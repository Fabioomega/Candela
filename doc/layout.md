# Memory Layout

Tensor libraries need to separate two things that are easy to confuse: the *logical*
description of a tensor (its shape and how to address its elements) from the *physical*
memory that stores the numbers. Candela does this with a `Layout` descriptor that lives
alongside the buffer rather than being baked into it.

The payoff is that operations like reshape, slice, and transpose can return instantly -
they just build a new `Layout` pointing into the same memory, without touching a single
element.

---

## The fields

```
shape      - the logical dimensions, e.g. [3, 4] for a 3×4 matrix
stride     - how many elements to skip per step in each dimension
adj_stride - how much the memory position changes when a dimension steps (see below)
offset     - where in the buffer this tensor starts
len        - total number of elements (product of shape)
```

To read element `[i, j]` of a 2D tensor you compute:

```
buffer[offset + i * stride[0] + j * stride[1]]
```

That's it. Transposes, slices, and views are all just different values for these fields
pointing into the same underlying buffer.

---

## adj_stride - the iteration trick

`adj_stride` is probably the most unusual field here. It came out of a simple question:
the stride-based position formula works, but can the per-element recomputation be
eliminated?

The Tensorken blog post on strides made the naive approach clear - and immediately
raised the question of whether it could be improved. So: open a text file, sketch a
few tensors by hand with arbitrary shapes up to five dimensions, apply some slices, and
stare at the result. The memory positions of a sliced matrix laid out on paper made
the answer obvious: there were gaps between elements, and they were completely regular.
Not random jumps - fixed deltas determined entirely by the shape and stride. If the gaps
are regular, they can be precomputed per dimension, and iteration becomes: keep a running
position, add one delta per step instead of a full multiply-and-accumulate from scratch.

The formula was worked out from enough concrete cases that the pattern became clear.
One consequence only noticed much later: `adj_stride[0] == 1` is an exact contiguity
check - a single field read, no scan. Any negative component means a dimension is
reversed. Neither was planned; both fall out of the math.

The naive way to iterate over a tensor is to recompute the full memory position from
the current multi-dimensional index at every step:

```
pos = offset + counter[0]*stride[0] + counter[1]*stride[1] + ...
```

That's a multiply-and-accumulate per element, which adds up.

The key observation is: when you move from logical element N to element N+1, you're
stepping in exactly one dimension (the innermost one that didn't wrap). The delta you
need to add to `pos` depends on *which* dimension just stepped.

`adj_stride[d]` is that delta for dimension `d`. It's defined as:

```
adj_stride[last]  =  stride[last]
adj_stride[d]     =  stride[d]  −  sum( stride[k] * (shape[k]−1)  for k = d+1..last )
```

**Where the formula comes from.** When dimension `d` increments, two things have
happened simultaneously to the running position. Every inner dimension (`k > d`) has
just wrapped from `shape[k]−1` back to zero - meaning the iterator has traveled
`stride[k] * (shape[k]−1)` forward through each one since the last time `d` stepped,
all of which needs to be undone. Then dimension `d` itself steps forward once, adding
`stride[d]`. The net delta is `stride[d]` minus the total walk through all inner
dimensions. That's the formula.

**Example: contiguous `[3, 4]` with stride `[4, 1]`**

```
adj_stride[1]  =  1
adj_stride[0]  =  4  −  1*(4−1)  =  4 − 3  =  1
```

Both are 1 - stepping in any dimension always moves by 1 from the previous position.
This is the fast path and is why `adj_stride = [1, ..., 1]` is hardcoded for freshly
allocated tensors.

**Example: transposed `[4, 3]` (originally `[3, 4]`, stride reversed to `[1, 4]`)**

```
adj_stride[1]  =  4
adj_stride[0]  =  1  −  4*(3−1)  =  1 − 8  =  −7
```

`adj_stride[1] = 4` means stepping along the inner logical dimension skips 4 physical
elements. `adj_stride[0] = -7` means when the outer dimension increments, the position
jumps *backwards* by 7 - because you've been walking forward through a column and now
need to back up to the top of the next column.

---

## Zero-copy operations

### View (reshape)

Changes the shape without touching data. Returns a new `Layout` with the same `offset`
and a freshly computed `stride` for the new shape. The buffer is shared.

The constraint: the tensor must already be contiguous. You can't reshape a transposed or
sliced tensor directly - pack it first with `AsContiguous`, then view.

### Slice

Narrows the view to a subregion. Adjusts `offset` to point at the first element of the
slice and updates the shape, but keeps the original strides. No data moves.

### Transpose

Reverses `shape` and `stride` (swapping all axes). The `adj_stride` is recomputed for
the new arrangement. The buffer is untouched; the new layout will have negative
`adj_stride` components where the iteration direction reversed.

### Permuting axes

`transpose` is the all-axes-reversed special case of a more general move: reordering the
axes by an arbitrary permutation. `transpose_axes(axes)` takes an explicit permutation -
`transpose_axes(&[1, 0])` is the 2D transpose, `transpose_axes(&[0, 2, 1])` swaps only the
last two axes of a rank-3 tensor and leaves the batch axis alone. It rejects a permutation
that isn't a bijection over the existing axes. Like `transpose`, it only shuffles `shape`
and `stride` and recomputes `adj_stride` - no data moves.

`rotate_axis_innermost(axis)` is a convenience wrapper for one common pattern: cyclically
rotate the axes so that `axis` (and everything before it) ends up innermost. For a rank-4
tensor, `rotate_axis_innermost(2)` produces the axis order `[3, 0, 1, 2]` - axis 2 becomes
the last (innermost) dimension. It's built on `transpose_axes`, so it's the same zero-copy
layout shuffle underneath.

---

## Broadcasting

Broadcasting lets a tensor with fewer elements stand in for a larger one along specific
dimensions - the layout pretends the same elements are accessible at multiple positions
without duplicating any data.

Candela implements this entirely through zero strides. A dimension with `stride = 0`
means every step along that axis reads from the same physical offset as the one before -
the position does not advance. A `[4]` vector broadcast to `[3, 4]` gains a new leading
dimension with `stride = 0`; reading element `[i, j]` computes `buffer[i*0 + j*1] = buffer[j]`,
so all three rows map to the same four values. The buffer is untouched.

```
source: shape=[4],   stride=[1],    len=4    (physical buffer)
result: shape=[3,4], stride=[0, 1], len=12   (logical view)
```

Like `View`, `Slice`, and `Transpose`, `Broadcast` is a zero-copy reference op. The
executor applies the broadcast layout to the existing buffer without allocating or copying
anything.

### Rules

Candela follows NumPy-style rules, aligned right:

- Dimensions beyond the source rank are prepended with `stride = 0`.
- A source dimension of size 1 expands to the target size with `stride = 0`.
- Any other mismatch (source dim is neither 1 nor equal to target) is rejected.

```
[1, 4] → [3, 4]   ✓  size-1 leading dim expands; last dim matches
[3, 1] → [3, 4]   ✓  size-1 trailing dim expands
[4]    → [3, 4]   ✓  rank promoted; new leading dim added with stride 0
[3, 4] → [2, 4]   ✗  dim 0: 3 is not 1 and 3 ≠ 2
```

### How adj_stride handles zero strides

The `adj_stride` formula requires no special case for broadcast dimensions. When `stride[d] = 0`, it subtracts the accumulated inner walk from zero - leaving
a negative value that resets the position back to the start of the inner sequence.
Combined with the zero stride, the net effect is: the outer step changes nothing. The
inner sequence restarts from the same offset.

**Example: `[4]` broadcast to `[3, 4]`, stride `[0, 1]`:**

```
adj_stride[1] = 1
adj_stride[0] = 0 − 1*(4−1) = −3
```

When dimension 0 increments, position backs up by 3 (undoing the inner walk) and
`stride[0] = 0` contributes nothing forward. The inner row starts over from the same
base position. The buffer is never read out of bounds.

### Structural flags

`is_contiguous()` returns false for any broadcast layout. The check looks for zeros in
the inner strides - a zero stride guarantees that elements are revisited rather than
advanced, which is the opposite of sequential layout.

`is_transposed()` returns false for broadcast-only layouts. The check is
`adj_stride[d] < 0 && stride[d] != 0`. The `stride != 0` guard explicitly excludes
broadcast dimensions: a negative `adj_stride` from a zero-stride dim is a positional
reset, not a reversal. Code that needs to distinguish "non-contiguous because transposed"
from "non-contiguous because broadcast" (e.g., deciding whether a BLAS pack is needed)
can use `is_transposed()` without false positives from broadcast layouts.

---

## When a copy is unavoidable

Some operations need elements laid out sequentially - BLAS routines being the obvious
example. When that's the case, Candela inserts an `AsContiguous` node into the graph.
It allocates a fresh buffer and copies the elements in row-major order, producing a
clean layout with `adj_stride = [1, ..., 1]`.

This only happens when it has to. A scalar op on a transposed tensor, for instance,
iterates through the transposed layout directly using `adj_stride` - no copy needed.

