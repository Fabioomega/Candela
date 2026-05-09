# Memory Layout

Tensor libraries need to separate two things that are easy to confuse: the *logical*
description of a tensor (its shape and how to address its elements) from the *physical*
memory that stores the numbers. Candela does this with a `Layout` descriptor that lives
alongside the buffer rather than being baked into it.

The payoff is that operations like reshape, slice, and transpose can return instantly —
they just build a new `Layout` pointing into the same memory, without touching a single
element.

---

## The fields

```
shape      — the logical dimensions, e.g. [3, 4] for a 3×4 matrix
stride     — how many elements to skip per step in each dimension
adj_stride — how much the memory position changes when a dimension steps (see below)
offset     — where in the buffer this tensor starts
len        — total number of elements (product of shape)
```

To read element `[i, j]` of a 2D tensor you compute:

```
buffer[offset + i * stride[0] + j * stride[1]]
```

That's it. Transposes, slices, and views are all just different values for these fields
pointing into the same underlying buffer.

---

## adj_stride — the iteration trick

`adj_stride` is probably the most unusual field here. It's what makes non-contiguous
iteration fast.

The naive way to iterate over a tensor is to recompute the full memory position from
the current multi-dimensional index at every step:

```
pos = offset + counter[0]*stride[0] + counter[1]*stride[1] + ...
```

That's a multiply-and-accumulate per element, which adds up. Instead, the iterator
keeps a running `pos` and just adds a single precomputed delta at each step.

The key observation is: when you move from logical element N to element N+1, you're
stepping in exactly one dimension (the innermost one that didn't wrap). The delta you
need to add to `pos` depends on *which* dimension just stepped.

`adj_stride[d]` is that delta for dimension `d`. It's defined as:

```
adj_stride[last]  =  stride[last]
adj_stride[d]     =  stride[d]  −  sum( stride[k] * (shape[k]−1)  for k = d+1..last )
```

The subtracted term undoes the movement that happened while iterating through the
lower dimensions. When dimension `d` increments, all dimensions below it have just
wrapped back to zero — so you need to step *back* over everything you walked through
at the inner level, then step forward by `stride[d]`.

**Example: contiguous `[3, 4]` with stride `[4, 1]`**

```
adj_stride[1]  =  1
adj_stride[0]  =  4  −  1*(4−1)  =  4 − 3  =  1
```

Both are 1 — stepping in any dimension always moves by 1 from the previous position.
This is the fast path and is why `adj_stride = [1, ..., 1]` is hardcoded for freshly
allocated tensors.

**Example: transposed `[4, 3]` (originally `[3, 4]`, stride reversed to `[1, 4]`)**

```
adj_stride[1]  =  4
adj_stride[0]  =  1  −  4*(3−1)  =  1 − 8  =  −7
```

`adj_stride[1] = 4` means stepping along the inner logical dimension skips 4 physical
elements. `adj_stride[0] = -7` means when the outer dimension increments, the position
jumps *backwards* by 7 — because you've been walking forward through a column and now
need to back up to the top of the next column.

The sign of `adj_stride` is actually how Candela detects transposed tensors: any
negative value means the tensor has a reversed dimension. Similarly, `adj_stride[0] == 1`
means the entire tensor is laid out contiguously in memory.

---

## Zero-copy operations

### View (reshape)

Changes the shape without touching data. Returns a new `Layout` with the same `offset`
and a freshly computed `stride` for the new shape. The buffer is shared.

The constraint: the tensor must already be contiguous. You can't reshape a transposed or
sliced tensor directly — pack it first with `AsContiguous`, then view.

### Slice

Narrows the view to a subregion. Adjusts `offset` to point at the first element of the
slice and updates the shape, but keeps the original strides. No data moves.

### Transpose

Reverses `shape` and `stride` (swapping all axes). The `adj_stride` is recomputed for
the new arrangement. The buffer is untouched; the new layout will have negative
`adj_stride` components where the iteration direction reversed.

---

## When a copy is unavoidable

Some operations need elements laid out sequentially — BLAS routines being the obvious
example. When that's the case, Candela inserts an `AsContiguous` node into the graph.
It allocates a fresh buffer and copies the elements in row-major order, producing a
clean layout with `adj_stride = [1, ..., 1]`.

This only happens when it has to. A scalar op on a transposed tensor, for instance,
iterates through the transposed layout directly using `adj_stride` — no copy needed.

---

## What's coming: broadcasting

Broadcasting will be wired up in Phase 4. The trick is setting a dimension's stride to
`0`. When the iterator reads element `[i, ...]` in that dimension, it adds
`0 * (shape[k]−1)` to `adj_stride` for that dimension, which evaluates as if the
dimension were size 1 — so the same slice of memory is reused for every index along
that axis. A `[1, 4]` row vector broadcast to `[3, 4]` just sets `stride[0] = 0`; no
data is copied.

`broadcast_to_shape` already exists in the layout system; it has a few bugs that need
fixing (tracked in Phase 2) before it can be wired into the element-wise ops.
