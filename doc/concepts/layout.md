# Memory Layout

A `Layout` is the descriptor that says how to address a tensor's elements. It sits
alongside the buffer, so the operations that only change *how* elements are addressed -
reshape, slice, transpose, broadcast - rewrite the layout and copy nothing.

---

## The fields

```text
shape      - the logical dimensions, e.g. [3, 4] for a 3×4 matrix
stride     - how many elements to skip per step in each dimension
adj_stride - how much the memory position changes when a dimension steps (see below)
offset     - where in the buffer the tensor starts
len        - total number of elements (product of shape)
```

Element `[i, j]` of a 2D tensor sits at:

```text
buffer[offset + i * stride[0] + j * stride[1]]
```

Transposes, slices, and views are just different values of these fields over the same
buffer.

---

## adj_stride - the iteration trick

`adj_stride` exists to remove the per-element multiply-and-accumulate that stride-based
addressing would otherwise cost. It models the gap between two consecutive elements at the
edge of a contiguous block: the jump the running position takes when an inner dimension
wraps and the next dimension out steps forward.

The naive way to iterate recomputes the full position from the multi-dimensional index at
every element:

```text
pos = offset + counter[0]*stride[0] + counter[1]*stride[1] + ...
```

Moving from logical element N to N+1 only ever steps one dimension: the innermost one that
did not wrap. The delta added to `pos` depends on *which* dimension stepped, and
`adj_stride[d]` is that delta:

```text
adj_stride[last]  =  stride[last]
adj_stride[d]     =  stride[d]  −  sum( stride[k] * (shape[k]−1)  for k = d+1..last )
```

**Where the formula comes from.** When dimension `d` increments, every inner dimension
(`k > d`) has just wrapped from `shape[k]−1` back to zero - so the iterator has traveled
`stride[k] * (shape[k]−1)` forward through each one since `d` last stepped, all of which
must be undone. Then `d` steps forward once, adding `stride[d]`. The net delta is
`stride[d]` minus the walk through the inner dimensions. A negative component falls out of
this directly: it marks a dimension whose steps run backwards through memory.

**Example: contiguous `[3, 4]`, stride `[4, 1]`**

```text
adj_stride[1]  =  1
adj_stride[0]  =  4  −  1*(4−1)  =  1
```

Both are 1 - stepping any dimension moves by 1 from the previous position. This is the fast
path, and why `adj_stride = [1, ..., 1]` is hardcoded for freshly allocated tensors.

**Example: transposed `[4, 3]` (from `[3, 4]`, stride `[1, 4]`)**

```text
adj_stride[1]  =  4
adj_stride[0]  =  1  −  4*(3−1)  =  −7
```

`adj_stride[1] = 4` skips 4 physical elements per inner step; `adj_stride[0] = -7` jumps
backwards to the top of the next column when the outer dimension increments.

---

## Zero-copy operations

- **View (reshape).** New `shape` and `stride`, same `offset`, shared buffer. Requires a
  contiguous input - a transposed or sliced tensor must be packed with `AsContiguous` first.
- **Slice.** Adjusts `offset` to the start of the subregion and updates `shape`, keeping the
  original strides.
- **Transpose.** Reverses `shape` and `stride` and recomputes `adj_stride`; reversed axes
  give negative `adj_stride` components.
- **Permuting axes.** `transpose_axes` generalizes transpose to any axis permutation (and
  rejects non-bijections); it is the same shape/stride shuffle, no data moved.

---

## Broadcasting

Broadcasting lets a smaller tensor stand in for a larger one, and Candela implements it
entirely through zero strides. A dimension with `stride = 0` reads the same physical offset
at every step along that axis. A `[4]` vector broadcast to `[3, 4]` gains a leading dimension
with `stride = 0`, so `[i, j]` reads `buffer[i*0 + j*1] = buffer[j]` - three rows over the
same four values, buffer untouched.

```text
source: shape=[4],   stride=[1],    len=4    (physical buffer)
result: shape=[3,4], stride=[0, 1], len=12   (logical view)
```

The right-aligned rules follow NumPy: dimensions beyond the source rank are prepended with
`stride = 0`, a source dimension of size 1 expands with `stride = 0`, and any other mismatch
is rejected.

The `adj_stride` formula needs no special case for a zero stride: it subtracts the inner
walk from zero, leaving a negative delta that resets the position to the start of the inner
sequence while the zero stride adds nothing - so the outer step restarts the inner row from
the same offset.

---

## Structural flags

Two cheap checks read straight off the layout instead of scanning the elements:

- **`is_contiguous()`** is `adj_stride[0] == 1` together with no zero stride in the inner
  axes. The first part means every step advances by exactly one element; the zero-stride
  guard excludes broadcasts, whose repeated reads are not a contiguous run.
- **`is_transposed()`** is true when some axis has `adj_stride[d] < 0 && stride[d] != 0`. The
  `stride != 0` guard again excludes broadcast dimensions: a negative `adj_stride` from a
  zero stride is the positional reset described above, not a reversal.

Together they let code decide whether a BLAS pack is needed without mistaking a broadcast for
a transpose.
