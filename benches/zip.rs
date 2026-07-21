use candela::backend::{Backend, ComputeFor, CpuPure};
use candela::skeleton::{Skeleton, SkeletonSlot};
use candela::{Dimension, Layout, OpError};
use criterion::{
    AxisScale, BenchmarkId, Criterion, PlotConfiguration, criterion_group, criterion_main,
};
use std::hint::black_box;
use std::iter::zip;

mod common;
use common::{FillConfig, ShapePolicy, SizeSpec, Variant};

struct Walk {
    start: usize,
    rows: usize,
    cols: usize,
    pitch: isize,
    step: isize,
}

fn walk(layout: &Layout) -> Walk {
    match layout.shape().len() {
        1 => Walk {
            start: layout.offset(),
            rows: 1,
            cols: layout.shape()[0],
            pitch: 0,
            step: layout.stride()[0] as isize,
        },
        2 => Walk {
            start: layout.offset(),
            rows: layout.shape()[0],
            cols: layout.shape()[1],
            pitch: layout.stride()[0] as isize,
            step: layout.stride()[1] as isize,
        },
        _ => panic!("no best zip kernel for layout {layout:?}"),
    }
}

//////////////////////////////////////////////////////////////////

// TODO: Axis reordering tensors is only worthwhile when the input count
// is bigger than 2. We can use the implementation of with some changes:
// https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/TensorIterator.cpp
// Pytorch implementation does not assume the ordering of the result while we assume
// the ouput stride always being one and in row-major order so we need to take that
// into consideration. Also should look into tiling for the transposed case.

const MAX_DIMS: usize = 8;

fn simplify_duo_layout(
    layout_a: &Layout,
    layout_b: &Layout,
) -> (usize, [usize; MAX_DIMS], [i32; MAX_DIMS], [i32; MAX_DIMS]) {
    let rank: usize = layout_a.shape().len();
    let mut shape = [0usize; MAX_DIMS];
    let mut adj_stride_a = [0i32; MAX_DIMS];
    let mut adj_stride_b = [0i32; MAX_DIMS];

    let mut w: usize = 0;

    shape[0] = layout_a.shape()[0];
    adj_stride_a[0] = layout_a.adj_stride()[0];
    adj_stride_b[0] = layout_b.adj_stride()[0];
    for i in 1..rank {
        if layout_a.adj_stride()[i] == layout_a.adj_stride()[i - 1]
            && layout_b.adj_stride()[i] == layout_b.adj_stride()[i - 1]
        {
            shape[w] *= layout_a.shape()[i];
        } else {
            w += 1;
            shape[w] = layout_a.shape()[i];
            adj_stride_a[w] = layout_a.adj_stride()[i];
            adj_stride_b[w] = layout_b.adj_stride()[i];
        }
    }

    (w + 1, shape, adj_stride_a, adj_stride_b)
}

/// Same collapse as [`simplify_duo_layout`], but returns *raw* strides per
/// merged dim instead of adjusted ones. The adjusted stride bakes in the rewind
/// of the inner dims, which is exactly what you want for an accumulating walk
/// (`pos += adj_stride`) and exactly what you don't want for computing a chunk's
/// position from its index directly (`pos = offset + Σ iₖ·strideₖ`). A merged
/// group's raw stride is its innermost original stride - stepping the combined
/// dimension by one is stepping the innermost original dimension by one.
fn simplify_duo_layout_raw(
    layout_a: &Layout,
    layout_b: &Layout,
) -> (usize, [usize; MAX_DIMS], [i32; MAX_DIMS], [i32; MAX_DIMS]) {
    let rank: usize = layout_a.shape().len();
    let mut shape = [0usize; MAX_DIMS];
    let mut raw_a = [0i32; MAX_DIMS];
    let mut raw_b = [0i32; MAX_DIMS];

    let mut w: usize = 0;

    shape[0] = layout_a.shape()[0];
    raw_a[0] = layout_a.stride()[0];
    raw_b[0] = layout_b.stride()[0];
    for i in 1..rank {
        if layout_a.adj_stride()[i] == layout_a.adj_stride()[i - 1]
            && layout_b.adj_stride()[i] == layout_b.adj_stride()[i - 1]
        {
            shape[w] *= layout_a.shape()[i];
            // The innermost dim of the group carries its raw stride.
            raw_a[w] = layout_a.stride()[i];
            raw_b[w] = layout_b.stride()[i];
        } else {
            w += 1;
            shape[w] = layout_a.shape()[i];
            raw_a[w] = layout_a.stride()[i];
            raw_b[w] = layout_b.stride()[i];
        }
    }

    (w + 1, shape, raw_a, raw_b)
}

struct IterWith<'a, T> {
    data_a: &'a [T],
    pos_a: usize,
    data_b: &'a [T],
    pos_b: usize,
    counter: [usize; MAX_DIMS],
    layout_a: &'a Layout,
    layout_b: &'a Layout,
    left_over: usize,
}

impl<'a, T> IterWith<'a, T> {
    pub fn new<'b: 'a>(
        data_a: &'a [T],
        layout_a: &'a Layout,
        data_b: &'b [T],
        layout_b: &'b Layout,
    ) -> Result<Self, OpError> {
        debug_assert!(
            layout_a.shape().len() <= MAX_DIMS,
            "dimensions higher than 8 are not support for iteration!"
        );

        debug_assert!(
            layout_b.shape().len() <= MAX_DIMS,
            "dimensions higher than 8 are not support for iteration!"
        );

        if layout_a.shape() != layout_b.shape() {
            return Err(OpError::NotSameShape(
                layout_a.shape().into(),
                layout_b.shape().into(),
            ));
        }

        Ok(Self {
            data_a,
            pos_a: layout_a.offset(),
            data_b,
            pos_b: layout_b.offset(),
            layout_a,
            layout_b,
            counter: [0; MAX_DIMS],
            left_over: layout_a.len(),
        })
    }

    pub fn new_unchecked<'b: 'a>(
        data_a: &'a [T],
        layout_a: &'a Layout,
        data_b: &'b [T],
        layout_b: &'b Layout,
    ) -> Self {
        debug_assert!(
            layout_a.shape().len() <= MAX_DIMS,
            "dimensions higher than 8 are not support for iteration!"
        );

        debug_assert!(
            layout_b.shape().len() <= MAX_DIMS,
            "dimensions higher than 8 are not support for iteration!"
        );

        Self {
            data_a,
            pos_a: layout_a.offset(),
            data_b,
            pos_b: layout_b.offset(),
            layout_a,
            layout_b,
            counter: [0; MAX_DIMS],
            left_over: layout_a.len(),
        }
    }
}

impl<'a, T> Iterator for IterWith<'a, T> {
    type Item = (&'a T, &'a T);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.left_over == 0 {
            return None;
        }

        let last = self.layout_a.shape().len() - 1;
        self.counter[last] += 1;
        let mut step_dim = last;

        for dim in (1..self.layout_a.shape().len()).rev() {
            if self.counter[dim] == self.layout_a.shape()[dim] {
                self.counter[dim] = 0;
                self.counter[dim - 1] += 1;

                step_dim = dim - 1;
                continue;
            }
            break;
        }

        let item_a = &self.data_a[self.pos_a];

        let item_b = &self.data_b[self.pos_b];

        self.pos_a = self
            .pos_a
            .wrapping_add_signed(self.layout_a.adj_stride()[step_dim] as isize);

        self.pos_b = self
            .pos_b
            .wrapping_add_signed(self.layout_b.adj_stride()[step_dim] as isize);

        self.left_over -= 1;

        Some((item_a, item_b))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.left_over, Some(self.left_over))
    }

    fn fold<B, F>(self, init: B, mut f: F) -> B
    where
        F: FnMut(B, Self::Item) -> B,
    {
        let mut acc = init;

        if self.layout_a.is_contiguous() && self.layout_b.is_contiguous() {
            let a = self.data_a
                [self.layout_a.offset()..self.layout_a.offset() + self.layout_a.len()]
                .iter();

            let b = self.data_b
                [self.layout_b.offset()..self.layout_b.offset() + self.layout_b.len()]
                .iter();

            for el in zip(a, b) {
                acc = f(acc, el);
            }

            return acc;
        }

        let (mut rank, mut shape, mut adj_stride_a, mut adj_stride_b) =
            simplify_duo_layout(self.layout_a, self.layout_b);

        if rank == 1 {
            shape[1] = shape[0];
            shape[0] = 1;

            adj_stride_a[1] = adj_stride_a[0];
            adj_stride_b[1] = adj_stride_b[0];

            rank += 1;
        }

        let last = rank - 1;
        let mut counter: [usize; MAX_DIMS] = [0; MAX_DIMS];

        let mut pos_a = self.layout_a.offset();
        let mut pos_b = self.layout_b.offset();

        let left_over: usize = shape[0..last].iter().product();

        let n = shape[last];
        let step_a = adj_stride_a[last] as isize * (shape[last] - 1) as isize;
        let stride_a = adj_stride_a[last] as isize;

        let step_b = adj_stride_b[last] as isize * (shape[last] - 1) as isize;
        let stride_b = adj_stride_b[last] as isize;

        let next_chunk = |counter: &mut [usize; MAX_DIMS]| -> usize {
            let last_counter = last - 1;
            counter[last_counter] += 1;
            let mut step_dim = last_counter;
            for dim in (1..last).rev() {
                if counter[dim] == shape[dim] {
                    counter[dim] = 0;
                    counter[dim - 1] += 1;
                    step_dim = dim - 1;
                    continue;
                }
                break;
            }

            step_dim
        };

        if stride_a == 1 && stride_b == 1 {
            for _ in 0..left_over {
                for el in zip(
                    self.data_a[pos_a..pos_a + n].iter(),
                    self.data_b[pos_b..pos_b + n].iter(),
                ) {
                    acc = f(acc, el);
                }

                let step_dim = next_chunk(&mut counter);

                pos_a = pos_a.wrapping_add_signed(adj_stride_a[step_dim] as isize + step_a);
                pos_b = pos_b.wrapping_add_signed(adj_stride_b[step_dim] as isize + step_b);
            }
        } else if stride_a == 0 && stride_b == 0 {
            for _ in 0..left_over {
                acc = f(acc, (&self.data_a[pos_a], &self.data_b[pos_b]));

                let step_dim = next_chunk(&mut counter);

                pos_a = pos_a.wrapping_add_signed(adj_stride_a[step_dim] as isize + step_a);
                pos_b = pos_b.wrapping_add_signed(adj_stride_b[step_dim] as isize + step_b);
            }
        } else if stride_a == 0 {
            for _ in 0..left_over {
                for el in self.data_b[pos_b..pos_b + n].iter() {
                    acc = f(acc, (&self.data_a[pos_a], el));
                }

                let step_dim = next_chunk(&mut counter);

                pos_a = pos_a.wrapping_add_signed(adj_stride_a[step_dim] as isize + step_a);
                pos_b = pos_b.wrapping_add_signed(adj_stride_b[step_dim] as isize + step_b);
            }
        } else if stride_b == 0 {
            for _ in 0..left_over {
                for el in self.data_a[pos_a..pos_a + n].iter() {
                    acc = f(acc, (el, &self.data_b[pos_b]));
                }

                let step_dim = next_chunk(&mut counter);

                pos_a = pos_a.wrapping_add_signed(adj_stride_a[step_dim] as isize + step_a);
                pos_b = pos_b.wrapping_add_signed(adj_stride_b[step_dim] as isize + step_b);
            }
        } else {
            for _ in 0..left_over {
                let mut pos_inner_a = pos_a;
                let mut pos_inner_b = pos_b;
                for _ in 0..n {
                    debug_assert!(pos_inner_a < self.data_a.len());
                    debug_assert!(pos_inner_b < self.data_b.len());
                    // SAFETY: a well-formed layout only ever visits in-bounds
                    // positions of its own buffer, so `pos_inner` is a valid
                    // index. Dropping the bounds check keeps the strided read
                    // from stalling memory-level parallelism on gather-heavy
                    // layouts (e.g. transposed).
                    acc = f(acc, unsafe {
                        (
                            self.data_a.get_unchecked(pos_inner_a),
                            self.data_b.get_unchecked(pos_inner_b),
                        )
                    });

                    pos_inner_a = pos_inner_a.wrapping_add_signed(stride_a);
                    pos_inner_b = pos_inner_b.wrapping_add_signed(stride_b);
                }

                let step_dim = next_chunk(&mut counter);

                pos_a = pos_a.wrapping_add_signed(adj_stride_a[step_dim] as isize + step_a);
                pos_b = pos_b.wrapping_add_signed(adj_stride_b[step_dim] as isize + step_b);
            }
        }

        acc
    }
}

impl<'a, T: Copy> IterWith<'a, T> {
    /// Like `fold`, but drives the output too: writes `f(a, b)` into `out` in
    /// row-major order. The fold hands the consumer scalars and lets it pick the
    /// write pattern (typically `out[i] = ...` with an external counter, an
    /// indexed store). Here the inner loop owns `out` as a contiguous slice and
    /// pairs it with the source via `iter_mut().zip(..)` - the same shape the
    /// hand-rolled kernel uses, so the compiler emits a streaming pointer walk
    /// instead of a base+index store plus a separate induction variable.
    fn apply(self, out: &mut [T], f: impl Fn(&T, &T) -> T) {
        if self.layout_a.is_contiguous() && self.layout_b.is_contiguous() {
            let a =
                &self.data_a[self.layout_a.offset()..self.layout_a.offset() + self.layout_a.len()];
            let b =
                &self.data_b[self.layout_b.offset()..self.layout_b.offset() + self.layout_b.len()];
            for ((o, x), y) in out.iter_mut().zip(a).zip(b) {
                *o = f(x, y);
            }
            return;
        }

        let (mut rank, mut shape, mut raw_a, mut raw_b) =
            simplify_duo_layout_raw(self.layout_a, self.layout_b);

        if rank == 1 {
            shape[1] = shape[0];
            shape[0] = 1;
            raw_a[1] = raw_a[0];
            raw_b[1] = raw_b[0];
            rank += 1;
        }

        let last = rank - 1;
        let n = shape[last];
        let inner_a = raw_a[last] as isize;
        let inner_b = raw_b[last] as isize;

        // The output is contiguous row-major of length (#chunks * n), so
        // `chunks_exact_mut(n)` hands out each destination row with no index
        // math - the same fixed-stride output walk the hand kernel gets.
        //
        // Source addressing is dependency-free instead of accumulated. The
        // second-innermost dim `mid` is the hot loop: its position is the pure
        // affine `base + j * stride`, so every row's address is independent and
        // the out-of-order engine can keep many rows' loads in flight (this is
        // what let the rank-2 path catch the hand kernel). The block base -
        // everything above `mid` - is decomposed from the block index with raw
        // strides; that carries the only division, but it runs once per block
        // (`outer` times), not once per chunk, so it is amortized over `mid`
        // rows. For rank 2, `outer == 1` and the whole thing degenerates to a
        // single affine loop with no decomposition at all.
        let mid_dim = last - 1;
        let mid = shape[mid_dim];
        let mid_a = raw_a[mid_dim] as isize;
        let mid_b = raw_b[mid_dim] as isize;
        let outer: usize = shape[0..mid_dim].iter().product();

        let off_a = self.layout_a.offset();
        let off_b = self.layout_b.offset();

        let block_base = |b: usize| -> (usize, usize) {
            let mut pa = off_a;
            let mut pb = off_b;
            let mut rem = b;
            for k in (0..mid_dim).rev() {
                let ik = rem % shape[k];
                rem /= shape[k];
                pa = pa.wrapping_add_signed(ik as isize * raw_a[k] as isize);
                pb = pb.wrapping_add_signed(ik as isize * raw_b[k] as isize);
            }
            (pa, pb)
        };

        let mut out_chunks = out.chunks_exact_mut(n);

        // Stamps the block/mid nest with a per-row body that sees `row_out`,
        // `pa`, `pb`. The inner-stride kind is loop-invariant, so it is matched
        // once out here rather than per chunk - each arm gets its own vectorized
        // row loop, none of them re-tests the strides.
        macro_rules! drive {
            ($row_out:ident, $pa:ident, $pb:ident, $row:block) => {{
                for b in 0..outer {
                    let (base_a, base_b) = block_base(b);
                    for j in 0..mid {
                        let $row_out = out_chunks.next().unwrap();
                        let $pa = base_a.wrapping_add_signed(j as isize * mid_a);
                        let $pb = base_b.wrapping_add_signed(j as isize * mid_b);
                        $row
                    }
                }
            }};
        }

        if inner_a == 1 && inner_b == 1 {
            drive!(row_out, pa, pb, {
                for ((o, x), y) in row_out
                    .iter_mut()
                    .zip(self.data_a[pa..pa + n].iter())
                    .zip(self.data_b[pb..pb + n].iter())
                {
                    *o = f(x, y);
                }
            });
        } else if inner_a == 0 && inner_b == 0 {
            drive!(row_out, pa, pb, {
                let v = f(&self.data_a[pa], &self.data_b[pb]);
                row_out.fill(v);
            });
        } else if inner_b == 0 {
            // `a` streams the row, `b` is one splatted scalar (bcast_inner).
            drive!(row_out, pa, pb, {
                let bv = self.data_b[pb];
                for (o, x) in row_out.iter_mut().zip(self.data_a[pa..pa + n].iter()) {
                    *o = f(x, &bv);
                }
            });
        } else if inner_a == 0 {
            drive!(row_out, pa, pb, {
                let av = self.data_a[pa];
                for (o, y) in row_out.iter_mut().zip(self.data_b[pb..pb + n].iter()) {
                    *o = f(&av, y);
                }
            });
        } else {
            drive!(row_out, pa, pb, {
                let mut ia = pa;
                let mut ib = pb;
                for o in row_out.iter_mut() {
                    debug_assert!(ia < self.data_a.len());
                    debug_assert!(ib < self.data_b.len());
                    // SAFETY: a well-formed layout only ever visits in-bounds
                    // positions of its own buffer.
                    *o = unsafe { f(self.data_a.get_unchecked(ia), self.data_b.get_unchecked(ib)) };
                    ia = ia.wrapping_add_signed(inner_a);
                    ib = ib.wrapping_add_signed(inner_b);
                }
            });
        }
    }

    /// Same output-driving structure as [`apply`], but advances the source
    /// positions the *old* way: accumulate `pos += adj_stride` per chunk with a
    /// carry counter, instead of computing each chunk's position affinely from
    /// its index. Kept only to isolate the addressing change - everything else
    /// (output ownership, inner-row bodies) is identical to [`apply`], so a diff
    /// between the two is purely raw-affine vs adj-accumulate.
    fn apply_adj(self, out: &mut [T], f: impl Fn(&T, &T) -> T) {
        if self.layout_a.is_contiguous() && self.layout_b.is_contiguous() {
            let a =
                &self.data_a[self.layout_a.offset()..self.layout_a.offset() + self.layout_a.len()];
            let b =
                &self.data_b[self.layout_b.offset()..self.layout_b.offset() + self.layout_b.len()];
            for ((o, x), y) in out.iter_mut().zip(a).zip(b) {
                *o = f(x, y);
            }
            return;
        }

        let (mut rank, mut shape, mut adj_a, mut adj_b) =
            simplify_duo_layout(self.layout_a, self.layout_b);

        if rank == 1 {
            shape[1] = shape[0];
            shape[0] = 1;
            adj_a[1] = adj_a[0];
            adj_b[1] = adj_b[0];
            rank += 1;
        }

        let last = rank - 1;
        let mut counter: [usize; MAX_DIMS] = [0; MAX_DIMS];
        let mut pos_a = self.layout_a.offset();
        let mut pos_b = self.layout_b.offset();
        let n = shape[last];
        let step_a = adj_a[last] as isize * (shape[last] - 1) as isize;
        let inner_a = adj_a[last] as isize;
        let step_b = adj_b[last] as isize * (shape[last] - 1) as isize;
        let inner_b = adj_b[last] as isize;

        let next_chunk = |counter: &mut [usize; MAX_DIMS]| -> usize {
            let last_counter = last - 1;
            counter[last_counter] += 1;
            let mut step_dim = last_counter;
            for dim in (1..last).rev() {
                if counter[dim] == shape[dim] {
                    counter[dim] = 0;
                    counter[dim - 1] += 1;
                    step_dim = dim - 1;
                    continue;
                }
                break;
            }
            step_dim
        };

        macro_rules! drive {
            ($row_out:ident, $pa:ident, $pb:ident, $row:block) => {{
                for $row_out in out.chunks_exact_mut(n) {
                    let $pa = pos_a;
                    let $pb = pos_b;
                    $row
                    let step_dim = next_chunk(&mut counter);
                    pos_a = pos_a.wrapping_add_signed(adj_a[step_dim] as isize + step_a);
                    pos_b = pos_b.wrapping_add_signed(adj_b[step_dim] as isize + step_b);
                }
            }};
        }

        if inner_a == 1 && inner_b == 1 {
            drive!(row_out, pa, pb, {
                for ((o, x), y) in row_out
                    .iter_mut()
                    .zip(self.data_a[pa..pa + n].iter())
                    .zip(self.data_b[pb..pb + n].iter())
                {
                    *o = f(x, y);
                }
            });
        } else if inner_a == 0 && inner_b == 0 {
            drive!(row_out, pa, pb, {
                let v = f(&self.data_a[pa], &self.data_b[pb]);
                row_out.fill(v);
            });
        } else if inner_b == 0 {
            drive!(row_out, pa, pb, {
                let bv = self.data_b[pb];
                for (o, x) in row_out.iter_mut().zip(self.data_a[pa..pa + n].iter()) {
                    *o = f(x, &bv);
                }
            });
        } else if inner_a == 0 {
            drive!(row_out, pa, pb, {
                let av = self.data_a[pa];
                for (o, y) in row_out.iter_mut().zip(self.data_b[pb..pb + n].iter()) {
                    *o = f(&av, y);
                }
            });
        } else {
            drive!(row_out, pa, pb, {
                let mut ia = pa;
                let mut ib = pb;
                for o in row_out.iter_mut() {
                    debug_assert!(ia < self.data_a.len());
                    debug_assert!(ib < self.data_b.len());
                    // SAFETY: a well-formed layout only ever visits in-bounds
                    // positions of its own buffer.
                    *o = unsafe { f(self.data_a.get_unchecked(ia), self.data_b.get_unchecked(ib)) };
                    ia = ia.wrapping_add_signed(inner_a);
                    ib = ib.wrapping_add_signed(inner_b);
                }
            });
        }
    }
}

//////////////////////////////////////////////////////////////////

struct UIterWith<'a, T> {
    data_a: &'a [T],
    pos_a: usize,
    data_b: &'a [T],
    pos_b: usize,
    counter: [usize; MAX_DIMS],
    layout_a: &'a Layout,
    layout_b: &'a Layout,
    left_over: usize,
}

impl<'a, T> UIterWith<'a, T> {
    pub fn new<'b: 'a>(
        data_a: &'a [T],
        layout_a: &'a Layout,
        data_b: &'b [T],
        layout_b: &'b Layout,
    ) -> Result<Self, OpError> {
        debug_assert!(
            layout_a.shape().len() <= MAX_DIMS,
            "dimensions higher than 8 are not support for iteration!"
        );

        debug_assert!(
            layout_b.shape().len() <= MAX_DIMS,
            "dimensions higher than 8 are not support for iteration!"
        );

        if layout_a.shape() != layout_b.shape() {
            return Err(OpError::NotSameShape(
                layout_a.shape().into(),
                layout_b.shape().into(),
            ));
        }

        Ok(Self {
            data_a,
            pos_a: layout_a.offset(),
            data_b,
            pos_b: layout_b.offset(),
            layout_a,
            layout_b,
            counter: [0; MAX_DIMS],
            left_over: layout_a.len(),
        })
    }

    pub fn new_unchecked<'b: 'a>(
        data_a: &'a [T],
        layout_a: &'a Layout,
        data_b: &'b [T],
        layout_b: &'b Layout,
    ) -> Self {
        debug_assert!(
            layout_a.shape().len() <= MAX_DIMS,
            "dimensions higher than 8 are not support for iteration!"
        );

        debug_assert!(
            layout_b.shape().len() <= MAX_DIMS,
            "dimensions higher than 8 are not support for iteration!"
        );

        Self {
            data_a,
            pos_a: layout_a.offset(),
            data_b,
            pos_b: layout_b.offset(),
            layout_a,
            layout_b,
            counter: [0; MAX_DIMS],
            left_over: layout_a.len(),
        }
    }
}

impl<'a, T> Iterator for UIterWith<'a, T> {
    type Item = (usize, &'a T, &'a T);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.left_over == 0 {
            return None;
        }

        let last = self.layout_a.shape().len() - 1;
        self.counter[last] += 1;
        let mut step_dim = last;

        for dim in (1..self.layout_a.shape().len()).rev() {
            if self.counter[dim] == self.layout_a.shape()[dim] {
                self.counter[dim] = 0;
                self.counter[dim - 1] += 1;

                step_dim = dim - 1;
                continue;
            }
            break;
        }

        let item_a = &self.data_a[self.pos_a];

        let item_b = &self.data_b[self.pos_b];

        self.pos_a = self
            .pos_a
            .wrapping_add_signed(self.layout_a.adj_stride()[step_dim] as isize);

        self.pos_b = self
            .pos_b
            .wrapping_add_signed(self.layout_b.adj_stride()[step_dim] as isize);

        self.left_over -= 1;

        Some((self.layout_a.len() - self.left_over - 1, item_a, item_b))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.left_over, Some(self.left_over))
    }

    fn fold<B, F>(self, init: B, mut f: F) -> B
    where
        F: FnMut(B, Self::Item) -> B,
    {
        let mut acc = init;

        if self.layout_a.is_contiguous() && self.layout_b.is_contiguous() {
            let a = self.data_a
                [self.layout_a.offset()..self.layout_a.offset() + self.layout_a.len()]
                .iter();

            let b = self.data_b
                [self.layout_b.offset()..self.layout_b.offset() + self.layout_b.len()]
                .iter();

            for (i, (x, y)) in zip(a, b).enumerate() {
                acc = f(acc, (i, x, y));
            }

            return acc;
        }

        let (mut rank, mut shape, mut adj_stride_a, mut adj_stride_b) =
            simplify_duo_layout(self.layout_a, self.layout_b);

        if rank == 1 {
            shape[1] = shape[0];
            shape[0] = 1;

            adj_stride_a[1] = adj_stride_a[0];
            adj_stride_b[1] = adj_stride_b[0];

            rank += 1;
        }

        let last = rank - 1;
        let mut counter: [usize; MAX_DIMS] = [0; MAX_DIMS];

        let mut pos_a = self.layout_a.offset();
        let mut pos_b = self.layout_b.offset();

        let left_over: usize = shape[0..last].iter().product();

        let n = shape[last];
        let step_a = adj_stride_a[last] as isize * (shape[last] - 1) as isize;
        let stride_a = adj_stride_a[last] as isize;

        let step_b = adj_stride_b[last] as isize * (shape[last] - 1) as isize;
        let stride_b = adj_stride_b[last] as isize;

        let next_chunk = |counter: &mut [usize; MAX_DIMS]| -> usize {
            let last_counter = last - 1;
            counter[last_counter] += 1;
            let mut step_dim = last_counter;
            for dim in (1..last).rev() {
                if counter[dim] == shape[dim] {
                    counter[dim] = 0;
                    counter[dim - 1] += 1;
                    step_dim = dim - 1;
                    continue;
                }
                break;
            }

            step_dim
        };

        if stride_a == 1 && stride_b == 1 {
            let mut i = 0;
            for _ in 0..left_over {
                for (x, y) in zip(
                    self.data_a[pos_a..pos_a + n].iter(),
                    self.data_b[pos_b..pos_b + n].iter(),
                ) {
                    acc = f(acc, (i, x, y));

                    i += 1;
                }

                let step_dim = next_chunk(&mut counter);

                pos_a = pos_a.wrapping_add_signed(adj_stride_a[step_dim] as isize + step_a);
                pos_b = pos_b.wrapping_add_signed(adj_stride_b[step_dim] as isize + step_b);
            }
        } else {
            let mut i = 0;
            for _ in 0..left_over {
                let mut pos_inner_a = pos_a;
                let mut pos_inner_b = pos_b;
                for _ in 0..n {
                    debug_assert!(pos_inner_a < self.data_a.len());
                    debug_assert!(pos_inner_b < self.data_b.len());
                    // SAFETY: a well-formed layout only ever visits in-bounds
                    // positions of its own buffer, so `pos_inner` is a valid
                    // index. Dropping the bounds check keeps the strided read
                    // from stalling memory-level parallelism on gather-heavy
                    // layouts (e.g. transposed).
                    acc = f(acc, unsafe {
                        (
                            i,
                            self.data_a.get_unchecked(pos_inner_a),
                            self.data_b.get_unchecked(pos_inner_b),
                        )
                    });

                    i += 1;

                    pos_inner_a = pos_inner_a.wrapping_add_signed(stride_a);
                    pos_inner_b = pos_inner_b.wrapping_add_signed(stride_b);
                }

                let step_dim = next_chunk(&mut counter);

                pos_a = pos_a.wrapping_add_signed(adj_stride_a[step_dim] as isize + step_a);
                pos_b = pos_b.wrapping_add_signed(adj_stride_b[step_dim] as isize + step_b);
            }
        }

        acc
    }
}

//////////////////////////////////////////////////////////////////

/// Hand-rolled `a * b`, the fair best for a two-operand element-wise op.
fn zip_mul_kernel(a: &[f32], la: &Layout, b: &[f32], lb: &Layout, out: &mut [f32]) {
    let (wa, wb) = (walk(la), walk(lb));
    assert_eq!(
        (wa.rows, wa.cols),
        (wb.rows, wb.cols),
        "operands disagree on shape"
    );
    assert_eq!(out.len(), wa.rows * wa.cols, "output is the wrong size");

    let cols = wa.cols;
    let row_start = |w: &Walk, r: usize| w.start.wrapping_add_signed(r as isize * w.pitch);

    if wb.step == 0 && wa.step == 1 {
        // `b` is inner-broadcast: one value per row, constant along it. Hoist
        // it out of the inner loop so the row becomes a splat-and-multiply the
        // compiler can vectorize, while `a` streams contiguously. This is the
        // whole point of the broadcast fast path - the naive strided walk below
        // would re-read the same `b` element `cols` times and never vectorize.
        for (r, row_out) in out.chunks_exact_mut(cols).enumerate() {
            let ra = row_start(&wa, r);
            let s = b[row_start(&wb, r)];
            for (o, x) in row_out.iter_mut().zip(&a[ra..ra + cols]) {
                *o = *x * s;
            }
        }
    } else if wa.step == 0 && wb.step == 1 {
        // Symmetric: `a` is the inner-broadcast operand.
        for (r, row_out) in out.chunks_exact_mut(cols).enumerate() {
            let s = a[row_start(&wa, r)];
            let rb = row_start(&wb, r);
            for (o, y) in row_out.iter_mut().zip(&b[rb..rb + cols]) {
                *o = s * *y;
            }
        }
    } else if wa.step == 0 && wb.step == 0 {
        // Both inner-broadcast: the whole row collapses to one product.
        for (r, row_out) in out.chunks_exact_mut(cols).enumerate() {
            let v = a[row_start(&wa, r)] * b[row_start(&wb, r)];
            row_out.fill(v);
        }
    } else if wa.step == 1 && wb.step == 1 {
        // Both inner-contiguous. Outer-broadcast lands here too: its pitch is 0,
        // so `row_start` pins every row to the same source slice - a reuse the
        // row loop reads straight out of cache.
        for (r, row_out) in out.chunks_exact_mut(cols).enumerate() {
            let (ra, rb) = (row_start(&wa, r), row_start(&wb, r));
            let row_a = &a[ra..ra + cols];
            let row_b = &b[rb..rb + cols];
            for ((o, x), y) in row_out.iter_mut().zip(row_a).zip(row_b) {
                *o = *x * *y;
            }
        }
    } else {
        for (r, row_out) in out.chunks_exact_mut(cols).enumerate() {
            let (ra, rb) = (row_start(&wa, r), row_start(&wb, r));
            for (c, o) in row_out.iter_mut().enumerate() {
                let c = c as isize;
                *o =
                    a[ra.wrapping_add_signed(c * wa.step)] * b[rb.wrapping_add_signed(c * wb.step)];
            }
        }
    }
}

fn zip_mul<B: Backend>(layouts: &[Layout]) -> Skeleton<f32, B>
where
    f32: ComputeFor<B>,
{
    let a: SkeletonSlot<f32, B> = SkeletonSlot::new(layouts[0].clone());
    let b: SkeletonSlot<f32, B> = SkeletonSlot::new(layouts[1].clone());
    (&a * &b).into_skeleton(&[a, b]).unwrap()
}

fn zip_multiplication(c: &mut Criterion) {
    let plot_config = PlotConfiguration::default().summary_scale(AxisScale::Logarithmic);

    let mut group = c.benchmark_group("zip_multiplication");
    group.plot_config(plot_config);

    let cfg = FillConfig::new(2)
        .variants(&[
            Variant::Contig,
            Variant::Step,
            Variant::Padded,
            Variant::Transposed,
        ])
        .sizes(&[
            SizeSpec::Elems(64),
            SizeSpec::Elems(1024),
            SizeSpec::L1,
            SizeSpec::L2,
            SizeSpec::L3,
            SizeSpec::Dram,
        ]);

    for case in common::fill_cases(&cfg, zip_mul::<CpuPure>) {
        group.throughput(case.throughput);

        let (lhs, rhs) = (&case.inputs[0], &case.inputs[1]);
        let (data_a, la) = (lhs.data(), lhs.layout());
        let (data_b, lb) = (rhs.data(), rhs.layout());
        let n = case.skeleton.len();

        let mut expected = vec![0.0f32; n];
        zip_mul_kernel(data_a, la, data_b, lb, &mut expected);

        let from_iter: Vec<f32> = lhs.iter().zip(rhs.iter()).map(|(x, y)| x * y).collect();

        let it = unsafe { IterWith::new(data_a, la, data_b, lb).unwrap_unchecked() };
        let from_iter_with: Vec<f32> = it.map(|(x, y)| *x * *y).collect();

        let it = unsafe { IterWith::new(data_a, la, data_b, lb).unwrap_unchecked() };
        let from_iter_fold: Vec<f32> = it.fold(Vec::new(), |mut v, x| {
            v.push(*x.0 * *x.1);
            v
        });

        let it = unsafe { UIterWith::new(data_a, la, data_b, lb).unwrap_unchecked() };
        let mut from_uiter_fold: Vec<f32> = Vec::with_capacity(it.size_hint().0);
        from_uiter_fold.resize(it.size_hint().0, 0.0);
        from_uiter_fold = it.fold(from_uiter_fold, |mut v, (i, x, y)| {
            v[i] = *x * *y;
            v
        });

        let it = unsafe { IterWith::new(data_a, la, data_b, lb).unwrap_unchecked() };
        let mut from_apply = vec![0.0f32; n];
        it.apply(&mut from_apply, |x, y| *x * *y);

        let it = unsafe { IterWith::new(data_a, la, data_b, lb).unwrap_unchecked() };
        let mut from_apply_adj = vec![0.0f32; n];
        it.apply_adj(&mut from_apply_adj, |x, y| *x * *y);

        assert_eq!(from_iter, expected, "zip order wrong at {}", case.label);
        assert_eq!(
            from_iter_with, expected,
            "zip order wrong at {}",
            case.label
        );
        assert_eq!(
            from_iter_fold, expected,
            "zip order wrong at {}",
            case.label
        );
        assert_eq!(
            from_uiter_fold, expected,
            "zip order wrong at {}",
            case.label
        );
        assert_eq!(from_apply, expected, "apply order wrong at {}", case.label);
        assert_eq!(
            from_apply_adj, expected,
            "apply_adj order wrong at {}",
            case.label
        );

        let mut out = vec![0.0f32; n];

        group.bench_function(BenchmarkId::new("best", &case.label), |bencher| {
            bencher.iter(|| {
                zip_mul_kernel(
                    black_box(data_a),
                    black_box(la),
                    black_box(data_b),
                    black_box(lb),
                    &mut out,
                );
                black_box(&out);
            });
        });

        group.bench_function(BenchmarkId::new("iter", &case.label), |bencher| {
            bencher.iter(|| {
                let pairs = black_box(lhs).iter().zip(black_box(rhs).iter());
                for (o, (x, y)) in out.iter_mut().zip(pairs) {
                    *o = x * y;
                }
                black_box(&out);
            });
        });

        group.bench_function(BenchmarkId::new("iter_with", &case.label), |bencher| {
            bencher.iter(|| {
                let it =
                    black_box(unsafe { IterWith::new(data_a, la, data_b, lb).unwrap_unchecked() });

                for (i, (a, b)) in it.enumerate() {
                    out[i] = *a * *b;
                }

                black_box(&out);
            });
        });

        group.bench_function(BenchmarkId::new("iter_with_fold", &case.label), |bencher| {
            bencher.iter(|| {
                let it =
                    unsafe { black_box(IterWith::new(data_a, la, data_b, lb)).unwrap_unchecked() };

                it.enumerate().for_each(|(i, (&x, &y))| {
                    out[i] = x * y;
                });

                black_box(&out);
            });
        });

        group.bench_function(BenchmarkId::new("uiter_with", &case.label), |bencher| {
            bencher.iter(|| {
                let it =
                    black_box(unsafe { UIterWith::new(data_a, la, data_b, lb).unwrap_unchecked() });

                for (i, a, b) in it {
                    out[i] = *a * *b;
                }

                black_box(&out);
            });
        });

        group.bench_function(
            BenchmarkId::new("uiter_with_fold", &case.label),
            |bencher| {
                bencher.iter(|| {
                    let it = unsafe {
                        black_box(UIterWith::new(data_a, la, data_b, lb)).unwrap_unchecked()
                    };

                    it.for_each(|(i, &x, &y)| {
                        out[i] = x * y;
                    });

                    black_box(&out);
                });
            },
        );

        group.bench_function(BenchmarkId::new("iter_apply", &case.label), |bencher| {
            bencher.iter(|| {
                let it =
                    black_box(unsafe { IterWith::new(data_a, la, data_b, lb).unwrap_unchecked() });
                it.apply(&mut out, |x, y| *x * *y);
                black_box(&out);
            });
        });

        group.bench_function(BenchmarkId::new("iter_apply_adj", &case.label), |bencher| {
            bencher.iter(|| {
                let it =
                    black_box(unsafe { IterWith::new(data_a, la, data_b, lb).unwrap_unchecked() });
                it.apply_adj(&mut out, |x, y| *x * *y);
                black_box(&out);
            });
        });
    }

    group.finish();
}

/// Naive reference: `out` in logical row-major order, each element gathered
/// straight from the multi-index. Independent of `apply`'s chunking/decomposition.
fn reference_mul(data_a: &[f32], la: &Layout, data_b: &[f32], lb: &Layout) -> Vec<f32> {
    let shape = la.shape();
    let total: usize = shape.iter().product();
    let mut out = vec![0.0f32; total];
    for (lin, o) in out.iter_mut().enumerate() {
        let mut rem = lin;
        let mut pa = la.offset() as isize;
        let mut pb = lb.offset() as isize;
        for d in (0..shape.len()).rev() {
            let i = rem % shape[d];
            rem /= shape[d];
            pa += i as isize * la.stride()[d] as isize;
            pb += i as isize * lb.stride()[d] as isize;
        }
        *o = data_a[pa as usize] * data_b[pb as usize];
    }
    out
}

/// `apply` is only exercised at rank 2 by the sweep (where the block loop runs
/// once and never decomposes an index). These multi-dim layouts drive the
/// `outer > 1` path - including a 4D case whose block base needs a real
/// `mod`/`div` - and check every inner-stride kind against [`reference_mul`].
fn validate_apply_nrank() {
    // distinct values so a mis-addressed element changes the product.
    let data: Vec<f32> = (0..256).map(|i| (i as f32) * 0.5 + 1.0).collect();

    // (shape, stride_a, stride_b) - same logical shape, differing strides.
    let cases: [(&[usize], &[i32], &[i32]); 6] = [
        // 3D contiguous vs outer-broadcast (dim 0 reused).
        (&[2, 3, 4], &[12, 4, 1], &[0, 4, 1]),
        // 3D contiguous vs middle-broadcast (dim 1 reused).
        (&[2, 3, 4], &[12, 4, 1], &[12, 0, 1]),
        // 3D contiguous vs inner-broadcast (dim 2 splatted).
        (&[2, 3, 4], &[12, 4, 1], &[12, 4, 0]),
        // 3D contiguous vs inner-strided (gather, non-unit inner stride).
        (&[2, 3, 4], &[12, 4, 1], &[24, 8, 2]),
        // 3D both broadcast on the inner dim (whole row collapses).
        (&[2, 3, 4], &[3, 1, 0], &[12, 4, 0]),
        // 4D contiguous vs inner-broadcast - forces the block loop to decompose
        // its index over dims 0 and 1 with mod/div (mid_dim == 2, outer == 6).
        (&[2, 3, 2, 4], &[24, 8, 4, 1], &[24, 8, 4, 0]),
    ];

    for (shape, sa, sb) in cases {
        let la = Layout::from_strided(shape, sa, 0);
        let lb = Layout::from_strided(shape, sb, 0);
        let expected = reference_mul(&data, &la, &data, &lb);

        let it = unsafe { IterWith::new(&data, &la, &data, &lb).unwrap_unchecked() };
        let mut got = vec![0.0f32; expected.len()];
        it.apply(&mut got, |x, y| *x * *y);

        assert_eq!(
            got, expected,
            "apply n-rank wrong for shape {shape:?} sa {sa:?} sb {sb:?}"
        );
    }
}

fn zip_broadcast(c: &mut Criterion) {
    validate_apply_nrank();

    let plot_config = PlotConfiguration::default().summary_scale(AxisScale::Logarithmic);

    let mut group = c.benchmark_group("zip_broadcast");
    group.plot_config(plot_config);

    let sizes = &[SizeSpec::L1, SizeSpec::L2, SizeSpec::L3, SizeSpec::Dram];

    for &inner in &[64usize, 1024usize] {
        let cfg = FillConfig::new(2)
            .shape(ShapePolicy::Rows { inner })
            .variant_combos(&[
                // &[Variant::Contig, Variant::Contig],
                &[Variant::Contig, Variant::BcastInner],
                &[Variant::Contig, Variant::BcastOuter],
            ])
            .sizes(sizes)
            .variant_sizes(sizes);

        for case in common::fill_cases(&cfg, zip_mul::<CpuPure>) {
            group.throughput(case.throughput);

            let (lhs, rhs) = (&case.inputs[0], &case.inputs[1]);
            let (data_a, la) = (lhs.data(), lhs.layout());
            let (data_b, lb) = (rhs.data(), rhs.layout());
            let n = case.skeleton.len();

            let mut expected = vec![0.0f32; n];
            zip_mul_kernel(data_a, la, data_b, lb, &mut expected);

            let it = unsafe { IterWith::new(data_a, la, data_b, lb).unwrap_unchecked() };
            let mut from_iter = vec![0.0f32; n];
            it.enumerate()
                .for_each(|(i, (&x, &y))| from_iter[i] = x * y);
            assert_eq!(
                from_iter, expected,
                "zip broadcast order wrong at n{inner} {}",
                case.label
            );

            let it = unsafe { IterWith::new(data_a, la, data_b, lb).unwrap_unchecked() };
            let mut from_apply = vec![0.0f32; n];
            it.apply(&mut from_apply, |x, y| *x * *y);
            assert_eq!(
                from_apply, expected,
                "zip apply order wrong at n{inner} {}",
                case.label
            );

            let mut out = vec![0.0f32; n];
            let tag = format!("n{inner}");

            group.bench_function(
                BenchmarkId::new(format!("best_{tag}"), &case.label),
                |bencher| {
                    bencher.iter(|| {
                        zip_mul_kernel(
                            black_box(data_a),
                            black_box(la),
                            black_box(data_b),
                            black_box(lb),
                            &mut out,
                        );
                        black_box(&out);
                    });
                },
            );

            group.bench_function(
                BenchmarkId::new(format!("iter_fold_{tag}"), &case.label),
                |bencher| {
                    bencher.iter(|| {
                        let it = unsafe {
                            black_box(IterWith::new(data_a, la, data_b, lb)).unwrap_unchecked()
                        };
                        it.enumerate().for_each(|(i, (&x, &y))| {
                            out[i] = x * y;
                        });
                        black_box(&out);
                    });
                },
            );

            group.bench_function(
                BenchmarkId::new(format!("iter_apply_{tag}"), &case.label),
                |bencher| {
                    bencher.iter(|| {
                        let it = unsafe {
                            black_box(IterWith::new(data_a, la, data_b, lb)).unwrap_unchecked()
                        };
                        it.apply(&mut out, |x, y| *x * *y);
                        black_box(&out);
                    });
                },
            );
        }
    }

    group.finish();
}

criterion_group!(benches, zip_multiplication, zip_broadcast);
criterion_main!(benches);
