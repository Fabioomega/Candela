//! `walker` bench: the reusable `DimWalker` driver vs the `count` baseline.
//!
//! `count` (copied verbatim from `addressing.rs`) is the fastest chunk-addressing
//! scheme we have - hand-inlined per call site, rank-specialized, with macro-
//! stamped row bodies. `DimWalker` packages that walk once and hands it out as a
//! reusable N-operand driver, so any gap this bench shows is the cost of the
//! *abstraction*, not of a different algorithm.
//!
//! Three contenders are timed:
//!
//! - **`count`** - the hand-inlined baseline.
//! - **`walker_spec`** (`DimWalkerSpec`) - an intermediate shape that hands the
//!   running positions to the row body as *separate scalar arguments*, with a
//!   rank-2 / rank-3 / rank-4+ dispatch.
//! - **`new_walker_spec`** (`DimWalker`) - the driver: `for_each` pushes the
//!   per-operand offsets as a `[usize; N]` array, the consumer matches the inner
//!   strides once and writes each output chunk through a raw cursor.
//!
//! `new_walker_spec` now ties or beats `count` at every compute-bound size (L2
//! and up) on rank 2/3/6, and beats `walker_spec` across the board. The only
//! residual is L1-tiny, where `DimWalker::new`'s setup (collapse + adjacent
//! strides + baked strides) can't amortize over so few chunks; it is gone by L2.
//! Memory-bound (DRAM) sizes tie for everyone.
//!
//! What was measured and settled - kept so it isn't re-run:
//!
//! - **The `[usize; N]` offsets array is free.** It crosses the `for_each`
//!   closure boundary as an aggregate, but SROA scalarizes it: the running
//!   positions live in registers and each `offsets[i] += ...` folds into a plain
//!   register add. The array in the closure signature is not a cost here.
//! - **The output cursor was the hidden cost.** Pulling each chunk with
//!   `chunks_exact_mut().next().unwrap()` layers a second, redundant loop-carried
//!   countdown (the `ChunksExactMut` length) plus a panic edge on top of the
//!   walk's own counter. The panic is an optimization barrier, and the extra
//!   length register tips the tight arms into spilling their strides to the
//!   stack. `.next().unwrap_unchecked()` makes the `None` arm unreachable, the
//!   length check dies, and the iterator collapses to a bare `ptr += len` - this
//!   alone brought rank 2 to parity.
//! - **Flat single loop + register countdown beats the nested walk at every mid
//!   extent.** One loop over all chunks that advances the position each chunk
//!   and, when a countdown hits zero, adds a block correction (the odometer only
//!   *selects* which correction, at the boundary) is the shape `count` uses at
//!   every rank. Splitting it into an outer/inner nested loop forces a two-level
//!   induction hierarchy - integer offsets kept live for the bounds checks and
//!   the deferred correction, *plus* strength-reduced running pointers - which
//!   overflows the register file and spills. The "wasteful"-looking per-chunk
//!   countdown is 2-3 predictable uops; the tidy nest is several extra
//!   loop-carried values. Flat won even at mid=128 (+3.5%), and the penalty of
//!   the nest grew with mid (+25% at mid=2 -> +36% at mid=128), confirming the
//!   cost lives in the inner per-chunk state, not the per-block correction.
//! - **Rank-3 specialization on its own is a hair.** Once the generic path is
//!   flat it already handles rank 3 (the odometer degenerates to a constant
//!   step-0 correction); a dedicated rank-3 arm only saves one counter increment
//!   per block. Kept because `count` keeps one, but it is not what closes the gap.
//! - **The `[s1, s2]` gather arm needs `get_unchecked`.** Both operands strided
//!   (inner stride neither 0 nor 1) is the one arm that reads inputs element by
//!   element; safe indexing bounds-checks every element and left `r2_transposed`
//!   ~12% behind `count` at L2. Matching `count`'s `get_unchecked` (guarded by a
//!   `debug_assert` on the position) closed it to parity. The contiguous and
//!   broadcast arms slice whole rows, so their one bounds check per chunk is
//!   free - only the per-element arm cared.

use candela::Layout;
use criterion::{
    AxisScale, BenchmarkId, Criterion, PlotConfiguration, Throughput, criterion_group,
    criterion_main,
};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};
use std::hint::black_box;
use std::iter::zip;
use std::sync::LazyLock;

const MAX_DIMS: usize = 8;
const SEED: u64 = 0xADD2E55;

static L1_BYTES: LazyLock<usize> =
    LazyLock::new(|| cache_size::l1_cache_size().unwrap_or(64 * 1024));
static L2_BYTES: LazyLock<usize> =
    LazyLock::new(|| cache_size::l2_cache_size().unwrap_or(256 * 1024));
static L3_BYTES: LazyLock<usize> =
    LazyLock::new(|| cache_size::l3_cache_size().unwrap_or(16 * 1024 * 1024));

//////////////////////////////////////////////////////////////////
// count - copied verbatim from addressing.rs (adjusted-stride flat walk with a
// register countdown), so the two contenders share an identical baseline.

fn calculate_adjacent_dim_stride(stride: &[i32], slice_shape: &[usize]) -> [i32; MAX_DIMS] {
    let rank = stride.len();
    debug_assert!(rank >= 1, "stride must have rank >= 1");

    let mut v = [0i32; MAX_DIMS];
    v[..rank].copy_from_slice(stride);

    let mut accum: i32 = 0;
    for i in (0..rank - 1).rev() {
        accum += stride[i + 1] * (slice_shape[i + 1] as i32 - 1);
        v[i] -= accum;
    }

    v
}

fn simplify_duo_layout(
    layout_a: &Layout,
    layout_b: &Layout,
) -> (usize, [usize; MAX_DIMS], [i32; MAX_DIMS], [i32; MAX_DIMS]) {
    let rank: usize = layout_a.shape().len();
    let mut shape = [0usize; MAX_DIMS];
    let mut adj_stride_a = [0i32; MAX_DIMS];
    let mut adj_stride_b = [0i32; MAX_DIMS];

    let mut w: usize = 0;

    let before_adj_stride_a = calculate_adjacent_dim_stride(layout_a.stride(), layout_a.shape());
    let before_adj_stride_b = calculate_adjacent_dim_stride(layout_b.stride(), layout_b.shape());

    shape[0] = layout_a.shape()[0];
    adj_stride_a[0] = before_adj_stride_a[0];
    adj_stride_b[0] = before_adj_stride_b[0];
    for i in 1..rank {
        if before_adj_stride_a[i] == before_adj_stride_a[i - 1]
            && before_adj_stride_b[i] == before_adj_stride_b[i - 1]
        {
            shape[w] *= layout_a.shape()[i];
        } else {
            w += 1;
            shape[w] = layout_a.shape()[i];
            adj_stride_a[w] = before_adj_stride_a[i];
            adj_stride_b[w] = before_adj_stride_b[i];
        }
    }

    (w + 1, shape, adj_stride_a, adj_stride_b)
}

macro_rules! stamp_arms {
    ($drive:ident, $data_a:ident, $data_b:ident, $n:ident, $inner_a:ident, $inner_b:ident, $f:ident) => {
        if $inner_a == 1 && $inner_b == 1 {
            $drive!(row_out, pa, pb, {
                for ((o, x), y) in row_out
                    .iter_mut()
                    .zip($data_a[pa..pa + $n].iter())
                    .zip($data_b[pb..pb + $n].iter())
                {
                    *o = $f(*x, *y);
                }
            });
        } else if $inner_a == 0 && $inner_b == 0 {
            $drive!(row_out, pa, pb, {
                let v = $f($data_a[pa], $data_b[pb]);
                row_out.fill(v);
            });
        } else if $inner_b == 0 {
            $drive!(row_out, pa, pb, {
                let bv = $data_b[pb];
                for (o, x) in row_out.iter_mut().zip($data_a[pa..pa + $n].iter()) {
                    *o = $f(*x, bv);
                }
            });
        } else if $inner_a == 0 {
            $drive!(row_out, pa, pb, {
                let av = $data_a[pa];
                for (o, y) in row_out.iter_mut().zip($data_b[pb..pb + $n].iter()) {
                    *o = $f(av, *y);
                }
            });
        } else {
            $drive!(row_out, pa, pb, {
                let mut ia = pa;
                let mut ib = pb;
                for o in row_out.iter_mut() {
                    debug_assert!(ia < $data_a.len());
                    debug_assert!(ib < $data_b.len());
                    // SAFETY: a well-formed layout only ever visits in-bounds
                    // positions of its own buffer.
                    *o = unsafe { $f(*$data_a.get_unchecked(ia), *$data_b.get_unchecked(ib)) };
                    ia = ia.wrapping_add_signed($inner_a);
                    ib = ib.wrapping_add_signed($inner_b);
                }
            });
        }
    };
}

fn apply_count(
    data_a: &[f32],
    la: &Layout,
    data_b: &[f32],
    lb: &Layout,
    out: &mut [f32],
    f: impl Fn(f32, f32) -> f32,
) {
    if la.is_contiguous() && lb.is_contiguous() {
        let a = &data_a[la.offset()..la.offset() + la.len()];
        let b = &data_b[lb.offset()..lb.offset() + lb.len()];
        for ((o, x), y) in out.iter_mut().zip(a).zip(b) {
            *o = f(*x, *y);
        }
        return;
    }

    let (mut rank, mut shape, mut adj_a, mut adj_b) = simplify_duo_layout(la, lb);

    if rank == 1 {
        shape[1] = shape[0];
        shape[0] = 1;
        adj_a[1] = adj_a[0];
        adj_b[1] = adj_b[0];
        rank += 1;
    }

    let last = rank - 1;
    let n = shape[last];
    let inner_a = adj_a[last] as isize;
    let inner_b = adj_b[last] as isize;
    let step_a = inner_a * (n - 1) as isize;
    let step_b = inner_b * (n - 1) as isize;

    if rank == 2 {
        let row_da = adj_a[0] as isize + step_a;
        let row_db = adj_b[0] as isize + step_b;
        let mut pos_a = la.offset();
        let mut pos_b = lb.offset();

        macro_rules! drive {
            ($row_out:ident, $pa:ident, $pb:ident, $row:block) => {{
                for $row_out in out.chunks_exact_mut(n) {
                    let $pa = pos_a;
                    let $pb = pos_b;
                    $row
                    pos_a = pos_a.wrapping_add_signed(row_da);
                    pos_b = pos_b.wrapping_add_signed(row_db);
                }
            }};
        }

        stamp_arms!(drive, data_a, data_b, n, inner_a, inner_b, f);
    } else if rank == 3 {
        let mid = shape[1];
        let row_da = adj_a[1] as isize + step_a;
        let row_db = adj_b[1] as isize + step_b;
        let corr_a = adj_a[0] as isize + step_a - row_da;
        let corr_b = adj_b[0] as isize + step_b - row_db;
        let mut pos_a = la.offset();
        let mut pos_b = lb.offset();
        let mut cd = mid;

        macro_rules! drive {
            ($row_out:ident, $pa:ident, $pb:ident, $row:block) => {{
                for $row_out in out.chunks_exact_mut(n) {
                    let $pa = pos_a;
                    let $pb = pos_b;
                    $row
                    pos_a = pos_a.wrapping_add_signed(row_da);
                    pos_b = pos_b.wrapping_add_signed(row_db);
                    cd -= 1;
                    if cd == 0 {
                        cd = mid;
                        pos_a = pos_a.wrapping_add_signed(corr_a);
                        pos_b = pos_b.wrapping_add_signed(corr_b);
                    }
                }
            }};
        }

        stamp_arms!(drive, data_a, data_b, n, inner_a, inner_b, f);
    } else {
        let mid_dim = last - 1;
        let mid = shape[mid_dim];
        let row_da = adj_a[mid_dim] as isize + step_a;
        let row_db = adj_b[mid_dim] as isize + step_b;
        let mut corr_a = [0isize; MAX_DIMS];
        let mut corr_b = [0isize; MAX_DIMS];
        for k in 0..mid_dim {
            corr_a[k] = adj_a[k] as isize + step_a - row_da;
            corr_b[k] = adj_b[k] as isize + step_b - row_db;
        }
        let mut counter: [usize; MAX_DIMS] = [0; MAX_DIMS];
        let mut pos_a = la.offset();
        let mut pos_b = lb.offset();
        let mut cd = mid;

        macro_rules! drive {
            ($row_out:ident, $pa:ident, $pb:ident, $row:block) => {{
                for $row_out in out.chunks_exact_mut(n) {
                    let $pa = pos_a;
                    let $pb = pos_b;
                    $row
                    pos_a = pos_a.wrapping_add_signed(row_da);
                    pos_b = pos_b.wrapping_add_signed(row_db);
                    cd -= 1;
                    if cd == 0 {
                        cd = mid;
                        let mut step = mid_dim - 1;
                        counter[step] += 1;
                        for dim in (1..mid_dim).rev() {
                            if counter[dim] == shape[dim] {
                                counter[dim] = 0;
                                counter[dim - 1] += 1;
                                step = dim - 1;
                                continue;
                            }
                            break;
                        }
                        pos_a = pos_a.wrapping_add_signed(corr_a[step]);
                        pos_b = pos_b.wrapping_add_signed(corr_b[step]);
                    }
                }
            }};
        }

        stamp_arms!(drive, data_a, data_b, n, inner_a, inner_b, f);
    }
}

//////////////////////////////////////////////////////////////////
// DimWalker - the reusable N-operand walk.

/// Collapse the shared shape, merging any run of dims that every operand walks
/// with the same adjacent stride. A dim boundary survives only where at least
/// one operand's adjacent stride changes - matching `simplify_duo_layout`, but
/// for `N` operands.
fn simplify_layout<const N: usize>(
    layouts: [&Layout; N],
    l_adj_strides: [[i32; MAX_DIMS]; N],
) -> (usize, [usize; MAX_DIMS], [[i32; MAX_DIMS]; N]) {
    let rank: usize = layouts[0].shape().len();
    let mut shape = [0usize; MAX_DIMS];
    let mut adj_strides = [[0i32; MAX_DIMS]; N];

    let mut w: usize = 0;

    shape[0] = layouts[0].shape()[0];
    for n in 0..N {
        adj_strides[n][0] = l_adj_strides[n][0];
    }

    for i in 1..rank {
        let mut mergeable = true;
        for n in 0..N {
            if l_adj_strides[n][i] != l_adj_strides[n][i - 1] {
                mergeable = false;
                break;
            }
        }

        if mergeable {
            shape[w] *= layouts[0].shape()[i];
        } else {
            w += 1;
            shape[w] = layouts[0].shape()[i];
            for n in 0..N {
                adj_strides[n][w] = l_adj_strides[n][i];
            }
        }
    }

    (w + 1, shape, adj_strides)
}

struct DimWalkerSpec<'a, const N: usize> {
    rank: usize,
    layouts: [&'a Layout; N],
    shape: [usize; MAX_DIMS],
    /// Per operand, per collapsed dim: the pre-folded advance the walk adds when
    /// it steps that dim. Column `last` holds the inner *element* stride (read
    /// by [`strides`](Self::strides), never by the walk); column `last - 1`
    /// holds the mid-loop row advance; columns `0..last - 1` hold the block
    /// corrections applied at odometer carries.
    baked_stride: [[isize; MAX_DIMS]; N],
    is_fully_contiguous: bool,
}

impl<'a> DimWalkerSpec<'a, 2> {
    /// Two-operand walk that mirrors `count`: the running positions are scalar
    /// locals (`pos0`/`pos1`, register-resident), the closure receives them as
    /// *separate scalar arguments* rather than a `[usize; 2]` array (an array in
    /// the closure signature is what forced the earlier variants to materialize
    /// the offsets to the stack), and the loop is rank-specialized - a constant-
    /// correction countdown at rank 3, the generic odometer only at rank 4+.
    #[inline]
    fn for_each_spec(&self, mut f: impl FnMut(usize, usize, usize)) {
        if self.is_fully_contiguous {
            f(
                self.layouts[0].offset(),
                self.layouts[1].offset(),
                self.layouts[0].len(),
            );
            return;
        }

        let last = self.rank - 1;
        let n = self.shape[last];
        let mid = self.shape[last - 1];
        let row_da0 = self.baked_stride[0][last - 1];
        let row_da1 = self.baked_stride[1][last - 1];
        let mut pos0 = self.layouts[0].offset();
        let mut pos1 = self.layouts[1].offset();
        let mut cd = mid;

        if self.rank == 3 {
            let corr0 = self.baked_stride[0][0];
            let corr1 = self.baked_stride[1][0];
            let chunks = self.shape[0] * mid;
            for _ in 0..chunks {
                f(pos0, pos1, n);
                pos0 = pos0.wrapping_add_signed(row_da0);
                pos1 = pos1.wrapping_add_signed(row_da1);
                cd -= 1;
                if cd == 0 {
                    cd = mid;
                    pos0 = pos0.wrapping_add_signed(corr0);
                    pos1 = pos1.wrapping_add_signed(corr1);
                }
            }
        } else {
            let mid_dim = last - 1;
            let chunks: usize = self.shape[0..last].iter().product();
            let mut counter: [usize; MAX_DIMS] = [0; MAX_DIMS];
            for _ in 0..chunks {
                f(pos0, pos1, n);
                pos0 = pos0.wrapping_add_signed(row_da0);
                pos1 = pos1.wrapping_add_signed(row_da1);
                cd -= 1;
                if cd == 0 {
                    cd = mid;
                    let mut step = mid_dim - 1;
                    counter[step] += 1;
                    for dim in (1..mid_dim).rev() {
                        if counter[dim] == self.shape[dim] {
                            counter[dim] = 0;
                            counter[dim - 1] += 1;
                            step = dim - 1;
                            continue;
                        }
                        break;
                    }
                    pos0 = pos0.wrapping_add_signed(self.baked_stride[0][step]);
                    pos1 = pos1.wrapping_add_signed(self.baked_stride[1][step]);
                }
            }
        }
    }
}

impl<'a, const N: usize> DimWalkerSpec<'a, N> {
    #[inline]
    fn new(layouts: [&'a Layout; N]) -> Self {
        if layouts.iter().all(|l| l.is_contiguous()) {
            // One row over the whole buffer; the row body is a contiguous copy,
            // so the inner stride must read back as 1 (not 0, which would pick
            // the splat arm).
            let rank = layouts[0].shape().len();
            let mut baked_stride = [[0isize; MAX_DIMS]; N];
            for n in 0..N {
                baked_stride[n][rank - 1] = 1;
            }
            return Self {
                rank,
                layouts,
                shape: [0; MAX_DIMS],
                baked_stride,
                is_fully_contiguous: true,
            };
        }

        let l_adj_strides = layouts.map(|l| calculate_adjacent_dim_stride(l.stride(), l.shape()));

        let (mut rank, mut shape, mut adj_strides) = simplify_layout(layouts, l_adj_strides);

        // Promote to at least rank 3 by prepending unit dims, so the walk always
        // has a block odometer (dims `0..last - 1`), a mid dim (`last - 1`) and
        // a row (`last`). Prepended dims have extent 1, so they never step.
        while rank < 3 {
            for d in (0..rank).rev() {
                shape[d + 1] = shape[d];
                for n in 0..N {
                    adj_strides[n][d + 1] = adj_strides[n][d];
                }
            }
            shape[0] = 1;
            for n in 0..N {
                adj_strides[n][0] = adj_strides[n][1];
            }
            rank += 1;
        }

        let last = rank - 1;
        let steps: [isize; N] =
            adj_strides.map(|adj| adj[last] as isize * (shape[last] - 1) as isize);

        let mut idx = 0usize;
        let baked_stride: [[isize; MAX_DIMS]; N] = adj_strides.map(|adj| {
            let mut temp = [0isize; MAX_DIMS];
            let step = steps[idx];
            let chunk_stride = adj[last - 1] as isize + step;

            temp[last] = adj[last] as isize; // inner element stride
            temp[last - 1] = chunk_stride; // mid-loop row advance
            for x in 0..last - 1 {
                temp[x] = adj[x] as isize + step - chunk_stride; // block correction
            }

            idx += 1;
            temp
        });

        Self {
            rank,
            layouts,
            shape,
            baked_stride,
            is_fully_contiguous: false,
        }
    }

    /// The per-operand inner *element* stride - what the caller matches on to
    /// pick a row-body specialization (`1` contiguous, `0` broadcast, anything
    /// else a strided gather).
    #[inline]
    fn strides(&self) -> [isize; N] {
        let last = self.rank - 1;
        self.baked_stride.map(|s| s[last])
    }
}

/// The DimWalker analogue of `apply_count`: `count`'s rank-specialized,
/// register-resident, scalar-argument walk driven by [`DimWalker::for_each_spec`].
/// The inner strides are matched once, outside the walk; every arm writes `out`
/// in walk order, which is logical row-major (collapsed dims descend in
/// significance).
fn apply_walker_spec(
    data_a: &[f32],
    la: &Layout,
    data_b: &[f32],
    lb: &Layout,
    out: &mut [f32],
    f: impl Fn(f32, f32) -> f32,
) {
    let walker = DimWalkerSpec::new([la, lb]);

    match walker.strides() {
        [1, 1] => {
            let mut i = 0usize;
            walker.for_each_spec(|p0, p1, len| {
                let a = &data_a[p0..p0 + len];
                let b = &data_b[p1..p1 + len];
                for ((o, x), y) in out[i..i + len].iter_mut().zip(a).zip(b) {
                    *o = f(*x, *y);
                }
                i += len;
            });
        }
        [0, 0] => {
            let mut i = 0usize;
            walker.for_each_spec(|p0, p1, len| {
                let v = f(data_a[p0], data_b[p1]);
                out[i..i + len].fill(v);
                i += len;
            });
        }
        [_, 0] => {
            let mut i = 0usize;
            walker.for_each_spec(|p0, p1, len| {
                let bv = data_b[p1];
                let a = &data_a[p0..p0 + len];
                for (o, x) in out[i..i + len].iter_mut().zip(a) {
                    *o = f(*x, bv);
                }
                i += len;
            });
        }
        [0, _] => {
            let mut i = 0usize;
            walker.for_each_spec(|p0, p1, len| {
                let av = data_a[p0];
                let b = &data_b[p1..p1 + len];
                for (o, y) in out[i..i + len].iter_mut().zip(b) {
                    *o = f(av, *y);
                }
                i += len;
            });
        }
        [sa, sb] => {
            let mut i = 0usize;
            walker.for_each_spec(|p0, p1, len| {
                let mut pa = p0;
                let mut pb = p1;
                for o in out[i..i + len].iter_mut() {
                    debug_assert!(pa < data_a.len());
                    debug_assert!(pb < data_b.len());
                    // SAFETY: a well-formed layout only ever visits in-bounds
                    // positions of its own buffer.
                    *o = unsafe { f(*data_a.get_unchecked(pa), *data_b.get_unchecked(pb)) };
                    pa = pa.wrapping_add_signed(sa);
                    pb = pb.wrapping_add_signed(sb);
                }
                i += len;
            });
        }
    }
}

//////////////////////////////////////////////////////////////////

struct DimWalker<'a, const N: usize> {
    rank: usize,
    layouts: [&'a Layout; N],
    shape: [usize; MAX_DIMS],
    baked_stride: [[isize; MAX_DIMS]; N],
    chunk_len: usize,
    is_fully_contiguous: bool,
}

impl<'a, const N: usize> DimWalker<'a, N> {
    fn new(layouts: [&'a Layout; N]) -> Self {
        if layouts.iter().all(|l| l.is_contiguous()) {
            let rank = layouts[0].shape().len();
            let baked_stride = [[1isize; MAX_DIMS]; N];

            return Self {
                rank,
                layouts,
                shape: [0; MAX_DIMS],
                baked_stride,
                chunk_len: layouts[0].len(),
                is_fully_contiguous: true,
            };
        }

        let l_adj_strides = layouts.map(|l| calculate_adjacent_dim_stride(l.stride(), l.shape()));

        let (mut rank, mut shape, mut adj_strides) = simplify_layout(layouts, l_adj_strides);

        if rank == 1 {
            shape[1] = shape[0];
            shape[0] = 1;

            for i in 0..N {
                adj_strides[i][1] = adj_strides[i][0];
            }

            rank += 1;
        }

        let last = rank - 1;

        let steps =
            adj_strides.map(|adj_stride| adj_stride[last] as isize * (shape[last] - 1) as isize);

        let mut i: usize = 0;
        let baked_stride: [[isize; MAX_DIMS]; N] = adj_strides.map(|adj_stride| {
            let mut temp = [0isize; MAX_DIMS];
            let step = steps[i];
            let chunk_stride = adj_stride[last - 1] as isize + step;
            temp[last] = adj_stride[last] as isize;
            temp[last - 1] = chunk_stride;

            for x in 0..rank - 2 {
                temp[x] = adj_stride[x] as isize + step - chunk_stride;
            }
            i += 1;

            temp
        });

        Self {
            rank,
            layouts,
            shape,
            baked_stride,
            chunk_len: shape[rank - 1],
            is_fully_contiguous: false,
        }
    }

    fn strides(&self) -> (usize, [isize; N]) {
        (
            self.chunk_len,
            self.baked_stride
                .map(|adj_stride| adj_stride[self.rank - 1]),
        )
    }

    fn fold<A>(&self, init: A, mut f: impl FnMut(A, [usize; N]) -> A) -> A {
        if self.is_fully_contiguous {
            return f(init, self.layouts.map(|l| l.offset()));
        }

        let last = self.rank - 1;
        let mut counter: [usize; MAX_DIMS] = [0; MAX_DIMS];

        let n_chunks = self.shape[last - 1];
        let chunk_stride: [isize; N] = self.baked_stride.map(|s| s[last - 1]);

        let mut offsets: [usize; N] = self.layouts.map(|l| l.offset());
        let mut acc = init;

        if self.rank == 2 {
            let left_over = self.shape[0];
            for _ in 0..left_over {
                acc = f(acc, offsets);

                for i in 0..N {
                    offsets[i] = offsets[i].wrapping_add_signed(chunk_stride[i]);
                }
            }
            return acc;
        }

        if self.rank == 3 {
            let chunks = self.shape[0] * n_chunks;
            let mut count = n_chunks;
            for _ in 0..chunks {
                acc = f(acc, offsets);

                for i in 0..N {
                    offsets[i] = offsets[i].wrapping_add_signed(chunk_stride[i]);
                }

                count -= 1;
                if count == 0 {
                    count = n_chunks;
                    for i in 0..N {
                        offsets[i] = offsets[i].wrapping_add_signed(self.baked_stride[i][0]);
                    }
                }
            }
            return acc;
        }

        let chunks: usize = self.shape[0..last].iter().product();
        let mut count = n_chunks;
        for _ in 0..chunks {
            acc = f(acc, offsets);

            for i in 0..N {
                offsets[i] = offsets[i].wrapping_add_signed(chunk_stride[i]);
            }

            count -= 1;
            if count == 0 {
                count = n_chunks;

                let last_counter = last - 2;
                counter[last_counter] += 1;
                let mut step_dim = last_counter;
                for dim in (1..last - 1).rev() {
                    if counter[dim] == self.shape[dim] {
                        counter[dim] = 0;
                        counter[dim - 1] += 1;
                        step_dim = dim - 1;
                        continue;
                    }
                    break;
                }

                for i in 0..N {
                    offsets[i] = offsets[i].wrapping_add_signed(self.baked_stride[i][step_dim]);
                }
            }
        }

        acc
    }

    fn for_each(&self, mut f: impl FnMut([usize; N])) {
        self.fold((), |(), offsets| f(offsets));
    }
}

/// Pull the next output chunk from a `ChunksExactMut` driven in lockstep with a
/// `DimWalker` walk. The walk emits exactly `out.len() / len` chunks, so the
/// iterator can never run dry mid-walk - the `unwrap` is elided in release. The
/// debug assert turns a walk/`out` mismatch into a clear panic instead of a
/// silent grab of a non-existent chunk.
#[inline(always)]
unsafe fn next_chunk<'a, T>(chunks: &mut std::slice::ChunksExactMut<'a, T>) -> &'a mut [T] {
    let chunk = chunks.next();
    debug_assert!(chunk.is_some(), "walk emitted more chunks than `out` holds");
    // SAFETY: the caller drives this iterator in lockstep with the walk, which
    // yields exactly as many chunks as `out` was split into.
    unsafe { chunk.unwrap_unchecked() }
}

fn zip2<T: Copy, F: Fn(T, T) -> T>(
    inp1: &[T],
    l1: &Layout,
    inp2: &[T],
    l2: &Layout,
    out: &mut [T],
    f: F,
) {
    let walker = DimWalker::new([l1, l2]);
    let (len, strides) = walker.strides();

    let mut chunks = out.chunks_exact_mut(len);
    match strides {
        [1, 1] => {
            walker.for_each(|offsets| {
                let chunk = unsafe { next_chunk(&mut chunks) };
                let it1 = inp1[offsets[0]..offsets[0] + len].iter();
                let it2 = inp2[offsets[1]..offsets[1] + len].iter();

                for (o, (x, y)) in chunk.iter_mut().zip(zip(it1, it2)) {
                    *o = f(*x, *y);
                }
            });
        }
        [0, 0] => {
            walker.for_each(|offsets| {
                let chunk = unsafe { next_chunk(&mut chunks) };
                chunk.fill(f(inp1[offsets[0]], inp2[offsets[1]]));
            });
        }
        [0, _] => {
            walker.for_each(|offsets| {
                let chunk = unsafe { next_chunk(&mut chunks) };
                let x = inp1[offsets[0]];

                for (o, y) in chunk
                    .iter_mut()
                    .zip(inp2[offsets[1]..offsets[1] + len].iter())
                {
                    *o = f(x, *y);
                }
            });
        }
        [_, 0] => {
            walker.for_each(|offsets| {
                let chunk = unsafe { next_chunk(&mut chunks) };
                let y = inp2[offsets[1]];
                for (o, x) in chunk
                    .iter_mut()
                    .zip(inp1[offsets[0]..offsets[0] + len].iter())
                {
                    *o = f(*x, y);
                }
            });
        }
        [s1, s2] => {
            walker.for_each(|offsets| {
                let chunk = unsafe { next_chunk(&mut chunks) };
                let mut pos1 = offsets[0];
                let mut pos2 = offsets[1];

                for o in chunk.iter_mut() {
                    debug_assert!(pos1 < inp1.len());
                    debug_assert!(pos2 < inp2.len());
                    // SAFETY: a well-formed layout only ever visits in-bounds
                    // positions of its own buffer.
                    *o = unsafe { f(*inp1.get_unchecked(pos1), *inp2.get_unchecked(pos2)) };
                    pos1 = pos1.wrapping_add_signed(s1);
                    pos2 = pos2.wrapping_add_signed(s2);
                }
            });
        }
    }

    // The walk must have consumed every chunk `out` was split into; a leftover
    // means the walk under-emitted (partially written `out`). Pairs with the
    // over-emission guard in `next_chunk`.
    debug_assert!(
        chunks.next().is_none(),
        "walk emitted fewer chunks than `out` holds"
    );
}

//////////////////////////////////////////////////////////////////

// Correctness + cases (harness shared with addressing.rs, trimmed to the two
// timed contenders).

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

fn required_len(shape: &[usize], stride: &[i32], offset: usize) -> usize {
    let mut lo = offset as isize;
    let mut hi = offset as isize;
    for (d, s) in zip(shape, stride) {
        let extent = (*d as isize - 1) * *s as isize;
        if extent >= 0 {
            hi += extent;
        } else {
            lo += extent;
        }
    }
    assert!(lo >= 0, "layout visits negative indices");
    hi as usize + 1
}

fn validate_battery() {
    type VCase = (
        &'static str,
        &'static [usize],
        &'static [i32],
        &'static [i32],
        usize,
        usize,
        usize,
    );
    // (name, shape, stride_a, stride_b, offset_a, offset_b, collapsed rank)
    const CASES: &[VCase] = &[
        ("r1_contig", &[97], &[1], &[1], 5, 9, 1),
        ("r1_strided_offset", &[97], &[3], &[2], 5, 11, 1),
        ("r1_bcast", &[97], &[0], &[1], 0, 3, 1),
        (
            "r2_both_contig_offset",
            &[7, 13],
            &[13, 1],
            &[13, 1],
            3,
            17,
            1,
        ),
        ("r2_contig_vs_padded", &[7, 13], &[13, 1], &[26, 1], 2, 9, 2),
        ("r2_transposed_b", &[7, 13], &[13, 1], &[1, 7], 0, 0, 2),
        ("r2_bcast_inner", &[7, 13], &[13, 1], &[1, 0], 1, 4, 2),
        ("r2_bcast_outer", &[7, 13], &[13, 1], &[0, 1], 0, 2, 2),
        ("r2_reversed_b", &[7, 13], &[13, 1], &[13, -1], 0, 12, 2),
        ("r2_all_scalar", &[5, 9], &[0, 0], &[0, 0], 3, 8, 1),
        (
            "r3_contig_vs_midbcast",
            &[3, 5, 7],
            &[35, 7, 1],
            &[35, 0, 1],
            0,
            6,
            3,
        ),
        (
            "r3_inner_bcast_collapses",
            &[3, 5, 7],
            &[35, 7, 1],
            &[35, 7, 0],
            4,
            0,
            2,
        ),
        (
            "r3_gather_b",
            &[3, 5, 7],
            &[35, 7, 1],
            &[80, 15, 2],
            0,
            1,
            3,
        ),
        (
            "r3_collapses_to_r2",
            &[3, 5, 7],
            &[75, 15, 1],
            &[75, 15, 1],
            0,
            0,
            2,
        ),
        (
            "r4_padded",
            &[3, 2, 5, 4],
            &[61, 29, 5, 1],
            &[61, 29, 5, 1],
            0,
            7,
            4,
        ),
        (
            "r4_bcast_mix",
            &[3, 2, 5, 4],
            &[40, 20, 4, 1],
            &[0, 20, 0, 1],
            2,
            0,
            4,
        ),
        (
            "r8_max_dims",
            &[2, 2, 2, 2, 2, 2, 2, 2],
            &[255, 127, 63, 31, 15, 7, 3, 1],
            &[255, 127, 63, 31, 15, 7, 3, 1],
            0,
            0,
            8,
        ),
    ];

    let mut rng = StdRng::seed_from_u64(SEED ^ 0xBA77E51);

    for &(name, shape, sa, sb, off_a, off_b, want_rank) in CASES {
        let la = Layout::from_strided(shape, sa, off_a);
        let lb = Layout::from_strided(shape, sb, off_b);

        let (rank, ..) = simplify_duo_layout(&la, &lb);
        assert_eq!(
            rank, want_rank,
            "{name}: collapse produced rank {rank}, case written for rank {want_rank}"
        );

        let data_a: Vec<f32> = (0..required_len(shape, sa, off_a))
            .map(|_| rng.random())
            .collect();
        let data_b: Vec<f32> = (0..required_len(shape, sb, off_b))
            .map(|_| rng.random())
            .collect();

        let expected = reference_mul(&data_a, &la, &data_b, &lb);
        let mut got: Vec<f32> = vec![0.0f32; expected.len()];

        apply_count(&data_a, &la, &data_b, &lb, &mut got, |x, y| x * y);
        assert_eq!(got, expected, "count wrong at {name}");

        got.fill(0.0);
        apply_walker_spec(&data_a, &la, &data_b, &lb, &mut got, |x, y| x * y);
        assert_eq!(got, expected, "walker_spec wrong at {name}");

        got.fill(0.0);
        zip2(&data_a, &la, &data_b, &lb, &mut got, |x, y| x * y);
        assert_eq!(got, expected, "walker_spec wrong at {name}");
    }
}

#[derive(Clone, Copy)]
enum Size {
    L1,
    L2,
    L3,
    Dram,
}

impl Size {
    fn budget(self) -> usize {
        match self {
            Size::L1 => *L1_BYTES,
            Size::L2 => *L2_BYTES,
            Size::L3 => *L3_BYTES,
            Size::Dram => 8 * *L3_BYTES,
        }
    }

    fn label(self) -> &'static str {
        match self {
            Size::L1 => "L1",
            Size::L2 => "L2",
            Size::L3 => "L3",
            Size::Dram => "DRAM",
        }
    }
}

struct Recipe {
    name: &'static str,
    sizes: &'static [Size],
    min_rank: usize,
    make: fn(usize) -> (Layout, Layout),
}

fn mk_r2_bcast_inner_n8(d: usize) -> (Layout, Layout) {
    (
        Layout::from_strided(&[d, 8], &[8, 1], 0),
        Layout::from_strided(&[d, 8], &[1, 0], 0),
    )
}

fn mk_r2_bcast_inner_n64(d: usize) -> (Layout, Layout) {
    (
        Layout::from_strided(&[d, 64], &[64, 1], 0),
        Layout::from_strided(&[d, 64], &[1, 0], 0),
    )
}

fn mk_r2_padded_n8(d: usize) -> (Layout, Layout) {
    (
        Layout::from_strided(&[d, 8], &[16, 1], 0),
        Layout::from_strided(&[d, 8], &[16, 1], 0),
    )
}

fn mk_r2_padded_n64(d: usize) -> (Layout, Layout) {
    (
        Layout::from_strided(&[d, 64], &[128, 1], 0),
        Layout::from_strided(&[d, 64], &[128, 1], 0),
    )
}

fn mk_r2_transposed(d: usize) -> (Layout, Layout) {
    (
        Layout::from_strided(&[d, d], &[1, d as i32], 0),
        Layout::from_strided(&[d, d], &[1, d as i32], 0),
    )
}

fn mk_r3_midtiny(d: usize) -> (Layout, Layout) {
    (
        Layout::from_strided(&[d, 2, 32], &[256, 64, 1], 0),
        Layout::from_strided(&[d, 2, 32], &[256, 64, 1], 0),
    )
}

fn mk_r6_tiny(d: usize) -> (Layout, Layout) {
    (
        Layout::from_strided(&[d, 2, 2, 2, 2, 4], &[2048, 512, 128, 32, 8, 1], 0),
        Layout::from_strided(&[d, 2, 2, 2, 2, 4], &[2048, 512, 128, 32, 8, 1], 0),
    )
}

fn mk_r5_dense_bcast(d: usize) -> (Layout, Layout) {
    (
        Layout::from_strided(&[d, 2, 2, 2, 4], &[32, 16, 8, 4, 1], 0),
        Layout::from_strided(&[d, 2, 2, 2, 4], &[8, 0, 4, 0, 1], 0),
    )
}

fn mk_r3_mid8(d: usize) -> (Layout, Layout) {
    (
        Layout::from_strided(&[d, 8, 32], &[1024, 64, 1], 0),
        Layout::from_strided(&[d, 8, 32], &[1024, 64, 1], 0),
    )
}

fn mk_r3_mid32(d: usize) -> (Layout, Layout) {
    (
        Layout::from_strided(&[d, 32, 32], &[4096, 64, 1], 0),
        Layout::from_strided(&[d, 32, 32], &[4096, 64, 1], 0),
    )
}

fn mk_r3_mid128(d: usize) -> (Layout, Layout) {
    (
        Layout::from_strided(&[d, 128, 32], &[16384, 64, 1], 0),
        Layout::from_strided(&[d, 128, 32], &[16384, 64, 1], 0),
    )
}

fn mk_r3_midlarge(d: usize) -> (Layout, Layout) {
    let s0 = (d as i32) * 256;
    (
        Layout::from_strided(&[2, d, 64], &[s0, 128, 1], 0),
        Layout::from_strided(&[2, d, 64], &[s0, 128, 1], 0),
    )
}

static RECIPES: &[Recipe] = &[
    Recipe {
        name: "r2_bcast_inner_n8",
        sizes: &[Size::L1, Size::L2, Size::Dram],
        min_rank: 2,
        make: mk_r2_bcast_inner_n8,
    },
    Recipe {
        name: "r2_bcast_inner_n64",
        sizes: &[Size::L1, Size::L2, Size::Dram],
        min_rank: 2,
        make: mk_r2_bcast_inner_n64,
    },
    Recipe {
        name: "r2_padded_n8",
        sizes: &[Size::L1, Size::L2, Size::Dram],
        min_rank: 2,
        make: mk_r2_padded_n8,
    },
    Recipe {
        name: "r2_padded_n64",
        sizes: &[Size::L1, Size::L2, Size::Dram],
        min_rank: 2,
        make: mk_r2_padded_n64,
    },
    Recipe {
        name: "r2_transposed",
        sizes: &[Size::L2, Size::L3, Size::Dram],
        min_rank: 2,
        make: mk_r2_transposed,
    },
    Recipe {
        name: "r3_midtiny",
        sizes: &[Size::L1, Size::L2, Size::Dram],
        min_rank: 3,
        make: mk_r3_midtiny,
    },
    Recipe {
        name: "r6_tiny",
        sizes: &[Size::L1, Size::L2, Size::Dram],
        min_rank: 6,
        make: mk_r6_tiny,
    },
    Recipe {
        name: "r5_dense_bcast",
        sizes: &[Size::L1, Size::L2, Size::Dram],
        min_rank: 5,
        make: mk_r5_dense_bcast,
    },
    Recipe {
        name: "r3_midlarge",
        sizes: &[Size::L2, Size::Dram],
        min_rank: 3,
        make: mk_r3_midlarge,
    },
    Recipe {
        name: "r3_mid8",
        sizes: &[Size::L2, Size::Dram],
        min_rank: 3,
        make: mk_r3_mid8,
    },
    Recipe {
        name: "r3_mid32",
        sizes: &[Size::L2, Size::Dram],
        min_rank: 3,
        make: mk_r3_mid32,
    },
    Recipe {
        name: "r3_mid128",
        sizes: &[Size::L2, Size::Dram],
        min_rank: 3,
        make: mk_r3_mid128,
    },
];

fn buffer_len(layout: &Layout) -> usize {
    layout.last() + 1
}

fn distinct_elems(layout: &Layout) -> usize {
    zip(layout.shape(), layout.stride())
        .map(|(d, s)| if *s == 0 { 1 } else { *d })
        .product()
}

fn footprint(make: fn(usize) -> (Layout, Layout), d: usize) -> usize {
    let (la, lb) = make(d);
    (buffer_len(&la) + buffer_len(&lb) + la.len()) * size_of::<f32>()
}

fn solve_d(make: fn(usize) -> (Layout, Layout), budget: usize) -> usize {
    if footprint(make, 2) >= budget {
        return 2;
    }
    let mut hi = 2usize;
    while footprint(make, hi) < budget {
        hi *= 2;
    }
    let mut lo = hi / 2;
    while hi - lo > 1 {
        let mid = (lo + hi) / 2;
        if footprint(make, mid) <= budget {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    lo.max(2)
}

fn walker(c: &mut Criterion) {
    validate_battery();

    let plot_config = PlotConfiguration::default().summary_scale(AxisScale::Logarithmic);

    let mut group = c.benchmark_group("walker");
    group.plot_config(plot_config);

    let mut rng = StdRng::seed_from_u64(SEED);

    for recipe in RECIPES {
        for &size in recipe.sizes {
            let d = solve_d(recipe.make, size.budget());
            let (la, lb) = (recipe.make)(d);
            let n = la.len();

            let (rank, ..) = simplify_duo_layout(&la, &lb);
            assert!(
                rank >= recipe.min_rank,
                "{}: collapse merged the case away (rank {} < {})",
                recipe.name,
                rank,
                recipe.min_rank
            );

            let data_a: Vec<f32> = (0..buffer_len(&la)).map(|_| rng.random()).collect();
            let data_b: Vec<f32> = (0..buffer_len(&lb)).map(|_| rng.random()).collect();

            let expected = reference_mul(&data_a, &la, &data_b, &lb);

            let mut got = vec![0.0f32; n];
            apply_count(&data_a, &la, &data_b, &lb, &mut got, |x, y| x * y);
            assert_eq!(got, expected, "count wrong at {}", recipe.name);

            got.fill(0.0);
            apply_walker_spec(&data_a, &la, &data_b, &lb, &mut got, |x, y| x * y);
            assert_eq!(got, expected, "walker_spec wrong at {}", recipe.name);

            got.fill(0.0);
            zip2(&data_a, &la, &data_b, &lb, &mut got, |x, y| x * y);
            assert_eq!(got, expected, "walker_spec wrong at {}", recipe.name);

            let label = format!("{}/{}", size.label(), recipe.name);
            let touched = distinct_elems(&la) + distinct_elems(&lb) + n;
            group.throughput(Throughput::Bytes((touched * size_of::<f32>()) as u64));

            let mut out = vec![0.0f32; n];

            group.bench_function(BenchmarkId::new("count", &label), |bencher| {
                bencher.iter(|| {
                    apply_count(
                        black_box(data_a.as_slice()),
                        black_box(&la),
                        black_box(data_b.as_slice()),
                        black_box(&lb),
                        &mut out,
                        |x, y| x * y,
                    );
                    black_box(&out);
                });
            });

            // group.bench_function(BenchmarkId::new("walker_spec", &label), |bencher| {
            //     bencher.iter(|| {
            //         apply_walker_spec(
            //             black_box(data_a.as_slice()),
            //             black_box(&la),
            //             black_box(data_b.as_slice()),
            //             black_box(&lb),
            //             &mut out,
            //             |x, y| x * y,
            //         );
            //         black_box(&out);
            //     });
            // });

            group.bench_function(BenchmarkId::new("new_walker_spec", &label), |bencher| {
                bencher.iter(|| {
                    zip2(
                        black_box(data_a.as_slice()),
                        black_box(&la),
                        black_box(data_b.as_slice()),
                        black_box(&lb),
                        &mut out,
                        |x, y| x * y,
                    );
                    black_box(&out);
                });
            });
        }
    }

    group.finish();
}

criterion_group!(benches, walker);
criterion_main!(benches);
