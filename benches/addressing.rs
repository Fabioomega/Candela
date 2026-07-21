//! Chunk addressing: adjusted-stride accumulation vs raw-stride affine
//! recomputation, isolated from everything else.
//!
//! Every contender walks the same simplified (collapsed) layout row by row and
//! uses byte-identical row bodies; the only thing that differs is how each
//! row's start position is produced:
//!
//! - `adj`: one running position per operand, advanced by the adjusted stride
//!   of whichever dimension the odometer carried into. No multiplies, no
//!   divisions - but the position is a serial register chain threaded through
//!   a carry loop whose branches the compiler cannot unroll through.
//! - `raw`: the second-innermost dimension is an affine hot loop
//!   (`base + j * stride`), so consecutive row starts are mutually
//!   independent; the block base above it is re-derived from the block index
//!   with div/mod (runtime divisors) once per block.
//! - `hybrid`: affine hot loop like `raw`, but block bases advance by
//!   accumulation like `adj` - no div/mod anywhere, and the serial chain only
//!   ticks once per block instead of once per row.
//! - `nested`: adjusted-stride accumulation like `adj`, but the loop nest is
//!   rank-specialized - dedicated rank-2 and rank-3 nests, and a register mid
//!   loop with a per-block odometer above rank 3 - so the walking state is
//!   plain scalars in registers instead of a runtime-indexed stack array.
//!   (The emitted asm showed `adj`'s generic odometer round-trips its counter
//!   through the stack every row; that memory serialization, not the
//!   accumulation itself, is what `adj` pays at rank 2.)
//! - `count`: `adj`'s flat single-loop shape - which beat `nested`'s two-level
//!   nest wherever the mid extent was tiny - but with the odometer's innermost
//!   counter replaced by a register countdown and the block correction folded
//!   to a constant (rank 3) or a per-dim lookup at 1/mid frequency (rank 4+).
//!   One loop, register state, no division: the bet is that loop shape and
//!   register residency were each worth something, and this takes both.
//!
//! The recipes pull the schemes apart on purpose. Rank-2 cases reproduce the
//! elementwise sweeps - there `raw` never divides (`outer == 1`), which is its
//! best possible case. Higher-rank cases with tiny mid extents charge `raw`'s
//! per-block decomposition against very few rows. Gather and broadcast rows
//! vary how much row-level memory parallelism independent addressing could
//! unlock. Note the adj chain itself is register-only arithmetic - it never
//! waits on a load - so wherever `adj` loses, the suspect is the carry-loop
//! branches / lost unrolling, not load latency stacking on the chain.

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
// Layout collapse - copied from the zip bench so the timed code sees the
// exact same simplified geometry.

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

/// Same collapse, but returns *raw* strides per merged dim. A merged group's
/// raw stride is its innermost original stride - stepping the combined
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

//////////////////////////////////////////////////////////////////
// The contenders. Row bodies are identical across all of them - the
// comparison is about row-start addressing only - so the five inner-stride
// bodies live in `stamp_arms!` exactly once, and each driver provides its
// loop structure as a local `drive!` macro taking `(row_out, pa, pb, body)`.
// The inner-stride kind is loop-invariant, so it is matched once per call;
// each arm gets its own vectorized row loop.

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

/// Raw-affine addressing: `pos = block_base + j * mid_stride`, block base
/// decomposed from the block index with div/mod once per block.
///
/// `inline(never)` on all three drivers keeps each one a standalone symbol:
/// the timed closures cannot specialize them against the surrounding loop,
/// and the emitted asm stays comparable across contenders.
#[inline(never)]
fn apply_raw(
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

    let (mut rank, mut shape, mut raw_a, mut raw_b) = simplify_duo_layout_raw(la, lb);

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

    let mid_dim = last - 1;
    let mid = shape[mid_dim];
    let mid_a = raw_a[mid_dim] as isize;
    let mid_b = raw_b[mid_dim] as isize;
    let outer: usize = shape[0..mid_dim].iter().product();

    let off_a = la.offset();
    let off_b = lb.offset();

    let block_base = |blk: usize| -> (usize, usize) {
        let mut pa = off_a;
        let mut pb = off_b;
        let mut rem = blk;
        for k in (0..mid_dim).rev() {
            let ik = rem % shape[k];
            rem /= shape[k];
            pa = pa.wrapping_add_signed(ik as isize * raw_a[k] as isize);
            pb = pb.wrapping_add_signed(ik as isize * raw_b[k] as isize);
        }
        (pa, pb)
    };

    let mut out_chunks = out.chunks_exact_mut(n);

    macro_rules! drive {
        ($row_out:ident, $pa:ident, $pb:ident, $row:block) => {{
            for blk in 0..outer {
                let (base_a, base_b) = block_base(blk);
                for j in 0..mid {
                    let $row_out = out_chunks.next().unwrap();
                    let $pa = base_a.wrapping_add_signed(j as isize * mid_a);
                    let $pb = base_b.wrapping_add_signed(j as isize * mid_b);
                    $row
                }
            }
        }};
    }

    stamp_arms!(drive, data_a, data_b, n, inner_a, inner_b, f);
}

/// Adjusted-stride addressing: one running position per operand, advanced by
/// `adj_stride[step_dim]` after each row, where `step_dim` comes from an
/// odometer carry loop.
#[inline(never)]
fn apply_adj(
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
    let mut counter: [usize; MAX_DIMS] = [0; MAX_DIMS];
    let mut pos_a = la.offset();
    let mut pos_b = lb.offset();
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

    stamp_arms!(drive, data_a, data_b, n, inner_a, inner_b, f);
}

/// [`apply_adj`] with the per-row delta pre-folded: `delta[k] = adj[k] + step`
/// is baked into one array per operand at walk start, so the row advance is a
/// single indexed load instead of an i32 load, a sign extend, and a scalar
/// add. The odometer still fires every row. This isolates how much of `adj`'s
/// per-row cost is the redundant arithmetic (prediction: almost none) versus
/// the counter round trip and the indexed load themselves (prediction: almost
/// all) - the part `count` demotes to 1/mid frequency instead of removing.
#[inline(never)]
fn apply_adj_baked(
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
    let mut counter: [usize; MAX_DIMS] = [0; MAX_DIMS];
    let mut pos_a = la.offset();
    let mut pos_b = lb.offset();
    let n = shape[last];
    let inner_a = adj_a[last] as isize;
    let inner_b = adj_b[last] as isize;
    let step_a = inner_a * (n - 1) as isize;
    let step_b = inner_b * (n - 1) as isize;

    let mut delta_a = [0isize; MAX_DIMS];
    let mut delta_b = [0isize; MAX_DIMS];
    for k in 0..last {
        delta_a[k] = adj_a[k] as isize + step_a;
        delta_b[k] = adj_b[k] as isize + step_b;
    }

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
                pos_a = pos_a.wrapping_add_signed(delta_a[step_dim]);
                pos_b = pos_b.wrapping_add_signed(delta_b[step_dim]);
            }
        }};
    }

    stamp_arms!(drive, data_a, data_b, n, inner_a, inner_b, f);
}

/// Affine hot loop like [`apply_raw`], accumulated block bases like
/// [`apply_adj`]: rows within a block stay independent (`base + j * stride`),
/// but the block base is advanced with block-local adjusted strides instead of
/// being decomposed with div/mod. The serial chain ticks once per block, and
/// nothing ever divides.
#[inline(never)]
fn apply_hybrid(
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

    let (mut rank, mut shape, mut raw_a, mut raw_b) = simplify_duo_layout_raw(la, lb);

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

    let mid_dim = last - 1;
    let mid = shape[mid_dim];
    let mid_a = raw_a[mid_dim] as isize;
    let mid_b = raw_b[mid_dim] as isize;
    let outer: usize = shape[0..mid_dim].iter().product();

    // Adjusted strides restricted to the block dims: stepping block dim `k`
    // rewinds the block dims inside it, but never the mid dim or the row -
    // the affine hot loop never moved the base, so there is nothing to unwind.
    let mut blk_adj_a = [0isize; MAX_DIMS];
    let mut blk_adj_b = [0isize; MAX_DIMS];
    for k in 0..mid_dim {
        let mut rew_a = 0isize;
        let mut rew_b = 0isize;
        for j in k + 1..mid_dim {
            rew_a += (shape[j] - 1) as isize * raw_a[j] as isize;
            rew_b += (shape[j] - 1) as isize * raw_b[j] as isize;
        }
        blk_adj_a[k] = raw_a[k] as isize - rew_a;
        blk_adj_b[k] = raw_b[k] as isize - rew_b;
    }

    let mut bcounter: [usize; MAX_DIMS] = [0; MAX_DIMS];
    let mut base_a = la.offset();
    let mut base_b = lb.offset();
    let mut out_chunks = out.chunks_exact_mut(n);

    macro_rules! drive {
        ($row_out:ident, $pa:ident, $pb:ident, $row:block) => {{
            for _ in 0..outer {
                for j in 0..mid {
                    let $row_out = out_chunks.next().unwrap();
                    let $pa = base_a.wrapping_add_signed(j as isize * mid_a);
                    let $pb = base_b.wrapping_add_signed(j as isize * mid_b);
                    $row
                }
                if mid_dim > 0 {
                    let mut step = mid_dim - 1;
                    bcounter[step] += 1;
                    for dim in (1..mid_dim).rev() {
                        if bcounter[dim] == shape[dim] {
                            bcounter[dim] = 0;
                            bcounter[dim - 1] += 1;
                            step = dim - 1;
                            continue;
                        }
                        break;
                    }
                    base_a = base_a.wrapping_add_signed(blk_adj_a[step]);
                    base_b = base_b.wrapping_add_signed(blk_adj_b[step]);
                }
            }
        }};
    }

    stamp_arms!(drive, data_a, data_b, n, inner_a, inner_b, f);
}

/// Rank-specialized nested loops with adjusted-stride accumulation.
///
/// Same no-division walk as [`apply_adj`], but the flat chunk loop plus
/// generic odometer is replaced by a loop nest chosen from the collapsed
/// rank, so every position and delta is a scalar the compiler can keep in a
/// register:
///
/// - rank 2: one loop; the per-row delta `adj[0] + adj[1]*(n-1)` is a
///   constant, so the position is a plain induction variable.
/// - rank 3: two loops; rows advance by a constant inside a block, block
///   bases advance by a constant outside. No odometer at all.
/// - rank 4+: register loop over the mid dim, generic odometer once per
///   block - 1/mid the frequency the flat walk pays it, on the only path
///   where collapse still leaves that many dims.
///
/// Splitting for parallelism stays possible: a one-time seek decomposes a
/// linear index into block/mid/row coordinates, and the walk resumes from
/// there with the same constants.
#[inline(never)]
fn apply_nested(
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
        // A block boundary steps dim 0 after (mid - 1) row advances, so the
        // block-to-block base delta folds both into one constant.
        let blk_da = (mid - 1) as isize * row_da + adj_a[0] as isize + step_a;
        let blk_db = (mid - 1) as isize * row_db + adj_b[0] as isize + step_b;
        let mut bpos_a = la.offset();
        let mut bpos_b = lb.offset();
        let mut out_chunks = out.chunks_exact_mut(n);

        macro_rules! drive {
            ($row_out:ident, $pa:ident, $pb:ident, $row:block) => {{
                for _ in 0..shape[0] {
                    let mut pos_a = bpos_a;
                    let mut pos_b = bpos_b;
                    for _ in 0..mid {
                        let $row_out = out_chunks.next().unwrap();
                        let $pa = pos_a;
                        let $pb = pos_b;
                        $row
                        pos_a = pos_a.wrapping_add_signed(row_da);
                        pos_b = pos_b.wrapping_add_signed(row_db);
                    }
                    bpos_a = bpos_a.wrapping_add_signed(blk_da);
                    bpos_b = bpos_b.wrapping_add_signed(blk_db);
                }
            }};
        }

        stamp_arms!(drive, data_a, data_b, n, inner_a, inner_b, f);
    } else {
        let mid_dim = last - 1;
        let mid = shape[mid_dim];
        let row_da = adj_a[mid_dim] as isize + step_a;
        let row_db = adj_b[mid_dim] as isize + step_b;
        // Per-dimension block-base deltas: crossing into dim k after a full
        // block folds the (mid - 1) row advances and the dim-k adjusted step
        // into one add, exactly like the rank-3 constant but one per dim.
        let mut blk_da = [0isize; MAX_DIMS];
        let mut blk_db = [0isize; MAX_DIMS];
        for k in 0..mid_dim {
            blk_da[k] = (mid - 1) as isize * row_da + adj_a[k] as isize + step_a;
            blk_db[k] = (mid - 1) as isize * row_db + adj_b[k] as isize + step_b;
        }
        let blocks: usize = shape[0..mid_dim].iter().product();
        let mut counter: [usize; MAX_DIMS] = [0; MAX_DIMS];
        let mut bpos_a = la.offset();
        let mut bpos_b = lb.offset();
        let mut out_chunks = out.chunks_exact_mut(n);

        macro_rules! drive {
            ($row_out:ident, $pa:ident, $pb:ident, $row:block) => {{
                for _ in 0..blocks {
                    let mut pos_a = bpos_a;
                    let mut pos_b = bpos_b;
                    for _ in 0..mid {
                        let $row_out = out_chunks.next().unwrap();
                        let $pa = pos_a;
                        let $pb = pos_b;
                        $row
                        pos_a = pos_a.wrapping_add_signed(row_da);
                        pos_b = pos_b.wrapping_add_signed(row_db);
                    }
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
                    bpos_a = bpos_a.wrapping_add_signed(blk_da[step]);
                    bpos_b = bpos_b.wrapping_add_signed(blk_db[step]);
                }
            }};
        }

        stamp_arms!(drive, data_a, data_b, n, inner_a, inner_b, f);
    }
}

/// Flat chunk loop with a register countdown: `adj`'s single-loop shape with
/// `nested`'s register-resident state.
///
/// One loop over every chunk, like [`apply_adj`]; but where `adj` runs the
/// generic odometer every row (stack-resident counter, dynamically indexed
/// stride arrays), the countdown keeps one register ticking from `mid` to 0
/// and only touches block bookkeeping when it hits zero:
///
/// - rank 2: identical to [`apply_nested`] - a pure induction variable.
/// - rank 3: `pos += row_delta` every row; on countdown expiry add a single
///   precomputed correction constant. No arrays, no odometer, one
///   predictable branch.
/// - rank 4+: same flat loop; expiry consults the odometer over the dims
///   above mid and adds a per-dim correction - the stack round trip survives
///   but at 1/mid the frequency, without giving up the flat loop shape.
#[inline(never)]
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
        // After `mid` uncorrected row advances the position sits one row_delta
        // past the block's last row; the correction lands it on the next
        // block's first row instead.
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
        // Per-dimension corrections, same folding as the rank-3 constant.
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

/// The `count` walk written flat and plain: no `drive!` macro, no broadcast
/// specializations, no dedicated rank arms - and no `adj_stride`. This is the
/// template for the `iter.rs` port; the macro version above is this plus
/// per-arm stamping and a rank-2 arm that skips the (nearly free) countdown.
///
/// Raw strides are all the walk needs, because everything telescopes:
///
/// - Advancing one row is `pos += stride[mid]`. The adjusted-stride form
///   (`adj[mid] + (n-1)*stride[last]`) folds a rewind in and then adds it
///   back out - the row body never moves `pos`, so there is nothing to
///   rewind.
/// - After `mid` uncorrected advances the position overshoots the block by
///   one whole mid extent. The correction into block dim `k` is NumPy's
///   backstride algebra: `stride[k] - Σ backstride[k+1..mid] - mid*stride[mid]`
///   where `backstride[j] = (shape[j]-1)*stride[j]`, all precomputed.
///
/// Rank needs no dispatch here: at rank 2 there are no block dims, so the
/// countdown expires exactly once - after the final row - and the guard makes
/// it a no-op. Both-contiguous inputs collapse to rank 1, which promotes to a
/// single row covering the whole buffer; the production early-out is just a
/// shortcut past the same result.
///
/// The `stride == 0` splat arms of the macro version are performance
/// specializations only - the strided row loop below handles broadcast
/// operands correctly (`ia += 0` re-reads the same element), it merely
/// doesn't hoist the reload out of the loop.
#[inline(never)]
fn apply_count_plain(
    data_a: &[f32],
    la: &Layout,
    data_b: &[f32],
    lb: &Layout,
    out: &mut [f32],
    f: impl Fn(f32, f32) -> f32,
) {
    let (mut rank, mut shape, mut raw_a, mut raw_b) = simplify_duo_layout_raw(la, lb);

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

    let mid_dim = last - 1;
    let mid = shape[mid_dim];
    let row_da = raw_a[mid_dim] as isize;
    let row_db = raw_b[mid_dim] as isize;

    let mut corr_a = [0isize; MAX_DIMS];
    let mut corr_b = [0isize; MAX_DIMS];
    for k in 0..mid_dim {
        let mut back_a = 0isize;
        let mut back_b = 0isize;
        for j in k + 1..mid_dim {
            back_a += (shape[j] - 1) as isize * raw_a[j] as isize;
            back_b += (shape[j] - 1) as isize * raw_b[j] as isize;
        }
        corr_a[k] = raw_a[k] as isize - back_a - mid as isize * row_da;
        corr_b[k] = raw_b[k] as isize - back_b - mid as isize * row_db;
    }

    let mut counter: [usize; MAX_DIMS] = [0; MAX_DIMS];
    let mut pos_a = la.offset();
    let mut pos_b = lb.offset();
    let mut cd = mid;

    if inner_a == 1 && inner_b == 1 {
        for row_out in out.chunks_exact_mut(n) {
            for ((o, x), y) in row_out
                .iter_mut()
                .zip(data_a[pos_a..pos_a + n].iter())
                .zip(data_b[pos_b..pos_b + n].iter())
            {
                *o = f(*x, *y);
            }

            pos_a = pos_a.wrapping_add_signed(row_da);
            pos_b = pos_b.wrapping_add_signed(row_db);

            cd -= 1;
            if cd == 0 {
                cd = mid;
                if mid_dim > 0 {
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
        }
    } else {
        for row_out in out.chunks_exact_mut(n) {
            let mut ia = pos_a;
            let mut ib = pos_b;
            for o in row_out.iter_mut() {
                debug_assert!(ia < data_a.len());
                debug_assert!(ib < data_b.len());
                // SAFETY: a well-formed layout only ever visits in-bounds
                // positions of its own buffer.
                *o = unsafe { f(*data_a.get_unchecked(ia), *data_b.get_unchecked(ib)) };
                ia = ia.wrapping_add_signed(inner_a);
                ib = ib.wrapping_add_signed(inner_b);
            }

            pos_a = pos_a.wrapping_add_signed(row_da);
            pos_b = pos_b.wrapping_add_signed(row_db);

            cd -= 1;
            if cd == 0 {
                cd = mid;
                if mid_dim > 0 {
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
        }
    }
}

//////////////////////////////////////////////////////////////////
// Cases.

/// Naive reference: `out` in logical row-major order, each element gathered
/// straight from the multi-index. Independent of any chunking/decomposition.
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

/// Smallest buffer that holds every position `layout` can visit. Supports
/// negative strides: the walk then descends below the offset, and the lowest
/// visited index must still be in bounds.
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

/// Exhaustive correctness battery, run once before any timing. Covers every
/// driver path the sweep's recipes don't reach: rank-1 promotion, the
/// both-contiguous early-out, nonzero offsets, negative strides, layouts that
/// collapse (to rank 1 and rank 2), genuine rank-4 block decomposition, and
/// the full MAX_DIMS odometer depth. Each case pins the rank the collapse
/// must produce, so a change to the merge rule cannot silently reroute a case
/// onto a different specialization than the one it was written to exercise.
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
        let mut got = vec![0.0f32; expected.len()];

        apply_raw(&data_a, &la, &data_b, &lb, &mut got, |x, y| x * y);
        assert_eq!(got, expected, "raw wrong at {name}");

        got.fill(0.0);
        apply_adj(&data_a, &la, &data_b, &lb, &mut got, |x, y| x * y);
        assert_eq!(got, expected, "adj wrong at {name}");

        got.fill(0.0);
        apply_hybrid(&data_a, &la, &data_b, &lb, &mut got, |x, y| x * y);
        assert_eq!(got, expected, "hybrid wrong at {name}");

        got.fill(0.0);
        apply_nested(&data_a, &la, &data_b, &lb, &mut got, |x, y| x * y);
        assert_eq!(got, expected, "nested wrong at {name}");

        got.fill(0.0);
        apply_count(&data_a, &la, &data_b, &lb, &mut got, |x, y| x * y);
        assert_eq!(got, expected, "count wrong at {name}");

        got.fill(0.0);
        apply_count_plain(&data_a, &la, &data_b, &lb, &mut got, |x, y| x * y);
        assert_eq!(got, expected, "count_plain wrong at {name}");

        got.fill(0.0);
        apply_adj_baked(&data_a, &la, &data_b, &lb, &mut got, |x, y| x * y);
        assert_eq!(got, expected, "adj_baked wrong at {name}");
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
            // 8x L3: first rung solidly on the bandwidth plateau (see the
            // harness's NL3 sweep note).
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

/// One layout recipe, parameterized by a free dimension `d` that the size
/// solver scales until the physical footprint (both buffers + output) hits the
/// rung's byte budget.
struct Recipe {
    name: &'static str,
    sizes: &'static [Size],
    /// The collapse must keep at least this many dims, or the case would
    /// silently degrade into a different (easier) one.
    min_rank: usize,
    make: fn(usize) -> (Layout, Layout),
}

/// Rank 2, contiguous x inner-broadcast: the case the elementwise sweeps
/// measured. `raw` never divides at rank 2 (`outer == 1`) - its best case.
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

/// Rank 2, both padded (row pitch 2x): contiguous rows, chunk-advance cost
/// scales inversely with row length.
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

/// Rank 2 square transpose: gather rows, where independent row addresses are
/// the memory-level-parallelism question.
fn mk_r2_transposed(d: usize) -> (Layout, Layout) {
    (
        Layout::from_strided(&[d, d], &[1, d as i32], 0),
        Layout::from_strided(&[d, d], &[1, d as i32], 0),
    )
}

/// Rank 3 with mid extent 2: `raw` decomposes a block base every 2 rows.
fn mk_r3_midtiny(d: usize) -> (Layout, Layout) {
    (
        Layout::from_strided(&[d, 2, 32], &[256, 64, 1], 0),
        Layout::from_strided(&[d, 2, 32], &[256, 64, 1], 0),
    )
}

/// Rank 6, every level padded so nothing collapses: `raw` pays 4 runtime-
/// divisor mod/div pairs per block of 8 elements. The affine scheme's worst
/// case by construction.
fn mk_r6_tiny(d: usize) -> (Layout, Layout) {
    (
        Layout::from_strided(&[d, 2, 2, 2, 2, 4], &[2048, 512, 128, 32, 8, 1], 0),
        Layout::from_strided(&[d, 2, 2, 2, 2, 4], &[2048, 512, 128, 32, 8, 1], 0),
    )
}

/// Rank 5, dense: contiguous operand against one broadcast on alternating
/// dims (the bias-against-activations shape of an NCHW-ish network). The
/// mixed adj strides block every merge, so this is the realistic high-rank
/// non-collapsible case.
fn mk_r5_dense_bcast(d: usize) -> (Layout, Layout) {
    (
        Layout::from_strided(&[d, 2, 2, 2, 4], &[32, 16, 8, 4, 1], 0),
        Layout::from_strided(&[d, 2, 2, 2, 4], &[8, 0, 4, 0, 1], 0),
    )
}

/// Mid-extent sweep between `r3_midtiny` (mid = 2) and `r3_midlarge`
/// (mid = d): same padded rank-3 family with n = 32 rows, mid pinned. Where
/// the nested/flat crossover sits along this axis is what decides whether a
/// rank-3 dispatch needs a mid-size threshold at all.
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

/// Rank 3 with a huge mid extent: `raw`'s decomposition is amortized over `d`
/// rows - the control where affine addressing should be safe.
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

/// Elements the backing buffer must hold to satisfy `layout`.
fn buffer_len(layout: &Layout) -> usize {
    layout.last() + 1
}

/// Distinct elements a full traversal reads - stride-0 axes contribute 1.
fn distinct_elems(layout: &Layout) -> usize {
    zip(layout.shape(), layout.stride())
        .map(|(d, s)| if *s == 0 { 1 } else { *d })
        .product()
}

/// Physical working set: both buffers plus the output, in bytes.
fn footprint(make: fn(usize) -> (Layout, Layout), d: usize) -> usize {
    let (la, lb) = make(d);
    (buffer_len(&la) + buffer_len(&lb) + la.len()) * size_of::<f32>()
}

/// Largest `d` whose footprint stays within `budget` (footprint is monotone in
/// `d` for every recipe here, affine for most, quadratic for the transpose).
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

fn addressing(c: &mut Criterion) {
    validate_battery();

    let plot_config = PlotConfiguration::default().summary_scale(AxisScale::Logarithmic);

    let mut group = c.benchmark_group("addressing");
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
            apply_raw(&data_a, &la, &data_b, &lb, &mut got, |x, y| x * y);
            assert_eq!(got, expected, "raw wrong at {}", recipe.name);

            got.fill(0.0);
            apply_adj(&data_a, &la, &data_b, &lb, &mut got, |x, y| x * y);
            assert_eq!(got, expected, "adj wrong at {}", recipe.name);

            got.fill(0.0);
            apply_hybrid(&data_a, &la, &data_b, &lb, &mut got, |x, y| x * y);
            assert_eq!(got, expected, "hybrid wrong at {}", recipe.name);

            got.fill(0.0);
            apply_nested(&data_a, &la, &data_b, &lb, &mut got, |x, y| x * y);
            assert_eq!(got, expected, "nested wrong at {}", recipe.name);

            got.fill(0.0);
            apply_count(&data_a, &la, &data_b, &lb, &mut got, |x, y| x * y);
            assert_eq!(got, expected, "count wrong at {}", recipe.name);

            let label = format!("{}/{}", size.label(), recipe.name);
            let touched = distinct_elems(&la) + distinct_elems(&lb) + n;
            group.throughput(Throughput::Bytes((touched * size_of::<f32>()) as u64));

            let mut out = vec![0.0f32; n];

            group.bench_function(BenchmarkId::new("raw", &label), |bencher| {
                bencher.iter(|| {
                    apply_raw(
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

            group.bench_function(BenchmarkId::new("adj", &label), |bencher| {
                bencher.iter(|| {
                    apply_adj(
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

            group.bench_function(BenchmarkId::new("hybrid", &label), |bencher| {
                bencher.iter(|| {
                    apply_hybrid(
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

            group.bench_function(BenchmarkId::new("nested", &label), |bencher| {
                bencher.iter(|| {
                    apply_nested(
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

            group.bench_function(BenchmarkId::new("adj_baked", &label), |bencher| {
                bencher.iter(|| {
                    apply_adj_baked(
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
        }
    }

    group.finish();
}

criterion_group!(benches, addressing);
criterion_main!(benches);
