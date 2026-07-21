use candela::{
    Dimension, Layout,
    backend::{Backend, ComputeFor, CpuExperimental, CpuPure},
    skeleton::{Skeleton, SkeletonSlot},
    walker::map_chunk,
};
use criterion::measurement::WallTime;
use criterion::{
    AxisScale, BenchmarkGroup, BenchmarkId, Criterion, PlotConfiguration, criterion_group,
    criterion_main,
};
use std::hint::black_box;

mod common;
use common::{FillConfig, SizeSpec, Variant};

use crate::common::{bench_fill, fill_cases};

/// `&x * 2 + 1` over one external slot: a single `FusedScalar` the planner must
/// `Allocate` for, so it runs the out-of-place `compute_scalar`.
fn build_scalar_alloc<B: Backend>(layouts: &[Layout]) -> Skeleton<f32, B>
where
    f32: ComputeFor<B>,
{
    let x: SkeletonSlot<f32, B> = SkeletonSlot::new(layouts[0].clone());
    (&x * 2.0 + 1.0)
        .into_skeleton(std::slice::from_ref(&x))
        .unwrap()
}

/// `(a + b) * 2 + 1`: the `Add` allocates a buffer the `FusedScalar` overwrites
/// in place (`InPlaceIdx`), so the scalar runs `compute_scalar_inplace`. The
/// `Add`'s cost is included; subtract [`build_add_baseline`] to isolate it.
fn build_scalar_inplace<B: Backend>(layouts: &[Layout]) -> Skeleton<f32, B>
where
    f32: ComputeFor<B>,
{
    let a: SkeletonSlot<f32, B> = SkeletonSlot::new(layouts[0].clone());
    let b: SkeletonSlot<f32, B> = SkeletonSlot::new(layouts[1].clone());
    ((&a + &b) * 2.0 + 1.0).into_skeleton(&[a, b]).unwrap()
}

/// `a + b`: the upstream half of [`build_scalar_inplace`], on its own. The
/// in-place scalar kernel's cost is `inplace - add`.
fn build_add_baseline<B: Backend>(layouts: &[Layout]) -> Skeleton<f32, B>
where
    f32: ComputeFor<B>,
{
    let a: SkeletonSlot<f32, B> = SkeletonSlot::new(layouts[0].clone());
    let b: SkeletonSlot<f32, B> = SkeletonSlot::new(layouts[1].clone());
    (&a + &b).into_skeleton(&[a, b]).unwrap()
}

/// The vectorizable ceiling for the `alloc` path: `2x + 1` written straight into
/// `out`, no op interpreter, no fn-pointers. Contiguous inputs hit a plain slice
/// loop the compiler can vectorize; strided inputs (rank <= 2, all this bench
/// produces) walk their layout, so their rows still vectorize but row jumps do
/// not - the same shape a good kernel is bounded by. `data` is the input's
/// backing buffer, indexed through `layout`.
fn handrolled_scalar(data: &[f32], layout: &Layout, out: &mut [f32]) {
    if layout.is_contiguous() {
        let off = layout.offset();
        let n = out.len();
        for (o, &x) in out.iter_mut().zip(&data[off..off + n]) {
            *o = 2.0 * x + 1.0;
        }
        return;
    }

    match layout.shape().len() {
        1 => {
            let n = layout.shape()[0];
            let stride = layout.stride()[0] as isize;
            let mut p = layout.offset();
            for o in out.iter_mut().take(n) {
                *o = 2.0 * data[p] + 1.0;
                p = p.wrapping_add_signed(stride);
            }
        }
        2 => {
            let (rows, cols) = (layout.shape()[0], layout.shape()[1]);
            let (row_stride, col_stride) =
                (layout.stride()[0] as isize, layout.stride()[1] as isize);
            let mut row = layout.offset();
            let mut oi = 0;
            for _ in 0..rows {
                let mut p = row;
                for _ in 0..cols {
                    out[oi] = 2.0 * data[p] + 1.0;
                    oi += 1;
                    p = p.wrapping_add_signed(col_stride);
                }
                row = row.wrapping_add_signed(row_stride);
            }
        }
        r => panic!("handrolled baseline only handles rank <= 2, got {r}"),
    }
}

/// Times [`handrolled_scalar`] over the same cases as the `alloc` skeletons, so
/// its line sits directly against `alloc_new`/`alloc_old`. The output buffer is
/// reused across iterations: this is the raw kernel ceiling, deliberately
/// excluding the output allocation and `run` scaffolding (both measured in
/// `examples/scalaroverhead.rs`).
fn bench_handrolled(group: &mut BenchmarkGroup<'_, WallTime>, label: &str, cfg: &FillConfig) {
    for case in fill_cases::<f32, CpuExperimental, _>(cfg, build_scalar_alloc::<CpuExperimental>) {
        let out_len = case.skeleton.len();
        group.throughput(case.throughput);

        let input = &case.inputs[0];
        let data = input.data();
        let layout = input.layout().clone();
        let mut out = vec![0.0f32; out_len];

        group.bench_function(BenchmarkId::new(label, &case.label), |b| {
            b.iter(|| {
                handrolled_scalar(black_box(data), black_box(&layout), black_box(&mut out));
                black_box(&out);
            });
        });
    }
}

/// Like [`bench_handrolled`], but allocates a fresh, uninitialized output buffer
/// every iteration - exactly as `Skeleton::run` now does. Isolates the cost of
/// the allocation itself (page faults on cold pages, allocator churn on large
/// buffers) from the kernel: the delta against the reused `alloc_handrolled` is
/// what buffer reuse (an arena) would save, and it should sit right on top of
/// `alloc_new` if that gap really is allocation and not scaffolding.
fn bench_handrolled_fresh(group: &mut BenchmarkGroup<'_, WallTime>, label: &str, cfg: &FillConfig) {
    for case in fill_cases::<f32, CpuExperimental, _>(cfg, build_scalar_alloc::<CpuExperimental>) {
        let out_len = case.skeleton.len();
        group.throughput(case.throughput);

        let input = &case.inputs[0];
        let data = input.data();
        let layout = input.layout().clone();

        group.bench_function(BenchmarkId::new(label, &case.label), |b| {
            b.iter(|| {
                let mut out: Vec<f32> = Vec::with_capacity(out_len);
                // SAFETY: handrolled_scalar writes all `out_len` elements before
                // the black_box below reads them, so the uninitialized contents
                // are never observed - the same contract `run`'s output relies on.
                unsafe { out.set_len(out_len) };
                handrolled_scalar(black_box(data), black_box(&layout), black_box(&mut out));
                black_box(&out);
            });
        });
    }
}

fn bench_walker_direct(group: &mut BenchmarkGroup<'_, WallTime>, label: &str, cfg: &FillConfig) {
    for case in fill_cases::<f32, CpuExperimental, _>(cfg, build_scalar_alloc::<CpuExperimental>) {
        let out_len = case.skeleton.len();
        group.throughput(case.throughput);

        let input = &case.inputs[0];
        let data = input.data();
        let layout = input.layout().clone();

        group.bench_function(BenchmarkId::new(label, &case.label), |b| {
            b.iter(|| {
                let mut out: Vec<f32> = Vec::with_capacity(out_len);
                // SAFETY: the walk writes all `out_len` elements before the
                // black_box reads them - same contract as `run`'s output.
                unsafe { out.set_len(out_len) };
                map_chunk(
                    black_box(data),
                    black_box(&layout),
                    black_box(&mut out),
                    |src: &[f32], dst: &mut [f32]| {
                        for (o, x) in dst.iter_mut().zip(src) {
                            *o = 2.0 * *x + 1.0;
                        }
                    },
                    |x: f32| 2.0 * x + 1.0,
                );
                black_box(&out);
            });
        });
    }
}

fn alloc(c: &mut Criterion) {
    let plot_config = PlotConfiguration::default().summary_scale(AxisScale::Logarithmic);

    let mut group = c.benchmark_group("scalar_alloc");
    group.plot_config(plot_config);

    let sizes = &[
        SizeSpec::Elems(64),
        SizeSpec::Elems(1024),
        SizeSpec::L1,
        SizeSpec::L2,
        SizeSpec::L3,
        SizeSpec::Dram,
    ];

    // Out-of-place path: one input, all layout variants - this is where strided
    // input iteration is exercised.
    let alloc = FillConfig::new(1)
        .variants(&[
            Variant::Contig,
            // Variant::Step,
            // Variant::Padded,
            // Variant::Transposed,
        ])
        .sizes(sizes);

    bench_fill(
        &mut group,
        "alloc_new",
        &alloc,
        build_scalar_alloc::<CpuExperimental>,
    );
    bench_fill(
        &mut group,
        "alloc_old",
        &alloc,
        build_scalar_alloc::<CpuPure>,
    );
    bench_handrolled(&mut group, "alloc_handrolled", &alloc);
    bench_handrolled_fresh(&mut group, "alloc_handrolled_fresh", &alloc);
    bench_walker_direct(&mut group, "alloc_walker_direct", &alloc);

    group.finish();
}

fn inplace(c: &mut Criterion) {
    let plot_config = PlotConfiguration::default().summary_scale(AxisScale::Logarithmic);

    let mut group = c.benchmark_group("scalar_inplace");
    group.plot_config(plot_config);

    let sizes = &[
        SizeSpec::Elems(64),
        SizeSpec::Elems(1024),
        SizeSpec::L1,
        SizeSpec::L2,
        SizeSpec::L3,
        SizeSpec::Dram,
    ];

    let pair = FillConfig::new(2).variants(&[Variant::Contig]).sizes(sizes);

    bench_fill(
        &mut group,
        "inplace_new",
        &pair,
        build_scalar_inplace::<CpuExperimental>,
    );
    bench_fill(
        &mut group,
        "inplace_old",
        &pair,
        build_scalar_inplace::<CpuPure>,
    );

    group.finish();
}

fn add(c: &mut Criterion) {
    let plot_config = PlotConfiguration::default().summary_scale(AxisScale::Logarithmic);

    let mut group = c.benchmark_group("scalar_add");
    group.plot_config(plot_config);

    let sizes = &[
        SizeSpec::Elems(64),
        SizeSpec::Elems(1024),
        SizeSpec::L1,
        SizeSpec::L2,
        SizeSpec::L3,
        SizeSpec::Dram,
    ];

    let pair = FillConfig::new(2).variants(&[Variant::Contig]).sizes(sizes);

    bench_fill(
        &mut group,
        "add_new",
        &pair,
        build_add_baseline::<CpuExperimental>,
    );
    bench_fill(&mut group, "add_old", &pair, build_add_baseline::<CpuPure>);

    group.finish();
}

criterion_group!(benches, alloc, inplace, add);
criterion_main!(benches);
