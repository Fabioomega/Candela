use candela::{
    Dimension, Layout,
    backend::{Backend, ComputeFor, CpuExperimental},
    skeleton::{Skeleton, SkeletonSlot},
};
use criterion::measurement::WallTime;
use criterion::{
    AxisScale, BenchmarkGroup, BenchmarkId, Criterion, PlotConfiguration, criterion_group,
    criterion_main,
};
use std::hint::black_box;
use wide::f32x8;

mod common;
use common::{FillConfig, SizeSpec, Variant, bench_fill, fill_cases};

fn build_chain<B: Backend>(layouts: &[Layout]) -> Skeleton<f32, B>
where
    f32: ComputeFor<B>,
{
    let x: SkeletonSlot<f32, B> = SkeletonSlot::new(layouts[0].clone());
    let y = (&x * 2.0 + 1.0).exp();
    let y = (y * 0.1 + 0.5).tanh();
    let y = (y * 1.5 + 0.2).exp();
    let y = (y * 0.05 + 0.1).ln();
    y.into_skeleton(std::slice::from_ref(&x)).unwrap()
}

const LANE_WIDTH: usize = 8;

#[inline]
fn unary_simd<F: Fn(f32x8) -> f32x8, U: Fn(f32) -> f32>(
    src: &[f32],
    dst: &mut [f32],
    f_simd: F,
    f: U,
) {
    let (in_chunks, in_remainder) = src.as_chunks::<LANE_WIDTH>();
    let (out_chunks, out_remainder) = dst.as_chunks_mut::<LANE_WIDTH>();

    for (chunk_in, chunk_out) in in_chunks.iter().zip(out_chunks) {
        let v = wide::f32x8::from(*chunk_in);

        let result = f_simd(v);

        chunk_out.copy_from_slice(&result.to_array());
    }

    for (x, y) in in_remainder.iter().zip(out_remainder) {
        *y = f(*x);
    }
}

#[inline]
fn unary_simd_inplace<F: Fn(f32x8) -> f32x8, U: Fn(f32) -> f32>(out: &mut [f32], f_simd: F, f: U) {
    let (out_chunks, out_remainder) = out.as_chunks_mut::<LANE_WIDTH>();

    for chunk_out in out_chunks.iter_mut() {
        let v = wide::f32x8::from(*chunk_out);

        let result = f_simd(v);

        chunk_out.copy_from_slice(&result.to_array());
    }

    for x in out_remainder.iter_mut() {
        *x = f(*x);
    }
}

#[inline]
fn apply_affine(input: wide::f32x8, a: f32, b: f32) -> wide::f32x8 {
    wide::f32x8::splat(a) * input + wide::f32x8::splat(b)
}

#[inline]
fn chain_hanrolled_multipass(input: &[f32], output: &mut [f32]) {
    unary_simd(
        input,
        output,
        |x| apply_affine(x, 2.0, 1.0),
        |x| 2.0 * x + 1.0,
    );
    unary_simd_inplace(output, |x| x.exp(), |x| x.exp());
    unary_simd_inplace(output, |x| apply_affine(x, 0.1, 0.5), |x| 0.1 * x + 0.5);
    unary_simd_inplace(output, |x| x.tanh(), |x| x.tanh());
    unary_simd_inplace(output, |x| apply_affine(x, 1.5, 0.2), |x| 1.5 * x + 0.2);
    unary_simd_inplace(output, |x| x.exp(), |x| x.exp());
    unary_simd_inplace(output, |x| apply_affine(x, 0.05, 0.1), |x| 0.5 * x + 0.1);
    unary_simd_inplace(output, |x| x.ln(), |x| x.ln());
}

// This only works for contiguous memory!
#[inline]
fn chain_hanrolled(input: &[f32], output: &mut [f32]) {
    unary_simd(
        input,
        output,
        |x| {
            let mut y = apply_affine(x, 2.0, 1.0).exp();
            y = apply_affine(y, 0.1, 0.5).tanh();
            y = apply_affine(y, 1.5, 0.2).exp();
            y = apply_affine(y, 0.05, 0.1).ln();
            y
        },
        |x| {
            let mut y = (x * 2.0 + 1.0).exp();
            y = (y * 0.1 + 0.5).tanh();
            y = (y * 1.5 + 0.2).exp();
            y = (y * 0.05 + 0.1).ln();
            y
        },
    );
}

const TILE_WIDTH: usize = 2048;

// This only works for contiguous memory!
#[inline]
fn chain_hanrolled_tiled(input: &[f32], output: &mut [f32]) {
    let (in_chunks, in_remainder) = input.as_chunks::<TILE_WIDTH>();
    let (out_chunks, out_remainder) = output.as_chunks_mut::<TILE_WIDTH>();

    for (i, o) in in_chunks.iter().zip(out_chunks.iter_mut()) {
        unary_simd(
            i,
            o,
            |x| {
                let mut y = apply_affine(x, 2.0, 1.0).exp();
                y = apply_affine(y, 0.1, 0.5).tanh();
                y = apply_affine(y, 1.5, 0.2).exp();
                y = apply_affine(y, 0.05, 0.1).ln();
                y
            },
            |x| {
                let mut y = (x * 2.0 + 1.0).exp();
                y = (y * 0.1 + 0.5).tanh();
                y = (y * 1.5 + 0.2).exp();
                y = (y * 0.05 + 0.1).ln();
                y
            },
        );
    }

    unary_simd(
        in_remainder,
        out_remainder,
        |x| {
            let mut y = apply_affine(x, 2.0, 1.0).exp();
            y = apply_affine(y, 0.1, 0.5).tanh();
            y = apply_affine(y, 1.5, 0.2).exp();
            y = apply_affine(y, 0.05, 0.1).ln();
            y
        },
        |x| {
            let mut y = (x * 2.0 + 1.0).exp();
            y = (y * 0.1 + 0.5).tanh();
            y = (y * 1.5 + 0.2).exp();
            y = (y * 0.05 + 0.1).ln();
            y
        },
    );
}

// This only works for contiguous memory!
#[inline]
fn chain_hanrolled_tiled_multipass(input: &[f32], output: &mut [f32]) {
    let (in_chunks, in_remainder) = input.as_chunks::<TILE_WIDTH>();
    let (out_chunks, out_remainder) = output.as_chunks_mut::<TILE_WIDTH>();

    for (i, o) in in_chunks.iter().zip(out_chunks.iter_mut()) {
        unary_simd(i, o, |x| apply_affine(x, 2.0, 1.0), |x| 2.0 * x + 1.0);
        unary_simd_inplace(o, |x| x.exp(), |x| x.exp());
        unary_simd_inplace(o, |x| apply_affine(x, 0.1, 0.5), |x| 0.1 * x + 0.5);
        unary_simd_inplace(o, |x| x.tanh(), |x| x.tanh());
        unary_simd_inplace(o, |x| apply_affine(x, 1.5, 0.2), |x| 1.5 * x + 0.2);
        unary_simd_inplace(o, |x| x.exp(), |x| x.exp());
        unary_simd_inplace(o, |x| apply_affine(x, 0.05, 0.1), |x| 0.5 * x + 0.1);
        unary_simd_inplace(o, |x| x.ln(), |x| x.ln());
    }

    unary_simd(
        in_remainder,
        out_remainder,
        |x| apply_affine(x, 2.0, 1.0),
        |x| 2.0 * x + 1.0,
    );
    unary_simd_inplace(out_remainder, |x| x.exp(), |x| x.exp());
    unary_simd_inplace(
        out_remainder,
        |x| apply_affine(x, 0.1, 0.5),
        |x| 0.1 * x + 0.5,
    );
    unary_simd_inplace(out_remainder, |x| x.tanh(), |x| x.tanh());
    unary_simd_inplace(
        out_remainder,
        |x| apply_affine(x, 1.5, 0.2),
        |x| 1.5 * x + 0.2,
    );
    unary_simd_inplace(out_remainder, |x| x.exp(), |x| x.exp());
    unary_simd_inplace(
        out_remainder,
        |x| apply_affine(x, 0.05, 0.1),
        |x| 0.5 * x + 0.1,
    );
    unary_simd_inplace(out_remainder, |x| x.ln(), |x| x.ln());
}

fn bench_handrolled(
    group: &mut BenchmarkGroup<'_, WallTime>,
    label: &str,
    cfg: &FillConfig,
    kernel: impl Fn(&[f32], &mut [f32]),
) {
    for case in fill_cases::<f32, CpuExperimental, _>(cfg, build_chain::<CpuExperimental>) {
        let out_len = case.skeleton.len();
        group.throughput(case.throughput);

        let input = &case.inputs[0];
        let data = input.data();
        let mut out = vec![0.0f32; out_len];

        group.bench_function(BenchmarkId::new(label, &case.label), |b| {
            b.iter(|| {
                kernel(black_box(data), black_box(&mut out));
                black_box(&out);
            });
        });
    }
}

fn tiling(c: &mut Criterion) {
    let plot_config = PlotConfiguration::default().summary_scale(AxisScale::Logarithmic);
    let mut group = c.benchmark_group("tiling");
    group.plot_config(plot_config);

    let chain = FillConfig::new(1).variants(&[Variant::Contig]).sizes(&[
        // SizeSpec::L1,
        SizeSpec::L2,
        SizeSpec::L3,
        SizeSpec::Dram,
    ]);

    bench_fill(&mut group, "chain", &chain, build_chain::<CpuExperimental>);
    bench_handrolled(&mut group, "chain_handrolled", &chain, chain_hanrolled);
    bench_handrolled(
        &mut group,
        "chain_handrolled_tiled",
        &chain,
        chain_hanrolled_tiled,
    );
    bench_handrolled(
        &mut group,
        "chain_handrolled_tiled_multipass",
        &chain,
        chain_hanrolled_tiled_multipass,
    );
    bench_handrolled(
        &mut group,
        "chain_handrolled_multipass",
        &chain,
        chain_hanrolled_multipass,
    );

    group.finish();
}

criterion_group!(benches, tiling);
criterion_main!(benches);
