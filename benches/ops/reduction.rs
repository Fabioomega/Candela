use candela::Layout;
use candela::backend::{Backend, ComputeFor};
use candela::skeleton::{Skeleton, SkeletonSlot};
use criterion::{Criterion, criterion_group};
use rand::distr::{Distribution, StandardUniform};

use crate::common::{FillConfig, ShapePolicy, SizeSpec, Variant, bench_fill};
use crate::ops::{
    Alloc, BenchScalar, SimdReduce, Tier, bench_cells, bench_reference, bench_reference_pair,
    bench_reference_reduce, group,
};

const INNER: usize = 1024;

#[derive(Clone, Copy)]
enum Red {
    Sum,
    Mean,
    Max,
}

const REDS: [Red; 3] = [Red::Sum, Red::Mean, Red::Max];

impl Red {
    fn name(self) -> &'static str {
        match self {
            Red::Sum => "sum",
            Red::Mean => "mean",
            Red::Max => "max",
        }
    }
}

#[derive(Clone, Copy)]
enum Axis {
    /// `axis = -1`: each output gathers one contiguous run.
    Along,
    /// `axis = 0`: each output accumulates a column across rows.
    Across,
}

impl Axis {
    fn name(self) -> &'static str {
        match self {
            Axis::Along => "inner",
            Axis::Across => "outer",
        }
    }

    fn index(self) -> isize {
        match self {
            Axis::Along => -1,
            Axis::Across => 0,
        }
    }
}

fn build_full<T, B>(red: Red) -> impl Fn(&[Layout]) -> Skeleton<T, B> + Copy
where
    T: BenchScalar + ComputeFor<B>,
    B: Backend,
{
    move |layouts: &[Layout]| {
        let x: SkeletonSlot<T, B> = SkeletonSlot::new(layouts[0].clone());
        let out = match red {
            Red::Sum => x.sum(),
            Red::Mean => x.mean(),
            Red::Max => x.max(),
        };
        out.into_skeleton(std::slice::from_ref(&x)).unwrap()
    }
}

fn build_axis<T, B>(red: Red, axis: Axis) -> impl Fn(&[Layout]) -> Skeleton<T, B> + Copy
where
    T: BenchScalar + ComputeFor<B>,
    B: Backend,
{
    move |layouts: &[Layout]| {
        let x: SkeletonSlot<T, B> = SkeletonSlot::new(layouts[0].clone());
        let a = axis.index();
        let out = match red {
            Red::Sum => x.sum_axis(a, false).unwrap(),
            Red::Mean => x.mean_axis(a, false).unwrap(),
            Red::Max => x.max_axis(a, false).unwrap(),
        };
        out.into_skeleton(std::slice::from_ref(&x)).unwrap()
    }
}

fn full_cfg() -> FillConfig {
    FillConfig::new(1).variants(&[Variant::Contig]).sizes(&[
        SizeSpec::Elems(4096),
        SizeSpec::L1,
        SizeSpec::L2,
        SizeSpec::L3,
        SizeSpec::Dram,
    ])
}

fn axis_cfg() -> FillConfig {
    FillConfig::new(1)
        .shape(ShapePolicy::Rows { inner: INNER })
        .variants(&[Variant::Contig])
        .sizes(&[SizeSpec::L1, SizeSpec::L2, SizeSpec::L3, SizeSpec::Dram])
}

fn axis_layout_cfg() -> FillConfig {
    FillConfig::new(1)
        .shape(ShapePolicy::Rows { inner: INNER })
        .variants(&[Variant::Contig, Variant::Padded, Variant::Step])
        .sizes(&[SizeSpec::L2, SizeSpec::Dram])
        .variant_sizes(&[SizeSpec::L2, SizeSpec::Dram])
}

fn run<T, B>(c: &mut Criterion, cell: &str)
where
    T: BenchScalar + ComputeFor<B>,
    StandardUniform: Distribution<T>,
    B: Backend,
{
    for red in REDS {
        let mut g = group(c, &format!("reduce/{}", red.name()), Tier::Fast);
        bench_fill(&mut g, cell, &full_cfg(), build_full::<T, B>(red));
        g.finish();

        for axis in [Axis::Along, Axis::Across] {
            let name = format!("reduce/{}_axis_{}", red.name(), axis.name());
            let mut g = group(c, &name, Tier::Fast);
            bench_fill(&mut g, cell, &axis_cfg(), build_axis::<T, B>(red, axis));
            g.finish();
        }
    }

    let mut g = group(c, "reduce/sum_axis_inner_layouts", Tier::Fast);
    bench_fill(
        &mut g,
        cell,
        &axis_layout_cfg(),
        build_axis::<T, B>(Red::Sum, Axis::Along),
    );
    g.finish();
}

fn run_reference<T>(c: &mut Criterion, dtype: &str)
where
    T: BenchScalar + ComputeFor<candela::backend::CpuPure>,
    StandardUniform: Distribution<T>,
{
    for red in REDS {
        let mut g = group(c, &format!("reduce/{}", red.name()), Tier::Fast);
        let cfg = full_cfg();

        bench_reference_reduce(
            &mut g,
            &format!("base/{dtype}"),
            &cfg,
            build_full::<T, _>(red),
            move |src: &[T]| fold_scalar(red, src),
        );
        bench_reference_reduce(
            &mut g,
            &format!("simd/{dtype}"),
            &cfg,
            build_full::<T, _>(red),
            move |src: &[T]| fold_simd(red, src),
        );
        g.finish();
    }

    for red in REDS {
        for axis in [Axis::Along, Axis::Across] {
            let name = format!("reduce/{}_axis_{}", red.name(), axis.name());
            let mut g = group(c, &name, Tier::Fast);
            let cfg = axis_cfg();

            bench_reference_pair(
                &mut g,
                dtype,
                &cfg,
                build_axis::<T, _>(red, axis),
                move |inputs: &[&[T]], out: &mut [T]| axis_kernel(red, axis, inputs[0], out),
            );
            g.finish();
        }

        let name = format!("reduce/{}_axis_inner", red.name());
        let mut g = group(c, &name, Tier::Fast);
        bench_reference(
            &mut g,
            &format!("simd/{dtype}"),
            &axis_cfg(),
            build_axis::<T, _>(red, Axis::Along),
            Alloc::Fresh,
            move |inputs: &[&[T]], out: &mut [T]| {
                for (o, row) in out.iter_mut().zip(inputs[0].chunks_exact(INNER)) {
                    *o = fold_simd(red, row);
                }
            },
        );
        g.finish();
    }
}

fn fold_scalar<T: BenchScalar>(red: Red, src: &[T]) -> T {
    match red {
        Red::Sum => src.iter().fold(T::from_f64(0.0), |a, b| a + *b),
        Red::Mean => {
            let s = src.iter().fold(T::from_f64(0.0), |a, b| a + *b);
            s / T::from_f64(src.len() as f64)
        }
        Red::Max => src
            .iter()
            .fold(T::from_f64(f64::NEG_INFINITY), |a, b| a.b_max(*b)),
    }
}

fn fold_simd<T: BenchScalar>(red: Red, src: &[T]) -> T {
    match red {
        Red::Sum => T::simd_reduce(src, SimdReduce::Sum),
        Red::Mean => T::simd_reduce(src, SimdReduce::Sum) / T::from_f64(src.len() as f64),
        Red::Max => T::simd_reduce(src, SimdReduce::Max),
    }
}

fn axis_kernel<T: BenchScalar>(red: Red, axis: Axis, src: &[T], out: &mut [T]) {
    let rows = src.len() / INNER;
    match axis {
        Axis::Along => {
            for (o, row) in out.iter_mut().zip(src.chunks_exact(INNER)) {
                *o = fold_scalar(red, row);
            }
        }
        Axis::Across => {
            let seed = match red {
                Red::Max => T::from_f64(f64::NEG_INFINITY),
                _ => T::from_f64(0.0),
            };
            out.fill(seed);
            for row in src.chunks_exact(INNER) {
                match red {
                    Red::Max => {
                        for (o, x) in out.iter_mut().zip(row) {
                            *o = o.b_max(*x);
                        }
                    }
                    _ => {
                        for (o, x) in out.iter_mut().zip(row) {
                            *o = *o + *x;
                        }
                    }
                }
            }
            if let Red::Mean = red {
                let n = T::from_f64(rows as f64);
                for o in out.iter_mut() {
                    *o = *o / n;
                }
            }
        }
    }
}

fn reduction(c: &mut Criterion) {
    bench_cells!(c, run);
    run_reference::<f32>(c, "f32");
    run_reference::<f64>(c, "f64");
}

criterion_group!(benches, reduction);
