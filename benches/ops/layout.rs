use candela::Layout;
use candela::backend::{Backend, ComputeFor};
use candela::skeleton::{Skeleton, SkeletonSlot};
use criterion::{Criterion, criterion_group};
use rand::distr::{Distribution, StandardUniform};

use crate::common::{FillConfig, ShapePolicy, SizeSpec, Variant, bench_fill};
use crate::ops::{BenchScalar, Tier, bench_cells, group};

fn build_as_contiguous<T, B>(layouts: &[Layout]) -> Skeleton<T, B>
where
    T: BenchScalar + ComputeFor<B>,
    B: Backend,
{
    let x: SkeletonSlot<T, B> = SkeletonSlot::new(layouts[0].clone());
    x.as_contiguous()
        .into_skeleton(std::slice::from_ref(&x))
        .unwrap()
}

fn build_view_then_op<T, B>(layouts: &[Layout]) -> Skeleton<T, B>
where
    T: BenchScalar + ComputeFor<B>,
    B: Backend,
{
    let x: SkeletonSlot<T, B> = SkeletonSlot::new(layouts[0].clone());
    let rows = layouts[0].len() / 64;
    let viewed = x.view([rows, 64]).unwrap();
    (viewed * T::from_f64(2.0) + T::from_f64(1.0))
        .into_skeleton(std::slice::from_ref(&x))
        .unwrap()
}

fn build_plain_op<T, B>(layouts: &[Layout]) -> Skeleton<T, B>
where
    T: BenchScalar + ComputeFor<B>,
    B: Backend,
{
    let x: SkeletonSlot<T, B> = SkeletonSlot::new(layouts[0].clone());
    (&x * T::from_f64(2.0) + T::from_f64(1.0))
        .into_skeleton(std::slice::from_ref(&x))
        .unwrap()
}

fn sizes() -> [SizeSpec; 3] {
    [SizeSpec::L2, SizeSpec::L3, SizeSpec::Dram]
}

fn copy_cfg() -> FillConfig {
    FillConfig::new(1)
        .variants(&[Variant::Contig, Variant::Transposed])
        .sizes(&sizes())
        .variant_sizes(&sizes())
}

fn view_cfg() -> FillConfig {
    FillConfig::new(1)
        .variants(&[Variant::Contig])
        .sizes(&sizes())
}

fn strided_cfg() -> FillConfig {
    FillConfig::new(1)
        .shape(ShapePolicy::Rows { inner: 256 })
        .variants(&[Variant::Contig, Variant::Padded, Variant::Step])
        .sizes(&sizes())
        .variant_sizes(&sizes())
}

fn run<T, B>(c: &mut Criterion, cell: &str)
where
    T: BenchScalar + ComputeFor<B>,
    StandardUniform: Distribution<T>,
    B: Backend,
{
    let mut g = group(c, "layout/as_contiguous", Tier::Fast);
    bench_fill(&mut g, cell, &copy_cfg(), build_as_contiguous::<T, B>);
    g.finish();

    let mut g = group(c, "layout/view_then_op", Tier::Fast);
    bench_fill(&mut g, cell, &view_cfg(), build_view_then_op::<T, B>);
    bench_fill(
        &mut g,
        &format!("{cell}/noview"),
        &view_cfg(),
        build_plain_op::<T, B>,
    );
    g.finish();

    let mut g = group(c, "layout/strided_op", Tier::Fast);
    bench_fill(&mut g, cell, &strided_cfg(), build_plain_op::<T, B>);
    g.finish();
}

fn layout(c: &mut Criterion) {
    bench_cells!(c, run);
}

criterion_group!(benches, layout);
