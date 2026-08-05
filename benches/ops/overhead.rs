use candela::Layout;
use candela::backend::{Backend, ComputeFor};
use candela::skeleton::{Skeleton, SkeletonSlot};
use criterion::{Criterion, criterion_group};
use rand::distr::{Distribution, StandardUniform};

use crate::common::{FillConfig, SizeSpec, Variant, bench_fill, fill_cases};
use crate::ops::{Alloc, BenchScalar, Tier, bench_cells, bench_reference, group};

fn tiny_sizes() -> [SizeSpec; 5] {
    [
        SizeSpec::Elems(64),
        SizeSpec::Elems(256),
        SizeSpec::Elems(1024),
        SizeSpec::Elems(4096),
        SizeSpec::Elems(16384),
    ]
}

fn floor_cfg() -> FillConfig {
    FillConfig::new(1)
        .variants(&[Variant::Contig])
        .sizes(&tiny_sizes())
}

fn build_floor<T, B>(layouts: &[Layout]) -> Skeleton<T, B>
where
    T: BenchScalar + ComputeFor<B>,
    B: Backend,
{
    let x: SkeletonSlot<T, B> = SkeletonSlot::new(layouts[0].clone());
    (&x * T::from_f64(2.0) + T::from_f64(1.0))
        .into_skeleton(std::slice::from_ref(&x))
        .unwrap()
}

fn build_depth<T, B>(depth: usize) -> impl Fn(&[Layout]) -> Skeleton<T, B> + Copy
where
    T: BenchScalar + ComputeFor<B>,
    B: Backend,
{
    move |layouts: &[Layout]| {
        let slots: Vec<SkeletonSlot<T, B>> = layouts
            .iter()
            .take(depth + 1)
            .map(|l| SkeletonSlot::new(l.clone()))
            .collect();

        let mut acc = &slots[0] + &slots[1];
        for slot in &slots[2..] {
            acc = acc + slot;
        }
        acc.into_skeleton(&slots).unwrap()
    }
}

const DEPTHS: [usize; 4] = [1, 2, 4, 8];

fn depth_cfg(depth: usize) -> FillConfig {
    FillConfig::new(depth + 1)
        .variants(&[Variant::Contig])
        .sizes(&[SizeSpec::Elems(1024)])
}

fn run<T, B>(c: &mut Criterion, cell: &str)
where
    T: BenchScalar + ComputeFor<B>,
    StandardUniform: Distribution<T>,
    B: Backend,
{
    let mut g = group(c, "overhead/run_floor", Tier::Fast);
    if cell == "pure/f32" {
        report_allocations::<T, B>();
    }
    bench_fill(&mut g, cell, &floor_cfg(), build_floor::<T, B>);
    g.finish();

    let mut g = group(c, "overhead/depth", Tier::Fast);
    for depth in DEPTHS {
        bench_fill(
            &mut g,
            &format!("{cell}/nodes{depth}"),
            &depth_cfg(depth),
            build_depth::<T, B>(depth),
        );
    }
    g.finish();
}

fn report_allocations<T, B>()
where
    T: BenchScalar + ComputeFor<B>,
    StandardUniform: Distribution<T>,
    B: Backend,
{
    for case in fill_cases::<T, B, _>(&floor_cfg(), build_floor::<T, B>) {
        let m = case.skeleton.memory_report();
        eprintln!(
            "[overhead/run_floor] {}: allocations={} buffers={:?} peak={}B",
            case.label,
            m.total_number_of_allocations,
            m.allocated_buffers_size,
            m.peak_memory_usage
        );
    }
}

fn run_reference<T>(c: &mut Criterion, dtype: &str)
where
    T: BenchScalar + ComputeFor<candela::backend::CpuPure>,
    StandardUniform: Distribution<T>,
{
    let mut g = group(c, "overhead/run_floor", Tier::Fast);
    let cfg = floor_cfg();

    bench_reference(
        &mut g,
        &format!("base/{dtype}"),
        &cfg,
        build_floor::<T, candela::backend::CpuPure>,
        Alloc::Fresh,
        |inputs: &[&[T]], out: &mut [T]| {
            for (o, x) in out.iter_mut().zip(inputs[0]) {
                *o = *x * T::from_f64(2.0) + T::from_f64(1.0);
            }
        },
    );
    g.finish();
}

fn overhead(c: &mut Criterion) {
    bench_cells!(c, run);
    run_reference::<f32>(c, "f32");
    run_reference::<f64>(c, "f64");
}

criterion_group!(benches, overhead);
