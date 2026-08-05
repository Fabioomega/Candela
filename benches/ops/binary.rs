use candela::Layout;
use candela::backend::{Backend, ComputeFor};
use candela::skeleton::{Skeleton, SkeletonSlot};
use criterion::{Criterion, criterion_group};
use rand::distr::{Distribution, StandardUniform};

use crate::common::{FillConfig, ShapePolicy, SizeSpec, Variant, bench_fill};
use crate::ops::{
    Alloc, BenchScalar, SimdBinary, Tier, bench_cells, bench_reference, bench_reference_pair, group,
};

#[derive(Clone, Copy)]
enum Bin {
    Add,
    Sub,
    Mul,
    Div,
}

impl Bin {
    fn name(self) -> &'static str {
        match self {
            Bin::Add => "add",
            Bin::Sub => "sub",
            Bin::Mul => "mul",
            Bin::Div => "div",
        }
    }

    fn simd(self) -> SimdBinary {
        match self {
            Bin::Add => SimdBinary::Add,
            Bin::Sub => SimdBinary::Sub,
            Bin::Mul => SimdBinary::Mul,
            Bin::Div => SimdBinary::Div,
        }
    }
}

fn builder<T, B>(op: Bin) -> impl Fn(&[Layout]) -> Skeleton<T, B> + Copy
where
    T: BenchScalar + ComputeFor<B>,
    B: Backend,
{
    move |layouts: &[Layout]| {
        let a: SkeletonSlot<T, B> = SkeletonSlot::new(layouts[0].clone());
        let b: SkeletonSlot<T, B> = SkeletonSlot::new(layouts[1].clone());
        let out = match op {
            Bin::Add => &a + &b,
            Bin::Sub => &a - &b,
            Bin::Mul => &a * &b,
            Bin::Div => &a / &b,
        };
        out.into_skeleton(&[a, b]).unwrap()
    }
}

fn contig_cfg() -> FillConfig {
    FillConfig::new(2).variants(&[Variant::Contig]).sizes(&[
        SizeSpec::L1,
        SizeSpec::L2,
        SizeSpec::L3,
        SizeSpec::Dram,
    ])
}

fn run<T, B>(c: &mut Criterion, cell: &str)
where
    T: BenchScalar + ComputeFor<B>,
    StandardUniform: Distribution<T>,
    B: Backend,
{
    for op in [Bin::Add, Bin::Sub, Bin::Mul, Bin::Div] {
        let mut g = group(c, &format!("binary/{}", op.name()), Tier::Fast);
        bench_fill(&mut g, cell, &contig_cfg(), builder::<T, B>(op));
        g.finish();
    }

    let mut g = group(c, "binary/add_layouts", Tier::Fast);
    bench_fill(&mut g, cell, &layout_cfg(), builder::<T, B>(Bin::Add));
    g.finish();
}

fn layout_cfg() -> FillConfig {
    FillConfig::new(2)
        .shape(ShapePolicy::Rows { inner: 256 })
        .variant_combos(&[
            &[Variant::Contig, Variant::Contig],
            &[Variant::Step, Variant::Step],
            &[Variant::Padded, Variant::Padded],
            &[Variant::Transposed, Variant::Transposed],
            &[Variant::Contig, Variant::BcastOuter],
            &[Variant::Contig, Variant::BcastInner],
        ])
        .sizes(&[SizeSpec::L2, SizeSpec::Dram])
        .variant_sizes(&[SizeSpec::L2, SizeSpec::Dram])
}

fn run_reference<T>(c: &mut Criterion, dtype: &str)
where
    T: BenchScalar + ComputeFor<candela::backend::CpuPure>,
    StandardUniform: Distribution<T>,
{
    for op in [Bin::Add, Bin::Sub, Bin::Mul, Bin::Div] {
        let mut g = group(c, &format!("binary/{}", op.name()), Tier::Fast);
        let cfg = contig_cfg();

        let scalar = move |inputs: &[&[T]], out: &mut [T]| {
            let (a, b) = (inputs[0], inputs[1]);
            match op {
                Bin::Add => {
                    for (o, (x, y)) in out.iter_mut().zip(a.iter().zip(b)) {
                        *o = *x + *y;
                    }
                }
                Bin::Sub => {
                    for (o, (x, y)) in out.iter_mut().zip(a.iter().zip(b)) {
                        *o = *x - *y;
                    }
                }
                Bin::Mul => {
                    for (o, (x, y)) in out.iter_mut().zip(a.iter().zip(b)) {
                        *o = *x * *y;
                    }
                }
                Bin::Div => {
                    for (o, (x, y)) in out.iter_mut().zip(a.iter().zip(b)) {
                        *o = *x / *y;
                    }
                }
            }
        };

        bench_reference_pair(&mut g, dtype, &cfg, builder::<T, _>(op), scalar);
        bench_reference(
            &mut g,
            &format!("simd/{dtype}"),
            &cfg,
            builder::<T, _>(op),
            Alloc::Fresh,
            move |inputs: &[&[T]], out: &mut [T]| T::simd_zip(inputs[0], inputs[1], out, op.simd()),
        );
        g.finish();
    }
}

fn binary(c: &mut Criterion) {
    bench_cells!(c, run);
    run_reference::<f32>(c, "f32");
    run_reference::<f64>(c, "f64");
}

criterion_group!(benches, binary);
