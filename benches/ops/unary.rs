use candela::Layout;
use candela::backend::{Backend, ComputeFor};
use candela::skeleton::{Skeleton, SkeletonSlot};
use criterion::{Criterion, criterion_group};
use rand::distr::{Distribution, StandardUniform};

use crate::common::{FillConfig, SizeSpec, Variant, bench_fill};
use crate::ops::{
    Alloc, BenchScalar, SimdUnary, Tier, bench_cells, bench_reference, bench_reference_pair, group,
};

#[derive(Clone, Copy)]
enum Un {
    AxBy,
    Exp,
    Ln,
    Log2,
    Recip,
    ReLU,
    Tanh,
}

const OPS: [Un; 7] = [
    Un::AxBy,
    Un::Exp,
    Un::Ln,
    Un::Log2,
    Un::Recip,
    Un::ReLU,
    Un::Tanh,
];

impl Un {
    fn name(self) -> &'static str {
        match self {
            Un::AxBy => "axby",
            Un::Exp => "exp",
            Un::Ln => "ln",
            Un::Log2 => "log2",
            Un::Recip => "recip",
            Un::ReLU => "relu",
            Un::Tanh => "tanh",
        }
    }

    fn simd(self) -> SimdUnary {
        match self {
            Un::AxBy => SimdUnary::AxBy,
            Un::Exp => SimdUnary::Exp,
            Un::Ln => SimdUnary::Ln,
            Un::Log2 => SimdUnary::Log2,
            Un::Recip => SimdUnary::Recip,
            Un::ReLU => SimdUnary::ReLU,
            Un::Tanh => SimdUnary::Tanh,
        }
    }

    fn scalar<T: BenchScalar>(self, x: T) -> T {
        match self {
            Un::AxBy => x * T::from_f64(2.0) + T::from_f64(1.0),
            Un::Exp => x.b_exp(),
            Un::Ln => x.b_ln(),
            Un::Log2 => x.b_log2(),
            Un::Recip => x.b_recip(),
            Un::ReLU => x.b_relu(),
            Un::Tanh => x.b_tanh(),
        }
    }
}

fn builder<T, B>(op: Un) -> impl Fn(&[Layout]) -> Skeleton<T, B> + Copy
where
    T: BenchScalar + ComputeFor<B>,
    B: Backend,
{
    move |layouts: &[Layout]| {
        let x: SkeletonSlot<T, B> = SkeletonSlot::new(layouts[0].clone());
        let out = match op {
            Un::AxBy => &x * T::from_f64(2.0) + T::from_f64(1.0),
            Un::Exp => x.exp(),
            Un::Ln => x.ln(),
            Un::Log2 => x.log2(),
            Un::Recip => x.recip(),
            Un::ReLU => x.relu(),
            Un::Tanh => x.tanh(),
        };
        out.into_skeleton(std::slice::from_ref(&x)).unwrap()
    }
}

fn contig_cfg() -> FillConfig {
    FillConfig::new(1).variants(&[Variant::Contig]).sizes(&[
        SizeSpec::L1,
        SizeSpec::L2,
        SizeSpec::L3,
        SizeSpec::Dram,
    ])
}

fn layout_cfg() -> FillConfig {
    FillConfig::new(1)
        .variants(&[
            Variant::Contig,
            Variant::Step,
            Variant::Padded,
            Variant::Transposed,
        ])
        .variant_sizes(&[SizeSpec::L2, SizeSpec::Dram])
        .sizes(&[SizeSpec::L2, SizeSpec::Dram])
}

fn run<T, B>(c: &mut Criterion, cell: &str)
where
    T: BenchScalar + ComputeFor<B>,
    StandardUniform: Distribution<T>,
    B: Backend,
{
    for op in OPS {
        let mut g = group(c, &format!("unary/{}", op.name()), Tier::Fast);
        bench_fill(&mut g, cell, &contig_cfg(), builder::<T, B>(op));
        g.finish();
    }

    let mut g = group(c, "unary/exp_layouts", Tier::Fast);
    bench_fill(&mut g, cell, &layout_cfg(), builder::<T, B>(Un::Exp));
    g.finish();
}

fn run_reference<T>(c: &mut Criterion, dtype: &str)
where
    T: BenchScalar + ComputeFor<candela::backend::CpuPure>,
    StandardUniform: Distribution<T>,
{
    for op in OPS {
        let mut g = group(c, &format!("unary/{}", op.name()), Tier::Fast);
        let cfg = contig_cfg();

        bench_reference_pair(
            &mut g,
            dtype,
            &cfg,
            builder::<T, _>(op),
            move |inputs: &[&[T]], out: &mut [T]| {
                for (o, x) in out.iter_mut().zip(inputs[0]) {
                    *o = op.scalar(*x);
                }
            },
        );
        bench_reference(
            &mut g,
            &format!("simd/{dtype}"),
            &cfg,
            builder::<T, _>(op),
            Alloc::Fresh,
            move |inputs: &[&[T]], out: &mut [T]| T::simd_map(inputs[0], out, op.simd()),
        );
        g.finish();
    }
}

fn unary(c: &mut Criterion) {
    bench_cells!(c, run);
    run_reference::<f32>(c, "f32");
    run_reference::<f64>(c, "f64");
}

criterion_group!(benches, unary);
