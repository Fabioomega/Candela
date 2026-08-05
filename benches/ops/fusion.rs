use candela::Layout;
use candela::backend::{Backend, ComputeFor};
use candela::skeleton::{Skeleton, SkeletonSlot};
use criterion::{Criterion, criterion_group};
use rand::distr::{Distribution, StandardUniform};

use crate::common::{FillConfig, ShapePolicy, SizeSpec, Variant, bench_fill};
use crate::ops::{
    Alloc, BenchScalar, Tier, bench_cells, bench_reference, bench_reference_pair, group,
};

#[derive(Clone, Copy)]
enum Chain {
    AffineCollapse,
    Chain2,
    Chain4,
    RecipScaled,
    Sigmoid,
}

const CHAINS: [Chain; 5] = [
    Chain::AffineCollapse,
    Chain::Chain2,
    Chain::Chain4,
    Chain::RecipScaled,
    Chain::Sigmoid,
];

impl Chain {
    fn name(self) -> &'static str {
        match self {
            Chain::AffineCollapse => "affine_collapse",
            Chain::Chain2 => "chain2",
            Chain::Chain4 => "chain4",
            Chain::RecipScaled => "recip_scaled",
            Chain::Sigmoid => "sigmoid",
        }
    }

    fn scalar<T: BenchScalar>(self, x: T) -> T {
        match self {
            Chain::AffineCollapse => (x + T::from_f64(1.0)) * T::from_f64(2.0) - T::from_f64(3.0),
            Chain::Chain2 => x.b_exp().b_ln(),
            Chain::Chain4 => {
                let y = (x * T::from_f64(2.0) + T::from_f64(1.0)).b_exp();
                let y = (y * T::from_f64(0.1) + T::from_f64(0.5)).b_tanh();
                let y = (y * T::from_f64(1.5) + T::from_f64(0.2)).b_exp();
                (y * T::from_f64(0.05) + T::from_f64(0.1)).b_ln()
            }
            Chain::RecipScaled => x.b_recip() * T::from_f64(3.0),
            Chain::Sigmoid => ((x * T::from_f64(-1.0)).b_exp() + T::from_f64(1.0)).b_recip(),
        }
    }
}

fn builder<T, B>(chain: Chain) -> impl Fn(&[Layout]) -> Skeleton<T, B> + Copy
where
    T: BenchScalar + ComputeFor<B>,
    B: Backend,
{
    move |layouts: &[Layout]| {
        let x: SkeletonSlot<T, B> = SkeletonSlot::new(layouts[0].clone());
        let out = match chain {
            Chain::AffineCollapse => (&x + T::from_f64(1.0)) * T::from_f64(2.0) - T::from_f64(3.0),
            Chain::Chain2 => x.exp().ln(),
            Chain::Chain4 => {
                let y = (&x * T::from_f64(2.0) + T::from_f64(1.0)).exp();
                let y = (y * T::from_f64(0.1) + T::from_f64(0.5)).tanh();
                let y = (y * T::from_f64(1.5) + T::from_f64(0.2)).exp();
                (y * T::from_f64(0.05) + T::from_f64(0.1)).ln()
            }
            Chain::RecipScaled => x.recip() * T::from_f64(3.0),
            Chain::Sigmoid => ((-&x).exp() + T::from_f64(1.0)).recip(),
        };
        out.into_skeleton(std::slice::from_ref(&x)).unwrap()
    }
}

fn build_softmax<T, B>(layouts: &[Layout]) -> Skeleton<T, B>
where
    T: BenchScalar + ComputeFor<B>,
    B: Backend,
{
    let x: SkeletonSlot<T, B> = SkeletonSlot::new(layouts[0].clone());
    let m = x.max_axis(-1, true).unwrap();
    let e = (&x - &m).exp();
    let s = e.sum_axis(-1, true).unwrap();
    (e / s).into_skeleton(std::slice::from_ref(&x)).unwrap()
}

fn contig_cfg() -> FillConfig {
    FillConfig::new(1).variants(&[Variant::Contig]).sizes(&[
        SizeSpec::L1,
        SizeSpec::L2,
        SizeSpec::L3,
        SizeSpec::Dram,
    ])
}

fn softmax_cfg() -> FillConfig {
    FillConfig::new(1)
        .shape(ShapePolicy::Rows { inner: 1024 })
        .variants(&[Variant::Contig])
        .sizes(&[SizeSpec::L2, SizeSpec::L3, SizeSpec::Dram])
}

fn run<T, B>(c: &mut Criterion, cell: &str)
where
    T: BenchScalar + ComputeFor<B>,
    StandardUniform: Distribution<T>,
    B: Backend,
{
    for chain in CHAINS {
        let mut g = group(c, &format!("fusion/{}", chain.name()), Tier::Fast);
        bench_fill(&mut g, cell, &contig_cfg(), builder::<T, B>(chain));
        g.finish();
    }

    let mut g = group(c, "fusion/softmax", Tier::Fast);
    bench_fill(&mut g, cell, &softmax_cfg(), build_softmax::<T, B>);
    g.finish();
}

fn run_reference<T>(c: &mut Criterion, dtype: &str)
where
    T: BenchScalar + ComputeFor<candela::backend::CpuPure>,
    StandardUniform: Distribution<T>,
{
    for chain in CHAINS {
        let mut g = group(c, &format!("fusion/{}", chain.name()), Tier::Fast);
        let cfg = contig_cfg();

        bench_reference_pair(
            &mut g,
            dtype,
            &cfg,
            builder::<T, _>(chain),
            move |inputs: &[&[T]], out: &mut [T]| {
                for (o, x) in out.iter_mut().zip(inputs[0]) {
                    *o = chain.scalar(*x);
                }
            },
        );
        g.finish();
    }

    let cfg = contig_cfg();
    let mut g = group(c, "fusion/chain4", Tier::Fast);
    bench_reference(
        &mut g,
        &format!("simd/{dtype}"),
        &cfg,
        builder::<T, _>(Chain::Chain4),
        Alloc::Fresh,
        |inputs: &[&[T]], out: &mut [T]| T::simd_chain4(inputs[0], out),
    );
    g.finish();

    let mut g = group(c, "fusion/sigmoid", Tier::Fast);
    bench_reference(
        &mut g,
        &format!("simd/{dtype}"),
        &cfg,
        builder::<T, _>(Chain::Sigmoid),
        Alloc::Fresh,
        |inputs: &[&[T]], out: &mut [T]| T::simd_sigmoid(inputs[0], out),
    );
    g.finish();
}

fn fusion(c: &mut Criterion) {
    bench_cells!(c, run);
    run_reference::<f32>(c, "f32");
    run_reference::<f64>(c, "f64");
}

criterion_group!(benches, fusion);
