use candela::backend::{ComputeFor, CpuPure};
use candela::skeleton::Skeleton;
use candela::{Dimension, FloatLikeTensorElement, Layout, Tensor};
use criterion::measurement::WallTime;
use criterion::{AxisScale, BenchmarkGroup, BenchmarkId, Criterion, PlotConfiguration};
use rand::distr::{Distribution, StandardUniform};
use std::hint::black_box;

use crate::common::{Case, FillConfig, SizeSpec, fill_cases};

pub mod binary;
pub mod fusion;
pub mod layout;
pub mod matmul;
pub mod overhead;
pub mod reduction;
pub mod unary;

macro_rules! bench_cells {
    ($c:expr, $run:ident) => {{
        $run::<f32, candela::backend::CpuPure>($c, "pure/f32");
        $run::<f64, candela::backend::CpuPure>($c, "pure/f64");
        #[cfg(feature = "mkl")]
        {
            $run::<f32, candela::backend::CpuMkl>($c, "mkl/f32");
            $run::<f64, candela::backend::CpuMkl>($c, "mkl/f64");
        }
    }};
}
pub(crate) use bench_cells;

#[derive(Clone, Copy)]
pub enum Tier {
    Fast,
    Heavy,
}

pub fn group<'a>(c: &'a mut Criterion, name: &str, tier: Tier) -> BenchmarkGroup<'a, WallTime> {
    let mut g = c.benchmark_group(name);
    g.plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));
    match tier {
        Tier::Fast => {
            g.sample_size(50)
                .warm_up_time(std::time::Duration::from_millis(500))
                .measurement_time(std::time::Duration::from_millis(1500));
        }
        Tier::Heavy => {
            g.sample_size(10)
                .warm_up_time(std::time::Duration::from_secs(1))
                .measurement_time(std::time::Duration::from_secs(3));
        }
    }
    g
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Alloc {
    /// Fresh buffer every iteration - what `Skeleton::run` does.
    Fresh,
    /// One buffer, allocated before the timed loop.
    Reuse,
}

fn alloc_vec<T: Default + Clone>(len: usize) -> Vec<T> {
    let mut buffer = Vec::with_capacity(len);
    #[allow(clippy::uninit_vec)]
    unsafe {
        buffer.set_len(len)
    };
    buffer
}

pub trait BenchScalar: FloatLikeTensorElement + Default + Send + Sync + 'static {
    /// Relative tolerance for validating a reference kernel against the
    /// skeleton. SIMD and libm transcendentals do not agree bit-for-bit.
    const TOL: f64;

    fn b_exp(self) -> Self;
    fn b_ln(self) -> Self;
    fn b_log2(self) -> Self;
    fn b_tanh(self) -> Self;
    fn b_recip(self) -> Self;
    fn b_relu(self) -> Self;
    fn b_max(self, other: Self) -> Self;

    fn simd_map(src: &[Self], dst: &mut [Self], op: SimdUnary);
    fn simd_zip(a: &[Self], b: &[Self], dst: &mut [Self], op: SimdBinary);
    /// The four-stage transcendental chain of `fusion/chain4`, fused: every
    /// stage runs on a register before the next lane group is loaded.
    fn simd_chain4(src: &[Self], dst: &mut [Self]);
    /// `1 / (1 + exp(-x))`, fused.
    fn simd_sigmoid(src: &[Self], dst: &mut [Self]);
    fn simd_reduce(src: &[Self], op: SimdReduce) -> Self;
}

#[derive(Clone, Copy)]
pub enum SimdUnary {
    /// `2x + 1`
    AxBy,
    Exp,
    Ln,
    Log2,
    Recip,
    ReLU,
    Tanh,
}

#[derive(Clone, Copy)]
pub enum SimdBinary {
    Add,
    Sub,
    Mul,
    Div,
}

#[derive(Clone, Copy)]
pub enum SimdReduce {
    Sum,
    Max,
}

/// Both dtypes get the same body; only the scalar type, the `wide` vector type
/// and the lane count differ.
macro_rules! impl_bench_scalar {
    ($t:ty, $v:ty, $lanes:expr, $tol:expr) => {
        impl BenchScalar for $t {
            const TOL: f64 = $tol;

            #[inline]
            fn b_exp(self) -> Self {
                self.exp()
            }
            #[inline]
            fn b_ln(self) -> Self {
                self.ln()
            }
            #[inline]
            fn b_log2(self) -> Self {
                self.log2()
            }
            #[inline]
            fn b_tanh(self) -> Self {
                self.tanh()
            }
            #[inline]
            fn b_recip(self) -> Self {
                1.0 / self
            }
            #[inline]
            fn b_relu(self) -> Self {
                if self > 0.0 { self } else { 0.0 }
            }
            #[inline]
            fn b_max(self, other: Self) -> Self {
                if self > other { self } else { other }
            }

            fn simd_map(src: &[Self], dst: &mut [Self], op: SimdUnary) {
                // Dispatch once, outside the loop - matching on the op per
                // element would measure the match, not the kernel.
                match op {
                    SimdUnary::AxBy => {
                        let (a, b) = (<$v>::splat(2.0), <$v>::splat(1.0));
                        simd_map_with::<$t, $v, $lanes>(src, dst, |x| x * a + b, |x| 2.0 * x + 1.0)
                    }
                    SimdUnary::Exp => {
                        simd_map_with::<$t, $v, $lanes>(src, dst, |x| x.exp(), |x| x.exp())
                    }
                    SimdUnary::Ln => {
                        simd_map_with::<$t, $v, $lanes>(src, dst, |x| x.ln(), |x| x.ln())
                    }
                    SimdUnary::Log2 => {
                        simd_map_with::<$t, $v, $lanes>(src, dst, |x| x.log2(), |x| x.log2())
                    }
                    SimdUnary::Recip => {
                        let one = <$v>::splat(1.0);
                        simd_map_with::<$t, $v, $lanes>(src, dst, |x| one / x, |x| 1.0 / x)
                    }
                    SimdUnary::ReLU => {
                        let zero = <$v>::splat(0.0);
                        simd_map_with::<$t, $v, $lanes>(
                            src,
                            dst,
                            |x| x.fast_max(zero),
                            |x| if x > 0.0 { x } else { 0.0 },
                        )
                    }
                    SimdUnary::Tanh => {
                        simd_map_with::<$t, $v, $lanes>(src, dst, |x| x.tanh(), |x| x.tanh())
                    }
                }
            }

            fn simd_zip(a: &[Self], b: &[Self], dst: &mut [Self], op: SimdBinary) {
                match op {
                    SimdBinary::Add => {
                        simd_zip_with::<$t, $v, $lanes>(a, b, dst, |x, y| x + y, |x, y| x + y)
                    }
                    SimdBinary::Sub => {
                        simd_zip_with::<$t, $v, $lanes>(a, b, dst, |x, y| x - y, |x, y| x - y)
                    }
                    SimdBinary::Mul => {
                        simd_zip_with::<$t, $v, $lanes>(a, b, dst, |x, y| x * y, |x, y| x * y)
                    }
                    SimdBinary::Div => {
                        simd_zip_with::<$t, $v, $lanes>(a, b, dst, |x, y| x / y, |x, y| x / y)
                    }
                }
            }

            fn simd_chain4(src: &[Self], dst: &mut [Self]) {
                simd_map_with::<$t, $v, $lanes>(
                    src,
                    dst,
                    |x| {
                        let y = (x * <$v>::splat(2.0) + <$v>::splat(1.0)).exp();
                        let y = (y * <$v>::splat(0.1) + <$v>::splat(0.5)).tanh();
                        let y = (y * <$v>::splat(1.5) + <$v>::splat(0.2)).exp();
                        (y * <$v>::splat(0.05) + <$v>::splat(0.1)).ln()
                    },
                    |x| {
                        let y = (2.0 * x + 1.0).exp();
                        let y = (0.1 * y + 0.5).tanh();
                        let y = (1.5 * y + 0.2).exp();
                        (0.05 * y + 0.1).ln()
                    },
                );
            }

            fn simd_sigmoid(src: &[Self], dst: &mut [Self]) {
                let one = <$v>::splat(1.0);
                let neg = <$v>::splat(-1.0);
                simd_map_with::<$t, $v, $lanes>(
                    src,
                    dst,
                    |x| one / ((x * neg).exp() + one),
                    |x| 1.0 / ((-x).exp() + 1.0),
                );
            }

            fn simd_reduce(src: &[Self], op: SimdReduce) -> Self {
                let (chunks, remainder) = src.as_chunks::<$lanes>();
                match op {
                    SimdReduce::Sum => {
                        // Four accumulators: one dependency chain per
                        // accumulator, so the adds pipeline instead of
                        // serializing on a single register.
                        let mut acc = [<$v>::splat(0.0); 4];
                        let mut it = chunks.chunks_exact(4);
                        for quad in &mut it {
                            for k in 0..4 {
                                acc[k] += <$v>::from(quad[k]);
                            }
                        }
                        for tail in it.remainder() {
                            acc[0] += <$v>::from(*tail);
                        }
                        let total = (acc[0] + acc[1]) + (acc[2] + acc[3]);
                        let mut out: $t = total.to_array().iter().sum();
                        for x in remainder {
                            out += *x;
                        }
                        out
                    }
                    SimdReduce::Max => {
                        let mut acc = <$v>::splat(<$t>::NEG_INFINITY);
                        for chunk in chunks {
                            acc = acc.fast_max(<$v>::from(*chunk));
                        }
                        let mut out = acc
                            .to_array()
                            .iter()
                            .copied()
                            .fold(<$t>::NEG_INFINITY, |a, b| if a > b { a } else { b });
                        for x in remainder {
                            if *x > out {
                                out = *x;
                            }
                        }
                        out
                    }
                }
            }
        }
    };
}

/// Lane-generic body of [`BenchScalar::simd_map`]: whole lanes through the
/// vector closure, the ragged tail through the scalar one.
#[inline]
fn simd_map_with<T, V, const N: usize>(
    src: &[T],
    dst: &mut [T],
    f_simd: impl Fn(V) -> V,
    f: impl Fn(T) -> T,
) where
    T: Copy,
    V: From<[T; N]> + ToArray<T, N>,
{
    let (in_chunks, in_tail) = src.as_chunks::<N>();
    let (out_chunks, out_tail) = dst.as_chunks_mut::<N>();

    for (chunk_in, chunk_out) in in_chunks.iter().zip(out_chunks) {
        *chunk_out = to_array(f_simd(V::from(*chunk_in)));
    }
    for (x, y) in in_tail.iter().zip(out_tail) {
        *y = f(*x);
    }
}

#[inline]
fn simd_zip_with<T, V, const N: usize>(
    a: &[T],
    b: &[T],
    dst: &mut [T],
    f_simd: impl Fn(V, V) -> V,
    f: impl Fn(T, T) -> T,
) where
    T: Copy,
    V: From<[T; N]> + ToArray<T, N>,
{
    let (a_chunks, a_tail) = a.as_chunks::<N>();
    let (b_chunks, b_tail) = b.as_chunks::<N>();
    let (out_chunks, out_tail) = dst.as_chunks_mut::<N>();

    for ((ca, cb), co) in a_chunks.iter().zip(b_chunks).zip(out_chunks) {
        *co = to_array(f_simd(V::from(*ca), V::from(*cb)));
    }
    for ((x, y), o) in a_tail.iter().zip(b_tail).zip(out_tail) {
        *o = f(*x, *y);
    }
}

/// `wide`'s vector types expose `to_array` inherently rather than through a
/// trait, so the generic helpers above go through this shim.
#[inline]
fn to_array<V, T, const N: usize>(v: V) -> [T; N]
where
    V: ToArray<T, N>,
{
    v.to_array()
}

trait ToArray<T, const N: usize> {
    fn to_array(self) -> [T; N];
}
impl ToArray<f32, 16> for wide::f32x16 {
    #[inline]
    fn to_array(self) -> [f32; 16] {
        wide::f32x16::to_array(self)
    }
}
impl ToArray<f64, 8> for wide::f64x8 {
    #[inline]
    fn to_array(self) -> [f64; 8] {
        wide::f64x8::to_array(self)
    }
}

// Tolerances are relative. f32 transcendentals through a 4-stage chain drift
// further than a single op does, so the bound is loose enough to cover the
// worst case in the suite without being loose enough to hide a wrong kernel.
impl_bench_scalar!(f32, wide::f32x16, 16, 1e-3);
impl_bench_scalar!(f64, wide::f64x8, 8, 1e-9);

/// True when two results agree to within `T::TOL`, relative for large values
/// and absolute near zero.
fn approx_eq<T: BenchScalar>(a: T, b: T) -> bool {
    let (a, b): (f64, f64) = (a.into(), b.into());
    if a == b {
        return true;
    }
    if !a.is_finite() || !b.is_finite() {
        // exp() overflowing to inf on both sides is agreement, not a mismatch
        return a.is_nan() == b.is_nan() && a.signum() == b.signum();
    }
    let diff = (a - b).abs();
    diff <= T::TOL * a.abs().max(b.abs()).max(1.0)
}

/// Element count at which reference kernels are checked against the skeleton.
///
/// Validation runs at this size and nowhere else.
const VALIDATE_ELEMS: usize = 8192;

/// Re-solves a sweep at [`VALIDATE_ELEMS`], preserving its shape policy and
/// variant combos so the validated layouts are the ones actually benched.
fn validation_cfg(cfg: &FillConfig) -> FillConfig {
    cfg.clone()
        .sizes(&[SizeSpec::Elems(VALIDATE_ELEMS)])
        .variant_sizes(&[SizeSpec::Elems(VALIDATE_ELEMS)])
}

fn run_kernel_cases<T, K>(
    group: &mut BenchmarkGroup<'_, WallTime>,
    cell: &str,
    cases: Vec<Case<T, CpuPure>>,
    alloc: Alloc,
    kernel: K,
) where
    T: BenchScalar + Clone + PartialEq + ComputeFor<CpuPure>,
    K: Fn(&[&[T]], &mut [T]),
{
    for case in cases {
        let out_len = case.skeleton.len();

        // Slice-of-slices is built once, outside the timed loop, so the tiny
        // size rungs measure the kernel rather than this bookkeeping.
        let data: Vec<&[T]> = case.inputs.iter().map(|t| t.data()).collect();

        group.throughput(case.throughput.clone());
        group.bench_function(BenchmarkId::new(cell, &case.label), |bencher| match alloc {
            Alloc::Fresh => bencher.iter(|| {
                let mut out = alloc_vec::<T>(out_len);
                kernel(black_box(&data), black_box(&mut out));
                out
            }),
            Alloc::Reuse => {
                let mut out = vec![T::default(); out_len];
                bencher.iter(|| {
                    kernel(black_box(&data), black_box(&mut out));
                    black_box(&out);
                })
            }
        });
    }
}

/// Checks a reference kernel against the skeleton on every layout combo the
/// sweep uses, at [`VALIDATE_ELEMS`].
fn validate<T, F, K>(cfg: &FillConfig, builder: F, kernel: &K, cell: &str, what: &str)
where
    T: BenchScalar + Clone + PartialEq + ComputeFor<CpuPure>,
    StandardUniform: Distribution<T>,
    F: Fn(&[Layout]) -> Skeleton<T, CpuPure>,
    K: Fn(&[&[T]], &mut [T]),
{
    let cases = fill_cases::<T, CpuPure, F>(&validation_cfg(cfg), builder);
    assert!(
        !cases.is_empty(),
        "{what} reference kernel [{cell}] produced no validation case"
    );

    for case in cases {
        let refs: Vec<&Tensor<T, CpuPure>> = case.inputs.iter().collect();
        let expected = case
            .skeleton
            .run(&refs)
            .expect("skeleton failed on its own inputs");

        let data: Vec<&[T]> = case.inputs.iter().map(|t| t.data()).collect();
        let mut got = vec![T::default(); case.skeleton.len()];
        kernel(&data, &mut got);

        for (i, (g, e)) in got.iter().zip(expected.data()).enumerate() {
            assert!(
                approx_eq(*g, *e),
                "{what} reference kernel [{cell}] disagrees with the skeleton at \
                 element {i} of case {}: got {}, skeleton produced {}",
                case.label,
                Into::<f64>::into(*g),
                Into::<f64>::into(*e),
            );
        }
    }
}

/// Registers a reference line for a map-shaped op (n inputs in, one buffer out).
pub fn bench_reference<T, F, K>(
    group: &mut BenchmarkGroup<'_, WallTime>,
    cell: &str,
    cfg: &FillConfig,
    builder: F,
    alloc: Alloc,
    kernel: K,
) where
    T: BenchScalar + Clone + PartialEq + ComputeFor<CpuPure>,
    StandardUniform: Distribution<T>,
    F: Fn(&[Layout]) -> Skeleton<T, CpuPure> + Copy,
    K: Fn(&[&[T]], &mut [T]),
{
    validate(cfg, builder, &kernel, cell, "map");

    let cases = fill_cases::<T, CpuPure, F>(cfg, builder);
    assert!(
        cases
            .iter()
            .all(|c| c.inputs.iter().all(|t| t.is_contiguous())),
        "reference kernels read flat slices; give bench_reference a contiguous-only FillConfig"
    );
    run_kernel_cases(group, cell, cases, alloc, kernel);
}

/// Registers the `base` and `base_reuse` lines for one kernel in one call.
pub fn bench_reference_pair<T, F, K>(
    group: &mut BenchmarkGroup<'_, WallTime>,
    dtype: &str,
    cfg: &FillConfig,
    builder: F,
    kernel: K,
) where
    T: BenchScalar + Clone + PartialEq + ComputeFor<CpuPure>,
    StandardUniform: Distribution<T>,
    F: Fn(&[Layout]) -> Skeleton<T, CpuPure> + Copy,
    K: Fn(&[&[T]], &mut [T]) + Copy,
{
    bench_reference(
        group,
        &format!("base/{dtype}"),
        cfg,
        builder,
        Alloc::Fresh,
        kernel,
    );
    bench_reference(
        group,
        &format!("base_reuse/{dtype}"),
        cfg,
        builder,
        Alloc::Reuse,
        kernel,
    );
}

/// Reference line for a full reduction: scalar out, so there is no output
/// buffer and therefore no `Fresh`/`Reuse` distinction to draw.
pub fn bench_reference_reduce<T, F, K>(
    group: &mut BenchmarkGroup<'_, WallTime>,
    cell: &str,
    cfg: &FillConfig,
    builder: F,
    kernel: K,
) where
    T: BenchScalar + Clone + PartialEq + ComputeFor<CpuPure>,
    StandardUniform: Distribution<T>,
    F: Fn(&[Layout]) -> Skeleton<T, CpuPure> + Copy,
    K: Fn(&[T]) -> T,
{
    // Validated at VALIDATE_ELEMS only
    for case in fill_cases::<T, CpuPure, F>(&validation_cfg(cfg), builder) {
        let refs: Vec<&Tensor<T, CpuPure>> = case.inputs.iter().collect();
        let expected = case
            .skeleton
            .run(&refs)
            .expect("skeleton failed on its own inputs");
        let got = kernel(case.inputs[0].data());
        assert!(
            approx_eq(got, expected.data()[0]),
            "reduction reference kernel [{cell}] disagrees with the skeleton on case {}: \
             got {}, skeleton produced {}",
            case.label,
            Into::<f64>::into(got),
            Into::<f64>::into(expected.data()[0]),
        );
    }

    for case in fill_cases::<T, CpuPure, F>(cfg, builder) {
        assert!(
            case.inputs[0].is_contiguous(),
            "reduction reference kernels read flat slices; use a contiguous-only FillConfig"
        );
        let data = case.inputs[0].data();
        group.throughput(case.throughput.clone());
        group.bench_function(BenchmarkId::new(cell, &case.label), |bencher| {
            bencher.iter(|| kernel(black_box(data)));
        });
    }
}
