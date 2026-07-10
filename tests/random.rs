#![cfg(feature = "rand")]

use candela::backend::CpuPure;
use candela::{FloatLikeTensorElement, Tensor};
use rand::SeedableRng;
use rand::distr::{Distribution, StandardUniform, Uniform};
use rand::rngs::StdRng;
use rand_distr::StandardNormal;
use rstest::rstest;

// ── rand ─────────────────────────────────────────────────────────────────────

#[rstest]
#[case::f64(0.0f64)]
#[case::f32(0.0f32)]
fn rand_multi_axis<T: FloatLikeTensorElement>(#[case] _t: T)
where
    StandardUniform: Distribution<T>,
{
    let t: Tensor<T> = Tensor::rand(&[2, 3, 4]);
    assert_eq!(t.data().len(), 24);
    assert!(t.data().iter().all(|&x| {
        let x: f64 = x.into();
        (0.0..1.0).contains(&x)
    }));
}

#[test]
fn rand_in_cpu_pure() {
    let t: Tensor<f64, CpuPure> = Tensor::rand_in(&[3, 3]);
    assert_eq!(t.data().len(), 9);
    assert!(t.data().iter().all(|&x| (0.0..1.0).contains(&x)));
}

#[rstest]
#[case::f64(0.0f64)]
#[case::f32(0.0f32)]
fn rand_with_same_seed<T: FloatLikeTensorElement>(#[case] _t: T)
where
    StandardUniform: Distribution<T>,
{
    let a: Tensor<T> = Tensor::rand_with(&[16], &mut StdRng::seed_from_u64(1));
    let b: Tensor<T> = Tensor::rand_with(&[16], &mut StdRng::seed_from_u64(1));
    assert_eq!(a.data(), b.data());
}

// ── randn ────────────────────────────────────────────────────────────────────

#[rstest]
#[case::f64(0.0f64)]
#[case::f32(0.0f32)]
fn randn_multi_axis<T: FloatLikeTensorElement>(#[case] _t: T)
where
    StandardNormal: Distribution<T>,
{
    let t: Tensor<T> = Tensor::randn(&[5, 5]);
    assert_eq!(t.data().len(), 25);
}

#[rstest]
#[case::f64(0.0f64)]
#[case::f32(0.0f32)]
fn randn_with_same_seed<T: FloatLikeTensorElement>(#[case] _t: T)
where
    StandardNormal: Distribution<T>,
{
    let a: Tensor<T> = Tensor::randn_with(&[16], &mut StdRng::seed_from_u64(3));
    let b: Tensor<T> = Tensor::randn_with(&[16], &mut StdRng::seed_from_u64(3));
    assert_eq!(a.data(), b.data());
}

// ── sample ───────────────────────────────────────────────────────────────────

#[test]
fn sample_uniform_range() {
    let dist = Uniform::new(-2.0_f64, 5.0).unwrap();
    let t: Tensor<f64> = Tensor::sample(&[50], dist);
    assert_eq!(t.data().len(), 50);
    assert!(t.data().iter().all(|&x| (-2.0..5.0).contains(&x)));
}

#[test]
fn sample_with_same_seed() {
    let a: Tensor<f64> = Tensor::sample_with(
        &[16],
        Uniform::new(0.0_f64, 100.0).unwrap(),
        &mut StdRng::seed_from_u64(9),
    );
    let b: Tensor<f64> = Tensor::sample_with(
        &[16],
        Uniform::new(0.0_f64, 100.0).unwrap(),
        &mut StdRng::seed_from_u64(9),
    );
    assert_eq!(a.data(), b.data());
}
