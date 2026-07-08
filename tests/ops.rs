mod common;

use candela::{FloatLikeTensorElement, Tensor, arange, ones};
use common::{assert_approx_eq, tensor_of};
use rstest::rstest;

// ── scalar ops ────────────────────────────────────────────────────────────────

#[rstest]
#[case::f64(0.0f64)]
#[case::f32(0.0f32)]
fn scalar_add<T: FloatLikeTensorElement>(#[case] _t: T) {
    let t = Tensor::from_scalar(T::from_f64(1.0), &[4]);
    assert_approx_eq((t + T::from_f64(2.0)).materialize().data(), &[3.0; 4]);
}

#[rstest]
#[case::f64(0.0f64)]
#[case::f32(0.0f32)]
fn scalar_sub<T: FloatLikeTensorElement>(#[case] _t: T) {
    let t = Tensor::from_scalar(T::from_f64(1.0), &[4]);
    assert_approx_eq((t - T::from_f64(2.0)).materialize().data(), &[-1.0; 4]);
}

#[rstest]
#[case::f64(0.0f64)]
#[case::f32(0.0f32)]
fn scalar_mul<T: FloatLikeTensorElement>(#[case] _t: T) {
    let t = Tensor::from_scalar(T::from_f64(1.0), &[4]);
    assert_approx_eq((t * T::from_f64(3.0)).materialize().data(), &[3.0; 4]);
}

#[rstest]
#[case::f64(0.0f64)]
#[case::f32(0.0f32)]
fn scalar_div<T: FloatLikeTensorElement>(#[case] _t: T) {
    let t = Tensor::from_scalar(T::from_f64(1.0), &[4]);
    assert_approx_eq((t / T::from_f64(4.0)).materialize().data(), &[0.25; 4]);
}

#[rstest]
#[case::f64(0.0f64)]
#[case::f32(0.0f32)]
fn scalar_exp<T: FloatLikeTensorElement>(#[case] _t: T) {
    // e^0 = 1
    let t = Tensor::from_scalar(T::from_f64(0.0), &[4]);
    assert_eq!(t.exp().materialize().data(), &vec![T::from_f64(1.0); 4]);
}

#[rstest]
#[case::f64(0.0f64)]
#[case::f32(0.0f32)]
fn scalar_ln<T: FloatLikeTensorElement>(#[case] _t: T) {
    // ln(1) = 0
    let t = Tensor::from_scalar(T::from_f64(1.0), &[4]);
    assert_eq!(t.ln().materialize().data(), &vec![T::from_f64(0.0); 4]);
}

#[rstest]
#[case::f64(0.0f64)]
#[case::f32(0.0f32)]
fn scalar_log2<T: FloatLikeTensorElement>(#[case] _t: T) {
    // log2(2) = 1
    let t = Tensor::from_scalar(T::from_f64(2.0), &[4]);
    assert_eq!(t.log2().materialize().data(), &vec![T::from_f64(1.0); 4]);
}

// ── scalar fusion chain ───────────────────────────────────────────────────────

#[test]
fn fused_chain_long() {
    // 20 sequential additions: x + 0 + 1 + ... + 19, starting from x = 1
    // expected: 1 + (0+1+2+...+19) = 1 + 190 = 191
    let t = ones!(&[4]);
    let mut p = t.to_promise();
    for i in 0..20_u32 {
        p += i as f64;
    }
    let result = p.materialize();
    assert_eq!(result.data(), &vec![190.0 + 1.0; 4]);
}

// ── non-commutativity ─────────────────────────────────────────────────────────

#[rstest]
#[case::f64(0.0f64)]
#[case::f32(0.0f32)]
fn sub_is_not_commutative<T: FloatLikeTensorElement>(#[case] _t: T) {
    // [0,1,2,3] - [1,1,1,1] = [-1,0,1,2]
    // [1,1,1,1] - [0,1,2,3] = [1,0,-1,-2]
    let a: Tensor<T> = arange!(4);
    let b = Tensor::from_scalar(T::from_f64(1.0), &[4]);
    let ab = (a.clone() - b.clone()).materialize();
    let ba = (b - a).materialize();
    assert_ne!(ab.data(), ba.data());
    assert_approx_eq(ab.data(), &[-1.0, 0.0, 1.0, 2.0]);
    assert_approx_eq(ba.data(), &[1.0, 0.0, -1.0, -2.0]);
}

#[rstest]
#[case::f64(0.0f64)]
#[case::f32(0.0f32)]
fn div_is_not_commutative<T: FloatLikeTensorElement>(#[case] _t: T) {
    // [4,6,8] / [2,2,2] = [2,3,4]
    // [2,2,2] / [4,6,8] != [2,3,4]
    let a = tensor_of::<T>(&[4.0, 6.0, 8.0], &[3]);
    let b = Tensor::from_scalar(T::from_f64(2.0), &[3]);
    let ab = (a.clone() / b.clone()).materialize();
    let ba = (b / a).materialize();
    assert_ne!(ab.data(), ba.data());
    assert_approx_eq(ab.data(), &[2.0, 3.0, 4.0]);
}

// ── binary tensor ops ─────────────────────────────────────────────────────────

#[rstest]
#[case::f64(0.0f64)]
#[case::f32(0.0f32)]
fn tensor_add<T: FloatLikeTensorElement>(#[case] _t: T) {
    // [1,2,3] + [4,5,6] = [5,7,9]
    let a = tensor_of::<T>(&[1.0, 2.0, 3.0], &[3]);
    let b = tensor_of::<T>(&[4.0, 5.0, 6.0], &[3]);
    assert_approx_eq((a + b).materialize().data(), &[5.0, 7.0, 9.0]);
}

#[rstest]
#[case::f64(0.0f64)]
#[case::f32(0.0f32)]
fn tensor_sub<T: FloatLikeTensorElement>(#[case] _t: T) {
    // [4,5,6] - [1,2,3] = [3,3,3]
    let a = tensor_of::<T>(&[4.0, 5.0, 6.0], &[3]);
    let b = tensor_of::<T>(&[1.0, 2.0, 3.0], &[3]);
    assert_approx_eq((a - b).materialize().data(), &[3.0, 3.0, 3.0]);
}

#[rstest]
#[case::f64(0.0f64)]
#[case::f32(0.0f32)]
fn tensor_mul<T: FloatLikeTensorElement>(#[case] _t: T) {
    // [1,2,3] * [4,5,6] = [4,10,18]
    let a = tensor_of::<T>(&[1.0, 2.0, 3.0], &[3]);
    let b = tensor_of::<T>(&[4.0, 5.0, 6.0], &[3]);
    assert_approx_eq((a * b).materialize().data(), &[4.0, 10.0, 18.0]);
}

#[rstest]
#[case::f64(0.0f64)]
#[case::f32(0.0f32)]
fn tensor_div<T: FloatLikeTensorElement>(#[case] _t: T) {
    // [4,6,8] / [2,3,4] = [2,2,2]
    let a = tensor_of::<T>(&[4.0, 6.0, 8.0], &[3]);
    let b = tensor_of::<T>(&[2.0, 3.0, 4.0], &[3]);
    assert_approx_eq((a / b).materialize().data(), &[2.0, 2.0, 2.0]);
}

#[test]
#[should_panic]
fn tensor_add_shape_mismatch_panics() {
    let a = ones!(&[3]);
    let b = ones!(&[4]);
    let _ = (a + b).materialize();
}
