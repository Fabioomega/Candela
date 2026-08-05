mod common;

use candela::{FloatLikeTensorElement, Tensor, arange, ones};
use common::{assert_approx_eq, cast, tensor_of};
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

#[rstest]
#[case::f64(0.0f64)]
#[case::f32(0.0f32)]
fn scalar_recip<T: FloatLikeTensorElement>(#[case] _t: T) {
    let t = tensor_of::<T>(&[1.0, 2.0, 4.0, 8.0], &[4]);
    assert_eq!(
        t.recip().materialize().data(),
        &cast::<T>(&[1.0, 0.5, 0.25, 0.125])
    );
}

#[rstest]
#[case::f64(0.0f64)]
#[case::f32(0.0f32)]
fn scalar_recip_non_contiguous<T: FloatLikeTensorElement>(#[case] _t: T) {
    let t = tensor_of::<T>(&[1.0, 2.0, 4.0, 8.0], &[2, 2]);
    assert_eq!(
        t.transpose().recip().materialize().data(),
        &cast::<T>(&[1.0, 0.25, 0.5, 0.125])
    );
}

#[rstest]
#[case::f64(0.0f64)]
#[case::f32(0.0f32)]
fn scalar_neg<T: FloatLikeTensorElement>(#[case] _t: T) {
    let t = tensor_of::<T>(&[1.0, -2.0, 3.0, -4.0], &[4]);
    assert_approx_eq((-&t).materialize().data(), &[-1.0, 2.0, -3.0, 4.0]);
    assert_approx_eq((-t).materialize().data(), &[-1.0, 2.0, -3.0, 4.0]);
}

#[rstest]
#[case::f64(0.0f64)]
#[case::f32(0.0f32)]
fn scalar_neg_twice_is_identity<T: FloatLikeTensorElement>(#[case] _t: T) {
    let t = tensor_of::<T>(&[1.0, -2.0, 3.0, -4.0], &[4]);
    assert_approx_eq((-(-&t)).materialize().data(), &[1.0, -2.0, 3.0, -4.0]);
}

// ── scalar on the left ────────────────────────────────────────────────────────

// These tests cannot be generic because the orphan rule sucks. f32 only.

#[test]
fn scalar_lhs_add() {
    let t = tensor_of::<f32>(&[1.0, 2.0, 3.0], &[3]);
    assert_approx_eq((2.0f32 + &t).materialize().data(), &[3.0, 4.0, 5.0]);
    assert_approx_eq((2.0f32 + t).materialize().data(), &[3.0, 4.0, 5.0]);
}

#[test]
fn scalar_lhs_sub() {
    // 2 - [1,2,3] = [1,0,-1], the mirror of [1,2,3] - 2 = [-1,0,1].
    let t = tensor_of::<f32>(&[1.0, 2.0, 3.0], &[3]);
    assert_approx_eq((2.0f32 - &t).materialize().data(), &[1.0, 0.0, -1.0]);
    assert_approx_eq((t - 2.0f32).materialize().data(), &[-1.0, 0.0, 1.0]);
}

#[test]
fn scalar_lhs_mul() {
    let t = tensor_of::<f32>(&[1.0, 2.0, 3.0], &[3]);
    assert_approx_eq((3.0f32 * &t).materialize().data(), &[3.0, 6.0, 9.0]);
    assert_approx_eq((3.0f32 * t).materialize().data(), &[3.0, 6.0, 9.0]);
}

#[test]
fn scalar_lhs_div() {
    // 8 / [1,2,4] = [8,4,2], the mirror of [1,2,4] / 8.
    let t = tensor_of::<f32>(&[1.0, 2.0, 4.0], &[3]);
    assert_approx_eq((8.0f32 / &t).materialize().data(), &[8.0, 4.0, 2.0]);
    assert_approx_eq((t / 8.0f32).materialize().data(), &[0.125, 0.25, 0.5]);
}

#[test]
fn scalar_lhs_div_matches_recip_then_scale() {
    // `s / t` lowers to a reciprocal plus a scale; the sugar must not drift from
    // the expression it replaces.
    let t = tensor_of::<f32>(&[1.0, 2.0, 4.0], &[3]);
    let sugar = (8.0f32 / &t).materialize();
    let manual = (t.recip() * 8.0f32).materialize();
    assert_eq!(sugar.data(), manual.data());
}

#[test]
fn scalar_lhs_leads_a_promise_chain() {
    // The impls cover promises, not just `Tensor`, so a scalar can lead an
    // expression whose right-hand side is itself an operation.
    let t = tensor_of::<f32>(&[1.0, 2.0, 3.0], &[3]);
    let r = 1.0f32 - (t * 2.0f32);
    assert_approx_eq(r.materialize().data(), &[-1.0, -3.0, -5.0]);
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
