mod common;

use candela::{Tensor, arange, ones, zeros};

// ── scalar ops ────────────────────────────────────────────────────────────────

#[test]
fn scalar_add() {
    let t = ones!(&[4]);
    assert_eq!((t + 2.0).materialize().data(), &vec![3.0; 4]);
}

#[test]
fn scalar_sub() {
    let t = ones!(&[4]);
    assert_eq!((t - 2.0).materialize().data(), &vec![-1.0; 4]);
}

#[test]
fn scalar_mul() {
    let t = ones!(&[4]);
    assert_eq!((t * 3.0).materialize().data(), &vec![3.0; 4]);
}

#[test]
fn scalar_div() {
    let t = ones!(&[4]);
    assert_eq!((t / 4.0).materialize().data(), &vec![0.25; 4]);
}

#[test]
fn scalar_exp() {
    // e^0 = 1
    let t = zeros!(&[4]);
    let r = t.exp().materialize();
    assert_eq!(r.data(), &vec![1.0; 4]);
}

#[test]
fn scalar_ln() {
    // ln(1) = 0
    let t = ones!(&[4]);
    let r = t.ln().materialize();
    assert_eq!(r.data(), &vec![0.0; 4]);
}

#[test]
fn scalar_log2() {
    // log2(2) = 1
    let t = Tensor::from_scalar(2.0, &[4]);
    let r = t.log2().materialize();
    assert_eq!(r.data(), &vec![1.0; 4]);
}

// ── scalar fusion chain ───────────────────────────────────────────────────────

#[test]
fn fused_chain_long() {
    // 20 sequential additions: x + 0 + 1 + ... + 19, starting from x = 1
    // expected: 1 + (0+1+2+...+19) = 1 + 190 = 191
    let t = ones!(&[4]);
    let mut p = t.as_promise();
    for i in 0..20_u32 {
        p = p + i as f64;
    }
    let result = p.materialize();
    assert_eq!(result.data(), &vec![190.0 + 1.0; 4]);
}

// ── non-commutativity ─────────────────────────────────────────────────────────

#[test]
fn sub_is_not_commutative() {
    // [0,1,2,3] - [1,1,1,1] = [-1,0,1,2]
    // [1,1,1,1] - [0,1,2,3] = [1,0,-1,-2]
    let a = arange!(4);
    let b = ones!(&[4]);
    let ab = (a.clone() - b.clone()).materialize();
    let ba = (b - a).materialize();
    assert_ne!(ab.data(), ba.data());
    assert_eq!(ab.data(), &vec![-1.0, 0.0, 1.0, 2.0]);
    assert_eq!(ba.data(), &vec![1.0, 0.0, -1.0, -2.0]);
}

#[test]
fn div_is_not_commutative() {
    // [4,6,8] / [2,2,2] = [2,3,4]
    // [2,2,2] / [4,6,8] != [2,3,4]
    let a = Tensor::from_slice(&[4.0, 6.0, 8.0], &[3]);
    let b = Tensor::from_scalar(2.0, &[3]);
    let ab = (a.clone() / b.clone()).materialize();
    let ba = (b / a).materialize();
    assert_ne!(ab.data(), ba.data());
    assert_eq!(ab.data(), &vec![2.0, 3.0, 4.0]);
}

// ── binary tensor ops ─────────────────────────────────────────────────────────

#[test]
fn tensor_add() {
    // [1,2,3] + [4,5,6] = [5,7,9]
    let a = Tensor::from_slice(&[1.0, 2.0, 3.0], &[3]);
    let b = Tensor::from_slice(&[4.0, 5.0, 6.0], &[3]);
    assert_eq!((a + b).materialize().data(), &vec![5.0, 7.0, 9.0]);
}

#[test]
fn tensor_sub() {
    // [4,5,6] - [1,2,3] = [3,3,3]
    let a = Tensor::from_slice(&[4.0, 5.0, 6.0], &[3]);
    let b = Tensor::from_slice(&[1.0, 2.0, 3.0], &[3]);
    assert_eq!((a - b).materialize().data(), &vec![3.0, 3.0, 3.0]);
}

#[test]
fn tensor_mul() {
    // [1,2,3] * [4,5,6] = [4,10,18]
    let a = Tensor::from_slice(&[1.0, 2.0, 3.0], &[3]);
    let b = Tensor::from_slice(&[4.0, 5.0, 6.0], &[3]);
    assert_eq!((a * b).materialize().data(), &vec![4.0, 10.0, 18.0]);
}

#[test]
fn tensor_div() {
    // [4,6,8] / [2,3,4] = [2,2,2]
    let a = Tensor::from_slice(&[4.0, 6.0, 8.0], &[3]);
    let b = Tensor::from_slice(&[2.0, 3.0, 4.0], &[3]);
    assert_eq!((a / b).materialize().data(), &vec![2.0, 2.0, 2.0]);
}

#[test]
#[should_panic]
fn tensor_add_shape_mismatch_panics() {
    let a = ones!(&[3]);
    let b = ones!(&[4]);
    let _ = (a + b).materialize();
}
