mod common;

use candela::{Dimension, FloatLikeTensorElement, Tensor, srange};
use common::assert_approx_eq;
use rstest::rstest;

// ── basic shape correctness ───────────────────────────────────────────────────

#[rstest]
#[case::f64(0.0f64)]
#[case::f32(0.0f32)]
fn matmul_2x3_by_3x4<T: FloatLikeTensorElement>(#[case] _t: T) {
    // [2,3] @ [3,4] = [2,4]
    let a: Tensor<T> = srange!(6, &[2, 3]); // [[0,1,2],[3,4,5]]
    let b = Tensor::<T>::eye(3, 4);
    let c = a.matmul(&b).unwrap().materialize();
    assert_eq!(c.shape(), &[2, 4]);
}

#[rstest]
#[case::f64(0.0f64)]
#[case::f32(0.0f32)]
fn matmul_identity_times_identity<T: FloatLikeTensorElement>(#[case] _t: T) {
    // I @ I = I
    let i = Tensor::<T>::eye(3, 3);
    let result = i.clone().matmul(&i).unwrap().materialize();
    let expected = Tensor::<T>::eye(3, 3);
    assert_eq!(result.data(), expected.data());
}

#[rstest]
#[case::f64(0.0f64)]
#[case::f32(0.0f32)]
fn matmul_known_values<T: FloatLikeTensorElement>(#[case] _t: T) {
    // [[1,2],[3,4]] @ [[1,0],[0,1]] = [[1,2],[3,4]]
    let one = T::ONE;
    let two = one + one;
    let three = two + one;
    let four = three + one;
    let a = Tensor::from_slice(&[one, two, three, four], &[2, 2]);
    let b = Tensor::<T>::eye(2, 2);
    let c = a.clone().matmul(&b).unwrap().materialize();
    assert_eq!(c.data(), a.data());
}

#[rstest]
#[case::f64(0.0f64)]
#[case::f32(0.0f32)]
fn matmul_non_square<T: FloatLikeTensorElement>(#[case] _t: T) {
    // [[1,0,0],[0,1,0],[0,0,1]] @ [[1,2],[3,4],[5,6]] = [[1,2],[3,4],[5,6]]
    let one = T::ONE;
    let two = one + one;
    let three = two + one;
    let four = three + one;
    let five = four + one;
    let six = five + one;
    let a = Tensor::<T>::eye(3, 3);
    let b = Tensor::from_slice(&[one, two, three, four, five, six], &[3, 2]);
    let c = a.matmul(&b).unwrap().materialize();
    assert_eq!(c.shape(), &[3, 2]);
    assert_eq!(c.data(), b.data());
}

#[rstest]
#[case::f64(0.0f64)]
#[case::f32(0.0f32)]
fn matmul_transposed_rhs<T: FloatLikeTensorElement>(#[case] _t: T) {
    // A = [2,3], B^T means B is stored transposed in memory but matmul treats it as [3,4]
    let a: Tensor<T> = srange!(6, &[2, 3]);
    let b: Tensor<T> = srange!(12, &[4, 3]); // will be transposed → [3,4]
    let bt = b.transpose();
    let c = a.matmul(&bt).unwrap().materialize();
    assert_eq!(c.shape(), &[2, 4]);
}

#[rstest]
#[case::f64(0.0f64)]
#[case::f32(0.0f32)]
fn matmul_batched_shape<T: FloatLikeTensorElement>(#[case] _t: T) {
    // [2,3,4] @ [2,4,5] = [2,3,5]
    let a = Tensor::from_scalar(T::ONE, &[2, 3, 4]);
    let b = Tensor::from_scalar(T::ONE, &[2, 4, 5]);
    let c = a.matmul(&b).unwrap().materialize();
    assert_eq!(c.shape(), &[2, 3, 5]);
}

#[rstest]
#[case::f64(0.0f64)]
#[case::f32(0.0f32)]
fn matmul_batched_values<T: FloatLikeTensorElement>(#[case] _t: T) {
    // All-ones [2,3,4] @ all-ones [2,4,5]: each output element = sum of 4 ones = 4.0
    let a = Tensor::from_scalar(T::ONE, &[2, 3, 4]);
    let b = Tensor::from_scalar(T::ONE, &[2, 4, 5]);
    let c = a.matmul(&b).unwrap().materialize();
    assert_approx_eq(c.data(), &vec![4.0; 2 * 3 * 5]);
}

#[test]
fn matmul_shape_mismatch() {
    // [3,4] @ [3,4] — inner dims 4 != 3
    let a = Tensor::from_scalar(1.0, &[3, 4]);
    let b = Tensor::from_scalar(1.0, &[3, 4]);
    assert!(a.matmul(&b).is_err());
}

#[rstest]
#[case::f64(0.0f64)]
#[case::f32(0.0f32)]
fn matmul_not_symmetric<T: FloatLikeTensorElement>(#[case] _t: T) {
    // A @ B != B @ A in general
    // [[1,2],[3,4]] @ [[5,6],[7,8]] vs [[5,6],[7,8]] @ [[1,2],[3,4]]
    let one = T::ONE;
    let two = one + one;
    let three = two + one;
    let four = three + one;
    let five = four + one;
    let six = five + one;
    let seven = six + one;
    let eight = seven + one;
    let a = Tensor::from_slice(&[one, two, three, four], &[2, 2]);
    let b = Tensor::from_slice(&[five, six, seven, eight], &[2, 2]);
    let ab = a.clone().matmul(&b.clone()).unwrap().materialize();
    let ba = b.matmul(&a).unwrap().materialize();
    assert_ne!(ab.data(), ba.data());
}

// ── error cases: bias shape mismatches ───────────────────────────────────────

#[test]
#[should_panic]
fn matmul_plus_bias_wrong_shape_panics() {
    // (A@B) shape is [2,2]; bias is [3,3] — incompatible, no broadcast possible
    let a = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let b = Tensor::from_slice(&[5.0, 6.0, 7.0, 8.0], &[2, 2]);
    let bias = Tensor::from_scalar(1.0, &[3, 3]);
    let _ = (a.matmul(&b).unwrap() + bias).materialize();
}

#[test]
#[should_panic]
fn matmul_minus_bias_wrong_shape_panics() {
    let a = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let b = Tensor::from_slice(&[5.0, 6.0, 7.0, 8.0], &[2, 2]);
    let bias = Tensor::from_scalar(1.0, &[3, 3]);
    let _ = (a.matmul(&b).unwrap() - bias).materialize();
}

// ── matmul scaled and fused with bias ────────────────────────────────────────
// These exercise alpha*(A@B) and alpha*(A@B) +/- bias chains.
// MatMulSum fusion is live: consecutive MatMul + ScalarOp / Add / Sub nodes
// fuse at graph-construction time, so these tests exercise the fused code path.
//
// Test matrices used throughout this section:
//   A = [[1,2],[3,4]], B = [[5,6],[7,8]]
//   A @ B = [[19,22],[43,50]]

#[rstest]
#[case::f64(0.0f64)]
#[case::f32(0.0f32)]
fn matmul_scaled<T: FloatLikeTensorElement>(#[case] _t: T) {
    let one = T::ONE;
    let two = one + one;
    let three = two + one;
    let four = three + one;
    let five = four + one;
    let six = five + one;
    let seven = six + one;
    let eight = seven + one;
    let a = Tensor::from_slice(&[one, two, three, four], &[2, 2]);
    let b = Tensor::from_slice(&[five, six, seven, eight], &[2, 2]);
    let c = (a.matmul(&b).unwrap() * two).materialize();
    assert_eq!(c.shape(), &[2, 2]);
    // 2 * [[19,22],[43,50]] = [[38,44],[86,100]]
    assert_approx_eq(c.data(), &[38.0, 44.0, 86.0, 100.0]);
}

#[rstest]
#[case::f64(0.0f64)]
#[case::f32(0.0f32)]
fn matmul_plus_bias<T: FloatLikeTensorElement>(#[case] _t: T) {
    let one = T::ONE;
    let two = one + one;
    let three = two + one;
    let four = three + one;
    let five = four + one;
    let six = five + one;
    let seven = six + one;
    let eight = seven + one;
    let a = Tensor::from_slice(&[one, two, three, four], &[2, 2]);
    let b = Tensor::from_slice(&[five, six, seven, eight], &[2, 2]);
    let bias = Tensor::from_slice(&[one, T::ZERO, T::ZERO, one], &[2, 2]); // identity
    let c = (a.matmul(&b).unwrap() + bias).materialize();
    assert_eq!(c.shape(), &[2, 2]);
    // [[19,22],[43,50]] + I = [[20,22],[43,51]]
    assert_approx_eq(c.data(), &[20.0, 22.0, 43.0, 51.0]);
}

#[rstest]
#[case::f64(0.0f64)]
#[case::f32(0.0f32)]
fn matmul_minus_bias<T: FloatLikeTensorElement>(#[case] _t: T) {
    let one = T::ONE;
    let two = one + one;
    let three = two + one;
    let four = three + one;
    let five = four + one;
    let six = five + one;
    let seven = six + one;
    let eight = seven + one;
    let a = Tensor::from_slice(&[one, two, three, four], &[2, 2]);
    let b = Tensor::from_slice(&[five, six, seven, eight], &[2, 2]);
    let bias = Tensor::from_slice(&[one, T::ZERO, T::ZERO, one], &[2, 2]); // identity
    let c = (a.matmul(&b).unwrap() - bias).materialize();
    assert_eq!(c.shape(), &[2, 2]);
    // [[19,22],[43,50]] - I = [[18,22],[43,49]]
    assert_approx_eq(c.data(), &[18.0, 22.0, 43.0, 49.0]);
}

#[rstest]
#[case::f64(0.0f64)]
#[case::f32(0.0f32)]
fn matmul_scaled_plus_bias<T: FloatLikeTensorElement>(#[case] _t: T) {
    let one = T::ONE;
    let two = one + one;
    let three = two + one;
    let four = three + one;
    let five = four + one;
    let six = five + one;
    let seven = six + one;
    let eight = seven + one;
    let a = Tensor::from_slice(&[one, two, three, four], &[2, 2]);
    let b = Tensor::from_slice(&[five, six, seven, eight], &[2, 2]);
    let bias = Tensor::from_slice(&[one, T::ZERO, T::ZERO, one], &[2, 2]);
    let c = (a.matmul(&b).unwrap() * two + bias).materialize();
    assert_eq!(c.shape(), &[2, 2]);
    // 2*[[19,22],[43,50]] + I = [[39,44],[86,101]]
    assert_approx_eq(c.data(), &[39.0, 44.0, 86.0, 101.0]);
}

#[rstest]
#[case::f64(0.0f64)]
#[case::f32(0.0f32)]
fn matmul_scaled_minus_bias<T: FloatLikeTensorElement>(#[case] _t: T) {
    let one = T::ONE;
    let two = one + one;
    let three = two + one;
    let four = three + one;
    let five = four + one;
    let six = five + one;
    let seven = six + one;
    let eight = seven + one;
    let a = Tensor::from_slice(&[one, two, three, four], &[2, 2]);
    let b = Tensor::from_slice(&[five, six, seven, eight], &[2, 2]);
    let bias = Tensor::from_slice(&[one, T::ZERO, T::ZERO, one], &[2, 2]);
    let c = (a.matmul(&b).unwrap() * two - bias).materialize();
    assert_eq!(c.shape(), &[2, 2]);
    // 2*[[19,22],[43,50]] - I = [[37,44],[86,99]]
    assert_approx_eq(c.data(), &[37.0, 44.0, 86.0, 99.0]);
}

// ── batched matmul with bias ──────────────────────────────────────────────────

#[rstest]
#[case::f64(0.0f64)]
#[case::f32(0.0f32)]
fn matmul_batched_plus_bias<T: FloatLikeTensorElement>(#[case] _t: T) {
    // [2,3,4] @ [2,4,5] = [2,3,5]; each element = 4.0 (sum of 4 ones)
    // Adding all-ones bias[2,3,5]: each result element = 5.0
    let a = Tensor::from_scalar(T::ONE, &[2, 3, 4]);
    let b = Tensor::from_scalar(T::ONE, &[2, 4, 5]);
    let bias = Tensor::from_scalar(T::ONE, &[2, 3, 5]);
    let c = (a.matmul(&b).unwrap() + bias).materialize();
    assert_eq!(c.shape(), &[2, 3, 5]);
    assert_approx_eq(c.data(), &vec![5.0; 2 * 3 * 5]);
}

// ── batch-dimension broadcasting ────────────────────────

#[rstest]
#[case::f64(0.0f64)]
#[case::f32(0.0f32)]
fn matmul_broadcast_batch_lhs_one<T: FloatLikeTensorElement>(#[case] _t: T) {
    // [1,3,4] @ [2,4,5] should broadcast lhs batch dim: result is [2,3,5]
    // Each output slice is the single lhs matrix multiplied by the corresponding rhs batch.
    let a = Tensor::from_scalar(T::ONE, &[1, 3, 4]);
    let b = Tensor::from_scalar(T::ONE, &[2, 4, 5]);
    let c = a.matmul(&b).unwrap().materialize();
    assert_eq!(c.shape(), &[2, 3, 5]);
    // Each output element is the sum of 4 ones.
    assert_approx_eq(c.data(), &vec![4.0; 2 * 3 * 5]);
}

#[rstest]
#[case::f64(0.0f64)]
#[case::f32(0.0f32)]
fn matmul_broadcast_batch_rhs_one<T: FloatLikeTensorElement>(#[case] _t: T) {
    // [2,3,4] @ [1,4,5] should broadcast rhs batch dim: result is [2,3,5]
    let a = Tensor::from_scalar(T::ONE, &[2, 3, 4]);
    let b = Tensor::from_scalar(T::ONE, &[1, 4, 5]);
    let c = a.matmul(&b).unwrap().materialize();
    assert_eq!(c.shape(), &[2, 3, 5]);
    assert_approx_eq(c.data(), &vec![4.0; 2 * 3 * 5]);
}
