mod common;

use candela::{Dimension, Tensor, srange};
use common::assert_approx_eq;

// ── basic shape correctness ───────────────────────────────────────────────────

#[test]
fn matmul_2x3_by_3x4() {
    // [2,3] @ [3,4] = [2,4]
    let a: Tensor<f64> = srange!(6, &[2, 3]); // [[0,1,2],[3,4,5]]
    let b = Tensor::eye(3, 4);
    let c = a.matmul(&b).unwrap().materialize();
    assert_eq!(c.shape(), &[2, 4]);
}

#[test]
fn matmul_identity_times_identity() {
    // I @ I = I
    let i = Tensor::<f64>::eye(3, 3);
    let result = i.clone().matmul(&i).unwrap().materialize();
    let expected = Tensor::<f64>::eye(3, 3);
    assert_approx_eq(result.data(), expected.data());
}

#[test]
fn matmul_known_values() {
    // [[1,2],[3,4]] @ [[1,0],[0,1]] = [[1,2],[3,4]]
    let a = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let b = Tensor::<f64>::eye(2, 2);
    let c = a.clone().matmul(&b).unwrap().materialize();
    assert_approx_eq(c.data(), a.data());
}

#[test]
fn matmul_non_square() {
    // [[1,0,0],[0,1,0],[0,0,1]] @ [[1,2],[3,4],[5,6]] = [[1,2],[3,4],[5,6]]
    let a = Tensor::<f64>::eye(3, 3);
    let b = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]);
    let c = a.matmul(&b).unwrap().materialize();
    assert_eq!(c.shape(), &[3, 2]);
    assert_approx_eq(c.data(), b.data());
}

#[test]
fn matmul_transposed_rhs() {
    // A = [2,3], B^T means B is stored transposed in memory but matmul treats it as [3,4]
    let a: Tensor<f64> = srange!(6, &[2, 3]);
    let b = srange!(12, &[4, 3]); // will be transposed → [3,4]
    let bt = b.transpose();
    let c = a.matmul(&bt).unwrap().materialize();
    assert_eq!(c.shape(), &[2, 4]);
}

#[test]
fn matmul_batched_shape() {
    // [2,3,4] @ [2,4,5] = [2,3,5]
    let a = Tensor::from_scalar(1.0, &[2, 3, 4]);
    let b = Tensor::from_scalar(1.0, &[2, 4, 5]);
    let c = a.matmul(&b).unwrap().materialize();
    assert_eq!(c.shape(), &[2, 3, 5]);
}

#[test]
fn matmul_batched_values() {
    // All-ones [2,3,4] @ all-ones [2,4,5]: each output element = sum of 4 ones = 4.0
    let a = Tensor::from_scalar(1.0, &[2, 3, 4]);
    let b = Tensor::from_scalar(1.0, &[2, 4, 5]);
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

#[test]
fn matmul_not_symmetric() {
    // A @ B != B @ A in general
    // [[1,2],[3,4]] @ [[5,6],[7,8]] vs [[5,6],[7,8]] @ [[1,2],[3,4]]
    let a = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let b = Tensor::from_slice(&[5.0, 6.0, 7.0, 8.0], &[2, 2]);
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

#[test]
fn matmul_scaled() {
    let a = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let b = Tensor::from_slice(&[5.0, 6.0, 7.0, 8.0], &[2, 2]);
    let c = (a.matmul(&b).unwrap() * 2.0).materialize();
    assert_eq!(c.shape(), &[2, 2]);
    // 2 * [[19,22],[43,50]] = [[38,44],[86,100]]
    assert_approx_eq(c.data(), &[38.0, 44.0, 86.0, 100.0]);
}

#[test]
fn matmul_plus_bias() {
    let a = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let b = Tensor::from_slice(&[5.0, 6.0, 7.0, 8.0], &[2, 2]);
    let bias = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], &[2, 2]); // identity
    let c = (a.matmul(&b).unwrap() + bias).materialize();
    assert_eq!(c.shape(), &[2, 2]);
    // [[19,22],[43,50]] + I = [[20,22],[43,51]]
    assert_approx_eq(c.data(), &[20.0, 22.0, 43.0, 51.0]);
}

#[test]
fn matmul_minus_bias() {
    let a = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let b = Tensor::from_slice(&[5.0, 6.0, 7.0, 8.0], &[2, 2]);
    let bias = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], &[2, 2]); // identity
    let c = (a.matmul(&b).unwrap() - bias).materialize();
    assert_eq!(c.shape(), &[2, 2]);
    // [[19,22],[43,50]] - I = [[18,22],[43,49]]
    assert_approx_eq(c.data(), &[18.0, 22.0, 43.0, 49.0]);
}

#[test]
fn matmul_scaled_plus_bias() {
    let a = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let b = Tensor::from_slice(&[5.0, 6.0, 7.0, 8.0], &[2, 2]);
    let bias = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], &[2, 2]);
    let c = (a.matmul(&b).unwrap() * 2.0 + bias).materialize();
    assert_eq!(c.shape(), &[2, 2]);
    // 2*[[19,22],[43,50]] + I = [[39,44],[86,101]]
    assert_approx_eq(c.data(), &[39.0, 44.0, 86.0, 101.0]);
}

#[test]
fn matmul_scaled_minus_bias() {
    let a = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let b = Tensor::from_slice(&[5.0, 6.0, 7.0, 8.0], &[2, 2]);
    let bias = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], &[2, 2]);
    let c = (a.matmul(&b).unwrap() * 2.0 - bias).materialize();
    assert_eq!(c.shape(), &[2, 2]);
    // 2*[[19,22],[43,50]] - I = [[37,44],[86,99]]
    assert_approx_eq(c.data(), &[37.0, 44.0, 86.0, 99.0]);
}

// ── batched matmul with bias ──────────────────────────────────────────────────

#[test]
fn matmul_batched_plus_bias() {
    // [2,3,4] @ [2,4,5] = [2,3,5]; each element = 4.0 (sum of 4 ones)
    // Adding all-ones bias[2,3,5]: each result element = 5.0
    let a = Tensor::from_scalar(1.0, &[2, 3, 4]);
    let b = Tensor::from_scalar(1.0, &[2, 4, 5]);
    let bias = Tensor::from_scalar(1.0, &[2, 3, 5]);
    let c = (a.matmul(&b).unwrap() + bias).materialize();
    assert_eq!(c.shape(), &[2, 3, 5]);
    assert_approx_eq(c.data(), &vec![5.0; 2 * 3 * 5]);
}

// ── batch-dimension broadcasting ────────────────────────
// Broadcasting the batch dimension of matmul requires fixing the `check_broadcast`
// predicate in `matmul_tensor_impl` (currently uses `==` instead of `!=`) and
// wiring broadcast strides (stride=0 for the expanded batch dim) through to the
// batched-GEMM call.

#[test]
fn matmul_broadcast_batch_lhs_one() {
    // [1,3,4] @ [2,4,5] should broadcast lhs batch dim: result is [2,3,5]
    // Each output slice is the single lhs matrix multiplied by the corresponding rhs batch.
    let a = Tensor::from_scalar(1.0, &[1, 3, 4]);
    let b = Tensor::from_scalar(1.0, &[2, 4, 5]);
    let c = a.matmul(&b).unwrap().materialize();
    assert_eq!(c.shape(), &[2, 3, 5]);
    // Each output element is the sum of 4 ones.
    assert_approx_eq(c.data(), &vec![4.0; 2 * 3 * 5]);
}

#[test]
fn matmul_broadcast_batch_rhs_one() {
    // [2,3,4] @ [1,4,5] should broadcast rhs batch dim: result is [2,3,5]
    let a = Tensor::from_scalar(1.0, &[2, 3, 4]);
    let b = Tensor::from_scalar(1.0, &[1, 4, 5]);
    let c = a.matmul(&b).unwrap().materialize();
    assert_eq!(c.shape(), &[2, 3, 5]);
    assert_approx_eq(c.data(), &vec![4.0; 2 * 3 * 5]);
}
