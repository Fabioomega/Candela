mod common;

use candela::{Dimension, FloatLikeTensorElement, Tensor, arange, ones};
use common::{assert_approx_eq, tensor_of};
use rstest::rstest;

// Bug: OpKindScalar::Sub was doing addition (copy-pasted from Add arm).
#[rstest]
#[case::f64(0.0f64)]
#[case::f32(0.0f32)]
fn regression_scalar_sub_was_adding<T: FloatLikeTensorElement>(#[case] _t: T) {
    let t = Tensor::from_scalar(T::from_f64(10.0), &[4]);
    assert_approx_eq((t - T::from_f64(3.0)).materialize().data(), &[7.0; 4]);
}

// Bug: Sub with scalar on a tensor starting from ones should not add.
#[test]
fn regression_scalar_sub_from_ones() {
    let t = ones!(&[4]);
    assert_eq!((t - 3.0).materialize().data(), &vec![-2.0; 4]);
}

// Bug: NotSameShape error reported inputs[0].shape() for both sides.
#[test]
fn regression_not_same_shape_error_shows_both_shapes() {
    let a = ones!(&[3, 4]);
    let b = ones!(&[3, 5]);
    let result = std::panic::catch_unwind(|| {
        let _ = (a + b).materialize();
    });
    assert!(result.is_err(), "expected a panic for shape mismatch");
}

// Bug: unordered buffer reuse could pick rhs as output, reversing Sub/Div.
#[rstest]
#[case::f64(0.0f64)]
#[case::f32(0.0f32)]
fn regression_sub_ordering_with_reusable_rhs<T: FloatLikeTensorElement>(#[case] _t: T) {
    // a - b must be a - b, not b - a
    let a = Tensor::from_scalar(T::from_f64(10.0), &[4]);
    let b = Tensor::from_scalar(T::from_f64(3.0), &[4]);
    // Adding 0 makes b go through a scalar op node (potentially reusable)
    let b_node = b + T::from_f64(0.0);
    let result = (a - b_node).materialize();
    assert_approx_eq(result.data(), &[7.0; 4]); // not [-7.0; 4]
}

// Bug: redirect table is static — applied to every input lookup regardless of
// execution order. If an independent consumer of node A (call it C) appears
// before AsContiguous(A) (call it B) in the topological sort, the redirect
// A→B causes C to request B's buffer before B has been computed.
#[rstest]
#[case::f64(0.0f64)]
#[case::f32(0.0f32)]
fn bug_redirect_timing_independent_consumer_before_as_contiguous<T: FloatLikeTensorElement>(#[case] _t: T) {
    let t = tensor_of::<T>(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let transposed = t.transpose(); // node_1
    let contiguous = transposed.as_contiguous(); // node_2
    let shifted = &transposed + T::from_f64(1.0); // node_3, independent of node_2

    // root.inputs = [node_2, node_3] → sort yields node_3 before node_2
    let result = (&contiguous + &shifted).materialize();

    // transposed = [[1,3],[2,4]], contiguous = [1,3,2,4], shifted = [[2,4],[3,5]]
    // contiguous + shifted = [[3,7],[5,9]]
    assert_approx_eq(result.data(), &[3.0, 7.0, 5.0, 9.0]);
}

// Bug: matmul inputs.pop() order was reversed (raw_a got rhs, raw_b got lhs).
#[rstest]
#[case::f64(0.0f64)]
#[case::f32(0.0f32)]
fn regression_matmul_input_order<T: FloatLikeTensorElement>(#[case] _t: T) {
    // identity @ B == B, not B^T
    let identity = Tensor::<T>::eye(2, 2);
    let b = tensor_of::<T>(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let result = identity.matmul(&b).unwrap().materialize();
    assert_eq!(result.data(), b.data());
}

// Bug: chain (t * 2) - 1 must produce correct values, not corrupt results.
#[rstest]
#[case::f64(0.0f64)]
#[case::f32(0.0f32)]
fn regression_buffer_reuse_chain_correctness<T: FloatLikeTensorElement>(#[case] _t: T) {
    // arange(6) * 2 - 1 = [-1, 1, 3, 5, 7, 9]
    let t: Tensor<T> = arange!(6);
    let result = (t * T::from_f64(2.0) - T::from_f64(1.0)).materialize();
    assert_approx_eq(result.data(), &[-1.0, 1.0, 3.0, 5.0, 7.0, 9.0]);
}

// Bug: is_last_axes_transposed returned true for contiguous [m, 1] tensors
// (stride = [1, 1]), causing BLAS to receive transb=CblasTrans and n=m instead
// of n=1, writing past the end of the one-element output buffer.
#[rstest]
#[case::f64(0.0f64)]
#[case::f32(0.0f32)]
fn regression_matmul_rhs_single_column<T: FloatLikeTensorElement>(#[case] _t: T) {
    let a = tensor_of::<T>(&[1.0, 2.0], &[1, 2]);
    let b = tensor_of::<T>(&[3.0, 4.0], &[2, 1]);
    let c = a.matmul(&b).unwrap().materialize();
    assert_eq!(c.shape(), &[1, 1]);
    // 1*3 + 2*4 = 11
    assert_eq!(c.data(), &[T::from_f64(11.0)]);
}

// Bug: matmul of mismatched 1-D vs 2-D shapes panicked deep in the broadcast
// helper (subtract overflow) instead of returning a clean CannotMatMul error.
// Root cause: the helper compared element counts, not ranks, when picking
// largest/smallest, then subtracted ranks.
#[test]
fn regression_matmul_1d_2d_shape_mismatch() {
    let a = Tensor::from_slice(&[3.0_f64, 4.0], &[2]);
    let b = Tensor::from_slice(&[1.0_f64, 2.0], &[1, 2]);
    let result = a.matmul(&b);
    assert!(matches!(
        result,
        Err(candela::errors::OpError::CannotMatMul(_, _))
    ));
}

// ── 0-D construction is rejected ─────────────────────────────────────────────

#[test]
#[should_panic(expected = "rank >= 1")]
fn tensor_from_slice_empty_shape_panics() {
    let _ = Tensor::from_slice(&[1.0_f64], &[]);
}

#[test]
#[should_panic(expected = "rank >= 1")]
fn tensor_from_scalar_empty_shape_panics() {
    let _ = Tensor::from_scalar(0.0_f32, &[]);
}

#[test]
#[should_panic(expected = "rank >= 1")]
fn layout_from_shape_empty_panics() {
    let _ = candela::Layout::from_shape(&[], 0);
}
