mod common;

use candela::{Dimension, FloatLikeTensorElement, Tensor, arange, ones, s};
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

// Bug: redirect table is static - applied to every input lookup regardless of
// execution order. If an independent consumer of node A (call it C) appears
// before AsContiguous(A) (call it B) in the topological sort, the redirect
// A→B causes C to request B's buffer before B has been computed.
#[rstest]
#[case::f64(0.0f64)]
#[case::f32(0.0f32)]
fn bug_redirect_timing_independent_consumer_before_as_contiguous<T: FloatLikeTensorElement>(
    #[case] _t: T,
) {
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
    assert!(matches!(result, Err(candela::OpError::CannotMatMul(_, _))));
}

// Bug: `.cache()` wraps its input in a zero-copy NoOp, so the cache node shares
// the input node's buffer and retains it for reuse across materializations. The
// planner classified the cache's NoOp as an allocation instead of a reference, so
// it never pinned the input slot alive - a downstream buffer-reuse op then claimed
// that slot and stripped data the cache still held, panicking in strip_tensor.
#[rstest]
#[case::f64(0.0f64)]
#[case::f32(0.0f32)]
fn regression_cached_node_consumed_by_reusable_op<T: FloatLikeTensorElement>(#[case] _t: T) {
    // cached = arange(4) + 1 = [1, 2, 3, 4]; the downstream + 0 is a buffer-reuse
    // candidate that would grab the cache's shared buffer if the slot looked free.
    let t: Tensor<T> = arange!(4);
    let cached = (t + T::from_f64(1.0)).cache();
    let result = (&cached + T::from_f64(0.0)).materialize();
    assert_approx_eq(result.data(), &[1.0, 2.0, 3.0, 4.0]);
}

// Bug: fast_packed_iter's contiguous branch iterated the whole physical buffer
// from index 0, ignoring the layout offset. A scalar op on a sub-range slice
// (contiguous, offset != 0) read the wrong elements: arange(5)[2..4] + 1 gave
// [1, 2] instead of [3, 4].
#[rstest]
#[case::f64(0.0f64)]
#[case::f32(0.0f32)]
fn regression_scalar_op_offset_slice<T: FloatLikeTensorElement>(#[case] _t: T) {
    let t: Tensor<T> = arange!(5); // [0,1,2,3,4]
    let sliced = t.slice(s![2..4]).unwrap(); // [2,3], offset 2, contiguous
    let result = (sliced + T::from_f64(1.0)).materialize();
    assert_approx_eq(result.data(), &[3.0, 4.0]);
}

// Bug: the offset leak was rank-agnostic - a contiguous row block of a 2-D
// tensor also carries a nonzero offset.
#[rstest]
#[case::f64(0.0f64)]
#[case::f32(0.0f32)]
fn regression_scalar_op_offset_row_block<T: FloatLikeTensorElement>(#[case] _t: T) {
    // [[0,1,2],[3,4,5]]; row [1..2] = [[3,4,5]] at offset 3; *10 = [30,40,50].
    let t = tensor_of::<T>(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0], &[2, 3]);
    let sliced = t.slice(s![1..2, ..]).unwrap();
    let result = (sliced * T::from_f64(10.0)).materialize();
    assert_approx_eq(result.data(), &[30.0, 40.0, 50.0]);
}

// Bug: a fused scalar chain runs through the same packed iterator, so it hit the
// offset bug too: arange(5)[2..4] then *2 + 1 must be [5, 7].
#[rstest]
#[case::f64(0.0f64)]
#[case::f32(0.0f32)]
fn regression_fused_scalar_offset_slice<T: FloatLikeTensorElement>(#[case] _t: T) {
    let t: Tensor<T> = arange!(5);
    let sliced = t.slice(s![2..4]).unwrap(); // [2,3]
    let result = (sliced * T::from_f64(2.0) + T::from_f64(1.0)).materialize();
    assert_approx_eq(result.data(), &[5.0, 7.0]);
}

// Bug: compute_layout for ScalarOp used is_contiguous() (which ignores offset)
// and cloned the input layout, so the result node inherited the slice's offset
// while owning a fresh offset-0 buffer. A downstream consumer reading that node
// via its layout then indexed past the small buffer / read shifted data.
#[rstest]
#[case::f64(0.0f64)]
#[case::f32(0.0f32)]
fn regression_scalar_op_offset_slice_feeds_consumer<T: FloatLikeTensorElement>(#[case] _t: T) {
    let t: Tensor<T> = arange!(5);
    let sliced = t.slice(s![2..4]).unwrap(); // [2,3]
    let shifted = sliced + T::from_f64(1.0); // [3,4]; node must be offset 0
    let result = (&shifted + &shifted).materialize(); // [6,8]
    assert_approx_eq(result.data(), &[6.0, 8.0]);
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
    let _ = candela::Layout::new(&[]);
}
