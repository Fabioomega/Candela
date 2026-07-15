mod common;

use candela::skeleton::SkeletonSlot;
use candela::{Dimension, FloatLikeTensorElement, Layout, Tensor, srange};
use common::{assert_approx_eq, assert_approx_eq_by, cast, tensor_of};
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
    let a = tensor_of::<T>(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let b = Tensor::<T>::eye(2, 2);
    let c = a.clone().matmul(&b).unwrap().materialize();
    assert_eq!(c.data(), a.data());
}

#[rstest]
#[case::f64(0.0f64)]
#[case::f32(0.0f32)]
fn matmul_non_square<T: FloatLikeTensorElement>(#[case] _t: T) {
    // [[1,0,0],[0,1,0],[0,0,1]] @ [[1,2],[3,4],[5,6]] = [[1,2],[3,4],[5,6]]
    let a = Tensor::<T>::eye(3, 3);
    let b = tensor_of::<T>(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]);
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
    let a = Tensor::from_scalar(T::from_f64(1.0), &[2, 3, 4]);
    let b = Tensor::from_scalar(T::from_f64(1.0), &[2, 4, 5]);
    let c = a.matmul(&b).unwrap().materialize();
    assert_eq!(c.shape(), &[2, 3, 5]);
}

#[rstest]
#[case::f64(0.0f64)]
#[case::f32(0.0f32)]
fn matmul_batched_values<T: FloatLikeTensorElement>(#[case] _t: T) {
    // All-ones [2,3,4] @ all-ones [2,4,5]: each output element = sum of 4 ones = 4.0
    let a = Tensor::from_scalar(T::from_f64(1.0), &[2, 3, 4]);
    let b = Tensor::from_scalar(T::from_f64(1.0), &[2, 4, 5]);
    let c = a.matmul(&b).unwrap().materialize();
    assert_approx_eq(c.data(), &vec![4.0; 2 * 3 * 5]);
}

#[test]
fn matmul_shape_mismatch() {
    // [3,4] @ [3,4] - inner dims 4 != 3
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
    let a = tensor_of::<T>(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let b = tensor_of::<T>(&[5.0, 6.0, 7.0, 8.0], &[2, 2]);
    let ab = a.clone().matmul(&b.clone()).unwrap().materialize();
    let ba = b.matmul(&a).unwrap().materialize();
    assert_ne!(ab.data(), ba.data());
}

// ── error cases: bias shape mismatches ───────────────────────────────────────

#[test]
#[should_panic]
fn matmul_plus_bias_wrong_shape_panics() {
    // (A@B) shape is [2,2]; bias is [3,3] - incompatible, no broadcast possible
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
    let a = tensor_of::<T>(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let b = tensor_of::<T>(&[5.0, 6.0, 7.0, 8.0], &[2, 2]);
    let c = (a.matmul(&b).unwrap() * T::from_f64(2.0)).materialize();
    assert_eq!(c.shape(), &[2, 2]);
    // 2 * [[19,22],[43,50]] = [[38,44],[86,100]]
    assert_approx_eq(c.data(), &[38.0, 44.0, 86.0, 100.0]);
}

#[rstest]
#[case::f64(0.0f64)]
#[case::f32(0.0f32)]
fn matmul_plus_bias<T: FloatLikeTensorElement>(#[case] _t: T) {
    let a = tensor_of::<T>(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let b = tensor_of::<T>(&[5.0, 6.0, 7.0, 8.0], &[2, 2]);
    let bias = tensor_of::<T>(&[1.0, 0.0, 0.0, 1.0], &[2, 2]); // identity
    let c = (a.matmul(&b).unwrap() + bias).materialize();
    assert_eq!(c.shape(), &[2, 2]);
    // [[19,22],[43,50]] + I = [[20,22],[43,51]]
    assert_approx_eq(c.data(), &[20.0, 22.0, 43.0, 51.0]);
}

#[rstest]
#[case::f64(0.0f64)]
#[case::f32(0.0f32)]
fn matmul_minus_bias<T: FloatLikeTensorElement>(#[case] _t: T) {
    let a = tensor_of::<T>(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let b = tensor_of::<T>(&[5.0, 6.0, 7.0, 8.0], &[2, 2]);
    let bias = tensor_of::<T>(&[1.0, 0.0, 0.0, 1.0], &[2, 2]); // identity
    let c = (a.matmul(&b).unwrap() - bias).materialize();
    assert_eq!(c.shape(), &[2, 2]);
    // [[19,22],[43,50]] - I = [[18,22],[43,49]]
    assert_approx_eq(c.data(), &[18.0, 22.0, 43.0, 49.0]);
}

#[rstest]
#[case::f64(0.0f64)]
#[case::f32(0.0f32)]
fn matmul_scaled_plus_bias<T: FloatLikeTensorElement>(#[case] _t: T) {
    let a = tensor_of::<T>(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let b = tensor_of::<T>(&[5.0, 6.0, 7.0, 8.0], &[2, 2]);
    let bias = tensor_of::<T>(&[1.0, 0.0, 0.0, 1.0], &[2, 2]);
    let c = (a.matmul(&b).unwrap() * T::from_f64(2.0) + bias).materialize();
    assert_eq!(c.shape(), &[2, 2]);
    // 2*[[19,22],[43,50]] + I = [[39,44],[86,101]]
    assert_approx_eq(c.data(), &[39.0, 44.0, 86.0, 101.0]);
}

#[rstest]
#[case::f64(0.0f64)]
#[case::f32(0.0f32)]
fn matmul_scaled_minus_bias<T: FloatLikeTensorElement>(#[case] _t: T) {
    let a = tensor_of::<T>(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let b = tensor_of::<T>(&[5.0, 6.0, 7.0, 8.0], &[2, 2]);
    let bias = tensor_of::<T>(&[1.0, 0.0, 0.0, 1.0], &[2, 2]);
    let c = (a.matmul(&b).unwrap() * T::from_f64(2.0) - bias).materialize();
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
    let a = Tensor::from_scalar(T::from_f64(1.0), &[2, 3, 4]);
    let b = Tensor::from_scalar(T::from_f64(1.0), &[2, 4, 5]);
    let bias = Tensor::from_scalar(T::from_f64(1.0), &[2, 3, 5]);
    let c = (a.matmul(&b).unwrap() + bias).materialize();
    assert_eq!(c.shape(), &[2, 3, 5]);
    assert_approx_eq(c.data(), &vec![5.0; 2 * 3 * 5]);
}

// ── broadcast (stride-0) matrix axes ─────────────────────────────────────────

#[rstest]
#[case::f64(0.0f64)]
#[case::f32(0.0f32)]
fn matmul_broadcast_lhs_rows<T: FloatLikeTensorElement>(#[case] _t: T) {
    // A is [1,3] widened to [2,3] with stride [0,1]: both rows read [1,2,3].
    // [1,2,3] @ [[1,2],[3,4],[5,6]] = [1+6+15, 2+8+18] = [22,28], twice.
    let a = Tensor::from_vec_with_layout(
        cast::<T>(&[1.0, 2.0, 3.0]),
        Layout::new((1, 3)).broadcast((2, 3)).unwrap(),
    );
    let b = tensor_of::<T>(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]);
    let c = a.matmul(&b).unwrap().materialize();
    assert_eq!(c.shape(), &[2, 2]);
    assert_approx_eq(c.data(), &[22.0, 28.0, 22.0, 28.0]);
}

#[rstest]
#[case::f64(0.0f64)]
#[case::f32(0.0f32)]
fn matmul_broadcast_lhs_cols<T: FloatLikeTensorElement>(#[case] _t: T) {
    // A is [2,1] widened to [2,3] with stride [1,0]: A = [[1,1,1],[2,2,2]].
    // Row 0 = column sums of B = [9,12]; row 1 is twice that = [18,24].
    let a = Tensor::from_vec_with_layout(
        cast::<T>(&[1.0, 2.0]),
        Layout::new((2, 1)).broadcast((2, 3)).unwrap(),
    );
    let b = tensor_of::<T>(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]);
    let c = a.matmul(&b).unwrap().materialize();
    assert_approx_eq(c.data(), &[9.0, 12.0, 18.0, 24.0]);
}

#[rstest]
#[case::f64(0.0f64)]
#[case::f32(0.0f32)]
fn matmul_broadcast_rhs_rows<T: FloatLikeTensorElement>(#[case] _t: T) {
    // B is [1,2] widened to [3,2] with stride [0,1]: every row of B reads [1,2].
    // Each output row is [sum(A_row), 2*sum(A_row)] = [6,12] and [15,30].
    let a = tensor_of::<T>(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let b = Tensor::from_vec_with_layout(
        cast::<T>(&[1.0, 2.0]),
        Layout::new((1, 2)).broadcast((3, 2)).unwrap(),
    );
    let c = a.matmul(&b).unwrap().materialize();
    assert_approx_eq(c.data(), &[6.0, 12.0, 15.0, 30.0]);
}

#[rstest]
#[case::f64(0.0f64)]
#[case::f32(0.0f32)]
fn matmul_broadcast_rhs_cols<T: FloatLikeTensorElement>(#[case] _t: T) {
    // B is [3,1] widened to [3,2] with stride [1,0]: B = [[1,1],[2,2],[3,3]],
    // so both output columns are equal. Row 0 = 1+4+9 = 14; row 1 = 4+10+18 = 32.
    let a = tensor_of::<T>(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let b = Tensor::from_vec_with_layout(
        cast::<T>(&[1.0, 2.0, 3.0]),
        Layout::new((3, 1)).broadcast((3, 2)).unwrap(),
    );
    let c = a.matmul(&b).unwrap().materialize();
    assert_approx_eq(c.data(), &[14.0, 14.0, 32.0, 32.0]);
}

#[rstest]
#[case::f64(0.0f64)]
#[case::f32(0.0f32)]
fn matmul_broadcast_lhs_both_axes<T: FloatLikeTensorElement>(#[case] _t: T) {
    // A is [1,1] widened to [2,3] with stride [0,0]: every element reads 2.0.
    // Each output row = 2 * column sums of B = [18,24].
    let a = Tensor::from_vec_with_layout(
        cast::<T>(&[2.0]),
        Layout::new((1, 1)).broadcast((2, 3)).unwrap(),
    );
    let b = tensor_of::<T>(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]);
    let c = a.matmul(&b).unwrap().materialize();
    assert_approx_eq(c.data(), &[18.0, 24.0, 18.0, 24.0]);
}

#[rstest]
#[case::f64(0.0f64)]
#[case::f32(0.0f32)]
fn matmul_noncontiguous_lhs_matches_materialized<T: FloatLikeTensorElement>(#[case] _t: T) {
    // Each layout below addresses the same values as its materialized twin, so
    // handing the kernel the strided view and handing it a dense copy must agree
    // element for element. `as_contiguous` forces the copy that a backend
    // accepting arbitrary strides would otherwise skip, which makes this a
    // differential check on the strided path rather than on any one expected
    // value. The buffer is deliberately larger than the narrow layouts require.
    let b = tensor_of::<T>(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]);

    let layouts = [
        Layout::new((2, 3)),                            // dense
        Layout::new((3, 2)).transpose(),                // transposed view
        Layout::new((1, 3)).broadcast((2, 3)).unwrap(), // rows repeated
        Layout::new((2, 1)).broadcast((2, 3)).unwrap(), // cols repeated
        Layout::new((1, 1)).broadcast((2, 3)).unwrap(), // single value
    ];

    for layout in layouts {
        let a = Tensor::from_vec_with_layout(
            cast::<T>(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
            layout.clone(),
        );
        let strided = a.matmul(&b).unwrap().materialize();
        let copied = a.as_contiguous().matmul(&b).unwrap().materialize();
        assert_eq!(
            strided.shape(),
            copied.shape(),
            "shape disagrees for {layout:?}"
        );
        assert_approx_eq_by(strided.data(), copied.data(), 1e-6);
    }
}

#[test]
fn matmul_broadcast_allocations() {
    // A backend taking arbitrary strides reaches the kernel with the stride-0 view
    // intact. An inserted `AsContiguous` would compute identical values, so it is
    // invisible to every other test in this group; it surfaces here as the extra
    // buffer it would have to allocate for the widened [2,3] operand.
    let a: SkeletonSlot<f64> = SkeletonSlot::new(Layout::new((1, 3)).broadcast((2, 3)).unwrap());
    let b: SkeletonSlot<f64> = SkeletonSlot::new(Layout::new((3, 2)));
    let skeleton = a.matmul(&b).unwrap().into_skeleton(&[a, b]).unwrap();

    // The [2,2] f64 output, and nothing else.
    assert_eq!(skeleton.memory_report().allocated_buffers_size, vec![32]);
}

// ── batch-dimension broadcasting ────────────────────────

#[rstest]
#[case::f64(0.0f64)]
#[case::f32(0.0f32)]
fn matmul_broadcast_batch_lhs_one<T: FloatLikeTensorElement>(#[case] _t: T) {
    // [1,3,4] @ [2,4,5] should broadcast lhs batch dim: result is [2,3,5]
    // Each output slice is the single lhs matrix multiplied by the corresponding rhs batch.
    let a = Tensor::from_scalar(T::from_f64(1.0), &[1, 3, 4]);
    let b = Tensor::from_scalar(T::from_f64(1.0), &[2, 4, 5]);
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
    let a = Tensor::from_scalar(T::from_f64(1.0), &[2, 3, 4]);
    let b = Tensor::from_scalar(T::from_f64(1.0), &[1, 4, 5]);
    let c = a.matmul(&b).unwrap().materialize();
    assert_eq!(c.shape(), &[2, 3, 5]);
    assert_approx_eq(c.data(), &vec![4.0; 2 * 3 * 5]);
}

#[rstest]
#[case::f64(0.0f64)]
#[case::f32(0.0f32)]
fn matmul_broadcast_batch_axis<T: FloatLikeTensorElement>(#[case] _t: T) {
    // A is [1,2,3] widened to [2,2,3] with stride [0,3,1]: one matrix re-read per
    // batch, against a distinct B per batch.
    // A     = [[1,2,3],[4,5,6]]
    // B[0]  = [[1,2],[3,4],[5,6]]   -> [[22,28],[49,64]]
    // B[1]  = [[7,8],[9,10],[11,12]] -> [[58,64],[139,154]]
    let a = Tensor::from_vec_with_layout(
        cast::<T>(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
        Layout::new((1, 2, 3)).broadcast((2, 2, 3)).unwrap(),
    );
    let b = tensor_of::<T>(
        &[
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ],
        &[2, 3, 2],
    );
    let c = a.matmul(&b).unwrap().materialize();
    assert_eq!(c.shape(), &[2, 2, 2]);
    assert_approx_eq(
        c.data(),
        &[22.0, 28.0, 49.0, 64.0, 58.0, 64.0, 139.0, 154.0],
    );
}

// ── NumPy-style 1-D promotion ────────────────────────────────────────────────

// [K] @ [K, N]: lhs gets a prepended 1 for the matmul, stripped on output.
#[test]
fn matmul_1d_lhs() {
    let a = Tensor::from_slice(&[1.0_f64, 2.0, 3.0], &[3]);
    let b = Tensor::from_slice(&[1.0_f64, 0.0, 0.0, 1.0, 1.0, 1.0], &[3, 2]);
    let c = a.matmul(&b).unwrap().materialize();
    assert_eq!(c.shape(), &[2]);
    // [1,2,3] @ [[1,0],[0,1],[1,1]] = [1+0+3, 0+2+3] = [4, 5]
    assert_eq!(c.data(), &[4.0, 5.0]);
}

// [M, K] @ [K]: rhs gets an appended 1 for the matmul, stripped on output.
#[test]
fn matmul_1d_rhs() {
    let a = Tensor::from_slice(&[1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let b = Tensor::from_slice(&[1.0_f64, 0.0, 1.0], &[3]);
    let c = a.matmul(&b).unwrap().materialize();
    assert_eq!(c.shape(), &[2]);
    // [[1,2,3],[4,5,6]] @ [1,0,1] = [1+0+3, 4+0+6] = [4, 10]
    assert_eq!(c.data(), &[4.0, 10.0]);
}

// [K] @ [K]: dot product. NumPy returns 0-D; we deviate to [1] to keep the
// rank >= 1 invariant.
#[test]
fn matmul_1d_dot() {
    let a = Tensor::from_slice(&[1.0_f64, 2.0, 3.0], &[3]);
    let b = Tensor::from_slice(&[4.0_f64, 5.0, 6.0], &[3]);
    let c = a.matmul(&b).unwrap().materialize();
    assert_eq!(c.shape(), &[1]);
    // 1*4 + 2*5 + 3*6 = 32
    assert_eq!(c.data(), &[32.0]);
}
