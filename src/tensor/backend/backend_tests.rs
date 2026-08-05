//! One backend conformance suite, run against every `(Backend, Dtype)` pair.
//!
//! Each test is a generic function over `B: Backend` and `T: ComputeFor<B>`, so
//! the assertions exist once. The `backend_suite!` invocation at the bottom of
//! the file expands the list of test names into one `#[test]` wrapper per pair,
//! which is what `cargo test` actually collects. Adding a test means writing the
//! generic function and adding its name to that list.
//!
//! Entry into the backend goes through `B::compute` / `B::compute_inplace`
//! rather than a backend-private `compute_op`, which is what makes the body
//! generic; it also puts the per-`(T, B)` `ComputeFor` dispatch under test.

use crate::tensor::backend::cpu_pure::CpuPure;
use crate::tensor::backend::{Backend, ComputeFor, Dtype};
use crate::tensor::mem_formats::layout::Layout;
use crate::tensor::ops::def_op::{OpKind, OpKindScalar, Sign};
use crate::tensor::storage::TensorData;
use crate::tensor::traits::{Dimension, FromIndex};

#[cfg(feature = "mkl")]
use crate::tensor::backend::cpu_mkl::CpuMkl;

// ── helpers ───────────────────────────────────────────────────────────────────

/// Lets test bodies write plain `f64` literals regardless of the dtype under
/// test. Kept test-local so the real trait hierarchy stays free of it.
///
/// `Display` is a supertrait because `TensorData<T>: PartialEq` requires it, and
/// several bodies want `assert_eq!` on whole tensors. `PartialOrd` is not on
/// `Dtype`, but comparison assertions need it.
trait TestScalar: Dtype + FromIndex + PartialOrd + std::fmt::Display {
    fn v(x: f64) -> Self;
}

impl TestScalar for f32 {
    fn v(x: f64) -> Self {
        x as f32
    }
}

impl TestScalar for f64 {
    fn v(x: f64) -> Self {
        x
    }
}

fn vals<T: TestScalar>(xs: &[f64]) -> Vec<T> {
    xs.iter().copied().map(T::v).collect()
}

fn td<T: TestScalar>(data: &[f64], shape: &[usize]) -> TensorData<T> {
    TensorData::from_vec(vals(data), shape, 0)
}

fn arange<T: TestScalar>(n: usize, shape: &[usize]) -> TensorData<T> {
    TensorData::from_iter((0..n).map(T::from_index), shape)
}

/// Every body carries this bound; naming it once keeps the signatures readable.
trait Case<B: Backend>: TestScalar + ComputeFor<B> {}
impl<B: Backend, T: TestScalar + ComputeFor<B>> Case<B> for T {}

// ── compute ───────────────────────────────────────────────────────────────────

fn as_contiguous_non_contiguous_input<B: Backend, T: Case<B>>() {
    let t = TensorData::from_scalar(T::ONE, &[7, 7]).slice(s![.., 1..2]);
    let buffer = vec![T::ONE; 7];

    let output = B::compute(
        &OpKind::AsContiguous,
        buffer,
        &Layout::new((7, 1)),
        std::slice::from_ref(&t),
    );
    assert_eq!(output, t);
    assert!(output.is_contiguous());
}

fn scalar_op_axby_contiguous<B: Backend, T: Case<B>>() {
    let input = td(&[1.0, 2.0, 3.0], &[3]);
    let output = B::compute(
        &OpKind::ScalarOp(OpKindScalar::AxBy(T::v(2.0), T::v(1.0))),
        vec![T::ZERO; 3],
        &Layout::new(3),
        &[input],
    );
    assert_eq!(output.data(), &vals(&[3.0, 5.0, 7.0]));
}

fn scalar_op_axby_non_contiguous<B: Backend, T: Case<B>>() {
    // Column slice of [3,4] → shape [3,1], stride [4,1], non-contiguous.
    let base = TensorData::from_scalar(T::ONE, &[3, 4]);
    let input = base.slice(s![.., 0..1]);
    let output = B::compute(
        &OpKind::ScalarOp(OpKindScalar::AxBy(T::v(2.0), T::v(3.0))),
        vec![T::ZERO; 3],
        &Layout::new((3, 1)),
        &[input],
    );
    assert_eq!(output.data(), &vals(&[5.0, 5.0, 5.0])); // 2*1 + 3 = 5
}

fn scalar_op_exp<B: Backend, T: Case<B>>() {
    let input = td(&[0.0], &[1]);
    let output = B::compute(
        &OpKind::ScalarOp(OpKindScalar::Exp),
        vec![T::ZERO; 1],
        &Layout::new(1),
        &[input],
    );
    assert_eq!(output.data(), &vals(&[1.0]));
}

fn scalar_op_ln<B: Backend, T: Case<B>>() {
    let input = td(&[1.0], &[1]);
    let output = B::compute(
        &OpKind::ScalarOp(OpKindScalar::Ln),
        vec![T::ZERO; 1],
        &Layout::new(1),
        &[input],
    );
    assert_eq!(output.data(), &vals(&[0.0]));
}

fn scalar_op_log2<B: Backend, T: Case<B>>() {
    let input = td(&[1.0, 2.0], &[2]);
    let output = B::compute(
        &OpKind::ScalarOp(OpKindScalar::Log2),
        vec![T::ZERO; 2],
        &Layout::new(2),
        &[input],
    );
    assert_eq!(output.data(), &vals(&[0.0, 1.0]));
}

fn scalar_op_inv<B: Backend, T: Case<B>>() {
    let input = td(&[1.0, 2.0, 4.0], &[3]);
    let output = B::compute(
        &OpKind::ScalarOp(OpKindScalar::Recip),
        vec![T::ZERO; 3],
        &Layout::new(3),
        &[input],
    );
    assert_eq!(output.data(), &vals(&[1.0, 0.5, 0.25]));
}

fn scalar_op_inv_non_contiguous<B: Backend, T: Case<B>>() {
    // Column slice of [3,4] → shape [3,1], stride [4,1], non-contiguous, so this
    // takes the per-element path rather than the contiguous-chunk one.
    let base = TensorData::from_scalar(T::v(4.0), &[3, 4]);
    let input = base.slice(s![.., 0..1]);
    let output = B::compute(
        &OpKind::ScalarOp(OpKindScalar::Recip),
        vec![T::ZERO; 3],
        &Layout::new((3, 1)),
        &[input],
    );
    assert_eq!(output.data(), &vals(&[0.25, 0.25, 0.25]));
}

fn fused_scalar<B: Backend, T: Case<B>>() {
    // AxBy(2, 1): 2*3+1=7, then AxBy(3, 0): 3*7+0=21
    let input = td(&[3.0], &[1]);
    let ops = Box::new([
        OpKindScalar::AxBy(T::v(2.0), T::v(1.0)),
        OpKindScalar::AxBy(T::v(3.0), T::v(0.0)),
    ]);
    let output = B::compute(
        &OpKind::FusedScalar(ops),
        vec![T::ZERO; 1],
        &Layout::new(1),
        &[input],
    );
    assert_eq!(output.data(), &vals(&[21.0]));
}

fn add_contiguous<B: Backend, T: Case<B>>() {
    let lhs = td(&[1.0, 2.0, 3.0], &[3]);
    let rhs = td(&[4.0, 5.0, 6.0], &[3]);
    let output = B::compute(&OpKind::Add, vec![T::ZERO; 3], &Layout::new(3), &[lhs, rhs]);
    assert_eq!(output.data(), &vals(&[5.0, 7.0, 9.0]));
}

fn add_lhs_non_contiguous<B: Backend, T: Case<B>>() {
    // lhs: [3,4] sliced to [3,2] - non-contiguous, all 1.0
    // rhs: contiguous [3,2], all 2.0
    let base = TensorData::from_scalar(T::ONE, &[3, 4]);
    let lhs = base.slice(s![.., 0..2]);
    let rhs = TensorData::from_scalar(T::v(2.0), &[3, 2]);
    let output = B::compute(
        &OpKind::Add,
        vec![T::ZERO; 6],
        &Layout::new((3, 2)),
        &[lhs, rhs],
    );
    assert_eq!(output.data(), &vec![T::v(3.0); 6]);
}

fn add_rhs_non_contiguous<B: Backend, T: Case<B>>() {
    // lhs: contiguous [3,2], all 1.0
    // rhs: [3,4] sliced to [3,2] - non-contiguous, all 2.0
    let lhs = TensorData::from_scalar(T::ONE, &[3, 2]);
    let base = TensorData::from_scalar(T::v(2.0), &[3, 4]);
    let rhs = base.slice(s![.., 0..2]);
    let output = B::compute(
        &OpKind::Add,
        vec![T::ZERO; 6],
        &Layout::new((3, 2)),
        &[lhs, rhs],
    );
    assert_eq!(output.data(), &vec![T::v(3.0); 6]);
}

fn add_both_non_contiguous<B: Backend, T: Case<B>>() {
    let base_a = TensorData::from_scalar(T::ONE, &[3, 4]);
    let lhs = base_a.slice(s![.., 0..2]);
    let base_b = TensorData::from_scalar(T::v(2.0), &[3, 4]);
    let rhs = base_b.slice(s![.., 0..2]);
    let output = B::compute(
        &OpKind::Add,
        vec![T::ZERO; 6],
        &Layout::new((3, 2)),
        &[lhs, rhs],
    );
    assert_eq!(output.data(), &vec![T::v(3.0); 6]);
}

fn sub_contiguous<B: Backend, T: Case<B>>() {
    let lhs = td(&[5.0, 6.0], &[2]);
    let rhs = td(&[1.0, 2.0], &[2]);
    let output = B::compute(&OpKind::Sub, vec![T::ZERO; 2], &Layout::new(2), &[lhs, rhs]);
    assert_eq!(output.data(), &vals(&[4.0, 4.0]));
}

fn mul_contiguous<B: Backend, T: Case<B>>() {
    let lhs = td(&[2.0, 3.0], &[2]);
    let rhs = td(&[4.0, 5.0], &[2]);
    let output = B::compute(&OpKind::Mul, vec![T::ZERO; 2], &Layout::new(2), &[lhs, rhs]);
    assert_eq!(output.data(), &vals(&[8.0, 15.0]));
}

fn div_contiguous<B: Backend, T: Case<B>>() {
    let lhs = td(&[6.0, 10.0], &[2]);
    let rhs = td(&[2.0, 5.0], &[2]);
    let output = B::compute(&OpKind::Div, vec![T::ZERO; 2], &Layout::new(2), &[lhs, rhs]);
    assert_eq!(output.data(), &vals(&[3.0, 2.0]));
}

// ── compute_inplace ───────────────────────────────────────────────────────────

fn scalar_axby_inplace<B: Backend, T: Case<B>>() {
    let input = td(&[1.0, 2.0, 3.0], &[3]);
    let layout = Layout::new(3);
    let output = B::compute_inplace(
        &OpKind::ScalarOp(OpKindScalar::AxBy(T::v(2.0), T::v(1.0))),
        &layout,
        vec![input],
        0,
    );
    assert_eq!(output.data(), &vals(&[3.0, 5.0, 7.0]));
}

fn scalar_exp_inplace<B: Backend, T: Case<B>>() {
    let input: TensorData<T> = td(&[0.0], &[1]);
    let layout = Layout::new(1);
    let output = B::compute_inplace(
        &OpKind::ScalarOp(OpKindScalar::Exp),
        &layout,
        vec![input],
        0,
    );
    assert_eq!(output.data(), &vals(&[1.0]));
}

fn scalar_ln_inplace<B: Backend, T: Case<B>>() {
    let input: TensorData<T> = td(&[1.0], &[1]);
    let layout = Layout::new(1);
    let output = B::compute_inplace(&OpKind::ScalarOp(OpKindScalar::Ln), &layout, vec![input], 0);
    assert_eq!(output.data(), &vals(&[0.0]));
}

fn scalar_log2_inplace<B: Backend, T: Case<B>>() {
    let input: TensorData<T> = td(&[1.0, 2.0], &[2]);
    let layout = Layout::new(2);
    let output = B::compute_inplace(
        &OpKind::ScalarOp(OpKindScalar::Log2),
        &layout,
        vec![input],
        0,
    );
    assert_eq!(output.data(), &vals(&[0.0, 1.0]));
}

fn scalar_inv_inplace<B: Backend, T: Case<B>>() {
    let input: TensorData<T> = td(&[1.0, 2.0, 4.0], &[3]);
    let layout = Layout::new(3);
    let output = B::compute_inplace(
        &OpKind::ScalarOp(OpKindScalar::Recip),
        &layout,
        vec![input],
        0,
    );
    assert_eq!(output.data(), &vals(&[1.0, 0.5, 0.25]));
}

fn fused_scalar_inplace<B: Backend, T: Case<B>>() {
    // AxBy(2, 1): 2*3+1=7, then AxBy(3, 0): 3*7+0=21
    let input: TensorData<T> = td(&[3.0], &[1]);
    let layout = Layout::new(1);
    let ops = Box::new([
        OpKindScalar::AxBy(T::v(2.0), T::v(1.0)),
        OpKindScalar::AxBy(T::v(3.0), T::v(0.0)),
    ]);
    let output = B::compute_inplace(&OpKind::FusedScalar(ops), &layout, vec![input], 0);
    assert_eq!(output.data(), &vals(&[21.0]));
}

fn add_inplace_reuse_lhs<B: Backend, T: Case<B>>() {
    let lhs: TensorData<T> = td(&[1.0, 2.0, 3.0], &[3]);
    let rhs = td(&[4.0, 5.0, 6.0], &[3]);
    let layout = Layout::new(3);
    let output = B::compute_inplace(&OpKind::Add, &layout, vec![lhs, rhs], 0);
    assert_eq!(output.data(), &vals(&[5.0, 7.0, 9.0]));
}

fn add_inplace_reuse_rhs<B: Backend, T: Case<B>>() {
    let lhs: TensorData<T> = td(&[1.0, 2.0, 3.0], &[3]);
    let rhs = td(&[4.0, 5.0, 6.0], &[3]);
    let layout = Layout::new(3);
    let output = B::compute_inplace(&OpKind::Add, &layout, vec![lhs, rhs], 1);
    assert_eq!(output.data(), &vals(&[5.0, 7.0, 9.0]));
}

fn sub_inplace<B: Backend, T: Case<B>>() {
    let lhs: TensorData<T> = td(&[5.0, 6.0], &[2]);
    let rhs = td(&[1.0, 2.0], &[2]);
    let layout = Layout::new(2);
    let output = B::compute_inplace(&OpKind::Sub, &layout, vec![lhs, rhs], 0);
    assert_eq!(output.data(), &vals(&[4.0, 4.0]));
}

fn mul_inplace<B: Backend, T: Case<B>>() {
    let lhs: TensorData<T> = td(&[2.0, 3.0], &[2]);
    let rhs = td(&[4.0, 5.0], &[2]);
    let layout = Layout::new(2);
    let output = B::compute_inplace(&OpKind::Mul, &layout, vec![lhs, rhs], 0);
    assert_eq!(output.data(), &vals(&[8.0, 15.0]));
}

fn div_inplace<B: Backend, T: Case<B>>() {
    let lhs: TensorData<T> = td(&[6.0, 10.0], &[2]);
    let rhs = td(&[2.0, 5.0], &[2]);
    let layout = Layout::new(2);
    let output = B::compute_inplace(&OpKind::Div, &layout, vec![lhs, rhs], 0);
    assert_eq!(output.data(), &vals(&[3.0, 2.0]));
}

fn slice_inplace<B: Backend, T: Case<B>>() {
    // [[0,1,2],[3,4,5],[6,7,8]]; take columns 1..3 → logical [[1,2],[4,5],[7,8]]
    let input: TensorData<T> = arange(9, &[3, 3]);
    let output = B::compute_inplace(
        &OpKind::Slice,
        &Layout::new((3, 3)).slice(s![.., 1..3]).unwrap(),
        vec![input],
        0,
    );
    assert_eq!(output.shape(), &[3, 2]);
    assert_eq!(output, td(&[1.0, 2.0, 4.0, 5.0, 7.0, 8.0], &[3, 2]));
}

fn view_inplace<B: Backend, T: Case<B>>() {
    let input: TensorData<T> = arange(6, &[6]);
    let new_layout = input.layout().view((2, 3)).unwrap();
    let output_layout = new_layout.clone();
    let output = B::compute_inplace(&OpKind::View, &output_layout, vec![input], 0);
    assert_eq!(output.shape(), &[2, 3]);
    assert_eq!(output.data(), &vals(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]));
}

fn transpose_inplace<B: Backend, T: Case<B>>() {
    // [[0,1,2],[3,4,5]] → shape [3,2], row-major iteration [0,3,1,4,2,5]
    let input: TensorData<T> = arange(6, &[2, 3]);
    let output = B::compute_inplace(
        &OpKind::Transpose,
        &Layout::new((2, 3)).transpose(),
        vec![input],
        0,
    );
    assert_eq!(output.shape(), &[3, 2]);
    assert_eq!(output, td(&[0.0, 3.0, 1.0, 4.0, 2.0, 5.0], &[3, 2]));
}

fn transpose_axes_inplace<B: Backend, T: Case<B>>() {
    let input: TensorData<T> = arange(6, &[2, 3]);
    let new_layout = input.layout().transpose_axes((1, 0)).unwrap();
    let output_layout = new_layout.clone();
    let output = B::compute_inplace(&OpKind::TransposeAxes, &output_layout, vec![input], 0);
    assert_eq!(output.shape(), &[3, 2]);
    assert_eq!(output, td(&[0.0, 3.0, 1.0, 4.0, 2.0, 5.0], &[3, 2]));
}

fn no_op_inplace<B: Backend, T: Case<B>>() {
    let input: TensorData<T> = td(&[1.0, 2.0, 3.0], &[3]);
    let layout = Layout::new(3);
    let output = B::compute_inplace(&OpKind::NoOp, &layout, vec![input], 0);
    assert_eq!(output.data(), &vals(&[1.0, 2.0, 3.0]));
}

// ── matmul ────────────────────────────────────────────────────────────────────

fn matmul_identity_2x2<B: Backend, T: Case<B>>() {
    // A @ I = A
    let a = td(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let eye = td(&[1.0, 0.0, 0.0, 1.0], &[2, 2]);
    let output = B::compute(
        &OpKind::MatMul(T::ONE),
        vec![T::ZERO; 4],
        &Layout::new((2, 2)),
        &[a.clone(), eye],
    );
    assert_eq!(output.data(), a.data());
}

fn matmul_rectangular<B: Backend, T: Case<B>>() {
    // [2,3] @ [3,2] = [2,2]
    // A = [[1,2,3],[4,5,6]], B = [[7,8],[9,10],[11,12]]
    // C[0,0] = 1*7+2*9+3*11 = 58,  C[0,1] = 1*8+2*10+3*12 = 64
    // C[1,0] = 4*7+5*9+6*11 = 139, C[1,1] = 4*8+5*10+6*12 = 154
    let a = td(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let b = td(&[7.0, 8.0, 9.0, 10.0, 11.0, 12.0], &[3, 2]);
    let output = B::compute(
        &OpKind::MatMul(T::ONE),
        vec![T::ZERO; 4],
        &Layout::new((2, 2)),
        &[a, b],
    );
    assert_eq!(output.data(), &vals(&[58.0, 64.0, 139.0, 154.0]));
}

fn matmul_batched<B: Backend, T: Case<B>>() {
    // [2,2,2] @ [2,2,2] = [2,2,2]
    // Both batches of A are all-ones; B batch 0 is I, B batch 1 is 2*I.
    let a = TensorData::from_scalar(T::ONE, &[2, 2, 2]);
    let b = td(&[1.0, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 2.0], &[2, 2, 2]);
    let output = B::compute(
        &OpKind::MatMul(T::ONE),
        vec![T::ZERO; 8],
        &Layout::new((2, 2, 2)),
        &[a, b],
    );
    // batch 0: [[1,1],[1,1]] @ I = [[1,1],[1,1]]
    // batch 1: [[1,1],[1,1]] @ 2I = [[2,2],[2,2]]
    assert_eq!(
        output.data(),
        &vals(&[1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0])
    );
}

// ── matmul_sum ────────────────────────────────────────────────────────────────
// MatMulSum(alpha, beta, sign) computes: alpha*(A@B) +/- beta*C
// All tests use A = identity [[1,0],[0,1]], B = [[2,3],[4,5]], so A@B = [[2,3],[4,5]].

fn matmul_sum_plus<B: Backend, T: Case<B>>() {
    let a = td(&[1.0, 0.0, 0.0, 1.0], &[2, 2]);
    let b = td(&[2.0, 3.0, 4.0, 5.0], &[2, 2]);
    let c = td(&[1.0, 1.0, 1.0, 1.0], &[2, 2]);
    let output = B::compute(
        &OpKind::MatMulSum(T::ONE, T::ONE, Sign::Plus),
        vec![T::ZERO; 4],
        &Layout::new((2, 2)),
        &[a, b, c],
    );
    // 1*(A@B) + 1*C = [[2,3],[4,5]] + [[1,1],[1,1]] = [[3,4],[5,6]]
    assert_eq!(output.data(), &vals(&[3.0, 4.0, 5.0, 6.0]));
}

fn matmul_sum_minus<B: Backend, T: Case<B>>() {
    let a = td(&[1.0, 0.0, 0.0, 1.0], &[2, 2]);
    let b = td(&[2.0, 3.0, 4.0, 5.0], &[2, 2]);
    let c = td(&[1.0, 1.0, 1.0, 1.0], &[2, 2]);
    let output = B::compute(
        &OpKind::MatMulSum(T::ONE, T::ONE, Sign::Minus),
        vec![T::ZERO; 4],
        &Layout::new((2, 2)),
        &[a, b, c],
    );
    // 1*(A@B) - 1*C = [[2,3],[4,5]] - [[1,1],[1,1]] = [[1,2],[3,4]]
    assert_eq!(output.data(), &vals(&[1.0, 2.0, 3.0, 4.0]));
}

fn matmul_sum_scaled_alpha<B: Backend, T: Case<B>>() {
    let a = td(&[1.0, 0.0, 0.0, 1.0], &[2, 2]);
    let b = td(&[2.0, 3.0, 4.0, 5.0], &[2, 2]);
    let c = td(&[1.0, 1.0, 1.0, 1.0], &[2, 2]);
    let output = B::compute(
        &OpKind::MatMulSum(T::v(2.0), T::ONE, Sign::Plus),
        vec![T::ZERO; 4],
        &Layout::new((2, 2)),
        &[a, b, c],
    );
    // 2*(A@B) + 1*C = 2*[[2,3],[4,5]] + [[1,1],[1,1]] = [[5,7],[9,11]]
    assert_eq!(output.data(), &vals(&[5.0, 7.0, 9.0, 11.0]));
}

fn matmul_sum_scaled_beta<B: Backend, T: Case<B>>() {
    let a = td(&[1.0, 0.0, 0.0, 1.0], &[2, 2]);
    let b = td(&[2.0, 3.0, 4.0, 5.0], &[2, 2]);
    let c = td(&[1.0, 1.0, 1.0, 1.0], &[2, 2]);
    let output = B::compute(
        &OpKind::MatMulSum(T::ONE, T::v(2.0), Sign::Plus),
        vec![T::ZERO; 4],
        &Layout::new((2, 2)),
        &[a, b, c],
    );
    // 1*(A@B) + 2*C = [[2,3],[4,5]] + 2*[[1,1],[1,1]] = [[4,5],[6,7]]
    assert_eq!(output.data(), &vals(&[4.0, 5.0, 6.0, 7.0]));
}

// ── broadcast ─────────────────────────────────────────────────────────────────

fn broadcast_inplace_row_to_matrix<B: Backend, T: Case<B>>() {
    let input: TensorData<T> = td(&[7.0, 8.0, 9.0], &[1, 3]);
    let new_layout = input.layout().broadcast((2, 3)).unwrap();
    let output = B::compute_inplace(&OpKind::Broadcast, &new_layout, vec![input], 0);
    assert_eq!(output.shape(), &[2, 3]);
    let logical: Vec<T> = output.iter().copied().collect();
    assert_eq!(logical, vals(&[7.0, 8.0, 9.0, 7.0, 8.0, 9.0]));
}

// ── Sum / SumAxis ─────────────────────────────────────────────────────────────

fn sum_1d<B: Backend, T: Case<B>>() {
    // [0,1,2,3,4] → 10
    let input = arange(5, &[5]);
    let output = B::compute(&OpKind::Sum, vec![T::ZERO; 1], &Layout::new(1), &[input]);
    assert_eq!(output.data(), &vals(&[10.0]));
}

fn sum_non_contiguous<B: Backend, T: Case<B>>() {
    // Column 0 of [[0,1,2],[3,4,5]] is [0,3] → sum = 3
    let base: TensorData<T> = arange(6, &[2, 3]);
    let input = base.slice(s![.., 0..1]);
    let output = B::compute(&OpKind::Sum, vec![T::ZERO; 1], &Layout::new(1), &[input]);
    assert_eq!(output.data(), &vals(&[3.0]));
}

fn sum_axis_0_2d<B: Backend, T: Case<B>>() {
    // [[0,1],[2,3],[4,5]] sum axis 0 → [6, 9]
    let input = arange(6, &[3, 2]);
    let output = B::compute(
        &OpKind::SumAxis(0, false),
        vec![T::ZERO; 2],
        &Layout::new(2),
        &[input],
    );
    assert_eq!(output.data(), &vals(&[6.0, 9.0]));
}

fn sum_axis_1_2d<B: Backend, T: Case<B>>() {
    // [[0,1],[2,3],[4,5]] sum axis 1 → [1, 5, 9]
    let input = arange(6, &[3, 2]);
    let output = B::compute(
        &OpKind::SumAxis(1, false),
        vec![T::ZERO; 3],
        &Layout::new(3),
        &[input],
    );
    assert_eq!(output.data(), &vals(&[1.0, 5.0, 9.0]));
}

fn sum_axis_negative<B: Backend, T: Case<B>>() {
    // axis=-1 on [3,2] resolves to axis 1: same result as sum_axis_1_2d
    let input = arange(6, &[3, 2]);
    let output = B::compute(
        &OpKind::SumAxis(-1, false),
        vec![T::ZERO; 3],
        &Layout::new(3),
        &[input],
    );
    assert_eq!(output.data(), &vals(&[1.0, 5.0, 9.0]));
}

fn sum_axis_0_3d<B: Backend, T: Case<B>>() {
    // [2,3,4] sum axis 0 → [3,4]; result[i] = i + (i+12)
    let input = arange(24, &[2, 3, 4]);
    let output = B::compute(
        &OpKind::SumAxis(0, false),
        vec![T::ZERO; 12],
        &Layout::new((3, 4)),
        &[input],
    );
    let expected: Vec<T> = (0..12)
        .map(|i| T::from_index(i) + T::from_index(i + 12))
        .collect();
    assert_eq!(output.data(), &expected);
}

fn sum_axis_middle_3d<B: Backend, T: Case<B>>() {
    // [2,3,1] sum axis 1 → [2,1]
    // data: [0,1,2, 3,4,5]; batch 0 sum = 0+1+2 = 3, batch 1 sum = 3+4+5 = 12
    let input = arange(6, &[2, 3, 1]);
    let output = B::compute(
        &OpKind::SumAxis(1, false),
        vec![T::ZERO; 2],
        &Layout::new((2, 1)),
        &[input],
    );
    assert_eq!(output.data(), &vals(&[3.0, 12.0]));
}

// ── Max / MaxAxis ─────────────────────────────────────────────────────────────

fn max_1d<B: Backend, T: Case<B>>() {
    // max of [0,1,2,3,4] = 4
    let input = arange(5, &[5]);
    let output = B::compute(&OpKind::Max, vec![T::ZERO; 1], &Layout::new(1), &[input]);
    assert_eq!(output.data(), &vals(&[4.0]));
}

fn max_non_contiguous<B: Backend, T: Case<B>>() {
    // Column 0 of [[0,1,2],[3,4,5]] is [0,3] → max = 3
    let base: TensorData<T> = arange(6, &[2, 3]);
    let input = base.slice(s![.., 0..1]);
    let output = B::compute(&OpKind::Max, vec![T::ZERO; 1], &Layout::new(1), &[input]);
    assert_eq!(output.data(), &vals(&[3.0]));
}

fn max_axis_0_2d<B: Backend, T: Case<B>>() {
    // [[0,1],[2,3],[4,5]] max axis 0 → [4, 5]
    let input = arange(6, &[3, 2]);
    let output = B::compute(
        &OpKind::MaxAxis(0, false),
        vec![T::ZERO; 2],
        &Layout::new(2),
        &[input],
    );
    assert_eq!(output.data(), &vals(&[4.0, 5.0]));
}

fn max_axis_1_2d<B: Backend, T: Case<B>>() {
    // [[0,1],[2,3],[4,5]] max axis 1 → [1, 3, 5]
    let input = arange(6, &[3, 2]);
    let output = B::compute(
        &OpKind::MaxAxis(1, false),
        vec![T::ZERO; 3],
        &Layout::new(3),
        &[input],
    );
    assert_eq!(output.data(), &vals(&[1.0, 3.0, 5.0]));
}

fn max_axis_negative<B: Backend, T: Case<B>>() {
    // axis=-1 on [3,2] resolves to axis 1: same result as max_axis_1_2d
    let input = arange(6, &[3, 2]);
    let output = B::compute(
        &OpKind::MaxAxis(-1, false),
        vec![T::ZERO; 3],
        &Layout::new(3),
        &[input],
    );
    assert_eq!(output.data(), &vals(&[1.0, 3.0, 5.0]));
}

// ── Mean / MeanAxis ───────────────────────────────────────────────────────────

fn mean_1d<B: Backend, T: Case<B>>() {
    // mean of [0,1,2,3,4] = 2.0
    let input = arange(5, &[5]);
    let output = B::compute(&OpKind::Mean, vec![T::ZERO; 1], &Layout::new(1), &[input]);
    assert_eq!(output.data(), &vals(&[2.0]));
}

fn mean_non_contiguous<B: Backend, T: Case<B>>() {
    // Column 0 of [[0,1,2],[3,4,5]] is [0,3] → mean = 1.5
    let base: TensorData<T> = arange(6, &[2, 3]);
    let input = base.slice(s![.., 0..1]);
    let output = B::compute(&OpKind::Mean, vec![T::ZERO; 1], &Layout::new(1), &[input]);
    assert_eq!(output.data(), &vals(&[1.5]));
}

fn mean_axis_0_2d<B: Backend, T: Case<B>>() {
    // [[0,1],[2,3],[4,5]] mean axis 0 → [2.0, 3.0]
    let input = arange(6, &[3, 2]);
    let output = B::compute(
        &OpKind::MeanAxis(0, false),
        vec![T::ZERO; 2],
        &Layout::new(2),
        &[input],
    );
    assert_eq!(output.data(), &vals(&[2.0, 3.0]));
}

fn mean_axis_1_2d<B: Backend, T: Case<B>>() {
    // [[0,1],[2,3],[4,5]] mean axis 1 → [0.5, 2.5, 4.5]
    let input = arange(6, &[3, 2]);
    let output = B::compute(
        &OpKind::MeanAxis(1, false),
        vec![T::ZERO; 3],
        &Layout::new(3),
        &[input],
    );
    assert_eq!(output.data(), &vals(&[0.5, 2.5, 4.5]));
}

fn mean_axis_negative<B: Backend, T: Case<B>>() {
    // axis=-1 on [3,2] resolves to axis 1: same result as mean_axis_1_2d
    let input = arange(6, &[3, 2]);
    let output = B::compute(
        &OpKind::MeanAxis(-1, false),
        vec![T::ZERO; 3],
        &Layout::new(3),
        &[input],
    );
    assert_eq!(output.data(), &vals(&[0.5, 2.5, 4.5]));
}

// ── expansion ─────────────────────────────────────────────────────────────────

macro_rules! backend_suite {
    ($($name:ident),+ $(,)?) => {
        mod cpu_pure_f32 {
            use super::*;
            $(#[test] fn $name() { super::$name::<CpuPure, f32>() })+
        }
        mod cpu_pure_f64 {
            use super::*;
            $(#[test] fn $name() { super::$name::<CpuPure, f64>() })+
        }
        #[cfg(feature = "mkl")]
        mod cpu_mkl_f32 {
            use super::*;
            $(#[test] fn $name() { super::$name::<CpuMkl, f32>() })+
        }
        #[cfg(feature = "mkl")]
        mod cpu_mkl_f64 {
            use super::*;
            $(#[test] fn $name() { super::$name::<CpuMkl, f64>() })+
        }
    };
}

backend_suite![
    // compute
    as_contiguous_non_contiguous_input,
    scalar_op_axby_contiguous,
    scalar_op_axby_non_contiguous,
    scalar_op_exp,
    scalar_op_ln,
    scalar_op_log2,
    scalar_op_inv,
    scalar_op_inv_non_contiguous,
    fused_scalar,
    add_contiguous,
    add_lhs_non_contiguous,
    add_rhs_non_contiguous,
    add_both_non_contiguous,
    sub_contiguous,
    mul_contiguous,
    div_contiguous,
    // compute_inplace
    scalar_axby_inplace,
    scalar_exp_inplace,
    scalar_ln_inplace,
    scalar_log2_inplace,
    scalar_inv_inplace,
    fused_scalar_inplace,
    add_inplace_reuse_lhs,
    add_inplace_reuse_rhs,
    sub_inplace,
    mul_inplace,
    div_inplace,
    slice_inplace,
    view_inplace,
    transpose_inplace,
    transpose_axes_inplace,
    no_op_inplace,
    // matmul
    matmul_identity_2x2,
    matmul_rectangular,
    matmul_batched,
    matmul_sum_plus,
    matmul_sum_minus,
    matmul_sum_scaled_alpha,
    matmul_sum_scaled_beta,
    // broadcast
    broadcast_inplace_row_to_matrix,
    // reductions
    sum_1d,
    sum_non_contiguous,
    sum_axis_0_2d,
    sum_axis_1_2d,
    sum_axis_negative,
    sum_axis_0_3d,
    sum_axis_middle_3d,
    max_1d,
    max_non_contiguous,
    max_axis_0_2d,
    max_axis_1_2d,
    max_axis_negative,
    mean_1d,
    mean_non_contiguous,
    mean_axis_0_2d,
    mean_axis_1_2d,
    mean_axis_negative,
];
