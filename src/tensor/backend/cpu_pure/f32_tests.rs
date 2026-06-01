use super::*;
use crate::tensor::Dimension;
use crate::tensor::ops::def_op::{OpKindScalar, Sign};

fn td(data: Vec<f32>, shape: &[usize]) -> TensorData<f32> {
    TensorData::from_vec(data, shape, 0)
}

fn arange(n: usize, shape: &[usize]) -> TensorData<f32> {
    TensorData::from_iter((0..n).map(|i| i as f32), shape)
}

// ── cpu_compute_op_f32 ────────────────────────────────────────────────────────

#[test]
fn as_contiguous_non_contiguous_input() {
    let t = TensorData::from_scalar(1.0, &[7, 7]).slice(s![.., 1..2]);
    let buffer = vec![1.0; 7];

    let output = compute_op(
        &OpKind::AsContiguous,
        buffer,
        &Layout::from_shape(&[7, 1], 0),
        std::slice::from_ref(&t),
    );
    assert_eq!(output, t);
    assert!(output.is_contiguous());
}

#[test]
fn scalar_op_axby_contiguous() {
    let input = td(vec![1.0, 2.0, 3.0], &[3]);
    let output = compute_op(
        &OpKind::ScalarOp(OpKindScalar::AxBy(2.0, 1.0)),
        vec![0.0; 3],
        &Layout::from_shape(&[3], 0),
        &[input],
    );
    assert_eq!(output.data(), &vec![3.0, 5.0, 7.0]);
}

#[test]
fn scalar_op_axby_non_contiguous() {
    // Column slice of [3,4] → shape [3,1], stride [4,1], non-contiguous.
    let base = TensorData::from_scalar(1.0_f32, &[3, 4]);
    let input = base.slice(s![.., 0..1]);
    let output = compute_op(
        &OpKind::ScalarOp(OpKindScalar::AxBy(2.0, 3.0)),
        vec![0.0; 3],
        &Layout::from_shape(&[3, 1], 0),
        &[input],
    );
    assert_eq!(output.data(), &vec![5.0, 5.0, 5.0]); // 2*1 + 3 = 5
}

#[test]
fn scalar_op_axby_transposed() {
    // [2,3] transposed to [3,2]: fully non-contiguous, stride [1,3].
    // logical order [0,3,1,4,2,5]; 2x+1 → [1,7,3,9,5,11].
    let base = arange(6, &[2, 3]);
    let input = base.as_layout(base.layout().transpose());
    let output = compute_op(
        &OpKind::ScalarOp(OpKindScalar::AxBy(2.0, 1.0)),
        vec![0.0; 6],
        &Layout::from_shape(&[3, 2], 0),
        &[input],
    );
    assert_eq!(output.data(), &vec![1.0, 7.0, 3.0, 9.0, 5.0, 11.0]);
    assert!(output.is_contiguous());
}

#[test]
fn scalar_op_exp() {
    let input = td(vec![0.0], &[1]);
    let output = compute_op(
        &OpKind::ScalarOp(OpKindScalar::Exp),
        vec![0.0; 1],
        &Layout::from_shape(&[1], 0),
        &[input],
    );
    assert_eq!(output.data(), &vec![1.0]);
}

#[test]
fn scalar_op_ln() {
    let input = td(vec![1.0], &[1]);
    let output = compute_op(
        &OpKind::ScalarOp(OpKindScalar::Ln),
        vec![0.0; 1],
        &Layout::from_shape(&[1], 0),
        &[input],
    );
    assert_eq!(output.data(), &vec![0.0]);
}

#[test]
fn scalar_op_log2() {
    let input = td(vec![1.0, 2.0], &[2]);
    let output = compute_op(
        &OpKind::ScalarOp(OpKindScalar::Log2),
        vec![0.0; 2],
        &Layout::from_shape(&[2], 0),
        &[input],
    );
    assert_eq!(output.data(), &vec![0.0, 1.0]);
}

#[test]
fn fused_scalar() {
    // AxBy(2, 1): 2*3+1=7, then AxBy(3, 0): 3*7+0=21
    let input = td(vec![3.0], &[1]);
    let ops = Box::new([OpKindScalar::AxBy(2.0, 1.0), OpKindScalar::AxBy(3.0, 0.0)]);
    let output = compute_op(
        &OpKind::FusedScalar(ops),
        vec![0.0; 1],
        &Layout::from_shape(&[1], 0),
        &[input],
    );
    assert_eq!(output.data(), &vec![21.0]);
}

#[test]
fn add_contiguous() {
    let lhs = td(vec![1.0, 2.0, 3.0], &[3]);
    let rhs = td(vec![4.0, 5.0, 6.0], &[3]);
    let output = compute_op(
        &OpKind::Add,
        vec![0.0; 3],
        &Layout::from_shape(&[3], 0),
        &[lhs, rhs],
    );
    assert_eq!(output.data(), &vec![5.0, 7.0, 9.0]);
}

#[test]
fn add_lhs_non_contiguous() {
    // lhs: [3,4] sliced to [3,2] - non-contiguous, all 1.0
    // rhs: contiguous [3,2], all 2.0
    let base = TensorData::from_scalar(1.0_f32, &[3, 4]);
    let lhs = base.slice(s![.., 0..2]);
    let rhs = TensorData::from_scalar(2.0_f32, &[3, 2]);
    let output = compute_op(
        &OpKind::Add,
        vec![0.0; 6],
        &Layout::from_shape(&[3, 2], 0),
        &[lhs, rhs],
    );
    assert_eq!(output.data(), &vec![3.0; 6]);
}

#[test]
fn add_rhs_non_contiguous() {
    // lhs: contiguous [3,2], all 1.0
    // rhs: [3,4] sliced to [3,2] - non-contiguous, all 2.0
    let lhs = TensorData::from_scalar(1.0_f32, &[3, 2]);
    let base = TensorData::from_scalar(2.0_f32, &[3, 4]);
    let rhs = base.slice(s![.., 0..2]);
    let output = compute_op(
        &OpKind::Add,
        vec![0.0; 6],
        &Layout::from_shape(&[3, 2], 0),
        &[lhs, rhs],
    );
    assert_eq!(output.data(), &vec![3.0; 6]);
}

#[test]
fn add_both_non_contiguous() {
    let base_a = TensorData::from_scalar(1.0_f32, &[3, 4]);
    let lhs = base_a.slice(s![.., 0..2]);
    let base_b = TensorData::from_scalar(2.0_f32, &[3, 4]);
    let rhs = base_b.slice(s![.., 0..2]);
    let output = compute_op(
        &OpKind::Add,
        vec![0.0; 6],
        &Layout::from_shape(&[3, 2], 0),
        &[lhs, rhs],
    );
    assert_eq!(output.data(), &vec![3.0; 6]);
}

#[test]
fn add_transposed() {
    // Both operands are [2,3] transposed to [3,2], fully non-contiguous.
    // logical order [0,3,1,4,2,5]; element-wise sum → [0,6,2,8,4,10].
    let base_a = arange(6, &[2, 3]);
    let lhs = base_a.as_layout(base_a.layout().transpose());
    let base_b = arange(6, &[2, 3]);
    let rhs = base_b.as_layout(base_b.layout().transpose());
    let output = compute_op(
        &OpKind::Add,
        vec![0.0; 6],
        &Layout::from_shape(&[3, 2], 0),
        &[lhs, rhs],
    );
    let logical: Vec<f32> = output.iter().copied().collect();
    assert_eq!(logical, vec![0.0, 6.0, 2.0, 8.0, 4.0, 10.0]);
}

#[test]
fn sub_contiguous() {
    let lhs = td(vec![5.0, 6.0], &[2]);
    let rhs = td(vec![1.0, 2.0], &[2]);
    let output = compute_op(
        &OpKind::Sub,
        vec![0.0; 2],
        &Layout::from_shape(&[2], 0),
        &[lhs, rhs],
    );
    assert_eq!(output.data(), &vec![4.0, 4.0]);
}

#[test]
fn mul_contiguous() {
    let lhs = td(vec![2.0, 3.0], &[2]);
    let rhs = td(vec![4.0, 5.0], &[2]);
    let output = compute_op(
        &OpKind::Mul,
        vec![0.0; 2],
        &Layout::from_shape(&[2], 0),
        &[lhs, rhs],
    );
    assert_eq!(output.data(), &vec![8.0, 15.0]);
}

#[test]
fn div_contiguous() {
    let lhs = td(vec![6.0, 10.0], &[2]);
    let rhs = td(vec![2.0, 5.0], &[2]);
    let output = compute_op(
        &OpKind::Div,
        vec![0.0; 2],
        &Layout::from_shape(&[2], 0),
        &[lhs, rhs],
    );
    assert_eq!(output.data(), &vec![3.0, 2.0]);
}

// ── cpu_compute_op_f32_inplace ────────────────────────────────────────────────

#[test]
fn scalar_axby_inplace() {
    let input = td(vec![1.0, 2.0, 3.0], &[3]);
    let layout = Layout::from_shape(&[3], 0);
    let output = compute_op_inplace(
        &OpKind::ScalarOp(OpKindScalar::AxBy(2.0, 1.0)),
        &layout,
        vec![input],
        0,
    );
    assert_eq!(output.data(), &vec![3.0, 5.0, 7.0]);
}

#[test]
fn scalar_exp_inplace() {
    let input = td(vec![0.0], &[1]);
    let layout = Layout::from_shape(&[1], 0);
    let output = compute_op_inplace(
        &OpKind::ScalarOp(OpKindScalar::Exp),
        &layout,
        vec![input],
        0,
    );
    assert_eq!(output.data(), &vec![1.0]);
}

#[test]
fn scalar_ln_inplace() {
    let input = td(vec![1.0], &[1]);
    let layout = Layout::from_shape(&[1], 0);
    let output = compute_op_inplace(&OpKind::ScalarOp(OpKindScalar::Ln), &layout, vec![input], 0);
    assert_eq!(output.data(), &vec![0.0]);
}

#[test]
fn scalar_log2_inplace() {
    let input = td(vec![1.0, 2.0], &[2]);
    let layout = Layout::from_shape(&[2], 0);
    let output = compute_op_inplace(
        &OpKind::ScalarOp(OpKindScalar::Log2),
        &layout,
        vec![input],
        0,
    );
    assert_eq!(output.data(), &vec![0.0, 1.0]);
}

#[test]
fn fused_scalar_inplace() {
    // AxBy(2, 1): 2*3+1=7, then AxBy(3, 0): 3*7+0=21
    let input = td(vec![3.0], &[1]);
    let layout = Layout::from_shape(&[1], 0);
    let ops = Box::new([OpKindScalar::AxBy(2.0, 1.0), OpKindScalar::AxBy(3.0, 0.0)]);
    let output = compute_op_inplace(&OpKind::FusedScalar(ops), &layout, vec![input], 0);
    assert_eq!(output.data(), &vec![21.0]);
}

#[test]
fn add_inplace_reuse_lhs() {
    let lhs = td(vec![1.0, 2.0, 3.0], &[3]);
    let rhs = td(vec![4.0, 5.0, 6.0], &[3]);
    let layout = Layout::from_shape(&[3], 0);
    let output = compute_op_inplace(&OpKind::Add, &layout, vec![lhs, rhs], 0);
    assert_eq!(output.data(), &vec![5.0, 7.0, 9.0]);
}

#[test]
fn add_inplace_reuse_rhs() {
    let lhs = td(vec![1.0, 2.0, 3.0], &[3]);
    let rhs = td(vec![4.0, 5.0, 6.0], &[3]);
    let layout = Layout::from_shape(&[3], 0);
    let output = compute_op_inplace(&OpKind::Add, &layout, vec![lhs, rhs], 1);
    assert_eq!(output.data(), &vec![5.0, 7.0, 9.0]);
}

#[test]
fn sub_inplace() {
    let lhs = td(vec![5.0, 6.0], &[2]);
    let rhs = td(vec![1.0, 2.0], &[2]);
    let layout = Layout::from_shape(&[2], 0);
    let output = compute_op_inplace(&OpKind::Sub, &layout, vec![lhs, rhs], 0);
    assert_eq!(output.data(), &vec![4.0, 4.0]);
}

#[test]
fn mul_inplace() {
    let lhs = td(vec![2.0, 3.0], &[2]);
    let rhs = td(vec![4.0, 5.0], &[2]);
    let layout = Layout::from_shape(&[2], 0);
    let output = compute_op_inplace(&OpKind::Mul, &layout, vec![lhs, rhs], 0);
    assert_eq!(output.data(), &vec![8.0, 15.0]);
}

#[test]
fn div_inplace() {
    let lhs = td(vec![6.0, 10.0], &[2]);
    let rhs = td(vec![2.0, 5.0], &[2]);
    let layout = Layout::from_shape(&[2], 0);
    let output = compute_op_inplace(&OpKind::Div, &layout, vec![lhs, rhs], 0);
    assert_eq!(output.data(), &vec![3.0, 2.0]);
}

#[test]
fn slice_inplace() {
    // [[0,1,2],[3,4,5],[6,7,8]]; take columns 1..3 → logical [[1,2],[4,5],[7,8]]
    let input = arange(9, &[3, 3]);
    let new_layout = input.layout().slice(s![.., 1..3]).unwrap();
    let output = compute_op_inplace(
        &OpKind::Slice(new_layout),
        &Layout::from_shape(&[3, 2], 0),
        vec![input],
        0,
    );
    assert_eq!(output.shape(), &[3, 2]);
    assert_eq!(output, td(vec![1.0, 2.0, 4.0, 5.0, 7.0, 8.0], &[3, 2]));
}

#[test]
fn view_inplace() {
    let input = arange(6, &[6]);
    let new_layout = input.layout().view(&[2, 3]).unwrap();
    let output_layout = new_layout.clone();
    let output = compute_op_inplace(&OpKind::View(new_layout), &output_layout, vec![input], 0);
    assert_eq!(output.shape(), &[2, 3]);
    assert_eq!(output.data(), &vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
}

#[test]
fn transpose_inplace() {
    // [[0,1,2],[3,4,5]] → shape [3,2], row-major iteration [0,3,1,4,2,5]
    let input = arange(6, &[2, 3]);
    let output = compute_op_inplace(
        &OpKind::Transpose,
        &Layout::from_shape(&[3, 2], 0),
        vec![input],
        0,
    );
    assert_eq!(output.shape(), &[3, 2]);
    assert_eq!(output, td(vec![0.0, 3.0, 1.0, 4.0, 2.0, 5.0], &[3, 2]));
}

#[test]
fn transpose_axes_inplace() {
    let input = arange(6, &[2, 3]);
    let new_layout = input.layout().transpose_axes(&[1, 0]).unwrap();
    let output_layout = new_layout.clone();
    let output = compute_op_inplace(
        &OpKind::TransposeAxes(new_layout),
        &output_layout,
        vec![input],
        0,
    );
    assert_eq!(output.shape(), &[3, 2]);
    assert_eq!(output, td(vec![0.0, 3.0, 1.0, 4.0, 2.0, 5.0], &[3, 2]));
}

#[test]
fn no_op_inplace() {
    let input = td(vec![1.0, 2.0, 3.0], &[3]);
    let layout = Layout::from_shape(&[3], 0);
    let output = compute_op_inplace(&OpKind::NoOp, &layout, vec![input], 0);
    assert_eq!(output.data(), &vec![1.0, 2.0, 3.0]);
}

// ── cpu_compute_op_f32 (matmul) ──────────────────────────────────────────────

#[test]
fn matmul_identity_2x2() {
    // A @ I = A
    let a = td(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let eye = td(vec![1.0, 0.0, 0.0, 1.0], &[2, 2]);
    let output = compute_op(
        &OpKind::MatMul(1.0),
        vec![0.0; 4],
        &Layout::from_shape(&[2, 2], 0),
        &[a.clone(), eye],
    );
    assert_eq!(output.data(), a.data());
}

#[test]
fn matmul_rectangular() {
    // [2,3] @ [3,2] = [2,2]
    // A = [[1,2,3],[4,5,6]], B = [[7,8],[9,10],[11,12]]
    // C[0,0] = 1*7+2*9+3*11 = 58,  C[0,1] = 1*8+2*10+3*12 = 64
    // C[1,0] = 4*7+5*9+6*11 = 139, C[1,1] = 4*8+5*10+6*12 = 154
    let a = td(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let b = td(vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0], &[3, 2]);
    let output = compute_op(
        &OpKind::MatMul(1.0),
        vec![0.0; 4],
        &Layout::from_shape(&[2, 2], 0),
        &[a, b],
    );
    assert_eq!(output.data(), &vec![58.0, 64.0, 139.0, 154.0]);
}

#[test]
fn matmul_batched() {
    // [2,2,2] @ [2,2,2] = [2,2,2]
    // Both batches of A are all-ones; B batch 0 is I, B batch 1 is 2*I.
    let a = TensorData::from_scalar(1.0_f32, &[2, 2, 2]);
    let b = td(vec![1.0, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 2.0], &[2, 2, 2]);
    let output = compute_op(
        &OpKind::MatMul(1.0),
        vec![0.0; 8],
        &Layout::from_shape(&[2, 2, 2], 0),
        &[a, b],
    );
    // batch 0: [[1,1],[1,1]] @ I = [[1,1],[1,1]]
    // batch 1: [[1,1],[1,1]] @ 2I = [[2,2],[2,2]]
    assert_eq!(output.data(), &vec![1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0]);
}

#[test]
fn matmul_transposed() {
    // Both operands are physically transposed 2x2 views (stride [1,2]).
    // A = [[1,2],[3,4]], B = [[5,6],[7,8]]; A@B = [[19,22],[43,50]].
    let a_phys = td(vec![1.0, 3.0, 2.0, 4.0], &[2, 2]);
    let a = a_phys.as_layout(a_phys.layout().transpose());
    let b_phys = td(vec![5.0, 7.0, 6.0, 8.0], &[2, 2]);
    let b = b_phys.as_layout(b_phys.layout().transpose());
    let output = compute_op(
        &OpKind::MatMul(1.0),
        vec![0.0; 4],
        &Layout::from_shape(&[2, 2], 0),
        &[a, b],
    );
    assert_eq!(output.data(), &vec![19.0, 22.0, 43.0, 50.0]);
}

#[test]
fn matmul_batched_transposed() {
    // A is [2,2,2] with its last two axes transposed (non-contiguous batch
    // matrices); B is contiguous identity per batch, so A@B = A.
    // A logical: batch 0 [[1,2],[3,4]], batch 1 [[5,6],[7,8]].
    let a_phys = td(vec![1.0, 3.0, 2.0, 4.0, 5.0, 7.0, 6.0, 8.0], &[2, 2, 2]);
    let a = a_phys.as_layout(a_phys.layout().transpose_axes(&[0, 2, 1]).unwrap());
    let b = td(vec![1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0], &[2, 2, 2]);
    let output = compute_op(
        &OpKind::MatMul(1.0),
        vec![0.0; 8],
        &Layout::from_shape(&[2, 2, 2], 0),
        &[a, b],
    );
    assert_eq!(output.data(), &vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
}

// ── cpu_compute_op_f32 (matmul_sum) ──────────────────────────────────────────
// MatMulSum(alpha, beta, sign) computes: alpha*(A@B) +/- beta*C
// All tests use A = identity [[1,0],[0,1]], B = [[2,3],[4,5]], so A@B = [[2,3],[4,5]].

#[test]
fn matmul_sum_plus() {
    let a = td(vec![1.0, 0.0, 0.0, 1.0], &[2, 2]);
    let b = td(vec![2.0, 3.0, 4.0, 5.0], &[2, 2]);
    let c = td(vec![1.0, 1.0, 1.0, 1.0], &[2, 2]);
    let output = compute_op(
        &OpKind::MatMulSum(1.0, 1.0, Sign::Plus),
        vec![0.0; 4],
        &Layout::from_shape(&[2, 2], 0),
        &[a, b, c],
    );
    // 1*(A@B) + 1*C = [[2,3],[4,5]] + [[1,1],[1,1]] = [[3,4],[5,6]]
    assert_eq!(output.data(), &vec![3.0, 4.0, 5.0, 6.0]);
}

#[test]
fn matmul_sum_minus() {
    let a = td(vec![1.0, 0.0, 0.0, 1.0], &[2, 2]);
    let b = td(vec![2.0, 3.0, 4.0, 5.0], &[2, 2]);
    let c = td(vec![1.0, 1.0, 1.0, 1.0], &[2, 2]);
    let output = compute_op(
        &OpKind::MatMulSum(1.0, 1.0, Sign::Minus),
        vec![0.0; 4],
        &Layout::from_shape(&[2, 2], 0),
        &[a, b, c],
    );
    // 1*(A@B) - 1*C = [[2,3],[4,5]] - [[1,1],[1,1]] = [[1,2],[3,4]]
    assert_eq!(output.data(), &vec![1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn matmul_sum_scaled_alpha() {
    let a = td(vec![1.0, 0.0, 0.0, 1.0], &[2, 2]);
    let b = td(vec![2.0, 3.0, 4.0, 5.0], &[2, 2]);
    let c = td(vec![1.0, 1.0, 1.0, 1.0], &[2, 2]);
    let output = compute_op(
        &OpKind::MatMulSum(2.0, 1.0, Sign::Plus),
        vec![0.0; 4],
        &Layout::from_shape(&[2, 2], 0),
        &[a, b, c],
    );
    // 2*(A@B) + 1*C = 2*[[2,3],[4,5]] + [[1,1],[1,1]] = [[5,7],[9,11]]
    assert_eq!(output.data(), &vec![5.0, 7.0, 9.0, 11.0]);
}

#[test]
fn matmul_sum_scaled_beta() {
    let a = td(vec![1.0, 0.0, 0.0, 1.0], &[2, 2]);
    let b = td(vec![2.0, 3.0, 4.0, 5.0], &[2, 2]);
    let c = td(vec![1.0, 1.0, 1.0, 1.0], &[2, 2]);
    let output = compute_op(
        &OpKind::MatMulSum(1.0, 2.0, Sign::Plus),
        vec![0.0; 4],
        &Layout::from_shape(&[2, 2], 0),
        &[a, b, c],
    );
    // 1*(A@B) + 2*C = [[2,3],[4,5]] + 2*[[1,1],[1,1]] = [[4,5],[6,7]]
    assert_eq!(output.data(), &vec![4.0, 5.0, 6.0, 7.0]);
}

// ── cpu_compute_op_f32 (broadcast) ───────────────────────────────────────────
// Failure cases: Layout::broadcast rejects dimensions that are neither 1 nor equal.

#[test]
fn broadcast_non_one_dim_mismatch_returns_error() {
    // [2] cannot broadcast to [3]: dim 2 is not 1 and 2 != 3
    let layout = Layout::from_shape(&[2], 0);
    assert!(layout.broadcast(&[3]).is_err());
}

#[test]
fn broadcast_inner_dim_mismatch_returns_error() {
    // [2, 3] cannot broadcast to [2, 4]: last dim 3 is not 1 and 3 != 4
    let layout = Layout::from_shape(&[2, 3], 0);
    assert!(layout.broadcast(&[2, 4]).is_err());
}

#[test]
fn broadcast_smaller_rank_mismatch_returns_error() {
    // [4] cannot broadcast to [2, 3]: dim 4 is not 1 and 4 != 3
    let layout = Layout::from_shape(&[4], 0);
    assert!(layout.broadcast(&[2, 3]).is_err());
}

#[test]
fn broadcast_rank_reduction_returns_error() {
    // Cannot broadcast to a shape with fewer dimensions: [3,4] → [4] shrinks rank
    let layout = Layout::from_shape(&[3, 4], 0);
    assert!(layout.broadcast(&[4]).is_err());
}

// Success cases

#[test]
fn broadcast_row_to_matrix() {
    // [1,3] broadcast to [2,3]: the single row is accessible twice (stride[0]=0)
    let input = td(vec![1.0, 2.0, 3.0], &[1, 3]);
    let new_layout = input.layout().broadcast(&[2, 3]).unwrap();
    let output = compute_op(
        &OpKind::Broadcast(new_layout.clone()),
        vec![0.0; 6],
        &new_layout,
        &[input],
    );
    assert_eq!(output.shape(), &[2, 3]);
    let logical: Vec<f32> = output.iter().copied().collect();
    assert_eq!(logical, vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0]);
}

#[test]
fn broadcast_vector_to_matrix() {
    // [3] broadcast to [2,3]: inserts a leading dim with stride 0
    let input = td(vec![4.0, 5.0, 6.0], &[3]);
    let new_layout = input.layout().broadcast(&[2, 3]).unwrap();
    let output = compute_op(
        &OpKind::Broadcast(new_layout.clone()),
        vec![0.0; 6],
        &new_layout,
        &[input],
    );
    assert_eq!(output.shape(), &[2, 3]);
    let logical: Vec<f32> = output.iter().copied().collect();
    assert_eq!(logical, vec![4.0, 5.0, 6.0, 4.0, 5.0, 6.0]);
}

#[test]
fn broadcast_transposed() {
    // [1,3] transposed to [3,1] (non-contiguous, stride [1,3]), then the
    // size-1 trailing dim is broadcast to [3,4].
    let base = td(vec![10.0, 20.0, 30.0], &[1, 3]);
    let input = base.as_layout(base.layout().transpose());
    let new_layout = input.layout().broadcast(&[3, 4]).unwrap();
    let output = compute_op(
        &OpKind::Broadcast(new_layout.clone()),
        vec![0.0; 12],
        &new_layout,
        &[input],
    );
    assert_eq!(output.shape(), &[3, 4]);
    let logical: Vec<f32> = output.iter().copied().collect();
    assert_eq!(
        logical,
        vec![
            10.0, 10.0, 10.0, 10.0, 20.0, 20.0, 20.0, 20.0, 30.0, 30.0, 30.0, 30.0
        ]
    );
}

// ── cpu_compute_op_f32_inplace (broadcast) ───────────────────────────────────

#[test]
fn broadcast_inplace_row_to_matrix() {
    let input = td(vec![7.0, 8.0, 9.0], &[1, 3]);
    let new_layout = input.layout().broadcast(&[2, 3]).unwrap();
    let output = compute_op_inplace(
        &OpKind::Broadcast(new_layout.clone()),
        &new_layout,
        vec![input],
        0,
    );
    assert_eq!(output.shape(), &[2, 3]);
    let logical: Vec<f32> = output.iter().copied().collect();
    assert_eq!(logical, vec![7.0, 8.0, 9.0, 7.0, 8.0, 9.0]);
}

// ── Sum / SumAxis ─────────────────────────────────────────────────────────────

#[test]
fn sum_1d() {
    // [0,1,2,3,4] → 10
    let input = arange(5, &[5]);
    let output = compute_op(
        &OpKind::Sum,
        vec![0.0; 1],
        &Layout::from_shape(&[1], 0),
        &[input],
    );
    assert_eq!(output.data(), &vec![10.0]);
}

#[test]
fn sum_non_contiguous() {
    // Column 0 of [[0,1,2],[3,4,5]] is [0,3] → sum = 3
    let base = arange(6, &[2, 3]);
    let input = base.slice(s![.., 0..1]);
    let output = compute_op(
        &OpKind::Sum,
        vec![0.0; 1],
        &Layout::from_shape(&[1], 0),
        &[input],
    );
    assert_eq!(output.data(), &vec![3.0]);
}

#[test]
fn sum_transposed() {
    // [2,3] transposed to [3,2], fully non-contiguous; total sum = 15.
    let base = arange(6, &[2, 3]);
    let input = base.as_layout(base.layout().transpose());
    let output = compute_op(
        &OpKind::Sum,
        vec![0.0; 1],
        &Layout::from_shape(&[1], 0),
        &[input],
    );
    assert_eq!(output.data(), &vec![15.0]);
}

#[test]
fn sum_axis_0_2d() {
    // [[0,1],[2,3],[4,5]] sum axis 0 → [6, 9]
    let input = arange(6, &[3, 2]);
    let output = compute_op(
        &OpKind::SumAxis(0, false),
        vec![0.0; 2],
        &Layout::from_shape(&[2], 0),
        &[input],
    );
    assert_eq!(output.data(), &vec![6.0, 9.0]);
}

#[test]
fn sum_axis_1_2d() {
    // [[0,1],[2,3],[4,5]] sum axis 1 → [1, 5, 9]
    let input = arange(6, &[3, 2]);
    let output = compute_op(
        &OpKind::SumAxis(1, false),
        vec![0.0; 3],
        &Layout::from_shape(&[3], 0),
        &[input],
    );
    assert_eq!(output.data(), &vec![1.0, 5.0, 9.0]);
}

#[test]
fn sum_axis_negative() {
    // axis=-1 on [3,2] resolves to axis 1: same result as sum_axis_1_2d
    let input = arange(6, &[3, 2]);
    let output = compute_op(
        &OpKind::SumAxis(-1, false),
        vec![0.0; 3],
        &Layout::from_shape(&[3], 0),
        &[input],
    );
    assert_eq!(output.data(), &vec![1.0, 5.0, 9.0]);
}

#[test]
fn sum_axis_0_3d() {
    // [2,3,4] sum axis 0 → [3,4]; result[i] = i + (i+12)
    let input = arange(24, &[2, 3, 4]);
    let output = compute_op(
        &OpKind::SumAxis(0, false),
        vec![0.0; 12],
        &Layout::from_shape(&[3, 4], 0),
        &[input],
    );
    let expected: Vec<f32> = (0..12).map(|i| i as f32 + (i + 12) as f32).collect();
    assert_eq!(output.data(), &expected);
}

#[test]
fn sum_axis_middle_3d() {
    // [2,3,1] sum axis 1 → [2,1]
    // data: [0,1,2, 3,4,5]; batch 0 sum = 0+1+2 = 3, batch 1 sum = 3+4+5 = 12
    let input = arange(6, &[2, 3, 1]);
    let output = compute_op(
        &OpKind::SumAxis(1, false),
        vec![0.0; 2],
        &Layout::from_shape(&[2, 1], 0),
        &[input],
    );
    assert_eq!(output.data(), &vec![3.0, 12.0]);
}

// ── Max / MaxAxis ─────────────────────────────────────────────────────────────

#[test]
fn max_1d() {
    // max of [0,1,2,3,4] = 4
    let input = arange(5, &[5]);
    let output = compute_op(
        &OpKind::Max,
        vec![0.0; 1],
        &Layout::from_shape(&[1], 0),
        &[input],
    );
    assert_eq!(output.data(), &vec![4.0]);
}

#[test]
fn max_non_contiguous() {
    // Column 0 of [[0,1,2],[3,4,5]] is [0,3] → max = 3
    let base = arange(6, &[2, 3]);
    let input = base.slice(s![.., 0..1]);
    let output = compute_op(
        &OpKind::Max,
        vec![0.0; 1],
        &Layout::from_shape(&[1], 0),
        &[input],
    );
    assert_eq!(output.data(), &vec![3.0]);
}

#[test]
fn max_axis_0_2d() {
    // [[0,1],[2,3],[4,5]] max axis 0 → [4, 5]
    let input = arange(6, &[3, 2]);
    let output = compute_op(
        &OpKind::MaxAxis(0, false),
        vec![0.0; 2],
        &Layout::from_shape(&[2], 0),
        &[input],
    );
    assert_eq!(output.data(), &vec![4.0, 5.0]);
}

#[test]
fn max_axis_1_2d() {
    // [[0,1],[2,3],[4,5]] max axis 1 → [1, 3, 5]
    let input = arange(6, &[3, 2]);
    let output = compute_op(
        &OpKind::MaxAxis(1, false),
        vec![0.0; 3],
        &Layout::from_shape(&[3], 0),
        &[input],
    );
    assert_eq!(output.data(), &vec![1.0, 3.0, 5.0]);
}

#[test]
fn max_axis_negative() {
    // axis=-1 on [3,2] resolves to axis 1: same result as max_axis_1_2d
    let input = arange(6, &[3, 2]);
    let output = compute_op(
        &OpKind::MaxAxis(-1, false),
        vec![0.0; 3],
        &Layout::from_shape(&[3], 0),
        &[input],
    );
    assert_eq!(output.data(), &vec![1.0, 3.0, 5.0]);
}

// ── Mean / MeanAxis ───────────────────────────────────────────────────────────

#[test]
fn mean_1d() {
    // mean of [0,1,2,3,4] = 2.0
    let input = arange(5, &[5]);
    let output = compute_op(
        &OpKind::Mean,
        vec![0.0; 1],
        &Layout::from_shape(&[1], 0),
        &[input],
    );
    assert_eq!(output.data(), &vec![2.0]);
}

#[test]
fn mean_non_contiguous() {
    // Column 0 of [[0,1,2],[3,4,5]] is [0,3] → mean = 1.5
    let base = arange(6, &[2, 3]);
    let input = base.slice(s![.., 0..1]);
    let output = compute_op(
        &OpKind::Mean,
        vec![0.0; 1],
        &Layout::from_shape(&[1], 0),
        &[input],
    );
    assert_eq!(output.data(), &vec![1.5]);
}

#[test]
fn mean_axis_0_2d() {
    // [[0,1],[2,3],[4,5]] mean axis 0 → [2.0, 3.0]
    let input = arange(6, &[3, 2]);
    let output = compute_op(
        &OpKind::MeanAxis(0, false),
        vec![0.0; 2],
        &Layout::from_shape(&[2], 0),
        &[input],
    );
    assert_eq!(output.data(), &vec![2.0, 3.0]);
}

#[test]
fn mean_axis_1_2d() {
    // [[0,1],[2,3],[4,5]] mean axis 1 → [0.5, 2.5, 4.5]
    let input = arange(6, &[3, 2]);
    let output = compute_op(
        &OpKind::MeanAxis(1, false),
        vec![0.0; 3],
        &Layout::from_shape(&[3], 0),
        &[input],
    );
    assert_eq!(output.data(), &vec![0.5, 2.5, 4.5]);
}

#[test]
fn mean_axis_negative() {
    // axis=-1 on [3,2] resolves to axis 1: same result as mean_axis_1_2d
    let input = arange(6, &[3, 2]);
    let output = compute_op(
        &OpKind::MeanAxis(-1, false),
        vec![0.0; 3],
        &Layout::from_shape(&[3], 0),
        &[input],
    );
    assert_eq!(output.data(), &vec![0.5, 2.5, 4.5]);
}
