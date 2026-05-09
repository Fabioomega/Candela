use super::*;
use crate::tensor::ops::def_op::OpKindScalar;

fn td(data: Vec<f64>, shape: &[usize]) -> TensorData<f64> {
    TensorData::from_vec(data, shape, 0)
}

fn arange(n: usize, shape: &[usize]) -> TensorData<f64> {
    TensorData::from_iter((0..n).map(|i| i as f64), shape)
}

// ── cpu_compute_op_f64 ────────────────────────────────────────────────────────

#[test]
fn compute_as_contiguous_f64() {
    let t = TensorData::from_scalar(1.0, &[7, 7]).slice(s![.., 1..2]);
    let buffer = vec![1.0; 7];

    let output = cpu_compute_op_f64(
        &OpKind::AsContiguous,
        buffer,
        &Layout::from_shape(&[7, 1], 0),
        &[t.clone()],
    );
    assert_eq!(output, t);
    assert!(output.is_contiguous());
}

#[test]
fn scalar_op_axby_contiguous() {
    let input = td(vec![1.0, 2.0, 3.0], &[3]);
    let output = cpu_compute_op_f64(
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
    let base = TensorData::from_scalar(1.0_f64, &[3, 4]);
    let input = base.slice(s![.., 0..1]);
    let output = cpu_compute_op_f64(
        &OpKind::ScalarOp(OpKindScalar::AxBy(2.0, 3.0)),
        vec![0.0; 3],
        &Layout::from_shape(&[3, 1], 0),
        &[input],
    );
    assert_eq!(output.data(), &vec![5.0, 5.0, 5.0]); // 2*1 + 3 = 5
}

#[test]
fn scalar_op_exp() {
    let input = td(vec![0.0], &[1]);
    let output = cpu_compute_op_f64(
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
    let output = cpu_compute_op_f64(
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
    let output = cpu_compute_op_f64(
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
    let output = cpu_compute_op_f64(
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
    let output = cpu_compute_op_f64(
        &OpKind::Add,
        vec![0.0; 3],
        &Layout::from_shape(&[3], 0),
        &[lhs, rhs],
    );
    assert_eq!(output.data(), &vec![5.0, 7.0, 9.0]);
}

#[test]
fn add_lhs_non_contiguous() {
    // lhs: [3,4] sliced to [3,2] — non-contiguous, all 1.0
    // rhs: contiguous [3,2], all 2.0
    let base = TensorData::from_scalar(1.0_f64, &[3, 4]);
    let lhs = base.slice(s![.., 0..2]);
    let rhs = TensorData::from_scalar(2.0_f64, &[3, 2]);
    let output = cpu_compute_op_f64(
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
    // rhs: [3,4] sliced to [3,2] — non-contiguous, all 2.0
    let lhs = TensorData::from_scalar(1.0_f64, &[3, 2]);
    let base = TensorData::from_scalar(2.0_f64, &[3, 4]);
    let rhs = base.slice(s![.., 0..2]);
    let output = cpu_compute_op_f64(
        &OpKind::Add,
        vec![0.0; 6],
        &Layout::from_shape(&[3, 2], 0),
        &[lhs, rhs],
    );
    assert_eq!(output.data(), &vec![3.0; 6]);
}

#[test]
fn add_both_non_contiguous() {
    let base_a = TensorData::from_scalar(1.0_f64, &[3, 4]);
    let lhs = base_a.slice(s![.., 0..2]);
    let base_b = TensorData::from_scalar(2.0_f64, &[3, 4]);
    let rhs = base_b.slice(s![.., 0..2]);
    let output = cpu_compute_op_f64(
        &OpKind::Add,
        vec![0.0; 6],
        &Layout::from_shape(&[3, 2], 0),
        &[lhs, rhs],
    );
    assert_eq!(output.data(), &vec![3.0; 6]);
}

#[test]
fn sub_contiguous() {
    let lhs = td(vec![5.0, 6.0], &[2]);
    let rhs = td(vec![1.0, 2.0], &[2]);
    let output = cpu_compute_op_f64(
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
    let output = cpu_compute_op_f64(
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
    let output = cpu_compute_op_f64(
        &OpKind::Div,
        vec![0.0; 2],
        &Layout::from_shape(&[2], 0),
        &[lhs, rhs],
    );
    assert_eq!(output.data(), &vec![3.0, 2.0]);
}

// ── cpu_compute_op_f64_inplace ────────────────────────────────────────────────

#[test]
fn scalar_axby_inplace() {
    let input = td(vec![1.0, 2.0, 3.0], &[3]);
    let layout = Layout::from_shape(&[3], 0);
    let output = cpu_compute_op_f64_inplace(
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
    let output =
        cpu_compute_op_f64_inplace(&OpKind::ScalarOp(OpKindScalar::Exp), &layout, vec![input], 0);
    assert_eq!(output.data(), &vec![1.0]);
}

#[test]
fn scalar_ln_inplace() {
    let input = td(vec![1.0], &[1]);
    let layout = Layout::from_shape(&[1], 0);
    let output =
        cpu_compute_op_f64_inplace(&OpKind::ScalarOp(OpKindScalar::Ln), &layout, vec![input], 0);
    assert_eq!(output.data(), &vec![0.0]);
}

#[test]
fn scalar_log2_inplace() {
    let input = td(vec![1.0, 2.0], &[2]);
    let layout = Layout::from_shape(&[2], 0);
    let output = cpu_compute_op_f64_inplace(
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
    let output = cpu_compute_op_f64_inplace(&OpKind::FusedScalar(ops), &layout, vec![input], 0);
    assert_eq!(output.data(), &vec![21.0]);
}

#[test]
fn add_inplace_reuse_lhs() {
    let lhs = td(vec![1.0, 2.0, 3.0], &[3]);
    let rhs = td(vec![4.0, 5.0, 6.0], &[3]);
    let layout = Layout::from_shape(&[3], 0);
    let output = cpu_compute_op_f64_inplace(&OpKind::Add, &layout, vec![lhs, rhs], 0);
    assert_eq!(output.data(), &vec![5.0, 7.0, 9.0]);
}

#[test]
fn add_inplace_reuse_rhs() {
    let lhs = td(vec![1.0, 2.0, 3.0], &[3]);
    let rhs = td(vec![4.0, 5.0, 6.0], &[3]);
    let layout = Layout::from_shape(&[3], 0);
    let output = cpu_compute_op_f64_inplace(&OpKind::Add, &layout, vec![lhs, rhs], 1);
    assert_eq!(output.data(), &vec![5.0, 7.0, 9.0]);
}

#[test]
fn sub_inplace() {
    let lhs = td(vec![5.0, 6.0], &[2]);
    let rhs = td(vec![1.0, 2.0], &[2]);
    let layout = Layout::from_shape(&[2], 0);
    let output = cpu_compute_op_f64_inplace(&OpKind::Sub, &layout, vec![lhs, rhs], 0);
    assert_eq!(output.data(), &vec![4.0, 4.0]);
}

#[test]
fn mul_inplace() {
    let lhs = td(vec![2.0, 3.0], &[2]);
    let rhs = td(vec![4.0, 5.0], &[2]);
    let layout = Layout::from_shape(&[2], 0);
    let output = cpu_compute_op_f64_inplace(&OpKind::Mul, &layout, vec![lhs, rhs], 0);
    assert_eq!(output.data(), &vec![8.0, 15.0]);
}

#[test]
fn div_inplace() {
    let lhs = td(vec![6.0, 10.0], &[2]);
    let rhs = td(vec![2.0, 5.0], &[2]);
    let layout = Layout::from_shape(&[2], 0);
    let output = cpu_compute_op_f64_inplace(&OpKind::Div, &layout, vec![lhs, rhs], 0);
    assert_eq!(output.data(), &vec![3.0, 2.0]);
}

#[test]
fn slice_inplace() {
    // [[0,1,2],[3,4,5],[6,7,8]]; take columns 1..3 → logical [[1,2],[4,5],[7,8]]
    let input = arange(9, &[3, 3]);
    let new_layout = input.layout().slice(s![.., 1..3]).unwrap();
    let output = cpu_compute_op_f64_inplace(
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
    let output =
        cpu_compute_op_f64_inplace(&OpKind::View(new_layout), &output_layout, vec![input], 0);
    assert_eq!(output.shape(), &[2, 3]);
    assert_eq!(output.data(), &vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
}

#[test]
fn transpose_inplace() {
    // [[0,1,2],[3,4,5]] → shape [3,2], row-major iteration [0,3,1,4,2,5]
    let input = arange(6, &[2, 3]);
    let output = cpu_compute_op_f64_inplace(
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
    let output = cpu_compute_op_f64_inplace(
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
    let output = cpu_compute_op_f64_inplace(&OpKind::NoOp, &layout, vec![input], 0);
    assert_eq!(output.data(), &vec![1.0, 2.0, 3.0]);
}
