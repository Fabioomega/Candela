use super::*;
use crate::tensor::errors::OpError;
use crate::tensor::ops::def_op::{OpKindScalar, Sign};

// ── scalar / noop ops ─────────────────────────────────────────────────────────

#[test]
fn scalar_op_same_shape() {
    let input = Layout::new((3, 4));
    let result =
        compute_layout::<f64>(&OpKind::ScalarOp(OpKindScalar::AxBy(2.0, 1.0)), &[&input]).unwrap();
    assert_eq!(result.shape(), input.shape());
    assert_eq!(result.len(), input.len());
}

#[test]
fn noop_same_shape() {
    let input = Layout::new(5);
    let result = compute_layout::<f64>(&OpKind::NoOp, &[&input]).unwrap();
    assert_eq!(result.shape(), input.shape());
}

// ── binary elementwise ops ────────────────────────────────────────────────────

#[test]
fn add_equal_shapes() {
    let a = Layout::new((2, 3));
    let b = Layout::new((2, 3));
    let result = compute_layout::<f64>(&OpKind::Add, &[&a, &b]).unwrap();
    assert_eq!(result.shape(), &[2, 3]);
}

#[test]
fn add_shape_mismatch() {
    let a = Layout::new((3, 4));
    let b = Layout::new((3, 5));
    let err = compute_layout::<f64>(&OpKind::Add, &[&a, &b]).unwrap_err();
    match err {
        OpError::NotSameShape(s1, s2) => {
            assert_eq!(&*s1, &[3, 4]);
            assert_eq!(&*s2, &[3, 5]);
        }
        _ => panic!("expected NotSameShape, got {err:?}"),
    }
}

#[test]
fn matmul_output_shape() {
    // [2,3] @ [3,4] = [2,4]
    let a = Layout::new((2, 3));
    let b = Layout::new((3, 4));
    let result = compute_layout::<f64>(&OpKind::MatMul(1.0), &[&a, &b]).unwrap();
    assert_eq!(result.shape(), &[2, 4]);
}

#[test]
fn matmul_dimension_mismatch() {
    // [2,3] @ [4,5] - inner dims 3 != 4
    let a = Layout::new((2, 3));
    let b = Layout::new((4, 5));
    let err = compute_layout::<f64>(&OpKind::MatMul(1.0), &[&a, &b]).unwrap_err();
    assert!(matches!(err, OpError::CannotMatMul(3, 4)));
}

#[test]
fn matmul_batched_output_shape() {
    // [2,3,4] @ [2,4,5] = [2,3,5]
    let a = Layout::new((2, 3, 4));
    let b = Layout::new((2, 4, 5));
    let result = compute_layout::<f64>(&OpKind::MatMul(1.0), &[&a, &b]).unwrap();
    assert_eq!(result.shape(), &[2, 3, 5]);
}

#[test]
fn sub_equal_shapes() {
    let a = Layout::new((2, 3));
    let b = Layout::new((2, 3));
    let result = compute_layout::<f64>(&OpKind::Sub, &[&a, &b]).unwrap();
    assert_eq!(result.shape(), &[2, 3]);
}

#[test]
fn mul_equal_shapes() {
    let a = Layout::new((2, 3));
    let b = Layout::new((2, 3));
    let result = compute_layout::<f64>(&OpKind::Mul, &[&a, &b]).unwrap();
    assert_eq!(result.shape(), &[2, 3]);
}

#[test]
fn div_equal_shapes() {
    let a = Layout::new((2, 3));
    let b = Layout::new((2, 3));
    let result = compute_layout::<f64>(&OpKind::Div, &[&a, &b]).unwrap();
    assert_eq!(result.shape(), &[2, 3]);
}

// ── layout-only ops ───────────────────────────────────────────────────────────

#[test]
fn as_contiguous_output_layout() {
    let transposed = Layout::new((3, 4)).transpose(); // shape [4,3], stride [1,4] - not contiguous
    let result = compute_layout::<f64>(&OpKind::AsContiguous, &[&transposed]).unwrap();
    assert_eq!(result.shape(), &[4, 3]);
    assert!(result.is_contiguous());
    assert_eq!(result.stride(), &[3_i32, 1]);
}

// ── MatMulSum layout ──────────────────────────────────────────────────────────

#[test]
fn matmulsum_output_shape() {
    // [2,3] @ [3,4] + bias[2,4] → [2,4]
    let a = Layout::new((2, 3));
    let b = Layout::new((3, 4));
    let bias = Layout::new((2, 4));
    let result =
        compute_layout::<f64>(&OpKind::MatMulSum(1.0, 1.0, Sign::Plus), &[&a, &b, &bias]).unwrap();
    assert_eq!(result.shape(), &[2, 4]);
}

#[test]
fn matmulsum_bias_shape_mismatch() {
    // Batched: [2,3,4] @ [2,4,5] = [2,3,5], but bias is [2,3,3]
    // The 2D code path has an early return that skips bias validation (see impl_layout.rs:54).
    // Using 3D inputs to reach the validation branch.
    let a = Layout::new((2, 3, 4));
    let b = Layout::new((2, 4, 5));
    let bias = Layout::new((2, 3, 3)); // wrong last dim: output is [2,3,5]
    let err = compute_layout::<f64>(&OpKind::MatMulSum(1.0, 1.0, Sign::Plus), &[&a, &b, &bias])
        .unwrap_err();
    match err {
        OpError::NotSameShape(expected, got) => {
            assert_eq!(&*expected, &[2, 3, 5]);
            assert_eq!(&*got, &[2, 3, 3]);
        }
        _ => panic!("expected NotSameShape, got {err:?}"),
    }
}

#[test]
fn matmulsum_2d_bias_shape_mismatch() {
    let a = Layout::new((2, 3));
    let b = Layout::new((3, 4));
    let bias = Layout::new((3, 3));
    let result = compute_layout::<f64>(&OpKind::MatMulSum(1.0, 1.0, Sign::Plus), &[&a, &b, &bias]);
    assert!(
        result.is_err(),
        "expected Err for mismatched 2D bias, got Ok({:?})",
        result.unwrap().shape()
    );
}

// ── Sum / SumAxis layout ──────────────────────────────────────────────────────

#[test]
fn sum_scalar_output() {
    // Any shape reduces to a single-element tensor
    let input = Layout::new((3, 4));
    let result = compute_layout::<f64>(&OpKind::Sum, &[&input]).unwrap();
    assert_eq!(result.shape(), &[1]);
}

#[test]
fn sum_axis_0_no_keepdim() {
    // [3,4] reduce axis 0 → [4]
    let input = Layout::new((3, 4));
    let result = compute_layout::<f64>(&OpKind::SumAxis(0, false), &[&input]).unwrap();
    assert_eq!(result.shape(), &[4]);
}

#[test]
fn sum_axis_1_no_keepdim() {
    // [3,4] reduce axis 1 → [3]
    let input = Layout::new((3, 4));
    let result = compute_layout::<f64>(&OpKind::SumAxis(1, false), &[&input]).unwrap();
    assert_eq!(result.shape(), &[3]);
}

#[test]
fn sum_axis_keepdim() {
    // [3,4] reduce axis 0, keepdim=true → [1,4]
    let input = Layout::new((3, 4));
    let result = compute_layout::<f64>(&OpKind::SumAxis(0, true), &[&input]).unwrap();
    assert_eq!(result.shape(), &[1, 4]);
}

#[test]
fn sum_axis_negative() {
    // axis=-1 on [3,4] resolves to axis 1 → [3]
    let input = Layout::new((3, 4));
    let result = compute_layout::<f64>(&OpKind::SumAxis(-1, false), &[&input]).unwrap();
    assert_eq!(result.shape(), &[3]);
}

#[test]
fn sum_axis_out_of_bounds() {
    let input = Layout::new((3, 4));
    let err = compute_layout::<f64>(&OpKind::SumAxis(5, false), &[&input]).unwrap_err();
    assert!(matches!(err, OpError::AxesOutOfBounds));
}

// ── Max / MaxAxis layout ──────────────────────────────────────────────────────

#[test]
fn max_scalar_output() {
    let input = Layout::new((3, 4));
    let result = compute_layout::<f64>(&OpKind::Max, &[&input]).unwrap();
    assert_eq!(result.shape(), &[1]);
}

#[test]
fn max_axis_0_no_keepdim() {
    // [3,4] reduce axis 0 → [4]
    let input = Layout::new((3, 4));
    let result = compute_layout::<f64>(&OpKind::MaxAxis(0, false), &[&input]).unwrap();
    assert_eq!(result.shape(), &[4]);
}

#[test]
fn max_axis_1_no_keepdim() {
    // [3,4] reduce axis 1 → [3]
    let input = Layout::new((3, 4));
    let result = compute_layout::<f64>(&OpKind::MaxAxis(1, false), &[&input]).unwrap();
    assert_eq!(result.shape(), &[3]);
}

#[test]
fn max_axis_keepdim() {
    // [3,4] reduce axis 0, keepdim=true → [1,4]
    let input = Layout::new((3, 4));
    let result = compute_layout::<f64>(&OpKind::MaxAxis(0, true), &[&input]).unwrap();
    assert_eq!(result.shape(), &[1, 4]);
}

#[test]
fn max_axis_negative() {
    // axis=-1 on [3,4] resolves to axis 1 → [3]
    let input = Layout::new((3, 4));
    let result = compute_layout::<f64>(&OpKind::MaxAxis(-1, false), &[&input]).unwrap();
    assert_eq!(result.shape(), &[3]);
}

#[test]
fn max_axis_out_of_bounds() {
    let input = Layout::new((3, 4));
    let err = compute_layout::<f64>(&OpKind::MaxAxis(5, false), &[&input]).unwrap_err();
    assert!(matches!(err, OpError::AxesOutOfBounds));
}

// ── Mean / MeanAxis layout ────────────────────────────────────────────────────

#[test]
fn mean_scalar_output() {
    let input = Layout::new((3, 4));
    let result = compute_layout::<f64>(&OpKind::Mean, &[&input]).unwrap();
    assert_eq!(result.shape(), &[1]);
}

#[test]
fn mean_axis_0_no_keepdim() {
    // [3,4] reduce axis 0 → [4]
    let input = Layout::new((3, 4));
    let result = compute_layout::<f64>(&OpKind::MeanAxis(0, false), &[&input]).unwrap();
    assert_eq!(result.shape(), &[4]);
}

#[test]
fn mean_axis_1_no_keepdim() {
    // [3,4] reduce axis 1 → [3]
    let input = Layout::new((3, 4));
    let result = compute_layout::<f64>(&OpKind::MeanAxis(1, false), &[&input]).unwrap();
    assert_eq!(result.shape(), &[3]);
}

#[test]
fn mean_axis_keepdim() {
    // [3,4] reduce axis 0, keepdim=true → [1,4]
    let input = Layout::new((3, 4));
    let result = compute_layout::<f64>(&OpKind::MeanAxis(0, true), &[&input]).unwrap();
    assert_eq!(result.shape(), &[1, 4]);
}

#[test]
fn mean_axis_negative() {
    // axis=-1 on [3,4] resolves to axis 1 → [3]
    let input = Layout::new((3, 4));
    let result = compute_layout::<f64>(&OpKind::MeanAxis(-1, false), &[&input]).unwrap();
    assert_eq!(result.shape(), &[3]);
}

#[test]
fn mean_axis_out_of_bounds() {
    let input = Layout::new((3, 4));
    let err = compute_layout::<f64>(&OpKind::MeanAxis(5, false), &[&input]).unwrap_err();
    assert!(matches!(err, OpError::AxesOutOfBounds));
}
