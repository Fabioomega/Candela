use super::*;
use crate::tensor::backend::DefaultBackend;
use crate::tensor::graph::TensorGraphEdge;
use crate::tensor::ops::def_op::Sign;
use crate::tensor::storage::TensorData;
use std::sync::Arc;

fn edge(val: f64, shape: &[usize]) -> NodeKind<f64, DefaultBackend> {
    NodeKind::Edge(Arc::new(TensorGraphEdge::from_tensor_data(
        TensorData::from_scalar(val, shape),
    )))
}

fn axby(a: f64, b: f64) -> OpKind<f64> {
    OpKind::ScalarOp(OpKindScalar::AxBy(a, b))
}

fn assert_fused_scalar(op: &OpKind<f64>, expected_len: usize) -> &[OpKindScalar<f64>] {
    match op {
        OpKind::FusedScalar(ops) => {
            assert_eq!(ops.len(), expected_len, "FusedScalar length mismatch");
            ops
        }
        _ => panic!("expected FusedScalar, got {:?}", op.as_str()),
    }
}

fn assert_axby(op: &OpKindScalar<f64>, expected_a: f64, expected_b: f64) {
    match op {
        OpKindScalar::AxBy(a, b) => {
            assert!(
                (*a - expected_a).abs() < 1e-12,
                "a: expected {expected_a}, got {a}"
            );
            assert!(
                (*b - expected_b).abs() < 1e-12,
                "b: expected {expected_b}, got {b}"
            );
        }
        _ => panic!("expected AxBy"),
    }
}

// ── scalar fusion ─────────────────────────────────────────────────────────────

#[test]
fn axby_axby_fused_constants() {
    // (a=2, b=1) then (a=3, b=4) → AxBy(2*3, 3*1+4) = AxBy(6, 7)
    let input = edge(1.0, &[4]);
    let fusion = compute_fusion(
        &axby(2.0, 1.0),
        std::slice::from_ref(&input),
        &axby(3.0, 4.0),
        std::slice::from_ref(&input),
        0,
    );
    let result = fusion.unwrap();
    let ops = assert_fused_scalar(&result.op, 1);
    assert_axby(&ops[0], 6.0, 7.0);
}

#[test]
fn axby_then_exp() {
    let input = edge(1.0, &[4]);
    let fusion = compute_fusion(
        &axby(2.0, 0.0),
        std::slice::from_ref(&input),
        &OpKind::ScalarOp(OpKindScalar::Exp),
        std::slice::from_ref(&input),
        0,
    );
    let result = fusion.unwrap();
    let ops = assert_fused_scalar(&result.op, 2);
    assert!(matches!(ops[0], OpKindScalar::AxBy(_, _)));
    assert!(matches!(ops[1], OpKindScalar::Exp));
}

#[test]
fn exp_then_axby() {
    let input = edge(1.0, &[4]);
    let exp_op = OpKind::ScalarOp(OpKindScalar::Exp);
    let fusion = compute_fusion(
        &exp_op,
        std::slice::from_ref(&input),
        &axby(2.0, 0.0),
        std::slice::from_ref(&input),
        0,
    );
    let result = fusion.unwrap();
    let ops = assert_fused_scalar(&result.op, 2);
    assert!(matches!(ops[0], OpKindScalar::Exp));
    assert!(matches!(ops[1], OpKindScalar::AxBy(_, _)));
}

#[test]
fn non_scalar_no_fusion() {
    let input = edge(1.0, &[4]);
    let fusion = compute_fusion(
        &OpKind::Add,
        &[input.clone(), input.clone()],
        &OpKind::Mul,
        &[input.clone(), input.clone()],
        0,
    );
    assert!(fusion.is_none());
}

// ── MatMulSum fusion ──────────────────────────────────────────────────────────

#[test]
fn matmul_plus_bias() {
    // MatMul(alpha=2) + C → MatMulSum(alpha=2, beta=1, Plus)
    // inputs2[0] is the matmul slot (skip_input_idx=0); inputs2[1] is the bias.
    let a = edge(1.0, &[2, 3]);
    let b = edge(1.0, &[3, 4]);
    let c = edge(0.0, &[2, 4]);
    let fusion = compute_fusion(
        &OpKind::MatMul(2.0),
        &[a.clone(), b.clone()],
        &OpKind::Add,
        &[edge(0.0, &[2, 4]), c.clone()], // [matmul_placeholder, bias]
        0,
    );
    let result = fusion.unwrap();
    match result.op {
        OpKind::MatMulSum(alpha, beta, sign) => {
            assert!((alpha - 2.0).abs() < 1e-12);
            assert!((beta - 1.0).abs() < 1e-12); // MUL_NEUTRAL since bias is a plain edge
            assert!(matches!(sign, Sign::Plus));
        }
        _ => panic!("expected MatMulSum, got {:?}", result.op.as_str()),
    }
    assert_eq!(result.inputs.len(), 3); // [A, B, C]
}

#[test]
fn matmul_minus_bias() {
    // MatMul - C (left operand) → MatMulSum(..., Minus)
    // C - MatMul (right operand) must NOT fuse.
    let a = edge(1.0, &[2, 3]);
    let b = edge(1.0, &[3, 4]);
    let c = edge(0.0, &[2, 4]);

    let fusion = compute_fusion(
        &OpKind::MatMul(2.0),
        &[a.clone(), b.clone()],
        &OpKind::Sub,
        &[edge(0.0, &[2, 4]), c.clone()], // matmul is left operand
        0,
    );
    let result = fusion.unwrap();
    match result.op {
        OpKind::MatMulSum(alpha, beta, sign) => {
            assert!((alpha - 2.0).abs() < 1e-12);
            assert!((beta - 1.0).abs() < 1e-12);
            assert!(matches!(sign, Sign::Minus));
        }
        _ => panic!("expected MatMulSum, got {:?}", result.op.as_str()),
    }

    // C - MatMul cannot be expressed as MatMulSum; fusion must return None
    let no_fusion = compute_fusion(
        &OpKind::MatMul(2.0),
        &[a.clone(), b.clone()],
        &OpKind::Sub,
        &[c.clone(), edge(0.0, &[2, 4])], // matmul is right operand
        1,
    );
    assert!(no_fusion.is_none());
}

#[test]
fn matmulsum_then_axby() {
    // MatMulSum(alpha=2, beta=1.5, Plus) followed by AxBy(3, 0)
    // → MatMulSum(alpha=3*2=6, beta=1.5 unchanged, Plus)
    let a = edge(1.0, &[2, 3]);
    let b = edge(1.0, &[3, 4]);
    let c = edge(0.0, &[2, 4]);
    let fusion = compute_fusion(
        &OpKind::MatMulSum(2.0, 1.5, Sign::Plus),
        &[a.clone(), b.clone(), c.clone()],
        &axby(3.0, 0.0), // b2=0 == SUM_NEUTRAL for f64
        &[edge(0.0, &[2, 4])],
        0,
    );
    let result = fusion.unwrap();
    match result.op {
        OpKind::MatMulSum(alpha, beta, sign) => {
            assert!((alpha - 6.0).abs() < 1e-12);
            assert!((beta - 1.5).abs() < 1e-12);
            assert!(matches!(sign, Sign::Plus));
        }
        _ => panic!("expected MatMulSum, got {:?}", result.op.as_str()),
    }
    assert_eq!(result.inputs.len(), 3); // inputs unchanged: [A, B, C]
}
