use std::ops::{Add, Div, Mul, Neg, Sub};
use std::str::Matches;

use crate::tensor::definitions::NumberLike;
use crate::tensor::graph::NodeKind;
use crate::tensor::ops::def_op::{OpKind, OpKindScalar};

///////////////////////////////////////////

#[derive(Debug)]
pub(crate) struct Fusion<T: Copy> {
    pub(crate) op: OpKind<T>,
    pub(crate) inputs: Box<[NodeKind<T>]>,
}

#[cfg_attr(feature = "tracing", tracing::instrument(level = "trace", skip_all))]
pub fn try_fuse<T: NumberLike>(op: OpKind<T>, inputs: Box<[NodeKind<T>]>) -> Fusion<T> {
    let mut current_fusion: Fusion<T> = Fusion {
        op,
        inputs: inputs.clone(),
    };

    for (idx, inp) in inputs.iter().enumerate() {
        match inp {
            NodeKind::Edge(_) => continue,
            NodeKind::Node(node) => {
                let fused = compute_fusion(
                    &node.op,
                    &node.inputs,
                    &current_fusion.op,
                    &current_fusion.inputs,
                    idx,
                );

                if let Some(f) = fused {
                    current_fusion = f;
                }
            }
            NodeKind::Cache(cache) => {
                let node = cache.get_node();

                let fused = compute_fusion(
                    &node.op,
                    &node.inputs,
                    &current_fusion.op,
                    &current_fusion.inputs,
                    idx,
                );

                if let Some(f) = fused {
                    current_fusion = f;
                }
            }
        }
    }

    current_fusion
}

fn fuse_sum_scalar<T: NumberLike>(
    op1: &OpKindScalar<T>, // Parent
    inputs1: &[NodeKind<T>],
    op2: &OpKindScalar<T>, // Child
) -> Fusion<T> {
    let s1: T = match op1 {
        OpKindScalar::Sum(scalar) => *scalar,
        OpKindScalar::Sub(scalar) => -*scalar,
        _ => unreachable!("no other op should appear here"),
    };

    let s2: T = match op2 {
        OpKindScalar::Sum(scalar) => *scalar,
        OpKindScalar::Sub(scalar) => -*scalar,
        _ => unreachable!("no other op should appear here"),
    };

    Fusion {
        op: OpKind::ScalarOp(OpKindScalar::Sum(s1 + s2)),
        inputs: inputs1.into(),
    }
}

fn fuse_mul_scalar<T: NumberLike>(
    op1: &OpKindScalar<T>, // Parent
    inputs1: &[NodeKind<T>],
    op2: &OpKindScalar<T>, // Child
) -> Fusion<T> {
    match op1 {
        OpKindScalar::Mul(s1) => match op2 {
            OpKindScalar::Mul(s2) => Fusion {
                op: OpKind::ScalarOp(OpKindScalar::Mul(*s1 * *s2)),
                inputs: inputs1.into(),
            },
            OpKindScalar::Div(s2) => Fusion {
                op: OpKind::ScalarOp(OpKindScalar::Mul(*s1 / *s2)),
                inputs: inputs1.into(),
            },
            _ => unreachable!("no other op should appear here"),
        },
        OpKindScalar::Div(s1) => match op2 {
            OpKindScalar::Mul(s2) => Fusion {
                op: OpKind::ScalarOp(OpKindScalar::Mul(*s2 / *s1)),
                inputs: inputs1.into(),
            },
            OpKindScalar::Div(s2) => Fusion {
                op: OpKind::ScalarOp(OpKindScalar::Div(*s1 * *s2)),
                inputs: inputs1.into(),
            },
            _ => unreachable!("no other op should appear here"),
        },
        _ => unreachable!("no other op should appear here"),
    }
}

#[inline]
fn fuse_scalars_into_combination<T: NumberLike>(
    op1: &OpKindScalar<T>,
    inputs1: &[NodeKind<T>],
    op2: &OpKindScalar<T>,
) -> Fusion<T> {
    let ops = Box::new([op1.clone(), op2.clone()]);
    Fusion {
        op: OpKind::FusedScalar(ops),
        inputs: inputs1.into(),
    }
}

fn fuse_scalars<T: NumberLike>(
    op1: &OpKindScalar<T>,
    inputs1: &[NodeKind<T>],
    op2: &OpKindScalar<T>,
) -> Fusion<T> {
    match op1 {
        OpKindScalar::Sum(_) => match op2 {
            OpKindScalar::Sum(_) => fuse_sum_scalar(op1, inputs1, op2),
            OpKindScalar::Sub(_) => fuse_sum_scalar(op1, inputs1, op2),
            _ => fuse_scalars_into_combination(op1, inputs1, op2),
        },
        OpKindScalar::Sub(_) => match op2 {
            OpKindScalar::Sum(_) => fuse_sum_scalar(op1, inputs1, op2),
            OpKindScalar::Sub(_) => fuse_sum_scalar(op1, inputs1, op2),
            _ => fuse_scalars_into_combination(op1, inputs1, op2),
        },
        OpKindScalar::Mul(_) => match op2 {
            OpKindScalar::Mul(_) => fuse_mul_scalar(op1, inputs1, op2),
            OpKindScalar::Div(_) => fuse_mul_scalar(op1, inputs1, op2),
            _ => fuse_scalars_into_combination(op1, inputs1, op2),
        },
        OpKindScalar::Div(_) => match op2 {
            OpKindScalar::Mul(_) => fuse_mul_scalar(op1, inputs1, op2),
            OpKindScalar::Div(_) => fuse_mul_scalar(op1, inputs1, op2),
            _ => fuse_scalars_into_combination(op1, inputs1, op2),
        },
    }
}

fn fuse_scalar_combination<T: NumberLike>(
    ops: &[OpKindScalar<T>],
    inputs1: &[NodeKind<T>],
    op2: &OpKindScalar<T>,
) -> Fusion<T> {
    let tail = &ops[ops.len() - 1];
    let fused = fuse_scalars(tail, inputs1, op2);

    let op = fused.op;
    let inputs = fused.inputs;

    let new_ops = match op {
        OpKind::FusedScalar(_) => {
            let mut vec: Vec<OpKindScalar<T>> = Vec::with_capacity(ops.len() + 1);
            vec.extend(ops[..ops.len() - 1].iter().cloned());
            vec.push(op2.clone());

            vec.into_boxed_slice()
        }
        OpKind::ScalarOp(scalar_op) => {
            let mut vec: Vec<OpKindScalar<T>> = Vec::with_capacity(ops.len());
            vec.extend(ops[..ops.len() - 1].iter().cloned());
            vec.push(scalar_op.clone());

            vec.into_boxed_slice()
        }
        _ => unreachable!("no other op should appear here"),
    };

    Fusion {
        op: OpKind::FusedScalar(new_ops),
        inputs,
    }
}

#[cfg_attr(feature = "tracing", tracing::instrument(level = "trace", skip_all))]
pub fn compute_fusion<T>(
    op1: &OpKind<T>, // This is the father operand
    inputs1: &[NodeKind<T>],
    op2: &OpKind<T>, // This is the child operand
    inputs2: &[NodeKind<T>],
    skip_input_idx: usize, // Skips one of the inputs2 Nodes
) -> Option<Fusion<T>>
where
    T: NumberLike,
{
    match op1 {
        OpKind::ScalarOp(s1) => match op2 {
            OpKind::ScalarOp(s2) => Some(fuse_scalars(s1, inputs1, s2)),
            _ => None,
        },
        OpKind::FusedScalar(ops) => match op2 {
            OpKind::ScalarOp(s2) => Some(fuse_scalar_combination(ops, inputs1, s2)),
            _ => None,
        },
        OpKind::View(_) => match op2 {
            OpKind::AsContiguous => Some(Fusion {
                op: op1.clone(),
                inputs: inputs1.into(),
            }),
            _ => None,
        },

        _ => None,
    }
}
