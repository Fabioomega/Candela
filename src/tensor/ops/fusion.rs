use crate::tensor::backend::Backend;
use crate::tensor::graph::NodeKind;
use crate::tensor::ops::def_op::{OpKind, OpKindScalar, Sign};
use crate::tensor::traits::Numeric;

///////////////////////////////////////////

pub(crate) struct Fusion<T, B: Backend> {
    pub(crate) op: OpKind<T>,
    pub(crate) inputs: Box<[NodeKind<T, B>]>,
}

#[cfg_attr(feature = "tracing", tracing::instrument(level = "trace", skip_all))]
pub(crate) fn try_fuse<T: Numeric, B: Backend>(
    op: OpKind<T>,
    inputs: Box<[NodeKind<T, B>]>,
) -> Fusion<T, B> {
    let mut current_fusion: Fusion<T, B> = Fusion {
        op,
        inputs: inputs.clone(),
    };

    for (idx, inp) in inputs.iter().enumerate() {
        match inp {
            NodeKind::Edge(_) | NodeKind::Slot(_) => continue,
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

#[inline]
pub(crate) fn fuse_scalar_op<T: Numeric, B: Backend>(
    op1: &[OpKindScalar<T>],
    inputs1: &[NodeKind<T, B>],
    op2: &[OpKindScalar<T>],
) -> Fusion<T, B> {
    let op1_last = &op1[op1.len() - 1];
    let op2_first = &op2[0];

    let fused = if let OpKindScalar::AxBy(a1, b1) = op1_last
        && let OpKindScalar::AxBy(a2, b2) = op2_first
    {
        Some(OpKindScalar::AxBy(*a1 * *a2, *a2 * *b1 + *b2))
    } else {
        None
    };

    if let Some(op) = fused {
        let b: Box<[OpKindScalar<T>]> = op1[..op1.len() - 1]
            .iter()
            .cloned()
            .chain(std::iter::once(op))
            .chain(op2[1..].iter().cloned())
            .collect();

        Fusion {
            op: OpKind::FusedScalar(b),
            inputs: inputs1.into(),
        }
    } else {
        let b: Box<[OpKindScalar<T>]> = op1.iter().cloned().chain(op2.iter().cloned()).collect();

        Fusion {
            op: OpKind::FusedScalar(b),
            inputs: inputs1.into(),
        }
    }
}

#[inline]
pub(crate) fn fuse_scalar_ops<T, B>(
    op1: &OpKind<T>,
    inputs1: &[NodeKind<T, B>],
    op2: &OpKind<T>,
) -> Fusion<T, B>
where
    T: Numeric,
    B: Backend,
{
    match (op1, op2) {
        (OpKind::ScalarOp(s1), OpKind::ScalarOp(s2)) => {
            fuse_scalar_op(std::slice::from_ref(s1), inputs1, std::slice::from_ref(s2))
        }
        (OpKind::FusedScalar(f1), OpKind::ScalarOp(s1)) => {
            fuse_scalar_op(f1, inputs1, std::slice::from_ref(s1))
        }
        (OpKind::ScalarOp(s1), OpKind::FusedScalar(f2)) => {
            fuse_scalar_op(std::slice::from_ref(s1), inputs1, f2)
        }
        (OpKind::FusedScalar(f1), OpKind::FusedScalar(f2)) => fuse_scalar_op(f1, inputs1, f2),
        _ => unreachable!(),
    }
}

#[inline]
pub(crate) fn try_fuse_matmul_ops<T, B>(
    op1: &OpKind<T>, // This is the father operand
    inputs1: &[NodeKind<T, B>],
    op2: &OpKind<T>, // This is the child operand
    inputs2: &[NodeKind<T, B>],
    skip_input_idx: usize, // Skips one of the inputs2 Nodes
) -> Option<Fusion<T, B>>
where
    T: Numeric,
    B: Backend,
{
    match (op1, op2) {
        (OpKind::MatMul(a1), OpKind::ScalarOp(op)) => {
            if let OpKindScalar::AxBy(a2, b2) = op
                && *b2 == T::SUM_NEUTRAL
            {
                Some(Fusion {
                    op: OpKind::MatMul(*a2 * *a1),
                    inputs: inputs1.into(),
                })
            } else {
                None
            }
        }
        (OpKind::MatMulSum(a1, b1, sign), OpKind::ScalarOp(op)) => {
            if let OpKindScalar::AxBy(a2, b2) = op
                && *b2 == T::SUM_NEUTRAL
            {
                Some(Fusion {
                    op: OpKind::MatMulSum(*a2 * *a1, *b1, sign.clone()),
                    inputs: inputs1.into(),
                })
            } else {
                None
            }
        }
        (OpKind::MatMul(a1), OpKind::Add) => {
            let other = 1 - skip_input_idx;
            let other = &inputs2[other];
            let node = match other {
                NodeKind::Node(node) => Some(node.as_ref()),
                NodeKind::Cache(cache) => Some(cache.get_node()),
                _ => None,
            };

            let inputs: Box<[NodeKind<T, B>]> = inputs1
                .iter()
                .cloned()
                .chain(std::iter::once(other.clone()))
                .collect();

            let mut b1 = T::MUL_NEUTRAL;

            if let Some(node) = node
                && let OpKind::ScalarOp(scalar) = &node.op
                && let OpKindScalar::AxBy(a2, b2) = scalar
                && *b2 == T::SUM_NEUTRAL
            {
                b1 = *a2;
            }

            Some(Fusion {
                op: OpKind::MatMulSum(*a1, b1, Sign::Plus),
                inputs,
            })
        }
        (OpKind::MatMul(a1), OpKind::Sub) => {
            // Only fuse when MatMul is the left operand: MatMul - C.
            // C - MatMul cannot be expressed as MatMulSum.
            if skip_input_idx != 0 {
                return None;
            }

            let other = &inputs2[1];
            let node = match other {
                NodeKind::Node(node) => Some(node.as_ref()),
                NodeKind::Cache(cache) => Some(cache.get_node()),
                _ => None,
            };

            let inputs: Box<[NodeKind<T, B>]> = inputs1
                .iter()
                .cloned()
                .chain(std::iter::once(other.clone()))
                .collect();

            let mut b1 = T::MUL_NEUTRAL;

            if let Some(node) = node
                && let OpKind::ScalarOp(scalar) = &node.op
                && let OpKindScalar::AxBy(a2, b2) = scalar
                && *b2 == T::SUM_NEUTRAL
            {
                b1 = *a2;
            }

            Some(Fusion {
                op: OpKind::MatMulSum(*a1, b1, Sign::Minus),
                inputs,
            })
        }
        _ => None,
    }
}

#[cfg(test)]
#[path = "fusion_tests.rs"]
mod tests;

#[cfg_attr(feature = "tracing", tracing::instrument(level = "trace", skip_all))]
pub(crate) fn compute_fusion<T, B>(
    op1: &OpKind<T>, // This is the father operand
    inputs1: &[NodeKind<T, B>],
    op2: &OpKind<T>, // This is the child operand
    inputs2: &[NodeKind<T, B>],
    skip_input_idx: usize, // Skips one of the inputs2 Nodes
) -> Option<Fusion<T, B>>
where
    T: Numeric,
    B: Backend,
{
    match (op1, op2) {
        (OpKind::ScalarOp(_), OpKind::ScalarOp(_))
        | (OpKind::FusedScalar(_), OpKind::ScalarOp(_))
        | (OpKind::ScalarOp(_), OpKind::FusedScalar(_))
        | (OpKind::FusedScalar(_), OpKind::FusedScalar(_)) => {
            Some(fuse_scalar_ops(op1, inputs1, op2))
        }
        (OpKind::View(_), OpKind::AsContiguous) => Some(Fusion {
            op: op1.clone(),
            inputs: inputs1.into(),
        }),
        (OpKind::MatMul(_), _) | (OpKind::MatMulSum(_, _, _), _) => {
            try_fuse_matmul_ops(op1, inputs1, op2, inputs2, skip_input_idx)
        }
        (_, OpKind::AsContiguous) => {
            let is_contiguous = match &inputs2[0] {
                NodeKind::Node(node) => node.layout().is_contiguous(),
                NodeKind::Edge(node) => node.layout().is_contiguous(),
                NodeKind::Slot(node) => node.layout().is_contiguous(),
                NodeKind::Cache(cache) => cache.get_node().layout.is_contiguous(),
            };

            if is_contiguous {
                Some(Fusion {
                    op: op1.clone(),
                    inputs: inputs1.into(),
                })
            } else {
                None
            }
        }
        _ => None,
    }
}
