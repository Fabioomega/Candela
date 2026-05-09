use crate::tensor::definitions::NumberLike;
use crate::tensor::graph::NodeKind;
use crate::tensor::ops::def_op::{OpKind, OpKindScalar};
use crate::tensor::traits::Promising;

///////////////////////////////////////////

#[derive(Debug)]
pub(crate) struct Fusion<T: Copy> {
    pub(crate) op: OpKind<T>,
    pub(crate) inputs: Box<[NodeKind<T>]>,
}

#[cfg_attr(feature = "tracing", tracing::instrument(level = "trace", skip_all))]
pub(crate) fn try_fuse<T: NumberLike>(op: OpKind<T>, inputs: Box<[NodeKind<T>]>) -> Fusion<T> {
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

#[inline]
pub(crate) fn fuse_scalar_op<T: NumberLike>(
    op1: &[OpKindScalar<T>],
    inputs1: &[NodeKind<T>],
    op2: &[OpKindScalar<T>],
) -> Fusion<T> {
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
pub(crate) fn fuse_scalar_ops<T>(op1: &OpKind<T>, inputs1: &[NodeKind<T>], op2: &OpKind<T>) -> Fusion<T>
where
    T: NumberLike,
{
    match (op1, op2) {
        (OpKind::ScalarOp(s1), OpKind::ScalarOp(s2)) => {
            fuse_scalar_op(&[s1.clone()], inputs1, &[s2.clone()])
        }
        (OpKind::FusedScalar(f1), OpKind::ScalarOp(s1)) => {
            fuse_scalar_op(&f1, inputs1, &[s1.clone()])
        }
        (OpKind::ScalarOp(s1), OpKind::FusedScalar(f2)) => {
            fuse_scalar_op(&[s1.clone()], inputs1, f2)
        }
        (OpKind::FusedScalar(f1), OpKind::FusedScalar(f2)) => fuse_scalar_op(f1, inputs1, f2),
        _ => unreachable!(),
    }
}

#[cfg_attr(feature = "tracing", tracing::instrument(level = "trace", skip_all))]
pub(crate) fn compute_fusion<T>(
    op1: &OpKind<T>, // This is the father operand
    inputs1: &[NodeKind<T>],
    op2: &OpKind<T>, // This is the child operand
    inputs2: &[NodeKind<T>],
    skip_input_idx: usize, // Skips one of the inputs2 Nodes
) -> Option<Fusion<T>>
where
    T: NumberLike,
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
        (OpKind::NoOp, OpKind::NoOp) => Some(Fusion {
            op: op1.clone(),
            inputs: inputs1.into(),
        }),
        (_, OpKind::AsContiguous) => {
            let is_contiguous = match &inputs1[0] {
                NodeKind::Node(node) => node.layout.is_contiguous(),
                NodeKind::Edge(node) => node.layout().is_contiguous(),
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
