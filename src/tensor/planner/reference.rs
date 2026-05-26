use std::collections::HashMap;

use crate::tensor::backend::Backend;
use crate::tensor::graph::NodeKind;
use crate::tensor::ops::def_op::OpKind;
use crate::tensor::planner::{get_id, get_red_id};

/// Outcome of [`is_a_reference`] for a single node.
///
/// Layout-only ops (`View`, `Slice`, `Transpose`, `Broadcast`, `NoOp`) produce no new
/// data — they alias an existing buffer. The planner uses this enum to decide how to
/// extend the aliased slot's lifetime and which input index to pass as the in-place
/// output.
pub(crate) enum ReferenceKind {
    /// The aliased input is a graph leaf (`TensorGraphEdge`). Pass input index 0 as the
    /// in-place output; no slot lifetime to extend.
    Edge(usize),
    /// The aliased input is a computed node whose buffer occupies `slot_idx`. Pass input
    /// index `input_idx` as the in-place output and extend the slot's lifetime to cover
    /// all aliases.
    Slot(usize, usize),
    /// This op is not a layout-only alias; handle it normally.
    NoRef,
}

/// Classify `op` as a layout-only alias or not.
///
/// Returns `Edge` or `Slot` for ops that share an existing buffer without computing new
/// data (`View`, `Slice`, `Transpose`, `Broadcast`, `NoOp`). Returns `NoRef` for
/// everything else.
#[inline]
pub(crate) fn is_a_reference<T, B: Backend>(
    op: &OpKind<T>,
    inputs: &[NodeKind<T, B>],
    id_slot_map: &HashMap<usize, usize>,
) -> ReferenceKind {
    match op {
        OpKind::Slice(_)
        | OpKind::View(_)
        | OpKind::Transpose
        | OpKind::TransposeAxes(_)
        | OpKind::Broadcast(_) => match &inputs[0] {
            NodeKind::Node(_) | NodeKind::Cache(_) => {
                let id = get_id(&inputs[0]);

                id_slot_map
                    .get(&id)
                    .map_or(ReferenceKind::NoRef, |slot_idx| {
                        ReferenceKind::Slot(*slot_idx, 0)
                    })
            }
            NodeKind::Edge(_) => ReferenceKind::Edge(0),
        },
        _ => ReferenceKind::NoRef,
    }
}
