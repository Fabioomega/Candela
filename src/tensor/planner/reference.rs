use std::collections::HashMap;

use crate::tensor::graph::NodeKind;
use crate::tensor::ops::def_op::OpKind;

pub enum ReferenceKind {
    Edge(usize),
    Slot(usize, usize),
    NoRef,
}

#[inline]
pub fn is_a_reference<T: Copy>(
    op: &OpKind<T>,
    inputs: &[NodeKind<T>],
    id_slot_map: &HashMap<usize, usize>,
) -> ReferenceKind {
    match op {
        OpKind::Slice(_)
        | OpKind::View(_)
        | OpKind::TransposeAxes(_)
        | OpKind::Transpose
        | OpKind::NoOp => match &inputs[0] {
            NodeKind::Node(node) => id_slot_map
                .get(&node.id)
                .map_or(ReferenceKind::NoRef, |slot_idx| {
                    ReferenceKind::Slot(*slot_idx, 0)
                }),
            NodeKind::Cache(cache) => id_slot_map
                .get(&cache.get_node().id)
                .map_or(ReferenceKind::NoRef, |slot_idx| {
                    ReferenceKind::Slot(*slot_idx, 0)
                }),
            NodeKind::Edge(_) => ReferenceKind::Edge(0),
        },
        _ => ReferenceKind::NoRef,
    }
}
