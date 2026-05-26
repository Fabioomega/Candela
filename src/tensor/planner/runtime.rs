use std::collections::HashMap;

use crate::Layout;
use crate::tensor::backend::Backend;
use crate::tensor::graph::NodeKind;
use crate::tensor::ops::def_op::OpKind;
use crate::tensor::planner::alias::AliasMap;
use crate::tensor::planner::{get_id, get_red_id};

pub(crate) struct Slot {
    id: usize,
    pub(crate) len: usize,
    pub(crate) end: Option<usize>,
}

#[inline]
fn find_slot(slots: &[Slot], op_start: usize, len: usize) -> Option<usize> {
    for (i, slot) in slots.iter().enumerate() {
        if slot.len == len && slot.end.map_or(false, |e| e < op_start) {
            return Some(i);
        }
    }

    None
}

#[inline]
fn slot_is_free(slot: &Slot, op_location: usize, required_len: usize) -> bool {
    slot.end
        .map_or(false, |e| e < op_location && slot.len == required_len)
}

pub(crate) enum ExecKind {
    Allocate,
    UseSlot { slot_idx: usize },
    InPlace { slot_idx: usize, inputs_idx: usize },
    ReferenceEdge { inputs_idx: usize },
    ReferenceSlot { slot_idx: usize, inputs_idx: usize },
}

#[inline]
pub(crate) fn classify<T, B: Backend>(
    op: &OpKind<T>,
    inputs: &[NodeKind<T, B>],
    output_layout: &Layout,
    op_location: usize,
    slots: &[Slot],
    id_slot_map: &HashMap<usize, usize>,
    alias_map: &AliasMap<'_, T, B>,
) -> ExecKind {
    match op {
        // References
        OpKind::Slice(_)
        | OpKind::View(_)
        | OpKind::Transpose
        | OpKind::TransposeAxes(_)
        | OpKind::Broadcast(_) => match alias_map.resolve(&inputs[0]) {
            NodeKind::Node(n) => {
                let slot_idx = id_slot_map.get(&n.id);

                if let Some(slot_idx) = slot_idx {
                    ExecKind::ReferenceSlot {
                        slot_idx: *slot_idx,
                        inputs_idx: 0,
                    }
                } else {
                    unreachable!("This should never happen unless the planner fucked up.")
                }
            }
            NodeKind::Cache(c) => {
                let n = c.get_node();
                let slot_idx = id_slot_map.get(&n.id);

                if let Some(slot_idx) = slot_idx {
                    ExecKind::ReferenceSlot {
                        slot_idx: *slot_idx,
                        inputs_idx: 0,
                    }
                } else {
                    unreachable!("This should never happen unless the planner fucked up.")
                }
            }
            NodeKind::Edge(_) => ExecKind::ReferenceEdge { inputs_idx: 0 },
        },
        // InPlace ops
        OpKind::ScalarOp(_) => {
            let node = alias_map.resolve(&inputs[0]);
            let id = get_id(node);

            id_slot_map
                .get(&id)
                .filter(|&&s| slot_is_free(&slots[s], op_location, output_layout.len()))
                .copied()
                .map_or(ExecKind::Allocate, |slot_idx| ExecKind::InPlace {
                    slot_idx,
                    inputs_idx: 0,
                })
        }
        OpKind::FusedScalar(scalars) => match scalars[0] {
            _ => {
                let node = alias_map.resolve(&inputs[0]);
                let id = get_id(node);

                id_slot_map
                    .get(&id)
                    .filter(|&&s| slot_is_free(&slots[s], op_location, output_layout.len()))
                    .copied()
                    .map_or(ExecKind::Allocate, |slot_idx| ExecKind::InPlace {
                        slot_idx,
                        inputs_idx: 0,
                    })
            }
        },
        OpKind::Add | OpKind::Sub | OpKind::Mul | OpKind::Div => {
            for (i, inp) in inputs.iter().enumerate() {
                let node = alias_map.resolve(inp);
                let id = get_id(node);
                let slot_idx = id_slot_map.get(&id);

                if let Some(idx) = slot_idx
                    && slot_is_free(&slots[*idx], op_location, output_layout.len())
                {
                    return ExecKind::InPlace {
                        slot_idx: *idx,
                        inputs_idx: i,
                    };
                }
            }

            ExecKind::Allocate
        }
        _ => ExecKind::Allocate,
    }
}
