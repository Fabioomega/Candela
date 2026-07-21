//! Buffer-assignment classifier for the planning pass.
//!
//! [`classify`] decides, for a single node, how the executor should produce its
//! output buffer - the [`ExecKind`] - from the node's op, its already-resolved
//! inputs, the output layout, the node's position in the plan, and the live
//! [`Slot`] table. Alias and lifetime analysis are done earlier by the
//! pre-planner; this stage only picks a buffer strategy.

use std::collections::HashMap;

use crate::Layout;
use crate::tensor::backend::Backend;
use crate::tensor::graph::NodeKind;
use crate::tensor::ops::def_op::OpKind;
use crate::tensor::planner::{get_id, get_layout};

/// A live buffer tracked during planning.
///
/// `id` is the node currently owning the buffer, `len` its element count, and
/// `end` the index of the last plan step that reads it. `end = None` marks a
/// buffer that is never reclaimed - a cache output or the root result.
pub(crate) struct Slot {
    pub(crate) id: usize,
    pub(crate) len: usize,
    pub(crate) end: Option<usize>,
}

#[inline]
fn find_slot(slots: &[Slot], op_start: usize, len: usize) -> Option<usize> {
    for (i, slot) in slots.iter().enumerate() {
        if slot.len == len && slot.end.is_some_and(|e| e < op_start) {
            return Some(i);
        }
    }

    None
}

/// Whether the op at `op_location` is the last reader of `slot`.
/// This is the condition for overwriting an input in place: the op reads the buffer
/// and nothing reads it afterwards.
#[inline]
fn slot_is_last_read(slot: &Slot, op_location: usize, required_len: usize) -> bool {
    slot.end == Some(op_location) && slot.len == required_len
}

#[inline]
fn assign_slot(slots: &[Slot], op_location: usize, output_layout: &Layout) -> ExecKind {
    let slot = find_slot(slots, op_location, output_layout.len());
    slot.map_or(ExecKind::Allocate, |slot| ExecKind::UseSlot {
        slot_idx: slot,
    })
}

/// How a node's output buffer is produced at execution time.
pub(crate) enum ExecKind {
    /// Allocate a fresh buffer.
    Allocate,
    /// Reclaim a previously freed, same-size buffer - the one currently owned by
    /// `slots[slot_idx]`.
    UseSlot { slot_idx: usize },
    /// Overwrite the input at `input_idx` in place: this op is the last reader of
    /// its buffer (`slots[slot_idx]`), which is the right size.
    InPlace { slot_idx: usize, input_idx: usize },
    /// Alias the input at `input_idx` with this node's layout, performing no
    /// computation. Used when the input is an edge or cache buffer the planner
    /// never frees, so no slot lifetime needs tracking.
    ReferenceEternal { input_idx: usize },
    /// Alias the slot-backed buffer of the input at `input_idx`; the planner
    /// extends `slots[slot_idx]`'s lifetime to cover the reference.
    ReferenceSlot { slot_idx: usize, input_idx: usize },
}

/// Pick the [`ExecKind`] for a node from its op and resolved inputs.
///
/// Layout-only ops (`View`, `Slice`, `Transpose`, `TransposeAxes`, `Broadcast`,
/// `NoOp`) alias their input. `AsContiguous` aliases an already-contiguous input
/// and otherwise reuses a freed buffer or allocates one to pack into. Scalar and
/// binary element-wise ops overwrite a free same-size input in place when one
/// exists, falling back to buffer reuse or fresh allocation.
#[inline]
pub(crate) fn classify<T, B: Backend>(
    op: &OpKind<T>,
    inputs: &[&NodeKind<T, B>],
    output_layout: &Layout,
    op_location: usize,
    slots: &[Slot],
    id_slot_map: &HashMap<usize, usize>,
) -> ExecKind {
    match op {
        // References
        OpKind::Slice
        | OpKind::View
        | OpKind::Transpose
        | OpKind::TransposeAxes
        | OpKind::Broadcast
        | OpKind::NoOp => match &inputs[0] {
            NodeKind::Node(n) => match id_slot_map.get(&n.id) {
                Some(slot_idx) => ExecKind::ReferenceSlot {
                    slot_idx: *slot_idx,
                    input_idx: 0,
                },
                // No slot means the input is itself an eternal reference (a chain of
                // layout-only ops bottoming out at an edge/cache), so its buffer is
                // never reclaimed, then this op aliases it eternally too.
                None => ExecKind::ReferenceEternal { input_idx: 0 },
            },
            NodeKind::Baked(c) => match id_slot_map.get(&c.id) {
                Some(slot_idx) => ExecKind::ReferenceSlot {
                    slot_idx: *slot_idx,
                    input_idx: 0,
                },
                // No slot means the input is itself an eternal reference (a chain of
                // layout-only ops bottoming out at an edge/cache), so its buffer is
                // never reclaimed, then this op aliases it eternally too.
                None => ExecKind::ReferenceEternal { input_idx: 0 },
            },
            NodeKind::Cache(_) | NodeKind::Edge(_) | NodeKind::Slot(_) => {
                ExecKind::ReferenceEternal { input_idx: 0 }
            }
        },
        OpKind::AsContiguous => match &inputs[0] {
            NodeKind::Node(n) => {
                if n.layout().is_contiguous() {
                    match id_slot_map.get(&n.id) {
                        Some(slot_idx) => ExecKind::ReferenceSlot {
                            slot_idx: *slot_idx,
                            input_idx: 0,
                        },
                        None => ExecKind::ReferenceEternal { input_idx: 0 },
                    }
                } else {
                    assign_slot(slots, op_location, output_layout)
                }
            }
            NodeKind::Cache(c) => {
                let n = c.get_node();
                if n.layout().is_contiguous() {
                    match id_slot_map.get(&n.id) {
                        Some(slot_idx) => ExecKind::ReferenceSlot {
                            slot_idx: *slot_idx,
                            input_idx: 0,
                        },
                        None => ExecKind::ReferenceEternal { input_idx: 0 },
                    }
                } else {
                    assign_slot(slots, op_location, output_layout)
                }
            }
            NodeKind::Baked(c) => {
                if c.layout().is_contiguous() {
                    match id_slot_map.get(&c.id) {
                        Some(slot_idx) => ExecKind::ReferenceSlot {
                            slot_idx: *slot_idx,
                            input_idx: 0,
                        },
                        None => ExecKind::ReferenceEternal { input_idx: 0 },
                    }
                } else {
                    assign_slot(slots, op_location, output_layout)
                }
            }
            NodeKind::Edge(e) => {
                if e.layout().is_contiguous() {
                    ExecKind::ReferenceEternal { input_idx: 0 }
                } else {
                    assign_slot(slots, op_location, output_layout)
                }
            }
            NodeKind::Slot(s) => {
                if s.layout().is_contiguous() {
                    ExecKind::ReferenceEternal { input_idx: 0 }
                } else {
                    assign_slot(slots, op_location, output_layout)
                }
            }
        },
        // InPlace ops
        OpKind::ScalarOp(_) => {
            let id = get_id(inputs[0]);

            id_slot_map
                .get(&id)
                .filter(|&&s| slot_is_last_read(&slots[s], op_location, output_layout.len()))
                .copied()
                .map_or(assign_slot(slots, op_location, output_layout), |slot_idx| {
                    ExecKind::InPlace {
                        slot_idx,
                        input_idx: 0,
                    }
                })
        }
        OpKind::FusedScalar(_) => {
            let id = get_id(inputs[0]);

            id_slot_map
                .get(&id)
                .filter(|&&s| {
                    slot_is_last_read(&slots[s], op_location, output_layout.len())
                        && get_layout(inputs[0]).is_contiguous()
                })
                .copied()
                .map_or(assign_slot(slots, op_location, output_layout), |slot_idx| {
                    ExecKind::InPlace {
                        slot_idx,
                        input_idx: 0,
                    }
                })
        }
        OpKind::Add | OpKind::Sub | OpKind::Mul | OpKind::Div => {
            for (i, inp) in inputs.iter().enumerate() {
                if !get_layout(*inp).is_contiguous() {
                    continue;
                }

                let id = get_id(inp);

                // Guards against same node in 2 inputs (i. e. t + t).
                // In that case InPlaceIdx cannot happen as the backends needs
                // the storage Arc to be unique to be able to reuse a buffer in-place.
                if inputs
                    .iter()
                    .enumerate()
                    .any(|(j, other)| j != i && get_id(other) == id)
                {
                    continue;
                }

                let slot_idx = id_slot_map.get(&id);

                if let Some(idx) = slot_idx
                    && slot_is_last_read(&slots[*idx], op_location, output_layout.len())
                {
                    return ExecKind::InPlace {
                        slot_idx: *idx,
                        input_idx: i,
                    };
                }
            }

            assign_slot(slots, op_location, output_layout)
        }
        _ => assign_slot(slots, op_location, output_layout),
    }
}
