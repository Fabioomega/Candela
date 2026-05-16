use crate::tensor::graph::NodeKind;
use crate::tensor::mem_formats::layout::Layout;
use crate::tensor::ops::def_op::OpKind;
use crate::tensor::planner::get_id;
use crate::tensor::planner::plan::Slot;
use std::collections::HashMap;

#[inline]
fn slot_is_free(slot: &Slot, op_location: usize, required_len: usize) -> bool {
    slot.end
        .map_or(false, |e| e < op_location && slot.len == required_len)
}

/// Find an input whose buffer can be overwritten in-place for this op.
///
/// Returns `(Some(slot_idx), input_idx)` when an input's slot is free at `op_location`
/// and has the same length as `output_layout`. The caller uses `slot_idx` to update the
/// slot's owner and `input_idx` as the `InPlaceIdx` output kind.
///
/// Returns `(None, 0)` when no in-place reuse is possible — the caller falls through to
/// redirect detection, buffer reuse, or fresh allocation.
#[cfg_attr(
    feature = "tracing",
    tracing::instrument(
        level = "trace",
        skip(op, inputs, output_layout, slots, id_slot_map),
        fields(slots_count = slots.len())
    )
)]
#[inline]
pub(crate) fn find_buffer_inplace<T: Copy>(
    op: &OpKind<T>,
    inputs: &[NodeKind<T>],
    output_layout: &Layout,
    op_location: usize,
    slots: &[Slot],
    id_slot_map: &HashMap<usize, usize>,
) -> (Option<usize>, usize) {
    let result = match op {
        OpKind::ScalarOp(_) => {
            let id = get_id(&inputs[0]);
            let slot_idx = id_slot_map
                .get(&id)
                .filter(|&&s| slot_is_free(&slots[s], op_location, output_layout.len()))
                .copied();

            (slot_idx, 0)
        }
        OpKind::FusedScalar(scalars) => match scalars[0] {
            _ => {
                let id = get_id(&inputs[0]);
                let slot_idx = id_slot_map
                    .get(&id)
                    .filter(|&&s| slot_is_free(&slots[s], op_location, output_layout.len()))
                    .copied();

                (slot_idx, 0)
            }
        },
        OpKind::Add | OpKind::Sub | OpKind::Mul | OpKind::Div => {
            for (i, inp) in inputs.iter().enumerate() {
                let id = get_id(inp);
                let slot_idx = id_slot_map.get(&id);

                if let Some(idx) = slot_idx
                    && slot_is_free(&slots[*idx], op_location, output_layout.len())
                {
                    return (Some(*idx), i);
                }
            }

            (None, 0)
        }
        OpKind::Slice(_)
        | OpKind::View(_)
        | OpKind::Transpose
        | OpKind::TransposeAxes(_)
        | OpKind::Broadcast(_)
        | OpKind::NoOp => (None, 0),
        _ => (None, 0),
    };

    #[cfg(feature = "tracing")]
    tracing::trace!(
        found_slot = result.0.is_some(),
        slot_idx = result.0,
        input_idx = result.1,
    );

    result
}
