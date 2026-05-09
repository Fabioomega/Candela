//! Static execution planner.
//!
//! [`plan_computation`] analyses the computation graph once and returns a
//! `Vec<ComputeKind>` that tells the executor exactly what to run, which
//! buffer to write into, and what to free after each step.

use std::collections::HashMap;

use crate::tensor::graph::{NodeKind, TensorGraphCacheNode, TensorGraphNode};
use crate::tensor::planner::get_id;
use crate::tensor::planner::in_place::find_buffer_inplace;
use crate::tensor::planner::reference::{ReferenceKind, is_a_reference};
use crate::tensor::planner::sort::topological_sort;

/// How the executor should produce the output buffer for a single operation.
pub(crate) enum OutputKind {
    /// Re-use the buffer previously owned by node `id`. The planner guarantees
    /// that buffer is no longer referenced by any live node at this point.
    Buffer(usize),
    /// Overwrite input at position `idx` in-place. The planner guarantees the
    /// input's buffer is not aliased by any other live node.
    InPlaceIdx(usize),
    /// Allocate a fresh `Vec<T>` of this length.
    Allocate(usize),
}

/// One step in the execution plan produced by [`plan_computation`].
pub(crate) enum ComputeKind<'a, T: Copy> {
    /// A regular computation node.
    Op {
        node: &'a TensorGraphNode<T>,
        output: OutputKind,
        /// Node IDs whose buffers should be dropped from the live-buffer cache
        /// immediately after this step completes.
        dealloc_after: Vec<usize>,
    },
    /// A cached computation node. The executor checks the cache before running;
    /// if already filled it inserts the cached result and cleans up any reserved
    /// buffers.
    CachedOp {
        cache: &'a TensorGraphCacheNode<T>,
        output: OutputKind,
        dealloc_after: Vec<usize>,
    },
}

/// A buffer slot tracked by the planner.
///
/// Each live buffer in the plan is represented by one slot. `end` is the index
/// of the last plan step that reads this buffer; after that step the executor
/// drops it. `end = None` means the buffer lives forever — this is used for
/// cached nodes whose results must survive across separate `.materialize()` calls.
pub(crate) struct Slot {
    id: usize,
    pub(crate) len: usize,
    pub(crate) end: Option<usize>,
}

struct OpPlan<'a, T: Copy> {
    node: &'a NodeKind<T>,
    end: Option<usize>,
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
fn extend_slot_life(slot_end1: Option<usize>, slot_end2: Option<usize>) -> Option<usize> {
    slot_end1.map_or(None, |e1| slot_end2.map_or(None, |e2| Some(e1.max(e2))))
}

#[cfg_attr(
    feature = "tracing",
    tracing::instrument(
        level = "trace",
        skip(node, plan, slots, id_slot_map, ref_deallocs),
        fields(node_id = node.id, output_len = node.layout.len(), slots_available = slots.len())
    )
)]
#[inline]
fn plan_node<'a, T: Copy>(
    op_start: usize,
    op_end: Option<usize>,
    node: &'a TensorGraphNode<T>,
    plan: &mut Vec<ComputeKind<'a, T>>,
    slots: &mut Vec<Slot>,
    id_slot_map: &mut HashMap<usize, usize>,
    ref_deallocs: &mut Vec<(usize, Option<usize>)>,
) {
    let (inplace_slot, input_idx) = find_buffer_inplace(
        &node.op,
        &node.inputs,
        &node.layout,
        op_start,
        slots,
        id_slot_map,
    );

    if let Some(slot_idx) = inplace_slot {
        #[cfg(feature = "tracing")]
        tracing::trace!(
            decision = "inplace",
            slot_idx,
            input_idx,
            reused_node_id = slots[slot_idx].id,
            "planned in-place reuse of input buffer"
        );

        id_slot_map.insert(node.id, slot_idx);
        slots[slot_idx].end = op_end;

        plan.push(ComputeKind::Op {
            node,
            output: OutputKind::InPlaceIdx(input_idx),
            dealloc_after: Vec::new(),
        });

        slots[slot_idx].id = node.id;

        return;
    }

    match is_a_reference(&node.op, &node.inputs, &id_slot_map) {
        ReferenceKind::Edge(input_idx) => {
            plan.push(ComputeKind::Op {
                node,
                output: OutputKind::InPlaceIdx(input_idx),
                dealloc_after: Vec::new(),
            });

            return;
        }
        ReferenceKind::Slot(slot_idx, input_idx) => {
            let source_node_id = slots[slot_idx].id;
            let extended_end = extend_slot_life(slots[slot_idx].end, op_end);
            slots[slot_idx].end = extended_end;
            id_slot_map.insert(node.id, slot_idx);
            ref_deallocs.push((node.id, extended_end));

            #[cfg(feature = "tracing")]
            tracing::trace!(
                decision = "reference",
                slot_idx,
                input_idx,
                source_node_id,
                ?extended_end,
                "planned reference op; extended slot lifetime to cover all aliases"
            );

            plan.push(ComputeKind::Op {
                node,
                output: OutputKind::InPlaceIdx(input_idx),
                dealloc_after: Vec::new(),
            });

            return;
        }
        ReferenceKind::NoRef => {}
    }

    let slot = find_slot(slots, op_start, node.layout.len());
    if let Some(slot_idx) = slot {
        #[cfg(feature = "tracing")]
        tracing::trace!(
            decision = "buffer_reuse",
            slot_idx,
            reused_node_id = slots[slot_idx].id,
            slot_len = slots[slot_idx].len,
            "planned reuse of a free same-size buffer"
        );

        id_slot_map.insert(node.id, slot_idx);
        slots[slot_idx].end = op_end;
        plan.push(ComputeKind::Op {
            node,
            output: OutputKind::Buffer(slots[slot_idx].id),
            dealloc_after: Vec::new(),
        });

        slots[slot_idx].id = node.id;
    } else {
        #[cfg(feature = "tracing")]
        tracing::trace!(
            decision = "allocate",
            len = node.layout.len(),
            new_slot_idx = slots.len(),
            "no free slot found, will allocate new buffer"
        );

        id_slot_map.insert(node.id, slots.len());
        slots.push(Slot {
            id: node.id,
            len: node.layout.len(),
            end: op_end,
        });

        plan.push(ComputeKind::Op {
            node,
            output: OutputKind::Allocate(node.layout.len()),
            dealloc_after: Vec::new(),
        });
    }
}

// The planner must assume that during some computation / before the planner finished planning
//  the cache may have been filled by another thread and cannot assume the current state of the cache.
// If that was not the case, it's possible to dumb the executor even further, as it would not have
//  to check the state of the cache before taking a decision.
#[cfg_attr(
    feature = "tracing",
    tracing::instrument(
        level = "trace",
        skip(cache, plan, slots, id_slot_map, ref_deallocs),
        fields(
            node_id = cache.get_node().id,
            output_len = cache.get_node().layout.len(),
            cache_filled = cache.is_cache_filled(),
            slots_available = slots.len()
        )
    )
)]
fn plan_cache_node<'a, T: Copy>(
    op_start: usize,
    cache: &'a TensorGraphCacheNode<T>,
    plan: &mut Vec<ComputeKind<'a, T>>,
    slots: &mut Vec<Slot>,
    id_slot_map: &mut HashMap<usize, usize>,
    ref_deallocs: &mut Vec<(usize, Option<usize>)>,
) {
    if cache.is_cache_filled() {
        #[cfg(feature = "tracing")]
        tracing::trace!(
            decision = "cache_hit",
            "cache already filled at plan time, skipping computation"
        );

        plan.push(ComputeKind::CachedOp {
            cache,
            output: OutputKind::Allocate(0),
            dealloc_after: Vec::new(),
        });
        return;
    }

    let node = cache.get_node();
    let (inplace_slot, input_idx) = find_buffer_inplace(
        &node.op,
        &node.inputs,
        &node.layout,
        op_start,
        slots,
        id_slot_map,
    );

    if let Some(slot_idx) = inplace_slot {
        #[cfg(feature = "tracing")]
        tracing::trace!(
            decision = "inplace",
            slot_idx,
            input_idx,
            reused_node_id = slots[slot_idx].id,
            "planned in-place reuse; slot will be kept alive for the cache"
        );

        // id_slot_map.insert(node.id, slot_idx);
        slots[slot_idx].end = None;

        plan.push(ComputeKind::CachedOp {
            cache,
            output: OutputKind::InPlaceIdx(input_idx),
            dealloc_after: Vec::new(),
        });

        return;
    }

    match is_a_reference(&node.op, &node.inputs, &id_slot_map) {
        ReferenceKind::Edge(input_idx) => {
            plan.push(ComputeKind::CachedOp {
                cache,
                output: OutputKind::InPlaceIdx(input_idx),
                dealloc_after: Vec::new(),
            });

            return;
        }
        ReferenceKind::Slot(slot_idx, input_idx) => {
            slots[slot_idx].end = extend_slot_life(slots[slot_idx].end, None);
            ref_deallocs.push((node.id, None));

            plan.push(ComputeKind::CachedOp {
                cache,
                output: OutputKind::InPlaceIdx(input_idx),
                dealloc_after: Vec::new(),
            });

            return;
        }
        ReferenceKind::NoRef => {}
    }

    let slot = find_slot(slots, op_start, node.layout.len());
    if let Some(slot_idx) = slot {
        #[cfg(feature = "tracing")]
        tracing::trace!(
            decision = "buffer_reuse",
            slot_idx,
            reused_node_id = slots[slot_idx].id,
            slot_len = slots[slot_idx].len,
            "planned buffer reuse; slot will be kept alive for the cache"
        );

        // id_slot_map.insert(node.id, slot_idx);
        slots[slot_idx].end = None;

        plan.push(ComputeKind::CachedOp {
            cache,
            output: OutputKind::Buffer(slots[slot_idx].id),
            dealloc_after: Vec::new(),
        });
    } else {
        #[cfg(feature = "tracing")]
        tracing::trace!(
            decision = "allocate",
            len = node.layout.len(),
            "no free slot found, will allocate new buffer for the cache"
        );

        // id_slot_map.insert(node.id, slots.len());
        // slots.push(Slot {
        //     id: node.id,
        //     len: node.layout.len(),
        //     end: None,
        // });

        plan.push(ComputeKind::CachedOp {
            cache,
            output: OutputKind::Allocate(node.layout.len()),
            dealloc_after: Vec::new(),
        });
    }
}

/// Build a static execution plan for the subgraph rooted at `base_node`.
///
/// This is called once per `.materialize()` invocation. It performs a topological
/// sort, analyses buffer lifetimes, and assigns each node an [`OutputKind`] that
/// tells the executor whether to allocate a new buffer, reuse a freed one, or
/// write in-place into an input.
///
/// The returned `Vec<ComputeKind>` is in dependency order and includes the root
/// node as its last element. See [doc/planner.md] for a full walkthrough of the
/// algorithm.
///
/// [doc/planner.md]: https://github.com/Fabioomega/candela/blob/main/doc/planner.md
// TODO: Add a planner that is very dumbed down and don't waste so much processing on planning
// TODO: That is specially useful for small computations where this planning time is significant
#[cfg_attr(
    feature = "tracing",
    tracing::instrument(
        level = "debug",
        skip(base_node),
        fields(
            node_id = base_node.id,
            ops_count = tracing::field::Empty,
            slots_count = tracing::field::Empty,
            dealloc_edges = tracing::field::Empty,
            ref_deallocs_count = tracing::field::Empty
        )
    )
)]
pub(crate) fn plan_computation<T: Copy>(base_node: &TensorGraphNode<T>) -> Vec<ComputeKind<'_, T>> {
    let dag_iter = topological_sort(base_node);

    let mut plan: Vec<ComputeKind<'_, T>> = Vec::with_capacity(32);
    let mut ops: Vec<OpPlan<'_, T>> = Vec::with_capacity(32);
    let mut slots: Vec<Slot> = Vec::with_capacity(32);
    let mut id_op: HashMap<usize, usize> = HashMap::with_capacity(32);
    let mut id_slot_map: HashMap<usize, usize> = HashMap::with_capacity(32);
    let mut ref_deallocs: Vec<(usize, Option<usize>)> = Vec::with_capacity(8);

    for node in dag_iter {
        match node {
            NodeKind::Edge(_) => {}
            NodeKind::Node(n) => {
                id_op.insert(n.id, ops.len());
                ops.push(OpPlan { node, end: None });

                for inp in n.inputs.iter() {
                    if let Some(&op_idx) = id_op.get(&get_id(inp)) {
                        ops[op_idx].end = Some(ops.len() - 1);
                    }
                }
            }
            NodeKind::Cache(cache) => {
                let n = cache.get_node();
                id_op.insert(n.id, ops.len());
                ops.push(OpPlan { node, end: None });

                for inp in n.inputs.iter() {
                    if let Some(&op_idx) = id_op.get(&get_id(inp)) {
                        ops[op_idx].end = Some(ops.len() - 1);
                    }
                }
            }
        }
    }

    let ops_len = ops.len();

    for (i, op) in ops.into_iter().enumerate() {
        match op.node {
            NodeKind::Node(arc_node) => {
                plan_node(
                    i,
                    op.end,
                    arc_node,
                    &mut plan,
                    &mut slots,
                    &mut id_slot_map,
                    &mut ref_deallocs,
                );
            }
            NodeKind::Cache(arc_cache) => {
                plan_cache_node(
                    i,
                    arc_cache,
                    &mut plan,
                    &mut slots,
                    &mut id_slot_map,
                    &mut ref_deallocs,
                );
            }
            NodeKind::Edge(_) => unreachable!(),
        }
    }

    plan_node(
        ops_len,
        None,
        &base_node,
        &mut plan,
        &mut slots,
        &mut id_slot_map,
        &mut ref_deallocs,
    );

    #[cfg(feature = "tracing")]
    {
        let dealloc_edges = slots.iter().filter(|s| s.end.is_some()).count();
        tracing::Span::current().record("ops_count", plan.len());
        tracing::Span::current().record("slots_count", slots.len());
        tracing::Span::current().record("dealloc_edges", dealloc_edges);
        tracing::Span::current().record("ref_deallocs_count", ref_deallocs.len());
    }

    for (node_id, dealloc_at) in &ref_deallocs {
        let Some(end) = dealloc_at else { continue };
        match &mut plan[*end] {
            ComputeKind::Op { dealloc_after, .. } | ComputeKind::CachedOp { dealloc_after, .. } => {
                dealloc_after.push(*node_id)
            }
        }
    }

    for slot in slots.into_iter() {
        let Some(end) = slot.end else { continue };

        match &mut plan[end] {
            ComputeKind::Op { dealloc_after, .. } => dealloc_after.push(slot.id),
            ComputeKind::CachedOp { dealloc_after, .. } => dealloc_after.push(slot.id),
        }
    }

    plan
}
