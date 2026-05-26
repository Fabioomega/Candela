//! Static execution planner.
//!
//! [`plan_computation`] analyses the computation graph once and returns a
//! `Vec<ComputeKind>` that tells the executor exactly what to run, which
//! buffer to write into, and what to free after each step.

use std::collections::HashMap;
use std::fmt::Debug;

use crate::tensor::backend::Backend;
use crate::tensor::graph::{NodeKind, TensorGraphCacheNode, TensorGraphEdge, TensorGraphNode};
use crate::tensor::planner::alias::{self, AliasKind, AliasMap};
use crate::tensor::planner::get_id;
use crate::tensor::planner::sort::topological_sort;

/// How the executor should produce the output buffer for a single operation.
#[derive(Debug)]
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
pub(crate) enum ComputeKind<'a, T, B: Backend> {
    Leaf {
        edge: &'a TensorGraphEdge<T, B>,
    },
    /// A regular computation node.
    Op {
        node: &'a TensorGraphNode<T, B>,
        output: OutputKind,
        /// Input node IDs resolved at plan time. Index `i` is the
        /// `computation_cache` key for `node.inputs[i]`.
        resolved_inputs: Vec<usize>,
        /// Node IDs whose buffers should be dropped from the live-buffer cache
        /// immediately after this step completes.
        dealloc_after: Vec<usize>,
    },
    /// A cached computation node. The executor checks the cache before running;
    /// if already filled it inserts the cached result and cleans up any reserved
    /// buffers.
    CachedOp {
        cache: &'a TensorGraphCacheNode<T, B>,
        output: OutputKind,
        /// Input node IDs resolved at plan time. Same semantics as `Op`.
        resolved_inputs: Vec<usize>,
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

#[derive(Debug)]
struct OpPlan<'a, T, B: Backend> {
    node: &'a NodeKind<T, B>,
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

/// Resolve each input's raw node ID through the current redirect map.
/// Called immediately before emitting a plan step so the node being planned
/// never sees its own redirect entry (which is inserted after the push).
#[inline]
fn build_resolved_inputs<T, B: Backend>(
    inputs: &[NodeKind<T, B>],
    id_redirect: &HashMap<usize, usize>,
) -> Vec<usize> {
    inputs
        .iter()
        .map(|inp| {
            let id = get_id(inp);
            *id_redirect.get(&id).unwrap_or(&id)
        })
        .collect()
}

#[cfg_attr(
    feature = "tracing",
    tracing::instrument(
        level = "trace",
        skip(node, plan, slots, id_slot_map, ref_deallocs, alias_map),
        fields(node_id = node.id, output_len = node.layout.len(), slots_available = slots.len())
    )
)]
#[inline]
fn plan_node<'a, T, B: Backend>(
    op_start: usize,
    mut op_end: Option<usize>,
    node: &'a TensorGraphNode<T, B>,
    plan: &mut Vec<ComputeKind<'a, T, B>>,
    slots: &mut Vec<Slot>,
    id_slot_map: &mut HashMap<usize, usize>,
    ref_deallocs: &mut Vec<(usize, Option<usize>)>,
    alias_map: &mut AliasMap<'_, T, B>,
) {
    let (inplace_slot, input_idx) = find_buffer_inplace(
        &node.op,
        &node.inputs,
        &node.layout,
        op_start,
        slots,
        id_slot_map,
        id_redirect,
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

        let resolved_inputs = build_resolved_inputs(&node.inputs, id_redirect);
        plan.push(ComputeKind::Op {
            node,
            output: OutputKind::InPlaceIdx(input_idx),
            resolved_inputs,
            dealloc_after: Vec::new(),
        });

        slots[slot_idx].id = node.id;

        return;
    }

    match is_a_reference(&node.op, &node.inputs, &id_slot_map) {
        ReferenceKind::Edge(input_idx) => {
            let resolved_inputs = build_resolved_inputs(&node.inputs, id_redirect);
            plan.push(ComputeKind::Op {
                node,
                output: OutputKind::InPlaceIdx(input_idx),
                resolved_inputs,
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

            let resolved_inputs = build_resolved_inputs(&node.inputs, id_redirect);
            plan.push(ComputeKind::Op {
                node,
                output: OutputKind::InPlaceIdx(input_idx),
                resolved_inputs,
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

        let resolved_inputs = build_resolved_inputs(&node.inputs, id_redirect);
        plan.push(ComputeKind::Op {
            node,
            output: OutputKind::Buffer(slots[slot_idx].id),
            resolved_inputs,
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

        let resolved_inputs = build_resolved_inputs(&node.inputs, id_redirect);
        plan.push(ComputeKind::Op {
            node,
            output: OutputKind::Allocate(node.layout.len()),
            resolved_inputs,
            dealloc_after: Vec::new(),
        });
    }

    // Activate the redirect after the step is emitted so this node's own
    // resolved_inputs used the pre-redirect state.
    if let Some(id) = pending_redirect_from {
        id_redirect.insert(id, node.id);
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
        skip(cache, plan, slots, id_slot_map, ref_deallocs, alias_map),
        fields(
            node_id = cache.get_node().id,
            output_len = cache.get_node().layout.len(),
            cache_filled = cache.is_cache_filled(),
            slots_available = slots.len()
        )
    )
)]
fn plan_cache_node<'a, T, B: Backend>(
    op_start: usize,
    cache: &'a TensorGraphCacheNode<T, B>,
    plan: &mut Vec<ComputeKind<'a, T, B>>,
    slots: &mut Vec<Slot>,
    id_slot_map: &mut HashMap<usize, usize>,
    ref_deallocs: &mut Vec<(usize, Option<usize>)>,
    alias_map: &mut AliasMap<'_, T, B>,
) {
    let node = cache.get_node();

    if cache.is_cache_filled() {
        #[cfg(feature = "tracing")]
        tracing::trace!(
            decision = "cache_hit",
            "cache already filled at plan time, skipping computation"
        );

        let resolved_inputs = build_resolved_inputs(&node.inputs, id_redirect);
        plan.push(ComputeKind::CachedOp {
            cache,
            output: OutputKind::Allocate(0),
            resolved_inputs,
            dealloc_after: Vec::new(),
        });
        return;
    }

    let (inplace_slot, input_idx) = find_buffer_inplace(
        &node.op,
        &node.inputs,
        &node.layout,
        op_start,
        slots,
        id_slot_map,
        id_redirect,
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

        let resolved_inputs = build_resolved_inputs(&node.inputs, id_redirect);
        plan.push(ComputeKind::CachedOp {
            cache,
            output: OutputKind::InPlaceIdx(input_idx),
            resolved_inputs,
            dealloc_after: Vec::new(),
        });

        return;
    }

    match is_a_reference(&node.op, &node.inputs, &id_slot_map, id_redirect) {
        ReferenceKind::Edge(input_idx) => {
            let resolved_inputs = build_resolved_inputs(&node.inputs, id_redirect);
            plan.push(ComputeKind::CachedOp {
                cache,
                output: OutputKind::InPlaceIdx(input_idx),
                resolved_inputs,
                dealloc_after: Vec::new(),
            });

            return;
        }
        ReferenceKind::Slot(slot_idx, input_idx) => {
            slots[slot_idx].end = None;
            ref_deallocs.push((node.id, None));

            let resolved_inputs = build_resolved_inputs(&node.inputs, id_redirect);
            plan.push(ComputeKind::CachedOp {
                cache,
                output: OutputKind::InPlaceIdx(input_idx),
                resolved_inputs,
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

        slots[slot_idx].end = None;

        let resolved_inputs = build_resolved_inputs(&node.inputs, id_redirect);
        plan.push(ComputeKind::CachedOp {
            cache,
            output: OutputKind::Buffer(slots[slot_idx].id),
            resolved_inputs,
            dealloc_after: Vec::new(),
        });
    } else {
        #[cfg(feature = "tracing")]
        tracing::trace!(
            decision = "allocate",
            len = node.layout.len(),
            "no free slot found, will allocate new buffer for the cache"
        );

        let resolved_inputs = build_resolved_inputs(&node.inputs, id_redirect);
        plan.push(ComputeKind::CachedOp {
            cache,
            output: OutputKind::Allocate(node.layout.len()),
            resolved_inputs,
            dealloc_after: Vec::new(),
        });
    }
}

/// The output of [`plan_computation`].
///
/// Contains the ordered execution schedule. All redirect resolution is done at
/// plan time — each step's `resolved_inputs` holds the concrete `computation_cache`
/// keys the executor should use.
pub(crate) struct Plan<'a, T, B: Backend> {
    /// Ordered list of steps to execute. Each step carries its [`OutputKind`],
    /// pre-resolved input IDs, and the list of buffer IDs to drop once the step
    /// completes.
    pub(crate) plan: Vec<ComputeKind<'a, T, B>>,
}

/// Build a static execution plan for the subgraph rooted at `base_node`.
///
/// This is called once per `.materialize()` invocation. It performs a topological
/// sort, analyses buffer lifetimes, and assigns each node an [`OutputKind`] that
/// tells the executor whether to allocate a new buffer, reuse a freed one, or
/// write in-place into an input.
///
/// All redirect resolution (deduplication of [`OpKind::AsContiguous`] nodes) is
/// performed here. Each plan step's `resolved_inputs` contains the concrete
/// `computation_cache` IDs to use.
///
/// Leaf tensors (graph inputs) appear as [`ComputeKind::Leaf`] steps. The executor
/// inserts them into `computation_cache` before any computation runs so all steps
/// resolve inputs uniformly by ID.
///
/// The plan is in dependency order and includes the root node as its last element.
/// See [doc/planner.md] for a full walkthrough of the algorithm.
///
/// [doc/planner.md]: https://github.com/Fabioomega/candela/blob/main/doc/planner.md
/// [`OpKind::AsContiguous`]: crate::tensor::ops::def_op::OpKind::AsContiguous
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
pub(crate) fn plan_computation<T, B: Backend>(base_node: &TensorGraphNode<T, B>) -> Plan<'_, T, B> {
    let dag_iter = topological_sort(base_node);

    let mut plan: Vec<ComputeKind<'_, T, B>> = Vec::with_capacity(32);
    let mut ops: Vec<OpPlan<'_, T, B>> = Vec::with_capacity(32);
    let mut slots: Vec<Slot> = Vec::with_capacity(32);
    let mut id_op: HashMap<usize, usize> = HashMap::with_capacity(32);
    let mut id_slot_map: HashMap<usize, usize> = HashMap::with_capacity(32);
    let mut alias_map: AliasMap<'_, T, B> = AliasMap::new();
    let mut ref_deallocs: Vec<(usize, Option<usize>)> = Vec::with_capacity(8);

    for node in dag_iter {
        match node {
            NodeKind::Edge(e) => {
                // Edges are leaves — give them a position in id_op so compute
                // nodes can find them when tracking lifetimes, but their end
                // stays None (never deallocated).
                id_op.insert(e.id, ops.len());
                ops.push(OpPlan { node, end: None });
            }
            NodeKind::Node(n) => {
                match alias::handle_alias(&node, n.id, &n.op, &n.inputs, &mut alias_map) {
                    AliasKind::NoAlias | AliasKind::OwningAlias => {
                        id_op.insert(n.id, ops.len());
                        ops.push(OpPlan { node, end: None });

                        for inp in n.inputs.iter() {
                            if let Some(&op_idx) = id_op.get(&get_id(inp)) {
                                ops[op_idx].end = Some(ops.len() - 1);
                            }
                        }
                    }
                    AliasKind::Alias => {}
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
            NodeKind::Edge(e) => {
                plan.push(ComputeKind::Leaf { edge: e });
            }
            NodeKind::Node(arc_node) => {
                plan_node(
                    i,
                    op.end,
                    arc_node,
                    &mut plan,
                    &mut slots,
                    &mut id_slot_map,
                    &mut ref_deallocs,
                    &mut alias_map,
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
                    &mut alias_map,
                );
            }
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
        &mut alias_map,
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
            ComputeKind::Leaf { .. } => unreachable!(),
        }
    }

    for slot in slots.into_iter() {
        let Some(end) = slot.end else { continue };

        match &mut plan[end] {
            ComputeKind::Op { dealloc_after, .. } => dealloc_after.push(slot.id),
            ComputeKind::CachedOp { dealloc_after, .. } => dealloc_after.push(slot.id),
            ComputeKind::Leaf { .. } => unreachable!(),
        }
    }

    Plan { plan }
}
