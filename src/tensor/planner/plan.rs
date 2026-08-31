//! Static execution planner.
//!
//! [`plan_computation`] analyses the computation graph once and returns a [`Plan`]
//! that tells the executor exactly what to run, which buffer to write into, what to
//! free after each step, and which buffer holds the final result.

use std::fmt::Debug;
use std::sync::Arc;

use fx_hash::{FxHashMap, FxHashMapExt};

use crate::tensor::backend::Backend;
use crate::tensor::graph::{
    NodeKind, TensorGraphBaked, TensorGraphCacheNode, TensorGraphEdge, TensorGraphNode,
};
use crate::tensor::planner::alias::{self, AliasKind, AliasMap};
use crate::tensor::planner::packing::{PackedSlots, alignment_of, greedy_offset_pack_slots};
use crate::tensor::planner::runtime::{ExecKind, Slot};
use crate::tensor::planner::sort::topological_sort;
use crate::tensor::planner::{get_id, runtime};

/// How the executor should produce the output buffer for a single operation.
#[derive(Debug, Clone)]
pub(crate) enum OutputKind {
    /// A region on an arena with start at `offset` of length `len`.
    Region { offset: usize, len: usize },
    /// Overwrite input at position `idx` in-place and, if `idx` is a reference,
    /// also removes the parent with `parent_id`. The planner guarantees the input's buffer is not
    /// aliased by any other live node.
    InPlaceIdx { idx: usize, parent_id: usize },
    /// Alias the input at position `idx` at this node's layout, copying no
    /// elements. The executor clones the input's handle and re-points its layout;
    /// the input keeps its buffer ownership and stays in the live-buffer cache, so
    /// nothing is allocated, freed, or renamed.
    Reference(usize),
    /// Allocate a fresh `Vec<T>` of this length.
    Allocate(usize),
}

/// One step in the execution plan produced by [`plan_computation`].
pub(crate) enum ComputeKind<'a, T, B: Backend> {
    Leaf {
        edge: &'a Arc<TensorGraphEdge<T, B>>,
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
        cache: &'a Arc<TensorGraphCacheNode<T, B>>,
        output: OutputKind,
        /// Input node IDs resolved at plan time. Same semantics as `Op`.
        resolved_inputs: Vec<usize>,
        dealloc_after: Vec<usize>,
    },
    Baked {
        baked: &'a Arc<TensorGraphBaked<T, B>>,
        arena_offset: usize,
        resolved_inputs: Vec<usize>,
        dealloc_after: Vec<usize>,
    },
}

/// A node staged by the pre-planner: the node itself, its inputs already resolved
/// through the alias map *at the node's position in the sort*, and `end` - the
/// index of the last step that reads its output, or `None` if never reclaimed.
#[derive(Debug)]
pub(crate) struct OpPlan<'a, T, B: Backend> {
    pub(crate) node: &'a NodeKind<T, B>,
    pub(crate) resolved_inputs: Vec<&'a NodeKind<T, B>>,
    pub(crate) end: Option<usize>,
}

#[inline]
fn extend_slot_life(slot_end1: Option<usize>, slot_end2: Option<usize>) -> Option<usize> {
    slot_end1.and_then(|e1| slot_end2.map(|e2| e1.max(e2)))
}

/// Project the input references resolved in [`pre_plan`] to their
/// `computation_cache` ids.
#[inline]
fn build_resolved_inputs<T, B: Backend>(resolved_inputs: &[&NodeKind<T, B>]) -> Vec<usize> {
    resolved_inputs.iter().map(|inp| get_id(*inp)).collect()
}

#[inline]
fn resolve_inputs<'a, T, B: Backend>(
    inputs: &'a [NodeKind<T, B>],
    alias_map: &AliasMap<'a, T, B>,
) -> Vec<&'a NodeKind<T, B>> {
    inputs.iter().map(|i| alias_map.resolve(i)).collect()
}

#[inline]
fn track_lifetimes<T, B: Backend>(
    resolved: &[&NodeKind<T, B>],
    pos: usize,
    id_op: &FxHashMap<usize, usize>,
    ops: &mut [OpPlan<'_, T, B>],
) {
    for inp in resolved {
        if let Some(&op_idx) = id_op.get(&get_id(inp)) {
            ops[op_idx].end = Some(pos);
        }
    }
}

/// Mutable accumulator state for the buffer-assignment pass. The `plan_*` methods
/// read and extend these four collections in lockstep as they walk the staged ops:
/// `plan` is the schedule under construction, `slots` tracks reusable buffers,
/// `id_slot_map` maps node ids to the slot holding their output, and `ref_deallocs`
/// records reference nodes whose buffers are reclaimed at a later step.
struct PlanState<'a, T, B: Backend> {
    plan: Vec<ComputeKind<'a, T, B>>,
    slots: Vec<Slot>,
    id_slot_map: FxHashMap<usize, usize>,
    dealloc_after: Vec<(usize, Option<usize>)>,
}

impl<'a, T, B: Backend> PlanState<'a, T, B> {
    fn with_capacity(capacity: usize) -> Self {
        let mut id_slot_map = FxHashMap::new();
        id_slot_map.reserve(capacity);

        Self {
            plan: Vec::with_capacity(capacity),
            slots: Vec::with_capacity(capacity / 2),
            id_slot_map,
            dealloc_after: Vec::with_capacity(capacity / 2),
        }
    }

    #[cfg_attr(
        feature = "tracing",
        tracing::instrument(
            level = "trace",
            skip(self, node, resolved_inputs),
            fields(
                node_id = node.id,
                output_len = node.layout.len(),
                slots_available = self.slots.len()
            )
        )
    )]
    #[inline]
    fn plan_node(
        &mut self,
        op_start: usize,
        op_end: Option<usize>,
        node: &'a TensorGraphNode<T, B>,
        resolved_inputs: &[&NodeKind<T, B>],
    ) {
        match runtime::classify(
            &node.op,
            resolved_inputs,
            &node.layout,
            op_start,
            &self.slots,
            &self.id_slot_map,
        ) {
            ExecKind::Allocate => {
                self.id_slot_map.insert(node.id, self.slots.len());
                self.slots.push(Slot {
                    id: node.id,
                    len: node.layout.len(),
                    start: op_start,
                    end: op_end,
                });

                let resolved_inputs = build_resolved_inputs(resolved_inputs);
                self.plan.push(ComputeKind::Op {
                    node,
                    output: OutputKind::Allocate(node.layout.len()),
                    resolved_inputs,
                    dealloc_after: Vec::new(),
                });
            }
            ExecKind::InPlace {
                slot_idx,
                input_idx,
            } => {
                self.id_slot_map.insert(node.id, slot_idx);
                self.slots[slot_idx].end = op_end;

                let resolved_inputs = build_resolved_inputs(resolved_inputs);
                self.plan.push(ComputeKind::Op {
                    node,
                    output: OutputKind::InPlaceIdx {
                        idx: input_idx,
                        parent_id: {
                            if self.slots[slot_idx].id == resolved_inputs[input_idx] {
                                usize::MAX
                            } else {
                                self.slots[slot_idx].id
                            }
                        },
                    },
                    resolved_inputs,
                    dealloc_after: Vec::new(),
                });

                self.slots[slot_idx].id = node.id;
            }
            ExecKind::ReferenceEternal { input_idx } => {
                let resolved_inputs = build_resolved_inputs(resolved_inputs);
                self.plan.push(ComputeKind::Op {
                    node,
                    output: OutputKind::Reference(input_idx),
                    resolved_inputs,
                    dealloc_after: Vec::new(),
                });
            }
            ExecKind::ReferenceSlot {
                slot_idx,
                input_idx,
            } => {
                let extended_end = extend_slot_life(self.slots[slot_idx].end, op_end);
                self.slots[slot_idx].end = extended_end;
                self.id_slot_map.insert(node.id, slot_idx);
                self.dealloc_after.push((node.id, extended_end));

                let resolved_inputs = build_resolved_inputs(resolved_inputs);
                self.plan.push(ComputeKind::Op {
                    node,
                    output: OutputKind::Reference(input_idx),
                    resolved_inputs,
                    dealloc_after: Vec::new(),
                });
            }
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
            skip(self, cache, resolved_inputs),
            fields(
                node_id = cache.get_node().id,
                output_len = cache.get_node().layout.len(),
                cache_filled = cache.is_cache_filled(),
                slots_available = self.slots.len()
            )
        )
    )]
    fn plan_cache_node(
        &mut self,
        op_start: usize,
        cache: &'a Arc<TensorGraphCacheNode<T, B>>,
        resolved_inputs: &[&NodeKind<T, B>],
    ) {
        let node = cache.get_node();

        if cache.is_cache_filled() {
            let resolved_inputs = build_resolved_inputs(resolved_inputs);
            self.plan.push(ComputeKind::CachedOp {
                cache,
                output: OutputKind::Allocate(0),
                resolved_inputs,
                dealloc_after: Vec::new(),
            });
            return;
        }

        match runtime::classify(
            &node.op,
            resolved_inputs,
            &node.layout,
            op_start,
            &self.slots,
            &self.id_slot_map,
        ) {
            ExecKind::Allocate => {
                let resolved_inputs = build_resolved_inputs(resolved_inputs);
                self.plan.push(ComputeKind::CachedOp {
                    cache,
                    output: OutputKind::Allocate(node.layout.len()),
                    resolved_inputs,
                    dealloc_after: Vec::new(),
                });
            }
            ExecKind::InPlace {
                slot_idx,
                input_idx,
            } => {
                self.slots[slot_idx].end = None;

                let resolved_inputs = build_resolved_inputs(resolved_inputs);
                self.plan.push(ComputeKind::CachedOp {
                    cache,
                    output: OutputKind::InPlaceIdx {
                        idx: input_idx,
                        parent_id: {
                            if self.slots[slot_idx].id == resolved_inputs[input_idx] {
                                usize::MAX
                            } else {
                                self.slots[slot_idx].id
                            }
                        },
                    },
                    resolved_inputs,
                    dealloc_after: Vec::new(),
                });

                // We don't need to change the id because it will never be read as it's an eternal
                // so it cannot be changed from this point onwards.
                // self.slots[slot_idx].id = node.id;
            }
            ExecKind::ReferenceEternal { input_idx } => {
                let resolved_inputs = build_resolved_inputs(resolved_inputs);
                self.plan.push(ComputeKind::CachedOp {
                    cache,
                    output: OutputKind::Reference(input_idx),
                    resolved_inputs,
                    dealloc_after: Vec::new(),
                });
            }
            ExecKind::ReferenceSlot {
                slot_idx,
                input_idx,
            } => {
                self.slots[slot_idx].end = None;
                self.dealloc_after.push((node.id, None));

                let resolved_inputs = build_resolved_inputs(resolved_inputs);
                self.plan.push(ComputeKind::CachedOp {
                    cache,
                    output: OutputKind::Reference(input_idx),
                    resolved_inputs,
                    dealloc_after: Vec::new(),
                });
            }
        }
    }
}

/// The root, resolved against the completed alias map. `id` is the
/// `computation_cache` key the result lands under - the root's own id, or, when the
/// root is a pure alias, the id of the node it resolves to. `resolved_inputs` is
/// used only when the root is planned as its own step (`id == base_node.id`).
struct RootNode<'a, T, B: Backend> {
    id: usize,
    resolved_inputs: Vec<&'a NodeKind<T, B>>,
}

struct PrePlan<'a, T, B: Backend> {
    pre_plan: Vec<OpPlan<'a, T, B>>,
    root: RootNode<'a, T, B>,
    /// Inputs that need to be added by an external source for the plan to run
    external_inputs: Vec<usize>,
}

/// Topologically sort the graph and, in one walk, classify each node's aliasing,
/// snapshot its resolved inputs, and record buffer lifetimes. Returns the staged
/// [`OpPlan`]s and the resolved [`RootNode`]. The alias map is built and consumed
/// entirely here; the buffer-assignment pass never sees it.
fn pre_plan<'a, T: PartialEq + Clone, B: Backend>(
    base_node: &'a TensorGraphNode<T, B>,
) -> PrePlan<'a, T, B> {
    let dag_iter = topological_sort(base_node);
    let mut id_op: FxHashMap<usize, usize> = FxHashMap::new();
    id_op.reserve(32);
    let mut ops: Vec<OpPlan<'_, T, B>> = Vec::with_capacity(32);
    let mut alias_map: AliasMap<'_, T, B> = AliasMap::new();
    let mut external_inputs: Vec<usize> = Vec::with_capacity(8);

    for node in dag_iter {
        match node {
            NodeKind::Edge(e) => {
                // Edges are leaves - give them a position in id_op so compute
                // nodes can find them when tracking lifetimes, but their end
                // stays None (never deallocated).
                id_op.insert(e.id, ops.len());
                ops.push(OpPlan {
                    node,
                    resolved_inputs: Vec::new(),
                    end: None,
                });
            }
            NodeKind::Slot(s) => {
                // Slots produce no plan step - their buffer arrives from outside - so
                // they're recorded as external inputs and deliberately kept out of
                // `ops`.
                external_inputs.push(s.id);
            }
            NodeKind::Node(n) => match alias::classify(&n.op, &n.inputs, &alias_map) {
                AliasKind::NoAlias => {
                    let resolved_inputs = resolve_inputs(&n.inputs, &alias_map);
                    let pos = ops.len();
                    id_op.insert(n.id, pos);

                    track_lifetimes(&resolved_inputs, pos, &id_op, &mut ops);

                    ops.push(OpPlan {
                        node,
                        resolved_inputs,
                        end: None,
                    });
                }
                AliasKind::Takeover(parent, tag) => {
                    let resolved_inputs = resolve_inputs(&n.inputs, &alias_map);
                    let pos = ops.len();
                    id_op.insert(n.id, ops.len());

                    track_lifetimes(&resolved_inputs, pos, &id_op, &mut ops);

                    ops.push(OpPlan {
                        node,
                        resolved_inputs,
                        end: None,
                    });

                    alias_map.takeover(parent, node, tag);
                }
                AliasKind::Alias(target, tag) => {
                    alias_map.insert(n.id, target, tag);
                }
            },
            NodeKind::Cache(cache) => {
                let n = cache.get_node();

                match alias::classify_cache(&n.inputs, &alias_map) {
                    AliasKind::Alias(target, tag) => {
                        alias_map.insert(n.id, target, tag);
                    }
                    AliasKind::Takeover(old_owner, tag) => {
                        let resolved_inputs = resolve_inputs(&n.inputs, &alias_map);
                        let pos = ops.len();

                        id_op.insert(n.id, ops.len());

                        track_lifetimes(&resolved_inputs, pos, &id_op, &mut ops);

                        ops.push(OpPlan {
                            node,
                            resolved_inputs,
                            end: None,
                        });

                        alias_map.takeover(old_owner, node, tag);
                    }
                    _ => unreachable!("classify_cache always aliases or takes over"),
                }
            }
            NodeKind::Baked(baked) => {
                let resolved_inputs = resolve_inputs(&baked.inputs, &alias_map);
                let pos = ops.len();

                id_op.insert(baked.id, pos);

                track_lifetimes(&resolved_inputs, pos, &id_op, &mut ops);

                ops.push(OpPlan {
                    node,
                    resolved_inputs,
                    end: None,
                });
            }
        }
    }

    let root_resolved = resolve_inputs(&base_node.inputs, &alias_map);

    let root_id = match alias::classify(&base_node.op, &base_node.inputs, &alias_map) {
        AliasKind::Alias(target, _) => {
            // Root is a pure alias: the result IS the target's buffer. Force it to
            // live to the end so the executor's final lookup finds it and pass 2
            // never reuses its slot.
            let id = get_id(target);
            if let Some(&op_idx) = id_op.get(&id) {
                ops[op_idx].end = None;
            }
            id
        }

        _ => {
            let root_pos = ops.len();
            track_lifetimes(&root_resolved, root_pos, &id_op, &mut ops);
            base_node.id
        }
    };

    PrePlan {
        pre_plan: ops,
        root: RootNode {
            id: root_id,
            resolved_inputs: root_resolved,
        },
        external_inputs,
    }
}

/// The output of [`plan_computation`]: the ordered execution schedule plus the id
/// the final result is stored under.
///
/// All alias resolution is done at plan time - each step's `resolved_inputs` holds
/// the concrete `computation_cache` keys the executor reads.
pub(crate) struct Plan<'a, T, B: Backend> {
    /// Steps in dependency order. Each carries its [`OutputKind`], pre-resolved
    /// input IDs, and the list of buffer IDs to drop once the step completes.
    pub(crate) plan: Vec<ComputeKind<'a, T, B>>,
    /// `computation_cache` key holding the root result - the root node's id, or the
    /// resolved target when the root is a pure alias and emits no step of its own.
    pub(crate) root_id: usize,
    /// Inputs that need to be added by an external source for the plan to run
    pub(crate) external_inputs: Vec<usize>,
    /// The size of the arena necessary to hold this plan
    pub(crate) arena_size: usize,
}

/// Build a static execution plan for the subgraph rooted at `base_node`.
///
/// Called once per `.materialize()` invocation. The pre-planner ([`pre_plan`])
/// topologically sorts the graph, classifies aliases, snapshots each node's
/// resolved inputs, and records buffer lifetimes; this function then assigns each
/// node an [`OutputKind`] - allocate, reuse a freed buffer, write in-place, or
/// alias an input - and fills the `dealloc_after` lists.
///
/// Alias resolution (deduplication and claiming of [`OpKind::AsContiguous`] and
/// `NoOp` nodes) is baked into each step's `resolved_inputs`. Leaf tensors appear
/// as [`ComputeKind::Leaf`] steps; the executor inserts them into
/// `computation_cache` before any computation runs so all steps resolve inputs
/// uniformly by ID.
///
/// Steps are in dependency order. The root is the last step unless it is a pure
/// alias, in which case it emits no step and [`Plan::root_id`] names the buffer
/// holding the result. See [doc/planner.md] for a full walkthrough of the algorithm.
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
#[inline]
pub(crate) fn plan_computation<T: PartialEq + Clone, B: Backend>(
    base_node: &TensorGraphNode<T, B>,
) -> Plan<'_, T, B> {
    let PrePlan {
        pre_plan,
        root,
        external_inputs,
    } = pre_plan(base_node);
    let mut state: PlanState<'_, T, B> = PlanState::with_capacity(pre_plan.len());

    let ops_len = pre_plan.len();

    for (i, op) in pre_plan.into_iter().enumerate() {
        match op.node {
            NodeKind::Edge(e) => {
                state.plan.push(ComputeKind::Leaf { edge: e });
            }
            NodeKind::Node(node) => {
                state.plan_node(i, op.end, node, &op.resolved_inputs);
            }
            NodeKind::Cache(cache) => {
                state.plan_cache_node(i, cache, &op.resolved_inputs);
            }
            NodeKind::Baked(baked) => {
                // Ask the packer for some memory, this does not count the root node which lives "forever".
                state.slots.push(Slot {
                    id: baked.id,
                    len: baked.plan.arena_size,
                    start: i,
                    end: Some(i),
                });

                // This dealloc the output of a baked plan not the slot itself which is free after 1 step.
                state.dealloc_after.push((baked.id, op.end));

                // TODO: Maybe we can find a way to make baked inject it's root into the parent arena
                state.plan.push(ComputeKind::Baked {
                    baked,
                    resolved_inputs: op.resolved_inputs.iter().map(|n| get_id(*n)).collect(),
                    arena_offset: 0,
                    dealloc_after: Vec::new(),
                });
            }
            NodeKind::Slot(_) => unreachable!("slots are pre-plan only nodes"),
        }
    }

    if root.id == base_node.id {
        state.plan_node(ops_len, None, base_node, &root.resolved_inputs);
    }

    let PlanState {
        mut plan,
        slots,
        dealloc_after,
        ..
    } = state;

    #[cfg(feature = "tracing")]
    {
        let span = tracing::Span::current();
        span.record("ops_count", ops_len);
        span.record("slots_count", slots.len());
        span.record("deallocs_after_count", dealloc_after.len());
    }

    for (node_id, dealloc_at) in &dealloc_after {
        let Some(end) = dealloc_at else { continue };

        // TODO: This is unnecessary because we guarantee that a slot is dead before using a region
        // and guarantees that the slot is "free" and there's nothing using the memory region.
        // It can be removed but is maintained to guarantee borrow-checking rules at runtime and
        // future-proof in case it becomes necessary again.
        match &mut plan[*end] {
            ComputeKind::Op { dealloc_after, .. }
            | ComputeKind::CachedOp { dealloc_after, .. }
            | ComputeKind::Baked { dealloc_after, .. } => dealloc_after.push(*node_id),
            ComputeKind::Leaf { .. } => unreachable!(),
        }
    }

    let PackedSlots { arena_size, slots } = greedy_offset_pack_slots(slots, alignment_of::<T>());

    for packed in slots {
        if let ComputeKind::Op { output, .. } = &mut plan[packed.start] {
            *output = OutputKind::Region {
                offset: packed.offset,
                len: packed.len,
            };
        } else if let ComputeKind::Baked { arena_offset, .. } = &mut plan[packed.start] {
            *arena_offset = packed.offset;

            // nothing to evict
            continue;
        }

        // TODO: This is unnecessary because we guarantee that a slot is dead before using a region
        // and guarantees that the allocated values "die" after they are used.
        // It can be removed but is maintained to guarantee borrow-checking rules at runtime and
        // future-proof in case it becomes necessary again.
        match &mut plan[packed.end] {
            ComputeKind::Op { dealloc_after, .. }
            | ComputeKind::CachedOp { dealloc_after, .. }
            | ComputeKind::Baked { dealloc_after, .. } => dealloc_after.push(packed.id),
            ComputeKind::Leaf { .. } => unreachable!(),
        }
    }

    #[cfg(feature = "tracing")]
    trace_plan(&plan, root.id);

    Plan {
        plan,
        root_id: root.id,
        external_inputs,
        arena_size,
    }
}

/// Emit one structured trace event per plan step: the op, the chosen
/// [`OutputKind`], the layout the executor attaches to the output
/// (`shape`/`stride`/`offset`), and the resolved input and dealloc ids.
#[cfg(feature = "tracing")]
fn trace_plan<T, B: Backend>(plan: &[ComputeKind<'_, T, B>], root_id: usize) {
    tracing::debug!(root_id, steps = plan.len(), "plan built");
    for (i, step) in plan.iter().enumerate() {
        match step {
            ComputeKind::Leaf { edge } => {
                tracing::debug!(step = i, kind = "Leaf", id = edge.id);
            }
            ComputeKind::Op {
                node,
                output,
                resolved_inputs,
                dealloc_after,
            } => {
                tracing::debug!(
                    step = i,
                    kind = "Op",
                    id = node.id,
                    op = node.op.as_str(),
                    output = ?output,
                    shape = ?node.layout.shape(),
                    stride = ?node.layout.stride(),
                    offset = node.layout.offset(),
                    inputs = ?resolved_inputs,
                    dealloc = ?dealloc_after,
                );
            }
            ComputeKind::CachedOp {
                cache,
                output,
                resolved_inputs,
                dealloc_after,
            } => {
                let node = cache.get_node();
                tracing::debug!(
                    step = i,
                    kind = "CachedOp",
                    id = node.id,
                    op = node.op.as_str(),
                    output = ?output,
                    shape = ?node.layout.shape(),
                    stride = ?node.layout.stride(),
                    offset = node.layout.offset(),
                    inputs = ?resolved_inputs,
                    dealloc = ?dealloc_after,
                );
            }
            ComputeKind::Baked {
                baked,
                arena_offset,
                resolved_inputs,
                dealloc_after,
            } => {
                tracing::debug!(
                    step = i,
                    kind = "Baked",
                    id = baked.id,
                    arena_offset = *arena_offset,
                    inputs = ?resolved_inputs,
                    dealloc = ?dealloc_after,
                );
            }
        }
    }
}
