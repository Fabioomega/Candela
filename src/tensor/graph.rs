//! Graph node types and the execution entry point.
//!
//! The computation graph is a DAG of [`NodeKind`] variants. Building a promise
//! chain constructs this graph without running anything; calling
//! [`Promising::compute`] on the root node triggers the planner and then
//! executes the resulting schedule. See [doc/graph.md] and [doc/planner.md]
//! for a detailed walkthrough.
//!
//! [doc/graph.md]: https://github.com/Fabioomega/candela/blob/main/doc/graph.md
//! [doc/planner.md]: https://github.com/Fabioomega/candela/blob/main/doc/planner.md

use std::boxed::Box;
use std::collections::HashMap;
use std::fmt::Debug;
use std::iter::zip;
use std::marker::PhantomData;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, OnceLock};

use crate::Dimension;
use crate::tensor::backend::{Backend, ComputeFor};
use crate::tensor::definitions::NumberLike;
use crate::tensor::errors::OpError;
use crate::tensor::mem_formats::layout::Layout;
use crate::tensor::ops::compute_layout;
use crate::tensor::ops::def_op::OpKind;
use crate::tensor::ops::fusion::try_fuse;
use crate::tensor::planner::{
    ComputeKind, OutputKind, OwnedComputeKind, OwnedCorePlan, core_plan_computation,
    from_borrowed_core_to_owned, plan_computation,
};
use crate::tensor::storage::TensorData;
use crate::tensor::traits::{Numeric, Promising};

static NEXT_ID: AtomicUsize = const { AtomicUsize::new(0) };

//////////////////////////////////////////////////////////////////////////////////

/// Every node in the computation graph is one of these three variants.
///
/// - `Edge` - a leaf that wraps a materialized tensor (no computation attached).
/// - `Cache` - a computation whose result is stored after the first evaluation
///   and returned directly on subsequent calls.
/// - `Node` - a regular computation that runs every time it's reached in the plan.
/// - `Slot` - an `Edge` that must be defined before computation.
pub enum NodeKind<T, B: Backend> {
    Edge(Arc<TensorGraphEdge<T, B>>),
    Cache(Arc<TensorGraphCacheNode<T, B>>),
    Node(Arc<TensorGraphNode<T, B>>),
    Slot(Arc<TensorGraphSlot<T, B>>),
    Compact(Arc<TensorGraphCompact<T, B>>),
}

impl<T, B: Backend> Clone for NodeKind<T, B> {
    fn clone(&self) -> Self {
        match self {
            NodeKind::Edge(e) => NodeKind::Edge(e.clone()),
            NodeKind::Cache(c) => NodeKind::Cache(c.clone()),
            NodeKind::Node(n) => NodeKind::Node(n.clone()),
            NodeKind::Slot(s) => NodeKind::Slot(s.clone()),
            NodeKind::Compact(c) => NodeKind::Compact(c.clone()),
        }
    }
}

impl<T: Debug, B: Backend> Debug for NodeKind<T, B> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            NodeKind::Edge(e) => f.debug_tuple("Edge").field(e).finish(),
            NodeKind::Cache(c) => f.debug_tuple("Cache").field(c).finish(),
            NodeKind::Node(n) => f.debug_tuple("Node").field(n).finish(),
            NodeKind::Slot(s) => f.debug_tuple("Slot").field(s).finish(),
            NodeKind::Compact(c) => f.debug_tuple("Compact").field(c).finish(),
        }
    }
}

//////////////////////////////////////////////////////////////////////////////////

/// A borrowing view of a single plan step, the unit the executor consumes.
///
/// Both the borrowed [`ComputeKind`] (one-shot planning) and the owned
/// [`OwnedComputeKind`] (a precompiled [`TensorGraphCompact`]) project into this so
/// [`run_plan`] is written once. The two carriers differ only in how an `Op` step
/// holds its `op`/`layout` - borrowed from the graph node, or from a boxed
/// [`OwnedOp`] - which collapses to the same `&OpKind`/`&Layout` here.
///
/// [`OwnedOp`]: crate::tensor::planner::OwnedOp
enum StepRef<'a, T, B: Backend> {
    Leaf {
        edge: &'a TensorGraphEdge<T, B>,
    },
    Op {
        id: usize,
        op: &'a OpKind<T>,
        layout: &'a Layout,
        output: &'a OutputKind,
        resolved_inputs: &'a [usize],
        dealloc_after: &'a [usize],
    },
    CachedOp {
        cache: &'a TensorGraphCacheNode<T, B>,
        output: &'a OutputKind,
        resolved_inputs: &'a [usize],
        dealloc_after: &'a [usize],
    },
    Compact {
        compact: &'a TensorGraphCompact<T, B>,
        resolved_inputs: &'a Vec<usize>,
        dealloc_after: &'a Vec<usize>,
    },
}

#[inline]
fn borrowed_step<'a, T, B: Backend>(step: &'a ComputeKind<'a, T, B>) -> StepRef<'a, T, B> {
    match step {
        ComputeKind::Leaf { edge } => StepRef::Leaf { edge },
        ComputeKind::Op {
            node,
            output,
            resolved_inputs,
            dealloc_after,
        } => StepRef::Op {
            id: node.id,
            op: &node.op,
            layout: &node.layout,
            output,
            resolved_inputs,
            dealloc_after,
        },
        ComputeKind::CachedOp {
            cache,
            output,
            resolved_inputs,
            dealloc_after,
        } => StepRef::CachedOp {
            cache,
            output,
            resolved_inputs,
            dealloc_after,
        },
        ComputeKind::Compact {
            compact,
            resolved_inputs,
            dealloc_after,
        } => StepRef::Compact {
            compact: compact,
            resolved_inputs,
            dealloc_after,
        },
    }
}

#[inline]
fn owned_step<T, B: Backend>(step: &OwnedComputeKind<T, B>) -> StepRef<'_, T, B> {
    match step {
        OwnedComputeKind::Leaf { edge } => StepRef::Leaf { edge },
        OwnedComputeKind::Op {
            node,
            output,
            resolved_inputs,
            dealloc_after,
        } => StepRef::Op {
            id: node.id,
            op: &node.op,
            layout: &node.layout,
            output,
            resolved_inputs,
            dealloc_after,
        },
        OwnedComputeKind::CachedOp {
            cache,
            output,
            resolved_inputs,
            dealloc_after,
        } => StepRef::CachedOp {
            cache,
            output,
            resolved_inputs,
            dealloc_after,
        },
        OwnedComputeKind::Compact {
            compact,
            resolved_inputs,
            dealloc_after,
        } => StepRef::Compact {
            compact: compact,
            resolved_inputs,
            dealloc_after,
        },
    }
}

/// Run a planned schedule against a fresh live-buffer cache and return the buffer
/// left under `root_id`. Shared by the borrowed and owned executors via [`StepRef`].
///
/// Leaf tensors are inserted first; every other step resolves its inputs by ID,
/// computes into its assigned buffer, then drops the IDs in `dealloc_after`.
//
// Steps had to be a dyn Iterator because of an nasty recursive type problem
fn run_plan<'a, T: NumberLike + ComputeFor<B> + 'a, B: Backend + 'a>(
    steps: &mut (dyn Iterator<Item = StepRef<'a, T, B>> + 'a),
    root_id: usize,
    external_inputs_ids: &[usize],
    external_inputs: Vec<TensorData<T>>,
) -> TensorData<T> {
    let mut computation_cache: HashMap<usize, TensorData<T>> = HashMap::new();

    for (&id, input) in zip(external_inputs_ids, external_inputs) {
        computation_cache.insert(id, input);
    }

    for step in steps {
        match step {
            StepRef::Leaf { edge } => {
                computation_cache.insert(edge.id, edge.data.clone());
            }
            StepRef::Op {
                id,
                op,
                layout,
                output,
                resolved_inputs,
                dealloc_after,
            } => {
                let result =
                    execute_output(op, layout, output, resolved_inputs, &mut computation_cache);

                computation_cache.insert(id, result);

                for &dealloc_id in dealloc_after {
                    computation_cache.remove(&dealloc_id);
                }
            }
            StepRef::CachedOp {
                cache,
                output,
                resolved_inputs,
                dealloc_after,
            } => {
                if cache.is_cache_filled() {
                    // The planner emits Allocate(0) for nodes that were already cached at
                    // plan time, so Allocate is the common case here. Buffer and InPlaceIdx
                    // are reached only when a race occurs: the cache was empty at plan time
                    // but filled by another thread before this executor step runs. In that
                    // case we still need to release the slot the planner reserved.
                    //
                    // TODO: is_cache_filled() returning true guarantees cache.get() is Some,
                    // so this unwrap can become unwrap_unchecked once the contract is verified.
                    computation_cache
                        .insert(cache.get_node().id, cache.cache.get().unwrap().clone());

                    match output {
                        OutputKind::Allocate(_) => {}
                        OutputKind::Buffer(id) => {
                            computation_cache.remove(id);
                        }
                        OutputKind::InPlaceIdx(idx) => {
                            computation_cache.remove(&resolved_inputs[*idx]);
                        }
                    }

                    for &dealloc_id in dealloc_after {
                        computation_cache.remove(&dealloc_id);
                    }

                    continue;
                }

                let node = cache.get_node();
                let result = execute_output(
                    &node.op,
                    &node.layout,
                    output,
                    resolved_inputs,
                    &mut computation_cache,
                );
                let _ = cache.cache.set(result.clone());
                computation_cache.insert(node.id, result);

                for &dealloc_id in dealloc_after {
                    computation_cache.remove(&dealloc_id);
                }
            }
            StepRef::Compact {
                compact,
                resolved_inputs,
                dealloc_after,
            } => {
                let result = run_plan(
                    &mut compact.plan.plan.iter().map(owned_step),
                    compact.plan.root_id,
                    &compact.plan.external_inputs,
                    build_inputs(&computation_cache, resolved_inputs),
                );

                computation_cache.insert(compact.id, result);

                for &dealloc_id in dealloc_after {
                    computation_cache.remove(&dealloc_id);
                }
            }
        }
    }

    // TODO: The plan always ends with the root computed and inserted into the cache, so
    // this is always Some. Can use unwrap_unchecked once the executor contract is verified.
    computation_cache.remove(&root_id).unwrap()
}

//////////////////////////////////////////////////////////////////////////////////

pub(crate) fn get_inputs_layout<T: NumberLike, B: Backend>(
    inputs: &[NodeKind<T, B>],
) -> Box<[&Layout]> {
    inputs
        .iter()
        .map(|node| match &node {
            NodeKind::Edge(edge) => edge.get().layout(),
            NodeKind::Node(node) => &node.layout,
            NodeKind::Cache(cache) => &cache.get_node().layout,
            NodeKind::Slot(slot) => &slot.layout,
            NodeKind::Compact(compact) => &compact.layout,
        })
        .collect()
}

#[inline]
fn strip_tensor<T: Copy>(tensor: TensorData<T>) -> Vec<T> {
    if let Ok(v) = Arc::try_unwrap(tensor.storage.buffer) {
        v
    } else {
        unreachable!("cannot strip a tensor that is being used!")
    }
}

#[inline]
fn alloc_vec<T: Default + Clone>(len: usize) -> Vec<T> {
    let mut output_buffer = Vec::with_capacity(len);
    output_buffer.resize(len, T::default());

    output_buffer
}

fn build_inputs<T: Clone>(
    computation_cache: &HashMap<usize, TensorData<T>>,
    ids: &[usize],
) -> Vec<TensorData<T>> {
    ids.iter()
        .map(|&id| computation_cache.get(&id).unwrap().clone())
        .collect()
}

fn execute_output<T: NumberLike + ComputeFor<B>, B: Backend>(
    op: &OpKind<T>,
    layout: &Layout,
    output: &OutputKind,
    resolved_inputs: &[usize],
    computation_cache: &mut HashMap<usize, TensorData<T>>,
) -> TensorData<T> {
    match output {
        OutputKind::Allocate(len) => {
            let output_buffer = alloc_vec(*len);
            let inputs = build_inputs(computation_cache, resolved_inputs);

            B::compute(op, output_buffer, layout, &inputs)
        }
        OutputKind::Buffer(id) => {
            // TODO: The planner guarantees this id is present in the cache, so this is
            // always Some. Can use unwrap_unchecked once the planner/executor contract
            // is verified to be sound.
            let reused = computation_cache.remove(id).unwrap();
            let output_buffer = strip_tensor(reused);
            let inputs = build_inputs(computation_cache, resolved_inputs);

            B::compute(op, output_buffer, layout, &inputs)
        }
        OutputKind::InPlaceIdx(idx) => {
            let inputs = build_inputs(computation_cache, resolved_inputs);

            B::compute_inplace(op, layout, inputs, *idx)
        }
    }
}

//////////////////////////////////////////////////////////////////////////////////

/// Leaf node in the computation graph - a plain [`Tensor`] entering the graph.
///
/// Created by [`Tensor::as_promise`], which wraps the underlying [`TensorData`]
/// in an edge and assigns it a unique ID. The edge carries no op; its only job
/// is to make existing data addressable within the graph.
///
/// [`Tensor`]: crate::tensor::tensor::Tensor
/// [`Tensor::as_promise`]: crate::tensor::tensor::Tensor::as_promise
pub struct TensorGraphEdge<T, B: Backend> {
    pub(crate) id: usize,
    data: TensorData<T>,
    marker: PhantomData<B>,
}

impl<T, B: Backend> TensorGraphEdge<T, B> {
    pub fn from_tensor_data(data: TensorData<T>) -> Self {
        Self {
            id: NEXT_ID.fetch_add(1, Ordering::Relaxed),
            data,
            marker: PhantomData {},
        }
    }

    pub fn get(&self) -> &TensorData<T> {
        &self.data
    }

    #[inline]
    pub(crate) fn layout(&self) -> &Layout {
        self.data.layout()
    }
}

impl<T: Copy, B: Backend> Promising for TensorGraphEdge<T, B> {
    type Output = T;

    #[inline]
    fn compute(&self) -> TensorData<T> {
        self.data.clone()
    }
}

impl<T, B: Backend> Debug for TensorGraphEdge<T, B> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "TensorGraphEdge {{ id: {}, data: [...] }}", self.id)
    }
}

//////////////////////////////////////////////////////////////////////////////////

/// A computation node in the graph. Holds an op, its inputs, and the output layout.
///
/// Constructed via [`TensorGraphNode::new`], which runs operator fusion and
/// computes the output layout before storing anything - so by the time a node
/// exists, compatible scalar chains have already been collapsed into a single
/// [`OpKind::FusedScalar`] and the output shape is known.
///
/// [`OpKind::FusedScalar`]: crate::tensor::ops::def_op::OpKind::FusedScalar
pub struct TensorGraphNode<T, B: Backend> {
    pub(crate) id: usize,
    pub(crate) op: OpKind<T>,
    pub(crate) inputs: Box<[NodeKind<T, B>]>,
    pub(crate) layout: Layout,
    marker: PhantomData<B>,
}

#[allow(private_bounds)]
impl<T: Numeric, B: Backend> TensorGraphNode<T, B> {
    pub fn new(op: OpKind<T>, inputs: Box<[NodeKind<T, B>]>) -> Result<Self, OpError> {
        let fused = try_fuse(op, inputs);

        let layouts = get_inputs_layout(&fused.inputs);
        let layout = compute_layout(&fused.op, &layouts);

        if let Err(err) = layout {
            return Err(err);
        }

        let unchecked_layout = unsafe { layout.unwrap_unchecked() };

        Ok(Self {
            id: NEXT_ID.fetch_add(1, Ordering::Relaxed),
            op: fused.op,
            inputs: fused.inputs,
            layout: unchecked_layout,
            marker: PhantomData {},
        })
    }

    pub fn with_layout(op: OpKind<T>, inputs: Box<[NodeKind<T, B>]>, layout: Layout) -> Self {
        let fused = try_fuse(op, inputs);

        Self {
            id: NEXT_ID.fetch_add(1, Ordering::Relaxed),
            op: fused.op,
            inputs: fused.inputs,
            layout,
            marker: PhantomData {},
        }
    }
}

impl<T, B: Backend> TensorGraphNode<T, B> {
    #[inline]
    pub(crate) fn layout(&self) -> &Layout {
        &self.layout
    }
}

impl<T: NumberLike + ComputeFor<B>, B: Backend> Promising for TensorGraphNode<T, B> {
    type Output = T;

    /// Execute the subgraph rooted at this node and return the result.
    ///
    /// This is the entry point for `.materialize()`. It calls [`plan_computation`]
    /// to build a static schedule, then steps through it in order - running each
    /// op, inserting its result into a live-buffer cache, and dropping entries
    /// listed in `dealloc_after` immediately so intermediate buffers are freed as
    /// soon as they're no longer needed.
    ///
    /// Three [`OutputKind`] variants drive execution:
    /// - **Allocate** - allocate a fresh buffer and compute into it.
    /// - **Buffer reuse** - extract a previously freed buffer from the cache and
    ///   compute into it without allocating.
    /// - **In-place** - mutate one of the op's inputs directly. Layout-only ops
    ///   (`View`, `Slice`, `Transpose`) also use this path; they share the input
    ///   buffer at a new layout without running any computation.
    ///
    /// All alias resolution is performed at plan time. Each step's
    /// `resolved_inputs` contains the concrete `computation_cache` IDs to use.
    ///
    /// Leaf tensors (graph inputs) are inserted into `computation_cache` first via
    /// [`ComputeKind::Leaf`] steps. The result is the buffer left under the plan's
    /// `root_id` once every step has run.
    fn compute(&self) -> TensorData<T> {
        let plan = plan_computation(self);
        run_plan(
            &mut plan.plan.iter().map(borrowed_step),
            plan.root_id,
            &plan.external_inputs,
            Vec::new(),
        )
    }
}

impl<T: Debug, B: Backend> Debug for TensorGraphNode<T, B> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "TensorGraphNode {{ id: {:?}, op: {:?},  inputs: [...] }}",
            self.id, self.op
        )
    }
}

//////////////////////////////////////////////////////////////////////////////////

/// A computation node whose result is kept alive after the first evaluation.
///
/// Wraps a [`TensorGraphNode`] and adds a `OnceLock<TensorData<T>>`. The inner
/// computation runs at most once; every subsequent call to `compute()` returns a
/// clone of the stored result without re-running the graph.
///
/// This is what you get when you call [`.cache()`] on a `TensorPromise`. The
/// planner never reclaims the slot owned by a cache node - its buffer survives
/// across separate `.materialize()` calls.
///
/// [`.cache()`]: crate::tensor::promise::TensorPromise::cache
pub struct TensorGraphCacheNode<T, B: Backend> {
    node: TensorGraphNode<T, B>,
    cache: OnceLock<TensorData<T>>,
}

impl<T, B: Backend> TensorGraphCacheNode<T, B> {
    pub fn from_node(node: TensorGraphNode<T, B>) -> Self {
        Self {
            node,
            cache: OnceLock::new(),
        }
    }

    pub fn get_node(&self) -> &TensorGraphNode<T, B> {
        &self.node
    }

    pub fn is_cache_filled(&self) -> bool {
        self.cache.get().is_some()
    }

    pub fn get_cache(&self) -> Option<&TensorData<T>> {
        self.cache.get()
    }

    #[inline]
    pub(crate) fn layout(&self) -> &Layout {
        &self.node.layout
    }
}

#[allow(private_bounds)]
impl<T: Numeric, B: Backend> TensorGraphCacheNode<T, B> {
    pub fn new(op: OpKind<T>, inputs: Box<[NodeKind<T, B>]>) -> Result<Self, OpError> {
        let node = TensorGraphNode::new(op, inputs);

        match node {
            Ok(node) => Ok(Self {
                node,
                cache: OnceLock::new(),
            }),
            Err(err) => Err(err),
        }
    }

    pub fn with_layout(op: OpKind<T>, inputs: Box<[NodeKind<T, B>]>, layout: Layout) -> Self {
        Self {
            node: TensorGraphNode::with_layout(op, inputs, layout),
            cache: OnceLock::new(),
        }
    }
}

impl<T: NumberLike + ComputeFor<B>, B: Backend> Promising for TensorGraphCacheNode<T, B> {
    type Output = T;

    fn compute(&self) -> TensorData<T> {
        // TODO: Once the cuda async is implemented, it would be ideal to change this to an async
        // OnceCell from tokio or some other library
        self.cache.get_or_init(|| self.node.compute()).clone()
    }
}

impl<T: Debug, B: Backend> Debug for TensorGraphCacheNode<T, B> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "TensorGraphNode {{ id: {:?}, op: {:?},  inputs: [...], cached: {} }}",
            self.node.id,
            self.node.op,
            self.is_cache_filled()
        )
    }
}

//////////////////////////////////////////////////////////////////////////////////

pub struct TensorGraphSlot<T, B: Backend> {
    pub(crate) id: usize,
    pub(crate) layout: Layout,
    marker: PhantomData<(T, B)>,
}

impl<T, B: Backend> TensorGraphSlot<T, B> {
    #[inline]
    pub(crate) fn layout(&self) -> &Layout {
        &self.layout
    }
}

impl<T: Debug, B: Backend> Debug for TensorGraphSlot<T, B> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "TensorGraphSlot {{ id: {},  }}", self.id)
    }
}

//////////////////////////////////////////////////////////////////////////////////
pub struct TensorGraphCompact<T, B: Backend> {
    pub(crate) id: usize,
    pub(crate) inputs: Box<[NodeKind<T, B>]>,
    pub(crate) plan: OwnedCorePlan<T, B>,
    layout: Layout,
}

impl<T: PartialEq + Clone, B: Backend> TensorGraphCompact<T, B> {
    pub fn from_node(node: TensorGraphNode<T, B>, external_inputs: Box<[NodeKind<T, B>]>) -> Self {
        let core = core_plan_computation(&node);
        Self {
            id: node.id,
            inputs: external_inputs,
            plan: from_borrowed_core_to_owned(core),
            layout: node.layout().clone(),
        }
    }
}

impl<T, B: Backend> TensorGraphCompact<T, B> {
    #[inline]
    pub(crate) fn layout(&self) -> &Layout {
        &self.layout
    }
}

impl<T: Debug, B: Backend> Debug for TensorGraphCompact<T, B> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "TensorGraphCompact {{ id: {:?}, plan: [...] }}", self.id)
    }
}
