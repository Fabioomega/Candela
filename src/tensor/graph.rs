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
use std::fmt::Debug;
use std::marker::PhantomData;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, OnceLock};

use crate::Dimension;
use crate::tensor::backend::{Backend, ComputeFor};
use crate::tensor::definitions::NumberLike;
use crate::tensor::errors::OpError;
use crate::tensor::executor::{borrowed_step, run_plan};
use crate::tensor::mem_formats::layout::Layout;
use crate::tensor::ops::compute_layout;
use crate::tensor::ops::def_op::OpKind;
use crate::tensor::ops::fusion::try_fuse;
use crate::tensor::planner::{OwnedCorePlan, plan_computation};
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
    Baked(Arc<TensorGraphBaked<T, B>>),
}

impl<T, B: Backend> Clone for NodeKind<T, B> {
    fn clone(&self) -> Self {
        match self {
            NodeKind::Edge(e) => NodeKind::Edge(e.clone()),
            NodeKind::Cache(c) => NodeKind::Cache(c.clone()),
            NodeKind::Node(n) => NodeKind::Node(n.clone()),
            NodeKind::Slot(s) => NodeKind::Slot(s.clone()),
            NodeKind::Baked(c) => NodeKind::Baked(c.clone()),
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
            NodeKind::Baked(c) => f.debug_tuple("Baked").field(c).finish(),
        }
    }
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
            NodeKind::Baked(baked) => &baked.layout,
        })
        .collect()
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
    pub(crate) data: TensorData<T>,
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

impl<T: Clone, B: Backend> Promising for TensorGraphEdge<T, B> {
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
    /// Four [`OutputKind`] variants drive execution:
    /// - **Allocate** - allocate a fresh buffer and compute into it.
    /// - **Buffer reuse** - extract a previously freed buffer from the cache and
    ///   compute into it without allocating.
    /// - **In-place** - take one of the op's inputs out of the cache and mutate its
    ///   buffer directly; only valid when that buffer is uniquely owned.
    /// - **Reference** - layout-only ops (`View`, `Slice`, `Transpose`) re-point an
    ///   input at a new layout, copying no elements. The input's handle is cloned and
    ///   its buffer stays in the cache, shared with the other nodes that read it.
    ///
    /// All alias resolution is performed at plan time. Each step's
    /// `resolved_inputs` contains the concrete `computation_cache` IDs to use.
    ///
    /// Leaf tensors (graph inputs) are inserted into `computation_cache` first via
    /// [`ComputeKind::Leaf`] steps. The result is the buffer left under the plan's
    /// `root_id` once every step has run.
    fn compute(&self) -> TensorData<T> {
        let plan = plan_computation(self);
        debug_assert!(plan.external_inputs.is_empty());

        run_plan(
            &mut plan.plan.iter().map(borrowed_step),
            plan.root_id,
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
    pub(crate) cache: OnceLock<TensorData<T>>,
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
    pub(crate) fn new(layout: Layout) -> Self {
        Self {
            id: NEXT_ID.fetch_add(1, Ordering::Relaxed),
            layout,
            marker: PhantomData {},
        }
    }

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
pub struct TensorGraphBaked<T, B: Backend> {
    pub(crate) id: usize,
    pub(crate) inputs: Box<[NodeKind<T, B>]>,
    pub(crate) inputs_ids: Box<[usize]>,
    pub(crate) plan: Arc<OwnedCorePlan<T, B>>,
    layout: Layout,
}

impl<T: PartialEq + Clone, B: Backend> TensorGraphBaked<T, B> {
    pub(crate) fn from_node(
        plan: &Arc<OwnedCorePlan<T, B>>,
        inputs: Box<[NodeKind<T, B>]>,
        inputs_ids: Box<[usize]>,
        layout: &Layout,
    ) -> Self {
        Self {
            id: NEXT_ID.fetch_add(1, Ordering::Relaxed),
            inputs,
            inputs_ids,
            plan: plan.clone(),
            layout: layout.clone(),
        }
    }
}

impl<T, B: Backend> TensorGraphBaked<T, B> {
    #[inline]
    pub(crate) fn layout(&self) -> &Layout {
        &self.layout
    }
}

impl<T: Debug, B: Backend> Debug for TensorGraphBaked<T, B> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "TensorGraphBaked {{ id: {:?}, plan: [...] }}", self.id)
    }
}
