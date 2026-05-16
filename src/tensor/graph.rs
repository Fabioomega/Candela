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
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, OnceLock};

use crate::tensor::definitions::NumberLike;
use crate::tensor::errors::OpError;
use crate::tensor::mem_formats::layout::Layout;
use crate::tensor::ops::def_op::OpKind;
use crate::tensor::ops::fusion::try_fuse;
use crate::tensor::ops::{ComputeWrapperSpec, compute_layout, cpu_compute, cpu_compute_inplace};
use crate::tensor::planner::{ComputeKind, OutputKind, plan_computation};
use crate::tensor::storage::TensorData;
use crate::tensor::traits::Promising;

static NEXT_ID: AtomicUsize = const { AtomicUsize::new(0) };

//////////////////////////////////////////////////////////////////////////////////

/// Every node in the computation graph is one of these three variants.
///
/// - `Edge` — a leaf that wraps a materialized tensor (no computation attached).
/// - `Cache` — a computation whose result is stored after the first evaluation
///   and returned directly on subsequent calls.
/// - `Node` — a regular computation that runs every time it's reached in the plan.
#[derive(Clone, Debug)]
pub enum NodeKind<T: Copy> {
    Edge(Arc<TensorGraphEdge<T>>),
    Cache(Arc<TensorGraphCacheNode<T>>),
    Node(Arc<TensorGraphNode<T>>),
}

//////////////////////////////////////////////////////////////////////////////////

pub(crate) fn get_inputs_layout<T: NumberLike>(inputs: &[NodeKind<T>]) -> Box<[&Layout]> {
    inputs
        .iter()
        .map(|node| match &node {
            NodeKind::Edge(edge) => edge.get().layout(),
            NodeKind::Node(node) => &node.layout,
            NodeKind::Cache(cache) => &cache.get_node().layout,
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

fn execute_output<T: NumberLike + ComputeWrapperSpec>(
    node: &TensorGraphNode<T>,
    output: OutputKind,
    resolved_inputs: &[usize],
    computation_cache: &mut HashMap<usize, TensorData<T>>,
) -> TensorData<T> {
    let fetch = |cache: &HashMap<usize, TensorData<T>>, id: usize| cache.get(&id).unwrap().clone();

    match output {
        OutputKind::Allocate(len) => {
            let output_buffer = alloc_vec(len);
            let inputs: Vec<_> = resolved_inputs
                .iter()
                .map(|&id| fetch(computation_cache, id))
                .collect();

            cpu_compute(&node.op, output_buffer, &node.layout, &inputs)
        }
        OutputKind::Buffer(id) => {
            // TODO: The planner guarantees this id is present in the cache, so this is
            // always Some. Can use unwrap_unchecked once the planner/executor contract
            // is verified to be sound.
            let reused = computation_cache.remove(&id).unwrap();
            let output_buffer = strip_tensor(reused);
            let inputs: Vec<_> = resolved_inputs
                .iter()
                .map(|&id| fetch(computation_cache, id))
                .collect();

            cpu_compute(&node.op, output_buffer, &node.layout, &inputs)
        }
        OutputKind::InPlaceIdx(idx) => {
            let inputs: Vec<_> = resolved_inputs
                .iter()
                .map(|&id| fetch(computation_cache, id))
                .collect();

            cpu_compute_inplace(&node.op, &node.layout, inputs, idx)
        }
    }
}

//////////////////////////////////////////////////////////////////////////////////

/// Leaf node in the computation graph — a plain [`Tensor`] entering the graph.
///
/// Created by [`Tensor::as_promise`], which wraps the underlying [`TensorData`]
/// in an edge and assigns it a unique ID. The edge carries no op; its only job
/// is to make existing data addressable within the graph.
///
/// [`Tensor`]: crate::tensor::tensor::Tensor
/// [`Tensor::as_promise`]: crate::tensor::tensor::Tensor::as_promise
pub struct TensorGraphEdge<T: Copy> {
    pub(crate) id: usize,
    data: TensorData<T>,
}

impl<T: Copy> TensorGraphEdge<T> {
    pub fn from_tensor_data(data: TensorData<T>) -> Self {
        Self {
            id: NEXT_ID.fetch_add(1, Ordering::Relaxed),
            data,
        }
    }

    pub fn get(&self) -> &TensorData<T> {
        &self.data
    }
}

impl<T: Copy> Promising for TensorGraphEdge<T> {
    type Output = T;

    #[inline]
    fn compute(&self) -> TensorData<T> {
        self.data.clone()
    }

    #[inline]
    fn layout(&self) -> &Layout {
        self.data.layout()
    }
}

impl<T: Copy> Debug for TensorGraphEdge<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "TensorGraphEdge {{ id: {}, data: [...] }}", self.id)
    }
}

//////////////////////////////////////////////////////////////////////////////////

/// A computation node in the graph. Holds an op, its inputs, and the output layout.
///
/// Constructed via [`TensorGraphNode::new`], which runs operator fusion and
/// computes the output layout before storing anything — so by the time a node
/// exists, compatible scalar chains have already been collapsed into a single
/// [`OpKind::FusedScalar`] and the output shape is known.
///
/// [`OpKind::FusedScalar`]: crate::tensor::ops::def_op::OpKind::FusedScalar
#[derive(Clone)]
pub struct TensorGraphNode<T: Copy> {
    pub(crate) id: usize,
    pub(crate) op: OpKind<T>,
    pub(crate) inputs: Box<[NodeKind<T>]>,
    pub(crate) layout: Layout,
}

#[allow(private_bounds)]
impl<T: NumberLike + ComputeWrapperSpec> TensorGraphNode<T> {
    pub fn new(op: OpKind<T>, inputs: Box<[NodeKind<T>]>) -> Result<Self, OpError> {
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
        })
    }

    pub fn with_layout(op: OpKind<T>, inputs: Box<[NodeKind<T>]>, layout: Layout) -> Self {
        let fused = try_fuse(op, inputs);

        Self {
            id: NEXT_ID.fetch_add(1, Ordering::Relaxed),
            op: fused.op,
            inputs: fused.inputs,
            layout,
        }
    }
}

impl<T: NumberLike + ComputeWrapperSpec> Promising for TensorGraphNode<T> {
    type Output = T;

    /// Execute the subgraph rooted at this node and return the result.
    ///
    /// This is the entry point for `.materialize()`. It calls [`plan_computation`]
    /// to build a static schedule, then steps through it in order — running each
    /// op, inserting its result into a live-buffer cache, and dropping entries
    /// listed in `dealloc_after` immediately so intermediate buffers are freed as
    /// soon as they're no longer needed.
    ///
    /// Three [`OutputKind`] variants drive execution:
    /// - **Allocate** — allocate a fresh buffer and compute into it.
    /// - **Buffer reuse** — extract a previously freed buffer from the cache and
    ///   compute into it without allocating.
    /// - **In-place** — mutate one of the op's inputs directly. Layout-only ops
    ///   (`View`, `Slice`, `Transpose`) also use this path; they share the input
    ///   buffer at a new layout without running any computation.
    ///
    /// All redirect resolution is performed at plan time. Each step's
    /// `resolved_inputs` contains the concrete `computation_cache` IDs to use..
    ///
    /// Leaf tensors (graph inputs) are inserted into `computation_cache` first via
    /// [`ComputeKind::Leaf`] steps.
    fn compute(&self) -> TensorData<T> {
        let plan = plan_computation(&self).plan;
        let mut computation_cache: HashMap<usize, TensorData<T>> = HashMap::new();

        for comp in plan.into_iter() {
            match comp {
                ComputeKind::Leaf { edge } => {
                    computation_cache.insert(edge.id, edge.data.clone());
                }
                ComputeKind::Op {
                    node,
                    output,
                    resolved_inputs,
                    dealloc_after,
                } => {
                    let result =
                        execute_output(node, output, &resolved_inputs, &mut computation_cache);

                    computation_cache.insert(node.id, result);

                    for dealloc_id in dealloc_after {
                        computation_cache.remove(&dealloc_id);
                    }
                }
                ComputeKind::CachedOp {
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
                                computation_cache.remove(&id);
                            }
                            OutputKind::InPlaceIdx(idx) => {
                                computation_cache.remove(&resolved_inputs[idx]);
                            }
                        }

                        for dealloc_id in dealloc_after {
                            computation_cache.remove(&dealloc_id);
                        }

                        continue;
                    }

                    let node = cache.get_node();
                    let result =
                        execute_output(node, output, &resolved_inputs, &mut computation_cache);
                    let _ = cache.cache.set(result.clone());
                    computation_cache.insert(node.id, result);

                    for dealloc_id in dealloc_after {
                        computation_cache.remove(&dealloc_id);
                    }
                }
            }
        }

        // TODO: The plan always ends with self computed and inserted into the cache, so this
        // is always Some. Can use unwrap_unchecked once the executor contract is verified.
        computation_cache.remove(&self.id).unwrap()
    }

    #[inline]
    fn layout(&self) -> &Layout {
        &self.layout
    }
}

impl<T: Copy + Debug> Debug for TensorGraphNode<T> {
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
/// planner never reclaims the slot owned by a cache node — its buffer survives
/// across separate `.materialize()` calls.
///
/// [`.cache()`]: crate::tensor::promise::TensorPromise::cache
pub struct TensorGraphCacheNode<T: Copy> {
    node: TensorGraphNode<T>,
    cache: OnceLock<TensorData<T>>,
}

impl<T: Copy> TensorGraphCacheNode<T> {
    pub fn from_node(node: TensorGraphNode<T>) -> Self {
        Self {
            node,
            cache: OnceLock::new(),
        }
    }

    pub fn get_node(&self) -> &TensorGraphNode<T> {
        &self.node
    }

    pub fn is_cache_filled(&self) -> bool {
        self.cache.get().is_some()
    }

    pub fn get_cache(&self) -> Option<&TensorData<T>> {
        self.cache.get()
    }
}

#[allow(private_bounds)]
impl<T: NumberLike + ComputeWrapperSpec> TensorGraphCacheNode<T> {
    pub fn new(op: OpKind<T>, inputs: Box<[NodeKind<T>]>) -> Result<Self, OpError> {
        let node = TensorGraphNode::new(op, inputs);

        match node {
            Ok(node) => Ok(Self {
                node: node,
                cache: OnceLock::new(),
            }),
            Err(err) => Err(err),
        }
    }

    pub fn with_layout(op: OpKind<T>, inputs: Box<[NodeKind<T>]>, layout: Layout) -> Self {
        Self {
            node: TensorGraphNode::with_layout(op, inputs, layout),
            cache: OnceLock::new(),
        }
    }
}

impl<T: NumberLike + ComputeWrapperSpec> Promising for TensorGraphCacheNode<T> {
    type Output = T;

    fn compute(&self) -> TensorData<T> {
        // TODO: Once the cuda async is implemented, it would be ideal to change this to an async
        // OnceCell from tokio or some other library
        self.cache.get_or_init(|| self.node.compute()).clone()
    }

    #[inline]
    fn layout(&self) -> &Layout {
        &self.get_node().layout
    }
}

impl<T: Copy + Debug> Debug for TensorGraphCacheNode<T> {
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
