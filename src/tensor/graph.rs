use std::boxed::Box;
use std::cell::OnceCell;
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
use crate::tensor::planner::{ComputeKind, OutputKind, get_id, plan_computation};
use crate::tensor::storage::TensorData;
use crate::tensor::traits::Promising;

static NEXT_ID: AtomicUsize = const { AtomicUsize::new(0) };

//////////////////////////////////////////////////////////////////////////////////

#[derive(Clone, Debug)]
pub enum NodeKind<T: Copy> {
    Edge(Arc<TensorGraphEdge<T>>),
    Cache(Arc<TensorGraphCacheNode<T>>),
    Node(Arc<TensorGraphNode<T>>),
}

//////////////////////////////////////////////////////////////////////////////////

pub fn get_inputs_layout<T: NumberLike>(inputs: &[NodeKind<T>]) -> Box<[&Layout]> {
    inputs
        .iter()
        .map(|node| match &node {
            NodeKind::Edge(edge) => edge.get().layout(),
            NodeKind::Node(node) => &node.layout,
            NodeKind::Cache(cache) => &cache.get_node().layout,
        })
        .collect()
}

fn get_inputs_tensor_data<T: Copy>(
    inputs: &[NodeKind<T>],
    computation_cache: &mut HashMap<usize, TensorData<T>>,
) -> Vec<TensorData<T>> {
    let mut inputs_data: Vec<TensorData<T>> = Vec::with_capacity(inputs.len());
    for kind in inputs.iter() {
        let id = get_id(kind);

        match kind {
            NodeKind::Node(_) => {
                let tensor = computation_cache.get(&id).unwrap();
                inputs_data.push(tensor.clone());
            }
            NodeKind::Cache(cache) => {
                // TODO: The topological sort guarantees cache nodes are computed before they
                // appear as inputs, so this is always Some. Can use unwrap_unchecked once
                // the planner/executor contract is verified to be sound.
                inputs_data.push(cache.cache.get().unwrap().clone());
            }
            NodeKind::Edge(edge) => {
                inputs_data.push(edge.compute());
            }
        }
    }

    inputs_data
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
pub fn alloc_vec<T: Default + Clone>(len: usize) -> Vec<T> {
    let mut output_buffer = Vec::with_capacity(len);
    output_buffer.resize(len, T::default());

    output_buffer
}

fn execute_output<T: NumberLike + ComputeWrapperSpec>(
    node: &TensorGraphNode<T>,
    output: OutputKind,
    computation_cache: &mut HashMap<usize, TensorData<T>>,
) -> TensorData<T> {
    match output {
        OutputKind::Allocate(len) => {
            let output_buffer = alloc_vec(len);
            let inputs = get_inputs_tensor_data(&node.inputs, computation_cache);
            cpu_compute(&node.op, output_buffer, &node.layout, &inputs)
        }
        OutputKind::Buffer(id) => {
            // TODO: The planner guarantees this id is present in the cache, so this is
            // always Some. Can use unwrap_unchecked once the planner/executor contract
            // is verified to be sound.
            let reused = computation_cache.remove(&id).unwrap();
            let output_buffer = strip_tensor(reused);
            let inputs = get_inputs_tensor_data(&node.inputs, computation_cache);
            cpu_compute(&node.op, output_buffer, &node.layout, &inputs)
        }
        OutputKind::InPlaceIdx(idx) => {
            let inputs = get_inputs_tensor_data(&node.inputs, computation_cache);
            cpu_compute_inplace(&node.op, &node.layout, inputs, idx)
        }
    }
}

//////////////////////////////////////////////////////////////////////////////////

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

#[derive(Clone)]
pub struct TensorGraphNode<T: Copy> {
    pub(crate) id: usize,
    pub(crate) op: OpKind<T>,
    pub(crate) inputs: Box<[NodeKind<T>]>,
    pub(crate) layout: Layout,
}

impl<T: NumberLike> TensorGraphNode<T> {
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

    fn compute(&self) -> TensorData<T> {
        let plan = plan_computation(&self);
        let mut computation_cache: HashMap<usize, TensorData<T>> = HashMap::new();

        for comp in plan.into_iter() {
            match comp {
                ComputeKind::Op {
                    node,
                    output,
                    dealloc_after,
                } => {
                    let result = execute_output(node, output, &mut computation_cache);
                    computation_cache.insert(node.id, result);

                    for dealloc_id in dealloc_after {
                        computation_cache.remove(&dealloc_id);
                    }
                }
                ComputeKind::CachedOp {
                    cache,
                    output,
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
                                let id = get_id(&cache.get_node().inputs[idx]);
                                computation_cache.remove(&id);
                            }
                        }

                        for dealloc_id in dealloc_after {
                            computation_cache.remove(&dealloc_id);
                        }

                        continue;
                    }

                    let node = cache.get_node();
                    let result = execute_output(node, output, &mut computation_cache);
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
}

impl<T: NumberLike> TensorGraphCacheNode<T> {
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
