use std::collections::HashMap;
use std::sync::Arc;

use crate::Layout;
use crate::tensor::backend::{Backend, ComputeFor};
use crate::tensor::definitions::NumberLike;
use crate::tensor::graph::{TensorGraphBaked, TensorGraphCacheNode, TensorGraphEdge};
use crate::tensor::ops::def_op::OpKind;
use crate::tensor::planner::{ComputeKind, OutputKind, OwnedComputeKind};
use crate::tensor::storage::TensorData;

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

#[inline]
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
        OutputKind::InPlaceIdx(idx) | OutputKind::Reference(idx) => {
            let inputs = build_inputs(computation_cache, resolved_inputs);

            B::compute_inplace(op, layout, inputs, *idx)
        }
    }
}

/// A borrowing view of a single plan step, the unit the executor consumes.
///
/// Both the borrowed [`ComputeKind`] (one-shot planning) and the owned
/// [`OwnedComputeKind`] (a precompiled plan) project into this so
/// [`run_plan`] is written once. The two carriers differ only in how an `Op` step
/// holds its `op`/`layout` - borrowed from the graph node, or from a boxed
/// [`OwnedOp`] - which collapses to the same `&OpKind`/`&Layout` here.
///
/// [`OwnedOp`]: crate::tensor::planner::OwnedOp
pub(crate) enum StepRef<'a, T, B: Backend> {
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
    Baked {
        baked: &'a TensorGraphBaked<T, B>,
        resolved_inputs: &'a Vec<usize>,
        dealloc_after: &'a Vec<usize>,
    },
}

#[inline]
pub(crate) fn borrowed_step<'a, T, B: Backend>(
    step: &'a ComputeKind<'a, T, B>,
) -> StepRef<'a, T, B> {
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
        ComputeKind::Baked {
            baked,
            resolved_inputs,
            dealloc_after,
        } => StepRef::Baked {
            baked,
            resolved_inputs,
            dealloc_after,
        },
    }
}

#[inline]
pub(crate) fn owned_step<T, B: Backend>(step: &OwnedComputeKind<T, B>) -> StepRef<'_, T, B> {
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
        OwnedComputeKind::Baked {
            baked,
            resolved_inputs,
            dealloc_after,
        } => StepRef::Baked {
            baked,
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
pub(crate) fn run_plan<'a, T: NumberLike + ComputeFor<B> + 'a, B: Backend + 'a>(
    steps: &mut (dyn Iterator<Item = StepRef<'a, T, B>> + 'a),
    root_id: usize,
    external_inputs: Vec<(usize, TensorData<T>)>,
) -> TensorData<T> {
    let mut computation_cache: HashMap<usize, TensorData<T>> = HashMap::new();

    for (id, input) in external_inputs.into_iter() {
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
                ..
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
                        OutputKind::Allocate(_) | OutputKind::Reference(_) => {}
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
            StepRef::Baked {
                baked,
                resolved_inputs,
                dealloc_after,
            } => {
                // The outer plan resolved each of `baked.inputs` to the id of the node
                // that produces its buffer (`resolved_inputs`). The inner plan, however,
                // keys those buffers by its own slot ids (`baked.inputs_ids`). The two
                // arrays are positionally aligned, so re-key each outer buffer under the
                // inner slot id it feeds before handing them to the inner run.
                let inputs: Vec<(usize, TensorData<T>)> = baked
                    .inputs_ids
                    .iter()
                    .zip(resolved_inputs.iter())
                    .map(|(&slot_id, &outer_id)| {
                        (slot_id, computation_cache.get(&outer_id).unwrap().clone())
                    })
                    .collect();

                let result = run_plan(
                    &mut baked.plan.plan.iter().map(owned_step),
                    baked.plan.root_id,
                    inputs,
                );

                computation_cache.insert(baked.id, result);

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
