use std::sync::Arc;

use crate::Layout;
use crate::tensor::backend::Backend;
use crate::tensor::graph::{TensorGraphBaked, TensorGraphCacheNode, TensorGraphEdge};
use crate::tensor::ops::def_op::OpKind;
use crate::tensor::planner::plan::Plan;
use crate::tensor::planner::{ComputeKind, OutputKind};

/// The owned, graph-detached payload of an [`OwnedComputeKind::Op`] step: the node's
/// id, op, and output layout cloned out of the graph so the plan owns them. Boxed in
/// `Op` to keep the enum small. The owned mirror of the borrowed `Op`'s `&node`.
pub(crate) struct OwnedOp<T> {
    pub(crate) id: usize,
    pub(crate) op: OpKind<T>,
    pub(crate) layout: Layout,
}

pub(crate) enum OwnedComputeKind<T, B: Backend> {
    Leaf {
        edge: Arc<TensorGraphEdge<T, B>>,
    },
    Op {
        node: Box<OwnedOp<T>>,
        output: OutputKind,
        resolved_inputs: Vec<usize>,
        dealloc_after: Vec<usize>,
    },
    CachedOp {
        cache: Arc<TensorGraphCacheNode<T, B>>,
        output: OutputKind,
        resolved_inputs: Vec<usize>,
        dealloc_after: Vec<usize>,
    },
    Baked {
        baked: Arc<TensorGraphBaked<T, B>>,
        arena_offset: usize,
        resolved_inputs: Vec<usize>,
        dealloc_after: Vec<usize>,
    },
}

pub(crate) struct OwnedPlan<T, B: Backend> {
    pub(crate) plan: Vec<OwnedComputeKind<T, B>>,
    pub(crate) root_id: usize,
    pub(crate) arena_size: usize,
}

#[inline]
fn from_borrowed_compute_kind_to_owned<T: Clone, B: Backend>(
    borrowed: Vec<ComputeKind<'_, T, B>>,
) -> Vec<OwnedComputeKind<T, B>> {
    borrowed
        .into_iter()
        .map(|borrowed| match borrowed {
            ComputeKind::Leaf { edge } => OwnedComputeKind::Leaf { edge: edge.clone() },
            ComputeKind::CachedOp {
                cache,
                output,
                resolved_inputs,
                dealloc_after,
            } => OwnedComputeKind::CachedOp {
                cache: cache.clone(),
                output,
                resolved_inputs,
                dealloc_after,
            },
            ComputeKind::Op {
                node,
                output,
                resolved_inputs,
                dealloc_after,
            } => OwnedComputeKind::Op {
                node: Box::new(OwnedOp {
                    id: node.id,
                    op: node.op.clone(),
                    layout: node.layout().clone(),
                }),
                output,
                resolved_inputs,
                dealloc_after,
            },
            ComputeKind::Baked {
                baked,
                arena_offset,
                resolved_inputs,
                dealloc_after,
            } => OwnedComputeKind::Baked {
                baked: baked.clone(),
                arena_offset,
                resolved_inputs,
                dealloc_after,
            },
        })
        .collect()
}

#[inline]
pub(crate) fn from_borrowed_core_to_owned<T: Clone, B: Backend>(
    core: Plan<'_, T, B>,
) -> OwnedPlan<T, B> {
    OwnedPlan {
        plan: from_borrowed_compute_kind_to_owned(core.plan),
        root_id: core.root_id,
        arena_size: core.arena_size,
    }
}
