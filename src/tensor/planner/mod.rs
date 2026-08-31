mod alias;
mod owned;
mod packing;
mod plan;
mod runtime;
mod sort;

pub(crate) use owned::{OwnedComputeKind, OwnedPlan, from_borrowed_core_to_owned};
pub(crate) use plan::ComputeKind;
pub(crate) use plan::OutputKind;
pub(crate) use plan::plan_computation;

use crate::Layout;
use crate::tensor::backend::Backend;
use crate::tensor::graph::NodeKind;

pub const ALIGNMENT_BYTES: usize = 128;

#[inline]
pub(crate) fn get_id<T, B: Backend>(node: &NodeKind<T, B>) -> usize {
    match node {
        NodeKind::Edge(edge) => edge.id,
        NodeKind::Slot(slot) => slot.id,
        NodeKind::Node(node) => node.id,
        NodeKind::Cache(cache) => cache.get_node().id,
        NodeKind::Baked(baked) => baked.id,
    }
}

#[inline]
pub(crate) fn get_layout<T, B: Backend>(node: &NodeKind<T, B>) -> &Layout {
    match node {
        NodeKind::Edge(edge) => edge.layout(),
        NodeKind::Slot(slot) => slot.layout(),
        NodeKind::Node(node) => node.layout(),
        NodeKind::Cache(cache) => cache.get_node().layout(),
        NodeKind::Baked(baked) => baked.layout(),
    }
}
