mod in_place;
mod plan;
mod reference;
mod sort;

pub use plan::ComputeKind;
pub use plan::OutputKind;
pub use plan::plan_computation;

use crate::tensor::graph::NodeKind;

#[inline]
pub fn get_id<T: Copy>(node: &NodeKind<T>) -> usize {
    match node {
        NodeKind::Edge(edge) => edge.id,
        NodeKind::Node(node) => node.id,
        NodeKind::Cache(cache) => cache.get_node().id,
    }
}
