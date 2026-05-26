mod alias;
mod plan;
mod runtime;
mod sort;

use std::collections::HashMap;

pub(crate) use plan::ComputeKind;
pub(crate) use plan::OutputKind;
pub(crate) use plan::plan_computation;

use crate::tensor::backend::Backend;
use crate::tensor::graph::NodeKind;

#[inline]
pub(crate) fn get_id<T, B: Backend>(node: &NodeKind<T, B>) -> usize {
    match node {
        NodeKind::Edge(edge) => edge.id,
        NodeKind::Node(node) => node.id,
        NodeKind::Cache(cache) => cache.get_node().id,
    }
}

#[inline]
pub(crate) fn get_red_id<T, B: Backend>(
    node: &NodeKind<T, B>,
    id_redirect: &HashMap<usize, usize>,
) -> usize {
    let id = get_id(node);

    id_redirect
        .get(&id)
        .map_or(id, |redirected_id| *redirected_id)
}
