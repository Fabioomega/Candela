use std::collections::HashSet;

use crate::tensor::graph::{NodeKind, TensorGraphNode};
use crate::tensor::planner::get_id;

pub struct TopologicalSortIter<'a, T: Copy> {
    stack: Vec<(&'a NodeKind<T>, bool)>,
    visited: HashSet<usize>,
}

impl<'a, T: Copy> TopologicalSortIter<'a, T> {
    pub fn new(base_node: &'a TensorGraphNode<T>) -> Self {
        let mut stack = Vec::new();
        stack.extend(base_node.inputs.iter().map(|i| (i, false)));
        Self {
            stack,
            visited: HashSet::new(),
        }
    }
}

impl<'a, T: Copy> Iterator for TopologicalSortIter<'a, T> {
    type Item = &'a NodeKind<T>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let (node, exiting) = self.stack.pop()?;

            if exiting {
                return Some(node);
            }

            let id = get_id(node);

            if !self.visited.insert(id) {
                continue;
            }

            self.stack.push((node, true));

            match node {
                NodeKind::Edge(_) => {}
                NodeKind::Node(n) => self.stack.extend(n.inputs.iter().rev().map(|i| (i, false))),
                NodeKind::Cache(cache) => {
                    if !cache.is_cache_filled() {
                        self.stack
                            .extend(cache.get_node().inputs.iter().rev().map(|i| (i, false)))
                    }
                }
            }
        }
    }
}

// Performs a DFS topological sort on the current DAG that this leaf (sink) is part of.
// NOTE: The base_node is not added to the iterator output,
//  but would naturally be the last element if added.
// NOTE 2: If a cache and non-cache node with the same id are present in the same DAG,
//  the cache will not be used. That will not be fixed as it would require
//  invalidating some elements after the iteration already went trough.
//  It's the user responsibility to use the cached node correctly.
#[cfg_attr(
    feature = "tracing",
    tracing::instrument(
        level = "trace",
        skip(base_node),
        fields(node_id = base_node.id, inputs_count = base_node.inputs.len())
    )
)]
pub fn topological_sort<T: Copy>(base_node: &TensorGraphNode<T>) -> TopologicalSortIter<'_, T> {
    TopologicalSortIter::new(base_node)
}
