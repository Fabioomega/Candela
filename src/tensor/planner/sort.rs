//! Topological sort for the computation graph.
//!
//! Yields graph nodes in post-order (inputs before the ops that consume them)
//! so the planner and executor can process them in a safe dependency order.

use std::collections::HashSet;

use crate::tensor::graph::{NodeKind, TensorGraphNode};
use crate::tensor::planner::get_id;

/// Iterator that yields the nodes of a computation DAG in topological order.
///
/// Built by [`topological_sort`]. Uses an explicit stack so arbitrarily deep
/// graphs don't overflow the call stack.
pub(crate) struct TopologicalSortIter<'a, T: Copy> {
    stack: Vec<(&'a NodeKind<T>, bool)>,
    visited: HashSet<usize>,
}

impl<'a, T: Copy> TopologicalSortIter<'a, T> {
    pub(crate) fn new(base_node: &'a TensorGraphNode<T>) -> Self {
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

/// Returns an iterator that visits every node reachable from `base_node` in
/// topological order (inputs before the ops that consume them).
///
/// `base_node` itself is **not** yielded — the planner adds it separately at the
/// end of the plan. All other nodes are deduplicated by ID, so shared nodes
/// appear exactly once.
///
/// If a [`TensorGraphCacheNode`] in the graph is already filled, its entire
/// subtree is skipped; the cache node is treated as a leaf.
///
/// [`TensorGraphCacheNode`]: crate::tensor::graph::TensorGraphCacheNode
///
/// # Note on mixed cache/non-cache nodes
///
/// If a cache node and a regular node share the same ID in the same DAG, the
/// regular node wins and the cache is ignored for that branch. This is a known
/// edge case — avoid constructing graphs where this can happen.
#[cfg_attr(
    feature = "tracing",
    tracing::instrument(
        level = "trace",
        skip(base_node),
        fields(node_id = base_node.id, inputs_count = base_node.inputs.len())
    )
)]
pub(crate) fn topological_sort<T: Copy>(base_node: &TensorGraphNode<T>) -> TopologicalSortIter<'_, T> {
    TopologicalSortIter::new(base_node)
}
