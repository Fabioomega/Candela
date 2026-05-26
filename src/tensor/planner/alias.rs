use std::collections::HashMap;

use crate::tensor::backend::Backend;
use crate::tensor::graph::NodeKind;
use crate::tensor::ops::def_op::OpKind;
use crate::tensor::planner::get_id;

#[derive(Clone, PartialEq)]
pub(crate) enum Tag {
    Alias,
    AsContiguous,
}

pub(crate) enum AliasKind {
    Alias,
    OwningAlias,
    NoAlias,
}

pub(crate) struct AliasMap<'a, T, B: Backend> {
    map: HashMap<usize, (&'a NodeKind<T, B>, Tag)>,
}

impl<'a, T, B: Backend> AliasMap<'a, T, B> {
    pub fn new() -> Self {
        Self {
            map: HashMap::with_capacity(32),
        }
    }

    #[inline]
    pub fn resolve(&self, node: &'a NodeKind<T, B>) -> &NodeKind<T, B> {
        let id = get_id(node);
        self.map.get(&id).map_or(node, |(node, _)| *node)
    }

    #[inline]
    pub fn insert_resolved(&mut self, id: usize, node: &'a NodeKind<T, B>) {
        let node_id = get_id(node);
        let node = self.map.get(&node_id).map_or(node, |(node, _)| *node);
        self.map.insert(id, (node, Tag::Alias));
    }

    // Inserts an alias by another alias, if it fails it inserts node.
    #[inline]
    pub fn insert_by_id_or(&mut self, id: usize, alias: usize, node: &NodeKind<T, B>) -> bool {
        if let Some((node, tag)) = self.map.get(&alias) {
            self.map.insert(id, (*node, tag.clone()));

            true
        } else {
            false
        }
    }

    #[inline]
    pub fn insert(&mut self, id: usize, alias: &'a NodeKind<T, B>) {
        self.map.insert(id, (alias, Tag::Alias));
    }

    #[inline]
    pub fn insert_tagged(&mut self, id: usize, alias: &'a NodeKind<T, B>, tag: Tag) {
        self.map.insert(id, (alias, tag));
    }
}

#[inline]
pub(crate) fn handle_alias<'a, T, B: Backend>(
    node: &'a NodeKind<T, B>,
    node_id: usize,
    op: &OpKind<T>,
    inputs: &'a [NodeKind<T, B>],
    alias_map: &'a mut AliasMap<'a, T, B>,
) -> AliasKind {
    match op {
        OpKind::AsContiguous => {
            if let NodeKind::Cache(_) = &inputs[0] {
                alias_map.insert(node_id, &inputs[0]);
                return AliasKind::Alias;
            }

            let id = get_id(&inputs[0]);
            if alias_map.insert_by_id(node_id, id) {
                AliasKind::Alias
            } else {
                alias_map.insert_tagged(id, node, Tag::AsContiguous);
                AliasKind::OwningAlias
            }
        }
        OpKind::NoOp => {
            alias_map.insert_resolved(node_id, &inputs[0]);
            AliasKind::Alias
        }
        _ => AliasKind::NoAlias,
    }
}
