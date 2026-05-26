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
    pub fn is_aliased(&self, id: usize) -> Option<(&NodeKind<T, B>, Tag)> {
        self.map.get(&id).map(|(node, tag)| (*node, tag.clone()))
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
pub(crate) fn classify<'a, T, B: Backend>(
    node_id: usize,
    op: &OpKind<T>,
    inputs: &'a [NodeKind<T, B>],
    id_slot_map: &HashMap<usize, usize>,
    alias_map: &'a mut AliasMap<'a, T, B>,
) -> AliasKind {
    match op {
        OpKind::AsContiguous => {
            if let NodeKind::Cache(_) = &inputs[0] {
                alias_map.insert(node_id, &inputs[0]);
                return AliasKind::Alias;
            }

            let id = get_id(&inputs[0]);
            if let Some((node, tag)) = alias_map.is_aliased(id)
                && tag == Tag::AsContiguous
            {
                alias_map.insert(id, node);
                AliasKind::Alias
            } else {
                AliasKind::OwningAlias
            }
        }
        OpKind::NoOp => {
            alias_map.insert(node_id, alias_map.resolve(&inputs[0]));
            AliasKind::Alias
        }
        _ => AliasKind::NoAlias,
    }
}
