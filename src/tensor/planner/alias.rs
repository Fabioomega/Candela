//! Alias classification for the pre-planner.
//!
//! A node is an *alias* when its output buffer is some other node's buffer rather
//! than freshly computed data. [`classify`] and [`classify_cache`] sort each node
//! into an [`AliasKind`]; [`AliasMap`] records the resulting `node id -> owning
//! node` mapping, and [`AliasMap::resolve`] is the single read path the planner
//! uses to turn an input into the node that actually produces its buffer.
//!
//! Two alias relationships exist. An [`AliasKind::Alias`] node contributes no
//! computation and resolves to an existing node - `NoOp`, or a duplicate
//! `AsContiguous` whose input is already contiguous. An [`AliasKind::Takeover`]
//! node is planned normally but *claims* an existing node: every entry pointing at
//! the claimed node is rewritten to point at the claimer, keeping the map
//! single-hop so later consumers of the claimed node resolve to the canonical
//! producer.

use std::collections::HashMap;

use crate::tensor::backend::Backend;
use crate::tensor::graph::NodeKind;
use crate::tensor::ops::def_op::OpKind;
use crate::tensor::planner::get_id;

/// What guarantee an alias target provides, so [`classify`] can tell whether a
/// later op is already satisfied by an existing alias or must claim it.
#[derive(Copy, Clone, PartialEq)]
pub(crate) enum Tag {
    /// Same data, no layout guarantee - a `NoOp` identity.
    Anything,
    /// A contiguous packing backed by a cache node, so it survives across
    /// separate `.materialize()` calls.
    AsContiguousCache,
    /// A contiguous packing.
    AsContiguous,
}

/// How the pre-planner should treat a node with respect to aliasing.
pub(crate) enum AliasKind<'a, T, B: Backend> {
    /// The node produces no buffer of its own; it resolves to the given node.
    /// Recorded in the [`AliasMap`] with no plan step emitted.
    Alias(&'a NodeKind<T, B>, Tag),
    /// The node is the canonical producer for the given node and claims it: it is
    /// planned normally, then every alias pointing at the claimed node is rewritten
    /// to point at this one. See [`AliasMap::takeover`].
    Takeover(&'a NodeKind<T, B>, Tag),
    /// Ordinary computation. Planned normally; no alias-map entry.
    NoAlias,
}

/// Maps a node id to the node whose buffer supersedes it, tagged with the
/// guarantee that buffer provides.
///
/// Absence of an entry means a node resolves to itself. [`takeover`] rewrites in
/// place rather than chaining, so the map is always single-hop and
/// [`resolve`] is one lookup. Identity is by node id, never by pointer.
///
/// [`resolve`]: AliasMap::resolve
/// [`takeover`]: AliasMap::takeover
pub(crate) struct AliasMap<'a, T, B: Backend> {
    map: HashMap<usize, (&'a NodeKind<T, B>, Tag)>,
}

impl<'a, T, B: Backend> AliasMap<'a, T, B> {
    pub fn new() -> Self {
        Self {
            map: HashMap::with_capacity(32),
        }
    }

    /// Return the node `node` resolves to: its alias-map entry if present,
    /// otherwise `node` itself.
    #[inline]
    pub fn resolve(&self, node: &'a NodeKind<T, B>) -> &'a NodeKind<T, B> {
        let id = get_id(node);
        self.map.get(&id).map_or(node, |(node, _)| *node)
    }

    /// Record that `id` resolves to `node`.
    #[inline]
    pub fn insert(&mut self, id: usize, node: &'a NodeKind<T, B>, tag: Tag) {
        self.map.insert(id, (node, tag));
    }

    /// Rewrite every entry currently pointing at `old_owner` to point at
    /// `new_owner`, then point `old_owner` itself at `new_owner`. Preserves the
    /// single-hop invariant: nothing in the map points at `old_owner` afterward.
    #[inline]
    pub fn takeover(
        &mut self,
        old_owner: &NodeKind<T, B>,
        new_owner: &'a NodeKind<T, B>,
        tag: Tag,
    ) {
        let old_owner_id = get_id(old_owner);
        for (_, value) in self.map.iter_mut() {
            if get_id(value.0) == old_owner_id {
                *value = (new_owner, tag);
            }
        }

        let id = get_id(old_owner);
        self.map.insert(id, (new_owner, tag));
    }

    #[inline]
    fn get_alias(&self, id: usize) -> Option<(&'a NodeKind<T, B>, Tag)> {
        self.map.get(&id).map(|(node, tag)| (*node, *tag))
    }
}

#[inline]
fn is_node_op<T: PartialEq, B: Backend>(node: &NodeKind<T, B>, op: &OpKind<T>) -> bool {
    match node {
        NodeKind::Edge(_) | NodeKind::Slot(_) | NodeKind::Baked(_) => false,
        NodeKind::Node(n) => n.op == *op,
        NodeKind::Cache(c) => c.get_node().op == *op,
    }
}

/// Classify a regular node's op into an [`AliasKind`] against the current alias map.
///
/// `AsContiguous` resolves to an input that is already contiguous-aliased
/// ([`AliasKind::Alias`]) - including a cache input, which always stores a
/// contiguous result - and otherwise claims its input ([`AliasKind::Takeover`]).
/// `NoOp` resolves to its input. Every other op is [`AliasKind::NoAlias`].
#[inline]
pub(crate) fn classify<'a, T: PartialEq, B: Backend>(
    op: &OpKind<T>,
    inputs: &'a [NodeKind<T, B>],
    alias_map: &AliasMap<'a, T, B>,
) -> AliasKind<'a, T, B> {
    match op {
        OpKind::AsContiguous => {
            if let NodeKind::Cache(_) = &inputs[0] {
                return AliasKind::Alias(&inputs[0], Tag::AsContiguous);
            }

            let id = get_id(&inputs[0]);
            if let Some((_, tag)) = alias_map.get_alias(id) {
                if tag == Tag::AsContiguous || tag == Tag::AsContiguousCache {
                    AliasKind::Alias(alias_map.resolve(&inputs[0]), Tag::AsContiguous)
                } else {
                    AliasKind::Takeover(alias_map.resolve(&inputs[0]), Tag::AsContiguous)
                }
            }
            // This should not happen if fusion and the api work right. But just in case.
            else if is_node_op(&inputs[0], op) {
                AliasKind::Alias(&inputs[0], Tag::AsContiguous)
            } else {
                AliasKind::Takeover(&inputs[0], Tag::AsContiguous)
            }
        }
        OpKind::NoOp => AliasKind::Alias(alias_map.resolve(&inputs[0]), Tag::Anything),
        _ => AliasKind::NoAlias,
    }
}

/// Classify a cache node into an [`AliasKind`]. A cache either deduplicates
/// against an input already backed by a contiguous cache ([`AliasKind::Alias`]) or
/// claims its input as the canonical contiguous-cache producer
/// ([`AliasKind::Takeover`]); it is never [`AliasKind::NoAlias`].
#[inline]
pub(crate) fn classify_cache<'a, T: PartialEq, B: Backend>(
    inputs: &'a [NodeKind<T, B>],
    alias_map: &AliasMap<'a, T, B>,
) -> AliasKind<'a, T, B> {
    let id = get_id(&inputs[0]);

    if let Some((owner, tag)) = alias_map.get_alias(id) {
        if tag == Tag::AsContiguousCache {
            // TODO: Maybe a nicer behavior would be to copy to the other cache instead of silently bypassing it.
            // TODO: Add at least a warning here if this happens
            AliasKind::Alias(owner, Tag::AsContiguousCache)
        } else {
            AliasKind::Takeover(owner, Tag::AsContiguousCache)
        }
    } else {
        AliasKind::Takeover(&inputs[0], Tag::AsContiguousCache)
    }
}
