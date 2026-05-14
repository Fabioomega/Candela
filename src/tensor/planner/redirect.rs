//! Redirect detection for the static execution planner.
//!
//! A *redirect* op produces a transformed version of a single input that
//! downstream consumers should use in place of the original. The canonical
//! example is [`OpKind::AsContiguous`]: it packs a non-contiguous tensor into a
//! fresh contiguous buffer so that ops like matmul can call into BLAS directly.
//!
//! When two [`AsContiguous`] nodes share the same input, the second one is
//! redundant — the planner detects this via [`is_a_redirect`] and skips
//! planning a second buffer entirely. It extends the first node's slot lifetime
//! to cover all consumers of both, and records the mapping in the redirect table
//! returned by [`plan_computation`]. At execution time the executor resolves the
//! duplicate node's ID through that table, transparently serving the result that
//! was already computed.
//!
//! [`AsContiguous`]: OpKind::AsContiguous
//! [`plan_computation`]: crate::tensor::planner::plan::plan_computation

use std::collections::HashMap;

use crate::tensor::graph::NodeKind;
use crate::tensor::ops::def_op::OpKind;
use crate::tensor::planner::get_id;

/// Outcome of [`is_a_redirect`] for a single node.
pub(crate) enum RedirectKind {
    /// This node is the *first* redirect for its input. The contained ID is the
    /// input node's ID. The planner registers this node as the canonical redirect
    /// target for that input and plans it normally.
    RedirectFrom(usize),
    /// A redirect for this input is already registered. The contained ID is the
    /// canonical node's ID. The planner shares its slot with this node and emits
    /// no plan step — the executor will resolve this node's ID via the redirect table.
    AlreadyRedirectingTo(usize),
    /// This op is not a redirect; handle it normally.
    NoRedirect,
}

/// Classify `op` as a redirect, a duplicate redirect, or neither.
///
/// Returns [`RedirectKind::RedirectFrom`] if this is the first [`OpKind::AsContiguous`]
/// seen for its input, [`RedirectKind::AlreadyRedirectingTo`] if a canonical redirect
/// for the same input already exists, or [`RedirectKind::NoRedirect`] otherwise.
#[inline]
pub(crate) fn is_a_redirect<T: Copy>(
    op: &OpKind<T>,
    inputs: &[NodeKind<T>],
    id_redirect: &HashMap<usize, usize>,
) -> RedirectKind {
    match op {
        // TODO: We should disregard redirect from a cache in AsContiguous case, when we make cache use a contiguous tensor.
        OpKind::AsContiguous => {
            let id = get_id(&inputs[0]);
            id_redirect
                .get(&id)
                .map_or(RedirectKind::RedirectFrom(id), |id| {
                    RedirectKind::AlreadyRedirectingTo(*id)
                })
        }
        _ => RedirectKind::NoRedirect,
    }
}
