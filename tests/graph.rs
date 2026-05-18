mod common;

use candela::{Tensor, arange, ones};
use common::assert_approx_eq;

// ── shared node (diamond graph) ───────────────────────────────────────────────

#[test]
fn shared_node_computed_once() {
    // t shared across two branches: t*2 - t = t
    let t = arange!(4);
    let p = t.as_promise();
    let lhs = &p * 2.0;
    let rhs = p.clone();
    let result = (lhs - rhs).materialize();
    // (2x - x) = x = [0, 1, 2, 3]
    assert_approx_eq(result.data(), &[0.0, 1.0, 2.0, 3.0]);
}

#[test]
fn diamond_graph_correctness() {
    // t shared: lhs = t * 2, rhs = t + 1, result = lhs - rhs = (2x) - (x+1) = x - 1
    let t = arange!(4);
    let p = t.as_promise();
    let lhs = &p * 2.0;
    let rhs = &p + 1.0;
    let result = (lhs - rhs).materialize();
    // [0,1,2,3] → [-1, 0, 1, 2]
    assert_approx_eq(result.data(), &[-1.0, 0.0, 1.0, 2.0]);
}

// ── CachedTensorPromise ───────────────────────────────────────────────────────

#[test]
fn cached_promise_stable_results() {
    // Two separate materializations using the same cached node yield the same data
    let t = arange!(4);
    let cached = (t + 1.0).cache();
    let r1 = (&cached * 2.0).materialize();
    let r2 = (&cached * 2.0).materialize();
    assert_eq!(r1.data(), r2.data());
}

#[test]
fn cached_promise_cache_empty_before_materialize() {
    let t = ones!(&[4]);
    let cached = (t + 1.0).cache();
    assert!(cached.get_cache().is_none());
}

#[test]
fn cached_promise_cache_filled_after_materialize() {
    let t = ones!(&[4]);
    let cached = (t + 1.0).cache();
    let _ = (&cached + 0.0).materialize(); // triggers cache fill
    assert!(cached.get_cache().is_some());
}

#[test]
fn cached_promise_feeds_two_downstream_graphs() {
    // cached = arange(4) + 10 = [10,11,12,13]
    // r1 = cached * 2 = [20,22,24,26]
    // r2 = cached + 1 = [11,12,13,14]
    let t = arange!(4);
    let cached = (t + 10.0).cache();
    let r1 = (&cached * 2.0).materialize();
    let r2 = (&cached + 1.0).materialize();
    assert_approx_eq(r1.data(), &[20.0, 22.0, 24.0, 26.0]);
    assert_approx_eq(r2.data(), &[11.0, 12.0, 13.0, 14.0]);
}

// ── Tensor clone ─────────────────────────────────────────────────────────────

#[test]
fn clone_shared_buffer() {
    // Clone shares the same underlying buffer pointer
    let a: Tensor<f64> = arange!(4);
    let b = a.clone();
    assert_eq!(a.data().as_ptr(), b.data().as_ptr());
}

#[test]
fn clone_deep_separate_buffer() {
    // clone_deep allocates a fresh buffer with same values
    let a = Tensor::from_slice(&[1.0, 2.0, 3.0], &[3]);
    let b = a.clone_deep();
    assert_ne!(a.data().as_ptr(), b.data().as_ptr());
    assert_eq!(a.data(), b.data());
}
