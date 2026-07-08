mod common;

use candela::{FloatLikeTensorElement, Tensor, arange, ones};
use common::assert_approx_eq;
use rstest::rstest;

// ── shared node (diamond graph) ───────────────────────────────────────────────

#[rstest]
#[case::f64(0.0f64)]
#[case::f32(0.0f32)]
fn shared_node_computed_once<T: FloatLikeTensorElement>(#[case] _t: T) {
    // t shared across two branches: t*2 - t = t
    let t: Tensor<T> = arange!(4);
    let p = t.to_promise();
    let lhs = &p * T::from_f64(2.0);
    let rhs = p.clone();
    let result = (lhs - rhs).materialize();
    // (2x - x) = x = [0, 1, 2, 3]
    assert_approx_eq(result.data(), &[0.0, 1.0, 2.0, 3.0]);
}

#[rstest]
#[case::f64(0.0f64)]
#[case::f32(0.0f32)]
fn diamond_graph_correctness<T: FloatLikeTensorElement>(#[case] _t: T) {
    // t shared: lhs = t * 2, rhs = t + 1, result = lhs - rhs = (2x) - (x+1) = x - 1
    let t: Tensor<T> = arange!(4);
    let p = t.to_promise();
    let lhs = &p * T::from_f64(2.0);
    let rhs = &p + T::from_f64(1.0);
    let result = (lhs - rhs).materialize();
    // [0,1,2,3] → [-1, 0, 1, 2]
    assert_approx_eq(result.data(), &[-1.0, 0.0, 1.0, 2.0]);
}

// ── CachedTensorPromise ───────────────────────────────────────────────────────

#[rstest]
#[case::f64(0.0f64)]
#[case::f32(0.0f32)]
fn cached_promise_stable_results<T: FloatLikeTensorElement>(#[case] _t: T) {
    // Two separate materializations using the same cached node yield the same data
    let t: Tensor<T> = arange!(4);
    let cached = (t + T::from_f64(1.0)).cache();
    let r1 = (&cached * T::from_f64(2.0)).materialize();
    let r2 = (&cached * T::from_f64(2.0)).materialize();
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

#[rstest]
#[case::f64(0.0f64)]
#[case::f32(0.0f32)]
fn cached_promise_feeds_two_downstream_graphs<T: FloatLikeTensorElement>(#[case] _t: T) {
    // cached = arange(4) + 10 = [10,11,12,13]
    // r1 = cached * 2 = [20,22,24,26]
    // r2 = cached + 1 = [11,12,13,14]
    let t: Tensor<T> = arange!(4);
    let cached = (t + T::from_f64(10.0)).cache();
    let r1 = (&cached * T::from_f64(2.0)).materialize();
    let r2 = (&cached + T::from_f64(1.0)).materialize();
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
fn deep_clone_separate_buffer() {
    // deep_clone allocates a fresh buffer with same values
    let a = Tensor::from_slice(&[1.0, 2.0, 3.0], &[3]);
    let b = a.deep_clone();
    assert_ne!(a.data().as_ptr(), b.data().as_ptr());
    assert_eq!(a.data(), b.data());
}
