mod common;

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use candela::backend::DefaultBackend;
use candela::skeleton::{BuildFunction, DynamicSkeleton, SkeletonSlot, UnboundedDynamicSkeleton};
use candela::{Dimension, FloatLikeTensorElement, Layout, Tensor, arange};
use common::{assert_approx_eq, assert_approx_eq_by, tensor_of};
use rstest::rstest;

// A `x * 2` build function over a single slot, paired with a counter that records
// how many times the cache asked it to build. The counter is how every test below
// distinguishes a cache hit from a miss.
fn counting_double<T: FloatLikeTensorElement>()
-> (BuildFunction<T, DefaultBackend>, Arc<AtomicUsize>) {
    let calls = Arc::new(AtomicUsize::new(0));
    let seen = calls.clone();
    let build: BuildFunction<T, DefaultBackend> = Box::new(move |inputs: &[Layout]| {
        seen.fetch_add(1, Ordering::Relaxed);
        let a = SkeletonSlot::new(inputs[0].clone());
        (&a * T::from_f64(2.0))
            .into_skeleton(std::slice::from_ref(&a))
            .unwrap()
    });
    (build, calls)
}

// An `a + b` build function over two slots, so the cache key is the pair of layouts.
fn counting_add<T: FloatLikeTensorElement>() -> (BuildFunction<T, DefaultBackend>, Arc<AtomicUsize>)
{
    let calls = Arc::new(AtomicUsize::new(0));
    let seen = calls.clone();
    let build: BuildFunction<T, DefaultBackend> = Box::new(move |inputs: &[Layout]| {
        seen.fetch_add(1, Ordering::Relaxed);
        let a = SkeletonSlot::new(inputs[0].clone());
        let b = SkeletonSlot::new(inputs[1].clone());
        (&a + &b).into_skeleton(&[a, b]).unwrap()
    });
    (build, calls)
}

// ── run ─────────────────────────────────────────────────────────────────────────

#[rstest]
#[case::f64(0.0f64)]
#[case::f32(0.0f32)]
fn dynamic_dispatch_by_shape<T: FloatLikeTensorElement>(#[case] _t: T) {
    // Two distinct shapes each build their own skeleton and compute correctly.

    let a: Tensor<T> = arange!(4);
    let b: Tensor<T> = arange!(8);

    let (build, calls) = counting_double::<T>();
    let sk: DynamicSkeleton<T> = DynamicSkeleton::new(4, build);

    let out_a = sk.run(&[&a]).unwrap();
    let out_b = sk.run(&[&b]).unwrap();

    assert_eq!(calls.load(Ordering::Relaxed), 2);
    assert_approx_eq(out_a.data(), &[0.0, 2.0, 4.0, 6.0]);
    assert_approx_eq(out_b.data(), &[0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0]);
}

#[test]
fn dynamic_repeated_shape() {
    // The same shape twice reuses the cached skeleton; build runs only once.

    let a: Tensor<f64> = arange!(4);

    let (build, calls) = counting_double::<f64>();
    let sk: DynamicSkeleton<f64> = DynamicSkeleton::new(4, build);

    sk.run(&[&a]).unwrap();
    sk.run(&[&a]).unwrap();

    assert_eq!(calls.load(Ordering::Relaxed), 1);
}

#[rstest]
#[case::f64(0.0f64)]
#[case::f32(0.0f32)]
fn dynamic_same_layout_distinct_data<T: FloatLikeTensorElement>(#[case] _t: T) {
    // Same layout, different values: the second run hits the cache yet the plan is
    // re-bound to the new input rather than replaying the first output.

    let x: Tensor<T> = tensor_of(&[1.0, 2.0, 3.0, 4.0], &[4]);
    let y: Tensor<T> = tensor_of(&[5.0, 6.0, 7.0, 8.0], &[4]);

    let (build, calls) = counting_double::<T>();
    let sk: DynamicSkeleton<T> = DynamicSkeleton::new(4, build);

    let out_x = sk.run(&[&x]).unwrap();
    let out_y = sk.run(&[&y]).unwrap();

    assert_eq!(calls.load(Ordering::Relaxed), 1);
    assert_approx_eq(out_x.data(), &[2.0, 4.0, 6.0, 8.0]);
    assert_approx_eq(out_y.data(), &[10.0, 12.0, 14.0, 16.0]);
}

#[test]
fn dynamic_multi_input_key() {
    // The key is the whole sequence of input layouts: a differing input forces a
    // rebuild, a repeated pair hits the cache.

    let a4: Tensor<f64> = arange!(4);
    let b4: Tensor<f64> = Tensor::from_scalar(1.0, &[4]);
    let a8: Tensor<f64> = arange!(8);
    let b8: Tensor<f64> = Tensor::from_scalar(1.0, &[8]);

    let (build, calls) = counting_add::<f64>();
    let sk: DynamicSkeleton<f64> = DynamicSkeleton::new(4, build);

    let out4 = sk.run(&[&a4, &b4]).unwrap();
    sk.run(&[&a4, &b4]).unwrap();
    let out8 = sk.run(&[&a8, &b8]).unwrap();

    assert_eq!(calls.load(Ordering::Relaxed), 2);
    assert_approx_eq(out4.data(), &[1.0, 2.0, 3.0, 4.0]);
    assert_approx_eq(out8.data(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
}

// ── compose ─────────────────────────────────────────────────────────────────────

#[rstest]
#[case::f64(0.0f64)]
#[case::f32(0.0f32)]
fn dynamic_compose_tensor<T: FloatLikeTensorElement>(#[case] _t: T) {
    // `compose` and `run` agree for the same cached skeleton.

    let a: Tensor<T> = arange!(4);

    let (build, _calls) = counting_double::<T>();
    let sk: DynamicSkeleton<T> = DynamicSkeleton::new(4, build);

    let run_output = sk.run(&[&a]).unwrap();
    let composed = sk.compose(&[&a]).unwrap().to_promise().materialize();

    assert_approx_eq_by(run_output.data(), composed.data(), 1e-6);
}

// ── containment ─────────────────────────────────────────────────────────────────

#[test]
fn dynamic_contains_key() {
    // A layout is absent before its first run and present afterwards; the tensor and
    // layout views agree.

    let a: Tensor<f64> = arange!(4);

    let (build, _calls) = counting_double::<f64>();
    let sk: DynamicSkeleton<f64> = DynamicSkeleton::new(4, build);

    assert!(!sk.contains_key(&[&a]));
    assert!(!sk.contains_key_by_layout(&[a.layout().clone()]));

    sk.run(&[&a]).unwrap();

    assert!(sk.contains_key(&[&a]));
    assert!(sk.contains_key_by_layout(&[a.layout().clone()]));
}

// ── remove ──────────────────────────────────────────────────────────────────────

#[test]
fn dynamic_remove_present() {
    // Removing a cached entry returns it, drops it from the cache, and forces the
    // next run to rebuild.

    let a: Tensor<f64> = arange!(4);

    let (build, calls) = counting_double::<f64>();
    let sk: DynamicSkeleton<f64> = DynamicSkeleton::new(4, build);

    sk.run(&[&a]).unwrap();
    assert_eq!(calls.load(Ordering::Relaxed), 1);

    assert!(sk.remove(&[&a]).is_some());
    assert!(!sk.contains_key(&[&a]));

    sk.run(&[&a]).unwrap();
    assert_eq!(calls.load(Ordering::Relaxed), 2);
}

#[test]
fn dynamic_remove_by_layout() {
    // The layout-keyed removal behaves like the tensor-keyed one.

    let a: Tensor<f64> = arange!(4);

    let (build, _calls) = counting_double::<f64>();
    let sk: DynamicSkeleton<f64> = DynamicSkeleton::new(4, build);

    sk.run(&[&a]).unwrap();

    assert!(sk.remove_by_layout(&[a.layout().clone()]).is_some());
    assert!(!sk.contains_key(&[&a]));
}

#[test]
fn dynamic_remove_absent() {
    let a: Tensor<f64> = arange!(4);

    let (build, _calls) = counting_double::<f64>();
    let sk: DynamicSkeleton<f64> = DynamicSkeleton::new(4, build);

    assert!(sk.remove(&[&a]).is_none());
}

// ── eviction ────────────────────────────────────────────────────────────────────

#[test]
fn dynamic_over_capacity() {
    // With room for two shapes, inserting a third evicts the least-recently-used one.

    let a: Tensor<f64> = arange!(4);
    let b: Tensor<f64> = arange!(8);
    let c: Tensor<f64> = arange!(12);

    let (build, calls) = counting_double::<f64>();
    let sk: DynamicSkeleton<f64> = DynamicSkeleton::new(2, build);

    sk.run(&[&a]).unwrap();
    sk.run(&[&b]).unwrap();
    sk.run(&[&c]).unwrap();

    assert_eq!(calls.load(Ordering::Relaxed), 3);
    assert!(!sk.contains_key(&[&a]));
    assert!(sk.contains_key(&[&b]));
    assert!(sk.contains_key(&[&c]));
}

#[test]
fn dynamic_lru_recency() {
    // Re-accessing `a` moves it back to most-recent, so the next insertion evicts `b`
    // instead.

    let a: Tensor<f64> = arange!(4);
    let b: Tensor<f64> = arange!(8);
    let c: Tensor<f64> = arange!(12);

    let (build, _calls) = counting_double::<f64>();
    let sk: DynamicSkeleton<f64> = DynamicSkeleton::new(2, build);

    sk.run(&[&a]).unwrap();
    sk.run(&[&b]).unwrap();
    sk.run(&[&a]).unwrap();
    sk.run(&[&c]).unwrap();

    assert!(sk.contains_key(&[&a]));
    assert!(!sk.contains_key(&[&b]));
    assert!(sk.contains_key(&[&c]));
}

#[test]
fn dynamic_unbounded() {
    // The unbounded policy keeps every distinct shape past its initial size.

    let (build, calls) = counting_double::<f64>();
    let sk: UnboundedDynamicSkeleton<f64, DefaultBackend> = DynamicSkeleton::new(1, build);

    let tensors: Vec<Tensor<f64>> = (1..=5)
        .map(|n| Tensor::from_scalar(1.0, &[n * 4]))
        .collect();

    for t in &tensors {
        sk.run(&[t]).unwrap();
    }

    assert_eq!(calls.load(Ordering::Relaxed), 5);
    for t in &tensors {
        assert!(sk.contains_key(&[t]));
    }
}
