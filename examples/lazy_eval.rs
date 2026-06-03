use candela::{Tensor, arange};

fn main() {
    // Arithmetic ops on Tensor return a TensorPromise. Nothing runs until
    // .materialize() triggers the planner and executes the whole plan at once.
    let t = Tensor::from_scalar(3.0_f64, &[4]);
    let result = (t * 2.0 + 1.0).materialize();
    assert_eq!(result.data(), &vec![7.0; 4]);

    // Borrowing &t creates two TensorPromises that both reference the same
    // TensorGraphEdge. The planner deduplicates by node ID so the shared input
    // is computed once and both branches read the same value.
    let t = Tensor::from_vec(vec![0.0_f64, 1.0, 2.0, 3.0], &[4]);
    let lhs = &t * 2.0; // 2t
    let rhs = &t + 1.0; // t + 1
    let result = (lhs - rhs).materialize(); // 2t - (t+1) = t - 1
    assert_eq!(result.data(), &[-1.0, 0.0, 1.0, 2.0]);

    // .cache() keeps the computed result alive in a OnceLock after the first
    // evaluation, making the same intermediate value reusable across multiple
    // independent .materialize() calls without recomputing the inner graph.
    // .get_cache() reads the stored value directly - useful for inspecting what
    // a preprocessing step produced without triggering a new computation.
    let t = arange!(4); // [0.0, 1.0, 2.0, 3.0]
    let preprocessed = (t * 2.0 + 1.0).cache(); // computes [1.0, 3.0, 5.0, 7.0] on first use

    assert!(preprocessed.get_cache().is_none()); // inner graph has not run yet

    let flow_a = (&preprocessed * 10.0).materialize(); // triggers computation
    let flow_b = (&preprocessed - 1.0).materialize(); // reuses cached result directly

    assert_eq!(flow_a.data(), &[10.0, 30.0, 50.0, 70.0]);
    assert_eq!(flow_b.data(), &[0.0, 2.0, 4.0, 6.0]);

    // read the intermediate value that was stored by the first materialization
    let mid = preprocessed.get_cache().unwrap();
    assert_eq!(mid.data(), &[1.0, 3.0, 5.0, 7.0]);
}
