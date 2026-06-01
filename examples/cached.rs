// Demonstrates CachedTensorPromise: a promise that stores its computed result
// after the first .materialize() and reuses it in every subsequent call without
// re-running the inner graph.
//
// The canonical use case is a preprocessing step that is shared across multiple
// independent downstream flows called at different points in time - like a bias
// correction or a feature normalisation that feeds several separate computations.

use candela::Tensor;

fn main() {
    let raw = Tensor::from_vec(vec![10.0_f64, 20.0, 30.0, 40.0], &[4]);

    // Build the preprocessing step and register it as cached.
    // Nothing runs here - the inner graph (raw - 2.0) is not executed yet.
    let preprocessed = (raw - 2.0).cache();

    // The cache starts empty.
    assert!(preprocessed.get_cache().is_none());

    // --- First downstream flow ---
    // Building &preprocessed * 2.0 links the cached node into a new graph.
    // Calling .materialize() runs the preprocessing for the first time:
    //   raw - 2.0  =  [8.0, 18.0, 28.0, 38.0]  ← stored in the cache
    //   * 2.0      =  [16.0, 36.0, 56.0, 76.0]  ← returned
    let flow_a = (&preprocessed * 2.0).materialize();
    assert_eq!(flow_a.data(), &[16.0, 36.0, 56.0, 76.0]);
    println!("flow_a: {:?}", flow_a.data());

    // The cache is now warm.
    assert!(preprocessed.get_cache().is_some());

    // --- Second downstream flow, called some time later ---
    // The preprocessing step is not re-run. The planner sees the cache is filled
    // and reads [8.0, 18.0, 28.0, 38.0] directly, skipping the inner graph entirely.
    let flow_b = (&preprocessed + 100.0).materialize();
    assert_eq!(flow_b.data(), &[108.0, 118.0, 128.0, 138.0]);
    println!("flow_b: {:?}", flow_b.data());

    // --- Inspecting the stored intermediate ---
    // get_cache() exposes whatever the first materialization computed and stored.
    // Useful for debugging a preprocessing step without triggering a new computation.
    let stored = preprocessed.get_cache().unwrap();
    assert_eq!(stored.data(), &[8.0, 18.0, 28.0, 38.0]);
    println!("cached intermediate: {:?}", stored.data());
}
