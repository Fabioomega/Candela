// Demonstrates the element-wise activation ops (relu, tanh). They are scalar
// ops, so they fuse with adjacent scalar arithmetic into a single FusedScalar
// pass instead of becoming separate graph nodes.

use candela::Tensor;

fn main() {
    // --- ReLU: max(x, 0), clamps negatives to zero ---
    let t = Tensor::from_slice(&[-2.0_f64, -0.5, 0.0, 1.5, 3.0], &[5]);
    let relu = t.relu().materialize();
    assert_eq!(relu.data(), &[0.0, 0.0, 0.0, 1.5, 3.0]);
    println!("relu:  {:?}", relu.data());

    // --- tanh: squashes every value into (-1, 1) ---
    // tanh(0) == 0 exactly; the rest land strictly inside the open interval.
    let t = Tensor::from_slice(&[-3.0_f64, 0.0, 3.0], &[3]);
    let tanh = t.tanh().materialize();
    assert_eq!(tanh.data()[1], 0.0);
    assert!(tanh.data().iter().all(|&x| x > -1.0 && x < 1.0));
    println!("tanh:  {:?}", tanh.data());

    // --- Fusion: a scalar op feeding an activation is one pass over the data ---
    // (x * 2 - 1) then relu. The multiply, subtract, and relu collapse into a
    // single FusedScalar node; the planner sees one op, not three.
    let t = Tensor::from_slice(&[0.0_f64, 0.25, 0.5, 1.0], &[4]);
    let fused = ((t * 2.0 - 1.0).relu()).materialize();
    // 2x - 1 = [-1, -0.5, 0, 1]  ->  relu = [0, 0, 0, 1]
    assert_eq!(fused.data(), &[0.0, 0.0, 0.0, 1.0]);
    println!("fused: {:?}", fused.data());
}
