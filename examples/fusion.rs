// Demonstrates scalar operator fusion: long chains of compatible scalar ops
// collapse into a single FusedScalar node during graph construction, so
// .materialize() makes exactly one pass over the data regardless of chain length.
use candela::arange;

fn main() {
    // 20 additions, each creating a new graph node. Because consecutive scalar
    // ops fuse during node construction (see OpKind::FusedScalar), the planner
    // sees a single fused node by the time materialize() is called.
    let t = arange!(1000); // [0.0, 1.0, ..., 999.0]
    let mut p = t.to_promise();
    for i in 0..20 {
        p += i as f64;
    }
    // sum(0..20) = 190; every element x becomes x + 190
    let result = p.materialize();
    assert_eq!(result.data()[0], 190.0);
    assert_eq!(result.data()[999], 1189.0);
    println!("first element: {}  (expected 190)", result.data()[0]);

    // Fusion composes with a trailing multiply: (x + 190) * 2 is still one pass.
    let t = arange!(4); // [0.0, 1.0, 2.0, 3.0]
    let mut p = t.to_promise();
    for i in 0..20 {
        p += i as f64;
    }
    let result = (p * 2.0).materialize();
    // [0+190, 1+190, 2+190, 3+190] * 2 = [380, 382, 384, 386]
    assert_eq!(result.data(), &[380.0, 382.0, 384.0, 386.0]);
    println!("fused chain * 2: {:?}", result.data());
}
