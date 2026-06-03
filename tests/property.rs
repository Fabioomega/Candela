use approx::assert_relative_eq;
use candela::Tensor;
use proptest::prelude::*;

proptest! {
    // a + b == b + a
    #[test]
    fn add_commutative(
        (a, b) in (1usize..=64).prop_flat_map(|n| {
            (
                prop::collection::vec(-1e6f64..1e6, n),
                prop::collection::vec(-1e6f64..1e6, n),
            )
        })
    ) {
        let n = a.len();
        let ta = Tensor::from_slice(&a, &[n]);
        let tb = Tensor::from_slice(&b, &[n]);
        let ab = (ta.clone() + tb.clone()).materialize();
        let ba = (tb + ta).materialize();
        assert_relative_eq!(ab.data(), ba.data(), max_relative = 1e-10);
    }

    // a * b == b * a
    #[test]
    fn mul_commutative(
        (a, b) in (1usize..=64).prop_flat_map(|n| {
            (
                prop::collection::vec(-1e3f64..1e3, n),
                prop::collection::vec(-1e3f64..1e3, n),
            )
        })
    ) {
        let n = a.len();
        let ta = Tensor::from_slice(&a, &[n]);
        let tb = Tensor::from_slice(&b, &[n]);
        let ab = (ta.clone() * tb.clone()).materialize();
        let ba = (tb * ta).materialize();
        assert_relative_eq!(ab.data(), ba.data(), max_relative = 1e-10);
    }

    // a - b == -(b - a)
    #[test]
    fn sub_anticommutative(
        (a, b) in (1usize..=64).prop_flat_map(|n| {
            (
                prop::collection::vec(-1e6f64..1e6, n),
                prop::collection::vec(-1e6f64..1e6, n),
            )
        })
    ) {
        let n = a.len();
        let ta = Tensor::from_slice(&a, &[n]);
        let tb = Tensor::from_slice(&b, &[n]);
        let ab = (ta.clone() - tb.clone()).materialize();
        let neg_ba = ((tb - ta) * -1.0).materialize();
        assert_relative_eq!(ab.data(), neg_ba.data(), max_relative = 1e-10);
    }

    // a + 0 == a
    #[test]
    fn add_zero_identity(a in prop::collection::vec(-1e6f64..1e6, 1..=64)) {
        let n = a.len();
        let t = Tensor::from_slice(&a, &[n]);
        let result = (t + 0.0).materialize();
        assert_relative_eq!(result.data(), a.as_slice(), max_relative = 1e-10);
    }

    // a * 1 == a
    #[test]
    fn mul_one_identity(a in prop::collection::vec(-1e6f64..1e6, 1..=64)) {
        let n = a.len();
        let t = Tensor::from_slice(&a, &[n]);
        let result = (t * 1.0).materialize();
        assert_relative_eq!(result.data(), a.as_slice(), max_relative = 1e-10);
    }

    // Fused scalar chain matches per-element sequential computation
    #[test]
    fn fused_scalar_matches_sequential(
        a in -1e3f64..1e3,
        b in -1e3f64..1e3,
        c in -1e3f64..1e3,
        data in prop::collection::vec(-1e6f64..1e6, 1..=256),
    ) {
        let n = data.len();
        let t = Tensor::from_slice(&data, &[n]);
        let fused = (t * a + b - c).materialize();
        let sequential: Vec<f64> = data.iter().map(|&x| x * a + b - c).collect();
        assert_relative_eq!(
            fused.data(),
            sequential.as_slice(),
            max_relative = 1e-8,
        );
    }

    // View preserves all elements
    #[test]
    fn view_preserves_data(data in prop::collection::vec(any::<f64>(), 12..=12)) {
        let t = Tensor::from_slice(&data, &[12]);
        let viewed = t.view(&[3, 4]).unwrap().materialize();
        assert_eq!(viewed.data(), &data);
    }
}
