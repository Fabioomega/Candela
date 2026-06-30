mod common;

use candela::{FloatLikeTensorElement, OpError, Tensor, arange, srange};
use common::assert_approx_eq;
use rstest::rstest;

use crate::common::assert_approx_eq_by;

// ── run ───────────────────────────────────────────────────────────────────────

#[rstest]
#[case::f64(0.0f64)]
#[case::f32(0.0f32)]
fn skeleton_scalar_chain<T: FloatLikeTensorElement>(#[case] _t: T) {
    // (s + s) * 0.5 + 1 = s + 1

    let base: Tensor<T> = arange!(4);
    let slot = base.as_slot();
    let sk = ((&slot + &slot) * T::from_f64(0.5) + T::from_f64(1.0))
        .into_skeleton(std::slice::from_ref(&slot))
        .unwrap();

    let out = sk.run(&[&arange!(4)]).unwrap();
    assert_approx_eq(out.data(), &[1.0, 2.0, 3.0, 4.0]);
}

#[rstest]
#[case::f64(0.0f64)]
#[case::f32(0.0f32)]
fn skeleton_repeated_runs<T: FloatLikeTensorElement>(#[case] _t: T) {
    // The same compiled plan executed against different inputs.

    let base: Tensor<T> = arange!(4);
    let slot = base.as_slot();
    let sk = ((&slot + &slot) * T::from_f64(0.5) + T::from_f64(1.0))
        .into_skeleton(std::slice::from_ref(&slot))
        .unwrap();

    let first = sk.run(&[&arange!(4)]).unwrap();
    let second = sk
        .run(&[&Tensor::from_scalar(T::from_f64(1.0), &[4])])
        .unwrap();

    assert_approx_eq(first.data(), &[1.0, 2.0, 3.0, 4.0]);
    assert_approx_eq(second.data(), &[2.0, 2.0, 2.0, 2.0]);
}

#[rstest]
#[case::f64(0.0f64)]
#[case::f32(0.0f32)]
fn skeleton_matmul_reduction<T: FloatLikeTensorElement>(#[case] _t: T) {
    // A plan carrying a BLAS matmul into a reduction: ([2, 3] @ [3, 2]) summed
    // along axis 0. We are comparing against the eager graph.

    let a: Tensor<T> = srange!(6, &[2, 3]);
    let b: Tensor<T> = srange!(6, &[3, 2]);

    let slot_a = a.as_slot();
    let slot_b = b.as_slot();

    let sk = slot_a
        .matmul(&slot_b)
        .unwrap()
        .sum_axis(0, false)
        .unwrap()
        .into_skeleton(&[slot_a, slot_b])
        .unwrap();

    let run_output = sk.run(&[&a, &b]).unwrap();
    let expected = a
        .matmul(&b)
        .unwrap()
        .sum_axis(0, false)
        .unwrap()
        .materialize();

    assert_approx_eq_by(run_output.data(), expected.data(), 1e-6);
}

// ── compose ───────────────────────────────────────────────────────────────────

#[rstest]
#[case::f64(0.0f64)]
#[case::f32(0.0f32)]
fn skeleton_compose_tensor<T: FloatLikeTensorElement>(#[case] _t: T) {
    let base: Tensor<T> = arange!(4);

    let slot = base.as_slot();
    let slot2 = base.as_slot();

    let sk = (&slot + &slot2).into_skeleton(&[slot, slot2]).unwrap();

    let run_output = sk.run(&[&base, &base]).unwrap();
    let composed = sk
        .compose(&[&base, &base])
        .unwrap()
        .as_promise()
        .materialize();

    assert_approx_eq_by(run_output.data(), composed.data(), 1e-6);
}

#[rstest]
#[case::f64(0.0f64)]
#[case::f32(0.0f32)]
fn skeleton_compose_promise<T: FloatLikeTensorElement>(#[case] _t: T) {
    let base: Tensor<T> = arange!(4);
    let base2: Tensor<T> = Tensor::from_scalar(T::from_f64(1.0), &[4]);

    let promise = (&base + T::from_f64(3.0)).log2();
    let promise_output = promise.clone_and_materialize();

    let promise2 = (&base2 + T::from_f64(3.0)).log2();
    let promise_output2 = promise2.clone_and_materialize();

    let slot = base.as_slot();
    let slot2 = slot.deep_clone();

    let sk = (&slot + &slot2).into_skeleton(&[slot, slot2]).unwrap();

    let run_output = sk.run(&[&promise_output, &promise_output2]).unwrap();
    let composed = sk
        .compose(&[&promise, &promise2])
        .unwrap()
        .as_promise()
        .materialize();

    assert_approx_eq_by(run_output.data(), composed.data(), 1e-6);
}

#[rstest]
#[case::f64(0.0f64)]
#[case::f32(0.0f32)]
fn skeleton_compose_promise_same_input<T: FloatLikeTensorElement>(#[case] _t: T) {
    let base: Tensor<T> = arange!(4);
    let promise = (&base + T::from_f64(3.0)).log2();
    let promise_output = promise.clone_and_materialize();

    let slot = base.as_slot();
    let slot2 = slot.deep_clone();

    let sk = (&slot + &slot2).into_skeleton(&[slot, slot2]).unwrap();

    let run_output = sk.run(&[&promise_output, &promise_output]).unwrap();
    let composed = sk
        .compose(&[&promise, &promise])
        .unwrap()
        .as_promise()
        .materialize();

    assert_approx_eq_by(run_output.data(), composed.data(), 1e-6);
}

// ── errors ─────────────────────────────────────────────────────────────────────

#[test]
fn skeleton_slot_count_mismatch() {
    // The skeleton declares one slot; running it with two inputs is rejected.

    let base: Tensor<f64> = arange!(4);
    let slot = base.as_slot();
    let sk = (&slot * 2.0)
        .into_skeleton(std::slice::from_ref(&slot))
        .unwrap();

    assert!(matches!(
        sk.run(&[&base, &base]),
        Err(OpError::IncorrectSlotAmount(1, 2))
    ));
}

#[test]
fn skeleton_unused_slot() {
    // The expression only references `used`, so declaring the unrelated `unused`
    // slot (same layout, distinct identity) does not match the graph.

    let base: Tensor<f64> = arange!(4);
    let used = base.as_slot();
    let unused = base.as_slot();

    assert!(matches!(
        (&used * 2.0).into_skeleton(std::slice::from_ref(&unused)),
        Err(OpError::NotSameSlot(_))
    ));
}

#[test]
fn skeleton_layout_mismatch() {
    // The slot was declared for shape [4]; a [8] input has an incompatible layout.

    let base: Tensor<f64> = arange!(4);
    let slot = base.as_slot();
    let sk = (&slot * 2.0)
        .into_skeleton(std::slice::from_ref(&slot))
        .unwrap();

    let wrong: Tensor<f64> = arange!(8);
    assert!(matches!(
        sk.run(&[&wrong]),
        Err(OpError::NotSameLayoutAtSlot(0))
    ));
}
