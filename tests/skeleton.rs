mod common;

use candela::{FloatLikeTensorElement, Tensor, arange};
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
