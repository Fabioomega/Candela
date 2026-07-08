// ── memory report ─────────────────────────────────────────────────────────────

use candela::{Tensor, arange, ones};

#[test]
fn memory_report_subtract_intermediate() {
    // a + b -> t (1 allocation)
    // t - b -> y (no allocations)

    let a: Tensor<f32> = arange!(4);
    let b: Tensor<f32> = ones!(&[4]);
    let sa = a.to_slot();
    let sb = b.to_slot();
    let sk = ((&sa + &sb) - &sb).into_skeleton(&[sa, sb]).unwrap();

    assert_eq!(sk.memory_report().total_number_of_allocations, 1);
}

#[test]
fn memory_report_aliased_operand() {
    // s + s -> t (1 allocation)
    // (t + t) -> y (2 allocation) (same buffer in 2 operands cannot be reused)

    let base: Tensor<f32> = arange!(4);
    let slot = base.to_slot();
    let t = &slot + &slot;
    let sk = (&t + &t)
        .into_skeleton(std::slice::from_ref(&slot))
        .unwrap();

    assert_eq!(sk.memory_report().total_number_of_allocations, 2);
}

// ── in-place memory reuse ─────────────────────────────────────────────────────────────

#[test]
fn memory_report_scalar_chain() {
    // (s + s) -> t (1 allocation)
    // (t * 0.5 + 1.0) -> collapses to f(x) = 0.5*x + 1.0 ; f(t) -> y (reuses allocation)

    let base: Tensor<f32> = arange!(4);
    let slot = base.to_slot();
    let sk = ((&slot + &slot) * 0.5 + 1.0)
        .into_skeleton(std::slice::from_ref(&slot))
        .unwrap();

    assert_eq!(sk.memory_report().total_number_of_allocations, 1);
}
