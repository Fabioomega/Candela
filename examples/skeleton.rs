use candela::skeleton::SkeletonSlot;
use candela::{Layout, Tensor};
use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    // A slot: a typed hole with a layout but no data
    let slot = SkeletonSlot::new(Layout::new(&[4]));

    // Build the plan once; planning happens here
    let skeleton = (&slot * 2.0 + 1.0).into_skeleton(std::slice::from_ref(&slot))?;

    // Run it on different inputs - no planning, just execution
    let a = skeleton.run(&[&Tensor::from_slice(&[0.0, 1.0, 2.0, 3.0], &[4])])?;
    let b = skeleton.run(&[&Tensor::from_scalar(5.0, &[4])])?;
    println!("a: {a}");
    println!("b: {b}");

    // Inputs must match the slot's exact layout
    assert!(skeleton.run(&[&Tensor::from_scalar(1.0, &[8])]).is_err());

    // compose splices the skeleton into a bigger graph as a BakedPromise
    let x = SkeletonSlot::new(Layout::new(&[4]));
    let y = x.deep_clone();
    let sum = (&x + &y).into_skeleton(&[x, y])?;

    let baked = sum.compose(&[
        &Tensor::from_scalar(1.0, &[4]),
        &Tensor::from_scalar(10.0, &[4]),
    ])?;
    let result = (baked * 2.0).materialize();
    println!("composed: {result}");

    Ok(())
}
