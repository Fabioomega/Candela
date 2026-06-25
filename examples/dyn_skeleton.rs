use candela::skeleton::{DynamicSkeleton, Skeleton, SkeletonSlot};
use candela::{Layout, Tensor};
use std::error::Error;

// Creates a build function that is called every
// time a skeleton encounters a unknown Layout
fn build(inputs: &[Layout]) -> Skeleton<f32> {
    let a = SkeletonSlot::new(inputs[0].clone());
    (&a * 2.0).into_skeleton(&[a]).unwrap()
}

fn main() -> Result<(), Box<dyn Error>> {
    // Creates tensors
    let a = Tensor::from_scalar(0.3, &[4]);
    let b = Tensor::from_scalar(0.3, &[8]);

    // Creates a dynamic skeleton hashmap with a cache
    // of size 12 and using the build function.
    let sk: DynamicSkeleton<f32> = DynamicSkeleton::new(12, Box::new(build));

    // Skeleton does not contain a layout like a
    assert!(!sk.contains_key(&[&a]));
    // Calculates the output of running a trough the skeleton
    let out_a = sk.run(&[&a])?;
    // Skeleton now contains a
    assert!(sk.contains_key(&[&a]));

    // Skeleton does not contain a layout like b
    assert!(!sk.contains_key(&[&b]));
    // Calculates the output of running a trough the skeleton
    let out_b = sk.run(&[&b])?;
    // Skeleton now contains b
    assert!(sk.contains_key(&[&b]));

    println!("{out_a}");
    println!("{out_b}");

    Ok(())
}
