use candela::skeleton::{DynamicSkeleton, Skeleton, SkeletonSlot};
use candela::{Layout, Tensor};
use std::error::Error;

fn build(inputs: &[Layout]) -> Skeleton<f32> {
    let a = SkeletonSlot::new(inputs[0].clone());
    (&a * 2.0).into_skeleton(&[a]).unwrap()
}

fn main() -> Result<(), Box<dyn Error>> {
    let a = Tensor::from_scalar(0.3, &[4]);
    let b = Tensor::from_scalar(0.3, &[8]);

    let sk: DynamicSkeleton<f32> = DynamicSkeleton::new(12, Box::new(build));
    let out_a = sk.run(&[&a])?;
    let out_b = sk.run(&[&b])?;

    println!("{out_a}");
    println!("{out_b}");

    Ok(())
}
