use candela::Tensor;
use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    // Creates tensors
    let a = Tensor::from_scalar(0.3, &[4]);
    let b = Tensor::from_scalar(0.3, &[8]);

    // Creates a slot for a tensor with the same shape as a
    let slot = a.as_slot();

    // Create a skeleton with that slot
    let skeleton = (&slot * 2.0 + 1.0).log2().into_skeleton(&[slot]).unwrap();

    // Running the skeleton
    let output_a = skeleton.run(&[&a]);

    // Running the skeleton for an invalid shape
    let output_b = skeleton.run(&[&b]);

    // Check the output of a
    println!("{}", output_a.unwrap());

    // Check the output is an error
    assert!(output_b.is_err());

    Ok(())
}
