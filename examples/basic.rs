// A first look at Candela: create tensors, do some arithmetic, and run it.
// If you read only one example, read this one - it covers the whole loop from
// "make a tensor" to "get numbers back".

use candela::{Dimension, Tensor, arange, ones, zeros};

fn main() {
    // --- Creating tensors ---
    // A tensor has data, a shape, and a stride. Here are a few ways to make one.
    let filled = Tensor::from_scalar(2.0_f64, &[2, 3]); // every element is 2.0
    let zeros = zeros!(&[2, 3]); // all zeros
    let ones = ones!(&[4]); // all ones
    let counted: Tensor<f64> = arange!(6); // [0, 1, 2, 3, 4, 5]

    assert_eq!(filled.shape(), &[2, 3]);
    assert_eq!(zeros.data(), &[0.0; 6]);
    assert_eq!(ones.data(), &[1.0; 4]);
    assert_eq!(counted.data(), &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);

    // --- Arithmetic builds a graph; nothing runs yet ---
    // Operators on a Tensor return a TensorPromise: a description of work to do,
    // not the result. You chain ops freely, then call .materialize() once to run
    // the whole thing in a single planned pass.
    let promise = counted * 2.0 + 1.0; // 2x + 1, still not computed
    let result = promise.materialize(); // now it actually runs
    assert_eq!(result.data(), &[1.0, 3.0, 5.0, 7.0, 9.0, 11.0]);
    println!("2x + 1 = {:?}", result.data());

    // --- Reshaping is zero-copy ---
    // view() reinterprets the same buffer with a new shape - no data is moved.
    let seq: Tensor<f64> = arange!(6); // annotate once, then chain freely
    let matrix = seq.view(&[2, 3]).unwrap().materialize();
    assert_eq!(matrix.shape(), &[2, 3]);
    println!("as a 2x3 matrix:\n{}", matrix);

    // --- A reduction ---
    // Sum every element down to a single value.
    let nums: Tensor<f64> = arange!(5);
    let total = nums.sum().materialize(); // 0+1+2+3+4
    assert_eq!(total.data(), &[10.0]);
    println!("sum(0..5) = {:?}", total.data());
}
