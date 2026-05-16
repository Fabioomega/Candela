// Demonstrates Candela's matrix multiplication: basic 2D matmul, batched matmul
// over a leading batch dimension, and batch broadcasting where one operand's
// batch size is 1.
//
// Matmul is the only built-in op that requires contiguous memory. Non-contiguous
// inputs (transposed, sliced, etc.) are packed automatically via an AsContiguous
// node before the BLAS call — zero extra effort from the caller.

use candela::{Dimension, Tensor};

fn main() {
    // --- Basic 2D matmul ---
    //
    // A [2,3] @ B [3,2] = C [2,2]
    //
    //   A = [[1, 2, 3],   B = [[ 7,  8],   C = [[ 58,  64],
    //        [4, 5, 6]]        [ 9, 10],        [139, 154]]
    //                         [11, 12]]
    //
    // C[0,0] = 1*7 + 2*9 + 3*11 = 58
    let a = Tensor::from_slice(&[1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let b = Tensor::from_slice(&[7.0_f64, 8.0, 9.0, 10.0, 11.0, 12.0], &[3, 2]);
    let c = a.matmul(&b).unwrap().materialize();
    assert_eq!(c.shape(), &[2, 2]);
    assert_eq!(c.data(), &[58.0, 64.0, 139.0, 154.0]);
    println!("2D matmul result: {:?}", c.data());

    // --- Batched matmul ---
    //
    // A [2,3,4] @ B [2,4,5] = C [2,3,5]
    //
    // The leading dimension is a batch index. Slice i of C equals slice i of A
    // multiplied by slice i of B — each pair is an independent [3,4] @ [4,5] matmul.
    //
    // Using all-ones matrices: every output element is the inner product of four
    // ones, so every element of C equals 4.0.
    let a = Tensor::from_scalar(1.0_f64, &[2, 3, 4]);
    let b = Tensor::from_scalar(1.0_f64, &[2, 4, 5]);
    let c = a.matmul(&b).unwrap().materialize();
    assert_eq!(c.shape(), &[2, 3, 5]);
    assert!(c.data().iter().all(|&x| x == 4.0));
    println!("batched matmul shape: {:?}", c.shape()); // [2, 3, 5]

    // --- Batch broadcasting: single lhs broadcast across rhs batches ---
    //
    // A [1,3,4] @ B [2,4,5] = C [2,3,5]
    //
    // The single A matrix is broadcast across both B slices. The result is
    // identical to stacking two copies of A and running a standard batched matmul.
    let a = Tensor::from_scalar(1.0_f64, &[1, 3, 4]);
    let b = Tensor::from_scalar(1.0_f64, &[2, 4, 5]);
    let c = a.matmul(&b).unwrap().materialize();
    assert_eq!(c.shape(), &[2, 3, 5]);
    assert!(c.data().iter().all(|&x| x == 4.0));
    println!("broadcast lhs batch shape: {:?}", c.shape()); // [2, 3, 5]

    // --- Batch broadcasting: single rhs broadcast across lhs batches ---
    //
    // A [2,3,4] @ B [1,4,5] = C [2,3,5]
    //
    // Symmetric case: each lhs slice is multiplied by the single rhs matrix.
    let a = Tensor::from_scalar(1.0_f64, &[2, 3, 4]);
    let b = Tensor::from_scalar(1.0_f64, &[1, 4, 5]);
    let c = a.matmul(&b).unwrap().materialize();
    assert_eq!(c.shape(), &[2, 3, 5]);
    assert!(c.data().iter().all(|&x| x == 4.0));
    println!("broadcast rhs batch shape: {:?}", c.shape()); // [2, 3, 5]
}
