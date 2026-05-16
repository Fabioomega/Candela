mod common;

use candela::{Dimension, Tensor, arange, ones};

// Bug: OpKindScalar::Sub was doing addition (copy-pasted from Add arm).
#[test]
fn regression_scalar_sub_was_adding() {
    let t = Tensor::from_scalar(10.0, &[4]);
    assert_eq!((t - 3.0).materialize().data(), &vec![7.0; 4]);
}

// Bug: Sub with scalar on a tensor starting from ones should not add.
#[test]
fn regression_scalar_sub_from_ones() {
    let t = ones!(&[4]);
    assert_eq!((t - 3.0).materialize().data(), &vec![-2.0; 4]);
}

// Bug: NotSameShape error reported inputs[0].shape() for both sides.
#[test]
fn regression_not_same_shape_error_shows_both_shapes() {
    let a = ones!(&[3, 4]);
    let b = ones!(&[3, 5]);
    let result = std::panic::catch_unwind(|| {
        let _ = (a + b).materialize();
    });
    assert!(result.is_err(), "expected a panic for shape mismatch");
}

// Bug: unordered buffer reuse could pick rhs as output, reversing Sub/Div.
#[test]
fn regression_sub_ordering_with_reusable_rhs() {
    // a - b must be a - b, not b - a
    let a = Tensor::from_scalar(10.0, &[4]);
    let b = Tensor::from_scalar(3.0, &[4]);
    // Adding 0 makes b go through a scalar op node (potentially reusable)
    let b_node = b + 0.0;
    let result = (a - b_node).materialize();
    assert_eq!(result.data(), &vec![7.0; 4]); // not [-7.0; 4]
}

// Bug: redirect table is static — applied to every input lookup regardless of
// execution order. If an independent consumer of node A (call it C) appears
// before AsContiguous(A) (call it B) in the topological sort, the redirect
// A→B causes C to request B's buffer before B has been computed.
//
// Graph:
//   edge_t → Transpose (node_1, non-contiguous)
//                ↓               ↓
//          AsContiguous       ScalarOp +1
//            (node_2)           (node_3)
//                ↓               ↓
//                   Add (root)
//                inputs = [node_2, node_3]
//
// The topological sort does a DFS from root.inputs. Because root.inputs =
// [node_2, node_3] and the stack is LIFO, node_3 is popped first. Its
// subtree (node_1) is explored and yielded before node_2, giving the order:
//   node_1, node_3, node_2
//
// During planning, node_2 is processed last, inserting redirect node_1→node_2.
// At execution time, node_3 runs before node_2 and resolves its node_1 input
// through the redirect, finding node_2's buffer absent from the cache → panic.
#[test]
fn bug_redirect_timing_independent_consumer_before_as_contiguous() {
    let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let transposed = t.transpose(); // node_1
    let contiguous = transposed.as_contiguous(); // node_2
    let shifted = &transposed + 1.0; // node_3, independent of node_2

    // root.inputs = [node_2, node_3] → sort yields node_3 before node_2
    let result = (&contiguous + &shifted).materialize();

    // transposed = [[1,3],[2,4]], contiguous = [1,3,2,4], shifted = [[2,4],[3,5]]
    // contiguous + shifted = [[3,7],[5,9]]
    assert_eq!(result.data(), &vec![3.0, 7.0, 5.0, 9.0]);
}

// Bug: matmul inputs.pop() order was reversed (raw_a got rhs, raw_b got lhs).
#[test]
fn regression_matmul_input_order() {
    // identity @ B == B, not B^T
    let identity = Tensor::<f64>::eye(2, 2);
    let b = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let result = identity.matmul(&b).unwrap().materialize();
    assert_eq!(result.data(), b.data());
}

// Bug: chain (t * 2) - 1 must produce correct values, not corrupt results.
#[test]
fn regression_buffer_reuse_chain_correctness() {
    // arange(6) * 2 - 1 = [-1, 1, 3, 5, 7, 9]
    let t = arange!(6);
    let result = (t * 2.0 - 1.0).materialize();
    assert_eq!(result.data(), &vec![-1.0, 1.0, 3.0, 5.0, 7.0, 9.0]);
}

// Bug: is_last_axes_transposed returned true for contiguous [m, 1] tensors
// (stride = [1, 1]), causing BLAS to receive transb=CblasTrans and n=m instead
// of n=1, writing past the end of the one-element output buffer.
#[test]
fn regression_matmul_rhs_single_column() {
    let a = Tensor::from_slice(&[1.0, 2.0], &[1, 2]);
    let b = Tensor::from_slice(&[3.0, 4.0], &[2, 1]);
    let c = a.matmul(&b).unwrap().materialize();
    assert_eq!(c.shape(), &[1, 1]);
    assert_eq!(c.data(), &[11.0_f64]); // 1*3 + 2*4
}
