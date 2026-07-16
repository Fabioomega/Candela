use candela::{Dimension, Tensor, arange, s};

// [[0,1,2],[3,4,5]], contiguous, offset 0.
fn t2x3() -> Tensor<f64> {
    Tensor::from_slice(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0], &[2, 3])
}

fn logical(t: &Tensor<f64>) -> Vec<f64> {
    t.iter().copied().collect()
}

// ── reference over a computed node ────────────────────────────────────────────

#[test]
fn root_transpose_computed() {
    // (t + 10) = [[10,11,12],[13,14,15]] is computed into a fresh buffer, then the
    // transpose root aliases it. The buffer stays row-major; only the view is
    // transposed.
    let out = (t2x3() + 10.0).transpose().materialize();
    assert_eq!(out.shape(), &[3, 2]);
    assert!(!out.is_contiguous());
    assert_eq!(logical(&out), &[10.0, 13.0, 11.0, 14.0, 12.0, 15.0]);
    assert_eq!(out.data(), &[10.0, 11.0, 12.0, 13.0, 14.0, 15.0]);
}

#[test]
fn root_slice_computed() {
    // Second row of the computed (t + 10); the composed offset lands on [13,14,15].
    let out = (t2x3() + 10.0).slice(s![1..2, 0..3]).unwrap().materialize();
    assert_eq!(out.shape(), &[1, 3]);
    assert_eq!(logical(&out), &[13.0, 14.0, 15.0]);
}

#[test]
fn root_slice_reused_chain() {
    // A chain that creates buffer-reuse pressure, capped by a slice root. If the
    // root's slot were reclaimed by the reuse pass the data would be corrupt.
    let a: Tensor<f64> = arange!(6); // [0,1,2,3,4,5]
    let out = ((a * 2.0) - 1.0).slice(s![2..5]).unwrap().materialize();
    // (2x - 1) = [-1,1,3,5,7,9]; [2..5] => [3,5,7].
    assert_eq!(out.shape(), &[3]);
    assert_eq!(logical(&out), &[3.0, 5.0, 7.0]);
}

// ── chained references at the root ────────────────────────────────────────────

#[test]
fn root_transpose_twice() {
    // Two transposes collapse back to the original layout, still aliasing the edge.
    let t = t2x3();
    let out = t.transpose().transpose().materialize();
    assert_eq!(out.shape(), &[2, 3]);
    assert!(out.is_contiguous());
    assert_eq!(logical(&out), &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
    assert_eq!(out.data().as_ptr(), t.data().as_ptr());
}

#[test]
fn root_slice_then_transpose() {
    // Columns [1..3] => [[1,2],[4,5]], then transposed => [[1,4],[2,5]]. Two
    // references compose into one layout over the edge buffer.
    let t = t2x3();
    let out = t.slice(s![0..2, 1..3]).unwrap().transpose().materialize();
    assert_eq!(out.shape(), &[2, 2]);
    assert_eq!(logical(&out), &[1.0, 4.0, 2.0, 5.0]);
    assert_eq!(out.data().as_ptr(), t.data().as_ptr());
}

// ── broadcast at the root ─────────────────────────────────────────────────────

#[test]
fn root_broadcast() {
    // [3,1] expands to [3,4] with a stride-0 axis, aliasing the source buffer.
    let col = Tensor::from_slice(&[1.0, 2.0, 3.0], &[3, 1]);
    let out = col.broadcast(&[3, 4]).unwrap().materialize();
    assert_eq!(out.shape(), &[3, 4]);
    assert_eq!(out.stride()[1], 0); // expanded axis
    assert_eq!(
        logical(&out),
        &[1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 3.0]
    );
    assert_eq!(out.data().as_ptr(), col.data().as_ptr());
}
