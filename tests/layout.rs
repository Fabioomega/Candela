mod common;

use candela::{Dimension, Tensor, arange, s};
use common::assert_approx_eq;

// ── zero-copy checks ──────────────────────────────────────────────────────────

#[test]
fn view_is_zero_copy() {
    let t = arange!(12);
    let viewed = t.view(&[3, 4]).unwrap().materialize();
    // The result shares the same underlying data buffer — no allocation
    assert_eq!(viewed.data().as_ptr(), t.data().as_ptr());
}

#[test]
fn transpose_is_zero_copy() {
    let t = arange!(12);
    let viewed = t.view(&[3, 4]).unwrap();
    let t2 = viewed.materialize();
    let transposed = t2.transpose().materialize();
    // Buffer pointer must be the same — no copy was made
    assert_eq!(transposed.data().as_ptr(), t2.data().as_ptr());
}

#[test]
fn slice_is_zero_copy() {
    let t = arange!(12);
    let viewed = t.view(&[3, 4]).unwrap().materialize();
    let sliced = viewed.slice(s![0..2, ..]).unwrap().materialize();
    // Slice shares the same underlying buffer
    assert_eq!(sliced.data().as_ptr(), viewed.data().as_ptr());
}

// ── transpose correctness ─────────────────────────────────────────────────────

#[test]
fn transpose_2x2() {
    // [[1,2],[3,4]].T = [[1,3],[2,4]]
    // Multiply by 1.0 to force a fresh contiguous buffer from the non-contiguous transposed layout
    let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let result = (t.transpose() * 1.0).materialize();
    assert_eq!(result.shape(), &[2, 2]);
    assert_approx_eq(result.data(), &[1.0, 3.0, 2.0, 4.0]);
}

#[test]
fn transpose_2x3() {
    // [[0,1,2],[3,4,5]].T = [[0,3],[1,4],[2,5]]
    let t = Tensor::from_slice(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0], &[2, 3]);
    let result = (t.transpose() * 1.0).materialize();
    assert_eq!(result.shape(), &[3, 2]);
    assert_approx_eq(result.data(), &[0.0, 3.0, 1.0, 4.0, 2.0, 5.0]);
}

// ── slice correctness ─────────────────────────────────────────────────────────

#[test]
fn slice_then_add() {
    // arange(5) = [0,1,2,3,4], slice [2..4] = [2,3], + 1.0 = [3,4]
    let t = arange!(5);
    let sliced = t.slice(s![2..4]).unwrap();
    let result = (sliced + 1.0).materialize();
    assert_approx_eq(result.data(), &[3.0, 4.0]);
}

#[test]
fn slice_2d_row_range() {
    // [[0,1,2,3],[4,5,6,7],[8,9,10,11]], slice rows [1..3] = [[4,5,6,7],[8,9,10,11]]
    let t = Tensor::from_slice(
        &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0],
        &[3, 4],
    );
    let sliced = t.slice(s![1..3, ..]).unwrap().materialize();
    let temp: Box<[f64]> = sliced.iter().cloned().collect();
    assert_eq!(sliced.shape(), &[2, 4]);
    assert_approx_eq(&temp, &[4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0]);
}

// ── as_contiguous ─────────────────────────────────────────────────────────────

#[test]
fn as_contiguous_transposed() {
    // [[0,1],[2,3]].T = [[0,2],[1,3]], as_contiguous should give [0,2,1,3]
    let t = Tensor::from_slice(&[0.0, 1.0, 2.0, 3.0], &[2, 2]);
    let transposed = t.transpose();
    let result = transposed.as_contiguous().materialize();
    assert!(result.is_contiguous());
    assert_approx_eq(result.data(), &[0.0, 2.0, 1.0, 3.0]);
}

#[test]
fn view_after_as_contiguous() {
    // Transposed tensor can't be viewed directly — must go through as_contiguous first
    let t = Tensor::from_slice(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0], &[2, 3]);
    let cont = t.transpose().as_contiguous().materialize();
    // [3,2] contiguous — can view as [6]
    let viewed = cont.view(&[6]).unwrap().materialize();
    assert_eq!(viewed.shape(), &[6]);
}
