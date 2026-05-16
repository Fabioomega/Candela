mod common;

use candela::{Dimension, Layout, Tensor, ones};
use common::assert_approx_eq;

// ── Layout-level broadcast unit ───────────────────────────────────────────────

#[test]
fn broadcast_layout_zero_stride_on_expanded_dim() {
    // [4] broadcast to [3,4]: new leading dim gets stride 0
    let l = Layout::from_shape(&[4], 0);
    let b = l.broadcast(&[3, 4]).unwrap();
    assert_eq!(b.shape(), &[3, 4]);
    assert_eq!(b.stride()[0], 0);
    assert_eq!(b.stride()[1], 1);
    assert_eq!(b.len(), 12);
}

// ── Integration: binary ops auto-broadcast ────────────────────────────────────

#[test]
fn broadcast_col_plus_row() {
    // [3,1] + [1,4] = [3,4], every element = 1 + 2 = 3
    let col = Tensor::from_scalar(1.0, &[3, 1]);
    let row = Tensor::from_scalar(2.0, &[1, 4]);
    let result = (col + row).materialize();
    assert_eq!(result.shape(), &[3, 4]);
    assert_approx_eq(result.data(), &vec![3.0; 12]);
}

#[test]
fn broadcast_1d_against_2d() {
    // [4] + [3,4]: each row of result should be [1,2,3,4]
    let v = Tensor::from_slice(&[0.0, 1.0, 2.0, 3.0], &[4]);
    let m = ones!(&[3, 4]);
    let result = (v + m).materialize();
    assert_eq!(result.shape(), &[3, 4]);
    // each row: [0+1, 1+1, 2+1, 3+1] = [1,2,3,4]
    let expected: Vec<f64> = [1.0, 2.0, 3.0, 4.0]
        .iter()
        .cycle()
        .take(12)
        .copied()
        .collect();
    assert_approx_eq(result.data(), &expected);
}

#[test]
fn broadcast_scalar_tensor_against_matrix() {
    // [1] * [3,3] = [3,3], all 5.0
    let s = Tensor::from_scalar(5.0, &[1]);
    let m = ones!(&[3, 3]);
    let result = (s * m).materialize();
    assert_approx_eq(result.data(), &vec![5.0; 9]);
}

#[test]
fn broadcast_size_one_dim_in_both() {
    // [3,1] + [1,4]: values differ
    // col: [[0],[1],[2]] + row: [[0,1,2,3]]
    // result[i][j] = i + j
    let col = Tensor::from_slice(&[0.0, 1.0, 2.0], &[3, 1]);
    let row = Tensor::from_slice(&[0.0, 1.0, 2.0, 3.0], &[1, 4]);
    let result = (col + row).materialize();
    assert_eq!(result.shape(), &[3, 4]);
    let expected = vec![
        0.0, 1.0, 2.0, 3.0, // row 0: 0+j
        1.0, 2.0, 3.0, 4.0, // row 1: 1+j
        2.0, 3.0, 4.0, 5.0, // row 2: 2+j
    ];
    assert_approx_eq(result.data(), &expected);
}

#[test]
fn broadcast_mul_with_row_vector() {
    // [3,1] * [1,3]: outer product style
    let col = Tensor::from_slice(&[1.0, 2.0, 3.0], &[3, 1]);
    let row = Tensor::from_slice(&[1.0, 2.0, 3.0], &[1, 3]);
    let result = (col * row).materialize();
    assert_eq!(result.shape(), &[3, 3]);
    let expected = vec![
        1.0, 2.0, 3.0, // 1 * [1,2,3]
        2.0, 4.0, 6.0, // 2 * [1,2,3]
        3.0, 6.0, 9.0, // 3 * [1,2,3]
    ];
    assert_approx_eq(result.data(), &expected);
}

#[test]
#[should_panic]
fn broadcast_incompatible_shapes_panics() {
    // [3,4] + [2,4] — dim 0 mismatch, neither is 1
    let a = ones!(&[3, 4]);
    let b = ones!(&[2, 4]);
    let _ = (a + b).materialize();
}
