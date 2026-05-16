use super::*;
use crate::tensor::errors::OpError;

// ── from_shape ────────────────────────────────────────────────────────────────

#[test]
fn from_shape_1d_strides() {
    let l = Layout::from_shape(&[5], 0);
    assert_eq!(l.shape(), &[5]);
    assert_eq!(l.stride(), &[1]);
    assert_eq!(l.len(), 5);
    assert_eq!(l.offset(), 0);
}

#[test]
fn from_shape_2d_strides() {
    let l = Layout::from_shape(&[3, 4], 0);
    assert_eq!(l.shape(), &[3, 4]);
    assert_eq!(l.stride(), &[4, 1]);
    assert_eq!(l.len(), 12);
}

#[test]
fn from_shape_3d_strides() {
    let l = Layout::from_shape(&[2, 3, 4], 0);
    assert_eq!(l.shape(), &[2, 3, 4]);
    assert_eq!(l.stride(), &[12, 4, 1]);
    assert_eq!(l.len(), 24);
}

// ── view ──────────────────────────────────────────────────────────────────────

#[test]
fn view_incompatible_size() {
    let l = Layout::from_shape(&[4], 0);
    assert!(matches!(l.view(&[3]), Err(OpError::InvalidViewShape)));
}

#[test]
fn view_non_contiguous() {
    let l = Layout::from_shape(&[4, 4], 0);
    let transposed = l.transpose();
    assert!(matches!(
        transposed.view(&[16]),
        Err(OpError::NonContiguousView)
    ));
}

#[test]
fn view_compatible_shape() {
    let l = Layout::from_shape(&[12], 0);
    let v = l.view(&[3, 4]).unwrap();
    assert_eq!(v.shape(), &[3, 4]);
    assert_eq!(v.len(), 12);
    assert!(v.is_contiguous());
}

// ── slice ─────────────────────────────────────────────────────────────────────

#[test]
fn slice_shape_and_offset() {
    use crate::tensor::mem_formats::slice::SliceRange;
    let l = Layout::from_shape(&[4, 5], 0);
    let sliced = l
        .slice(&[SliceRange::from(1..3), SliceRange::from(..)])
        .unwrap();
    assert_eq!(sliced.shape(), &[2, 5]);
    assert_eq!(sliced.offset(), 5); // row 1 starts at index 5
    assert_eq!(sliced.len(), 10);
}

#[test]
fn slice_too_many_ranges() {
    use crate::tensor::mem_formats::slice::SliceRange;
    let l = Layout::from_shape(&[5], 0); // 1D — only one axis
    let result = l.slice(&[SliceRange::from(..), SliceRange::from(..)]);
    assert!(matches!(result, Err(OpError::AxesOutOfBounds)));
}

#[test]
fn slice_out_of_bounds() {
    use crate::tensor::mem_formats::slice::SliceRange;
    let l = Layout::from_shape(&[4], 0);
    // index 5 on a size-4 dimension: offset 5 + len 1 = 6 > 4
    let result = l.slice(&[SliceRange::from(5_i32)]);
    assert!(matches!(result, Err(OpError::InvalidSliceShape(_, _))));
}

// ── transpose ─────────────────────────────────────────────────────────────────

#[test]
fn transpose_swaps_last_two_axes() {
    let l = Layout::from_shape(&[3, 4], 0);
    let t = l.transpose();
    assert_eq!(t.shape(), &[4, 3]);
    assert_eq!(t.stride(), &[1, 4]);
}

#[test]
fn transpose_3d_axis_order() {
    // transpose() reverses the full axis order for any rank
    let l = Layout::from_shape(&[2, 3, 4], 0);
    let t = l.transpose();
    assert_eq!(t.shape(), &[4, 3, 2]);
    assert_eq!(t.stride(), &[1, 4, 12]);
}

#[test]
fn transpose_axes_permutation() {
    let l = Layout::from_shape(&[2, 3, 4], 0);
    let t = l.transpose_axes(&[2, 0, 1]).unwrap();
    assert_eq!(t.shape(), &[4, 2, 3]);
    assert_eq!(t.stride(), &[1, 12, 4]);
}

#[test]
fn transpose_axes_out_of_bounds() {
    let l = Layout::from_shape(&[3, 4], 0);
    assert!(matches!(
        l.transpose_axes(&[0, 5]),
        Err(OpError::AxesOutOfBounds)
    ));
}

#[test]
fn transpose_axes_duplicate_axes() {
    // [0, 0, 1] is not a valid permutation — duplicates corrupt the layout
    let l = Layout::from_shape(&[2, 3, 4], 0);
    assert!(l.transpose_axes(&[0, 0, 1]).is_err());
}

// ── is_contiguous ─────────────────────────────────────────────────────────────

#[test]
fn is_contiguous_fresh_layout() {
    let l = Layout::from_shape(&[4, 4], 0);
    assert!(l.is_contiguous());
}

#[test]
fn is_contiguous_after_transpose() {
    let l = Layout::from_shape(&[4, 4], 0);
    assert!(!l.transpose().is_contiguous());
}

#[test]
fn is_contiguous_innermost_broadcast_2d() {
    // [3, 1] -> [3, 4]: stride [1, 0], adj_stride [1, 0]
    // adj_stride[0] == 1 but the last stride is zero — not contiguous
    let b = Layout::from_shape(&[3, 1], 0).broadcast(&[3, 4]).unwrap();
    assert!(!b.is_contiguous());
}

#[test]
fn is_contiguous_outermost_broadcast_2d() {
    // [1, 4] -> [3, 4]: stride [0, 1], adj_stride [-3, 1] — already correct before fix
    let b = Layout::from_shape(&[1, 4], 0).broadcast(&[3, 4]).unwrap();
    assert!(!b.is_contiguous());
}

#[test]
fn is_contiguous_innermost_broadcast_3d() {
    // [2, 3, 1] -> [2, 3, 4]: stride [3, 1, 0], adj_stride [1, 1, 0]
    let b = Layout::from_shape(&[2, 3, 1], 0)
        .broadcast(&[2, 3, 4])
        .unwrap();
    assert!(!b.is_contiguous());
}

#[test]
fn is_contiguous_middle_broadcast_3d() {
    // [2, 1, 4] -> [2, 3, 4]: stride [4, 0, 1], adj_stride [1, -3, 1]
    // both endpoints are 1 — only caught by checking stride for zeros
    let b = Layout::from_shape(&[2, 1, 4], 0)
        .broadcast(&[2, 3, 4])
        .unwrap();
    assert!(!b.is_contiguous());
}

// ── is_transposed ─────────────────────────────────────────────────────────────

#[test]
fn is_transposed_outermost_broadcast_2d() {
    // [1, 4] -> [3, 4]: adj_stride [-3, 1] — negative comes from zero stride, not a transpose
    let b = Layout::from_shape(&[1, 4], 0).broadcast(&[3, 4]).unwrap();
    assert!(!b.is_transposed());
}

#[test]
fn is_transposed_middle_broadcast_3d() {
    // [2, 1, 4] -> [2, 3, 4]: adj_stride [1, -3, 1] — negative at dim 1 is from zero stride
    let b = Layout::from_shape(&[2, 1, 4], 0)
        .broadcast(&[2, 3, 4])
        .unwrap();
    assert!(!b.is_transposed());
}

#[test]
fn is_transposed_after_transpose() {
    // regression: real transpositions must still be detected
    let t = Layout::from_shape(&[3, 4], 0).transpose();
    assert!(t.is_transposed());
}

// ── broadcast ─────────────────────────────────────────────────────────────────

#[test]
fn broadcast_expanded_dim_zero_stride() {
    // [4] broadcast to [3,4]: new leading dim gets stride 0
    let l = Layout::from_shape(&[4], 0);
    let b = l.broadcast(&[3, 4]).unwrap();
    assert_eq!(b.shape(), &[3, 4]);
    assert_eq!(b.stride()[0], 0);
    assert_eq!(b.stride()[1], 1);
}

#[test]
fn broadcast_size_one_dim_zero_stride() {
    let l = Layout::from_shape(&[1, 4], 0);
    let b = l.broadcast(&[3, 4]).unwrap();
    assert_eq!(b.stride()[0], 0);
    assert_eq!(b.stride()[1], 1);
}

#[test]
fn broadcast_output_len() {
    let l = Layout::from_shape(&[4], 0);
    let b = l.broadcast(&[3, 4]).unwrap();
    assert_eq!(b.len(), 12);
}

#[test]
fn broadcast_same_shape() {
    let l = Layout::from_shape(&[4], 0);
    let b = l.broadcast(&[4]).unwrap();
    assert_eq!(b.shape(), &[4]);
    assert_eq!(b.stride()[0], 1); // stride unchanged
}

#[test]
fn broadcast_incompatible_shapes() {
    let l = Layout::from_shape(&[2, 3, 4], 0);
    assert!(matches!(
        l.broadcast(&[2, 4, 4]),
        Err(OpError::CannotBroadcast)
    ));
}
