use crate::tensor::MAX_DIMS;

pub(super) fn calculate_dim_stride(shape: &[usize]) -> [i32; MAX_DIMS] {
    let rank = shape.len();
    let mut v = [1i32; MAX_DIMS];

    for i in (0..rank.saturating_sub(1)).rev() {
        v[i] = (shape[i + 1] as i32) * v[i + 1];
    }

    v
}

pub(super) fn calculate_adjacent_dim_stride(
    stride: &[i32],
    slice_shape: &[usize],
) -> [i32; MAX_DIMS] {
    let rank = stride.len();
    debug_assert!(rank >= 1, "stride must have rank >= 1");

    let mut v = [0i32; MAX_DIMS];
    v[..rank].copy_from_slice(stride);

    let mut accum: i32 = 0;
    for i in (0..rank - 1).rev() {
        accum += stride[i + 1] * (slice_shape[i + 1] as i32 - 1);
        v[i] -= accum;
    }

    v
}
