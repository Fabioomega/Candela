pub(super) fn calculate_dim_stride(shape: &[usize]) -> Box<[i32]> {
    let mut v: Box<[i32]> = vec![1; shape.len()].into_boxed_slice();

    for i in (0..shape.len().saturating_sub(1)).rev() {
        v[i] = (shape[i + 1] as i32) * v[i + 1];
    }

    v
}

pub(super) fn calculate_adjacent_dim_stride(stride: &[i32], slice_shape: &[usize]) -> Box<[i32]> {
    let mut v: Box<[i32]> = stride.into();

    let mut accum: i32 = 0;
    for i in (0..stride.len() - 1).rev() {
        accum += stride[i + 1] * (slice_shape[i + 1] as i32 - 1);
        v[i] -= accum;
    }

    v
}
