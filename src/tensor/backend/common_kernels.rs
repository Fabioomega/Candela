use crate::tensor::definitions::NumberLike;
use crate::tensor::storage::TensorData;
use crate::tensor::traits::Numeric;
use crate::{Dimension, branch_fast_iter};

#[inline]
fn compute_op_tensor<T: NumberLike, F: Fn(T, T) -> T>(
    inputs: &[TensorData<T>],
    output_buffer: &mut [T],
    start_value: T,
    op: F,
) {
    output_buffer[0] = start_value;

    branch_fast_iter!(inputs[0].fast_iter() => it, {
        for el in it {
            output_buffer[0] = op(output_buffer[0], *el);
        }
    });
}

#[inline]
pub(crate) fn compute_sum_tensor<T: Numeric>(inputs: &[TensorData<T>], output_buffer: &mut [T]) {
    compute_op_tensor(inputs, output_buffer, T::SUM_NEUTRAL, |a, b| a + b);
}

#[inline]
pub(crate) fn compute_max_tensor<T: Numeric, F: Fn(T, T) -> T>(
    inputs: &[TensorData<T>],
    output_buffer: &mut [T],
    max: F,
) {
    compute_op_tensor(inputs, output_buffer, T::MIN, max);
}

#[inline]
pub(crate) fn compute_mean_tensor<T: Numeric, F: FnOnce(T, usize) -> T>(
    inputs: &[TensorData<T>],
    output_buffer: &mut [T],
    div: F,
) {
    compute_op_tensor(inputs, output_buffer, T::SUM_NEUTRAL, |a, b| a + b);
    output_buffer[0] = div(output_buffer[0], inputs[0].layout().len());
}

// TODO: Confirm that the contiguous path can vectorize
// TODO: Benchmark if the contiguous path can be made faster by reducing striding code
fn compute_op_axis_tensor<T: NumberLike, F: Fn(T, T) -> T>(
    inputs: &[TensorData<T>],
    axis: usize,
    output_buffer: &mut [T],
    start_value: T,
    op: F,
) {
    let t = &inputs[0];
    let n_outer: usize = t.shape()[..axis].iter().product();
    let n_size: usize = t.shape()[axis];
    let n_inner: usize = t.shape()[axis + 1..].iter().product();

    let mut base_pos = 0;
    output_buffer.fill(start_value);

    branch_fast_iter!(
        t.fast_iter() => _it, {
            let mut it = _it;

            for _ in 0..n_outer {
                for _ in 0..n_size {
                    for inner in 0..n_inner {
                        let el = unsafe { it.next().unwrap_unchecked() };
                        let current = output_buffer[base_pos + inner];

                        output_buffer[base_pos + inner] = op(current, *el);
                    }
                }

                base_pos += n_inner;
            }
        }
    );
}

#[inline]
pub(crate) fn compute_sum_axis_tensor<T: Numeric>(
    inputs: &[TensorData<T>],
    axis: usize,
    output_buffer: &mut [T],
) {
    compute_op_axis_tensor(inputs, axis, output_buffer, T::SUM_NEUTRAL, |a, b| a + b);
}

#[inline]
pub(crate) fn compute_max_axis_tensor<T: Numeric, F: Fn(T, T) -> T>(
    inputs: &[TensorData<T>],
    axis: usize,
    output_buffer: &mut [T],
    max: F,
) {
    compute_op_axis_tensor(inputs, axis, output_buffer, T::MIN, max);
}

#[inline]
pub(crate) fn compute_mean_axis_tensor<T: Numeric, F: Fn(T, usize) -> T>(
    inputs: &[TensorData<T>],
    axis: usize,
    output_buffer: &mut [T],
    div: F,
) {
    compute_op_axis_tensor(inputs, axis, output_buffer, T::SUM_NEUTRAL, |a, b| a + b);

    let n = inputs[0].shape()[axis];
    for el in output_buffer.iter_mut() {
        *el = div(*el, n);
    }
}
