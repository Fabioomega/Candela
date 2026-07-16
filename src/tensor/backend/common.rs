use crate::{branch_fast_iter, tensor::storage::TensorData};

#[inline]
pub(crate) fn clone_to_buffer<T: Clone>(tensor: &TensorData<T>, buffer: &mut [T]) {
    branch_fast_iter!(tensor.fast_iter() => iter, {
        for (i, el) in iter.cloned().enumerate() {
            buffer[i] = el;
        }
    })
}

#[inline]
pub(crate) fn normalize_axis(axis: isize, shape_len: usize) -> usize {
    (if axis < 0 {
        shape_len as isize + axis
    } else {
        axis
    }) as usize
}
