use crate::{branch_fast_iter, tensor::storage::TensorData};

#[inline]
pub(crate) fn clone_to_buffer<T: Copy>(tensor: &TensorData<T>, mut buffer: Vec<T>) -> Vec<T> {
    branch_fast_iter!(tensor.fast_iter() => iter, {
        for (i, el) in iter.cloned().enumerate() {
            buffer[i] = el;
        }

        buffer
    })
}
