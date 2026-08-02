use crate::tensor::storage::TensorData;
use crate::tensor::walker::map_chunk;

#[inline]
pub(crate) fn clone_to_buffer<T: Clone>(tensor: &TensorData<T>, buffer: &mut [T]) {
    map_chunk(
        tensor.data(),
        tensor.layout(),
        buffer,
        |src, dst| {
            dst.clone_from_slice(src);
        },
        |x| x,
    );
}

#[inline]
pub(crate) fn normalize_axis(axis: isize, shape_len: usize) -> usize {
    (if axis < 0 {
        shape_len as isize + axis
    } else {
        axis
    }) as usize
}
