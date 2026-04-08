use std::sync::Arc;

use crate::branch_fast_iter;
use crate::tensor::Dimension;
use crate::tensor::storage::TensorData;

pub(crate) struct ReusableVec<T> {
    pub(crate) v: Vec<T>,
    pub(crate) offset: usize,
}

#[inline]
fn strip_tensor<T: Copy + Default>(tensor: TensorData<T>) -> ReusableVec<T> {
    let len = tensor.len();
    let offset = tensor.offset();

    if let Ok(mut v) = Arc::try_unwrap(tensor.storage.buffer) {
        v.resize(len + offset, T::default());
        ReusableVec { v, offset }
    } else {
        unreachable!("the fuck. if the tensor is reusable why the fuck is it not unique?")
    }
}

#[inline]
pub fn alloc_cont_tensor<T: Copy + Default>(tensor: &TensorData<T>) -> ReusableVec<T> {
    println!("Allocated memory!");
    branch_fast_iter!(tensor.copied_fast_iter() => iter, {
        let v = Vec::from_iter(iter);

        ReusableVec {v, offset: 0}
    })
}

#[inline]
pub fn get_reusable_or_alloc<T: Copy + Default>(tensor: TensorData<T>) -> ReusableVec<T> {
    if tensor.reusable && tensor.is_contiguous() {
        strip_tensor(tensor)
    } else {
        alloc_cont_tensor(&tensor)
    }
}

#[inline]
fn unordered_remove_tensor<T: Copy>(tensors: &mut Vec<TensorData<T>>, n: usize) -> TensorData<T> {
    let temp = tensors[n].clone_reference();
    tensors[n] = tensors.pop().unwrap();

    temp
}

pub fn unordered_get_reusable_or_alloc_n<T: Copy + Default>(
    tensors: &mut Vec<TensorData<T>>,
    n: usize,
) -> ReusableVec<T> {
    for i in 0..tensors.len() {
        if tensors[i].reusable {
            let removed = unordered_remove_tensor(tensors, i);

            return strip_tensor(removed);
        }
    }

    let v = alloc_cont_tensor(&tensors[n]);
    let _ = unordered_remove_tensor(tensors, n);

    v
}
