use cblas::daxpy;
use cblas_sys::cblas_daxpy;
use intel_mkl_sys::{vdAdd, vdExp, vdLn, vdLog10, vdLogb};

use crate::branch_fast_iter;
use crate::tensor::Dimension;
use crate::tensor::definitions::{ChunkedIter, NumberLike};
use crate::tensor::mem_formats::layout::Layout;
use crate::tensor::mkl_extension::vdAddI;
use crate::tensor::ops::def_op::OpKindScalar;
use crate::tensor::storage::TensorData;
use crate::tensor::traits::{StreamingIterator, StreamingZip};

pub(crate) struct CommonBLASOps<T> {
    pub add: unsafe extern "C" fn(i32, *const T, i32, *const T, i32, *mut T, i32),
    pub scal: unsafe extern "C" fn(i32, T, *mut T, i32),
    pub axby: unsafe extern "C" fn(i32, T, *const T, i32, *mut T, i32),
    pub exp: unsafe extern "C" fn(i32, *const T, *mut T),
    pub ln: unsafe extern "C" fn(i32, *const T, *mut T),
    pub log2: unsafe extern "C" fn(i32, *const T, *mut T),
}

#[inline]
pub(crate) fn clone_to_buffer<T: NumberLike>(tensor: TensorData<T>, mut buffer: Vec<T>) -> Vec<T> {
    branch_fast_iter!(tensor.copied_fast_iter() => iter, {
        for (i, el) in iter.enumerate() {
            buffer[i] = el;
        }

        buffer
    })
}

#[inline]
pub(crate) fn fill_buffer<T: Clone>(buffer: *mut T, len: usize, value: T) {
    let mut i = buffer;
    for _ in 0..len {
        unsafe { *i = value.clone() };
        unsafe { i = i.add(1) };
    }
}

// When inplace=true, input and output alias the same buffer.
#[inline]
fn compute_blas_scalar_op<T: NumberLike>(
    ops: &[OpKindScalar<T>],
    n: usize,
    input: *const T,
    output: *mut T,
    inplace: bool,
    blas: CommonBLASOps<T>,
) {
    let mut ops_iter = ops.iter();

    if let Some(op) = ops_iter.next() {
        match *op {
            OpKindScalar::AxBy(a, b) => {
                // TODO: Maybe find another solution for this that does not require another check.
                if inplace {
                    unsafe { (blas.scal)(n as i32, a, output, 1) };
                    unsafe { (blas.add)(n as i32, output, 1, &b as *const T, 0, output, 1) };
                } else {
                    fill_buffer(output, n, b);
                    unsafe { (blas.axby)(n as i32, a, input, 1, output, 1) };
                }
            }
            OpKindScalar::Exp => {
                unsafe { (blas.exp)(n as i32, input, output) };
            }
            OpKindScalar::Ln => {
                unsafe { (blas.ln)(n as i32, input, output) };
            }
            OpKindScalar::Log2 => {
                unsafe { (blas.log2)(n as i32, input, output) };
            }
        }
    }

    for op in ops_iter {
        match *op {
            OpKindScalar::AxBy(a, b) => {
                unsafe { (blas.scal)(n as i32, a, output, 1) };

                unsafe { (blas.add)(n as i32, output, 1, &b as *const T, 0, output, 1) };
            }
            OpKindScalar::Exp => {
                unsafe { (blas.exp)(n as i32, output, output) };
            }
            OpKindScalar::Ln => {
                unsafe { (blas.ln)(n as i32, output, output) };
            }
            OpKindScalar::Log2 => {
                unsafe { (blas.log2)(n as i32, output, output) };
            }
        }
    }
}

// If the input is non-contiguous, the output buffer is always different than the input. Ths is guaranteed by the planner.
#[inline]
fn compute_non_cont_scalar_op<T: NumberLike>(
    ops: &[OpKindScalar<T>],
    input: &TensorData<T>,
    output: *mut T,
    blas: CommonBLASOps<T>,
) {
    // TODO: For big ops tensors rayon would be ideal.
    let mut it: ChunkedIter<'_, T> = input.packed_iter();
    while let Some(chunk) = it.next() {
        let n = chunk.packing_buffer.len();
        let pos = unsafe { output.add(chunk.absolute_buffer_position) };
        let mut ops_iter = ops.iter();

        if let Some(op) = ops_iter.next() {
            match *op {
                OpKindScalar::AxBy(a, b) => {
                    fill_buffer(pos, chunk.packing_buffer.len(), b);

                    unsafe { (blas.axby)(n as i32, a, chunk.packing_buffer.as_ptr(), 1, pos, 1) };
                }
                OpKindScalar::Exp => {
                    unsafe { (blas.exp)(n as i32, chunk.packing_buffer.as_ptr(), pos) };
                }
                OpKindScalar::Ln => {
                    unsafe { (blas.ln)(n as i32, chunk.packing_buffer.as_ptr(), pos) };
                }
                OpKindScalar::Log2 => {
                    unsafe { (blas.log2)(n as i32, chunk.packing_buffer.as_ptr(), pos) };
                }
            }
        }

        for op in ops_iter {
            match *op {
                OpKindScalar::AxBy(a, b) => {
                    unsafe { (blas.scal)(n as i32, a, pos, 1) };

                    unsafe { (blas.add)(n as i32, pos, 1, &b as *const T, 0, pos, 1) };
                }
                OpKindScalar::Exp => {
                    unsafe { (blas.exp)(n as i32, pos, pos) };
                }
                OpKindScalar::Ln => {
                    unsafe { (blas.ln)(n as i32, pos, pos) };
                }
                OpKindScalar::Log2 => {
                    unsafe { (blas.log2)(n as i32, pos, pos) };
                }
            }
        }
    }
}

#[inline]
pub(crate) fn compute_scalar_inplace<T: NumberLike>(
    ops: &[OpKindScalar<T>],
    output_layout: &Layout,
    mut inputs: Vec<TensorData<T>>,
    blas: CommonBLASOps<T>,
) -> TensorData<T> {
    // TODO: If the planner is sound this should be safe to unwrap unchecked.
    let mut tensor = inputs.pop().unwrap();
    let ptr = tensor.as_mut_ptr().unwrap();

    if tensor.is_contiguous() {
        compute_blas_scalar_op(ops, output_layout.len(), tensor.as_ptr(), ptr, true, blas);
    } else {
        compute_non_cont_scalar_op(ops, &tensor, ptr, blas);
    }

    tensor
}

#[inline]
pub(crate) fn compute_scalar<T: NumberLike>(
    ops: &[OpKindScalar<T>],
    mut output_buffer: Vec<T>,
    output_layout: &Layout,
    inputs: &[TensorData<T>],
    blas: CommonBLASOps<T>,
) -> TensorData<T> {
    if inputs[0].is_contiguous() {
        compute_blas_scalar_op(
            ops,
            output_layout.len(),
            inputs[0].as_ptr(),
            output_buffer.as_mut_ptr(),
            false,
            blas,
        );
    } else {
        compute_non_cont_scalar_op(ops, &inputs[0], output_buffer.as_mut_ptr(), blas);
    }

    TensorData::new(
        crate::tensor::storage::Storage::from_vec(output_buffer),
        output_layout.clone(),
    )
}

pub(crate) fn compute_elementwise_tensor_tensor<T: Copy + Default>(
    inputs: &[TensorData<T>],
    mut output_buffer: Vec<T>,
    operation: unsafe extern "C" fn(i32, *const T, *const T, *mut T),
) -> TensorData<T> {
    match (inputs[0].is_contiguous(), inputs[1].is_contiguous()) {
        (true, true) => {
            let lhs_buffer = inputs[0].as_ptr();
            let rhs_buffer = inputs[1].as_ptr();

            unsafe {
                operation(
                    output_buffer.len() as i32,
                    lhs_buffer,
                    rhs_buffer,
                    output_buffer.as_mut_ptr(),
                )
            }
        }
        (true, false) => {
            let mut it: ChunkedIter<'_, T> = inputs[1].packed_iter();

            while let Some(chunk) = it.next() {
                let n = chunk.packing_buffer.len();
                let input_ptr = unsafe {
                    inputs[0]
                        .storage
                        .buffer
                        .as_ptr()
                        .add(chunk.absolute_buffer_position)
                };
                let output_ptr = unsafe {
                    output_buffer
                        .as_mut_ptr()
                        .add(chunk.absolute_buffer_position)
                };

                unsafe {
                    operation(
                        n as i32,
                        input_ptr,
                        chunk.packing_buffer.as_ptr(),
                        output_ptr,
                    )
                };
            }
        }
        (false, true) => {
            let mut it: ChunkedIter<'_, T> = inputs[0].packed_iter();

            while let Some(chunk) = it.next() {
                let n = chunk.packing_buffer.len();
                let input_ptr = unsafe {
                    inputs[1]
                        .storage
                        .buffer
                        .as_ptr()
                        .add(chunk.absolute_buffer_position)
                };

                let output_ptr = unsafe {
                    output_buffer
                        .as_mut_ptr()
                        .add(chunk.absolute_buffer_position)
                };

                unsafe {
                    operation(
                        n as i32,
                        chunk.packing_buffer.as_ptr(),
                        input_ptr,
                        output_ptr,
                    )
                };
            }
        }
        (false, false) => {
            let mut it: StreamingZip<ChunkedIter<'_, T>, ChunkedIter<'_, T>> =
                inputs[0].packed_iter().zip(inputs[1].packed_iter());

            while let Some((chunk1, chunk2)) = it.next() {
                let n = chunk1.packing_buffer.len();
                let output_ptr = unsafe {
                    output_buffer
                        .as_mut_ptr()
                        .add(chunk1.absolute_buffer_position)
                };

                unsafe {
                    operation(
                        n as i32,
                        chunk1.packing_buffer.as_ptr(),
                        chunk2.packing_buffer.as_ptr(),
                        output_ptr,
                    );
                }
            }
        }
    };

    TensorData::from_vec(output_buffer, inputs[0].shape(), 0)
}

// The reused tensor must be contiguous; guaranteed by the planner (as_mut_ptr returns None otherwise).
pub(crate) fn compute_elementwise_tensor_tensor_inplace<T: Copy + Default>(
    mut inputs: Vec<TensorData<T>>,
    reuse_index: usize,
    operation: unsafe extern "C" fn(i32, *const T, *const T, *mut T),
) -> TensorData<T> {
    let last_idx = inputs.len() - 1;
    inputs.swap(reuse_index, last_idx);
    let mut output = inputs.pop().unwrap();
    let output_ptr = output.as_mut_ptr().unwrap();
    let other_is_left = reuse_index != 0;

    if inputs[0].is_contiguous() {
        let n = output.storage.buffer.len() as i32;
        let other_ptr = inputs[0].as_ptr();
        if other_is_left {
            unsafe { operation(n, other_ptr, output_ptr as *const T, output_ptr) };
        } else {
            unsafe { operation(n, output_ptr as *const T, other_ptr, output_ptr) };
        }
    } else {
        let mut it: ChunkedIter<'_, T> = inputs[0].packed_iter();
        while let Some(chunk) = it.next() {
            let n = chunk.packing_buffer.len() as i32;
            let out_ptr = unsafe { output_ptr.add(chunk.absolute_buffer_position) };
            if other_is_left {
                unsafe {
                    operation(
                        n,
                        chunk.packing_buffer.as_ptr(),
                        out_ptr as *const T,
                        out_ptr,
                    )
                };
            } else {
                unsafe {
                    operation(
                        n,
                        out_ptr as *const T,
                        chunk.packing_buffer.as_ptr(),
                        out_ptr,
                    )
                };
            }
        }
    }

    output
}
