use cblas::daxpy;
use cblas_sys::cblas_daxpy;
use intel_mkl_sys::{vdAdd, vdExp, vdLn, vdLog10, vdLogb};

use crate::branch_fast_iter;
use crate::tensor::Dimension;
use crate::tensor::definitions::{ChunkedIter, NumberLike};
use crate::tensor::mkl_extension::vdAddI;
use crate::tensor::ops::def_op::OpKindScalar;
use crate::tensor::storage::TensorData;
use crate::tensor::traits::StreamingIterator;

struct CommonBLASOps<T> {
    add: unsafe extern "C" fn(i32, *const T, i32, *const T, i32, *mut T, i32),
    scal: unsafe extern "C" fn(i32, T, *mut T, i32),
    axby: unsafe extern "C" fn(i32, T, *const T, i32, *mut T, i32),
    exp: unsafe extern "C" fn(i32, *const T, *mut T),
    ln: unsafe extern "C" fn(i32, *const T, *mut T),
    log2: unsafe extern "C" fn(i32, *const T, *mut T),
}

#[inline]
pub fn clone_to_buffer<T: NumberLike>(tensor: TensorData<T>, mut buffer: Vec<T>) -> Vec<T> {
    branch_fast_iter!(tensor.copied_fast_iter() => iter, {
        for (i, el) in iter.enumerate() {
            buffer[i] = el;
        }

        buffer
    })
}

#[inline]
pub fn fill_buffer<T: Clone>(buffer: *mut T, len: usize, value: T) {
    let mut i = buffer;
    for _ in 0..len {
        unsafe { *i = value.clone() };
        unsafe { i = i.add(1) };
    }
}

// Supports inplace ops
fn compute_blas_scalar_op<T: NumberLike>(
    ops: &[OpKindScalar<T>],
    n: usize,
    input: *const T,
    output: *mut T,
    blas: CommonBLASOps<T>,
) {
    let mut ops_iter = ops.iter();

    if let Some(op) = ops_iter.next() {
        match *op {
            OpKindScalar::AxBy(a, b) => {
                fill_buffer(output, n, b);

                unsafe { (blas.axby)(n as i32, a, input, 1, output, 1) };
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
fn compute_non_cont_scalar_op_f64<T: NumberLike>(
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
                    unsafe { (blas.scal)(n as i32, a, output, 1) };

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

pub fn compute_elementwise_tensor_tensor<T: Copy + Default>(
    inputs: &[TensorData<T>],
    mut output_buffer: Vec<T>,
    operation: unsafe extern "C" fn(i32, *const T, *const T, *mut T),
) -> TensorData<T> {
    match (inputs[0].is_contiguous(), inputs[1].is_contiguous()) {
        (true, true) => {
            let lhs_buffer = &inputs[0].storage.buffer;
            let rhs_buffer = &inputs[1].storage.buffer;

            unsafe {
                operation(
                    output_buffer.len() as i32,
                    lhs_buffer.as_ptr(),
                    rhs_buffer.as_ptr(),
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

        _ => unreachable!(),
    };

    // Non-contiguous path
    if !inputs[0].is_contiguous() {
        // TODO: There's no need to pack the input. Maybe we should
        // allocate a full buffer and then operate directly
        let mut packed_iter: ChunkedIter<'_, T> = inputs[0].packed_iter();

        while let Some(chunk) = packed_iter.next() {
            let buffer_size: usize = chunk.packing_buffer.len();

            unsafe {
                operation(
                    buffer_size as i32,
                    output_buffer.as_ptr().add(chunk.absolute_buffer_position),
                    chunk.packing_buffer.as_ptr(),
                    output_buffer
                        .as_mut_ptr()
                        .add(chunk.absolute_buffer_position),
                )
            }
        }
    // Contiguous path
    } else {
        let lhs_buffer = &inputs[0].storage.buffer;

        unsafe {
            operation(
                output_buffer.len() as i32,
                output_buffer.as_ptr(),
                lhs_buffer.as_ptr(),
                output_buffer.as_mut_ptr(),
            )
        }
    }

    TensorData::from_vec(output_buffer, inputs[0].shape(), 0)
}
