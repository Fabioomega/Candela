use std::iter::zip;

use crate::PACKING_BUFFER_SIZE;
use crate::tensor::Dimension;
use crate::tensor::backend::common::clone_to_buffer;
use crate::tensor::definitions::{ChunkedIter, NumberLike};
use crate::tensor::mem_formats::layout::Layout;
use crate::tensor::ops::def_op::OpKindScalar;
use crate::tensor::storage::TensorData;
use crate::tensor::traits::{StreamingIterator, StreamingZip};
use cblas_sys::CBLAS_LAYOUT::{self, CblasRowMajor};
use cblas_sys::CBLAS_TRANSPOSE::{self, CblasNoTrans, CblasTrans};

pub(crate) struct CommonBLASOps<T> {
    pub add: unsafe extern "C" fn(i32, *const T, i32, *const T, i32, *mut T, i32),
    pub scal: unsafe extern "C" fn(i32, T, *mut T, i32),
    pub axby: unsafe extern "C" fn(i32, T, *const T, i32, *mut T, i32),
    pub exp: unsafe extern "C" fn(i32, *const T, *mut T),
    pub ln: unsafe extern "C" fn(i32, *const T, *mut T),
    pub log2: unsafe extern "C" fn(i32, *const T, *mut T),
    pub inv: unsafe extern "C" fn(i32, *const T, *mut T),
    pub tanh: unsafe extern "C" fn(i32, *const T, *mut T),
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
#[allow(clippy::too_many_arguments)]
#[inline]
fn compute_blas_scalar_op<T: NumberLike, F: Fn(T, T) -> T>(
    ops: &[OpKindScalar<T>],
    n: usize,
    input: *const T,
    output: *mut T,
    inplace: bool,
    blas: CommonBLASOps<T>,
    zero: T,
    max: F,
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
            OpKindScalar::Inv => unsafe { (blas.inv)(n as i32, input, output) },
            OpKindScalar::Tanh => unsafe { (blas.tanh)(n as i32, input, output) },
            OpKindScalar::ReLU => {
                let input = unsafe { std::slice::from_raw_parts(input, n) };
                let output = unsafe { std::slice::from_raw_parts_mut(output, n) };

                for (&i_el, o_el) in zip(input, output) {
                    *o_el = max(zero, i_el);
                }
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
            OpKindScalar::Inv => unsafe { (blas.inv)(n as i32, input, output) },
            OpKindScalar::Tanh => unsafe { (blas.tanh)(n as i32, input, output) },
            OpKindScalar::ReLU => {
                let input = unsafe { std::slice::from_raw_parts(input, n) };
                let output = unsafe { std::slice::from_raw_parts_mut(output, n) };

                for (&i_el, o_el) in zip(input, output) {
                    *o_el = max(zero, i_el);
                }
            }
        }
    }
}

// If the input is non-contiguous, the output buffer is always different than the input. Ths is guaranteed by the planner.
#[inline]
fn compute_non_cont_scalar_op<T: NumberLike, F: Fn(T, T) -> T>(
    ops: &[OpKindScalar<T>],
    input: &TensorData<T>,
    output: *mut T,
    blas: CommonBLASOps<T>,
    zero: T,
    max: F,
) {
    // TODO: For big ops tensors rayon would be ideal.
    let mut it: ChunkedIter<'_, T> = input.packed_iter(PACKING_BUFFER_SIZE);
    while let Some(chunk) = it.next_stream() {
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
                OpKindScalar::Inv => unsafe {
                    (blas.inv)(n as i32, chunk.packing_buffer.as_ptr(), pos)
                },
                OpKindScalar::Tanh => unsafe {
                    (blas.tanh)(n as i32, chunk.packing_buffer.as_ptr(), pos)
                },
                OpKindScalar::ReLU => {
                    let output = unsafe { std::slice::from_raw_parts_mut(pos, n) };

                    for (&i_el, o_el) in zip(chunk.packing_buffer.iter(), output) {
                        *o_el = max(zero, i_el);
                    }
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
                OpKindScalar::Inv => unsafe { (blas.inv)(n as i32, pos, pos) },
                OpKindScalar::Tanh => unsafe { (blas.tanh)(n as i32, pos, pos) },
                OpKindScalar::ReLU => {
                    let output = unsafe { std::slice::from_raw_parts_mut(pos, n) };

                    for o_el in output {
                        *o_el = max(zero, *o_el);
                    }
                }
            }
        }
    }
}

#[inline]
pub(crate) fn compute_scalar_inplace<T: NumberLike, F: Fn(T, T) -> T>(
    ops: &[OpKindScalar<T>],
    output_layout: &Layout,
    mut inputs: Vec<TensorData<T>>,
    blas: CommonBLASOps<T>,
    zero: T,
    max: F,
) -> TensorData<T> {
    // TODO: If the planner is sound this should be safe to unwrap unchecked.
    let mut tensor = inputs.pop().unwrap();
    // The output must be contiguous
    debug_assert!(tensor.is_contiguous());
    let ptr = tensor.as_mut_ptr().unwrap();

    if tensor.is_contiguous() {
        compute_blas_scalar_op(
            ops,
            output_layout.len(),
            tensor.as_ptr(),
            ptr,
            true,
            blas,
            zero,
            max,
        );
    } else {
        compute_non_cont_scalar_op(ops, &tensor, ptr, blas, zero, max);
    }

    tensor
}

#[inline]
pub(crate) fn compute_scalar<T: NumberLike, F: Fn(T, T) -> T>(
    ops: &[OpKindScalar<T>],
    output_buffer: &mut [T],
    output_layout: &Layout,
    inputs: &[TensorData<T>],
    blas: CommonBLASOps<T>,
    zero: T,
    max: F,
) {
    if inputs[0].is_contiguous() {
        compute_blas_scalar_op(
            ops,
            output_layout.len(),
            inputs[0].as_ptr(),
            output_buffer.as_mut_ptr(),
            false,
            blas,
            zero,
            max,
        );
    } else {
        compute_non_cont_scalar_op(ops, &inputs[0], output_buffer.as_mut_ptr(), blas, zero, max);
    }
}

pub(crate) fn compute_elementwise_tensor_tensor<T: Copy + Default>(
    inputs: &[TensorData<T>],
    output_buffer: &mut [T],
    operation: unsafe extern "C" fn(i32, *const T, *const T, *mut T),
) {
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
            let mut it: ChunkedIter<'_, T> = inputs[1].packed_iter(PACKING_BUFFER_SIZE);

            while let Some(chunk) = it.next_stream() {
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
            let mut it: ChunkedIter<'_, T> = inputs[0].packed_iter(PACKING_BUFFER_SIZE);

            while let Some(chunk) = it.next_stream() {
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
            let mut it: StreamingZip<ChunkedIter<'_, T>, ChunkedIter<'_, T>> = inputs[0]
                .packed_iter(PACKING_BUFFER_SIZE)
                .zip(inputs[1].packed_iter(PACKING_BUFFER_SIZE));

            while let Some((chunk1, chunk2)) = it.next_stream() {
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
}

// The reused tensor must be contiguous; guaranteed by the planner (as_mut_ptr returns None otherwise).
pub(crate) fn compute_elementwise_tensor_tensor_inplace<T: Copy + Default>(
    mut inputs: Vec<TensorData<T>>,
    reuse_index: usize,
    operation: unsafe extern "C" fn(i32, *const T, *const T, *mut T),
) -> TensorData<T> {
    let last_idx = inputs.len() - 1;
    inputs.swap(reuse_index, last_idx);
    // TODO: unwrap can be removed after checking that the plan is sound
    let mut output = inputs.pop().unwrap();
    // The output must be contiguous
    debug_assert!(output.is_contiguous());
    let output_ptr = output.as_mut_ptr().unwrap();
    let other_is_left = reuse_index != 0;

    if inputs[0].is_contiguous() {
        let n = output.layout().len() as i32;
        let other_ptr = inputs[0].as_ptr();
        if other_is_left {
            unsafe { operation(n, other_ptr, output_ptr as *const T, output_ptr) };
        } else {
            unsafe { operation(n, output_ptr as *const T, other_ptr, output_ptr) };
        }
    } else {
        let mut it: ChunkedIter<'_, T> = inputs[0].packed_iter(PACKING_BUFFER_SIZE);
        while let Some(chunk) = it.next_stream() {
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

// TODO: Add a variant for tensors contiguous at the second dimension.
pub(crate) fn compute_matmul_sum<T: Copy>(
    inputs: &[TensorData<T>],
    alpha: T,
    beta: T,
    output_buffer: &mut [T],
    fill_output_with_c: bool,
    gemm_batch_strided: unsafe extern "C" fn(
        CBLAS_LAYOUT,
        CBLAS_TRANSPOSE,
        CBLAS_TRANSPOSE,
        i32,
        i32,
        i32,
        T,
        *const T,
        i32,
        i32,
        *const T,
        i32,
        i32,
        T,
        *mut T,
        i32,
        i32,
        i32,
    ),
) {
    let a = &inputs[0];
    let b = &inputs[1];

    let a_shape = a.layout().shape_as_3d();
    let a_stride_len = a.stride().len();

    // This check is necessary because a or b may be broadcasted.
    let a_stride = if a.shape().len() >= 3 {
        a.stride()[a_stride_len - 3]
    } else {
        (a_shape[1] * a_shape[2]) as i32
    };

    let b_shape = b.layout().shape_as_3d();
    let b_stride_len = b.stride().len();

    let b_stride = if b.shape().len() >= 3 {
        b.stride()[b_stride_len - 3]
    } else {
        (b_shape[1] * b_shape[2]) as i32
    };

    // Check whether the tensor is transposed between the last 2 axis
    let (transa, lda, m, k) = if a.layout().is_last_axes_transposed() {
        (
            CblasTrans,
            a_shape[1] as i32,
            a_shape[2] as i32,
            a_shape[1] as i32,
        )
    } else {
        (
            CblasNoTrans,
            a_shape[2] as i32,
            a_shape[1] as i32,
            a_shape[2] as i32,
        )
    };

    let (transb, ldb, n) = if b.layout().is_last_axes_transposed() {
        (CblasTrans, k, b_shape[1] as i32)
    } else {
        (CblasNoTrans, b_shape[2] as i32, b_shape[2] as i32)
    };

    if fill_output_with_c {
        let c = &inputs[2];
        clone_to_buffer(c, output_buffer);
    }

    unsafe {
        gemm_batch_strided(
            CblasRowMajor,
            transa,
            transb,
            m,
            n,
            k,
            alpha,
            a.as_ptr(),
            lda,
            a_stride,
            b.as_ptr(),
            ldb,
            b_stride,
            beta,
            output_buffer.as_mut_ptr(),
            n,
            m * n,
            a_shape[0] as i32,
        );
    };
}
