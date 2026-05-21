use std::iter::zip;
use std::process::Output;

use crate::branch_fast_iter;
use crate::tensor::Dimension;
use crate::tensor::definitions::{ChunkedIter, NumberLike};
use crate::tensor::mem_formats::layout::Layout;
use crate::tensor::ops::def_op::OpKindScalar;
use crate::tensor::storage::{Storage, TensorData};
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
pub(crate) fn clone_to_buffer<T: Copy>(tensor: &TensorData<T>, mut buffer: Vec<T>) -> Vec<T> {
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

#[inline]
pub(crate) fn normalize_axis<T: Clone>(axis: isize, shape_len: usize) -> usize {
    (if axis < 0 {
        shape_len as isize + axis
    } else {
        axis
    }) as usize
}

// When inplace=true, input and output alias the same buffer.
#[inline]
fn compute_blas_scalar_op<T: NumberLike, F: Fn(T, T) -> T>(
    ops: &[OpKindScalar<T>],
    n: usize,
    input: *const T,
    output: *mut T,
    inplace: bool,
    blas: CommonBLASOps<T>,
    relu_base: T,
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
                    *o_el = max(relu_base, i_el);
                }
            }
            OpKindScalar::Sigmoid => {}
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
                    *o_el = max(relu_base, i_el);
                }
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
    // TODO: unwrap can be removed after checking that the plan is sound
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

// TODO: Add a variant for tensors contiguous at the second dimension.
pub(crate) fn cpu_compute_matmul_sum_scaled<T: Copy>(
    inputs: &[TensorData<T>],
    alpha: T,
    beta: T,
    mut output_buffer: Vec<T>,
    output_layout: &Layout,
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
) -> TensorData<T> {
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
        output_buffer = clone_to_buffer(c, output_buffer);
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
            a_stride as i32,
            b.as_ptr(),
            ldb,
            b_stride as i32,
            beta,
            output_buffer.as_mut_ptr(),
            n,
            m * n,
            a_shape[0] as i32,
        );
    };

    TensorData::new(Storage::from_vec(output_buffer), output_layout.clone())
}

#[inline]
fn cpu_compute_op_tensor<T: NumberLike, F: Fn(T, T) -> T>(
    inputs: &[TensorData<T>],
    output_buffer: &mut Vec<T>,
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
pub(crate) fn cpu_compute_sum_tensor<T: NumberLike>(
    inputs: &[TensorData<T>],
    mut output_buffer: Vec<T>,
    output_layout: &Layout,
    start_value: T,
) -> TensorData<T> {
    cpu_compute_op_tensor(inputs, &mut output_buffer, start_value, |a, b| a + b);

    TensorData::new(Storage::from_vec(output_buffer), output_layout.clone())
}

#[inline]
pub(crate) fn cpu_compute_max_tensor<T: NumberLike, F: Fn(T, T) -> T>(
    inputs: &[TensorData<T>],
    mut output_buffer: Vec<T>,
    output_layout: &Layout,
    start_value: T,
    max: F,
) -> TensorData<T> {
    cpu_compute_op_tensor(inputs, &mut output_buffer, start_value, max);

    TensorData::new(Storage::from_vec(output_buffer), output_layout.clone())
}

#[inline]
pub(crate) fn cpu_compute_mean_tensor<T: NumberLike, F: FnOnce(T, usize) -> T>(
    inputs: &[TensorData<T>],
    mut output_buffer: Vec<T>,
    output_layout: &Layout,
    start_value: T,
    div: F,
) -> TensorData<T> {
    cpu_compute_op_tensor(inputs, &mut output_buffer, start_value, |a, b| a + b);
    output_buffer[0] = div(output_buffer[0], inputs[0].layout().len());

    TensorData::new(Storage::from_vec(output_buffer), output_layout.clone())
}

// TODO: Confirm that the contiguous path can vectorize
// TODO: Benchmark if the contiguous path can be made faster by reducing striding code
fn cpu_compute_op_axis_tensor<T: NumberLike, F: Fn(T, T) -> T>(
    inputs: &[TensorData<T>],
    axis: usize,
    output_buffer: &mut Vec<T>,
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
pub(crate) fn cpu_compute_sum_axis_tensor<T: NumberLike>(
    inputs: &[TensorData<T>],
    axis: usize,
    mut output_buffer: Vec<T>,
    output_layout: &Layout,
    start_value: T,
) -> TensorData<T> {
    cpu_compute_op_axis_tensor(inputs, axis, &mut output_buffer, start_value, |a, b| a + b);

    TensorData::new(Storage::from_vec(output_buffer), output_layout.clone())
}

#[inline]
pub(crate) fn cpu_compute_max_axis_tensor<T: NumberLike, F: Fn(T, T) -> T>(
    inputs: &[TensorData<T>],
    axis: usize,
    mut output_buffer: Vec<T>,
    output_layout: &Layout,
    start_value: T,
    max: F,
) -> TensorData<T> {
    cpu_compute_op_axis_tensor(inputs, axis, &mut output_buffer, start_value, max);

    TensorData::new(Storage::from_vec(output_buffer), output_layout.clone())
}

#[inline]
pub(crate) fn cpu_compute_mean_axis_tensor<T: NumberLike, F: Fn(T, usize) -> T>(
    inputs: &[TensorData<T>],
    axis: usize,
    mut output_buffer: Vec<T>,
    output_layout: &Layout,
    start_value: T,
    div: F,
) -> TensorData<T> {
    cpu_compute_op_axis_tensor(inputs, axis, &mut output_buffer, start_value, |a, b| a + b);

    let n = inputs[0].shape()[axis];
    for el in output_buffer.iter_mut() {
        *el = div(*el, n);
    }

    TensorData::new(Storage::from_vec(output_buffer), output_layout.clone())
}
