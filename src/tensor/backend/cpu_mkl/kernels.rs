use crate::tensor::Dimension;
use crate::tensor::backend::common::clone_to_buffer;
use crate::tensor::storage::TensorData;
use cblas_sys::CBLAS_LAYOUT::{self, CblasRowMajor};
use cblas_sys::CBLAS_TRANSPOSE::{self, CblasNoTrans, CblasTrans};

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

    // Check whether the tensor is transposed between the last 2 axis.
    let (transa, lda, m, k) = if a.layout().is_last_axes_transposed() {
        (
            CblasTrans,
            a_shape[1] as i32,
            a_shape[1] as i32,
            a_shape[2] as i32,
        )
    } else {
        (
            CblasNoTrans,
            a_shape[2] as i32,
            a_shape[1] as i32,
            a_shape[2] as i32,
        )
    };

    // A transposed B is stored n-by-k, so its rows are `k` long.
    let (transb, ldb, n) = if b.layout().is_last_axes_transposed() {
        (CblasTrans, k, b_shape[2] as i32)
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
