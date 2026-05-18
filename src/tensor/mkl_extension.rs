#![allow(
    improper_ctypes,
    non_upper_case_globals,
    non_camel_case_types,
    non_snake_case
)]

use std::ffi::{c_double, c_float, c_int};

use cblas_sys::{CBLAS_LAYOUT, CBLAS_TRANSPOSE};

unsafe extern "C" {
    pub fn vdAddI(
        n: c_int,
        a: *const f64,
        inca: c_int,
        b: *const f64,
        incb: c_int,
        y: *mut f64,
        incy: c_int,
    );

    pub fn vsAddI(
        n: c_int,
        a: *const f32,
        inca: c_int,
        b: *const f32,
        incb: c_int,
        y: *mut f32,
        incy: c_int,
    );

    pub fn cblas_dgemm_batch_strided(
        layout: CBLAS_LAYOUT,
        transa: CBLAS_TRANSPOSE,
        transb: CBLAS_TRANSPOSE,
        m: c_int,
        n: c_int,
        k: c_int,
        alpha: c_double,
        a: *const f64,
        lda: c_int,
        stridea: c_int,
        b: *const f64,
        ldb: c_int,
        strideb: c_int,
        beta: c_double,
        c: *mut f64,
        ldc: c_int,
        stridec: c_int,
        batch_size: c_int,
    );

    pub fn cblas_sgemm_batch_strided(
        layout: CBLAS_LAYOUT,
        transa: CBLAS_TRANSPOSE,
        transb: CBLAS_TRANSPOSE,
        m: c_int,
        n: c_int,
        k: c_int,
        alpha: c_float,
        a: *const f32,
        lda: c_int,
        stridea: c_int,
        b: *const f32,
        ldb: c_int,
        strideb: c_int,
        beta: c_float,
        c: *mut f32,
        ldc: c_int,
        stridec: c_int,
        batch_size: c_int,
    );
}
