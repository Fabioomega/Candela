#![allow(dead_code)]
use candela::{Dimension, FloatLikeTensorElement, Tensor};

pub fn assert_approx_eq<T: FloatLikeTensorElement>(data: &[T], expected: &[f64]) {
    assert_eq!(data.len(), expected.len());
    for (a, b) in data.iter().zip(expected.iter()) {
        let a_f64: f64 = (*a).into();
        assert!((a_f64 - b).abs() < 1e-6, "Expected {b}, got {a_f64}");
    }
}

pub fn assert_approx_eq_by<T: FloatLikeTensorElement>(
    data: &[T],
    expected: &[T],
    max_relative: f64,
) {
    assert_eq!(data.len(), expected.len());
    for (a, b) in data.iter().zip(expected.iter()) {
        let a_f64: f64 = (*a).into();
        let b_f64: f64 = (*b).into();
        let abs_diff = (a_f64 - b_f64).abs();
        let largest = a_f64.abs().max(b_f64.abs());
        assert!(
            a_f64 == b_f64 || abs_diff <= largest * max_relative,
            "Expected {b_f64}, got {a_f64}"
        );
    }
}

pub fn assert_shape<D: Dimension>(t: &D, expected: &[usize]) {
    assert_eq!(t.shape(), expected, "shape mismatch");
}

pub fn tensor_data<T: FloatLikeTensorElement>(t: &Tensor<T>) -> Vec<T> {
    t.data().to_vec()
}

/// Cast an `f64` slice to a typed `Vec<T>` via `T::from_f64`. Loses precision
/// for `T = f32` if values don't fit, which is fine for the small literals
/// used throughout the test suite.
pub fn cast<T: FloatLikeTensorElement>(values: &[f64]) -> Vec<T> {
    values.iter().copied().map(T::from_f64).collect()
}

/// Build a `Tensor<T>` from an `f64` slice and shape - the typed equivalent
/// of writing `Tensor::from_slice(&[T::from_f64(...), ...], shape)`.
pub fn tensor_of<T: FloatLikeTensorElement>(values: &[f64], shape: &[usize]) -> Tensor<T> {
    Tensor::from_slice(&cast::<T>(values), shape)
}
