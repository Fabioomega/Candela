#![allow(dead_code)]
use candela::{Dimension, FloatLikeTensorElement, Tensor};

pub fn assert_approx_eq<T: FloatLikeTensorElement>(data: &[T], expected: &[f64]) {
    assert_eq!(data.len(), expected.len());
    for (a, b) in data.iter().zip(expected.iter()) {
        let a_f64: f64 = (*a).into();
        assert!((a_f64 - b).abs() < 1e-6, "Expected {b}, got {a_f64}");
    }
}

pub fn assert_shape<D: Dimension>(t: &D, expected: &[usize]) {
    assert_eq!(t.shape(), expected, "shape mismatch");
}

pub fn tensor_data<T: FloatLikeTensorElement>(t: &Tensor<T>) -> Vec<T> {
    t.data().clone()
}
