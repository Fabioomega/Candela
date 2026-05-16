#![allow(dead_code)]
use approx::assert_relative_eq;
use candela::{Dimension, Tensor};

pub fn assert_approx_eq(a: &[f64], b: &[f64]) {
    assert_eq!(a.len(), b.len(), "slice lengths differ");
    assert_relative_eq!(a, b, max_relative = 1e-10);
}

pub fn assert_shape<D: Dimension>(t: &D, expected: &[usize]) {
    assert_eq!(t.shape(), expected, "shape mismatch");
}

pub fn tensor_data(t: &Tensor<f64>) -> Vec<f64> {
    t.data().clone()
}
