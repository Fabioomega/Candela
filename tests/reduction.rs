mod common;

use candela::errors::OpError;
use candela::{Dimension, FloatLikeTensorElement, Tensor, srange};
use common::assert_approx_eq;
use rstest::rstest;

// ── total sum ─────────────────────────────────────────────────────────────────

#[rstest]
#[case::f64(0.0f64)]
#[case::f32(0.0f32)]
fn sum_1d<T: FloatLikeTensorElement>(#[case] _t: T) {
    // [0,1,2,3,4] → 10
    let t: Tensor<T> = srange!(5, &[5]);
    assert_approx_eq(t.sum().materialize().data(), &[10.0]);
}

#[rstest]
#[case::f64(0.0f64)]
#[case::f32(0.0f32)]
fn sum_2d<T: FloatLikeTensorElement>(#[case] _t: T) {
    // [[1,2],[3,4]] → all elements sum to 10
    let one = T::ONE;
    let two = one + one;
    let three = two + one;
    let four = three + one;
    let t = Tensor::from_slice(&[one, two, three, four], &[2, 2]);
    assert_approx_eq(t.sum().materialize().data(), &[10.0]);
}

#[rstest]
#[case::f64(0.0f64)]
#[case::f32(0.0f32)]
fn sum_uniform<T: FloatLikeTensorElement>(#[case] _t: T) {
    // 3×3 tensor of 4.0 → 36.0
    let four = T::ONE + T::ONE + T::ONE + T::ONE;
    let t = Tensor::from_scalar(four, &[3, 3]);
    assert_approx_eq(t.sum().materialize().data(), &[36.0]);
}

// ── sum along axis ────────────────────────────────────────────────────────────

#[rstest]
#[case::f64(0.0f64)]
#[case::f32(0.0f32)]
fn sum_axis_0_1d<T: FloatLikeTensorElement>(#[case] _t: T) {
    // 1D: axis 0 collapses the only dimension → same as total sum
    let t: Tensor<T> = srange!(5, &[5]); // [0,1,2,3,4]
    assert_approx_eq(t.sum_axis(0, false).unwrap().materialize().data(), &[10.0]);
}

#[rstest]
#[case::f64(0.0f64)]
#[case::f32(0.0f32)]
fn sum_axis_0_2d<T: FloatLikeTensorElement>(#[case] _t: T) {
    // [[1,2],[3,4]] sum axis 0 → [4, 6]
    let one = T::ONE;
    let two = one + one;
    let three = two + one;
    let four = three + one;
    let t = Tensor::from_slice(&[one, two, three, four], &[2, 2]);
    assert_approx_eq(t.sum_axis(0, false).unwrap().materialize().data(), &[4.0, 6.0]);
}

#[rstest]
#[case::f64(0.0f64)]
#[case::f32(0.0f32)]
fn sum_axis_1_2d<T: FloatLikeTensorElement>(#[case] _t: T) {
    // [[1,2],[3,4]] sum axis 1 → [3, 7]
    let one = T::ONE;
    let two = one + one;
    let three = two + one;
    let four = three + one;
    let t = Tensor::from_slice(&[one, two, three, four], &[2, 2]);
    assert_approx_eq(t.sum_axis(1, false).unwrap().materialize().data(), &[3.0, 7.0]);
}

#[rstest]
#[case::f64(0.0f64)]
#[case::f32(0.0f32)]
fn sum_axis_keepdim<T: FloatLikeTensorElement>(#[case] _t: T) {
    // [[1,2],[3,4]] sum axis 0, keepdim=true → shape [1,2], values [4,6]
    let one = T::ONE;
    let two = one + one;
    let three = two + one;
    let four = three + one;
    let t = Tensor::from_slice(&[one, two, three, four], &[2, 2]);
    let result = t.sum_axis(0, true).unwrap().materialize();
    assert_eq!(result.shape(), &[1, 2]);
    assert_approx_eq(result.data(), &[4.0, 6.0]);
}

#[rstest]
#[case::f64(0.0f64)]
#[case::f32(0.0f32)]
fn sum_axis_negative<T: FloatLikeTensorElement>(#[case] _t: T) {
    // axis=-1 on [2,2] resolves to axis 1: same as sum_axis_1_2d
    let one = T::ONE;
    let two = one + one;
    let three = two + one;
    let four = three + one;
    let t = Tensor::from_slice(&[one, two, three, four], &[2, 2]);
    assert_approx_eq(t.sum_axis(-1, false).unwrap().materialize().data(), &[3.0, 7.0]);
}

#[rstest]
#[case::f64(0.0f64)]
#[case::f32(0.0f32)]
fn sum_axis_uniform<T: FloatLikeTensorElement>(#[case] _t: T) {
    // 3×3 of 4.0, sum axis 0 → [12, 12, 12]
    let four = T::ONE + T::ONE + T::ONE + T::ONE;
    let t = Tensor::from_scalar(four, &[3, 3]);
    assert_approx_eq(t.sum_axis(0, false).unwrap().materialize().data(), &[12.0; 3]);
}

#[test]
fn sum_axis_out_of_bounds() {
    let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let err = t.sum_axis(5, false).err().expect("expected Err");
    assert!(matches!(err, OpError::AxesOutOfBounds));
}

// ── total max ─────────────────────────────────────────────────────────────────

#[rstest]
#[case::f64(0.0f64)]
#[case::f32(0.0f32)]
fn max_1d<T: FloatLikeTensorElement>(#[case] _t: T) {
    // [0,1,2,3,4] → 4
    let t: Tensor<T> = srange!(5, &[5]);
    let four = T::ONE + T::ONE + T::ONE + T::ONE;
    assert_eq!(t.max().materialize().data(), &vec![four]);
}

#[rstest]
#[case::f64(0.0f64)]
#[case::f32(0.0f32)]
fn max_2d<T: FloatLikeTensorElement>(#[case] _t: T) {
    // [[1,5],[3,2]] → 5
    let one = T::ONE;
    let two = one + one;
    let three = two + one;
    let five = two + three;
    let t = Tensor::from_slice(&[one, five, three, two], &[2, 2]);
    assert_eq!(t.max().materialize().data(), &vec![five]);
}

#[rstest]
#[case::f64(0.0f64)]
#[case::f32(0.0f32)]
fn max_uniform<T: FloatLikeTensorElement>(#[case] _t: T) {
    // 3×3 tensor of 4.0 → 4.0
    let four = T::ONE + T::ONE + T::ONE + T::ONE;
    let t = Tensor::from_scalar(four, &[3, 3]);
    assert_eq!(t.max().materialize().data(), &vec![four]);
}

// ── max along axis ────────────────────────────────────────────────────────────

#[rstest]
#[case::f64(0.0f64)]
#[case::f32(0.0f32)]
fn max_axis_0_1d<T: FloatLikeTensorElement>(#[case] _t: T) {
    // 1D: axis 0 collapses the only dimension → same as total max
    let t: Tensor<T> = srange!(5, &[5]); // [0,1,2,3,4]
    let four = T::ONE + T::ONE + T::ONE + T::ONE;
    assert_eq!(t.max_axis(0, false).unwrap().materialize().data(), &vec![four]);
}

#[rstest]
#[case::f64(0.0f64)]
#[case::f32(0.0f32)]
fn max_axis_0_2d<T: FloatLikeTensorElement>(#[case] _t: T) {
    // [[1,2],[3,4]] max axis 0 → [3, 4]
    let one = T::ONE;
    let two = one + one;
    let three = two + one;
    let four = three + one;
    let t = Tensor::from_slice(&[one, two, three, four], &[2, 2]);
    assert_eq!(t.max_axis(0, false).unwrap().materialize().data(), &vec![three, four]);
}

#[rstest]
#[case::f64(0.0f64)]
#[case::f32(0.0f32)]
fn max_axis_1_2d<T: FloatLikeTensorElement>(#[case] _t: T) {
    // [[1,2],[3,4]] max axis 1 → [2, 4]
    let one = T::ONE;
    let two = one + one;
    let three = two + one;
    let four = three + one;
    let t = Tensor::from_slice(&[one, two, three, four], &[2, 2]);
    assert_eq!(t.max_axis(1, false).unwrap().materialize().data(), &vec![two, four]);
}

#[rstest]
#[case::f64(0.0f64)]
#[case::f32(0.0f32)]
fn max_axis_keepdim<T: FloatLikeTensorElement>(#[case] _t: T) {
    // [[1,2],[3,4]] max axis 0, keepdim=true → shape [1,2], values [3,4]
    let one = T::ONE;
    let two = one + one;
    let three = two + one;
    let four = three + one;
    let t = Tensor::from_slice(&[one, two, three, four], &[2, 2]);
    let result = t.max_axis(0, true).unwrap().materialize();
    assert_eq!(result.shape(), &[1, 2]);
    assert_eq!(result.data(), &vec![three, four]);
}

#[rstest]
#[case::f64(0.0f64)]
#[case::f32(0.0f32)]
fn max_axis_negative<T: FloatLikeTensorElement>(#[case] _t: T) {
    // axis=-1 on [2,2] resolves to axis 1: same as max_axis_1_2d
    let one = T::ONE;
    let two = one + one;
    let three = two + one;
    let four = three + one;
    let t = Tensor::from_slice(&[one, two, three, four], &[2, 2]);
    assert_eq!(t.max_axis(-1, false).unwrap().materialize().data(), &vec![two, four]);
}

#[rstest]
#[case::f64(0.0f64)]
#[case::f32(0.0f32)]
fn max_axis_uniform<T: FloatLikeTensorElement>(#[case] _t: T) {
    // 3×3 of 4.0, max axis 0 → [4, 4, 4]
    let four = T::ONE + T::ONE + T::ONE + T::ONE;
    let t = Tensor::from_scalar(four, &[3, 3]);
    assert_eq!(t.max_axis(0, false).unwrap().materialize().data(), &vec![four; 3]);
}

#[test]
fn max_axis_out_of_bounds() {
    let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let err = t.max_axis(5, false).err().expect("expected Err");
    assert!(matches!(err, OpError::AxesOutOfBounds));
}

// ── total mean ────────────────────────────────────────────────────────────────

#[rstest]
#[case::f64(0.0f64)]
#[case::f32(0.0f32)]
fn mean_1d<T: FloatLikeTensorElement>(#[case] _t: T) {
    // [0,1,2,3,4] → 2.0
    let t: Tensor<T> = srange!(5, &[5]);
    assert_approx_eq(t.mean().materialize().data(), &[2.0]);
}

#[rstest]
#[case::f64(0.0f64)]
#[case::f32(0.0f32)]
fn mean_2d<T: FloatLikeTensorElement>(#[case] _t: T) {
    // [[1,2],[3,4]] → mean = 2.5
    let one = T::ONE;
    let two = one + one;
    let three = two + one;
    let four = three + one;
    let t = Tensor::from_slice(&[one, two, three, four], &[2, 2]);
    assert_approx_eq(t.mean().materialize().data(), &[2.5]);
}

#[rstest]
#[case::f64(0.0f64)]
#[case::f32(0.0f32)]
fn mean_uniform<T: FloatLikeTensorElement>(#[case] _t: T) {
    // 3×3 tensor of 4.0 → 4.0
    let four = T::ONE + T::ONE + T::ONE + T::ONE;
    let t = Tensor::from_scalar(four, &[3, 3]);
    assert_approx_eq(t.mean().materialize().data(), &[4.0]);
}

// ── mean along axis ───────────────────────────────────────────────────────────

#[rstest]
#[case::f64(0.0f64)]
#[case::f32(0.0f32)]
fn mean_axis_0_1d<T: FloatLikeTensorElement>(#[case] _t: T) {
    // 1D: axis 0 collapses the only dimension → same as total mean
    let t: Tensor<T> = srange!(5, &[5]); // [0,1,2,3,4]
    assert_approx_eq(t.mean_axis(0, false).unwrap().materialize().data(), &[2.0]);
}

#[rstest]
#[case::f64(0.0f64)]
#[case::f32(0.0f32)]
fn mean_axis_0_2d<T: FloatLikeTensorElement>(#[case] _t: T) {
    // [[1,2],[3,4]] mean axis 0 → [2, 3]
    let one = T::ONE;
    let two = one + one;
    let three = two + one;
    let four = three + one;
    let t = Tensor::from_slice(&[one, two, three, four], &[2, 2]);
    assert_approx_eq(t.mean_axis(0, false).unwrap().materialize().data(), &[2.0, 3.0]);
}

#[rstest]
#[case::f64(0.0f64)]
#[case::f32(0.0f32)]
fn mean_axis_1_2d<T: FloatLikeTensorElement>(#[case] _t: T) {
    // [[1,2],[3,4]] mean axis 1 → [1.5, 3.5]
    let one = T::ONE;
    let two = one + one;
    let three = two + one;
    let four = three + one;
    let t = Tensor::from_slice(&[one, two, three, four], &[2, 2]);
    assert_approx_eq(t.mean_axis(1, false).unwrap().materialize().data(), &[1.5, 3.5]);
}

#[rstest]
#[case::f64(0.0f64)]
#[case::f32(0.0f32)]
fn mean_axis_keepdim<T: FloatLikeTensorElement>(#[case] _t: T) {
    // [[1,2],[3,4]] mean axis 0, keepdim=true → shape [1,2], values [2,3]
    let one = T::ONE;
    let two = one + one;
    let three = two + one;
    let four = three + one;
    let t = Tensor::from_slice(&[one, two, three, four], &[2, 2]);
    let result = t.mean_axis(0, true).unwrap().materialize();
    assert_eq!(result.shape(), &[1, 2]);
    assert_approx_eq(result.data(), &[2.0, 3.0]);
}

#[rstest]
#[case::f64(0.0f64)]
#[case::f32(0.0f32)]
fn mean_axis_negative<T: FloatLikeTensorElement>(#[case] _t: T) {
    // axis=-1 on [2,2] resolves to axis 1: same as mean_axis_1_2d
    let one = T::ONE;
    let two = one + one;
    let three = two + one;
    let four = three + one;
    let t = Tensor::from_slice(&[one, two, three, four], &[2, 2]);
    assert_approx_eq(t.mean_axis(-1, false).unwrap().materialize().data(), &[1.5, 3.5]);
}

#[rstest]
#[case::f64(0.0f64)]
#[case::f32(0.0f32)]
fn mean_axis_uniform<T: FloatLikeTensorElement>(#[case] _t: T) {
    // 3×3 of 4.0, mean axis 0 → [4, 4, 4]
    let four = T::ONE + T::ONE + T::ONE + T::ONE;
    let t = Tensor::from_scalar(four, &[3, 3]);
    assert_approx_eq(t.mean_axis(0, false).unwrap().materialize().data(), &[4.0; 3]);
}

#[test]
fn mean_axis_out_of_bounds() {
    let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let err = t.mean_axis(5, false).err().expect("expected Err");
    assert!(matches!(err, OpError::AxesOutOfBounds));
}
