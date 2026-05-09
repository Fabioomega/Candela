pub mod tensor;

pub use tensor::arange;
pub use tensor::errors;
pub use tensor::{CachedTensorPromise, Dimension, Layout, SliceRange, Tensor, TensorPromise};
