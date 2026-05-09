extern crate cblas;
extern crate intel_mkl_src;
extern crate intel_mkl_sys;

pub(crate) const PACKING_BUFFER_SIZE: usize = 128;

#[macro_use]
mod convenience;

mod definitions;
pub mod errors;
mod impl_generics;
mod internals;
mod iter;
mod macros;
mod mem_formats;
mod mkl_extension;
mod planner;
mod storage;
mod traits;

pub(crate) mod graph;
pub(crate) mod ops;
pub mod promise;
pub mod tensor;
pub use convenience::*;

pub use mem_formats::layout::Layout;
pub use mem_formats::slice::SliceRange;
pub use promise::{CachedTensorPromise, TensorPromise};
pub use tensor::Tensor;
pub use traits::Dimension;
