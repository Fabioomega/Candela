#[macro_use]
mod convenience;

pub(crate) const MAX_DIMS: usize = 8;

pub(crate) mod backend;
pub(crate) mod definitions;
pub mod errors;
mod executor;
mod internals;
mod iter;
mod mem_formats;
mod planner;
pub mod shape;
pub(crate) mod skeleton;
mod storage;
pub(crate) mod traits;
pub(crate) mod walker;

pub(crate) mod graph;
pub(crate) mod ops;
pub mod promise;
pub mod tensor_interface;
pub use convenience::*;

pub use iter::{InformedIter, Iter, StepInfo};
pub use mem_formats::layout::Layout;
pub use mem_formats::slice::SliceRange;
pub use promise::{CachedTensorPromise, TensorPromise};
pub use shape::IntoShape;
pub use tensor_interface::Tensor;
pub use traits::Dimension;
pub(crate) use traits::FromIndex;
