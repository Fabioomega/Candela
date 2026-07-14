#[macro_use]
mod convenience;

pub(crate) mod backend;
pub(crate) mod definitions;
pub mod errors;
mod executor;
mod internals;
mod iter;
mod macros;
mod mem_formats;
mod planner;
pub mod shape;
pub(crate) mod skeleton;
mod storage;
pub(crate) mod traits;

pub(crate) mod graph;
pub(crate) mod ops;
pub mod promise;
pub mod tensor_interface;
pub use convenience::*;

pub use iter::{ChunkIter, ChunkKind, InformedIter, Iter, StepInfo};
pub use mem_formats::layout::Layout;
pub use mem_formats::slice::SliceRange;
pub use promise::{CachedTensorPromise, TensorPromise};
pub use shape::IntoShape;
pub use tensor_interface::Tensor;
pub use traits::Dimension;
pub(crate) use traits::FromIndex;
