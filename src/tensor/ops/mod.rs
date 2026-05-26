mod capabilities;
pub mod def_op;

pub(crate) use capabilities::{CanMatMul, FloatLike};
pub mod fusion;
mod impl_layout;
pub mod impl_op;

pub use impl_layout::compute_layout;
