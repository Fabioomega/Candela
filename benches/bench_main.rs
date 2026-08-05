//! Entry point for the whole benchmark suite.
//!
//! One target, every group, so a single `cargo bench` produces one comparable
//! report. Select a subset with criterion's filter - `cargo bench -- matmul`,
//! `cargo bench -- '/add$'` - rather than by building a different target.
//!
//! See `benches/README.md` for the filter recipes and the two build settings
//! (`target-cpu=native`, the `tracing` default feature) that every number here
//! depends on.

mod common;
mod ops;

use criterion::criterion_main;

criterion_main!(
    ops::binary::benches,
    ops::unary::benches,
    ops::fusion::benches,
    ops::reduction::benches,
    ops::matmul::benches,
    ops::layout::benches,
    ops::overhead::benches,
);
