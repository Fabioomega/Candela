//! Shared benchmark harness.
//!
//! The methodology lives here so every bench file measures the same way:
//!
//! - **Sizes** are solved from a byte budget. The cache-relative rungs
//!   (`L1`/`L2`/`L3`/`Dram`) are solved against the *physical* footprint of the
//!   case - every allocated input element plus the output - so a case labeled
//!   "L2" actually fits in L2 regardless of stream count, dtype, or stride
//!   gaps. `Elems(n)` rungs are exact logical element counts, for sweeps that
//!   want fixed points (threshold hunting, cross-machine comparison).
//! - **Layout variants** (contiguous / transposed / inner-strided / broadcast)
//!   are constructed here, identically for every benchmark, so the
//!   contiguous-vs-strided ratio is comparable across bench files. The step
//!   constant is one cache line - the worst case, one element touched per
//!   line - and is deliberately not configurable per bench.
//! - **Throughput** is reported as distinct bytes touched (unique input
//!   elements read + output elements written), i.e. effective bandwidth,
//!   comparable to STREAM numbers. For gapped or broadcast inputs this
//!   deliberately differs from the physical footprint the size solver uses:
//!   each number answers its own question.
//!
//! A bench file describes a computation as a `Fn(&[Layout]) -> Skeleton`
//! builder: it receives the layouts the harness decided on, makes slots from
//! them, and returns the compiled skeleton. [`bench_fill`] and [`bench_shapes`]
//! are the default runners (they time `Skeleton::run`); benches that need a
//! custom timed loop call [`fill_cases`] / [`shape_cases`] directly and reuse
//! the same sizing, labels, and throughput accounting.

// Compiled once per bench binary, and each binary uses only a slice of the
// harness - dead-code analysis can't see across binaries.
#![allow(dead_code)]

use candela::backend::{Backend, ComputeFor};
use candela::skeleton::Skeleton;
use candela::{Dimension, Layout, Tensor};
use criterion::measurement::WallTime;
use criterion::{BenchmarkGroup, BenchmarkId, Throughput};
use rand::distr::{Distribution, StandardUniform};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};
use std::hint::black_box;
use std::sync::LazyLock;

pub const K: usize = 1024;
pub const M: usize = 1024 * 1024;

/// Fixed seed: cases are built once at startup, so runs are reproducible.
/// Kernels are value-independent, so averaging over inputs buys nothing.
const SEED: u64 = 0xCA9DE1A;

static L1_BYTES: LazyLock<usize> = LazyLock::new(|| cache_size::l1_cache_size().unwrap_or(64 * K));
static L2_BYTES: LazyLock<usize> = LazyLock::new(|| cache_size::l2_cache_size().unwrap_or(256 * K));
static L3_BYTES: LazyLock<usize> = LazyLock::new(|| cache_size::l3_cache_size().unwrap_or(16 * M));
static LINE_BYTES: LazyLock<usize> =
    LazyLock::new(|| cache_size::l1_cache_line_size().unwrap_or(64));

/// Elements to step per index for [`Variant::Step`]: one cache line, so every
/// touched element costs a full line - the worst-case stride.
fn step_elems<T>() -> usize {
    (*LINE_BYTES / size_of::<T>()).max(2)
}

/// Row pitch multiplier for [`Variant::Padded`]: every other row, the
/// canonical sliced-tensor shape. Global like the step constant - pitch
/// sweeps are a study for [`bench_shapes`], not a harness knob.
const PAD_FACTOR: usize = 2;

/// How each input's memory is laid out. Applied uniformly to every input of a
/// case; mixed-layout cases (e.g. one transposed operand) belong in
/// [`bench_shapes`], where layouts are explicit.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Variant {
    /// Dense row-major - the fast path.
    Contig,
    /// Same logical shape over column-major memory (stride `[1, rows]`).
    /// Under [`ShapePolicy::Flat`] the shape becomes a square to have two
    /// axes to swap.
    Transposed,
    /// Every stride multiplied by one cache line's worth of elements.
    Step,
    /// Contiguous rows with a gap between them (row pitch = [`PAD_FACTOR`] x
    /// row length) - what a column-slice of a wider matrix produces. Rows
    /// stay vectorizable; the stress is the prefetcher/TLB across row jumps.
    /// Under [`ShapePolicy::Flat`] the shape becomes a square to have an
    /// outer axis to gap.
    Padded,
    /// The outermost axis has stride 0: one row of data reused across every
    /// outer index - a `(1, n)` operand broadcast up to `(m, n)`. The inner
    /// run stays contiguous, so each output row re-reads the same source
    /// slice. Under [`ShapePolicy::Flat`] the whole tensor reads one element.
    BcastOuter,
    /// The innermost axis has stride 0: one value per outer index, splatted
    /// across the row - an `(m, 1)` operand broadcast up to `(m, n)`. Under
    /// [`ShapePolicy::Flat`] the whole tensor reads one element.
    BcastInner,
}

impl Variant {
    fn label(self) -> &'static str {
        match self {
            Variant::Contig => "contig",
            Variant::Transposed => "transposed",
            Variant::Step => "step",
            Variant::Padded => "padded",
            Variant::BcastOuter => "bcast_outer",
            Variant::BcastInner => "bcast_inner",
        }
    }
}

/// Logical shape of every input as a function of the fill size.
#[derive(Clone, Copy, Debug)]
pub enum ShapePolicy {
    /// One axis: `[n]`.
    Flat,
    /// `[n / inner, inner]` - for reductions and anything row-structured.
    Rows { inner: usize },
}

/// One rung of a size sweep. Cache rungs are byte budgets for the whole
/// working set; `Elems` is an exact logical element count. `Dram` is 8x L3:
/// a measured NL3 sweep (July 2026) showed 2x L3 still sits below the
/// bandwidth plateau (L3-thrash transition) and 4x is only borderline clear
/// of it - 8x is the first rung solidly on the plateau.
#[derive(Clone, Copy, Debug)]
pub enum SizeSpec {
    L1,
    L2,
    L3,
    NL3(usize),
    Dram,
    Elems(usize),
}

impl SizeSpec {
    fn budget_bytes(self) -> Option<usize> {
        match self {
            SizeSpec::L1 => Some(*L1_BYTES),
            SizeSpec::L2 => Some(*L2_BYTES),
            SizeSpec::L3 => Some(*L3_BYTES),
            SizeSpec::NL3(n) => Some(n * *L3_BYTES),
            SizeSpec::Dram => Some(8 * *L3_BYTES),
            SizeSpec::Elems(_) => None,
        }
    }

    fn label(self) -> String {
        match self {
            SizeSpec::L1 => "L1".into(),
            SizeSpec::L2 => "L2".into(),
            SizeSpec::L3 => "L3".into(),
            SizeSpec::NL3(n) => format!("{}*L3", n),
            SizeSpec::Dram => "DRAM".into(),
            SizeSpec::Elems(n) if n % M == 0 => format!("{}M", n / M),
            SizeSpec::Elems(n) if n % K == 0 => format!("{}K", n / K),
            SizeSpec::Elems(n) => n.to_string(),
        }
    }
}

const DEFAULT_SIZES: &[SizeSpec] = &[SizeSpec::L1, SizeSpec::L2, SizeSpec::L3, SizeSpec::Dram];
/// Non-contiguous variants default to the two rungs where the answer differs:
/// cache-resident and memory-bound. The full cross product is rarely worth
/// the wall-clock time.
const DEFAULT_VARIANT_SIZES: &[SizeSpec] = &[SizeSpec::L2, SizeSpec::Dram];

/// Declares a fill-style sweep: n same-shaped inputs, sizes solved from byte
/// budgets, layout variants applied uniformly.
#[derive(Clone)]
pub struct FillConfig {
    n_inputs: usize,
    /// One entry per case; each entry is one variant per input. A uniform
    /// sweep is just combos of the shape `[v; n_inputs]` - see
    /// [`variants`](Self::variants) - and mixed cases (broadcast pairs) are
    /// built with [`variant_combos`](Self::variant_combos).
    combos: Vec<Vec<Variant>>,
    sizes: Vec<SizeSpec>,
    variant_sizes: Vec<SizeSpec>,
    shape: ShapePolicy,
}

impl FillConfig {
    pub fn new(n_inputs: usize) -> Self {
        assert!(n_inputs > 0, "a skeleton needs at least one input");
        Self {
            n_inputs,
            combos: vec![vec![Variant::Contig; n_inputs]],
            sizes: DEFAULT_SIZES.to_vec(),
            variant_sizes: DEFAULT_VARIANT_SIZES.to_vec(),
            shape: ShapePolicy::Flat,
        }
    }

    /// Which layout variants to run, applied uniformly to every input. The
    /// all-`Contig` case uses [`sizes`](Self::sizes); every other uses
    /// [`variant_sizes`](Self::variant_sizes).
    pub fn variants(mut self, variants: &[Variant]) -> Self {
        self.combos = variants.iter().map(|&v| vec![v; self.n_inputs]).collect();
        self
    }

    /// Per-input layout variants: one inner slice per case, each of length
    /// `n_inputs`. Unlike [`variants`](Self::variants) - which lays every input
    /// out the same way - this builds mixed-layout cases, e.g. a contiguous
    /// operand against a broadcast one. A case uses [`sizes`](Self::sizes) only
    /// when every input in its combo is `Contig`, else [`variant_sizes`](Self::variant_sizes).
    pub fn variant_combos(mut self, combos: &[&[Variant]]) -> Self {
        for combo in combos {
            assert_eq!(
                combo.len(),
                self.n_inputs,
                "each variant combo must have one entry per input"
            );
        }
        self.combos = combos.iter().map(|c| c.to_vec()).collect();
        self
    }

    pub fn sizes(mut self, sizes: &[SizeSpec]) -> Self {
        self.sizes = sizes.to_vec();
        self
    }

    pub fn variant_sizes(mut self, sizes: &[SizeSpec]) -> Self {
        self.variant_sizes = sizes.to_vec();
        self
    }

    pub fn shape(mut self, shape: ShapePolicy) -> Self {
        self.shape = shape;
        self
    }

    fn sizes_for(&self, combo: &[Variant]) -> &[SizeSpec] {
        if combo.iter().all(|&v| v == Variant::Contig) {
            &self.sizes
        } else {
            &self.variant_sizes
        }
    }
}

/// A fully-built benchmark case: compiled skeleton, inputs matching its slot
/// layouts, and the throughput denominator. Benches with a custom timed loop
/// consume these directly instead of going through [`bench_fill`].
pub struct Case<T, B: Backend> {
    pub label: String,
    pub skeleton: Skeleton<T, B>,
    pub inputs: Vec<Tensor<T, B>>,
    pub throughput: Throughput,
}

/// Smallest fill a policy supports; also the probe's starting point.
fn min_fill(policy: ShapePolicy) -> usize {
    match policy {
        ShapePolicy::Flat => 64,
        ShapePolicy::Rows { inner } => inner,
    }
}

/// Round a requested logical element count to one the policy/variant pair can
/// realize exactly. Returns the realized count.
fn normalize_fill(policy: ShapePolicy, variant: Variant, n: usize) -> usize {
    match (policy, variant) {
        (ShapePolicy::Flat, Variant::Transposed | Variant::Padded) => {
            let side = n.isqrt().max(2);
            side * side
        }
        (ShapePolicy::Flat, _) => n.max(2),
        (ShapePolicy::Rows { inner }, _) => (n / inner).max(1) * inner,
    }
}

/// Realized fill for a whole combo. Every input's variant must round the
/// requested count to the same value; otherwise the operands would carry
/// different logical shapes and could not form an element-wise case (pairing
/// `Contig` with `Transposed` under `Flat`, say, which squares the count).
/// Uniform combos and broadcast pairs under `Rows` agree trivially.
fn normalize_fill_combo(policy: ShapePolicy, combo: &[Variant], n: usize) -> usize {
    let mut agreed: Option<usize> = None;
    for &v in combo {
        let f = normalize_fill(policy, v, n);
        match agreed {
            None => agreed = Some(f),
            Some(a) => assert_eq!(
                a, f,
                "variant combo normalizes to inconsistent fills; operands would desync shape"
            ),
        }
    }
    agreed.expect("a combo has at least one input")
}

/// Case label for a combo: the bare variant name when every input shares it
/// (so uniform sweeps keep their old labels), else the per-input names joined
/// with `+`.
fn combo_label(combo: &[Variant]) -> String {
    if combo.iter().all(|&v| v == combo[0]) {
        combo[0].label().to_string()
    } else {
        combo
            .iter()
            .map(|v| v.label())
            .collect::<Vec<_>>()
            .join("+")
    }
}

/// The target layout for one input at a (normalized) fill size.
fn make_layout<T>(policy: ShapePolicy, variant: Variant, fill: usize) -> Layout {
    let step = step_elems::<T>() as i32;
    match (policy, variant) {
        (ShapePolicy::Flat, Variant::Contig) => Layout::new([fill]),
        (ShapePolicy::Flat, Variant::Step) => Layout::from_strided(&[fill], &[step], 0),
        (ShapePolicy::Flat, Variant::BcastOuter | Variant::BcastInner) => {
            Layout::from_strided(&[fill], &[0], 0)
        }
        (ShapePolicy::Flat, Variant::Transposed) => {
            let side = fill.isqrt();
            Layout::from_strided(&[side, side], &[1, side as i32], 0)
        }
        (ShapePolicy::Flat, Variant::Padded) => {
            let side = fill.isqrt();
            Layout::from_strided(&[side, side], &[(PAD_FACTOR * side) as i32, 1], 0)
        }
        (ShapePolicy::Rows { inner }, Variant::Contig) => Layout::new([fill / inner, inner]),
        (ShapePolicy::Rows { inner }, Variant::Step) => {
            Layout::from_strided(&[fill / inner, inner], &[inner as i32 * step, step], 0)
        }
        (ShapePolicy::Rows { inner }, Variant::Padded) => {
            Layout::from_strided(&[fill / inner, inner], &[(PAD_FACTOR * inner) as i32, 1], 0)
        }
        (ShapePolicy::Rows { inner }, Variant::BcastOuter) => {
            Layout::from_strided(&[fill / inner, inner], &[0, 1], 0)
        }
        (ShapePolicy::Rows { inner }, Variant::BcastInner) => {
            Layout::from_strided(&[fill / inner, inner], &[1, 0], 0)
        }
        (ShapePolicy::Rows { inner }, Variant::Transposed) => {
            let rows = fill / inner;
            Layout::from_strided(&[rows, inner], &[1, rows as i32], 0)
        }
    }
}

/// Elements the backing buffer must hold to satisfy `layout`.
fn buffer_len(layout: &Layout) -> usize {
    layout.last() + 1
}

/// Distinct elements a full traversal of `layout` reads. Stride-0 axes revisit
/// the same memory, so they contribute 1. (Assumes non-zero-stride axes don't
/// self-overlap - true for every layout this module produces.)
fn distinct_elems(layout: &Layout) -> usize {
    std::iter::zip(layout.shape(), layout.stride())
        .map(|(d, s)| if *s == 0 { 1 } else { *d })
        .product()
}

/// Physical working set of one case in bytes: all input allocations plus the
/// output, which is what must fit in a cache for the rung label to be honest.
fn footprint_bytes<T, B, F>(cfg: &FillConfig, builder: &F, combo: &[Variant], fill: usize) -> usize
where
    B: Backend,
    T: Clone + PartialEq + ComputeFor<B>,
    F: Fn(&[Layout]) -> Skeleton<T, B>,
{
    let layouts: Vec<Layout> = combo
        .iter()
        .map(|&v| make_layout::<T>(cfg.shape, v, fill))
        .collect();

    let skeleton = builder(&layouts);

    let input_elems: usize = layouts.iter().map(buffer_len).sum();
    (input_elems + skeleton.len()) * size_of::<T>()
}

/// Solve for the fill size whose footprint hits `budget` bytes. Footprint is
/// affine in the fill for every recipe this module produces, so two probes
/// pin the line exactly; compiling the skeleton at the probe sizes is what
/// picks up the output's contribution (reductions shrink it, and only the
/// plan knows by how much).
fn solve_fill<T, B, F>(cfg: &FillConfig, builder: &F, combo: &[Variant], budget: usize) -> usize
where
    B: Backend,
    T: Clone + PartialEq + ComputeFor<B>,
    F: Fn(&[Layout]) -> Skeleton<T, B>,
{
    let n1 = normalize_fill_combo(cfg.shape, combo, min_fill(cfg.shape) * 8);
    let n2 = normalize_fill_combo(cfg.shape, combo, n1 * 4);
    assert!(n2 > n1, "probe fills collapsed; ShapePolicy too coarse");

    let f1 = footprint_bytes::<T, B, F>(cfg, builder, combo, n1) as f64;
    let f2 = footprint_bytes::<T, B, F>(cfg, builder, combo, n2) as f64;

    let slope = (f2 - f1) / (n2 - n1) as f64;
    assert!(
        slope > 0.0,
        "footprint does not grow with fill size; this workload cannot be size-swept"
    );
    let intercept = f1 - slope * n1 as f64;

    let target = ((budget as f64 - intercept) / slope).max(0.0) as usize;
    normalize_fill_combo(cfg.shape, combo, target.max(min_fill(cfg.shape)))
}

fn tensor_from_layout<T, B>(layout: &Layout, rng: &mut StdRng) -> Tensor<T, B>
where
    B: Backend,
    T: ComputeFor<B>,
    StandardUniform: Distribution<T>,
{
    let buffer: Vec<T> = (0..buffer_len(layout)).map(|_| rng.random()).collect();
    Tensor::from_vec_with_layout_in(buffer, layout.clone())
}

/// Build every case of a fill sweep. The builder is called once per case (plus
/// twice per variant to probe the footprint); only `Skeleton::run` should be
/// timed afterwards.
pub fn fill_cases<T, B, F>(cfg: &FillConfig, builder: F) -> Vec<Case<T, B>>
where
    B: Backend,
    T: Clone + PartialEq + ComputeFor<B>,
    StandardUniform: Distribution<T>,
    F: Fn(&[Layout]) -> Skeleton<T, B>,
{
    let mut rng = StdRng::seed_from_u64(SEED);
    let mut cases = Vec::new();

    for combo in &cfg.combos {
        for &size in cfg.sizes_for(combo) {
            let fill: usize = match size.budget_bytes() {
                Some(budget) => solve_fill::<T, B, F>(cfg, &builder, combo, budget),
                None => match size {
                    SizeSpec::Elems(n) => normalize_fill_combo(cfg.shape, combo, n),
                    _ => unreachable!(),
                },
            };

            let layouts: Vec<Layout> = combo
                .iter()
                .map(|&v| make_layout::<T>(cfg.shape, v, fill))
                .collect();

            let skeleton = builder(&layouts);

            let touched: usize = layouts.iter().map(distinct_elems).sum::<usize>() + skeleton.len();
            let inputs = layouts
                .iter()
                .map(|l| tensor_from_layout(l, &mut rng))
                .collect();

            cases.push(Case {
                label: format!("{}/{}", size.label(), combo_label(combo)),
                skeleton,
                inputs,
                throughput: Throughput::Bytes((touched * size_of::<T>()) as u64),
            });
        }
    }

    cases
}

/// One explicit-shape case: matmul-family benches where the shape points and
/// the throughput unit (flops, elements) are statements about the workload.
pub struct ShapeCase {
    pub label: String,
    pub layouts: Vec<Layout>,
    pub throughput: Throughput,
}

pub fn shape_cases<T, B, F>(specs: &[ShapeCase], builder: F) -> Vec<Case<T, B>>
where
    B: Backend,
    T: Clone + PartialEq + ComputeFor<B>,
    StandardUniform: Distribution<T>,
    F: Fn(&[Layout]) -> Skeleton<T, B>,
{
    let mut rng = StdRng::seed_from_u64(SEED);
    specs
        .iter()
        .map(|spec| Case {
            label: spec.label.clone(),
            skeleton: builder(&spec.layouts),
            inputs: spec
                .layouts
                .iter()
                .map(|l| tensor_from_layout(l, &mut rng))
                .collect(),
            throughput: spec.throughput.clone(),
        })
        .collect()
}

/// The default timed loop: `Skeleton::run` over pre-built inputs.
pub fn run_cases<T, B>(
    group: &mut BenchmarkGroup<'_, WallTime>,
    label: &str,
    cases: Vec<Case<T, B>>,
) where
    B: Backend,
    T: Clone + PartialEq + ComputeFor<B>,
{
    for case in cases {
        group.throughput(case.throughput);
        let references: Vec<&Tensor<T, B>> = case.inputs.iter().collect();

        group.bench_function(BenchmarkId::new(label, &case.label), |bencher| {
            bencher.iter(|| case.skeleton.run(black_box(&references)).unwrap());
        });
    }
}

pub fn bench_fill<T, B, F>(
    group: &mut BenchmarkGroup<'_, WallTime>,
    label: &str,
    cfg: &FillConfig,
    builder: F,
) where
    B: Backend,
    T: Clone + PartialEq + ComputeFor<B>,
    StandardUniform: Distribution<T>,
    F: Fn(&[Layout]) -> Skeleton<T, B>,
{
    run_cases(group, label, fill_cases(cfg, builder));
}

pub fn bench_shapes<T, B, F>(
    group: &mut BenchmarkGroup<'_, WallTime>,
    label: &str,
    specs: &[ShapeCase],
    builder: F,
) where
    B: Backend,
    T: Clone + PartialEq + ComputeFor<B>,
    StandardUniform: Distribution<T>,
    F: Fn(&[Layout]) -> Skeleton<T, B>,
{
    run_cases(group, label, shape_cases(specs, builder));
}
