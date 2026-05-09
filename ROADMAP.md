# Candela — Roadmap

Phases are ordered by dependency. Complete earlier phases before starting later ones.

Status markers: `[ ]` not started · `[~]` in progress · `[x]` done

---

## Phase 1 — Execution Planner (Done)

**Goal:** Replace the current ad-hoc buffer-reuse mechanism with a correct static
execution plan derived from the computation graph.

**Why first:** The existing `reusable: bool` approach produces wrong results for
non-commutative ops (`Sub`, `Div`) and does not scale. Everything that comes after
depends on execution being correct. Fix this before building on top of it.

**Known bugs to fix:**
- The `reusable` mechanism can pick the wrong buffer as the output of `Sub` and `Div`,
  silently reversing the operands. The result of `a - b` can come out as `b - a`.
- `Sub` in the scalar op path was copy-pasted from another arm and computes the wrong
  operation entirely.

### Tests
```rust
// Sub correctness (was silently computing addition before)
let t = ones(&[4]);
assert_eq!((t - 3.0).materialize().data(), &[-2.0; 4]);

// Div correctness
let t = Tensor::from_scalar(6.0, &[4]);
assert_eq!((t / 2.0).materialize().data(), &[3.0; 4]);

// Sub is non-commutative: (a - b) != (b - a)
let a = arange(4);   // [0, 1, 2, 3]
let b = ones(&[4]);
assert_ne!((a - b).materialize().data(), (b - a).materialize().data());

// Buffer reuse does not corrupt results in a longer chain
let t = arange(6);
let result = ((t * 2.0) - 1.0).materialize();
// expected: [-1, 1, 3, 5, 7, 9]
assert_eq!(result.data(), &[-1.0, 1.0, 3.0, 5.0, 7.0, 9.0]);

// A node used by two consumers is computed once and produces the same value for both
let t   = arange(4).as_promise();
let lhs = &t * 2.0;
let rhs = &t + 1.0;
let result = (lhs - rhs).materialize(); // (2x) - (x+1) = x - 1
assert_eq!(result.data(), &[-1.0, 0.0, 1.0, 2.0]);
```

### Documentation (Done)
- Document the execution plan struct and the liveness analysis algorithm
- Add a doc comment to `TensorGrap
- hNode::compute` explaining the execution model

---

## Phase 2 — Cleanup and Test Foundation

**Why here:** Small, high-leverage housekeeping while the Phase 1 architecture is fresh.
The test suite must exist before Phase 3 so regressions are caught as they are introduced,
not discovered later.

### Implementation
- [X] Delete `src/tensor/tensor_old.rs` — dead code, never referenced
- [X] Remove unused `lapacke` dependency from `Cargo.toml`
- [X] Mark internal modules `pub(crate)`: `OpKind`, `NodeKind`, `Promising`,
      `TensorGraphEdge`, `TensorGraphNode`, `TensorGraphCacheNode`
- [X] Fix `Clone` semantics: make `Clone` a shallow Arc-bump; rename the current deep
      copy to `deep_clone()`. A `Vec<Tensor<T>>::clone()` should not silently copy every
      buffer
- [kinda?] Replace remaining `unreachable!()` / `todo!()` in `graph.rs`, `impl_layout.rs`,
      and `impl_compute` with `Err(OpError::...)` or `unimplemented!()` with a clear
      message
- [ ] Fix the three bugs in `Layout::broadcast_to_shape` before Phase 4 can use it:
      the `cfg_debug_only!` guard predicate is inverted; stride is built from `shape`
      instead of `self.stride`; `len` is computed from `self.shape` instead of the
      target shape
- [ ] Set up `tests/` directory with an integration test module
- [fuck] Add a GitHub Actions workflow: `cargo fmt --check`, `cargo clippy -- -D warnings`,
      `cargo test`

### Tests
```rust
// Clone is shallow — both tensors share the same Arc, no allocation
let a = arange(1000);
let b = a.clone();
assert!(Arc::ptr_eq(&a.storage().buffer, &b.storage().buffer));

// deep_clone allocates a fresh buffer
let c = a.deep_clone();
assert!(!Arc::ptr_eq(&a.storage().buffer, &c.storage().buffer));
assert_eq!(a.data(), c.data());

// broadcast_to_shape produces correct strides
let layout = Layout::from_shape(&[4], 0);
let broadcast = layout.broadcast_to_shape(&[3, 4]).unwrap();
assert_eq!(broadcast.shape(), &[3, 4]);
assert_eq!(broadcast.stride()[0], 0); // broadcast dimension
assert_eq!(broadcast.len(), 12);      // total elements in broadcasted shape

// One test per OpKindScalar arm (catches copy-paste regressions)
let t = Tensor::from_scalar(3.0, &[4]);
assert_eq!((t.as_promise() + 2.0).materialize().data(), &[5.0; 4]);
assert_eq!((t.as_promise() - 2.0).materialize().data(), &[1.0; 4]);
assert_eq!((t.as_promise() * 2.0).materialize().data(), &[6.0; 4]);
assert_eq!((t.as_promise() / 2.0).materialize().data(), &[1.5; 4]);
```

### Documentation
- Add `# Examples` to the three public types: `Tensor`, `TensorPromise`,
  `CachedTensorPromise`
- Create `examples/lazy_eval.rs`: a walkthrough of building a graph and materializing it
- Create `examples/fusion.rs`: a demonstration of scalar fusion collapsing 20 additions

---

## Phase 3 — Complete Matmul

**Goal:** Make matrix multiplication actually compute the correct result.

**Why here:** Matmul is the single most important missing operation. Every model building
block in Phase 8 depends on it. Currently the implementation allocates a zeroed output
and returns it without computing anything.

**Known bugs to fix:**
- The `inputs.pop()` order is wrong: `raw_a` receives the second input (rhs) and `raw_b`
  receives the first (lhs). The labels are swapped, so any non-symmetric matmul will
  produce the wrong answer.

### Tests
```rust
// Basic 2x3 @ 3x4
let a = arange_2d(2, 3);         // [[0,1,2],[3,4,5]]
let b = Tensor::eye(3, 4);       // identity-ish
let c = a.matmul(&b)?.materialize();
assert_eq!(c.shape(), &[2, 4]);

// identity @ identity = identity
let i = Tensor::eye(3, 3);
let result = i.matmul(&i)?.materialize();
assert_approx_eq!(result.data(), Tensor::eye(3,3).data());

// Transposed matmul: A @ B^T where B is already transposed
let a = arange_2d(2, 3);
let b = arange_2d(4, 3).as_promise().transpose();
let c = a.matmul(&b)?.materialize();
assert_eq!(c.shape(), &[2, 4]);

// Batched: [2,3,4] @ [2,4,5] = [2,3,5]
let a = Tensor::from_scalar(1.0, &[2, 3, 4]);
let b = Tensor::from_scalar(1.0, &[2, 4, 5]);
assert_eq!(a.matmul(&b)?.materialize().shape(), &[2, 3, 5]);

// Shape mismatch returns error, does not panic
assert!(a_3x4.matmul(&b_3x4).is_err());
```

### Documentation
- Document the `Matmul` op in `def_op.rs` with the shape conventions used
- Add a `matmul` example in `examples/matmul.rs`

---

## Phase 4 — Broadcasting

**Goal:** Allow ops to work on tensors with compatible but non-identical shapes by
expanding dimensions implicitly rather than erroring.

**Why here:** Required for almost all real ops and for the batched matmul broadcast case.
The `broadcast_to_shape` bugs must be fixed in Phase 2 before this phase can proceed.

### Tests
```rust
// Column vector broadcast against row: [3,1] + [1,4] = [3,4]
let col = Tensor::from_scalar(1.0, &[3, 1]);
let row = Tensor::from_scalar(2.0, &[1, 4]);
let result = (col + row).materialize();
assert_eq!(result.shape(), &[3, 4]);
assert_eq!(result.data(), &[3.0; 12]);

// 1D against 2D: [4] + [3,4] = [3,4]
let v = arange(4);         // [0,1,2,3]
let m = ones(&[3, 4]);
let result = (v + m).materialize();
assert_eq!(result.shape(), &[3, 4]);
// each row of result should be [1,2,3,4]

// Scalar broadcast: [1] against [3,3]
let s = Tensor::from_scalar(5.0, &[1]);
let m = ones(&[3, 3]);
let result = (s * m).materialize();
assert_eq!(result.data(), &[5.0; 9]);

// Incompatible shapes still error
assert!((ones(&[3, 4]) + ones(&[2, 4])).materialize().is_err());
```

### Documentation
- Add broadcasting semantics to the Memory Layout section of `README.md`
- Document stride-0 broadcast in `Layout::broadcast_to_shape`

---

## Phase 5 — Reduction Ops

**Goal:** Add `sum`, `mean`, and `max` reductions along a specified axis, with optional
`keepdim` support.

**Why here:** Required for any loss function, normalization layer, or metric. Reductions
are also needed to implement `Softmax` in Phase 8.

### Tests
```rust
// 1D sum
let t = arange(5);  // [0,1,2,3,4]
assert_eq!(t.sum(0, false).materialize().data(), &[10.0]);

// 2D sum along axis 0
let t = Tensor::from_data(&[[1.0,2.0],[3.0,4.0]], &[2,2]);
assert_eq!(t.sum(0, false).materialize().data(), &[4.0, 6.0]);

// 2D sum along axis 1
assert_eq!(t.sum(1, false).materialize().data(), &[3.0, 7.0]);

// keepdim preserves rank
let result = t.sum(0, true).materialize();
assert_eq!(result.shape(), &[1, 2]);

// mean
let t = Tensor::from_scalar(4.0, &[3, 3]);
assert_eq!(t.mean(0, false).materialize().data(), &[4.0; 3]);

// max along axis
let t = Tensor::from_data(&[1.0, 5.0, 2.0, 4.0, 3.0], &[5]);
assert_eq!(t.max(0, false).materialize().data(), &[5.0]);
```

### Documentation
- Add `reduce` section to `README.md` features list
- Add doc examples to each reduction method

---

## Phase 6 — f32 Support and Activation Ops

**Goal:** Support `f32` tensors in addition to `f64`, and add `relu`, `sigmoid`, and
`tanh` as first-class ops.

**Why here:** These are two independent, straightforward additions that together greatly
expand usability. Being scalar ops, activations automatically participate in fusion
without any new fusion rules.

### Tests
```rust
// f32 basic ops
let t: Tensor<f32> = Tensor::from_scalar(2.0f32, &[4]);
assert_eq!((t * 3.0f32).materialize().data(), &[6.0f32; 4]);

// ReLU
let t = Tensor::from_data(&[-2.0, -1.0, 0.0, 1.0, 2.0], &[5]);
assert_eq!(t.relu().materialize().data(), &[0.0, 0.0, 0.0, 1.0, 2.0]);

// Sigmoid output is in (0, 1)
let t = Tensor::from_data(&[-10.0, 0.0, 10.0], &[3]);
let s = t.sigmoid().materialize();
assert!(s.data().iter().all(|&x| x > 0.0 && x < 1.0));
assert_approx_eq!(s.data()[1], 0.5); // sigmoid(0) = 0.5

// Activation fuses with preceding scalar op
let t = arange(4).as_promise();
let p = (t * 2.0).relu();  // should be one fused node
let result = p.materialize();
assert_eq!(result.data(), &[0.0, 2.0, 4.0, 6.0]);
```

### Documentation
- Add f32 examples to `README.md`
- Create `examples/activations.rs`

---

## Phase 7 — Model Building Blocks

**Goal:** Implement `Linear`, `Softmax`, and `LayerNorm` by composing the primitives
from Phases 3–6.

**Why here:** With matmul, broadcasting, reductions, and activations all in place,
these layers are pure composition — no new kernel work needed.

### Tests
```rust
// Linear: [batch=2, in=3] @ [out=4, in=3]^T + [4] = [2, 4]
let x      = Tensor::from_scalar(1.0, &[2, 3]);
let weight = Tensor::from_scalar(1.0, &[4, 3]);
let bias   = Tensor::from_scalar(0.5, &[4]);
let out    = linear(x, weight, bias).materialize();
assert_eq!(out.shape(), &[2, 4]);
// each output element = dot([1,1,1], [1,1,1]) + 0.5 = 3.5
assert_approx_eq!(out.data(), &[3.5; 8]);

// Softmax output sums to 1 along last axis
let logits = Tensor::from_data(&[1.0, 2.0, 3.0, 1.0, 2.0, 3.0], &[2, 3]);
let probs  = softmax(&logits, 1).materialize();
let row0_sum: f64 = probs.data()[..3].iter().sum();
assert_approx_eq!(row0_sum, 1.0);
```

### Documentation
- Create `examples/linear_layer.rs` showing a single forward pass
- Document each function with shape conventions

---

## Phase 8 — Symbolic Autodiff

**Goal:** Implement automatic differentiation by traversing the existing computation
graph in reverse and building a new gradient graph.

**Why symbolic:** The graph already carries the op and inputs at every node. Building
the backward pass as a new graph means gradients get fusion, execution planning, and
buffer reuse for free — the same machinery applies to both passes.

**Gradient rules you will need** (math, not implementation):
- `AxBy(a, b)`: `grad_input = a * grad_output`
- `Add`: `grad_lhs = grad_output`, `grad_rhs = grad_output`
- `Sub`: `grad_lhs = grad_output`, `grad_rhs = -grad_output`
- `Mul`: `grad_lhs = rhs * grad_output`, `grad_rhs = lhs * grad_output`
- `Matmul`: `grad_lhs = grad @ rhs^T`, `grad_rhs = lhs^T @ grad`
- `ReduceSum`: gradient is the broadcast of `grad_output` back to the input shape

### Tests
```rust
// Gradient of sum(x^2) w.r.t. x = 2x
let x = Parameter::new(Tensor::from_data(&[1.0, 2.0, 3.0], &[3]));
let loss = (x.as_promise() * x.as_promise()).sum(0, false);
let grads = loss.backward();
assert_approx_eq!(grads[&x].data(), &[2.0, 4.0, 6.0]);

// Chain rule: d/dx (3x + 1)^2 = 2(3x+1)*3 = 6(3x+1)
let x = Parameter::new(Tensor::from_scalar(2.0, &[1]));
let y = ((x.as_promise() * 3.0 + 1.0) * (x.as_promise() * 3.0 + 1.0)).sum(0, false);
let grads = y.backward();
// at x=2: 6*(3*2+1) = 6*7 = 42
assert_approx_eq!(grads[&x].data(), &[42.0]);

// Gradient flows through matmul
let w = Parameter::new(Tensor::from_scalar(1.0, &[3, 3]));
let x = Tensor::from_scalar(1.0, &[2, 3]);
let loss = linear(x, w, zeros(&[3])).sum(0, false).sum(0, false);
let grads = loss.backward();
assert_eq!(grads[&w].shape(), &[3, 3]);
```

### Documentation
- Add an `Autodiff` section to `README.md`
- Create `examples/gradient_descent.rs`: simple 1D regression for 10 steps
- Document the gradient rule for each `OpKind`

---

## Phase 9 — PromiseSkeleton

**Goal:** Pre-compile the execution plan once for a fixed graph topology and reuse it
across iterations with different data, avoiding the cost of re-planning on every call.

**Why:** In a training loop, graph topology and shapes are identical across iterations —
only leaf data changes. This is equivalent to CUDA graphs applied to the CPU path.

### Tests
```rust
// Skeleton and materialize produce identical results
let t_slot = Slot::new(&[4]);
let skeleton = (t_slot.as_promise() * 2.0 + 1.0).into_skeleton();

let input = arange(4);
let from_skeleton    = skeleton.run(&[&input]);
let from_materialize = (input.as_promise() * 2.0 + 1.0).materialize();
assert_eq!(from_skeleton.data(), from_materialize.data());

// Re-running with different data uses the same plan
let input2  = Tensor::from_scalar(5.0, &[4]);
let result2 = skeleton.run(&[&input2]);
assert_eq!(result2.data(), &[11.0; 4]);
```

### Documentation
- Add `PromiseSkeleton` to `README.md` with a training-loop example
- Document the slot-binding API

---

## Phase 10 — Backend / Dtype Split

**Goal:** Decouple the compute device (CPU-MKL) from the element type (f64, f32) so
that adding a no-MKL fallback and a future CUDA backend are well-defined extension
points rather than hacks. The no-MKL path also removes the Intel toolchain as a hard
requirement, which matters for CI and non-Intel platforms.

---

## Phase 11 — CUDA Backend

Long-horizon. Depends on the backend abstraction introduced in Phase 10. Each `OpKind`
maps to a CUDA kernel; async execution uses CUDA streams.

---

## Automated Testing

Tests should be written alongside each feature as it lands, not after the fact.

### Recommended crates

Add to `[dev-dependencies]` in `Cargo.toml`:

```toml
[dev-dependencies]
approx    = "0.5"   # assert_relative_eq!, assert_abs_diff_eq!
proptest  = "1"     # property-based testing
criterion = "0.5"   # benchmarks (goes under [[bench]])
```

### Directory structure

```
src/
  tensor/
    layout.rs        ← unit tests live here in #[cfg(test)] mod tests { }
    ops/
      fusion.rs      ← unit tests for every fusion rule
      impl_layout.rs ← unit tests for compute_layout per OpKind
tests/
  ops.rs             ← end-to-end op correctness (integration)
  graph.rs           ← materialization, shared nodes, caching
  layout.rs          ← slice/view/transpose/broadcast round-trips
  regression.rs      ← one test per bug that was found and fixed
benches/
  throughput.rs      ← criterion benchmarks
```

---

### Unit tests

Unit tests live inside the module they test using `#[cfg(test)] mod tests { }`.
They test one function or one code path in isolation.

**`mem_formats/layout.rs`**
- [ ] `from_shape` produces correct strides for 1D, 2D, 3D tensors
- [ ] `view` rejects incompatible total size
- [ ] `view` rejects non-contiguous input (`NonContiguousView`)
- [ ] `slice` produces correct shape, stride, and offset for a 2D subview
- [ ] `transpose` swaps shape and stride of the last two axes
- [ ] `transpose_axes` permutes correctly and rejects out-of-bound axes
- [ ] `is_contiguous` returns true for fresh `from_shape`, false after transpose
- [ ] `broadcast_to_shape` sets stride to 0 for expanded dimensions
- [ ] `broadcast_to_shape` sets `len` to the product of the *new* shape
- [ ] `broadcast_to_shape` returns `CannotBroadcast` for incompatible shapes

**`ops/fusion.rs`**
- [ ] Two consecutive `AxBy` ops fuse into one with correct `(a, b)` constants
- [ ] `AxBy` followed by `Exp` produces `FusedScalar([AxBy, Exp])`, not a single op
- [ ] `Exp` followed by `AxBy` produces `FusedScalar([Exp, AxBy])`
- [ ] View followed by AsContiguous fuses to just the View
- [ ] A node that is not scalar-compatible produces no fusion (`None`)

**`ops/impl_layout.rs`**
- [ ] `compute_layout` for `ScalarOp` returns the same layout as the input
- [ ] `compute_layout` for `Add` with equal shapes returns that shape
- [ ] `compute_layout` for `Add` with mismatched shapes returns `NotSameShape`
      with *both* shapes correctly reported (regression for the copy-paste bug)
- [ ] `compute_layout` for `Matmul` returns `[m, n]` for a `[m, k] @ [k, n]` input
- [ ] `compute_layout` for `Matmul` returns `NotSameBatch` for incompatible batch dims

---

### Integration tests (`tests/`)

Integration tests build graphs and materialize them end-to-end.
All expected values should be hand-computed and commented.

**`tests/ops.rs` — scalar ops**
```rust
// One test per OpKindScalar arm — catches copy-paste regressions
#[test] fn scalar_add() { /* ones(4) + 2.0 == [3;4] */ }
#[test] fn scalar_sub() { /* ones(4) - 2.0 == [-1;4] */ }
#[test] fn scalar_mul() { /* ones(4) * 3.0 == [3;4] */ }
#[test] fn scalar_div() { /* ones(4) / 4.0 == [0.25;4] */ }
#[test] fn scalar_exp() { /* from_scalar(0.0).exp() == [1.0;4] */ }
#[test] fn scalar_ln()  { /* from_scalar(1.0).ln()  == [0.0;4] */ }

// Fusion produces the same result as sequential application
#[test] fn fused_chain_correctness() {
    // (x * 2 + 3 - 1) applied with 20 ops matches single-pass
}

// Non-commutative ops are not reversed
#[test] fn sub_is_not_commutative() {
    let a = arange(4);      // [0,1,2,3]
    let b = ones(&[4]);
    let ab = (a - b).materialize(); // [-1,0,1,2]
    let ba = (b - a).materialize(); // [1,0,-1,-2]
    assert_ne!(ab.data(), ba.data());
}
#[test] fn div_is_not_commutative() { /* similar */ }
```

**`tests/ops.rs` — binary tensor ops**
```rust
#[test] fn tensor_add() { /* [1,2,3] + [4,5,6] == [5,7,9] */ }
#[test] fn tensor_sub() { /* [4,5,6] - [1,2,3] == [3,3,3] */ }
#[test] fn tensor_mul() { /* [1,2,3] * [4,5,6] == [4,10,18] */ }
#[test] fn tensor_div() { /* [4,6,8] / [2,3,4] == [2,2,2] */ }

// Shape mismatch panics (inline ops) and does not silently produce wrong output
#[test]
#[should_panic]
fn tensor_add_shape_mismatch_panics() {
    let _ = (ones(&[3]) + ones(&[4])).materialize();
}
```

**`tests/layout.rs` — shape operations**
```rust
#[test] fn view_zero_copy()       { /* same Arc pointer before and after */ }
#[test] fn transpose_zero_copy()  { /* same Arc pointer */ }
#[test] fn slice_zero_copy()      { /* same Arc pointer */ }
#[test] fn transpose_then_materialize_matches_manual() {
    // [[1,2],[3,4]].T == [[1,3],[2,4]]
}
#[test] fn slice_then_add()       { /* slice [2..4] of [0,1,2,3,4] + 1 == [3,4] */ }
#[test] fn as_contiguous_on_transposed_produces_correct_data() { }
```

**`tests/graph.rs` — graph execution**
```rust
// A node used by two branches is computed exactly once
#[test] fn shared_node_computed_once() {
    let t = arange(4).as_promise();
    let result = (&t * 2.0 - &t).materialize(); // == t
    assert_eq!(result.data(), &[0.0, 1.0, 2.0, 3.0]);
}

// CachedTensorPromise returns the same result on repeated materializations
#[test] fn cached_promise_stable() {
    let raw = arange(4);
    let cached = (raw.as_promise() + 1.0).cache();
    let r1 = (&cached * 2.0).materialize();
    let r2 = (&cached * 2.0).materialize();
    assert_eq!(r1.data(), r2.data());
}

// CachedTensorPromise is computed only once (result is shared, not recomputed)
#[test] fn cached_promise_computed_once() {
    // Build a graph where the cached node would produce different results
    // if re-evaluated (e.g., wrapping a counter). Verify it does not change.
}
```

---

### Regression tests (`tests/regression.rs`)

One test per bug that was found. Named clearly so it's obvious what broke.

```rust
// Bug: OpKindScalar::Sub was doing addition (copy-paste from Sum arm).
#[test] fn regression_scalar_sub_was_adding() {
    let t = Tensor::from_scalar(10.0, &[4]);
    assert_eq!((t - 3.0).materialize().data(), &[7.0; 4]);
}

// Bug: NotSameShape error reported inputs[0].shape() for both sides.
#[test] fn regression_not_same_shape_error_shows_both_shapes() {
    let a = ones(&[3, 4]);
    let b = ones(&[3, 5]);
    // The panic message must contain both "4" and "5".
    // Use std::panic::catch_unwind + check the string.
}

// Bug: unordered buffer reuse could pick rhs for output, reversing Sub/Div.
#[test] fn regression_sub_ordering_with_reusable_rhs() {
    let a = Tensor::from_scalar(10.0, &[4]);
    let b = (Tensor::from_scalar(3.0, &[4]).as_promise() + 0.0); // reusable
    assert_eq!((a - b).materialize().data(), &[7.0; 4]); // not [-7.0; 4]
}

// Bug: matmul inputs.pop() order was reversed (raw_a got rhs, raw_b got lhs).
// Add once matmul is implemented.
#[test] fn regression_matmul_input_order() {
    let a = Tensor::from_data(&[1.0,0.0, 0.0,1.0], &[2,2]); // identity
    let b = Tensor::from_data(&[1.0,2.0, 3.0,4.0], &[2,2]);
    // identity @ b == b, not b^T
    assert_eq!(a.matmul(&b).unwrap().materialize().data(), b.data());
}
```

---

### Property tests (`tests/property.rs`)

Property tests use `proptest` to generate random inputs and verify algebraic laws.
These catch edge cases that hand-written tests miss (large values, zero, negatives).

```rust
use proptest::prelude::*;

proptest! {
    // Commutativity of Add
    #[test]
    fn add_commutative(a in prop::collection::vec(-1e6f64..1e6, 1..100),
                       b in prop::collection::vec(-1e6f64..1e6, 1..100)) {
        prop_assume!(a.len() == b.len());
        let ta = Tensor::from_data(&a, &[a.len()]);
        let tb = Tensor::from_data(&b, &[b.len()]);
        let ab = (ta + tb).materialize();
        let ba = (tb + ta).materialize();
        assert_relative_eq!(ab.data(), ba.data(), max_relative = 1e-10);
    }

    // Commutativity of Mul
    #[test]
    fn mul_commutative(/* similar */) { }

    // Sub is anti-commutative: a - b == -(b - a)
    #[test]
    fn sub_anticommutative(a in vec, b in vec) {
        let ab  = (ta - tb).materialize();
        let neg_ba = (tb - ta) * -1.0).materialize();
        assert_relative_eq!(ab.data(), neg_ba.data(), max_relative = 1e-10);
    }

    // Additive identity: a + 0 == a
    #[test]
    fn add_zero_identity(a in vec) {
        let t    = Tensor::from_data(&a, &[a.len()]);
        let zero = Tensor::from_scalar(0.0, &[a.len()]);
        let result = (t + zero).materialize();
        assert_relative_eq!(result.data(), &a, max_relative = 1e-10);
    }

    // Multiplicative identity: a * 1 == a
    #[test]
    fn mul_one_identity(a in vec) { }

    // Scalar fusion produces the same result as sequential materialization
    #[test]
    fn fused_scalar_matches_sequential(
        a in -1e3f64..1e3,
        b in -1e3f64..1e3,
        c in -1e3f64..1e3,
        data in vec(-1e6f64..1e6, 1..1000),
    ) {
        let t = Tensor::from_data(&data, &[data.len()]);
        let fused      = (t.as_promise() * a + b - c).materialize();
        let sequential = data.iter().map(|&x| x * a + b - c).collect::<Vec<_>>();
        assert_relative_eq!(fused.data(), sequential.as_slice(), max_relative = 1e-10);
    }

    // View preserves all elements
    #[test]
    fn view_preserves_data(data in vec(any::<f64>(), 12..=12)) {
        let t = Tensor::from_data(&data, &[12]);
        let viewed = t.view(&[3, 4]).unwrap().materialize();
        assert_eq!(viewed.data(), &data);
    }
}
```

---

### Benchmarks (`benches/throughput.rs`)

Use `criterion`. Run with `cargo bench`.

```rust
use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};

fn bench_scalar_fusion(c: &mut Criterion) {
    let sizes = [1_000, 10_000, 100_000, 1_000_000];
    let mut group = c.benchmark_group("scalar_fusion");

    for &n in &sizes {
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, &n| {
            let t = arange(n);
            b.iter(|| {
                let mut p = t.as_promise();
                for i in 0..20 {
                    p = p + black_box(i as f64);
                }
                black_box(p.materialize())
            });
        });
    }
}

fn bench_matmul(c: &mut Criterion) {
    // 128x128, 512x512, 1024x1024
}

fn bench_non_contiguous_vs_contiguous(c: &mut Criterion) {
    // Compare (transposed + op).materialize() vs (contiguous + op).materialize()
    // to quantify the non-contiguous penalty.
}

criterion_group!(benches, bench_scalar_fusion, bench_matmul, bench_non_contiguous_vs_contiguous);
criterion_main!(benches);
```

---

### CI checklist

The GitHub Actions workflow should run all of the following on every push:

- [ ] `cargo fmt --check` — no formatting drift
- [ ] `cargo clippy -- -D warnings` — no lint regressions
- [ ] `cargo test` — all unit and integration tests pass
- [ ] `cargo test --features debug_only_check` — validation paths also exercised
- [ ] `cargo bench --no-run` — benchmarks compile (do not run timing on CI)

---

## Documentation Backlog

Items that are not blocked on any phase but should be done incrementally.

- [ ] Replace `simple_tensor` package name in `Cargo.toml` with `candela` to match
      `README.md` and the project name
- [ ] Add `#![doc = include_str!("../README.md")]` to `src/lib.rs` so `cargo doc`
      renders the README as the crate root
- [ ] Add `# Safety` doc comments to every `pub unsafe fn` explaining the invariants
      the caller must uphold
- [ ] Add `# Panics` doc comments to every function that can panic, including the
      inline `+`, `-`, `*`, `/` operators
- [ ] Create `examples/` directory with at minimum:
      - `lazy_eval.rs` — basic graph construction and materialization
      - `fusion.rs` — scalar fusion collapsing a 20-op chain
      - `cached_promise.rs` — shared preprocessing across multiple flows
      - `matmul.rs` — matrix multiplication
      - `gradient_descent.rs` — simple regression (after Phase 8)
- [ ] Conformance tests against NumPy for every op: generate inputs in Python, export
      as JSON, import in Rust tests and compare outputs within floating-point tolerance
- [ ] Add `benches/` with at minimum:
      - Materialization throughput for a scalar-fused chain (large tensor)
      - Matmul throughput vs naive loop
      - Non-contiguous vs contiguous op throughput ratio
