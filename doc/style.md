# Documentation Style Guide

This guide defines what each documentation context is for and how to write it.
The goal is consistency: a reader should always know what kind of document they're
in and what they can expect from it.

---

## Two contexts, two registers

### 1. Source code docs — `///` rustdoc comments

**Who reads this:** Someone mid-coding who hits F1 or opens the docs looking for
a specific detail. They want a fast, precise answer.

#### Prose section

Describe what the type or function **is**, structurally. Name the actual
implementation types (`Arc<TensorGraphEdge<T>>`, `OnceLock`, etc.). State
invariants and guarantees explicitly — if the planner guarantees something, say
so. If a value can be `None` only under a specific condition, say so.

Don't narrate usage in the prose section. Don't write "to do X, call Y".
Don't repeat behaviour already documented by a related type — cross-reference
with a doc link instead.

**Good example:** [`OutputKind::Buffer`](../src/tensor/planner/plan.rs) — names
the type involved, states the planner guarantee in a single sentence:
> *"Re-use the buffer previously owned by node `id`. The planner guarantees that
> buffer is no longer referenced by any live node at this point."*

**Good example:** [`TensorGraphCacheNode`](../src/tensor/graph.rs) — names the
concrete implementation detail (`OnceLock<TensorData<T>>`), states the invariant
("runs at most once"), and explains the planner consequence ("its buffer survives
across separate `.materialize()` calls") — all without telling the user how to
use it.

**Bad pattern to avoid:**
```rust
// Tutorial narration — does not belong in the prose section
/// A tensor you can use to do computations. Call `.as_promise()` to start
/// building a graph and then `.materialize()` to run it.
```

#### `# Examples` section

This is exactly where usage guidance lives. Show realistic, runnable patterns.
One scenario per block; multiple blocks are fine. Examples are doctests — they
must compile and pass. Asserting the output is preferred over just printing it.

**Good example:** [`TensorPromise`](../src/tensor/promise.rs) and
[`CachedTensorPromise`](../src/tensor/promise.rs) — each has a focused `# Examples`
block that shows one realistic pattern and asserts the output.

**Good example:** [`Tensor::as_promise`](../src/tensor/tensor.rs) — shows the
specific pattern (`as_promise()` seeding a loop variable) that motivated the
method's existence.

#### Other sections

| Section | When to use |
|---------|-------------|
| `# Panics` | Any function that can panic, including operator overloads (`+`, `-`, …). Always include. |
| `# Safety` | Every `pub unsafe fn`. Explain the invariants the caller must uphold. |
| `# Note` | A known edge case, footgun, or surprising interaction. See [`topological_sort`](../src/tensor/planner/sort.rs) for an example of when to use this. |
| `# Errors` | Functions returning `Result`. Describe what `Err` variants can be returned and why. |

#### Tone

Technical. Precise. Short declarative sentences. No fluff.

---

### 2. User-facing docs — `README.md` and `doc/*.md`

**Who reads this:** Someone building a mental model of the project — either a
new contributor or someone trying to understand a subsystem before touching it.
They're reading top to bottom, not scanning for a specific fact.

#### What these documents do

- Lead with **motivation** — the problem being solved, and why this design and not another.
  See how [planner.md](planner.md) opens with "Why not just execute the graph
  directly?" before explaining anything about the algorithm.
- Tell the story behind decisions; describe what was **rejected** and why — often
  more illuminating than describing what was chosen. See the `reusable: bool` story
  in [planner.md](planner.md).
- Explain trade-offs explicitly: what you give up, what you gain. See the cache
  explanation in the [README](../README.md#cachedtensorpromiset):
  *"You pay the memory cost of keeping that tensor alive, which is why this is opt-in."*
- Address the reader directly when it helps. See [planner.md](planner.md):
  *"If you haven't read graph.md yet, it's worth a quick look first."*
- Can have personality. See [README](../README.md#memory-layout):
  *"To be fair, this is what every tensor library worth its salt does."*

#### What these documents don't do

- They don't replace source docs as a reference. They link to them.
- They don't need to cover every API detail. Cover the concepts; let the source
  docs handle the specifics.

#### Tone

Conversational but technically grounded. Narrative. Precise where precision matters.

---

### 3. In-code comments — `//`

Explain **why** something non-obvious is done: a hidden constraint, a subtle
invariant, a workaround for a specific bug. Do not explain *what* the code does —
well-named identifiers do that. See the executor comments in
[`TensorGraphNode::compute`](../src/tensor/graph.rs) for examples: each `//` comment
explains a race condition or a planner contract, not what the surrounding code does.

---

### 4. Example files — `examples/*.rs`

Walkthroughs of a feature or use case aimed at someone encountering the API for
the first time. Short explanatory comments before each section are appropriate and
expected — unlike source docs, these files are meant to be read linearly.
See [lazy_eval.rs](../examples/lazy_eval.rs) and [fusion.rs](../examples/fusion.rs).

The file should compile, run, and produce correct output. Assertions are preferred
over `println!` alone.

---

### 5. Tests — `#[test]` functions

**Who reads this:** Someone reading a test failure in `cargo test` output, or
someone scanning the test file to understand what a module guarantees.

#### Naming

Use `subject_scenario[_qualifier]`:

- **subject** — the thing under test. Usually matches the section comment or the
  primary function/type being exercised (`matmul`, `broadcast`, `slice`).
- **scenario** — the input condition or variation that makes this test distinct
  (`batched`, `non_contiguous`, `transposed_rhs`, `shape_mismatch`).
- **qualifier** — only when two tests cover the same scenario from different
  angles (`_lhs` vs `_rhs`, `_2x2` vs `_rectangular`).

Don't encode the expected outcome in the name. The assertion does that job, and
`cargo test`'s one-line summary only shows the name — telling you what scenario
was exercised is more useful than telling you what the assertion was.

**One exception: `_panics`.** This suffix encodes `#[should_panic]`, which
structurally changes how the test works — without the attribute, a panicking
test would *pass* for the wrong reason. That structural signal is worth keeping
in the name. No equivalent exception exists for `_returns_error`: an `is_err()`
check is just a normal assertion.

**Good examples** from [tests/matmul.rs](../tests/matmul.rs) and
[tests/broadcast.rs](../tests/broadcast.rs):
```rust
fn matmul_batched_values()              // scenario: batched inputs, distinct from shape-only test
fn matmul_transposed_rhs()              // scenario: rhs arrives pre-transposed
fn matmul_shape_mismatch()              // scenario: inner dims don't align
fn matmul_plus_bias_wrong_shape_panics()// scenario + _panics: structurally different test
fn broadcast_col_plus_row()             // scenario: column [3,1] + row [1,4]
```

**Bad patterns to avoid:**
```rust
fn add_equal_shapes_returns_that_shape()           // outcome in name: add_equal_shapes
fn matmul_returns_correct_output_shape()           // too vague + outcome: matmul_output_shape
fn matmul_shape_mismatch_returns_error()           // _returns_error is outcome: matmul_shape_mismatch
fn add_mismatched_shapes_returns_not_same_shape_with_both() // encodes error variant: add_shape_mismatch
fn matmul_batched_cpu()                            // _cpu is noise in a CPU-only test file
fn compute_as_contiguous_f64()                     // _f64 is noise in an f64-only test file
```

#### Boundary cases

**Boolean predicates.** When the subject is a boolean predicate method (`is_contiguous`,
`is_transposed`, `is_empty`), follow `predicate_scenario` — not
`predicate_true_for_scenario` or `predicate_false_for_scenario`. The `true`/`false` is
the assertion; the scenario is the input condition.

```rust
// Wrong — encodes the assertion result
fn is_contiguous_true_for_fresh_layout()
fn is_contiguous_false_after_transpose()

// Correct — predicate is the subject, input condition is the scenario
fn is_contiguous_fresh_layout()
fn is_contiguous_after_transpose()
```

**Qualifier vs outcome.** A qualifier like `_shape` or `_values` is fine when it
distinguishes two tests of the same scenario from different angles. It becomes outcome
noise when it appears only because that's what the assertion checks.

```rust
// Fine — two tests of the batched scenario from different angles
fn matmul_batched_shape()    // asserts output shape
fn matmul_batched_values()   // asserts computed values

// Wrong — _output_shape is pure outcome when there's no corresponding _values test
fn matmul_batched_output_shape()  // should just be matmul_batched
```

**Outcome verbs.** Words like "rejects", "produces", "fuses_to" describe what the code
*does in response* to an input, not the input condition itself. Drop them; the noun
carries the scenario.

```rust
// Wrong
fn view_rejects_incompatible_size()
fn axby_followed_by_exp_produces_fused_chain()

// Correct
fn view_incompatible_size()
fn axby_then_exp()
```

#### Structure within a file

Group related tests under a `// ──` banner comment. Every test in a group should share
the group's prefix — if the banner says `matmul`, every test below it starts with
`matmul_`. A file with more than one test group always uses banners — there are no
implicit groupings. See [tests/matmul.rs](../tests/matmul.rs) and
[tests/regression.rs](../tests/regression.rs) for existing examples.

---

## What not to do (applies everywhere)

- Don't write defensive guidance: *"you don't need to call X first"* or *"note that
  Y is not required."* Just describe what X is for.
- Don't use marketing language (*"the final answer"*, *"powerful and flexible"*).
- Don't repeat the function signature in prose form. Rustdoc already shows it.
- Don't put `#[inline]` between the doc comment and `pub fn`. The correct order is:
  doc comment → attributes → item. See the existing inconsistency in
  [`clone_deep` and `clone_detached`](../src/tensor/tensor.rs) as a counter-example
  to avoid.

```rust
// Correct
/// Create a tensor with every element set to `scalar`.
#[inline]
pub fn from_scalar(scalar: T, shape: &[usize]) -> Self { ... }

// Wrong — attribute between doc and fn
#[inline]
/// Create a tensor with every element set to `scalar`.
pub fn from_scalar(scalar: T, shape: &[usize]) -> Self { ... }
```
