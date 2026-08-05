# Benchmarks

One target, `bench_main`, containing every group. `benches/common/mod.rs` holds
the sizing/layout/throughput methodology; `benches/ops/` holds the operation
suite, one module per category.

```
cargo bench                       # everything (~900 benchmarks, ~45 min)
cargo bench --features mkl        # adds the CpuMkl lines to every group
cargo bench -- matmul             # just the matmul groups
```

## Reading a group

Groups are named `<category>/<op>`. Inside each, the criterion "function" is the
line and the size label is the x-axis, so the comparison plot overlays every
line against size:

| Line | What it is |
|---|---|
| `pure/f32`, `pure/f64` | `Skeleton::run` on `CpuPure` |
| `mkl/f32`, `mkl/f64` | the same on `CpuMkl`, only under `--features mkl` |
| `base/<dtype>` | plain scalar loop, **fresh output buffer every iteration** - the honest match for `Skeleton::run`, which allocates afresh |
| `base_reuse/<dtype>` | the same kernel into a buffer allocated once. `base` minus `base_reuse` is the per-run allocation cost |
| `simd/<dtype>` | hand-written `wide` kernel at the backend's own lane width (`f32x16`, `f64x8`) |

Every reference kernel is checked against the skeleton's own output before it is
timed, at a small fixed size. A kernel that computes the wrong thing usually
computes it faster.

## Filtering

Criterion matches the filter as a regex against the **full** benchmark id -
`binary/add/pure/f32/L2/contig` - not against the group name. So:

```
cargo bench -- "binary/add/"              # add only, not add_layouts
cargo bench -- "unary/.*/pure/f32/"       # one cell across the unary category
cargo bench -- "/DRAM/"                   # the memory-bound rung everywhere
cargo bench -- "matmul/transposed_b/"     # one group
```

Anchoring with `$` does not do what it looks like: `binary/add$` matches nothing,
because the id continues past the group name.