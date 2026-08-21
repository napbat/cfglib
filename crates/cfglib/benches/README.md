# Performance benchmark

`performance.rs` is the crate root for a synthetic benchmark whose support
modules live in `performance/`. It has two deliberately separate build modes
so allocation instrumentation cannot perturb the CPU numbers used for
comparisons.

The latest local before/after matrix, retained changes, tradeoffs, and rejected
experiments are recorded in [RESULTS.md](RESULTS.md).

## CPU timing (default)

The default build installs `std::alloc::System` directly as the global
allocator. There is no counting wrapper or atomic operation on allocation
paths in this binary.

On Linux, pinning both runs to the same otherwise-idle core makes comparisons
less noisy:

```sh
env -u RUSTFLAGS -u CARGO_ENCODED_RUSTFLAGS \
  CFGLIB_BENCH_MS=300 taskset -c 2 \
  cargo bench -p cfglib --locked --bench performance -- cfg_dominators
```

The final argument is an optional substring filter. Remove
`cfg_dominators` to run every case. A nonempty filter that matches no case
fails with status 2 instead of producing an empty successful run.
`CFGLIB_BENCH_MS` is the minimum duration of each timing sample and defaults
to 75 ms; zero, non-integer, and non-UTF-8 values also fail with status 2.

Before measurement, every selected operation runs once and its complete result
is checked by a case-specific semantic oracle. The harness then calibrates an
iteration count to reach the target duration, runs seven timing samples, and
prints the median and minimum nanoseconds per operation. Analysis fixtures are
constructed before timing; cases whose names contain `build` intentionally
measure construction. Every measured operation returns its complete result,
which the harness passes through `black_box` and then drops inside the measured
iteration.

Mutation cases clone their fixture inside the measured operation and return the
complete mutated CFG together with any scalar status. Their names therefore use
`clone_…`, and adjacent clone-only controls expose setup cost; they are not
transform-only measurements.

## Allocation pressure (instrumented build)

Rebuild the benchmark with its local allocation cfg enabled:

```sh
env -u CARGO_ENCODED_RUSTFLAGS \
  RUSTFLAGS='--cfg cfglib_bench_alloc' \
  taskset -c 2 \
  cargo bench -p cfglib --locked --bench performance -- cfg_dominators
```

This mode wraps `System` and prints three additional per-operation fields:

- `allocs`: successful allocations and reallocations;
- `bytes`: total requested allocation bytes, counting a reallocation's full
  new requested size;
- `peak`: maximum incremental live requested bytes above the pre-operation
  baseline.

The instrumented binary tracks live bytes continuously and verifies that each
measured operation returns to its starting live-byte baseline. The figures are
allocator requests, not process RSS, and do not include allocator metadata or
fragmentation.

Allocation mode runs the semantic oracle before enabling counters, then
performs one additional unmeasured warm-up operation followed by one measured
operation. It does not report CPU timing because every allocator call in this
build passes through the counting wrapper. Pair its allocation results with
default-mode CPU results from the same benchmark case and revision.
