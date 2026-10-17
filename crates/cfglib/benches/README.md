# Performance benchmark

`performance/main.rs` is the crate root for a dependency-free synthetic
benchmark. It registers the original hot-path cases (`performance/hot_paths/`)
and the facade API cases (`performance/api_cases/`) into one
`BenchmarkSuite`; the harness, fixtures, and oracles live beside it. The
harness has separate CPU and allocation builds so allocator instrumentation
cannot perturb timing results.

[RESULTS.md](RESULTS.md) is a dated record of one optimization session and its
measurement limits, not a description of the current tree.

`graph-store.rs` is a separate, self-contained target measuring `Graph` at
the two scales it exists for. The first table is symbol-graph scale: one
million nodes, four million skewed-degree edges, a hundred thousand
incremental appends, and compaction. The second is procedure scale: five
hundred nodes and fifteen hundred edges built without a stated capacity and
thrown away, a thousand times over, where the whole measurement is the fixed
per-append cost rather than the memory system. It follows the same two-build
convention, prints one line per case, and takes the fastest of five runs.

The large table is sensitive to whatever else the machine is doing; pin the
process to one core and raise its priority before comparing two revisions of
it. The procedure-scale table is stable without that.

`analysis-scratch/main.rs` is a third self-contained target. It reports what
one callable's flow build allocates, phase by phase (reverse postorder, the
dominator tree, dominance frontiers, phi placement, SSA, the memory trace,
memory SSA, and memory-carried value flow), on a chain of if/else diamonds at
three procedure sizes, and what each scratch-taking entry point costs once its
working storage is warm. The CPU build prints best-of-seven nanoseconds per
call; the allocation build prints allocations and requested bytes per call. An
optional argument filters by substring of a `phase/size` name such as
`ssa-form-in/large`.

## Adding a benchmark

Every case is registered through `benchmark_case!`, which keeps the operation,
its semantic oracle, and the public facade functions it measures together:

```rust,ignore
benchmark_case!(
    suite,
    "api_topological_sort",
    covers [topological_sort],
    || topological_sort(&graph),
    |order: &Option<Vec<_>>| assert_eq!(order.as_ref().map(Vec::len), Some(node_count)),
);
```

The `covers` list can contain several exact aliases that share the measured
implementation. `benchmark_coverage!` attaches an alias family to an existing
case when the case lives in the original hot-path registry. `BenchmarkSuite`
rejects duplicate case names, duplicate API assignments, unknown API names,
filters that select nothing, and any public facade function without coverage.
The operation's oracle always runs before measurement.

Hot-path cases belong in the `performance/hot_paths/` module that owns their
responsibility. API-focused cases are split by responsibility under
`performance/api_cases/`; reusable instruction and graph fixtures live in
their sibling fixture modules. Add a focused case to the appropriate module
instead of extending `main.rs`. The `PUBLIC_API_FUNCTIONS` inventory lists
every root-level free function re-exported by `lib.rs`; update it when the
facade changes. The last line of a run (`coverage:`) prints the current counts.
Constructors, accessors, and associated analysis entry points are exercised by
the workloads but are not individually timed as facade functions.

`api_cases/traversal.rs` holds the small-run shapes that the 1,024-node chains
of `graph.rs` hide behind per-step costs: many short chases and searches, a
dense fan-out read once per parser step, layered route and fold walks, and a
linear stack reverse index. Cases there that time a method or constructor name
no facade function.

`performance/rtl_placement.rs` holds the one fixture that is not a `Cfg` or a
`Graph`: a synthetic register-transfer function, the consumer-owned view that
`ir::rtl` facts read, with the oracle behind `rtl_param_placement`. The case
times `ParamPlacement::compute` alone, with the dominator tree, frontiers,
summaries, and liveness built before timing.

## CPU timing

The default build installs `std::alloc::System` directly as the global
allocator. From the workspace root in PowerShell:

```powershell
$env:CFGLIB_BENCH_MS = "300"
cargo bench -p cfglib --bench performance -- cfg_dominators
Remove-Item Env:CFGLIB_BENCH_MS
```

The final argument is an optional substring filter. Remove `cfg_dominators` to
run every case. A nonempty filter that matches no case fails instead of
silently producing an empty run. `CFGLIB_BENCH_MS` controls the minimum duration
of each timing sample and defaults to 75 ms.

Before timing, every selected operation runs once and a case-specific semantic
oracle checks its complete result. The harness then calibrates an iteration
count, runs seven samples, and prints the median and minimum nanoseconds per
operation. Analysis fixtures are constructed before timing; cases containing
`build` intentionally measure construction.

Mutation cases clone their fixture inside the measured operation and return the
complete mutated CFG with any scalar status. The original hot-path cases name
that setup with `clone` and include adjacent clone-only controls. API coverage
cases also clone before mutation so repeated iterations remain independent;
interpret their result as clone-plus-operation cost.

## Allocation pressure

Rebuild the benchmark with its local allocation configuration enabled:

```powershell
$env:RUSTFLAGS = "--cfg cfglib_bench_alloc"
cargo bench -p cfglib --bench performance -- cfg_dominators
Remove-Item Env:RUSTFLAGS
```

This mode wraps `System` and reports three per-operation fields:

- `allocs`: successful allocations and reallocations;
- `bytes`: total requested bytes, including a reallocation's full new size;
- `peak`: maximum incremental live requested bytes above the operation's
  starting point.

The instrumented binary checks that each operation returns to its starting
live-byte baseline. These figures describe allocator requests, not process RSS,
and exclude allocator metadata and fragmentation. Allocation mode runs the
semantic oracle before enabling counters, performs an unmeasured warm-up, and
then measures one operation. Pair these results with CPU-mode timing from the
same case and revision.
