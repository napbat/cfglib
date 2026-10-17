# Repository Agent Guidelines

These instructions apply to the entire repository unless a more specific
`AGENTS.md` exists in a subdirectory.

## Priorities

- Optimize for correctness, clarity, maintainability, and a small review surface.
- Make the smallest cohesive change that fully solves the problem.
- Preserve public behavior and compatibility unless the task explicitly calls for
  a breaking change.
- Prefer straightforward code over clever code. Make invariants obvious.
- Do not add speculative abstractions, dependencies, configuration, or features.

## Rust Quality Gates

Before handing off a Rust change, run the relevant targeted tests while iterating,
then run all of these from the workspace root:

```text
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings -D clippy::pedantic
cargo clippy --workspace --all-targets --all-features -- -D warnings -D clippy::pedantic
cargo nextest run --workspace --all-features
cargo test --workspace --all-features --doc
```

Both feature sets are lint gates: code behind `#[cfg(feature = "serde")]` is
checked only by the all-features run, and code the feature does not touch must
also build without it. The `serde` feature is additive, so one all-features
test run covers every line. `cargo-nextest` runs each test in its own process,
in parallel; it does not run doctests, so `cargo test --doc` runs them. CI
(`.github/workflows/lint.yml`) runs a `lint` job and a `test` job in parallel.
The `lint` job also lints the allocation-counting benchmarks and checks the
`no_std` build for `x86_64-unknown-none`, a target without `std`, so a
dependency feature that turns `std` on fails the build. The `test` job also
runs the `control-tree` example.

Also build documentation with warnings denied when public APIs or docs change:

```text
RUSTDOCFLAGS="-D warnings" cargo doc --workspace --all-features --no-deps
```

In PowerShell, set `RUSTDOCFLAGS` with
`$env:RUSTDOCFLAGS = "-D warnings"` before running the documentation command.

### Clippy

- Treat the workspace's `clippy::pedantic` policy and every compiler or Clippy
  warning as an error.
- Fix the cause of a lint instead of hiding it or weakening workspace lint
  configuration.
- Do not add crate-, module-, or file-wide lint allowances.
- A narrow `#[allow(...)]` is acceptable only when the lint makes the code less
  correct or less clear. Keep it on the smallest item and add a short comment
  explaining why it is justified.
- Do not use meaningless rewrites, dummy reads, or unnecessary conversions merely
  to silence a warning.

## File and Module Size

- Keep every hand-written source file at or below 1,000 physical lines, including
  inline tests.
- Split files before they cross the limit. Extract modules by responsibility, not
  by arbitrary line ranges.
- Do not create catch-all modules or move complexity into a differently named
  oversized file.
- If a pre-existing file already exceeds 1,000 lines, do not make it larger. A
  substantial edit to that file must include a cohesive split that moves it toward
  compliance.
- Generated files are exempt only when they are clearly marked as generated and
  are not intended for manual editing.

## Cargo Project Layout

- Follow the [Cargo Book package-layout conventions](https://doc.rust-lang.org/cargo/guide/project-layout.html)
  within every workspace package.
- Keep package manifests at the package root and Rust implementation code under
  `src/`. Use `src/lib.rs` and `src/main.rs` for the default library and binary
  targets, and `src/bin/` for additional binaries.
- Put integration tests in `tests/`, examples in `examples/`, and benchmarks in
  `benches/` at the package root.
- Name binary, example, benchmark, and integration-test targets in `kebab-case`.
  Name Rust modules within those targets in `snake_case`.
- Give a multi-file binary, example, benchmark, or integration test its own
  `kebab-case` directory with a `main.rs` entry point and `snake_case` module
  files.
- Keep workspace-only tooling outside package `src/` trees unless it is an actual
  Cargo target.

## Naming and Layout Conventions

- Module files use the named-file style — `parent.rs` beside a `parent/`
  directory. `mod.rs` is forbidden.
- Top-level namespace modules (`analysis`, `dataflow`, `graph`, `ir`, `transform`)
  declare `pub mod` children and hoist nothing; the crate root is the single
  flat facade that re-exports every public item, one `pub use` per module,
  alphabetized. `ir` is the one exception: its three levels share names
  (`Dialect`, `FunctionView`), so the root re-exports none of them and
  callers write `ir::hlil::FunctionView`. Domain modules with a
  narrow API (`cfg`, `exception`, `ir::rtl`, `ir::mlil`, `ir::hlil`, and
  split coordinators such as `graph/search/events`) keep children private
  and re-export their public surface.
- Compound words in file and identifier names are underscore-separated and
  spelled in full (`constant_propagation`, `value_numbering`, `call_graph`);
  domain-standard acronyms (`ssa`, `scc`, `sccp`, `dce`, `pre`, `cdg`, `pdg`,
  `eh`, `seh`, `veh`, `clr`, `expr`) stay.
- A derived analysis or index is constructed with `Type::compute(...)`
  (`DominatorTree::compute`, `SsaForm::compute`, `EhModel::compute`).
  `build` is reserved for fallible builder entry points (`CfgBuilder::build`).
  Functions returning raw graphs use noun names (`condensation`, `call_graph`,
  `interference_graph`).
- `*Result` names are reserved for types used in `Result` positions; plain
  outputs are named for what they are (`Facts`, `VerifyReport`,
  `SccpAnalysis`, `PropagationStats`). Error types end in `Error` and
  implement `core::error::Error`.
- Counting accessors use the `*_count` suffix (`block_count`, `edge_count`,
  `value_count`); do not introduce `num_*` names.
- `_mapped` marks the `Rewrite`-returning variant of a transform that also
  exists without one; a transform whose only form returns a map takes no
  suffix. Passes that only add blocks or edges return the new identities
  directly instead of a map.
- Generic pass composition owns only stable identity, declared order, change
  reporting, and fallible execution. Consumers own pass selection, dependency
  order, dialect semantics, and any analysis-cache invalidation policy.
- Every generic fixpoint solver (`solve_problem`, `solve_node_problem`,
  `solve_edge_problem`) carries the identical facility matrix — full solve,
  `_from` seeding, `_with_config` bounds, and fallible `try_` counterparts —
  built as a fallible core with infallible wrappers, sharing `SolveConfig`,
  `SolveError`, and `TrySolveError`, and running on one shared worklist. A
  derived analysis (`Liveness`, `ReachingDefs`, `SccpAnalysis`, ...) wraps a
  solver unbounded or runs its own convergence loop; it does not grow the
  matrix.
- Use American English spelling in identifiers, documentation, and comments
  (`analyze`, `color`, `normalize`, `initialize`).
- Unit tests live inline in a `#[cfg(test)] mod tests` in the file they test;
  split them into `<parent>/tests.rs` when the file approaches the 1,000-line
  cap. Test modules are always named `tests`.

## Clean and Ergonomic Code

- Keep functions focused and control flow shallow. Prefer guard clauses when they
  make the happy path easier to follow.
- Keep semantic counterpart APIs parallel in naming, configuration shape, module
  placement, documentation, and tests. Prefer full-word canonical names; retain
  abbreviations only as compatibility aliases when existing callers require them.
- Give sibling implementations matching module paths and names (for example,
  `breadth_first.rs` and `depth_first.rs`) behind a small coordinator module that
  exposes their shared public surface.
- Use descriptive domain names and explicit domain types. Avoid boolean parameters,
  magic values, and ambiguous tuples when a small enum or struct communicates the
  contract better.
- Design public APIs so correct use is easy and invalid states are difficult to
  represent. Keep constructors, accessors, trait bounds, and ownership requirements
  no more restrictive than necessary.
- Preserve this workspace's `no_std` plus `alloc` compatibility unless the task
  explicitly changes that contract.
- Keep `ir::rtl`, `ir::mlil`, and `ir::hlil` language-neutral and trait-only.
  The library owns the IR traits and the generic algorithms that read through
  them and report facts. The consumer owns every IR storage, builder
  implementation, and editing machinery, and a consumer-defined dialect owns
  operations, roots, constants, value types, variable metadata, effects, edge
  payloads, outcomes, and constant folding. An algorithm may produce a new
  function through a consumer-implemented builder trait
  (`ir::hlil::FunctionBuilder`, `ir::hlil::FlatBuilder`): it appends to a
  function that the consumer's builder owns, it seals that function only
  through the builder's `finish`, and it reads its input only through view
  traits. It never edits the function it reads. Do not add IR storage, or
  language-, VM-, ABI-, or ISA-specific variants, to any generic layer.
- Avoid unnecessary cloning, allocation, collection, dynamic dispatch, and generic
  indirection. Optimize for a clear ownership story before micro-optimizing.
- Return structured errors for recoverable failures. Do not panic in public APIs
  unless the panic is part of a documented contract.
- Use `unwrap` or `expect` only for a proven invariant; make an `expect` message
  explain the violated invariant rather than repeat the immediate operation.
- Comments should explain intent, invariants, tradeoffs, or safety reasoning. Do not
  narrate code that is already self-explanatory.
- Do not use decorative section-divider comments made from repeated dashes, equals
  signs, or box-drawing characters. Express meaningful structure with cohesive
  modules, types, functions, and interface boundaries.
- Do not use a standalone comment merely to name or divide a section (for example,
  `// Parsing` or `// Constructors`). If the section is a distinct responsibility,
  extract a module with a narrow interface; otherwise omit the label.
- Keep public items documented, and update examples and crate-level docs when their
  behavior changes.

## DRY and Abstraction Boundaries

- Maintain one source of truth for each rule, representation, and conversion.
- Reuse existing domain types, graph views, algorithms, and helpers before adding a
  parallel implementation.
- Extract repeated behavior when it represents the same policy or invariant.
- Do not merge code that is merely syntactically similar but has different domain
  semantics or is likely to evolve independently.
- Prefer small, purpose-specific helpers over broad utility modules.
- Keep conversions and normalization at explicit boundaries instead of scattering
  them across callers.

## Tests

- Add a regression test for every bug fix and every externally observable behavior
  change.
- Test behavior and invariants rather than implementation details.
- Cover meaningful edge cases, failure paths, and interactions with exceptional
  control flow where relevant.
- Exercise generic RTL/MLIL/HLIL with distinct non-production toy dialects so
  bridge storage, edge translation, provenance, and analyses cannot
  accidentally depend on one consumer's semantic vocabulary or native-location
  type.
- Keep tests deterministic, focused, and readable. Use table-driven tests only when
  the cases share the same semantics and setup.
- Do not remove or weaken a test merely to make a change pass.

## Repository Hygiene

- Inspect the working tree before editing and preserve unrelated user changes.
- Keep formatting-only churn and opportunistic refactors out of focused changes.
- Do not use destructive Git commands, discard local work, commit, push, or rewrite
  history unless the user explicitly requests it.
- Add a dependency only when the benefit clearly outweighs its maintenance and
  compatibility cost. Confirm that it supports the repository's target and
  `no_std` requirements.
- Report the checks actually run and any checks that could not be run.
