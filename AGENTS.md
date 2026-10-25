# Agent rules

These rules apply to all agents that change this repository. Obey all rules in all changes.

- Make the smallest cohesive change that solves the whole problem.
- Preserve public behavior and compatibility unless the task calls for a breaking change.
- Do not add speculative abstractions, dependencies, configuration or features.

## Writing

- Use American English spelling (`analyze`, `color`, `normalize`).
- Document every public item. Update examples and crate documentation when behavior changes.
- Write comments that explain intent, invariants, tradeoffs or safety. Do not restate the code.
- Do not write section-divider or section-label comments. Extract a module instead.

## Lints: Clippy pedantic

- Treat the workspace `clippy::pedantic` policy and every warning as an error.
- Keep `[lints] workspace = true` in the `Cargo.toml` file of each crate.
- Fix the cause of a lint. Do not weaken the workspace lint configuration, and do not add
  crate-, module- or file-wide lint exceptions.
- A narrow `#[expect(lint, reason = "...")]` is acceptable only when the lint makes the code
  less correct or less clear. Put it on the smallest item. The workspace denies `#[allow]`.
- Do not use meaningless rewrites, dummy reads or conversions only to silence a warning.

## File size: 1000 lines

- Keep each hand-written source file at a maximum of 1000 lines, inline tests included.
- Split a file before it grows beyond that size. Split by responsibility, not at a random line.
- If a file is already longer, do not make it longer. Split it as part of a large change.
- A file marked as generated is exempt.

## Layout: Cargo package layout

Follow the Cargo package layout:
<https://doc.rust-lang.org/cargo/guide/project-layout.html>.

- Put source in `src/`, integration tests in `tests/`, examples in `examples/` and benchmarks in
  `benches/`.
- Give a target with many files its own directory with a `main.rs` file.
- Use kebab-case names for targets. Use snake_case names for modules.
- Put a module in `foo.rs` and its child modules in `foo/`. Do not use `mod.rs` files.

## IR boundary

- Keep `ir::rtl`, `ir::mlil` and `ir::hlil` language-neutral and trait-only.
- The library owns the IR traits and the generic algorithms that read through them.
- The consumer owns all IR storage, builders and editing. Its dialect owns operations, roots,
  constants, value types, variable metadata, effects, edge payloads, outcomes and constant
  folding.
- An algorithm can make a new function only through a consumer builder trait
  (`ir::hlil::FunctionBuilder`, `ir::hlil::FlatBuilder`). It seals the function only through
  `finish`, and it reads its input only through view traits. It never edits its input.
- Do not add IR storage, or a language-, VM-, ABI- or ISA-specific variant, to a generic layer.
- Keep the crate `no_std` with `alloc`. Add a dependency only if it supports `no_std`.

## API conventions

- The namespace modules (`analysis`, `dataflow`, `graph`, `ir`, `transform`) declare `pub mod`
  children and re-export nothing. The crate root re-exports every public item, one `pub use`
  for each module, in alphabetical order. The root re-exports nothing from `ir`, because its
  levels share names (`ir::hlil::FunctionView`).
- A domain module with a narrow API (`cfg`, `exception`, `ir::rtl`, `ir::mlil`, `ir::hlil`)
  keeps its children private and re-exports its public items.
- Spell compound names in full with underscores (`constant_propagation`). Keep the standard
  acronyms (`ssa`, `scc`, `sccp`, `dce`, `pre`, `cdg`, `pdg`, `eh`, `seh`, `veh`, `clr`, `expr`).
- Make a derived analysis with `Type::compute`. Use `build` only for a fallible builder
  (`CfgBuilder::build`). Give a function that returns a graph a noun name (`call_graph`).
- Use a `*Result` name only for a type in a `Result` position. Give each error type a name that
  ends in `Error`, and implement `core::error::Error`.
- Use `*_count` for counting accessors. Do not use `num_*`.
- `_mapped` marks the variant of a transform that also returns a `Rewrite`. A pass that only
  adds blocks or edges returns the new identities.
- Pass composition owns identity, declared order, change reports and fallible execution. The
  consumer owns pass selection, dependency order, dialect semantics and cache invalidation.
- Each fixpoint solver (`solve_problem`, `solve_node_problem`, `solve_edge_problem`) has the
  same set: full solve, `_from`, `_with_config` and `try_` forms. Each is a fallible core with
  infallible wrappers, on one shared worklist, with `SolveConfig`, `SolveError` and
  `TrySolveError`. A derived analysis wraps a solver or runs its own loop. It does not add to
  the set.
- Give sibling implementations parallel module paths and names (`breadth_first.rs`,
  `depth_first.rs`) behind one small coordinator module.
- Return structured errors. Do not panic in a public API unless the panic is documented.
- Use `unwrap` or `expect` only for a proven invariant. Make an `expect` message state the
  invariant.

## Tests

- Add a regression test for each defect and each change of observable behavior.
- Test behavior and invariants, not implementation details.
- Test the generic RTL, MLIL and HLIL with toy dialects that are not a real consumer.
- Put unit tests in a `#[cfg(test)] mod tests` in the file that they test. Move them to
  `<parent>/tests.rs` when the file gets near the size limit.
- Cover edge cases, failure paths and exceptional control flow where they apply.
- Keep tests deterministic.
- Do not remove or weaken a test to make a change pass.

## Checks

Make sure that all these checks pass before you report a change as complete:

```text
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings -D clippy::pedantic
cargo clippy --workspace --all-targets --all-features -- -D warnings -D clippy::pedantic
cargo nextest run --workspace --all-features
cargo test --workspace --all-features --doc
RUSTDOCFLAGS="-D warnings" cargo doc --workspace --all-features --no-deps
cargo check -p cfglib --all-features --target x86_64-unknown-none
```

`.github/workflows/lint.yml` runs the same checks, the benchmark lints and the `control-tree`
example.
