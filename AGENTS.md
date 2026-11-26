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

- Keep `ir::llil`, `ir::rtl`, `ir::mlil` and `ir::hlil` language-neutral and trait-only.
- The library owns the IR traits and the vocabulary that more than one storage or consumer
  shares: the dialect, view and builder traits, the instruction and memory-event traits, and the
  block, edge, region and handler identities.
- The library owns the graph algorithms, the generic fixpoint solvers, the transforms of the
  CFG shape and the generic `Cfg` container. A graph algorithm reads the shape of a graph
  (blocks, edges and instruction counts). It can also read the control-flow classification of
  an instruction through the shared traits (`FlowControl`, `CallInfo`, `AddressInstruction`),
  because that classification makes the graph. A renderer (DOT, text, control-tree
  pseudocode) takes the text of an instruction from the caller (`DisplayInstr`).
- Do not add an IR algorithm. An IR algorithm reads the operands, values, effects or
  operations of instructions: dataflow over instructions, SSA construction, memory SSA,
  lifting, lowering, and the verification or printing of IR. The consumer that owns the IR
  owns these algorithms and the hook traits that only they use.
- The consumer owns all IR storage, builders and editing. Its dialect owns operations, roots,
  constants, value types, variable metadata, effects, edge payloads, outcomes and constant
  folding.
- `effects` is the one exception to the rule on IR algorithms. It holds the shared effect
  vocabulary, the summarizer of RTL statements, the capture of one instruction and the statement
  tree, because every level and every consumer must compute effects with one set of rules. Keep
  it free of instruction-set names, and store no more than one instruction there.
- Do not add IR storage, or a language-, VM-, ABI- or ISA-specific variant, to a generic layer.
- Keep the crate `no_std` with `alloc`. Add a dependency only if it supports `no_std`.

## API conventions

- The namespace modules (`analysis`, `dataflow`, `graph`, `ir`, `transform`) declare `pub mod`
  children and re-export nothing. The crate root re-exports every public item, one `pub use`
  for each module, in alphabetical order. The root re-exports nothing from `ir`, because its
  levels share names (`ir::hlil::FunctionView`).
- A domain module with a narrow API (`cfg`, `exception`, `ir::llil`, `ir::rtl`, `ir::mlil`, `ir::hlil`)
  keeps its children private and re-exports its public items.
- Spell compound names in full with underscores (`control_dependence`). Keep the standard
  acronyms (`scc`, `cdg`, `eh`, `seh`, `veh`, `clr`).
- Make a derived analysis with `Type::compute`. Use `build` only for a fallible builder
  (`CfgBuilder::build`). Give a function that returns a graph a noun name (`call_graph`). A free
  function returns a standard type (`Vec`, `bool`, `Option`) or a graph; anything with its own
  result type is `Type::compute`.
- Mark a variant of a function only with these affixes: `try_` (it runs a fallible problem),
  `_in` (it takes a caller-owned scratch as its first argument) and `_mapped` (it also returns
  the identity remapping). Do not use `_with`, `_from`, `_of` or `_with_config` as variant
  markers: pass a config, a seed descriptor or a view instead, and give a genuinely different
  operation its own name.
- An analysis that starts from a root takes a `RootedView`. Choose the root with `Rooted` and the
  direction with `Reversed`; do not add a separate root or direction parameter. Where a
  direction is a value, use the one `Direction { Forward, Backward }`.
- A config is a plain struct with public fields that implements `Default`. Do not add `with_*`
  setters. Share a bound instead of redeclaring it (one depth bound, one step limit).
- A scratch type implements `Default`, starts empty and grows on entry to fit the graph. Only
  the `_in` form of a function takes it.
- A node-generic result type defaults its id parameters to `BlockId` and `EdgeId`.
- Use a `*Result` name only for a type in a `Result` position. Give each error type a name that
  ends in `Error`, and implement `core::error::Error`.
- Use `*_count` for counting accessors. Do not use `num_*`.
- `_mapped` marks the variant of a transform that also returns a `Rewrite`. A pass that only
  adds blocks or edges returns the new identities.
- Pass composition owns identity, declared order, change reports and fallible execution. The
  consumer owns pass selection, dependency order, dialect semantics and cache invalidation.
- The node and edge fixpoint solvers each have one entry point that takes the problem and one
  descriptor of the seeds and the `SolveConfig`, plus their `stream_` and `resume_` forms. A
  problem trait comes in an infallible and a fallible form; the fallible form has a blanket
  implementation for every infallible problem, and the solvers take the fallible form. A
  derived analysis wraps a solver or runs its own loop. It does not add entry points.
- The slot fixpoint drivers (`ordered_fixpoint`, `wave_fixpoint`) are schedulers over keyed
  slots that the caller owns. They have no graph and no problem trait, so they are not in the
  solver set above.
- Give sibling implementations parallel module paths and names (`breadth_first.rs`,
  `depth_first.rs`) behind one small coordinator module.
- Code that mutates a `Cfg` lives under `transform/` or `cfg/`; `graph/` holds analyses over
  views. The methods of `Cfg` live under `cfg/`.
- Build on the crate's shared pieces instead of hand-rolling them: the walk cores of
  `graph::traverse` and `graph::search` for breadth-first and depth-first walks; the views of
  `graph::view` (`Rooted`, `Reversed`, `Induced`, `VirtualRoot`) and the crate-private
  renumbered component view; `EpochMarks` for visited stamps; `DenseFanout::from_pairs` for
  counting-sort adjacency; `define_dense_id!` for index newtypes; and the shared no-index
  sentinel. When a module needs something the shared piece lacks, extend the shared piece.
- Return structured errors. Do not panic in a public API unless the panic is documented. Panic
  (under `# Panics`) only for a caller bug the caller can check cheaply, such as an id that is
  out of range or removed. Return a `Result` for input the caller cannot check cheaply: derived
  inputs that do not match, invalid structure, an exhausted budget, arithmetic overflow. Prefer
  a type that excludes invalid input (an unsigned cost) to a panic.
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
