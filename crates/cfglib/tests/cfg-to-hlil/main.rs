//! The cross-module pipeline of a source-language frontend (symtree-ir
//! shaped): build a `Cfg` of statements, analyse it, structure it with
//! `ControlTree::compute_with_report`, and lift it to the structured IR with
//! `lift_with_tree`, verify, and print.
//!
//! The toy here is deliberately small and unrelated to the one of the
//! `hlil-pipeline` target: a `Tiny` dialect over numbered variables, a
//! statement type that is the instruction of the graph, and `usize` handles.
//! Its two reference interpreters, one over the CFG and one over the
//! structured function, judge each lift by what the programs observe,
//! including the `goto` fallback of an irreducible graph. `pipeline` runs the
//! whole chain: dominators, post-dominators, SSA, SCCP, pruning of what the
//! constants prove dead, structuring, and the lift.

/// The package's golden-file comparison, shared with the unit tests.
#[path = "../../src/test_util/golden.rs"]
mod golden;

mod dialect;
mod graphs;
mod machine;
mod pipeline;
mod storage;

mod structuring;
