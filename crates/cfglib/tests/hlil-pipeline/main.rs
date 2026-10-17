//! The structured IR pipeline as a consumer of `cfglib::ir::hlil` drives it
//! (the shape of the isaspec and symtree-ir integrations): `lower` a
//! structured function into flat bytecode, `lift` the bytecode back,
//! `recover_structure`, `verify`, and print.
//!
//! The consumer here is a toy with its own vocabulary: the `Calc` dialect,
//! `Vec`-backed structured storage, and a register bytecode over a `Cfg`.
//! The central oracle is execution. A reference interpreter for each level
//! runs a program before and after every transformation, and the observable
//! trace (the output events, the way the run ends, and the final memory)
//! must not change.

/// The package's golden-file comparison, shared with the unit tests.
#[path = "../../src/test_util/golden.rs"]
mod golden;

mod describe;
mod dialect;
mod flat;
mod flat_run;
mod machine;
mod pipeline;
mod programs;
mod source;
mod structured;
mod structured_run;

mod correspondence;
mod equivalence;
mod lift_errors;
mod lifting;
mod lowering;
mod placements;
mod recovery;
mod rendering;
mod verification;
mod walks;
