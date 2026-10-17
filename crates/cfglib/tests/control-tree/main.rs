//! Consumer-style tests of `ControlTree`, modeling symtree-cfg and the
//! decompilers built on it: a source-like program is lowered to a `Cfg` of toy
//! statements, structured into a tree, and read back.
//!
//! Every tree is held to oracles written here, not to the structurer: the
//! blocks it covers partition into exactly one leaf each, its nodes form a tree
//! (one visit each in preorder, one parent each), a fully structured report has
//! no goto, and the flow its shape implies equals the graph's own edges once
//! the edges the report names as gotos are accounted for.

/// The package's golden-file comparison, shared with the unit tests.
#[path = "../../src/test_util/golden.rs"]
mod golden;

mod builders;
mod coverage;
mod example_scenario;
mod implied_flow;
mod loops;
mod oracles;
mod reading;
mod structured_flow;
mod switches;
mod traversal;
mod tree_shape;
mod try_catch;
mod unstructured;
mod variants;
