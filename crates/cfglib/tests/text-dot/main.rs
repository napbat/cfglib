//! Consumer-side tests of the text and DOT output formats: a tool that dumps
//! a code-intelligence graph or a CFG for a human, diffs two dumps, and reads
//! one back.
//!
//! Text is checked by algebraic law (write, parse, write is a fixed point, and
//! the parsed value is isomorphic to the input). DOT is checked against a
//! small scanner written here, so node and edge counts, identifiers, and
//! escaping are compared with the graph itself rather than with the writer's
//! own idea of its output. Goldens pin only the formatted text a reviewer reads.

/// The package's golden-file comparison, shared with the unit tests.
#[path = "../../src/test_util/golden.rs"]
mod golden;

mod cfg_text;
mod display;
mod dot_cfg;
mod dot_graph;
mod dot_scanner;
mod fixture;
mod graph_text;
mod instruction;
mod labels;
mod sink;
