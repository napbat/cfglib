//! Consumer-style tests of cfglib's memory event model, memory SSA, and
//! memory value flow, the way a binary lifter (reify, symtree-il) drives
//! them: its own instruction type reports ordered load, store, exchange, and
//! fence events over its own location vocabulary, and asks which stores can
//! reach which loads and which ordinary values memory carries.
//!
//! The oracles are brute-force: alias classes as connected components of the
//! alias relation, and the reaching-store sets found by exploring every
//! (program point, last write per class) state a random program can be in,
//! with unwind edges that leave before a throwing instruction.

mod alias;
mod machine;
mod oracle;
mod ssa_queries;
mod trace;
mod value_flow;
