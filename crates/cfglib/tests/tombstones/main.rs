//! Integration tests: analyses over stores with removed slots.
//!
//! The crate promises that a store with removed blocks, nodes, or edges is
//! neither unsound to analyze nor obliged to compact first: identities keep
//! their numbers until `compact` reports a `Renumbering`. A consumer that
//! runs an analysis between a transform and the compaction, or caches results
//! across a compaction, relies on that promise.
//!
//! Every analysis here is checked three ways against reference algorithms
//! that read only the plain topology (`oracle`), never a cfglib view: on the
//! store with holes, on the compacted store through the identities its
//! `Renumbering` reports, and on both a `Cfg` and a `Graph` so the contract is
//! the same for both stores.

mod cfg_analyses;
mod contracts;
mod oracle;
mod topology;
mod view_analyses;
