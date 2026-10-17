//! Consumer-side tests of the `serde` representation: a tool that persists a
//! CFG, a store, a scope or stack graph, or an exception model, and loads it
//! back in another process.
//!
//! A decoded value is checked against the original through the public
//! accessors and analyses a consumer would run on it (a dominator tree, the
//! text form, a resolution), and by continuing to mutate both copies the same
//! way. Documents written before a field existed must still load wherever the
//! field has a documented default, and documents that are not a value of the
//! type must be rejected with an error rather than loaded into something that
//! panics later. The serialized form of a small fixture is a golden.

#![cfg(feature = "serde")]

/// The package's golden-file comparison, shared with the unit tests.
#[path = "../../src/test_util/golden.rs"]
mod golden;

mod cfg_documents;
mod consumer;
mod exception_models;
mod graph_documents;
mod hostile_documents;
mod json;
mod region_documents;
mod scope_stack_documents;
