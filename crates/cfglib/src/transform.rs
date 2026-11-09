//! CFG transformation passes and the analyses that serve them.
//!
//! The passes that mutate a [`Cfg`](crate::Cfg) edit it in place and each
//! reports its change in the form that suits it: a count of the blocks,
//! edges, or instructions affected; a [`Rewrite`](crate::Rewrite) of the
//! identities that changed (the `_mapped` variants); or the identities it
//! created. Linearization and layout relaxation read a graph and return a new
//! value. Every pass entry point is re-exported at the crate root.
//!
//! - [`cleanup`] — unreachable removal, block merging, empty-block bypass, combined simplify.
//! - [`contract`] — edge contraction and node splitting.
//! - [`critical`] — critical edge splitting (required for SSA).
//! - [`layout`] — width relaxation for symbolic linear layouts.
//! - [`linearize`] — re-serialize a CFG back to a flat instruction stream.
//! - [`pass`] — named, ordered, fallible pass composition and reports.

pub mod cleanup;
pub mod contract;
pub mod critical;
pub mod layout;
pub mod linearize;
pub mod pass;
