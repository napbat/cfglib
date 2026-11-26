//! Verification of a control-flow graph: its structure and a consumer's
//! semantics.
//!
//! A [`Cfg`] is an edge-aware rooted view of itself, so its structure is held
//! to the one definition of the invariants in
//! [`graph::verify`](crate::graph::verify), plus the exception metadata only a
//! CFG has. [`verify_with`] layers the checks of a consumer's
//! [`SemanticValidator`] on top and returns both kinds of error in one
//! [`VerifyReport`].

extern crate alloc;
use alloc::vec::Vec;

use crate::block::BlockId;
use crate::edge::EdgeId;
use crate::graph::verify::{VerifyError, VerifyReport, verify_edge_view};

use super::Cfg;

/// Consumer-defined semantic checks layered over structural CFG verification.
///
/// Hooks run in block allocation order, then live edge identity order, then
/// once for whole-graph checks. Errors are consumer types, allowing frontends
/// to report branch cardinality, handler order, continuation constraints, or
/// format-specific provenance without putting those semantics in cfglib.
pub trait SemanticValidator<I, E> {
    /// Structured consumer error.
    type Error;

    /// Validate one block and its ordered adjacency.
    fn validate_block(&self, _cfg: &Cfg<I, E>, _block: BlockId, _errors: &mut Vec<Self::Error>) {}

    /// Validate one stable live edge.
    fn validate_edge(&self, _cfg: &Cfg<I, E>, _edge: EdgeId, _errors: &mut Vec<Self::Error>) {}

    /// Validate relationships that span several blocks or edges.
    fn finish(&self, _cfg: &Cfg<I, E>, _errors: &mut Vec<Self::Error>) {}
}

/// Validate structural invariants of a CFG.
///
/// A [`Cfg`] is an edge-aware rooted view of itself, so this is
/// [`verify_edge_view`] over the entry block: the CFG store and consumer-owned
/// storage are held to one definition of every invariant, and one set of
/// messages. Checks performed:
/// 1. The entry block index is within bounds.
/// 2. Every live edge references valid source/target block IDs.
/// 3. Adjacency lists (`succs`/`preds`) are consistent with edges, and every
///    live edge is listed exactly once by each of its endpoints.
/// 4. Every reachable non-entry block has at least one predecessor.
/// 5. The exception metadata keeps the invariants of [`Cfg::check_regions`]:
///    in particular every block it names by a single reference (a handler
///    entry, a filter block, a cleanup exit block, a continuation target) is
///    live, the references [`Cfg::compact`] panics on. The first broken one
///    is reported. The entry cannot be removed, so it is live by
///    construction.
///
/// Returns a [`VerifyReport`] containing all violations found.
///
/// # Examples
///
/// ```
/// use cfglib::{Cfg, EdgeKind, verify};
///
/// let mut cfg = Cfg::<u32>::new();
/// let b1 = cfg.new_block();
/// cfg.add_edge(cfg.entry(), b1, EdgeKind::Fallthrough);
///
/// let result = verify(&cfg);
/// assert!(result.is_ok());
/// ```
#[must_use]
pub fn verify<I, E>(cfg: &Cfg<I, E>) -> VerifyReport {
    let mut report = verify_edge_view(cfg);
    if let Err(error) = cfg.check_regions() {
        report.errors.push(VerifyError {
            message: alloc::string::ToString::to_string(&error),
        });
    }
    report
}

/// Validate structural invariants and consumer semantics in stable order.
///
/// The report holds the violations [`verify`] finds and, in hook order, the
/// errors the validator emits.
#[must_use]
pub fn verify_with<I, E, S>(cfg: &Cfg<I, E>, validator: &S) -> VerifyReport<S::Error>
where
    S: SemanticValidator<I, E>,
{
    let errors = verify(cfg).errors;
    let mut semantic_errors = Vec::new();
    for block_id in cfg.block_ids() {
        validator.validate_block(cfg, block_id, &mut semantic_errors);
    }
    for edge in cfg.edges() {
        validator.validate_edge(cfg, edge.id(), &mut semantic_errors);
    }
    validator.finish(cfg, &mut semantic_errors);
    VerifyReport {
        errors,
        semantic_errors,
    }
}

#[cfg(test)]
mod tests;
