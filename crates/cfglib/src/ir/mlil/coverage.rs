//! Provably equivalent exception-coverage extension.
//!
//! Compilers trim exception-table ranges at the last throwing
//! instruction, splitting the non-throwing tail of a source construct (a
//! `return` after an inner catch, a cleanup epilogue) out of the coverage
//! its construct carries. The trim is semantically invisible — a block
//! that cannot throw is covered or not without any observable difference
//! — but it defeats tree structuring, which recovers source shapes from
//! region shapes.

extern crate alloc;

use alloc::collections::BTreeSet;

use crate::{BlockId, Cfg};

use super::{Dialect, Instruction};

/// Extends every region's protected set with provably equivalent coverage.
///
/// A block joins a protected set when it cannot throw (no instruction
/// with [`may_throw`](Instruction::may_throw)) and every sequential
/// predecessor already lies in the set, growing each region to a
/// fixpoint. Exceptional predecessor edges are disregarded: they deliver
/// control from the runtime, not from inside the protected extent.
///
/// Mutates the graph in place; run it through
/// [`Function::with_derived_cfg`](super::Function::with_derived_cfg) so
/// the canonical function keeps the exact declared coverage.
/// Returns the number of protected-block memberships added across all regions.
pub fn extend_equivalent_coverage<D: Dialect>(cfg: &mut Cfg<Instruction<D>, D::Edge>) -> usize {
    let mut extended = 0usize;
    for index in 0..cfg.regions().len() {
        let id = cfg.regions()[index].id;
        let mut protected = cfg.regions()[index].protected_blocks.clone();
        if protected.is_empty() {
            continue;
        }
        loop {
            let mut grown = false;
            let mut candidates: BTreeSet<BlockId> = BTreeSet::new();
            for &block in &protected {
                for edge in cfg.outgoing(block) {
                    let reference = cfg.edge(edge);
                    if !reference.kind().is_exceptional() {
                        candidates.insert(reference.target());
                    }
                }
            }
            for candidate in candidates {
                if protected.contains(&candidate) {
                    continue;
                }
                let throws = cfg
                    .block(candidate)
                    .instructions()
                    .iter()
                    .any(Instruction::may_throw);
                if throws {
                    continue;
                }
                let enclosed = cfg.incoming(candidate).all(|edge| {
                    let reference = cfg.edge(edge);
                    reference.kind().is_exceptional() || protected.contains(&reference.source())
                });
                if enclosed {
                    protected.insert(candidate);
                    extended += 1;
                    grown = true;
                }
            }
            if !grown {
                break;
            }
        }
        if let Some(region) = cfg.region_mut(id) {
            region.protected_blocks = protected;
        }
    }
    extended
}
