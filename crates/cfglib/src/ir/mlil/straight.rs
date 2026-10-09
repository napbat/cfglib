//! Joins of straight runs of blocks.
//!
//! A lift keeps one block for each machine block, and the reductions after
//! it empty many of them: a block that held only a comparison whose result
//! moved into a branch, or a jump, forwards control and does nothing else.
//! [`Function::merge_straight_blocks`] takes those blocks out of the graph
//! and joins each block with the one block that only it enters, so the
//! function states its control flow once.

extern crate alloc;

use alloc::collections::BTreeSet;
use alloc::vec::Vec;
use core::iter;

use crate::region::{HandlerBody, RegionId};
use crate::{BlockId, Cfg, EdgeId};

use super::{EntityId, Error, Function, Instruction, ProvenanceMap, Result, VerifyDialect};

impl<D: VerifyDialect> Function<D> {
    /// Returns the function with its straight runs of blocks joined, and
    /// how many blocks that removed.
    ///
    /// An empty block with one non-exceptional successor is bypassed: each
    /// edge into it goes to its successor instead. An empty block that a
    /// branch enters stays, so each arm of the branch still names its own
    /// path. A block whose only edge is a non-exceptional edge to a block
    /// that no other edge enters takes the instructions and the edges of
    /// that block. The synthetic root stays empty and keeps its one entry
    /// edge, and the block that the root enters stays, so a fact about the
    /// entry block keeps its block.
    ///
    /// A block that an exceptional edge leaves holds the throw site of that
    /// edge. A join therefore refuses a block that holds a throwing
    /// instruction when the block that it would take has an exceptional
    /// edge. A block that a handler enters, or that a cleanup resumes from or
    /// at, keeps its identity. A join also needs both blocks in the same
    /// protected regions and known handler bodies, and a removed block
    /// leaves every region and body that held it.
    ///
    /// Every kept block, edge, and instruction keeps its identity and its
    /// provenance, so a block still names the block of the level that the
    /// function was translated from. A removed block or edge leaves its slot
    /// unused, as [`Cfg::remove_block`] does, and a rebuild of the function
    /// mirrors such a slot.
    ///
    /// # Errors
    ///
    /// Returns an error when the joined function fails verification.
    pub fn merge_straight_blocks(&self) -> Result<(Self, usize)> {
        let mut joined = self.clone();
        let removed = straighten(&mut joined.cfg);
        if removed == 0 {
            return Ok((joined, 0));
        }
        joined.reindex_instructions();
        let mut provenance = ProvenanceMap::new(self.provenance.source().clone());
        for entry in self.provenance.entries() {
            let kept = match entry.entity {
                EntityId::Block(block) => joined.cfg.contains_block(block),
                EntityId::Edge(edge) => joined.cfg.contains_edge(edge),
                EntityId::Instruction(_) | EntityId::Variable(_) => true,
            };
            if kept {
                // A stored entry has a valid span, so the insert cannot fail.
                let _ = provenance.insert(entry.source.clone(), entry.entity);
            }
        }
        joined.provenance = provenance;
        let report = joined.verify();
        if !report.is_ok() {
            return Err(Error::Verification(report));
        }
        Ok((joined, removed))
    }
}

/// Bypasses the empty forwarding blocks of `cfg` and joins its straight
/// runs until neither applies, and returns how many blocks that removed.
fn straighten<D: VerifyDialect>(cfg: &mut Cfg<Instruction<D>, D::Edge>) -> usize {
    let root = cfg.entry();
    let pinned = pinned(cfg);
    let mut removed = 0;
    loop {
        let before = removed;
        let order: Vec<BlockId> = cfg.block_ids().collect();
        for block in order {
            if block == root || !cfg.contains_block(block) {
                continue;
            }
            if cfg.block(block).is_empty() {
                // An arm of a branch keeps its block, so each arm still
                // names the path that it takes, even when it does nothing.
                // The exceptional edge of a throw site is no arm.
                let kept = pinned.contains(&block)
                    || cfg.incoming(block).any(|edge| {
                        let source = cfg.edge(edge).source();
                        source == root
                            || cfg
                                .outgoing(source)
                                .filter(|edge| !cfg.edge(*edge).kind().is_exceptional())
                                .nth(1)
                                .is_some()
                    });
                if !kept && let Some(target) = forwarded(cfg, block) {
                    let edge = sole_outgoing(cfg, block).expect("a forwarding block has one edge");
                    cfg.remove_edge(edge);
                    cfg.redirect_edges_to(block, target);
                    remove(cfg, block);
                    removed += 1;
                }
                continue;
            }
            while let Some(target) = joined(cfg, block)
                .filter(|target| !pinned.contains(target) && same_regions(cfg, block, *target))
            {
                let edge = sole_outgoing(cfg, block).expect("a joined block has one edge");
                let moved = core::mem::take(cfg.block_mut(target).instructions_mut());
                cfg.block_mut(block).instructions_mut().extend(moved);
                cfg.remove_edge(edge);
                cfg.move_outgoing_edges(target, block);
                remove(cfg, target);
                removed += 1;
            }
        }
        if removed == before {
            return removed;
        }
    }
}

/// Returns the blocks that an exception region or a cleanup names on their
/// own: each handler entry, and the block that each cleanup resumes from and
/// each of its routes resumes at.
fn pinned<I, E>(cfg: &Cfg<I, E>) -> BTreeSet<BlockId> {
    let mut pinned: BTreeSet<BlockId> = cfg
        .regions()
        .iter()
        .flat_map(|region| region.handlers.iter().map(|handler| handler.entry))
        .collect();
    for cleanup in cfg.cleanups() {
        pinned.extend(cleanup.resume_from);
        pinned.extend(
            cleanup
                .continuations
                .iter()
                .map(|continuation| continuation.resume),
        );
    }
    pinned
}

/// Tests whether two blocks lie in the same protected regions and in the
/// same known handler bodies.
fn same_regions<I, E>(cfg: &Cfg<I, E>, first: BlockId, second: BlockId) -> bool {
    let membership = |block: BlockId| {
        cfg.regions().iter().flat_map(move |region| {
            iter::once(region.protected_blocks.contains(&block)).chain(region.handlers.iter().map(
                move |handler| {
                    handler
                        .body
                        .blocks()
                        .is_some_and(|blocks| blocks.contains(&block))
                },
            ))
        })
    };
    membership(first).eq(membership(second))
}

/// Removes `block` from `cfg`, and from every protected region and known
/// handler body that holds it.
fn remove<I, E>(cfg: &mut Cfg<I, E>, block: BlockId) {
    let regions: Vec<RegionId> = cfg.regions().iter().map(|region| region.id).collect();
    for id in regions {
        let Some(region) = cfg.region_mut(id) else {
            continue;
        };
        region.protected_blocks.remove(&block);
        for handler in &mut region.handlers {
            if let HandlerBody::Known(blocks) = &mut handler.body {
                blocks.remove(&block);
            }
        }
    }
    cfg.remove_block(block);
}

/// Returns the successor of an empty block that forwards control to it
/// along its only, non-exceptional edge.
fn forwarded<D: VerifyDialect>(
    cfg: &Cfg<Instruction<D>, D::Edge>,
    block: BlockId,
) -> Option<BlockId> {
    let edge = cfg.edge(sole_outgoing(cfg, block)?);
    (!edge.kind().is_exceptional() && edge.target() != block).then(|| edge.target())
}

/// Returns the block that `block` can take: the target of its only,
/// non-exceptional edge, when no other edge enters it.
///
/// A block with a throwing instruction takes no block with an exceptional
/// edge, so that each exceptional edge keeps one throw site.
fn joined<D: VerifyDialect>(cfg: &Cfg<Instruction<D>, D::Edge>, block: BlockId) -> Option<BlockId> {
    let edge = cfg.edge(sole_outgoing(cfg, block)?);
    let target = edge.target();
    if edge.kind().is_exceptional()
        || target == block
        || target == cfg.entry()
        || cfg.incoming(target).count() != 1
    {
        return None;
    }
    let unwinds = cfg
        .outgoing(target)
        .any(|edge| cfg.edge(edge).kind().is_exceptional());
    let throws = cfg
        .block(block)
        .instructions()
        .iter()
        .any(Instruction::may_throw);
    (!(unwinds && throws)).then_some(target)
}

/// The only outgoing edge of `block`, when it has exactly one.
fn sole_outgoing<I, E>(cfg: &Cfg<I, E>, block: BlockId) -> Option<EdgeId> {
    let mut outgoing = cfg.outgoing(block);
    let first = outgoing.next()?;
    outgoing.next().is_none().then_some(first)
}
