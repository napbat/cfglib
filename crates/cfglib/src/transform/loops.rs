//! Loop preheaders: give every loop one block that enters its header from
//! outside.
//!
//! A preheader is a block outside a loop whose only successor is the loop's
//! header and which is the header's only predecessor outside the loop. Code
//! hoisted out of the loop goes there, which is why loop-invariant code
//! motion, unrolling and similar passes ask for one. [`insert_preheader`] puts
//! one in front of a loop of a [`LoopForest`]; [`canonicalize_loops`] does it
//! for every loop and reports the exits as well.

extern crate alloc;
use alloc::vec;
use alloc::vec::Vec;
use smallvec::SmallVec;

use crate::block::BlockId;
use crate::cfg::Cfg;
use crate::edge::{EdgeId, EdgeKind};
use crate::graph::dominator::DominatorTree;
use crate::graph::loop_forest::{LoopForest, LoopId};

/// A loop after [`canonicalize_loops`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CanonicalLoop {
    /// The loop header.
    pub header: BlockId,
    /// The loop's preheader: the one block outside the loop that enters the
    /// header, whose only successor is the header.
    ///
    /// `None` when the header has no predecessor outside the loop, which is
    /// the case for a loop headed by the entry block: the entry cannot gain
    /// a predecessor, so no preheader can be put before it.
    pub preheader: Option<BlockId>,
    /// The exits, ascending: the blocks outside the loop that are targets of
    /// edges from inside it, read once every preheader is in.
    pub exits: Vec<BlockId>,
}

/// Insert a dedicated **preheader** block for loop `lp` of `loops`.
///
/// A preheader is a single-successor block that becomes the sole
/// predecessor of the loop header from outside the loop. This simplifies
/// many loop transformations (LICM, unrolling, etc.).
///
/// The header's edges from outside the loop are redirected to the preheader
/// in place: each keeps its identity, kind, weight and payload, and its
/// position among its source's outgoing edges, so a branch keeps its arm
/// order. The preheader reaches the header by a new fallthrough edge.
///
/// Returns the new preheader, or `None` if the header is already canonical:
/// it has no predecessor outside the loop, or exactly one whose only successor
/// is the header.
///
/// `loops` must be the loops of `cfg`. The new block is in none of them, so
/// compute the loops again before asking about it.
///
/// # Panics
///
/// Panics when `lp` is not a loop of `loops`.
///
/// # Examples
///
/// ```
/// use cfglib::{Cfg, DominatorTree, EdgeKind, LoopForest, insert_preheader};
///
/// // Two blocks enter the loop `header <-> body`.
/// let mut cfg = Cfg::<u32>::new();
/// let [left, right, header, body] = [(); 4].map(|()| cfg.new_block());
/// cfg.add_edge(cfg.entry(), left, EdgeKind::ConditionalTrue);
/// cfg.add_edge(cfg.entry(), right, EdgeKind::ConditionalFalse);
/// cfg.add_edge(left, header, EdgeKind::Jump);
/// cfg.add_edge(right, header, EdgeKind::Jump);
/// cfg.add_edge(header, body, EdgeKind::Fallthrough);
/// cfg.add_edge(body, header, EdgeKind::Back);
///
/// let loops = LoopForest::compute(&cfg, &DominatorTree::compute(&cfg));
/// let lp = loops.loop_of_header(header).expect("header heads a loop");
/// let preheader = insert_preheader(&mut cfg, &loops, lp).expect("two entries");
/// assert_eq!(cfg.successors(preheader).collect::<Vec<_>>(), [header]);
/// assert_eq!(cfg.predecessors(header).collect::<Vec<_>>(), [body, preheader]);
/// ```
pub fn insert_preheader<I, E: Default>(
    cfg: &mut Cfg<I, E>,
    loops: &LoopForest,
    lp: LoopId,
) -> Option<BlockId> {
    insert_preheader_outside(cfg, loops.header(lp), |block| loops.contains(lp, block))
}

/// Put a preheader before `header` for the loop whose blocks `inside` names,
/// as [`insert_preheader`] does.
fn insert_preheader_outside<I, E: Default>(
    cfg: &mut Cfg<I, E>,
    header: BlockId,
    inside: impl Fn(BlockId) -> bool,
) -> Option<BlockId> {
    let from_outside: SmallVec<[EdgeId; 4]> = cfg
        .incoming(header)
        .filter(|&edge| !inside(cfg.edge(edge).source()))
        .collect();
    match from_outside.as_slice() {
        [] => return None,
        &[only] if cfg.outgoing(cfg.edge(only).source()).nth(1).is_none() => return None,
        _ => {}
    }

    let preheader = cfg.new_block();
    for edge in from_outside {
        cfg.redirect_edge_target(edge, preheader);
    }
    cfg.add_edge(preheader, header, EdgeKind::Fallthrough);
    Some(preheader)
}

/// Canonicalize all loops: insert preheaders and identify exits.
///
/// The loops are those of [`LoopForest::compute`], so explicit
/// [`EdgeKind::Back`] tags are honored. They come outermost first, and by
/// header within one depth, each with the preheader [`insert_preheader`] puts
/// before it. A preheader that sits on the paths through a loop nested in
/// another is part of the outer loop, and exits are read once every preheader
/// is in, so an edge that leaves one loop for the preheader of the next ends at
/// the preheader.
///
/// `dom` is the dominator tree of `cfg` as it was passed in, and the inserted
/// blocks are not in it: compute it again before relying on it.
pub fn canonicalize_loops<I, E: Default>(
    cfg: &mut Cfg<I, E>,
    dom: &DominatorTree,
) -> Vec<CanonicalLoop> {
    let loops = LoopForest::compute(cfg, dom);
    let mut order: Vec<LoopId> = loops.loops().collect();
    order.sort_by_key(|&lp| (loops.depth(lp), loops.header(lp)));
    // The preheaders that joined each loop: the forest holds the blocks the
    // loops had when it was computed.
    let mut joined: Vec<SmallVec<[BlockId; 2]>> = vec![SmallVec::new(); loops.loop_count()];
    let holds = |joined: &[SmallVec<[BlockId; 2]>], lp: LoopId, block: BlockId| {
        loops.contains(lp, block) || joined[lp.index()].contains(&block)
    };

    let mut preheaders = Vec::with_capacity(order.len());
    for &lp in &order {
        let header = loops.header(lp);
        let inserted = insert_preheader_outside(cfg, header, |block| holds(&joined, lp, block));
        if let Some(preheader) = inserted {
            // The new block is reached from the sources of the redirected
            // edges and reaches the header, so it lies on a cycle of every
            // other loop that holds both the header and one of those sources.
            let sources: SmallVec<[BlockId; 4]> = cfg.predecessors(preheader).collect();
            for other in loops.loops() {
                if other != lp
                    && holds(&joined, other, header)
                    && sources.iter().any(|&source| holds(&joined, other, source))
                {
                    joined[other.index()].push(preheader);
                }
            }
        }
        // Without an insertion the header has no outside predecessor, or one
        // that is already the preheader.
        preheaders.push(inserted.or_else(|| {
            cfg.predecessors(header)
                .find(|&block| !holds(&joined, lp, block))
        }));
    }

    order
        .into_iter()
        .zip(preheaders)
        .map(|(lp, preheader)| {
            let mut exits: Vec<BlockId> = loops
                .body(lp)
                .chain(joined[lp.index()].iter().copied())
                .flat_map(|block| cfg.successors(block))
                .filter(|&successor| !holds(&joined, lp, successor))
                .collect();
            exits.sort_unstable();
            exits.dedup();
            CanonicalLoop {
                header: loops.header(lp),
                preheader,
                exits,
            }
        })
        .collect()
}

#[cfg(test)]
mod tests;
