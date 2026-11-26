//! Irreducible-to-reducible transformation through dispatch blocks.
//!
//! [`make_reducible_with_dispatch`] makes every irreducible cycle single-entry
//! without copying a block, as LLVM's `FixIrreducible` does over `CycleInfo`,
//! and as the predicate variables of Bahmann et al. (TACO 2015) do for a whole
//! graph. Every edge into an entry of the cycle goes to a selector block that
//! records which entry it was for, and on to one new hub block that branches
//! on that record. The hub is then the only way into the cycle, which becomes
//! a natural loop with the hub as its header. cfglib writes no instruction of
//! its own: a [`DispatchBuilder`] supplies the code of the selectors and the
//! hub and the payloads of the new edges.
//!
//! The cycles are those of [`LoopForest::compute_cycles`], fixed innermost
//! first. Fixing a cycle moves the entries of the cycles around it, so the
//! entries of a cycle are found when its turn comes. Once a cycle is fixed, its
//! hub is the only block of it that control can still enter from outside, and
//! it stands for every entry the cycle shared with the one around it. So a
//! block is looked at as an entry once: a block of the graph for the innermost
//! cycle it enters, and a hub for the cycle directly around the one it
//! dispatches.

extern crate alloc;
use alloc::vec;
use alloc::vec::Vec;
use core::cmp::Reverse;

use crate::block::BlockId;
use crate::cfg::Cfg;
use crate::edge::{Edge, EdgeId, EdgeKind};
use crate::graph::loop_forest::{LoopForest, LoopId};
use crate::graph::traverse::reachable;
use crate::graph::view::Direction;

/// The code [`make_reducible_with_dispatch`] puts in the blocks it adds, and
/// the payloads of the edges it adds.
///
/// cfglib does not invent instructions, so the consumer writes the selector
/// and dispatch code of its own IR here. Each hub branches on a value of its
/// own, its selector, which the selector blocks of that hub set: a selector
/// block for entry `index` sets it to `index`, and the hub's edge for `index`
/// goes to that entry. The value must be the hub's own. A cycle inside another
/// one is fixed first, and when the outer cycle is fixed an edge to the inner
/// hub gets a selector of the outer one, so a path runs through the inner
/// selector, then the outer selector and hub, then the inner hub, which must
/// still find the value its own selector set.
///
/// The calls come cycle by cycle, innermost first. For one cycle,
/// [`dispatch`](Self::dispatch) comes first, then [`case`](Self::case) for
/// each entry in index order, then [`select`](Self::select) and
/// [`jump`](Self::jump) for each redirected edge: the edges into the first
/// entry first, each entry's in the order of its incoming edges.
pub trait DispatchBuilder<I, E> {
    /// The instructions of a selector block that routes control to entry
    /// `index` of the cycle `hub` dispatches, which is `entry`: they set the
    /// selector of `hub` to `index` and end in the transfer to the hub. A new
    /// [`EdgeKind::Jump`] edge leads from the selector to the hub.
    fn select(&mut self, hub: BlockId, index: usize, entry: BlockId) -> Vec<I>;

    /// The instructions of `hub`, whose last one branches on the selector to
    /// the entry it names in `entries`: the header of the cycle first, then
    /// the others in ascending order.
    fn dispatch(&mut self, hub: BlockId, entries: &[BlockId]) -> Vec<I>;

    /// The payload of the [`EdgeKind::SwitchCase`] edge from `hub` to entry
    /// `index`.
    fn case(&mut self, hub: BlockId, index: usize) -> E;

    /// The payload of the [`EdgeKind::Jump`] edge from `selector` to `hub`.
    fn jump(&mut self, selector: BlockId, hub: BlockId) -> E;
}

/// The blocks [`make_reducible_with_dispatch`] added, in the order it added
/// them.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct DispatchReport {
    /// One hub for each irreducible cycle, innermost cycles first.
    pub hubs: Vec<BlockId>,
    /// One selector for each edge redirected to a hub.
    pub selectors: Vec<BlockId>,
}

/// Make `cfg` reducible without copying a block, by entering every
/// irreducible cycle through a new hub block that dispatches to its entries.
///
/// The cycles are those of [`LoopForest::compute_cycles`], fixed innermost
/// first. For a cycle that can be entered at more than one block, one hub is
/// added. Its entries are numbered: its header first, then the others in
/// ascending order. Every edge into an entry, from outside the cycle or from
/// inside it, now goes to a new selector block, which goes on to the hub along
/// a new [`EdgeKind::Jump`] edge. The hub has one [`EdgeKind::SwitchCase`]
/// edge to each entry, in index order. The redirected edge keeps its identity,
/// kind, weight and payload. An edge into an entry from inside a cycle that
/// the entry itself heads within this one is left alone, so that inner loop
/// stays a natural loop of its own, and so is an edge from code the root
/// does not reach.
///
/// The hub dominates the cycle afterwards and heads it as a natural loop. The
/// result is reducible, and every path of the old graph is a path of the new
/// one once the hubs follow the selectors. Each irreducible cycle adds one hub
/// and one selector per redirected edge, so the growth is linear in the edges
/// for each level of irreducible nesting, where node splitting
/// ([`make_reducible`](super::make_reducible)) can grow exponentially.
///
/// cfglib writes no instruction: `builder` supplies the code of every added
/// block and the payloads of the added edges, in the order
/// [`DispatchBuilder`] documents. A reducible CFG is left unchanged and gets
/// an empty report.
///
/// # Examples
///
/// ```
/// use cfglib::{
///     BlockId, Cfg, DispatchBuilder, DominatorTree, EdgeKind, is_reducible,
///     make_reducible_with_dispatch,
/// };
///
/// /// A toy instruction set: set the selector of a hub, or branch on it.
/// #[derive(Debug, Clone, PartialEq)]
/// enum Inst {
///     SetSelector { hub: BlockId, index: usize },
///     Switch { hub: BlockId },
/// }
///
/// struct Selectors;
///
/// impl DispatchBuilder<Inst, ()> for Selectors {
///     fn select(&mut self, hub: BlockId, index: usize, _entry: BlockId) -> Vec<Inst> {
///         vec![Inst::SetSelector { hub, index }]
///     }
///     fn dispatch(&mut self, hub: BlockId, _entries: &[BlockId]) -> Vec<Inst> {
///         vec![Inst::Switch { hub }]
///     }
///     fn case(&mut self, _hub: BlockId, _index: usize) {}
///     fn jump(&mut self, _selector: BlockId, _hub: BlockId) {}
/// }
///
/// // entry branches into `a` and `b`, which jump to each other.
/// let mut cfg = Cfg::<Inst>::new();
/// let a = cfg.new_block();
/// let b = cfg.new_block();
/// cfg.add_edge(cfg.entry(), a, EdgeKind::ConditionalTrue);
/// cfg.add_edge(cfg.entry(), b, EdgeKind::ConditionalFalse);
/// cfg.add_edge(a, b, EdgeKind::Jump);
/// cfg.add_edge(b, a, EdgeKind::Jump);
///
/// let report = make_reducible_with_dispatch(&mut cfg, &mut Selectors);
///
/// assert!(is_reducible(&cfg, &DominatorTree::compute(&cfg)));
/// let &[hub] = report.hubs.as_slice() else { panic!("one irreducible cycle") };
/// assert_eq!(cfg.block(hub).instructions(), [Inst::Switch { hub }]);
/// assert_eq!(cfg.successors(hub).collect::<Vec<_>>(), [a, b]);
/// // Two edges into each of the two entries.
/// assert_eq!(report.selectors.len(), 4);
/// ```
pub fn make_reducible_with_dispatch<I, E, B: DispatchBuilder<I, E>>(
    cfg: &mut Cfg<I, E>,
    builder: &mut B,
) -> DispatchReport {
    let forest = LoopForest::compute_cycles(&*cfg);
    let mut report = DispatchReport::default();
    let mut irreducible: Vec<LoopId> = forest
        .loops()
        .filter(|&lp| forest.is_irreducible(lp))
        .collect();
    if irreducible.is_empty() {
        return report;
    }
    // A cycle is deeper than every cycle around it.
    irreducible.sort_by_key(|&lp| Reverse(forest.depth(lp)));
    let mut dispatch = Dispatch::new(cfg, &forest);
    for cycle in irreducible {
        dispatch.fix(cfg, builder, cycle, &mut report);
    }
    report
}

/// What the transformation knows about every block, the blocks it adds
/// included.
struct Dispatch<'a> {
    forest: &'a LoopForest,
    /// The innermost cycle of the forest that holds each block.
    owner: Vec<Option<LoopId>>,
    /// The cycle each block heads: its own for a header of the forest, the
    /// one it dispatches for a hub.
    heads: Vec<Option<LoopId>>,
    /// Whether the root reaches each block.
    reached: Vec<bool>,
    /// The blocks to look at as entries of each cycle, besides its header.
    candidates: Vec<Vec<BlockId>>,
}

impl<'a> Dispatch<'a> {
    fn new<I, E>(cfg: &Cfg<I, E>, forest: &'a LoopForest) -> Self {
        let bound = cfg.block_bound();
        let reached = reachable(cfg, [cfg.entry()], Direction::Forward);
        let mut heads = vec![None; bound];
        for lp in forest.loops() {
            heads[forest.header(lp).index()] = Some(lp);
        }
        let mut dispatch = Self {
            forest,
            owner: (0..bound)
                .map(|index| forest.innermost(BlockId::from_index(index)))
                .collect(),
            heads,
            reached,
            candidates: vec![Vec::new(); forest.len()],
        };
        // A block enters every cycle from the innermost one around it that it
        // does not head, up to some outermost one, and every cycle it enters
        // is irreducible. Fixing the innermost sends the block's edges through
        // that cycle's hub, which takes the block's place in the others.
        for block in cfg.block_ids() {
            let first = match dispatch.heads[block.index()] {
                Some(own) => forest.parent(own),
                None => dispatch.owner[block.index()],
            };
            if let Some(first) = first
                && dispatch.reached[block.index()]
                && dispatch.entered_from_outside(cfg, first, block)
            {
                dispatch.candidates[first.index()].push(block);
            }
        }
        dispatch
    }

    /// Whether `block` is in `cycle`.
    fn holds(&self, cycle: LoopId, block: BlockId) -> bool {
        self.owner[block.index()].is_some_and(|owner| self.forest.encloses(cycle, owner))
    }

    /// Whether an edge from a block the root reaches enters `block` from
    /// outside `cycle`.
    fn entered_from_outside<I, E>(&self, cfg: &Cfg<I, E>, cycle: LoopId, block: BlockId) -> bool {
        cfg.incoming(block).any(|edge| {
            let from = cfg.edge(edge).source();
            self.reached[from.index()] && !self.holds(cycle, from)
        })
    }

    /// The innermost cycle around `cycle` that holds `block`, which `cycle`
    /// itself does not.
    fn around(&self, cycle: LoopId, block: BlockId) -> Option<LoopId> {
        let owner = self.owner[block.index()]?;
        let mut around = self.forest.parent(cycle);
        while let Some(lp) = around {
            if self.forest.encloses(lp, owner) {
                break;
            }
            around = self.forest.parent(lp);
        }
        around
    }

    fn add_block<I, E>(
        &mut self,
        cfg: &mut Cfg<I, E>,
        owner: Option<LoopId>,
        heads: Option<LoopId>,
    ) -> BlockId {
        let block = cfg.new_block();
        debug_assert_eq!(
            block.index(),
            self.owner.len(),
            "a new block takes the next slot"
        );
        self.owner.push(owner);
        self.heads.push(heads);
        self.reached.push(true);
        block
    }

    /// Enter `cycle` through a new hub, if it still has more than one entry.
    fn fix<I, E, B: DispatchBuilder<I, E>>(
        &mut self,
        cfg: &mut Cfg<I, E>,
        builder: &mut B,
        cycle: LoopId,
        report: &mut DispatchReport,
    ) {
        let header = self.forest.header(cycle);
        let mut entries = vec![header];
        let candidates = core::mem::take(&mut self.candidates[cycle.index()]);
        entries.extend(
            candidates
                .into_iter()
                .filter(|&block| block != header && self.entered_from_outside(cfg, cycle, block)),
        );
        entries[1..].sort_unstable();
        debug_assert!(
            entries.len() > 1,
            "fixing the cycles inside an irreducible cycle leaves it irreducible"
        );

        let hub = self.add_block(cfg, Some(cycle), Some(cycle));
        report.hubs.push(hub);
        let code = builder.dispatch(hub, &entries);
        cfg.block_mut(hub).instructions_mut().extend(code);

        // An entry that heads a cycle inside this one keeps the edges from
        // that cycle, which stays a loop of its own; the hub enters it.
        let mut redirected: Vec<(usize, EdgeId)> = Vec::new();
        for (index, &entry) in entries.iter().enumerate() {
            let own = self.heads[entry.index()].filter(|&inner| inner != cycle);
            redirected.extend(
                cfg.incoming(entry)
                    .filter(|&edge| {
                        let from = cfg.edge(edge).source();
                        self.reached[from.index()]
                            && !own.is_some_and(|inner| self.holds(inner, from))
                    })
                    .map(|edge| (index, edge)),
            );
        }
        for (index, &entry) in entries.iter().enumerate() {
            let payload = builder.case(hub, index);
            cfg.add_edge(
                hub,
                entry,
                Edge::new(EdgeKind::SwitchCase).with_payload(payload),
            );
        }
        for (index, edge) in redirected {
            let from = cfg.edge(edge).source();
            let owner = if self.holds(cycle, from) {
                Some(cycle)
            } else {
                self.around(cycle, from)
            };
            let selector = self.add_block(cfg, owner, None);
            report.selectors.push(selector);
            let code = builder.select(hub, index, entries[index]);
            cfg.block_mut(selector).instructions_mut().extend(code);
            cfg.redirect_edge_target(edge, selector);
            let payload = builder.jump(selector, hub);
            cfg.add_edge(
                selector,
                hub,
                Edge::new(EdgeKind::Jump).with_payload(payload),
            );
        }

        if let Some(around) = self.forest.parent(cycle)
            && self.forest.is_irreducible(around)
        {
            self.candidates[around.index()].push(hub);
        }
    }
}

#[cfg(test)]
mod tests;
