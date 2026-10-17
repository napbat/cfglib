//! Exception-region structuring for the control-tree structurer.

extern crate alloc;
use alloc::boxed::Box;
use alloc::collections::{BTreeMap, BTreeSet};
use alloc::rc::Rc;
use alloc::vec::Vec;

use super::super::node::{CatchHandler, ControlNode};
use super::frames::{Bound, Frame, Next, Returned};
use super::walk::{Lifted, RegionWalk};
use super::{Structurer, block_is_allowed, flow_successors};
use crate::block::BlockId;
use crate::cfg::Cfg;
use crate::graph::traverse::{TraversalDirection, reverse_postorder};
use crate::region::{Handler, HandlerKind, HandlerRef, Region};

/// The blocks of `inner` also admitted by the enclosing bound, so a nested
/// region never escapes the extent it was entered under.
fn intersect_bounds(
    outer: Option<&BTreeSet<BlockId>>,
    inner: &BTreeSet<BlockId>,
) -> BTreeSet<BlockId> {
    match outer {
        None => inner.clone(),
        Some(outer) => inner.intersection(outer).copied().collect(),
    }
}

/// Group regions by anchor — their first protected block in reverse
/// postorder, i.e. the region's entry in flow order rather than by block id.
pub(super) fn region_anchors<'c, I, E>(
    cfg: &'c Cfg<I, E>,
    order: &[BlockId],
) -> BTreeMap<BlockId, Vec<&'c Region>> {
    let mut position: BTreeMap<BlockId, usize> = BTreeMap::new();
    for (index, &block) in order.iter().enumerate() {
        position.insert(block, index);
    }
    let mut anchors: BTreeMap<BlockId, Vec<&Region>> = BTreeMap::new();
    for region in cfg.regions() {
        let anchor = region
            .protected_blocks
            .iter()
            .min_by_key(|block| position.get(block).copied().unwrap_or(usize::MAX))
            .copied();
        if let Some(anchor) = anchor {
            anchors.entry(anchor).or_default().push(region);
        }
    }
    for candidates in anchors.values_mut() {
        candidates.sort_by_key(|region| core::cmp::Reverse(region.protected_blocks.len()));
    }
    anchors
}

/// The blocks the handlers of every region own, in the order the sweep
/// reaches them: each handler's entry, its filter funclet, then the blocks of
/// its known extent that carry code, in flow order.
///
/// A handler body is source code (a catch clause, a finally block, a filter
/// predicate) even when no exception edge enters the pad, and a funclet is
/// invoked by the runtime, so these blocks carry code without any incoming
/// edge from the entry. Protected blocks are not handler code: one the entry
/// cannot reach is dead code.
pub(super) fn handler_roots<I, E>(
    cfg: &Cfg<I, E>,
    carries_code: impl Fn(BlockId) -> bool,
) -> Vec<BlockId> {
    let mut listed: BTreeSet<BlockId> = BTreeSet::new();
    let mut roots = Vec::new();
    for region in cfg.regions() {
        for handler in &region.handlers {
            for block in handler_blocks_in_order(cfg, handler, &carries_code) {
                if listed.insert(block) {
                    roots.push(block);
                }
            }
        }
    }
    roots
}

/// One handler's blocks: its entry, its filter funclet, then the blocks of
/// its known extent that carry code: those the entry reaches in reverse
/// postorder, then the rest by block.
///
/// An extent block without code, such as the jump a source `leave` compiles
/// to, holds nothing to emit; walking from it would only reach the code that
/// follows the handler.
fn handler_blocks_in_order<I, E>(
    cfg: &Cfg<I, E>,
    handler: &Handler,
    carries_code: &impl Fn(BlockId) -> bool,
) -> Vec<BlockId> {
    let mut listed: BTreeSet<BlockId> = BTreeSet::new();
    let mut blocks = alloc::vec![handler.entry];
    listed.insert(handler.entry);
    if let HandlerKind::Filter { filter_block } = handler.kind
        && listed.insert(filter_block)
    {
        blocks.push(filter_block);
    }
    if let Some(extent) = handler.body.blocks() {
        let flow = reverse_postorder(cfg, handler.entry, TraversalDirection::Outgoing);
        for block in flow.into_iter().chain(extent.iter().copied()) {
            if extent.contains(&block) && carries_code(block) && listed.insert(block) {
                blocks.push(block);
            }
        }
    }
    blocks
}

/// The complete handler extents of `region`, or `None` when any handler's
/// extent is unknown and the region must stay unstructured.
///
/// A [`HandlerBody::Known`](crate::region::HandlerBody::Known) that omits
/// its own entry is a frontend bug, distinct from a deliberate `Unknown`:
/// it debug-panics, and degrades to unstructured flow in release builds.
fn complete_handler_bodies(region: &Region) -> Option<Vec<&BTreeSet<BlockId>>> {
    region
        .handlers
        .iter()
        .map(|handler| match handler.body.blocks() {
            Some(blocks) if blocks.contains(&handler.entry) => Some(blocks),
            Some(_) => {
                debug_assert!(
                    false,
                    "HandlerBody::Known must contain its own handler entry"
                );
                None
            }
            None => None,
        })
        .collect()
}

/// The region's normal continuation: the unique target of ordinary flow
/// leaving the protected blocks. When those have no exit or several, the
/// unique target of ordinary flow leaving the region's own blocks, protected
/// and handler alike. `None` when the exits diverge (or none exist), leaving
/// the walk to its conservative boundary handling.
///
/// A handler that leaves elsewhere (a `catch` that breaks or returns through
/// another path) does not change where the protected code resumes.
fn region_continuation<I, E>(cfg: &Cfg<I, E>, region: &Region) -> Option<BlockId> {
    let mut inside: BTreeSet<BlockId> = region.protected_blocks.clone();
    for handler in &region.handlers {
        inside.insert(handler.entry);
        if let Some(blocks) = handler.body.blocks() {
            inside.extend(blocks.iter().copied());
        }
    }
    let exits_of = |blocks: &mut dyn Iterator<Item = BlockId>| -> BTreeSet<BlockId> {
        blocks
            .flat_map(|block| flow_successors(cfg, block))
            .filter(|target| !inside.contains(target))
            .collect()
    };
    let protected_exits = exits_of(&mut region.protected_blocks.iter().copied());
    the_only(&protected_exits).or_else(|| the_only(&exits_of(&mut inside.iter().copied())))
}

/// The one element of `blocks`, if it has exactly one.
fn the_only(blocks: &BTreeSet<BlockId>) -> Option<BlockId> {
    let mut blocks = blocks.iter();
    match (blocks.next(), blocks.next()) {
        (Some(&only), None) => Some(only),
        _ => None,
    }
}

/// A try region being lifted: the frame that lifts its body, then hands the
/// handlers to a [`Handlers`] frame.
pub(super) struct TryBuild<'cfg> {
    region: &'cfg Region,
    handler_bodies: Vec<&'cfg BTreeSet<BlockId>>,
    /// The region's anchor.
    block: BlockId,
    allowed: Bound,
    /// Where ordinary flow reconverges after the region: the walks inside
    /// stop there silently, since the region's `leave` is implicit.
    continuation: Option<BlockId>,
    /// The protected blocks within the enclosing bound.
    try_bound: Rc<BTreeSet<BlockId>>,
    try_nodes: Vec<ControlNode>,
    awaiting: TryWait,
}

/// What the try waits for while a nested frame runs.
enum TryWait {
    Nothing,
    /// A try nested at the same anchor.
    Inner,
    /// The construct at the anchor: a branch, dispatch, or loop header.
    Anchor,
    /// The walk after the anchor.
    Walk,
    /// The walk into a protected successor of the anchor; the rest follow.
    Successor(alloc::vec::IntoIter<BlockId>),
    Handlers,
}

/// The arms of a try's handlers.
pub(super) struct HandlerArms {
    handlers: Vec<CatchHandler>,
    finally_nodes: Vec<ControlNode>,
    /// Whether every handler's own flow ends in a `break`, `continue`, or
    /// return.
    all_transfer: bool,
}

/// The handler arms of a region being lifted: the frame of its handlers, in
/// declaration order.
pub(super) struct Handlers<'cfg> {
    region: &'cfg Region,
    handler_bodies: Vec<&'cfg BTreeSet<BlockId>>,
    allowed: Bound,
    continuation: Option<BlockId>,
    /// The handler being lifted.
    index: usize,
    /// The blocks the handler in flight may emit.
    bound: Rc<BTreeSet<BlockId>>,
    /// The statements of the handler in flight.
    nodes: Vec<ControlNode>,
    /// The blocks of the handler in flight's extent still to lift.
    extent: alloc::vec::IntoIter<BlockId>,
    /// The catch arms sealed so far.
    catches: Vec<CatchHandler>,
    finally_nodes: Vec<ControlNode>,
    all_transfer: bool,
    awaiting: HandlerWait,
}

enum HandlerWait {
    /// No handler is in flight.
    Nothing,
    /// The walk from the handler's entry.
    Entry,
    /// The walk from a block of the handler's extent.
    Extent,
}

impl<'cfg, I, E> Structurer<'cfg, I, E> {
    /// Whether one `try` can hold the live protected code of `region`, which
    /// starts at `anchor`.
    ///
    /// A `try` is one construct, entered at one point and nested in what
    /// encloses it. The region does not fit when live code outside it jumps
    /// into the middle (a protected range split in two, a pad that retries
    /// halfway), when part of it was already emitted or claimed (a loop's
    /// test block), or when part of it lies beyond the enclosing bound. The
    /// tree would then show that part outside the try, raising into a pad the
    /// try does not enclose, so the region stays ordinary flow. Dead code
    /// enters nothing, a block without instructions needs no covering, and
    /// exception edges are not ordinary flow.
    fn fits_one_try(
        &self,
        region: &Region,
        anchor: BlockId,
        allowed_blocks: Option<&BTreeSet<BlockId>>,
    ) -> bool {
        let cfg = self.cfg;
        region
            .protected_blocks
            .iter()
            .filter(|&&block| block != anchor && self.reachable[block.index()])
            .all(|&block| {
                let entered_inside = cfg.incoming(block).map(|edge| cfg.edge(edge)).all(|edge| {
                    edge.kind().is_exceptional()
                        || !self.reachable[edge.source().index()]
                        || region.protected_blocks.contains(&edge.source())
                });
                let coverable = self.instruction_count(block) == 0
                    || (!self.is_visited(block) && block_is_allowed(allowed_blocks, block));
                entered_inside && coverable
            })
    }

    /// Pick the try region anchored at `block`, if a structurable one exists,
    /// and return the frame that lifts it.
    ///
    /// Among the regions anchored at `block` (outermost first), the first with
    /// complete handler extents and a single entry is structured; the others
    /// remain ordinary control flow. Bounds intersect with the enclosing
    /// `allowed`, so a nested region never escapes the extent it was entered
    /// under.
    pub(super) fn select_try(&mut self, block: BlockId, allowed: &Bound) -> Option<Frame<'cfg>> {
        let cfg = self.cfg;
        let candidates = self.anchors.get(&block)?;
        let (region, handler_bodies) = candidates
            .iter()
            .filter(|region| !self.structured_regions.contains(&region.id.0))
            .filter(|region| self.fits_one_try(region, block, allowed.as_deref()))
            .find_map(|&region| complete_handler_bodies(region).map(|bodies| (region, bodies)))?;
        self.structured_regions.insert(region.id.0);

        // The region's normal continuation: the unique block ordinary flow
        // reconverges at, or the anchor's in-bound post-dominator when the
        // exits diverge. Inner walks stop there silently — the region's
        // `leave` is implicit, not a goto — and the enclosing walk resumes
        // there after the structured node.
        let continuation = region_continuation(cfg, region).or_else(|| {
            self.effective_merge(block, allowed.as_deref())
                .filter(|merge| {
                    !region.protected_blocks.contains(merge)
                        && region.handlers.iter().all(|handler| {
                            handler.entry != *merge
                                && handler
                                    .body
                                    .blocks()
                                    .is_none_or(|blocks| !blocks.contains(merge))
                        })
                })
        });

        let try_bound = Rc::new(intersect_bounds(
            allowed.as_deref(),
            &region.protected_blocks,
        ));
        Some(Frame::Try(Box::new(TryBuild {
            region,
            handler_bodies,
            block,
            allowed: allowed.clone(),
            continuation,
            try_bound,
            try_nodes: Vec::new(),
            awaiting: TryWait::Nothing,
        })))
    }

    pub(super) fn step_try(
        &mut self,
        build: &mut TryBuild<'cfg>,
        returned: Returned,
    ) -> Next<'cfg> {
        match core::mem::replace(&mut build.awaiting, TryWait::Nothing) {
            TryWait::Nothing => self.start_try_body(build),
            TryWait::Inner | TryWait::Anchor => {
                let (node, next) = returned.into_construct();
                build.try_nodes.push(node);
                self.continue_try_body(build, next)
            }
            TryWait::Walk => {
                build.try_nodes.extend(returned.into_nodes());
                self.lift_protected_successors(build)
            }
            TryWait::Successor(successors) => {
                build.try_nodes.extend(returned.into_nodes());
                self.next_protected_successor(build, successors)
            }
            TryWait::Handlers => self.finish_try(build, returned.into_handlers()),
        }
    }

    /// Lift the try body. A nested region can share this anchor (an inner try
    /// starting at the same instruction), so the region check runs again
    /// first — each pass structures exactly one more region. Otherwise the
    /// anchor goes through the shared classification path, so an anchor that
    /// is itself a branch, dispatch, or loop header keeps its structure.
    fn start_try_body(&mut self, build: &mut TryBuild<'cfg>) -> Next<'cfg> {
        let bound = Some(Rc::clone(&build.try_bound));
        if let Some(inner) = self.select_try(build.block, &bound) {
            build.awaiting = TryWait::Inner;
            return Next::Call(inner);
        }
        match self.lift_block(
            build.block,
            &bound,
            build.continuation,
            &mut build.try_nodes,
        ) {
            Lifted::Next(next) => self.continue_try_body(build, next),
            Lifted::Construct(frame) => {
                build.awaiting = TryWait::Anchor;
                Next::Call(frame)
            }
        }
    }

    /// Go on with the try body at `next`, the block after its anchor.
    fn continue_try_body(
        &mut self,
        build: &mut TryBuild<'cfg>,
        next: Option<BlockId>,
    ) -> Next<'cfg> {
        let Some(next) = next else {
            return self.lift_protected_successors(build);
        };
        build.awaiting = TryWait::Walk;
        Next::Call(Frame::Region(RegionWalk::new(
            next,
            Some(Rc::clone(&build.try_bound)),
            build.continuation,
        )))
    }

    /// Protected successors the classified walk did not reach (for example
    /// the targets of a multi-successor linear anchor) still belong to the
    /// try body.
    fn lift_protected_successors(&mut self, build: &mut TryBuild<'cfg>) -> Next<'cfg> {
        let successors: Vec<BlockId> = self.cfg.successors(build.block).collect();
        self.next_protected_successor(build, successors.into_iter())
    }

    /// Lift the next of `successors` that is protected and not yet emitted,
    /// or go on to the handlers when none is left.
    fn next_protected_successor(
        &mut self,
        build: &mut TryBuild<'cfg>,
        mut successors: alloc::vec::IntoIter<BlockId>,
    ) -> Next<'cfg> {
        while let Some(succ) = successors.next() {
            if build.region.protected_blocks.contains(&succ) && !self.is_visited(succ) {
                build.awaiting = TryWait::Successor(successors);
                return Next::Call(Frame::Region(RegionWalk::new(
                    succ,
                    Some(Rc::clone(&build.try_bound)),
                    build.continuation,
                )));
            }
        }
        build.awaiting = TryWait::Handlers;
        Next::Call(Frame::Handlers(Box::new(Handlers {
            region: build.region,
            handler_bodies: core::mem::take(&mut build.handler_bodies),
            allowed: build.allowed.clone(),
            continuation: build.continuation,
            index: 0,
            bound: Rc::new(BTreeSet::new()),
            nodes: Vec::new(),
            extent: Vec::new().into_iter(),
            catches: Vec::new(),
            finally_nodes: Vec::new(),
            all_transfer: true,
            awaiting: HandlerWait::Nothing,
        })))
    }

    /// Build the `try` node from its body and the arms of its handlers.
    fn finish_try(&mut self, build: &mut TryBuild<'cfg>, arms: HandlerArms) -> Next<'cfg> {
        // A region whose body and handlers all end in a `break`, `continue`,
        // or return has no end to resume at: the walk has nothing to continue
        // with when the continuation was already emitted.
        let body_transfers = build
            .try_nodes
            .last()
            .is_some_and(|node| self.never_completes(*node));
        let resume = build
            .continuation
            .filter(|&next| !(self.is_visited(next) && body_transfers && arms.all_transfer));
        let try_nodes = core::mem::take(&mut build.try_nodes);
        let node = ControlNode::TryCatch {
            try_body: self.tables.seal(try_nodes),
            handlers: self.tables.push_handlers(arms.handlers),
            finally_body: self.tables.seal(arms.finally_nodes),
        };
        // Resume at the region's own continuation, not the anchor's merge:
        // they differ whenever the try exits through an explicit leave.
        Next::Return(Returned::Construct { node, next: resume })
    }

    pub(super) fn step_handlers(
        &mut self,
        state: &mut Handlers<'cfg>,
        returned: Returned,
    ) -> Next<'cfg> {
        match core::mem::replace(&mut state.awaiting, HandlerWait::Nothing) {
            HandlerWait::Nothing => self.start_handler(state),
            HandlerWait::Entry => {
                state.nodes = returned.into_nodes();
                let transfers = state
                    .nodes
                    .last()
                    .is_some_and(|node| self.never_completes(*node));
                state.all_transfer &= transfers;
                // The walk ends where the pad's flow splits without a branch
                // (a suspension with an exit and a resume edge). The rest of
                // the extent that the entry cannot reach follows in flow
                // order; a reachable block waits for the sweep that always
                // emitted it, and a pad of a nested region waits for that
                // region.
                let handler = &state.region.handlers[state.index];
                state.extent = handler_blocks_in_order(self.cfg, handler, &|block| {
                    self.instruction_count(block) > 0
                })
                .into_iter();
                self.lift_handler_extent(state)
            }
            HandlerWait::Extent => {
                state.nodes.extend(returned.into_nodes());
                self.lift_handler_extent(state)
            }
        }
    }

    /// Lift the handler at `state.index` from its entry, or end the arms when
    /// every handler is done.
    fn start_handler(&mut self, state: &mut Handlers<'cfg>) -> Next<'cfg> {
        let Some(handler) = state.region.handlers.get(state.index) else {
            return Next::Return(Returned::Handlers(HandlerArms {
                handlers: core::mem::take(&mut state.catches),
                finally_nodes: core::mem::take(&mut state.finally_nodes),
                all_transfer: state.all_transfer,
            }));
        };
        // A synthetic landing entry belongs to its handler even when the
        // enclosing bound does not name it. So do the extent blocks the entry
        // cannot reach: they are the handler's own dead code, and only a
        // block that feeds an enclosing loop is in that loop's bound.
        let handler_blocks = state.handler_bodies[state.index];
        let mut bound = intersect_bounds(state.allowed.as_deref(), handler_blocks);
        bound.insert(handler.entry);
        bound.extend(
            handler_blocks
                .iter()
                .copied()
                .filter(|block| !self.reachable[block.index()]),
        );
        state.bound = Rc::new(bound);
        state.awaiting = HandlerWait::Entry;
        Next::Call(Frame::Region(RegionWalk::new(
            handler.entry,
            Some(Rc::clone(&state.bound)),
            state.continuation,
        )))
    }

    /// Lift the next block of the handler's extent that no walk reached, or
    /// seal the handler and go on to the next one.
    fn lift_handler_extent(&mut self, state: &mut Handlers<'cfg>) -> Next<'cfg> {
        let region = state.region;
        let handler_blocks = state.handler_bodies[state.index];
        let reached = state.extent.find(|&block| {
            handler_blocks.contains(&block)
                && !self.reachable[block.index()]
                && !self.is_visited(block)
                && !self.awaits_its_handler(region, block)
        });
        if let Some(block) = reached {
            state.awaiting = HandlerWait::Extent;
            return Next::Call(Frame::Region(RegionWalk::new(
                block,
                Some(Rc::clone(&state.bound)),
                state.continuation,
            )));
        }
        let handler = &region.handlers[state.index];
        let nodes = core::mem::take(&mut state.nodes);
        if handler.kind == HandlerKind::Finally {
            // Multiple finally handlers concatenate in declaration order.
            state.finally_nodes.extend(nodes);
        } else {
            state.catches.push(CatchHandler {
                handler: HandlerRef::new(region.id, state.index),
                entry: handler.entry,
                kind: handler.kind,
                body: self.tables.seal(nodes),
            });
        }
        state.index += 1;
        self.start_handler(state)
    }

    /// Whether `block` is a pad, or in the extent, of a handler of a region
    /// other than `region` that the walk has not structured yet. That region
    /// lifts the block when it reaches the region's anchor.
    fn awaits_its_handler(&self, region: &Region, block: BlockId) -> bool {
        self.handler_claims.get(&block).is_some_and(|regions| {
            regions
                .iter()
                .any(|&other| other != region.id.0 && !self.structured_regions.contains(&other))
        })
    }
}
