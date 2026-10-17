//! The walk along sequential flow: one region, block by block, and the
//! transfers that end it.

extern crate alloc;
use alloc::rc::Rc;
use alloc::vec::Vec;

use super::super::node::ControlNode;
use super::super::report::{GotoDiagnostic, GotoReason};
use super::bound::Bound;
use super::branches::Conditional;
use super::frames::{Frame, Next, Returned};
use super::{BlockFlowKind, Structurer, flow_successors};
use crate::block::BlockId;
use crate::edge::{EdgeId, EdgeKind};

/// The walk of one region: [`Structurer::lift_region`]'s frame.
///
/// The walk ends silently at `stop` (a legitimate structural convergence
/// point — an arm's merge, a post-tested loop's latch); every other transfer
/// that cannot proceed is resolved against the loop stack or made explicit as
/// a goto.
pub(super) struct RegionWalk {
    allowed: Bound,
    stop: Option<BlockId>,
    /// The block the walk enters next; `None` ends it.
    current: Option<BlockId>,
    result: Vec<ControlNode>,
    awaiting: Awaiting,
}

/// What the walk waits for while a nested frame runs.
enum Awaiting {
    Nothing,
    /// Out-of-bound code lifted inline, which ends the walk.
    Inline,
    /// The construct that the entered block opened.
    Construct,
}

/// How entering a block left the walk.
pub(super) enum Lifted<'cfg> {
    /// The block is emitted; flow goes on at this block, if any.
    Next(Option<BlockId>),
    /// The block opens a construct. Run its frame, emit the node it returns,
    /// and go on at the block it names.
    Construct(Frame<'cfg>),
}

impl RegionWalk {
    pub(super) const fn new(head: BlockId, allowed: Bound, stop: Option<BlockId>) -> Self {
        Self {
            allowed,
            stop,
            current: Some(head),
            result: Vec::new(),
            awaiting: Awaiting::Nothing,
        }
    }
}

impl<'cfg, I, E> Structurer<'cfg, I, E> {
    pub(super) fn step_region(&mut self, walk: &mut RegionWalk, returned: Returned) -> Next<'cfg> {
        match core::mem::replace(&mut walk.awaiting, Awaiting::Nothing) {
            Awaiting::Nothing => {}
            Awaiting::Inline => {
                walk.result.extend(returned.into_nodes());
                return Next::Return(Returned::Nodes(core::mem::take(&mut walk.result)));
            }
            Awaiting::Construct => {
                let (node, next) = returned.into_construct();
                walk.result.push(node);
                walk.current = next;
            }
        }
        let cfg = self.cfg;
        while let Some(block) = walk.current {
            if Some(block) == walk.stop {
                break;
            }
            if let Some(node) = self.resolve_loop_transfer(block) {
                walk.result.push(node);
                break;
            }
            if self.is_visited(block) {
                // Convergence is only silent at `stop`; any other transfer
                // into already-emitted code is a real control transfer.
                self.push_goto(&mut walk.result, block, GotoReason::RevisitedTarget);
                break;
            }
            if !walk.allowed.contains(block) {
                // Exclusively owned out-of-bound code — a single predecessor,
                // so this transfer is its only entry, and no exception region
                // claims it — inlines here instead of degrading to a goto
                // (javac hoists `break label` bodies out of line like this).
                // The inlined walk stays within unclaimed code, so it never
                // pulls another region's blocks into this construct.
                // Anything else is recorded, never dropped: the jump is
                // explicit and the completeness sweep emits the target under
                // the matching label.
                if cfg.incoming(block).take(2).count() == 1 && !self.is_claimed(block) {
                    walk.awaiting = Awaiting::Inline;
                    let bound = Bound::Set(Rc::clone(&self.unclaimed));
                    return Next::Call(Frame::Region(RegionWalk::new(block, bound, walk.stop)));
                }
                self.push_goto(&mut walk.result, block, GotoReason::BoundaryEscape);
                break;
            }

            self.visit(block);
            match self.lift_entered(block, &walk.allowed, walk.stop, &mut walk.result) {
                Lifted::Next(next) => walk.current = next,
                Lifted::Construct(frame) => {
                    walk.awaiting = Awaiting::Construct;
                    return Next::Call(frame);
                }
            }
        }
        Next::Return(Returned::Nodes(core::mem::take(&mut walk.result)))
    }

    /// Enter an already-visited block: the try region it anchors, or else its
    /// own classified form.
    ///
    /// [`step_region`](Self::step_region)'s walk and a switch arm, whose entry
    /// the switch visits ahead of the walk, both enter blocks here, so a
    /// region anchored at an arm entry is structured like any other.
    pub(super) fn lift_entered(
        &mut self,
        block: BlockId,
        allowed: &Bound,
        stop: Option<BlockId>,
        result: &mut Vec<ControlNode>,
    ) -> Lifted<'cfg> {
        if let Some(frame) = self.select_try(block, allowed) {
            return Lifted::Construct(frame);
        }
        self.lift_block(block, allowed, stop, result)
    }

    /// Classify one already-visited block and emit its form.
    ///
    /// This is the single classification path: [`lift_entered`](Self::lift_entered)
    /// and the try-body anchor both run it, so a region anchored at a branch,
    /// dispatch, or loop header keeps its structure inside the enclosing
    /// construct.
    pub(super) fn lift_block(
        &mut self,
        block: BlockId,
        allowed: &Bound,
        stop: Option<BlockId>,
        result: &mut Vec<ControlNode>,
    ) -> Lifted<'cfg> {
        let cfg = self.cfg;
        let outgoing: Vec<EdgeId> = cfg.outgoing(block).collect();

        match self.classify_block(block) {
            BlockFlowKind::LoopHeader => {
                return Lifted::Construct(self.begin_loop(block, allowed));
            }
            BlockFlowKind::Conditional => {
                let conditional = Conditional::new(block, allowed.clone());
                return Lifted::Construct(Frame::Conditional(conditional));
            }
            BlockFlowKind::Switch => {
                return Lifted::Construct(self.begin_switch(block, allowed));
            }
            BlockFlowKind::BackEdge => {
                self.push_block(result, block);
                if let Some(target) = self.flow_back_edge_target(block) {
                    if let Some(node) = self.resolve_loop_transfer(target) {
                        result.push(node);
                    } else {
                        self.push_goto(result, target, GotoReason::RevisitedTarget);
                    }
                }
                return Lifted::Next(None);
            }
            BlockFlowKind::Jump => {
                return Lifted::Next(self.lift_jump(block, &outgoing, stop, result));
            }
            BlockFlowKind::Linear => {}
        }

        if outgoing.is_empty() {
            let span = self.whole_span(block);
            if !span.is_empty() {
                result.push(ControlNode::Return(span));
            }
            return Lifted::Next(None);
        }

        self.push_block(result, block);

        // Advance along the single sequential successor; exception edges are
        // not sequential flow, so they never block the advance. The walk in
        // [`step_region`](Self::step_region) resolves the target against the
        // stop block, the loop stack, and the enclosing bound.
        let flow_succs = flow_successors(cfg, block);
        if let &[only] = flow_succs.as_slice() {
            return Lifted::Next(Some(only));
        }
        Lifted::Next(None)
    }

    /// Emit a block whose flow leaves through explicit jumps, and return the
    /// block sequential flow continues at (if any).
    fn lift_jump(
        &mut self,
        block: BlockId,
        outgoing: &[EdgeId],
        stop: Option<BlockId>,
        result: &mut Vec<ControlNode>,
    ) -> Option<BlockId> {
        let cfg = self.cfg;
        self.push_block(result, block);
        for &eid in outgoing {
            let edge = cfg.edge(eid);
            if edge.kind() != EdgeKind::Jump {
                continue;
            }
            let target = edge.target();
            if Some(target) == stop {
                // The jump lands exactly on the enclosing construct's
                // convergence point (an arm skipping its sibling to reach the
                // join): structurally the arm simply ends.
                continue;
            }
            if let Some(node) = self.resolve_loop_transfer(target) {
                result.push(node);
            } else if !self.is_visited(target)
                && cfg.incoming(target).take(2).count() == 1
                && !self.is_claimed(target)
            {
                // Exclusively owned jump target — this explicit jump is its
                // only entry and no exception region claims it (javac jumps
                // over handler code to reach a join): sequential flow simply
                // continues there.
                return Some(target);
            } else {
                self.push_goto(result, target, GotoReason::ExplicitJump);
            }
        }
        None
    }

    /// The ultimate target of a chain of pure jump trampolines: an unemitted,
    /// instruction-free block with a single unconditional successor forwards
    /// a resolution unchanged. The HLIL lift empties dialect-declared pure
    /// transfers in its working view before structuring (javac's `break`
    /// routes through such blocks); frontends structuring a raw [`Cfg`] whose
    /// jump encodings occupy an instruction clear them the same way.
    ///
    /// [`Cfg`]: crate::Cfg
    fn through_trampolines(&self, target: BlockId) -> (BlockId, Vec<BlockId>) {
        let cfg = self.cfg;
        let mut hops = Vec::new();
        let mut current = target;
        for _ in 0..8 {
            if self.is_visited(current) || self.instruction_count(current) != 0 {
                break;
            }
            let mut normal = cfg
                .outgoing(current)
                .map(|edge| cfg.edge(edge))
                .filter(|edge| !edge.kind().is_exceptional());
            let (Some(edge), None) = (normal.next(), normal.next()) else {
                break;
            };
            if edge.kind() != EdgeKind::Jump {
                break;
            }
            hops.push(current);
            current = edge.target();
        }
        (current, hops)
    }

    /// Resolve a control transfer against the enclosing loops: `continue`
    /// when it reaches a loop's continue point, `break` when it reaches a
    /// loop's follow. Inner loops win; resolving against an outer loop labels
    /// it.
    pub(super) fn resolve_loop_transfer(&mut self, target: BlockId) -> Option<ControlNode> {
        let (target, hops) = self.through_trampolines(target);
        // The innermost loop that names the target: the last position pushed
        // for it, since the loops leave in the order they entered.
        let position = self
            .loop_targets
            .get(&target)
            .and_then(|positions| positions.last().copied())?;
        // The consumed trampolines carry no content; the sweep must not
        // resurrect them as labeled residue.
        for hop in hops {
            self.visit(hop);
        }
        let innermost = position + 1 == self.loop_stack.len();
        let label = if innermost {
            None
        } else {
            self.loop_stack[position].labeled = true;
            Some(self.loop_stack[position].header)
        };
        let context = &self.loop_stack[position];
        Some(if context.continue_target == target {
            ControlNode::Continue { label }
        } else {
            ControlNode::Break { label }
        })
    }

    pub(super) fn push_goto(
        &mut self,
        result: &mut Vec<ControlNode>,
        target: BlockId,
        reason: GotoReason,
    ) {
        self.goto_targets.insert(target);
        if !self.is_visited(target) {
            self.unemitted_targets.insert(target);
        }
        self.report.gotos.push(GotoDiagnostic { target, reason });
        result.push(ControlNode::Goto { target });
    }
}
