//! Branch structuring: the merge of a two-way or multi-way branch, and the
//! arms that run up to it.

extern crate alloc;
use alloc::boxed::Box;
use alloc::collections::{BTreeMap, BTreeSet};
use alloc::vec::Vec;

use super::super::node::{ControlNode, SwitchCase};
use super::super::report::GotoReason;
use super::Structurer;
use super::bound::Bound;
use super::frames::{Frame, Next, Returned};
use super::walk::{Lifted, RegionWalk};
use crate::block::BlockId;
use crate::cfg::Cfg;
use crate::edge::{EdgeId, EdgeKind};
use crate::graph::dominator::PostNode;

/// The two conditional-branch targets of a block, when it ends in one.
pub(super) struct ConditionalTargets {
    pub(super) true_target: BlockId,
    pub(super) false_target: BlockId,
}

pub(super) fn conditional_targets<I, E>(
    cfg: &Cfg<I, E>,
    block: BlockId,
) -> Option<ConditionalTargets> {
    let mut true_target = None;
    let mut false_target = None;
    for eid in cfg.outgoing(block) {
        match cfg.edge(eid).kind() {
            EdgeKind::ConditionalTrue => true_target = Some(cfg.edge(eid).target()),
            EdgeKind::ConditionalFalse => false_target = Some(cfg.edge(eid).target()),
            _ => {}
        }
    }
    Some(ConditionalTargets {
        true_target: true_target?,
        false_target: false_target?,
    })
}

/// An if/else conditional being lifted: the frame that lifts both arms.
pub(super) struct Conditional {
    block: BlockId,
    allowed: Bound,
    merge: Option<BlockId>,
    arm: ConditionalArm,
}

enum ConditionalArm {
    /// Nothing is lifted yet.
    Start,
    /// The `then` arm is being lifted; the `else` arm's entry follows.
    Then { false_target: BlockId },
    /// The `else` arm is being lifted; the `then` arm's statements are done.
    Else { then_nodes: Vec<ControlNode> },
}

impl Conditional {
    /// The frame that lifts the conditional at `block`, within `allowed`.
    pub(super) const fn new(block: BlockId, allowed: Bound) -> Self {
        Self {
            block,
            allowed,
            merge: None,
            arm: ConditionalArm::Start,
        }
    }
}

/// A switch being lifted: the frame that lifts each arm in dispatch order.
pub(super) struct Switch {
    block: BlockId,
    allowed: Bound,
    merge: Option<BlockId>,
    /// The case arms lifted so far.
    cases: Vec<SwitchCase>,
    /// The case targets still to lift, in first-encounter order.
    pending: alloc::vec::IntoIter<BlockId>,
    case_edges: BTreeMap<BlockId, Vec<EdgeId>>,
    /// The arm entries the switch pre-visited and owns.
    owned: BTreeSet<BlockId>,
    default_target: Option<BlockId>,
    default_edge: Option<EdgeId>,
    arm: SwitchWait,
}

enum SwitchWait {
    /// No arm is being lifted.
    Idle,
    /// The case arm at `target` with the dispatch `edges` is being lifted.
    Case {
        target: BlockId,
        edges: Vec<EdgeId>,
    },
    Default,
}

/// One switch arm (a case body or the default body), lifted up to the merge.
pub(super) struct SwitchArm {
    target: BlockId,
    /// The switch pre-visited the arm entry and owns it.
    entered: bool,
    allowed: Bound,
    merge: Option<BlockId>,
    body: Vec<ControlNode>,
    awaiting: ArmWait,
}

enum ArmWait {
    Nothing,
    /// An arm whose entry earlier code owns is one walk: its statements are
    /// the arm.
    Walk,
    /// The construct at the arm entry.
    Construct,
    /// The walk after the arm entry, toward the merge.
    Rest,
}

impl<'cfg, I, E> Structurer<'cfg, I, E> {
    /// The post-dominator merge of `block`, kept only when it stays inside
    /// the enclosing structural bound.
    ///
    /// The merge may be code the walk already emitted: the walk that follows
    /// the construct resolves it like any other transfer, as a loop's
    /// `continue` or `break` or an explicit goto, and ends silently only at the
    /// enclosing construct's own stop block.
    pub(super) fn effective_merge(&self, block: BlockId, allowed: &Bound) -> Option<BlockId> {
        match self.pdom.ipdom(block.index())? {
            PostNode::Node(merge) => Some(BlockId::from_index(merge)),
            PostNode::Exit => None,
        }
        .filter(|&merge| allowed.contains(merge))
    }

    /// The block the walk resumes at after `node`, a construct whose merge is
    /// `merge`.
    ///
    /// A merge that was already emitted is a transfer (a `continue`, a
    /// `break`, or a goto), unless no path reaches the end of `node`: when
    /// every arm already ends in a `break`, `continue`, or return, one more
    /// after the construct would be dead code.
    pub(super) fn resume_after(
        &mut self,
        node: &ControlNode,
        merge: Option<BlockId>,
    ) -> Option<BlockId> {
        merge.filter(|&next| !(self.is_visited(next) && self.never_completes(*node)))
    }

    /// The construct a branch or dispatch at `block` returns, with the block
    /// the walk goes on at after it: the merge, when the construct can reach
    /// it.
    fn finish_branch(&mut self, block: BlockId, allowed: &Bound, node: ControlNode) -> Next<'cfg> {
        let merge = self.effective_merge(block, allowed);
        let next = self.resume_after(&node, merge);
        Next::Return(Returned::Construct { node, next })
    }

    pub(super) fn step_conditional(
        &mut self,
        conditional: &mut Conditional,
        returned: Returned,
    ) -> Next<'cfg> {
        let block = conditional.block;
        match core::mem::replace(&mut conditional.arm, ConditionalArm::Start) {
            ConditionalArm::Start => {
                let targets = conditional_targets(self.cfg, block)
                    .expect("a block classified as a conditional has a true and a false edge");
                conditional.merge = self.effective_merge(block, &conditional.allowed);
                conditional.arm = ConditionalArm::Then {
                    false_target: targets.false_target,
                };
                Next::Call(Frame::Region(RegionWalk::new(
                    targets.true_target,
                    conditional.allowed.clone(),
                    conditional.merge,
                )))
            }
            ConditionalArm::Then { false_target } => {
                conditional.arm = ConditionalArm::Else {
                    then_nodes: returned.into_nodes(),
                };
                Next::Call(Frame::Region(RegionWalk::new(
                    false_target,
                    conditional.allowed.clone(),
                    conditional.merge,
                )))
            }
            ConditionalArm::Else { then_nodes } => {
                let else_nodes = returned.into_nodes();
                let node = ControlNode::IfThenElse {
                    condition: self.whole_span(block),
                    then_body: self.tables.seal(then_nodes),
                    else_body: self.tables.seal(else_nodes),
                };
                let allowed = conditional.allowed.clone();
                self.finish_branch(block, &allowed, node)
            }
        }
    }

    /// Start lifting the switch at `block`.
    pub(super) fn begin_switch(&mut self, block: BlockId, allowed: &Bound) -> Frame<'cfg> {
        let cfg = self.cfg;
        let merge = self.effective_merge(block, allowed);

        // Group dispatch edges by target in first-encounter order, and find
        // the explicit default edge (the sequential non-case successor).
        let mut case_targets: Vec<BlockId> = Vec::new();
        let mut case_edges: BTreeMap<BlockId, Vec<EdgeId>> = BTreeMap::new();
        let mut default_edge = None;
        let mut default_target = None;
        for eid in cfg.outgoing(block) {
            let edge = cfg.edge(eid);
            match edge.kind() {
                EdgeKind::SwitchCase => {
                    let target = edge.target();
                    let edges = case_edges.entry(target).or_default();
                    if edges.is_empty() {
                        case_targets.push(target);
                    }
                    edges.push(eid);
                }
                EdgeKind::Fallthrough | EdgeKind::Unconditional if default_edge.is_none() => {
                    default_edge = Some(eid);
                    default_target = Some(edge.target());
                }
                _ => {}
            }
        }

        // Pre-visit every arm entry so one arm's walk cannot absorb another's;
        // an arm-to-arm transfer stays explicit instead. The switch owns only
        // an entry that nothing emitted or claimed before it: an arm that
        // targets earlier code (the enclosing loop's header, a shared tail, an
        // entry a sibling arm already owns) is a transfer, resolved like any
        // other, never a second copy of the block.
        let mut owned: BTreeSet<BlockId> = BTreeSet::new();
        for &target in case_targets.iter().chain(&default_target) {
            if Some(target) != merge && allowed.contains(target) && !self.is_visited(target) {
                self.visit(target);
                owned.insert(target);
            }
        }

        Frame::Switch(Box::new(Switch {
            block,
            allowed: allowed.clone(),
            merge,
            cases: Vec::new(),
            pending: case_targets.into_iter(),
            case_edges,
            owned,
            default_target,
            default_edge,
            arm: SwitchWait::Idle,
        }))
    }

    pub(super) fn step_switch(&mut self, switch: &mut Switch, returned: Returned) -> Next<'cfg> {
        match core::mem::replace(&mut switch.arm, SwitchWait::Idle) {
            SwitchWait::Idle => {}
            SwitchWait::Case { target, edges } => {
                let arm = returned.into_nodes();
                switch.cases.push(SwitchCase {
                    block: target,
                    edges: self.tables.push_edges(edges),
                    body: self.tables.seal(arm),
                });
            }
            SwitchWait::Default => {
                let default_nodes = returned.into_nodes();
                return self.finish_switch(switch, default_nodes);
            }
        }
        if let Some(target) = switch.pending.next() {
            let edges = switch.case_edges.remove(&target).unwrap_or_default();
            let entered = switch.owned.remove(&target);
            switch.arm = SwitchWait::Case { target, edges };
            return Next::Call(Self::switch_arm(switch, target, entered));
        }
        let Some(target) = switch.default_target else {
            return self.finish_switch(switch, Vec::new());
        };
        let entered = switch.owned.remove(&target);
        switch.arm = SwitchWait::Default;
        Next::Call(Self::switch_arm(switch, target, entered))
    }

    fn switch_arm(switch: &Switch, target: BlockId, entered: bool) -> Frame<'cfg> {
        Frame::SwitchArm(SwitchArm {
            target,
            entered,
            allowed: switch.allowed.clone(),
            merge: switch.merge,
            body: Vec::new(),
            awaiting: ArmWait::Nothing,
        })
    }

    fn finish_switch(
        &mut self,
        switch: &mut Switch,
        default_nodes: Vec<ControlNode>,
    ) -> Next<'cfg> {
        let cases = core::mem::take(&mut switch.cases);
        let node = ControlNode::Switch {
            condition: self.whole_span(switch.block),
            cases: self.tables.push_cases(cases),
            default_body: self.tables.seal(default_nodes),
            default_edge: switch.default_edge,
        };
        let allowed = switch.allowed.clone();
        self.finish_branch(switch.block, &allowed, node)
    }

    pub(super) fn step_switch_arm(
        &mut self,
        arm: &mut SwitchArm,
        returned: Returned,
    ) -> Next<'cfg> {
        match core::mem::replace(&mut arm.awaiting, ArmWait::Nothing) {
            ArmWait::Nothing => self.start_switch_arm(arm),
            ArmWait::Walk => Next::Return(returned),
            ArmWait::Construct => {
                let (node, next) = returned.into_construct();
                arm.body.push(node);
                Self::continue_switch_arm(arm, next)
            }
            ArmWait::Rest => {
                arm.body.extend(returned.into_nodes());
                Next::Return(Returned::Nodes(core::mem::take(&mut arm.body)))
            }
        }
    }

    fn start_switch_arm(&mut self, arm: &mut SwitchArm) -> Next<'cfg> {
        let target = arm.target;
        if Some(target) == arm.merge {
            // The arm transfers straight to the switch continuation.
            return Next::Return(Returned::Nodes(Vec::new()));
        }
        if !arm.allowed.contains(target) {
            // Beyond the enclosing bound. When that bound is a loop and the
            // target is its follow or header, the transfer is a break or a
            // continue, as it is for the arm of an `if`; anything else is an
            // explicit jump.
            let mut body = Vec::new();
            match self.resolve_loop_transfer(target) {
                Some(node) => body.push(node),
                None => self.push_goto(&mut body, target, GotoReason::BoundaryEscape),
            }
            return Next::Return(Returned::Nodes(body));
        }
        if !arm.entered {
            // Earlier code owns the entry: a continue, a break, or a goto.
            arm.awaiting = ArmWait::Walk;
            return Next::Call(Frame::Region(RegionWalk::new(
                target,
                arm.allowed.clone(),
                arm.merge,
            )));
        }
        // The arm entry was pre-visited by the switch; enter it through the
        // shared path so a try region, branch, or loop at the arm entry keeps
        // its structure, then continue the ordinary walk toward the merge.
        match self.lift_entered(target, &arm.allowed, arm.merge, &mut arm.body) {
            Lifted::Next(next) => Self::continue_switch_arm(arm, next),
            Lifted::Construct(frame) => {
                arm.awaiting = ArmWait::Construct;
                Next::Call(frame)
            }
        }
    }

    /// Go on with the arm at `next`, the block after its entry.
    fn continue_switch_arm(arm: &mut SwitchArm, next: Option<BlockId>) -> Next<'cfg> {
        let Some(next) = next else {
            return Next::Return(Returned::Nodes(core::mem::take(&mut arm.body)));
        };
        arm.awaiting = ArmWait::Rest;
        Next::Call(Frame::Region(RegionWalk::new(
            next,
            arm.allowed.clone(),
            arm.merge,
        )))
    }
}
