//! Branch structuring: the merge of a two-way or multi-way branch, and the
//! arms that run up to it.

extern crate alloc;
use alloc::collections::{BTreeMap, BTreeSet};
use alloc::vec::Vec;

use super::super::node::{ControlNode, SwitchCase};
use super::super::report::GotoReason;
use super::{Structurer, block_is_allowed};
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

impl<I, E> Structurer<'_, I, E> {
    /// The post-dominator merge of `block`, kept only when it stays inside
    /// the enclosing structural bound.
    ///
    /// The merge may be code the walk already emitted: the walk that follows
    /// the construct resolves it like any other transfer, as a loop's
    /// `continue` or `break` or an explicit goto, and ends silently only at the
    /// enclosing construct's own stop block.
    pub(super) fn effective_merge(
        &self,
        block: BlockId,
        allowed_blocks: Option<&BTreeSet<BlockId>>,
    ) -> Option<BlockId> {
        match self.pdom.ipdom(block.raw())? {
            PostNode::Node(merge) => Some(BlockId::from_raw(merge)),
            PostNode::Exit => None,
        }
        .filter(|&merge| block_is_allowed(allowed_blocks, merge))
    }

    /// The block the walk resumes at after `node`, a construct whose merge is
    /// `merge`.
    ///
    /// A merge that was already emitted is a transfer (a `continue`, a
    /// `break`, or a goto), unless no path reaches the end of `node`: when
    /// every arm already ends in a transfer, one more after the construct
    /// would be dead code.
    pub(super) fn resume_after(
        &self,
        node: &ControlNode,
        merge: Option<BlockId>,
    ) -> Option<BlockId> {
        merge.filter(|&next| !(self.is_visited(next) && self.tables.never_completes(*node)))
    }

    /// Lift an if/else conditional starting at `block`.
    pub(super) fn lift_conditional(
        &mut self,
        block: BlockId,
        allowed_blocks: Option<&BTreeSet<BlockId>>,
    ) -> ControlNode {
        let targets = conditional_targets(self.cfg, block)
            .expect("a block classified as a conditional has a true and a false edge");
        let merge = self.effective_merge(block, allowed_blocks);

        let then_nodes = self.lift_region(targets.true_target, allowed_blocks, merge);
        let else_nodes = self.lift_region(targets.false_target, allowed_blocks, merge);

        ControlNode::IfThenElse {
            condition: self.whole_span(block),
            then_body: self.tables.seal(then_nodes),
            else_body: self.tables.seal(else_nodes),
        }
    }

    /// Lift a switch starting at `block`.
    pub(super) fn lift_switch(
        &mut self,
        block: BlockId,
        allowed_blocks: Option<&BTreeSet<BlockId>>,
    ) -> ControlNode {
        let cfg = self.cfg;
        let merge = self.effective_merge(block, allowed_blocks);

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
            if Some(target) != merge
                && block_is_allowed(allowed_blocks, target)
                && !self.is_visited(target)
            {
                self.visit(target);
                owned.insert(target);
            }
        }

        let mut cases = Vec::new();
        for target in case_targets {
            let edges = case_edges.remove(&target).unwrap_or_default();
            let entered = owned.remove(&target);
            let arm = self.lift_switch_arm(target, entered, allowed_blocks, merge);
            cases.push(SwitchCase {
                block: target,
                edges: self.tables.push_edges(edges),
                body: self.tables.seal(arm),
            });
        }
        let default_nodes = default_target
            .map(|target| {
                let entered = owned.remove(&target);
                self.lift_switch_arm(target, entered, allowed_blocks, merge)
            })
            .unwrap_or_default();

        ControlNode::Switch {
            condition: self.whole_span(block),
            cases: self.tables.push_cases(cases),
            default_body: self.tables.seal(default_nodes),
            default_edge,
        }
    }

    /// Lift one switch arm (a case body or the default body) up to the merge.
    ///
    /// `entered` says the switch pre-visited the arm entry and owns it.
    fn lift_switch_arm(
        &mut self,
        target: BlockId,
        entered: bool,
        allowed_blocks: Option<&BTreeSet<BlockId>>,
        merge: Option<BlockId>,
    ) -> Vec<ControlNode> {
        if Some(target) == merge {
            // The arm transfers straight to the switch continuation.
            return Vec::new();
        }
        if !block_is_allowed(allowed_blocks, target) {
            // Beyond the enclosing bound. When that bound is a loop and the
            // target is its follow or header, the transfer is a break or a
            // continue, as it is for the arm of an `if`; anything else is an
            // explicit jump.
            let mut body = Vec::new();
            match self.resolve_loop_transfer(target) {
                Some(node) => body.push(node),
                None => self.push_goto(&mut body, target, GotoReason::BoundaryEscape),
            }
            return body;
        }
        if !entered {
            // Earlier code owns the entry: a continue, a break, or a goto.
            return self.lift_region(target, allowed_blocks, merge);
        }
        // The arm entry was pre-visited above; enter it through the shared
        // path so a try region, branch, or loop at the arm entry keeps its
        // structure, then continue the ordinary walk toward the merge.
        let mut body = Vec::new();
        let next = self.lift_entered(target, allowed_blocks, merge, &mut body);
        if let Some(next) = next {
            body.extend(self.lift_region(next, allowed_blocks, merge));
        }
        body
    }
}
