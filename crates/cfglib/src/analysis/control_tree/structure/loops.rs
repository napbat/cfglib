//! Natural-loop structuring: shape classification, bodies, and follows.
//!
//! A loop's body is lifted inside the natural-loop block set, so exits
//! resolve to `Break` / `Continue` against the loop stack instead of being
//! absorbed into the body — the follow block is returned to the caller as
//! the loop's sequential continuation.

extern crate alloc;
use alloc::collections::BTreeSet;
use alloc::vec::Vec;

use super::branches::conditional_targets;
use super::{BlockFlowKind, LoopContext, Structurer, block_is_allowed, branching, flow_successors};
use crate::block::BlockId;
use crate::cfg::Cfg;
use crate::edge::{EdgeId, EdgeKind};
use crate::graph::structure::NaturalLoop;

use super::super::node::{ControlNode, LoopKind};

/// The recognized shape of one natural loop before its body is lifted.
enum LoopShape {
    /// Pre-tested: a straight-line chain from the header reaches a
    /// conditional that exits the loop on one arm.
    While {
        /// The condition chain, header first, conditional block last.
        chain: Vec<BlockId>,
        body_start: BlockId,
        exit: BlockId,
        exit_on_true: bool,
    },
    /// Post-tested: the unique latch's conditional returns or exits.
    DoWhile {
        latch: BlockId,
        exit: BlockId,
        continue_on_true: bool,
    },
    /// Post-tested single-block loop: the header is its own latch.
    SelfLoop {
        exit: BlockId,
        continue_on_true: bool,
    },
    /// No recognized test.
    Endless,
}

/// Walk the straight-line condition chain from the header: every iteration
/// runs each link exactly once (one flow successor, and past the header one
/// flow predecessor), so a loop-exiting conditional at the end makes the
/// whole chain the pre-test.
fn while_chain<I, E>(
    cfg: &Cfg<I, E>,
    natural: &NaturalLoop,
    header: BlockId,
    visited: &[bool],
) -> Option<LoopShape> {
    let mut chain = alloc::vec![header];
    let mut current = header;
    loop {
        if let Some(targets) = conditional_targets(cfg, current) {
            let true_inside = natural.body.contains(&targets.true_target);
            let false_inside = natural.body.contains(&targets.false_target);
            if true_inside == false_inside {
                return None;
            }
            let (exit, body_start) = if true_inside {
                (targets.false_target, targets.true_target)
            } else {
                (targets.true_target, targets.false_target)
            };
            if body_start == header {
                if chain.len() == 1 {
                    // The in-loop arm is the back edge itself: the test
                    // runs after the block's instructions, a single-block
                    // do/while.
                    return Some(LoopShape::SelfLoop {
                        exit,
                        continue_on_true: true_inside,
                    });
                }
                // The whole chain repeats and then tests: that is the
                // post-tested shape, classified by the latch rule.
                return None;
            }
            return Some(LoopShape::While {
                chain,
                body_start,
                exit,
                exit_on_true: !true_inside,
            });
        }
        let successors = flow_successors(cfg, current);
        let [next] = successors.as_slice() else {
            return None;
        };
        let next = *next;
        if next == header
            || !natural.body.contains(&next)
            || chain.contains(&next)
            || visited[next.index()]
        {
            // A link already emitted elsewhere would be emitted again as
            // part of the condition.
            return None;
        }
        let sequential_predecessors = cfg
            .incoming(next)
            .filter(|&edge| !cfg.edge(edge).kind().is_exceptional())
            .count();
        if sequential_predecessors != 1 {
            return None;
        }
        chain.push(next);
        current = next;
    }
}

fn classify_loop<I, E>(
    cfg: &Cfg<I, E>,
    natural: &NaturalLoop,
    header: BlockId,
    visited: &[bool],
) -> LoopShape {
    if let Some(shape) = while_chain(cfg, natural, header, visited) {
        return shape;
    }
    if natural.latches.len() == 1 {
        let latch = *natural
            .latches
            .iter()
            .next()
            .expect("a one-element set has a first element");
        if latch != header
            && !visited[latch.index()]
            && let Some(targets) = conditional_targets(cfg, latch)
        {
            let true_back = targets.true_target == header;
            let false_back = targets.false_target == header;
            if true_back != false_back {
                let other = if true_back {
                    targets.false_target
                } else {
                    targets.true_target
                };
                if !natural.body.contains(&other) {
                    return LoopShape::DoWhile {
                        latch,
                        exit: other,
                        continue_on_true: true_back,
                    };
                }
            }
        }
    }
    LoopShape::Endless
}

/// The deterministic follow of a loop with no recognized test: the first
/// sequential exit target scanning body blocks in identity order, preferring
/// targets inside the enclosing bound.
fn endless_follow<I, E>(
    cfg: &Cfg<I, E>,
    natural: &NaturalLoop,
    reachable: &[bool],
    allowed_blocks: Option<&BTreeSet<BlockId>>,
) -> Option<BlockId> {
    let mut fallback = None;
    // The body walk also collects dead code that feeds the loop. Such a block
    // is still the loop's to lift, but an exit only it takes is no exit of a
    // loop the entry runs.
    for &block in natural.body.iter().filter(|block| reachable[block.index()]) {
        for eid in cfg.outgoing(block) {
            let edge = cfg.edge(eid);
            let sequential = matches!(
                edge.kind(),
                EdgeKind::Fallthrough
                    | EdgeKind::Unconditional
                    | EdgeKind::ConditionalTrue
                    | EdgeKind::ConditionalFalse
                    | EdgeKind::Jump
            );
            if sequential && !natural.body.contains(&edge.target()) {
                if block_is_allowed(allowed_blocks, edge.target()) {
                    return Some(edge.target());
                }
                fallback.get_or_insert(edge.target());
            }
        }
    }
    fallback
}

impl<I, E> Structurer<'_, I, E> {
    /// Lift the loop anchored at `header` (already visited by the caller) and
    /// return the lifted node with the loop's sequential continuation.
    pub(super) fn lift_loop(
        &mut self,
        header: BlockId,
        allowed_blocks: Option<&BTreeSet<BlockId>>,
    ) -> (ControlNode, Option<BlockId>) {
        let cfg = self.cfg;
        let fallback;
        let natural = if let Some(natural) = self.loops.get(&header) {
            natural
        } else {
            // The classification and the loop map derive from the same back
            // edges, so a header without a loop is a library bug — degrade
            // to a single-block loop rather than dropping code.
            debug_assert!(false, "loop header {header} has no natural loop");
            fallback = NaturalLoop {
                header,
                body: [header].into_iter().collect(),
                latches: [header].into_iter().collect(),
                depth: 0,
            };
            &fallback
        };
        let bound: BTreeSet<BlockId> = match allowed_blocks {
            None => natural.body.clone(),
            Some(allowed) => natural.body.intersection(allowed).copied().collect(),
        };

        let shape = classify_loop(cfg, natural, header, &self.visited);
        let (follow, continue_target) = match &shape {
            LoopShape::While { exit, .. } | LoopShape::SelfLoop { exit, .. } => {
                (Some(*exit), header)
            }
            LoopShape::DoWhile { latch, exit, .. } => (Some(*exit), *latch),
            LoopShape::Endless => (
                endless_follow(cfg, natural, &self.reachable, allowed_blocks),
                header,
            ),
        };

        // A multi-block condition chain belongs to the loop's test, not its
        // body or the completeness sweep.
        if let LoopShape::While { chain, .. } = &shape {
            for &block in &chain[1..] {
                self.visit(block);
            }
        }

        self.loop_stack.push(LoopContext {
            header,
            follow,
            continue_target,
            labeled: false,
        });

        let (kind, mut body) = self.lift_shape(header, &bound, shape);

        // A trailing plain `continue` at the structural end of the body is
        // the loop's own back edge — implicit in the representation.
        if matches!(body.last(), Some(ControlNode::Continue { label: None })) {
            body.pop();
        }

        let context = self
            .loop_stack
            .pop()
            .expect("the loop context pushed above is still on the stack");
        let node = ControlNode::Loop {
            header,
            kind,
            body: self.tables.seal(body),
        };
        let node = if context.labeled {
            self.labeled_blocks.insert(header);
            let wrapped = self.tables.seal(alloc::vec![node]);
            ControlNode::Label {
                block: header,
                body: wrapped,
            }
        } else {
            node
        };
        (node, follow)
    }

    /// Lift the loop's condition and body per its recognized shape.
    fn lift_shape(
        &mut self,
        header: BlockId,
        bound: &BTreeSet<BlockId>,
        shape: LoopShape,
    ) -> (LoopKind, Vec<ControlNode>) {
        match shape {
            LoopShape::While {
                chain,
                body_start,
                exit_on_true,
                ..
            } => {
                let spans: Vec<_> = chain.iter().map(|&block| self.whole_span(block)).collect();
                let condition = self.tables.push_spans(spans.into_iter());
                let body = if body_start == header {
                    // The conditional's in-loop arm re-enters the chain: the
                    // loop is its own condition.
                    Vec::new()
                } else {
                    self.lift_region(body_start, Some(bound), None)
                };
                (
                    LoopKind::While {
                        condition,
                        exit_on_true,
                    },
                    body,
                )
            }
            LoopShape::SelfLoop {
                continue_on_true, ..
            } => (
                LoopKind::DoWhile {
                    latch: self.whole_span(header),
                    continue_on_true,
                },
                Vec::new(),
            ),
            LoopShape::DoWhile {
                latch,
                continue_on_true,
                ..
            } => {
                // The latch is the condition, not part of the body: pre-visit
                // it so the body walk ends there silently, and transfers to it
                // resolve as `continue`.
                self.visit(latch);
                let body = self.lift_header_body(header, bound, Some(latch));
                (
                    LoopKind::DoWhile {
                        latch: self.whole_span(latch),
                        continue_on_true,
                    },
                    body,
                )
            }
            LoopShape::Endless => (
                LoopKind::Endless,
                self.lift_header_body(header, bound, None),
            ),
        }
    }

    /// Lift a loop body whose header's own instructions belong inside it (the
    /// post-tested and endless shapes): emit the header's structure manually —
    /// it is already visited — then continue the ordinary walk.
    ///
    /// A plain header's edge to itself is the loop's own back edge, implicit
    /// in the representation. Any other edge is a transfer the walk resolves,
    /// even when it is the back edge of an enclosing loop: that one is a
    /// `continue` of the enclosing loop, not of this one.
    fn lift_header_body(
        &mut self,
        header: BlockId,
        bound: &BTreeSet<BlockId>,
        stop: Option<BlockId>,
    ) -> Vec<ControlNode> {
        let cfg = self.cfg;
        let mut body = Vec::new();
        let outgoing: Vec<EdgeId> = cfg.outgoing(header).collect();

        match branching(cfg, &outgoing) {
            Some(BlockFlowKind::Conditional) => {
                let node = self.lift_conditional(header, Some(bound));
                body.push(node);
                if let Some(merge) = self.effective_merge(header, Some(bound)) {
                    body.extend(self.lift_region(merge, Some(bound), stop));
                }
            }
            Some(BlockFlowKind::Switch) => {
                let node = self.lift_switch(header, Some(bound));
                body.push(node);
                if let Some(merge) = self.effective_merge(header, Some(bound)) {
                    body.extend(self.lift_region(merge, Some(bound), stop));
                }
            }
            _ => {
                self.push_block(&mut body, header);
                for eid in outgoing {
                    let edge = cfg.edge(eid);
                    if edge.target() != header && !edge.kind().is_exceptional() {
                        body.extend(self.lift_region(edge.target(), Some(bound), stop));
                    }
                }
            }
        }
        body
    }
}
