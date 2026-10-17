//! Natural-loop structuring: shape classification, bodies, and follows.
//!
//! A loop's body is lifted inside the natural-loop block set, so exits
//! resolve to `Break` / `Continue` against the loop stack instead of being
//! absorbed into the body — the follow block is returned to the caller as
//! the loop's sequential continuation.

extern crate alloc;
use alloc::boxed::Box;
use alloc::rc::Rc;
use alloc::vec::Vec;

use super::bound::Bound;
use super::branches::{Conditional, conditional_targets};
use super::frames::{Frame, Next, Returned};
use super::walk::RegionWalk;
use super::{BlockFlowKind, LoopContext, Structurer, branching, flow_successors};
use crate::block::BlockId;
use crate::cfg::Cfg;
use crate::edge::EdgeId;
use crate::graph::loop_forest::{LoopForest, LoopId};

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

/// One loop of the loop forest.
#[derive(Clone, Copy)]
struct LoopRef<'a> {
    forest: &'a LoopForest,
    lp: LoopId,
}

impl<'a> LoopRef<'a> {
    fn contains(self, block: BlockId) -> bool {
        self.forest.contains(self.lp, block)
    }

    /// The latches, lowest block first.
    fn latches(self) -> &'a [BlockId] {
        self.forest.latches(self.lp)
    }

    /// The blocks, in no order.
    fn body(self) -> impl Iterator<Item = BlockId> + 'a {
        self.forest.body(self.lp)
    }
}

/// Walk the straight-line condition chain from the header: every iteration
/// runs each link exactly once (one flow successor, and past the header one
/// flow predecessor), so a loop-exiting conditional at the end makes the
/// whole chain the pre-test.
fn while_chain<I, E>(
    cfg: &Cfg<I, E>,
    natural: LoopRef<'_>,
    header: BlockId,
    visited: &[bool],
) -> Option<LoopShape> {
    let mut chain = alloc::vec![header];
    let mut current = header;
    loop {
        if let Some(targets) = conditional_targets(cfg, current) {
            let true_inside = natural.contains(targets.true_target);
            let false_inside = natural.contains(targets.false_target);
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
        if next == header || !natural.contains(next) || visited[next.index()] {
            // A link already emitted elsewhere would be emitted again as
            // part of the condition.
            return None;
        }
        // A link that is already in the chain has a second sequential
        // predecessor, the block that leads back to it, so this count also
        // keeps a link from entering the chain twice.
        let sequential_predecessors = cfg
            .incoming(next)
            .filter(|&edge| !cfg.edge(edge).kind().is_exceptional())
            .take(2)
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
    natural: LoopRef<'_>,
    header: BlockId,
    visited: &[bool],
) -> LoopShape {
    if let Some(shape) = while_chain(cfg, natural, header, visited) {
        return shape;
    }
    if let &[latch] = natural.latches()
        && latch != header
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
            if !natural.contains(other) {
                return LoopShape::DoWhile {
                    latch,
                    exit: other,
                    continue_on_true: true_back,
                };
            }
        }
    }
    LoopShape::Endless
}

/// The deterministic follow of a loop with no recognized test: the first
/// exit target of sequential flow (any edge but an exception edge, the rule
/// [`flow_successors`] states) scanning body blocks in identity order,
/// preferring targets inside the enclosing bound.
///
/// The body comes in no order, so this keeps the first exit of the lowest
/// block and the first allowed exit of the lowest block, which is what the
/// scan in identity order finds.
fn endless_follow<I, E>(
    cfg: &Cfg<I, E>,
    natural: LoopRef<'_>,
    reachable: &[bool],
    allowed: &Bound,
) -> Option<BlockId> {
    let mut first_exit: Option<(BlockId, BlockId)> = None;
    let mut first_allowed: Option<(BlockId, BlockId)> = None;
    // The body walk also collects dead code that feeds the loop. Such a block
    // is still the loop's to lift, but an exit only it takes is no exit of a
    // loop the entry runs.
    for block in natural.body().filter(|block| reachable[block.index()]) {
        let mut seen_exit = false;
        for eid in cfg.outgoing(block) {
            let edge = cfg.edge(eid);
            if edge.kind().is_exceptional() || natural.contains(edge.target()) {
                continue;
            }
            if !seen_exit {
                seen_exit = true;
                if first_exit.is_none_or(|(lowest, _)| block < lowest) {
                    first_exit = Some((block, edge.target()));
                }
            }
            if allowed.contains(edge.target()) {
                if first_allowed.is_none_or(|(lowest, _)| block < lowest) {
                    first_allowed = Some((block, edge.target()));
                }
                break;
            }
        }
    }
    first_allowed.or(first_exit).map(|(_, target)| target)
}

/// A natural loop being lifted: the frame of its body, from the loop's
/// setup until the node is built.
pub(super) struct LoopBuild {
    header: BlockId,
    /// The loop's sequential continuation.
    follow: Option<BlockId>,
    /// The loop's blocks within the enclosing bound.
    bound: Bound,
    /// The recognized shape; taken when the body is lifted.
    shape: Option<LoopShape>,
    /// The kind of the node, known once the shape is lifted.
    kind: Option<LoopKind>,
}

/// The body of a post-tested or endless loop, whose header's own
/// instructions belong inside it: the frame of
/// [`Structurer::begin_header_body`].
pub(super) struct HeaderBody {
    header: BlockId,
    bound: Bound,
    stop: Option<BlockId>,
    body: Vec<ControlNode>,
    awaiting: HeaderWait,
}

enum HeaderWait {
    /// Nothing is lifted yet.
    Nothing,
    /// The branch or dispatch at the header.
    Branch,
    /// The walk from the merge after the branch.
    Merge,
    /// The walk from one of the header's other targets; the rest follow.
    Target(alloc::vec::IntoIter<BlockId>),
}

impl<'cfg, I, E> Structurer<'cfg, I, E> {
    /// Enter a loop: it becomes the innermost, and its continue point and
    /// follow resolve transfers to it.
    fn push_loop_context(&mut self, context: LoopContext) {
        let position = self.loop_stack.len();
        let follow = context
            .follow
            .filter(|&follow| follow != context.continue_target);
        for target in core::iter::once(context.continue_target).chain(follow) {
            self.loop_targets.entry(target).or_default().push(position);
        }
        self.loop_stack.push(context);
    }

    /// Leave the innermost loop.
    fn pop_loop_context(&mut self) -> LoopContext {
        let context = self
            .loop_stack
            .pop()
            .expect("the loop context pushed at entry is still on the stack");
        let follow = context
            .follow
            .filter(|&follow| follow != context.continue_target);
        for target in core::iter::once(context.continue_target).chain(follow) {
            let positions = self
                .loop_targets
                .get_mut(&target)
                .expect("a loop on the stack is registered at its targets");
            positions.pop();
            if positions.is_empty() {
                self.loop_targets.remove(&target);
            }
        }
        context
    }

    /// Start lifting the loop anchored at `header` (already visited by the
    /// caller): classify its shape, fix its follow, and enter its context. The
    /// frame returns the lifted node with the loop's sequential continuation.
    pub(super) fn begin_loop(&mut self, header: BlockId, allowed: &Bound) -> Frame<'cfg> {
        let cfg = self.cfg;
        // The classification and the loops derive from the same back edges, so
        // a loop header always has its loop.
        let lp = self
            .forest
            .loop_of_header(header)
            .expect("a block with an incoming back edge heads a loop of the forest");
        let forest = Rc::clone(&self.forest);
        let natural = LoopRef {
            forest: &forest,
            lp,
        };
        let bound = allowed.within_loop(&forest, lp);

        // The test of a pre- or post-tested loop is consumed by the loop node,
        // so it has to lie inside the enclosing bound: a test block beyond it
        // (a protected loop whose latch the try does not protect, a block that
        // another region claims) would end up inside a construct that does not
        // own it. The loop is then an endless loop whose exits are transfers.
        let shape = match classify_loop(cfg, natural, header, &self.visited) {
            LoopShape::While { ref chain, .. } if !chain.iter().all(|&b| allowed.contains(b)) => {
                LoopShape::Endless
            }
            LoopShape::DoWhile { latch, .. } if !allowed.contains(latch) => LoopShape::Endless,
            shape => shape,
        };
        let (follow, continue_target) = match &shape {
            LoopShape::While { exit, .. } | LoopShape::SelfLoop { exit, .. } => {
                (Some(*exit), header)
            }
            LoopShape::DoWhile { latch, exit, .. } => (Some(*exit), *latch),
            LoopShape::Endless => (
                endless_follow(cfg, natural, &self.reachable, allowed),
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

        self.push_loop_context(LoopContext {
            header,
            follow,
            continue_target,
            labeled: false,
        });

        Frame::Loop(Box::new(LoopBuild {
            header,
            follow,
            bound,
            shape: Some(shape),
            kind: None,
        }))
    }

    pub(super) fn step_loop(&mut self, build: &mut LoopBuild, returned: Returned) -> Next<'cfg> {
        let Some(shape) = build.shape.take() else {
            return self.finish_loop(build, returned.into_nodes());
        };
        // Lift the loop's condition and body per its recognized shape.
        let header = build.header;
        let bound = build.bound.clone();
        match shape {
            LoopShape::While {
                chain,
                body_start,
                exit_on_true,
                ..
            } => {
                let spans: Vec<_> = chain.iter().map(|&block| self.whole_span(block)).collect();
                let condition = self.tables.push_spans(spans.into_iter());
                build.kind = Some(LoopKind::While {
                    condition,
                    exit_on_true,
                });
                if body_start == header {
                    // The conditional's in-loop arm re-enters the chain: the
                    // loop is its own condition.
                    return self.finish_loop(build, Vec::new());
                }
                Next::Call(Frame::Region(RegionWalk::new(body_start, bound, None)))
            }
            LoopShape::SelfLoop {
                continue_on_true, ..
            } => {
                build.kind = Some(LoopKind::DoWhile {
                    latch: self.whole_span(header),
                    continue_on_true,
                });
                self.finish_loop(build, Vec::new())
            }
            LoopShape::DoWhile {
                latch,
                continue_on_true,
                ..
            } => {
                // The latch is the condition, not part of the body: pre-visit
                // it so the body walk ends there silently, and transfers to it
                // resolve as `continue`.
                self.visit(latch);
                build.kind = Some(LoopKind::DoWhile {
                    latch: self.whole_span(latch),
                    continue_on_true,
                });
                Next::Call(Self::header_body(header, &build.bound, Some(latch)))
            }
            LoopShape::Endless => {
                build.kind = Some(LoopKind::Endless);
                Next::Call(Self::header_body(header, &build.bound, None))
            }
        }
    }

    fn finish_loop(&mut self, build: &mut LoopBuild, mut body: Vec<ControlNode>) -> Next<'cfg> {
        // A trailing plain `continue` at the structural end of the body is
        // the loop's own back edge — implicit in the representation.
        if matches!(body.last(), Some(ControlNode::Continue { label: None })) {
            body.pop();
        }

        let context = self.pop_loop_context();
        let header = build.header;
        let node = ControlNode::Loop {
            header,
            kind: build
                .kind
                .take()
                .expect("the shape is lifted before the loop finishes"),
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
        Next::Return(Returned::Construct {
            node,
            next: build.follow,
        })
    }

    /// The frame that lifts the body of a post-tested or endless loop whose
    /// header's own instructions belong inside it: emit the header's structure
    /// manually — it is already visited — then continue the ordinary walk.
    ///
    /// A plain header's edge to itself is the loop's own back edge, implicit
    /// in the representation. Any other edge is a transfer the walk resolves,
    /// even when it is the back edge of an enclosing loop: that one is a
    /// `continue` of the enclosing loop, not of this one.
    fn header_body(header: BlockId, bound: &Bound, stop: Option<BlockId>) -> Frame<'cfg> {
        Frame::HeaderBody(HeaderBody {
            header,
            bound: bound.clone(),
            stop,
            body: Vec::new(),
            awaiting: HeaderWait::Nothing,
        })
    }

    pub(super) fn step_header_body(
        &mut self,
        header_body: &mut HeaderBody,
        returned: Returned,
    ) -> Next<'cfg> {
        let header = header_body.header;
        let bound = header_body.bound.clone();
        match core::mem::replace(&mut header_body.awaiting, HeaderWait::Nothing) {
            HeaderWait::Nothing => {
                let cfg = self.cfg;
                let outgoing: Vec<EdgeId> = cfg.outgoing(header).collect();
                match branching(cfg, &outgoing) {
                    Some(BlockFlowKind::Conditional) => {
                        header_body.awaiting = HeaderWait::Branch;
                        return Next::Call(Frame::Conditional(Conditional::new(header, bound)));
                    }
                    Some(BlockFlowKind::Switch) => {
                        header_body.awaiting = HeaderWait::Branch;
                        return Next::Call(self.begin_switch(header, &bound));
                    }
                    _ => {}
                }
                self.push_block(&mut header_body.body, header);
                let targets: Vec<BlockId> = outgoing
                    .into_iter()
                    .map(|eid| cfg.edge(eid))
                    .filter(|edge| edge.target() != header && !edge.kind().is_exceptional())
                    .map(|edge| edge.target())
                    .collect();
                Self::next_target(header_body, targets.into_iter())
            }
            HeaderWait::Branch => {
                let (node, next) = returned.into_construct();
                header_body.body.push(node);
                let Some(merge) = next else {
                    return Next::Return(Returned::Nodes(core::mem::take(&mut header_body.body)));
                };
                header_body.awaiting = HeaderWait::Merge;
                Next::Call(Frame::Region(RegionWalk::new(
                    merge,
                    bound,
                    header_body.stop,
                )))
            }
            HeaderWait::Merge => {
                header_body.body.extend(returned.into_nodes());
                Next::Return(Returned::Nodes(core::mem::take(&mut header_body.body)))
            }
            HeaderWait::Target(targets) => {
                header_body.body.extend(returned.into_nodes());
                Self::next_target(header_body, targets)
            }
        }
    }

    /// Lift the walk from the next of the header's other targets, or end the
    /// body.
    fn next_target(
        header_body: &mut HeaderBody,
        mut targets: alloc::vec::IntoIter<BlockId>,
    ) -> Next<'cfg> {
        let Some(target) = targets.next() else {
            return Next::Return(Returned::Nodes(core::mem::take(&mut header_body.body)));
        };
        header_body.awaiting = HeaderWait::Target(targets);
        Next::Call(Frame::Region(RegionWalk::new(
            target,
            header_body.bound.clone(),
            header_body.stop,
        )))
    }
}
