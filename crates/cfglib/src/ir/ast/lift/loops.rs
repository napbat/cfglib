//! Natural-loop lifting: shape classification, bodies, and follows.
//!
//! A loop's body is lifted inside the natural-loop block set, so exits
//! resolve to `Break` / `Continue` against the loop stack instead of being
//! absorbed into the body — the follow block is returned to the caller as
//! the loop's sequential continuation.

extern crate alloc;
use alloc::collections::BTreeSet;
use alloc::vec;
use alloc::vec::Vec;

use super::super::node::{AstNode, LoopKind};
use super::{
    LiftState, LoopContext, advance_merge, block_is_allowed, block_label_name, flow_successors,
    has_edge_kind, is_back_edge, is_exception_edge, lift_conditional, lift_region, lift_switch,
    map_block, push_block,
};
use crate::block::BlockId;
use crate::cfg::Cfg;
use crate::edge::EdgeKind;
use crate::graph::structure::NaturalLoop;

/// The two conditional-branch targets of a block, when it ends in one.
struct ConditionalTargets {
    true_target: BlockId,
    false_target: BlockId,
}

fn conditional_targets<I, E>(cfg: &Cfg<I, E>, block: BlockId) -> Option<ConditionalTargets> {
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
fn while_chain<I, E>(cfg: &Cfg<I, E>, natural: &NaturalLoop, header: BlockId) -> Option<LoopShape> {
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
        if next == header || !natural.body.contains(&next) || chain.contains(&next) {
            return None;
        }
        let sequential_predecessors = cfg
            .incoming(next)
            .filter(|&edge| !is_exception_edge(cfg.edge(edge).kind()))
            .count();
        if sequential_predecessors != 1 {
            return None;
        }
        chain.push(next);
        current = next;
    }
}

fn classify_loop<I, E>(cfg: &Cfg<I, E>, natural: &NaturalLoop, header: BlockId) -> LoopShape {
    if let Some(shape) = while_chain(cfg, natural, header) {
        return shape;
    }
    if natural.latches.len() == 1 {
        let latch = *natural
            .latches
            .iter()
            .next()
            .expect("a one-element set has a first element");
        if latch != header {
            if let Some(targets) = conditional_targets(cfg, latch) {
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
    }
    LoopShape::Endless
}

/// The deterministic follow of a loop with no recognized test: the first
/// sequential exit target scanning body blocks in identity order, preferring
/// targets inside the enclosing bound.
fn endless_follow<I, E>(
    cfg: &Cfg<I, E>,
    natural: &NaturalLoop,
    allowed_blocks: Option<&BTreeSet<BlockId>>,
) -> Option<BlockId> {
    let mut fallback = None;
    for &block in &natural.body {
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

/// Lift the loop anchored at `header` (already visited by the caller) and
/// return the lifted node with the loop's sequential continuation.
pub(super) fn lift_loop<'a, I, E, O>(
    cfg: &'a Cfg<I, E>,
    state: &mut LiftState<'_>,
    header: BlockId,
    allowed_blocks: Option<&BTreeSet<BlockId>>,
    map: &mut impl FnMut(&'a I) -> O,
) -> (AstNode<O>, Option<BlockId>) {
    let fallback;
    let natural = if let Some(natural) = state.loops.get(&header.raw()) {
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

    let shape = classify_loop(cfg, natural, header);
    let (follow, continue_target) = match &shape {
        LoopShape::While { exit, .. } | LoopShape::SelfLoop { exit, .. } => (Some(*exit), header),
        LoopShape::DoWhile { latch, exit, .. } => (Some(*exit), *latch),
        LoopShape::Endless => (endless_follow(cfg, natural, allowed_blocks), header),
    };

    // A multi-block condition chain belongs to the loop's test, not its
    // body or the completeness sweep.
    if let LoopShape::While { chain, .. } = &shape {
        for &block in &chain[1..] {
            state.visit(block);
        }
    }

    state.loop_stack.push(LoopContext {
        header: header.raw(),
        follow: follow.map(BlockId::raw),
        continue_target: continue_target.raw(),
        labeled: false,
    });

    let (kind, mut body) = lift_shape(cfg, state, header, &bound, shape, map);

    // A trailing plain `continue` at the structural end of the body is the
    // loop's own back edge — implicit in the representation.
    if matches!(body.last(), Some(AstNode::Continue { label: None })) {
        body.pop();
    }

    let context = state
        .loop_stack
        .pop()
        .expect("the loop context pushed above is still on the stack");
    let node = AstNode::Loop { header, kind, body };
    let node = if context.labeled {
        state.labeled_blocks.insert(header.raw());
        AstNode::Label {
            name: block_label_name(cfg, header),
            body: vec![node],
        }
    } else {
        node
    };
    (node, follow)
}

/// Lift the loop's condition witness and body per its recognized shape.
fn lift_shape<'a, I, E, O>(
    cfg: &'a Cfg<I, E>,
    state: &mut LiftState<'_>,
    header: BlockId,
    bound: &BTreeSet<BlockId>,
    shape: LoopShape,
    map: &mut impl FnMut(&'a I) -> O,
) -> (LoopKind<O>, Vec<AstNode<O>>) {
    match shape {
        LoopShape::While {
            chain,
            body_start,
            exit_on_true,
            ..
        } => {
            let condition = chain
                .iter()
                .flat_map(|&block| cfg.block(block).instructions())
                .map(&mut *map)
                .collect();
            let condition_block = *chain.last().expect("a chain begins at its header");
            let body = if body_start == header {
                // The conditional's in-loop arm re-enters the chain: the
                // loop is its own condition.
                Vec::new()
            } else {
                lift_region(cfg, state, body_start, Some(bound), None, map)
            };
            (
                LoopKind::While {
                    condition_block,
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
                latch: header,
                condition: map_block(cfg, header, map),
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
            state.visit(latch);
            let body = lift_header_body(cfg, state, header, bound, Some(latch), map);
            (
                LoopKind::DoWhile {
                    latch,
                    condition: map_block(cfg, latch, map),
                    continue_on_true,
                },
                body,
            )
        }
        LoopShape::Endless => (
            LoopKind::Endless,
            lift_header_body(cfg, state, header, bound, None, map),
        ),
    }
}

/// Lift a loop body whose header's own instructions belong inside it (the
/// post-tested and endless shapes): emit the header's structure manually —
/// it is already visited — then continue the ordinary walk.
fn lift_header_body<'a, I, E, O>(
    cfg: &'a Cfg<I, E>,
    state: &mut LiftState<'_>,
    header: BlockId,
    bound: &BTreeSet<BlockId>,
    stop: Option<BlockId>,
    map: &mut impl FnMut(&'a I) -> O,
) -> Vec<AstNode<O>> {
    let mut body = Vec::new();
    let outgoing: Vec<crate::EdgeId> = cfg.outgoing(header).collect();
    let is_conditional = has_edge_kind(cfg, &outgoing, EdgeKind::ConditionalTrue)
        && has_edge_kind(cfg, &outgoing, EdgeKind::ConditionalFalse);
    let has_switch = has_edge_kind(cfg, &outgoing, EdgeKind::SwitchCase);

    if is_conditional {
        let node = lift_conditional(cfg, state, header, Some(bound), map);
        body.push(node);
        if let Some(merge) = advance_merge(cfg, state, header, Some(bound)) {
            body.extend(lift_region(cfg, state, merge, Some(bound), stop, map));
        }
    } else if has_switch {
        let node = lift_switch(cfg, state, header, Some(bound), map);
        body.push(node);
        if let Some(merge) = advance_merge(cfg, state, header, Some(bound)) {
            body.extend(lift_region(cfg, state, merge, Some(bound), stop, map));
        }
    } else {
        push_block(&mut body, cfg, header, map);
        for eid in outgoing {
            let edge = cfg.edge(eid);
            if !is_back_edge(cfg, state.back_edges, eid) && !is_exception_edge(edge.kind()) {
                body.extend(lift_region(
                    cfg,
                    state,
                    edge.target(),
                    Some(bound),
                    stop,
                    map,
                ));
            }
        }
    }
    body
}
