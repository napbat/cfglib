//! CFG → AST lifting algorithm.
//!
//! Uses the dominator and post-dominator trees, dominance-proven natural
//! loops, and edge classifications to reconstruct structured control flow
//! from a [`Cfg`].

extern crate alloc;
use alloc::collections::{BTreeMap, BTreeSet};
use alloc::vec::Vec;

use super::node::{AstNode, SwitchCase};
use super::report::{GotoDiagnostic, GotoReason, LiftReport};
use crate::block::BlockId;
use crate::cfg::Cfg;
use crate::edge::EdgeKind;
use crate::graph::dominator::DominatorTree;
use crate::graph::structure::{NaturalLoop, detect_loops_tagged};
use crate::region::Region;

mod labels;
mod loops;
mod predicated;
mod try_catch;

pub use predicated::lift_predicated;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BlockFlowKind {
    LoopHeader,
    Conditional,
    Switch,
    BackEdge,
    Jump,
    Linear,
}

/// One entered loop, innermost last, resolving break/continue transfers.
struct LoopContext {
    /// The loop header block.
    header: u32,
    /// Block a `break` transfers to, when the loop has a recognized follow.
    follow: Option<u32>,
    /// Block a `continue` transfers to: the header, or a post-tested loop's
    /// condition latch.
    continue_target: u32,
    /// Set when a labeled break or continue named this loop.
    labeled: bool,
}

struct LiftState<'a> {
    pdom: &'a DominatorTree,
    /// Dominance-proven and explicitly tagged back-edge endpoint pairs.
    ///
    /// Machine frontends normally describe every native jump as
    /// [`EdgeKind::Jump`]. Keeping the structural result separately lets the
    /// AST lifter recover those loops without requiring frontends to mutate
    /// the lossless native edge classification.
    back_edges: &'a BTreeSet<(u32, u32)>,
    /// Natural loops keyed by header, honoring tagged back edges.
    loops: &'a BTreeMap<u32, NaturalLoop>,
    /// Region anchor (first protected block in reverse postorder) → the
    /// regions anchored there, outermost (largest protected set) first.
    anchors: &'a BTreeMap<u32, Vec<&'a Region>>,
    visited: Vec<bool>,
    /// Blocks targeted by an emitted [`AstNode::Goto`]; the label post-pass
    /// wraps their eventual emission in the matching label.
    goto_targets: BTreeSet<u32>,
    /// Blocks whose label was already emitted (swept sequences, labeled
    /// loops), so the post-pass does not label them twice.
    labeled_blocks: BTreeSet<u32>,
    /// Enclosing loops, innermost last.
    loop_stack: Vec<LoopContext>,
    /// Regions structured as [`AstNode::TryCatch`].
    structured_regions: BTreeSet<u32>,
    report: LiftReport,
}

impl LiftState<'_> {
    fn is_visited(&self, block: BlockId) -> bool {
        self.visited[block.index()]
    }

    fn visit(&mut self, block: BlockId) {
        self.visited[block.index()] = true;
    }
}

fn has_edge_kind<I, E>(cfg: &Cfg<I, E>, edges: &[crate::EdgeId], kind: EdgeKind) -> bool {
    edges.iter().any(|&edge| cfg.edge(edge).kind() == kind)
}

fn is_back_edge<I, E>(
    cfg: &Cfg<I, E>,
    back_edges: &BTreeSet<(u32, u32)>,
    edge: crate::EdgeId,
) -> bool {
    let edge = cfg.edge(edge);
    back_edges.contains(&(edge.source().raw(), edge.target().raw()))
}

/// Whether an edge transfers control exceptionally rather than sequentially.
fn is_exception_edge(kind: EdgeKind) -> bool {
    kind.is_exceptional()
}

/// Successors reached by sequential control flow, excluding exception edges.
fn flow_successors<I, E>(cfg: &Cfg<I, E>, block: BlockId) -> Vec<BlockId> {
    cfg.outgoing(block)
        .filter(|&edge| !is_exception_edge(cfg.edge(edge).kind()))
        .map(|edge| cfg.edge(edge).target())
        .collect()
}

fn classify_block<I, E>(
    cfg: &Cfg<I, E>,
    back_edges: &BTreeSet<(u32, u32)>,
    block: BlockId,
) -> BlockFlowKind {
    let successors: Vec<crate::EdgeId> = cfg.outgoing(block).collect();
    let predecessors: Vec<crate::EdgeId> = cfg.incoming(block).collect();
    if predecessors
        .iter()
        .any(|&edge| is_back_edge(cfg, back_edges, edge))
    {
        BlockFlowKind::LoopHeader
    } else if has_edge_kind(cfg, &successors, EdgeKind::ConditionalTrue)
        && has_edge_kind(cfg, &successors, EdgeKind::ConditionalFalse)
    {
        BlockFlowKind::Conditional
    } else if has_edge_kind(cfg, &successors, EdgeKind::SwitchCase) {
        BlockFlowKind::Switch
    } else if successors
        .iter()
        .any(|&edge| is_back_edge(cfg, back_edges, edge))
    {
        BlockFlowKind::BackEdge
    } else if has_edge_kind(cfg, &successors, EdgeKind::Jump) {
        BlockFlowKind::Jump
    } else {
        BlockFlowKind::Linear
    }
}

fn map_block<'a, I, E, O>(
    cfg: &'a Cfg<I, E>,
    block: BlockId,
    map: &mut impl FnMut(&'a I) -> O,
) -> Vec<O> {
    cfg.block(block).instructions().iter().map(map).collect()
}

fn push_block<'a, I, E, O>(
    result: &mut Vec<AstNode<O>>,
    cfg: &'a Cfg<I, E>,
    block: BlockId,
    map: &mut impl FnMut(&'a I) -> O,
) {
    let instructions = map_block(cfg, block, map);
    if !instructions.is_empty() {
        result.push(AstNode::Block {
            id: block,
            instructions,
        });
    }
}

fn block_is_allowed(allowed_blocks: Option<&BTreeSet<BlockId>>, block: BlockId) -> bool {
    allowed_blocks.is_none_or(|blocks| blocks.contains(&block))
}

/// Produce a label name for a block (used in Goto/Label nodes).
fn block_label_name<I, E>(cfg: &Cfg<I, E>, id: BlockId) -> alloc::string::String {
    cfg.block(id).label().map_or_else(
        || alloc::format!(".bb{}", id.raw()),
        alloc::string::String::from,
    )
}

/// Lift a [`Cfg`] into a structured [`AstNode`] tree.
///
/// Convenience wrapper over [`lift_with_report`] that discards the
/// structural fidelity report.
#[must_use]
pub fn lift<I: Clone, E>(cfg: &Cfg<I, E>) -> AstNode<I> {
    lift_with_report(cfg).0
}

/// Borrows a [`Cfg`]'s instruction payloads into a structured [`AstNode`]
/// tree without cloning them.
///
/// The returned tree owns only its control structure and vectors of
/// references. It therefore cannot outlive `cfg`; use [`lift`] when the tree
/// must be stored independently.
#[must_use]
pub fn lift_borrowed<I, E>(cfg: &Cfg<I, E>) -> AstNode<&I> {
    lift_borrowed_with_report(cfg).0
}

/// Lift a [`Cfg`] into a structured [`AstNode`] tree and report exactly
/// which parts degraded to unstructured flow.
///
/// The instruction type `I` must implement `Clone` so that instructions
/// can be copied into the AST nodes.
///
/// The lifter handles:
/// - Structured flow: `IfThenElse`, `Loop` (classified pre-tested,
///   post-tested, or endless through [`LoopKind`](super::LoopKind)),
///   `Switch` with an explicit default arm and case edge identities
/// - Loop transfers: `Break` / `Continue` derived from natural-loop
///   membership, labeled when they cross an enclosing loop
/// - Exception regions: `TryCatch` when every handler has a complete
///   [`HandlerBody`](crate::HandlerBody); multiple `Finally` handlers
///   concatenate in declaration order
/// - Unstructured flow: `Label` / `Goto` (for `Jump` edges no loop context
///   absorbs, transfers that leave a structural bound, and transfers into
///   already-emitted code)
///
/// Regions with an unknown handler extent remain ordinary control flow; the
/// lifter does not guess which blocks belong inside a handler. **No
/// reachable code is ever dropped**: a transfer refused at a structural
/// boundary becomes an explicit [`AstNode::Goto`], and every reachable
/// block the structured walk did not emit (refused targets, handler pads
/// entered only through exception edges, filter funclets) is appended at
/// the end under its label. The returned [`LiftReport`] lists every emitted
/// goto, swept block, and unstructured region so consumers can degrade per
/// construct instead of guessing from the tree shape.
///
/// # Panics
///
/// Panics (debug) when a [`HandlerBody::Known`](crate::HandlerBody::Known)
/// omits its own handler entry — a frontend bug; release builds treat such
/// a region as unstructured flow.
#[must_use]
pub fn lift_with_report<I: Clone, E>(cfg: &Cfg<I, E>) -> (AstNode<I>, LiftReport) {
    lift_with_report_by(cfg, &mut Clone::clone)
}

/// Borrows a [`Cfg`]'s instruction payloads into a structured [`AstNode`]
/// tree and reports every construct that degraded to unstructured flow.
///
/// Unlike [`lift_with_report`], this variant performs no instruction clone;
/// the returned tree borrows every payload from `cfg`.
#[must_use]
pub fn lift_borrowed_with_report<I, E>(cfg: &Cfg<I, E>) -> (AstNode<&I>, LiftReport) {
    lift_with_report_by(cfg, &mut |instruction| instruction)
}

fn lift_with_report_by<'a, I, E, O>(
    cfg: &'a Cfg<I, E>,
    map: &mut impl FnMut(&'a I) -> O,
) -> (AstNode<O>, LiftReport) {
    let dom = DominatorTree::compute(cfg);
    let pdom = DominatorTree::compute_post(cfg);
    let natural_loops: BTreeMap<u32, NaturalLoop> = detect_loops_tagged(cfg, &dom)
        .into_iter()
        .map(|natural| (natural.header.raw(), natural))
        .collect();
    let back_edges = natural_loops
        .values()
        .flat_map(|natural| {
            natural
                .latches
                .iter()
                .map(|latch| (latch.raw(), natural.header.raw()))
        })
        .collect();
    let order = cfg.reverse_postorder();
    let anchors = try_catch::region_anchors(cfg, &order);
    let mut state = LiftState {
        pdom: &pdom,
        back_edges: &back_edges,
        loops: &natural_loops,
        anchors: &anchors,
        visited: alloc::vec![false; cfg.block_bound()],
        goto_targets: BTreeSet::new(),
        labeled_blocks: BTreeSet::new(),
        loop_stack: Vec::new(),
        structured_regions: BTreeSet::new(),
        report: LiftReport::default(),
    };
    let mut body = lift_region(cfg, &mut state, cfg.entry(), None, None, map);

    // Completeness sweep: emit every reachable block the structured walk
    // refused or never reached, each under its label so the boundary
    // `Goto`s and exception edges that lead there stay resolvable. Region
    // metadata blocks (handler entries, filter funclets) are swept too —
    // a funclet is invoked by the runtime, so it can carry code without
    // any incoming CFG edge.
    let metadata_roots = try_catch::metadata_roots(cfg);
    while let Some(&pending) = order
        .iter()
        .chain(metadata_roots.iter())
        .find(|block| !state.is_visited(**block))
    {
        let pending_body = lift_region(cfg, &mut state, pending, None, None, map);
        if !pending_body.is_empty() {
            state.labeled_blocks.insert(pending.raw());
            state.report.swept_blocks.push(pending);
            body.push(AstNode::Label {
                name: block_label_name(cfg, pending),
                body: pending_body,
            });
        }
    }

    // Exact labeling: wrap the emission of every goto target that has no
    // label yet; whatever cannot be anchored is a dangling goto worth
    // reporting.
    let mut pending_labels: BTreeSet<u32> = state
        .goto_targets
        .difference(&state.labeled_blocks)
        .copied()
        .collect();
    let body = labels::apply_labels(cfg, body, &mut pending_labels);
    state
        .report
        .unresolved_labels
        .extend(pending_labels.into_iter().map(BlockId::from_raw));
    for region in cfg.regions() {
        // Inert tombstones (removed regions) carry nothing to structure.
        if region.protected_blocks.is_empty() && region.handlers.is_empty() {
            continue;
        }
        if !state.structured_regions.contains(&region.id.0) {
            state.report.unstructured_regions.push(region.id);
        }
    }

    let ast = AstNode::Sequence { body };
    (ast.simplify(), state.report)
}

/// Recursively lift a region starting at `head`.
///
/// The walk ends silently at `stop` (a legitimate structural convergence
/// point — an arm's merge, a post-tested loop's latch); every other
/// transfer that cannot proceed is resolved against the loop stack or made
/// explicit as a goto.
fn lift_region<'a, I, E, O>(
    cfg: &'a Cfg<I, E>,
    state: &mut LiftState<'_>,
    head: BlockId,
    allowed_blocks: Option<&BTreeSet<BlockId>>,
    stop: Option<BlockId>,
    map: &mut impl FnMut(&'a I) -> O,
) -> Vec<AstNode<O>> {
    let mut result = Vec::new();
    let mut current = Some(head);

    while let Some(block) = current {
        if Some(block) == stop {
            break;
        }
        if let Some(node) = resolve_loop_transfer(cfg, state, block) {
            result.push(node);
            break;
        }
        if state.is_visited(block) {
            // Convergence is only silent at `stop`; any other transfer into
            // already-emitted code is a real control transfer.
            push_goto(cfg, state, &mut result, block, GotoReason::RevisitedTarget);
            break;
        }
        if !block_is_allowed(allowed_blocks, block) {
            // Exclusively owned out-of-bound code — a single predecessor,
            // so this transfer is its only entry, and no exception region
            // claims it — inlines here instead of degrading to a goto
            // (javac hoists `break label` bodies out of line like this).
            // Anything else is recorded, never dropped: the jump is
            // explicit and the completeness sweep emits the target under
            // the matching label.
            if cfg.incoming(block).count() == 1 && !region_member(cfg, block) {
                result.extend(lift_region(cfg, state, block, None, stop, map));
            } else {
                push_goto(cfg, state, &mut result, block, GotoReason::BoundaryEscape);
            }
            break;
        }

        state.visit(block);

        if let Some((node, continuation)) =
            try_catch::lift_try_catch(cfg, state, block, allowed_blocks, map)
        {
            result.push(node);
            // Resume at the region's own continuation, not the anchor's
            // merge — they differ whenever the try exits through an
            // explicit leave.
            current = continuation.filter(|&next| !state.is_visited(next));
            continue;
        }

        current = lift_block(cfg, state, block, allowed_blocks, stop, &mut result, map);
    }

    result
}

/// Classify one already-visited block, emit its lifted form, and return the
/// block sequential flow continues at (if any).
///
/// This is the single classification path: [`lift_region`]'s walk, the
/// try-body anchor, and switch case arms all run it, so a region anchored
/// at a branch, dispatch, or loop header keeps its structure inside the
/// enclosing construct.
fn lift_block<'a, I, E, O>(
    cfg: &'a Cfg<I, E>,
    state: &mut LiftState<'_>,
    block: BlockId,
    allowed_blocks: Option<&BTreeSet<BlockId>>,
    stop: Option<BlockId>,
    result: &mut Vec<AstNode<O>>,
    map: &mut impl FnMut(&'a I) -> O,
) -> Option<BlockId> {
    let outgoing: Vec<crate::EdgeId> = cfg.outgoing(block).collect();
    let flow = classify_block(cfg, state.back_edges, block);

    if flow == BlockFlowKind::LoopHeader {
        let (node, follow) = loops::lift_loop(cfg, state, block, allowed_blocks, map);
        result.push(node);
        return follow;
    }

    if flow == BlockFlowKind::Conditional {
        let node = lift_conditional(cfg, state, block, allowed_blocks, map);
        result.push(node);
        return advance_merge(cfg, state, block, allowed_blocks);
    }

    if flow == BlockFlowKind::Switch {
        let node = lift_switch(cfg, state, block, allowed_blocks, map);
        result.push(node);
        return advance_merge(cfg, state, block, allowed_blocks);
    }

    if flow == BlockFlowKind::BackEdge {
        push_block(result, cfg, block, map);
        let target = outgoing
            .iter()
            .find(|&&edge| is_back_edge(cfg, state.back_edges, edge))
            .map(|&edge| cfg.edge(edge).target());
        if let Some(target) = target {
            if let Some(node) = resolve_loop_transfer(cfg, state, target) {
                result.push(node);
            } else {
                push_goto(cfg, state, result, target, GotoReason::RevisitedTarget);
            }
        }
        return None;
    }

    if flow == BlockFlowKind::Jump {
        push_block(result, cfg, block, map);
        for eid in outgoing {
            let edge = cfg.edge(eid);
            if edge.kind() == EdgeKind::Jump {
                let target = edge.target();
                if Some(target) == stop {
                    // The jump lands exactly on the enclosing construct's
                    // convergence point (an arm skipping its sibling to
                    // reach the join): structurally the arm simply ends.
                    continue;
                }
                if let Some(node) = resolve_loop_transfer(cfg, state, target) {
                    result.push(node);
                } else if !state.is_visited(target)
                    && cfg.incoming(target).count() == 1
                    && !region_member(cfg, target)
                {
                    // Exclusively owned jump target — this explicit jump
                    // is its only entry and no exception region claims
                    // it (javac jumps over handler code to reach a
                    // join): sequential flow simply continues there.
                    return Some(target);
                } else {
                    push_goto(cfg, state, result, target, GotoReason::ExplicitJump);
                }
            }
        }
        return None;
    }

    if outgoing.is_empty() {
        let insts = map_block(cfg, block, map);
        if !insts.is_empty() {
            result.push(AstNode::Return {
                id: block,
                instructions: insts,
            });
        }
        return None;
    }

    push_block(result, cfg, block, map);

    // Advance along the single sequential successor; exception edges are
    // not sequential flow, so they never block the advance. The walk in
    // [`lift_region`] resolves the target against the stop block, the loop
    // stack, and the enclosing bound.
    let flow_succs = flow_successors(cfg, block);
    if flow_succs.len() == 1 {
        return Some(flow_succs[0]);
    }
    None
}

/// The ultimate target of a chain of pure jump trampolines: an unemitted,
/// instruction-free block with a single unconditional successor forwards
/// a resolution unchanged. The HLIL lift empties dialect-declared pure
/// transfers in its working view before structuring (javac's `break`
/// routes through such blocks); frontends structuring a raw [`Cfg`] whose
/// jump encodings occupy an instruction clear them the same way.
fn through_trampolines<I, E>(
    cfg: &Cfg<I, E>,
    state: &LiftState<'_>,
    target: BlockId,
) -> (BlockId, Vec<BlockId>) {
    let mut hops = Vec::new();
    let mut current = target;
    for _ in 0..8 {
        if state.is_visited(current) || !cfg.block(current).instructions().is_empty() {
            break;
        }
        let mut normal = cfg
            .outgoing(current)
            .map(|edge| cfg.edge(edge))
            .filter(|edge| !is_exception_edge(edge.kind()));
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

/// Resolve a control transfer against the enclosing loops: `continue` when
/// it reaches a loop's continue point, `break` when it reaches a loop's
/// follow. Inner loops win; resolving against an outer loop labels it.
fn resolve_loop_transfer<I, E, O>(
    cfg: &Cfg<I, E>,
    state: &mut LiftState<'_>,
    target: BlockId,
) -> Option<AstNode<O>> {
    let (target, hops) = through_trampolines(cfg, state, target);
    let position = state.loop_stack.iter().rposition(|context| {
        context.continue_target == target.raw() || context.follow == Some(target.raw())
    })?;
    // The consumed trampolines carry no content; the sweep must not
    // resurrect them as labeled residue.
    for hop in hops {
        state.visit(hop);
    }
    let innermost = position + 1 == state.loop_stack.len();
    let label = if innermost {
        None
    } else {
        state.loop_stack[position].labeled = true;
        Some(block_label_name(
            cfg,
            BlockId::from_raw(state.loop_stack[position].header),
        ))
    };
    let context = &state.loop_stack[position];
    Some(if context.continue_target == target.raw() {
        AstNode::Continue { label }
    } else {
        AstNode::Break { label }
    })
}

/// Whether any exception region names the block — as protected code, a
/// handler entry, or declared handler extent. Region-claimed code never
/// moves across structural boundaries.
fn region_member<I, E>(cfg: &Cfg<I, E>, block: BlockId) -> bool {
    cfg.regions().iter().any(|region| {
        region.protected_blocks.contains(&block)
            || region.handlers.iter().any(|handler| {
                handler.entry == block
                    || handler
                        .body
                        .blocks()
                        .is_some_and(|blocks| blocks.contains(&block))
            })
    })
}

fn push_goto<I, E, O>(
    cfg: &Cfg<I, E>,
    state: &mut LiftState<'_>,
    result: &mut Vec<AstNode<O>>,
    target: BlockId,
    reason: GotoReason,
) {
    state.goto_targets.insert(target.raw());
    state.report.gotos.push(GotoDiagnostic { target, reason });
    result.push(AstNode::Goto {
        target: block_label_name(cfg, target),
    });
}

/// The post-dominator merge of `block`, kept only when it stays inside the
/// enclosing structural bound.
fn effective_merge<I, E>(
    _cfg: &Cfg<I, E>,
    state: &LiftState<'_>,
    block: BlockId,
    allowed_blocks: Option<&BTreeSet<BlockId>>,
) -> Option<BlockId> {
    state
        .pdom
        .idom(block)
        .filter(|merge| block_is_allowed(allowed_blocks, *merge))
}

/// Get the in-bound post-dominator merge point if it hasn't been visited yet.
fn advance_merge<I, E>(
    cfg: &Cfg<I, E>,
    state: &LiftState<'_>,
    block: BlockId,
    allowed_blocks: Option<&BTreeSet<BlockId>>,
) -> Option<BlockId> {
    effective_merge(cfg, state, block, allowed_blocks).filter(|&merge| !state.is_visited(merge))
}

/// Lift an if/else conditional starting at `block`.
fn lift_conditional<'a, I, E, O>(
    cfg: &'a Cfg<I, E>,
    state: &mut LiftState<'_>,
    block: BlockId,
    allowed_blocks: Option<&BTreeSet<BlockId>>,
    map: &mut impl FnMut(&'a I) -> O,
) -> AstNode<O> {
    let mut true_target = None;
    let mut false_target = None;
    for eid in cfg.outgoing(block) {
        match cfg.edge(eid).kind() {
            EdgeKind::ConditionalTrue => true_target = Some(cfg.edge(eid).target()),
            EdgeKind::ConditionalFalse => false_target = Some(cfg.edge(eid).target()),
            _ => {}
        }
    }

    let merge = effective_merge(cfg, state, block, allowed_blocks);

    let then_body = true_target
        .map(|target| lift_region(cfg, state, target, allowed_blocks, merge, map))
        .unwrap_or_default();
    let else_body = false_target
        .map(|target| lift_region(cfg, state, target, allowed_blocks, merge, map))
        .unwrap_or_default();

    AstNode::IfThenElse {
        condition: block,
        condition_instructions: map_block(cfg, block, map),
        then_body,
        else_body,
    }
}

/// Lift a switch starting at `block`.
fn lift_switch<'a, I, E, O>(
    cfg: &'a Cfg<I, E>,
    state: &mut LiftState<'_>,
    block: BlockId,
    allowed_blocks: Option<&BTreeSet<BlockId>>,
    map: &mut impl FnMut(&'a I) -> O,
) -> AstNode<O> {
    let merge = effective_merge(cfg, state, block, allowed_blocks);

    // Group dispatch edges by target in first-encounter order, and find the
    // explicit default edge (the sequential non-case successor).
    let mut case_targets: Vec<BlockId> = Vec::new();
    let mut case_edges: BTreeMap<u32, Vec<crate::EdgeId>> = BTreeMap::new();
    let mut default_edge = None;
    let mut default_target = None;
    for eid in cfg.outgoing(block) {
        let edge = cfg.edge(eid);
        match edge.kind() {
            EdgeKind::SwitchCase => {
                let target = edge.target();
                let edges = case_edges.entry(target.raw()).or_default();
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
    // an arm-to-arm transfer stays explicit instead.
    for &target in &case_targets {
        if Some(target) != merge && block_is_allowed(allowed_blocks, target) {
            state.visit(target);
        }
    }
    if let Some(target) = default_target {
        if Some(target) != merge && block_is_allowed(allowed_blocks, target) {
            state.visit(target);
        }
    }

    let mut cases = Vec::new();
    for target in case_targets {
        let edges = case_edges.remove(&target.raw()).unwrap_or_default();
        let body = lift_switch_arm(cfg, state, target, allowed_blocks, merge, map);
        cases.push(SwitchCase {
            id: target,
            edges,
            body,
        });
    }
    let default_body = default_target
        .map(|target| lift_switch_arm(cfg, state, target, allowed_blocks, merge, map))
        .unwrap_or_default();

    AstNode::Switch {
        condition: block,
        condition_instructions: map_block(cfg, block, map),
        cases,
        default_body,
        default_edge,
    }
}

/// Lift one switch arm (a case body or the default body) up to the merge.
fn lift_switch_arm<'a, I, E, O>(
    cfg: &'a Cfg<I, E>,
    state: &mut LiftState<'_>,
    target: BlockId,
    allowed_blocks: Option<&BTreeSet<BlockId>>,
    merge: Option<BlockId>,
    map: &mut impl FnMut(&'a I) -> O,
) -> Vec<AstNode<O>> {
    if Some(target) == merge {
        // The arm transfers straight to the switch continuation.
        return Vec::new();
    }
    let mut body = Vec::new();
    if !block_is_allowed(allowed_blocks, target) {
        push_goto(cfg, state, &mut body, target, GotoReason::BoundaryEscape);
        return body;
    }
    // The arm entry was pre-visited above; classify it through the shared
    // path so branch or loop structure at the arm entry is kept, then
    // continue the ordinary walk toward the merge.
    let next = lift_block(cfg, state, target, allowed_blocks, merge, &mut body, map);
    if let Some(next) = next {
        body.extend(lift_region(cfg, state, next, allowed_blocks, merge, map));
    }
    body
}

#[cfg(test)]
mod tests;
