//! CFG → AST lifting algorithm.
//!
//! Uses the dominator tree and edge classifications to reconstruct
//! structured control flow from a [`Cfg`].

extern crate alloc;
use alloc::collections::BTreeSet;
use alloc::vec::Vec;

use super::node::{AstNode, CatchHandler, SwitchCase};
use crate::block::BlockId;
use crate::cfg::Cfg;
use crate::dataflow::Predicated;
use crate::edge::EdgeKind;
use crate::graph::dominator::DominatorTree;
use crate::region::HandlerKind;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BlockFlowKind {
    LoopHeader,
    Conditional,
    Switch,
    BackEdge,
    Jump,
    Linear,
}

#[derive(Debug, Clone, Copy)]
struct BlockFlow {
    kind: BlockFlowKind,
    needs_label: bool,
}

fn has_edge_kind<I>(cfg: &Cfg<I>, edges: &[crate::EdgeId], kind: EdgeKind) -> bool {
    edges.iter().any(|&edge| cfg.edge(edge).kind() == kind)
}

fn classify_block<I>(cfg: &Cfg<I>, block: BlockId) -> BlockFlow {
    let successors = cfg.successor_edges(block);
    let predecessors = cfg.predecessor_edges(block);
    let kind = if has_edge_kind(cfg, predecessors, EdgeKind::Back) {
        BlockFlowKind::LoopHeader
    } else if has_edge_kind(cfg, successors, EdgeKind::ConditionalTrue)
        && has_edge_kind(cfg, successors, EdgeKind::ConditionalFalse)
    {
        BlockFlowKind::Conditional
    } else if has_edge_kind(cfg, successors, EdgeKind::SwitchCase) {
        BlockFlowKind::Switch
    } else if has_edge_kind(cfg, successors, EdgeKind::Back) {
        BlockFlowKind::BackEdge
    } else if has_edge_kind(cfg, successors, EdgeKind::Jump) {
        BlockFlowKind::Jump
    } else {
        BlockFlowKind::Linear
    };
    BlockFlow {
        kind,
        needs_label: has_edge_kind(cfg, predecessors, EdgeKind::Jump),
    }
}

fn push_block<I: Clone>(result: &mut Vec<AstNode<I>>, cfg: &Cfg<I>, block: BlockId) {
    let instructions = cfg.block(block).instructions().to_vec();
    if !instructions.is_empty() {
        result.push(AstNode::Block {
            id: block,
            instructions,
        });
    }
}

/// Lift a [`Cfg`] into a structured [`AstNode`] tree.
///
/// The instruction type `I` must implement `Clone` so that instructions
/// can be copied into the AST nodes.
///
/// The lifter handles:
/// - Structured flow: `IfThenElse`, `Loop`, `Switch`
/// - Exception regions: `TryCatch` (from [`Cfg::regions`])
/// - Unstructured flow: `Label` / `Goto` (for `Jump` edges)
#[must_use]
pub fn lift<I: Clone>(cfg: &Cfg<I>) -> AstNode<I> {
    let dom = DominatorTree::compute(cfg);
    let pdom = DominatorTree::compute_post(cfg);
    let mut visited = BTreeSet::new();
    // Collect the entry blocks of each region so we know which blocks
    // start a try/catch scope.
    let region_entries: BTreeSet<u32> = cfg
        .regions()
        .iter()
        .filter_map(|r| r.protected_blocks.iter().next())
        .map(|b| b.0)
        .collect();
    let body = lift_region(cfg, &dom, &pdom, cfg.entry(), &mut visited, &region_entries);
    let ast = AstNode::Sequence { body };
    ast.simplify()
}

/// Recursively lift a region starting at `head`.
fn lift_region<I: Clone>(
    cfg: &Cfg<I>,
    dom: &DominatorTree,
    pdom: &DominatorTree,
    head: BlockId,
    visited: &mut BTreeSet<u32>,
    region_entries: &BTreeSet<u32>,
) -> Vec<AstNode<I>> {
    let mut result = Vec::new();
    let mut current = Some(head);

    while let Some(block) = current {
        if visited.contains(&block.0) {
            break;
        }

        visited.insert(block.0);
        current = None;

        // --- TryCatch region ---
        let region_node = if region_entries.contains(&block.0) {
            lift_try_catch(cfg, dom, pdom, block, visited, region_entries)
        } else {
            None
        };
        if let Some(node) = region_node {
            result.push(node);
            current = advance_merge(pdom, block, visited);
            continue;
        }

        let successor_edges = cfg.successor_edges(block);
        let flow = classify_block(cfg, block);
        let needs_label = flow.needs_label;

        // --- Loop header ---
        if flow.kind == BlockFlowKind::LoopHeader {
            let node = lift_loop(cfg, dom, pdom, block, visited, region_entries);
            if needs_label {
                result.push(wrap_label(block, node));
            } else {
                result.push(node);
            }
            current = find_loop_exit(cfg, block, visited);
            continue;
        }

        // --- Conditional (if/else) ---
        if flow.kind == BlockFlowKind::Conditional {
            let node = lift_conditional(cfg, dom, pdom, block, visited, region_entries);
            if needs_label {
                result.push(wrap_label(block, node));
            } else {
                result.push(node);
            }
            current = advance_merge(pdom, block, visited);
            continue;
        }

        // --- Switch ---
        if flow.kind == BlockFlowKind::Switch {
            let node = lift_switch(cfg, dom, pdom, block, visited, region_entries);
            if needs_label {
                result.push(wrap_label(block, node));
            } else {
                result.push(node);
            }
            current = advance_merge(pdom, block, visited);
            continue;
        }

        // --- Back edge (loop latch) ---
        if flow.kind == BlockFlowKind::BackEdge {
            push_block(&mut result, cfg, block);
            result.push(AstNode::Continue);
            continue;
        }

        // --- Jump edge (unstructured goto) ---
        if flow.kind == BlockFlowKind::Jump {
            push_block(&mut result, cfg, block);
            for &eid in successor_edges {
                let edge = cfg.edge(eid);
                if edge.kind() == EdgeKind::Jump {
                    result.push(AstNode::Goto {
                        target: block_label_name(cfg, edge.target()),
                    });
                }
            }
            continue;
        }

        // --- Terminal ---
        if successor_edges.is_empty() {
            let insts = cfg.block(block).instructions().to_vec();
            if !insts.is_empty() {
                result.push(AstNode::Return {
                    id: block,
                    instructions: insts,
                });
            }
            continue;
        }

        // --- Break relay block ---
        // The builder creates empty blocks with a single Unconditional
        // edge for `break` statements. Recognise these and emit Break.
        if cfg.block(block).is_empty()
            && successor_edges.len() == 1
            && cfg.edge(successor_edges[0]).kind() == EdgeKind::Unconditional
        {
            result.push(AstNode::Break);
            continue;
        }

        // --- Fallthrough / unconditional ---
        let block_node = AstNode::Block {
            id: block,
            instructions: cfg.block(block).instructions().to_vec(),
        };
        if needs_label {
            result.push(wrap_label(block, block_node));
        } else {
            result.push(block_node);
        }
        let succs: Vec<BlockId> = cfg.successors(block).collect();
        if succs.len() == 1 && !visited.contains(&succs[0].0) {
            current = Some(succs[0]);
        }
    }

    result
}

/// Lift a [`Cfg`] and regionise predicated instructions into
/// [`AstNode::Guarded`] nodes.
///
/// Runs [`lift`], then wraps every maximal run of instructions sharing the
/// same [`Predicated::predicate`] into a `Guarded` node whose witness is the
/// run's first instruction. Unpredicated instructions stay in plain blocks.
/// Predicated runs that land in a branch/dispatch header
/// (`IfThenElse`/`Switch` `condition_instructions`) are hoisted into
/// guarded segments before the node. Two ledgered limits: a predicate on
/// the branch/dispatch instruction itself stays inline (unrepresentable as
/// a region), and predicated runs inside a
/// [`SwitchCase`]'s `header_instructions` are not regionised (the case
/// structure has no place to hoist them to).
#[must_use]
pub fn lift_predicated<I: Clone + Predicated>(cfg: &Cfg<I>) -> AstNode<I> {
    wrap_predicated(lift(cfg)).simplify()
}

fn wrap_nodes<I: Clone + Predicated>(nodes: Vec<AstNode<I>>) -> Vec<AstNode<I>> {
    nodes.into_iter().map(wrap_predicated).collect()
}

/// Split a block's instructions into guarded segments.
fn wrap_block_runs<I: Clone + Predicated>(id: BlockId, instructions: Vec<I>) -> AstNode<I> {
    let segments = predicate_runs(instructions)
        .into_iter()
        .map(|(predicate, run)| {
            guard_segment(
                predicate.as_ref(),
                AstNode::Block {
                    id,
                    instructions: run,
                },
            )
        })
        .collect();
    sequence_or_single(segments)
}

/// Split a return block's instructions into guarded segments; the final run
/// keeps its `Return` semantics (a predicated final run is a conditional
/// return, e.g. ARM `bxeq lr`).
fn wrap_return_runs<I: Clone + Predicated>(id: BlockId, instructions: Vec<I>) -> AstNode<I> {
    let mut runs = predicate_runs(instructions);
    let last = runs.pop();
    let mut segments: Vec<AstNode<I>> = runs
        .into_iter()
        .map(|(predicate, run)| {
            guard_segment(
                predicate.as_ref(),
                AstNode::Block {
                    id,
                    instructions: run,
                },
            )
        })
        .collect();
    if let Some((predicate, run)) = last {
        segments.push(guard_segment(
            predicate.as_ref(),
            AstNode::Return {
                id,
                instructions: run,
            },
        ));
    }
    sequence_or_single(segments)
}

fn wrap_predicated<I: Clone + Predicated>(node: AstNode<I>) -> AstNode<I> {
    match node {
        AstNode::Block { id, instructions } => wrap_block_runs(id, instructions),
        AstNode::Return { id, instructions } => wrap_return_runs(id, instructions),
        AstNode::Sequence { body } => AstNode::Sequence {
            body: wrap_nodes(body),
        },
        AstNode::IfThenElse {
            condition,
            condition_instructions,
            then_body,
            else_body,
        } => {
            let (prefix, rest) = split_header_runs(condition, condition_instructions);
            with_prefix(
                prefix,
                AstNode::IfThenElse {
                    condition,
                    condition_instructions: rest,
                    then_body: wrap_nodes(then_body),
                    else_body: wrap_nodes(else_body),
                },
            )
        }
        AstNode::Loop { header, body } => AstNode::Loop {
            header,
            body: wrap_nodes(body),
        },
        AstNode::Switch {
            condition,
            condition_instructions,
            cases,
        } => {
            let (prefix, rest) = split_header_runs(condition, condition_instructions);
            with_prefix(
                prefix,
                AstNode::Switch {
                    condition,
                    condition_instructions: rest,
                    cases: cases
                        .into_iter()
                        .map(|case| SwitchCase {
                            id: case.id,
                            header_instructions: case.header_instructions,
                            body: wrap_nodes(case.body),
                        })
                        .collect(),
                },
            )
        }
        AstNode::Label { name, body } => AstNode::Label {
            name,
            body: wrap_nodes(body),
        },
        AstNode::TryCatch {
            try_body,
            handlers,
            finally_body,
        } => AstNode::TryCatch {
            try_body: wrap_nodes(try_body),
            handlers: handlers
                .into_iter()
                .map(|handler| CatchHandler {
                    entry: handler.entry,
                    body: wrap_nodes(handler.body),
                })
                .collect(),
            finally_body: wrap_nodes(finally_body),
        },
        AstNode::Guarded {
            predicate,
            when_true,
            body,
        } => AstNode::Guarded {
            predicate,
            when_true,
            body: wrap_nodes(body),
        },
        leaf @ (AstNode::Break | AstNode::Continue | AstNode::Goto { .. }) => leaf,
    }
}

/// A maximal instruction run sharing one predicate.
type PredicateRun<I> = (
    Option<(<I as crate::dataflow::InstrInfo>::Variable, bool)>,
    Vec<I>,
);

/// Group instructions into maximal runs sharing one predicate.
fn predicate_runs<I: Predicated>(instructions: Vec<I>) -> Vec<PredicateRun<I>> {
    let mut runs: Vec<PredicateRun<I>> = Vec::new();
    for instruction in instructions {
        let predicate = instruction.predicate();
        match runs.last_mut() {
            Some((run_predicate, run)) if *run_predicate == predicate => run.push(instruction),
            _ => runs.push((predicate, alloc::vec![instruction])),
        }
    }
    runs
}

/// Wrap a segment in [`AstNode::Guarded`] when its run is predicated.
fn guard_segment<I: Clone + Predicated>(
    predicate: Option<&(I::Variable, bool)>,
    segment: AstNode<I>,
) -> AstNode<I> {
    match predicate {
        Some((_, when_true)) => {
            let when_true = *when_true;
            let witness = match &segment {
                AstNode::Block { instructions, .. } | AstNode::Return { instructions, .. } => {
                    instructions[0].clone()
                }
                _ => unreachable!("guard_segment only wraps Block/Return segments"),
            };
            AstNode::Guarded {
                predicate: witness,
                when_true,
                body: alloc::vec![segment],
            }
        }
        None => segment,
    }
}

/// Collapse a single-segment vector; wrap several segments in a sequence.
fn sequence_or_single<I>(mut segments: Vec<AstNode<I>>) -> AstNode<I> {
    if segments.len() == 1 {
        segments.pop().expect("single segment")
    } else {
        AstNode::Sequence { body: segments }
    }
}

/// Split a branch/dispatch header's predicated PREFIX runs into guarded
/// segments to hoist before the node. The final run — which contains the
/// branch/dispatch instruction itself — always stays in place, and headers
/// with no predicated prefix pass through untouched.
fn split_header_runs<I: Clone + Predicated>(
    block: BlockId,
    instructions: Vec<I>,
) -> (Vec<AstNode<I>>, Vec<I>) {
    let mut runs = predicate_runs(instructions);
    let Some((_, last)) = runs.pop() else {
        return (Vec::new(), Vec::new());
    };
    if runs.iter().all(|(predicate, _)| predicate.is_none()) {
        // Nothing to regionise: reassemble the original instruction list.
        let mut rest: Vec<I> = runs.into_iter().flat_map(|(_, run)| run).collect();
        rest.extend(last);
        return (Vec::new(), rest);
    }
    let prefix = runs
        .into_iter()
        .map(|(predicate, run)| {
            guard_segment(
                predicate.as_ref(),
                AstNode::Block {
                    id: block,
                    instructions: run,
                },
            )
        })
        .collect();
    (prefix, last)
}

/// Hoist `prefix` segments before `node`, or return `node` unchanged when
/// there is nothing to hoist.
fn with_prefix<I>(prefix: Vec<AstNode<I>>, node: AstNode<I>) -> AstNode<I> {
    if prefix.is_empty() {
        return node;
    }
    let mut body = prefix;
    body.push(node);
    AstNode::Sequence { body }
}

/// Produce a label name for a block (used in Goto/Label nodes).
fn block_label_name<I>(cfg: &Cfg<I>, id: BlockId) -> alloc::string::String {
    cfg.block(id).label().map_or_else(
        || alloc::format!(".bb{}", id.0),
        alloc::string::String::from,
    )
}

/// Wrap a node in a Label node.
fn wrap_label<I>(block: BlockId, inner: AstNode<I>) -> AstNode<I> {
    AstNode::Label {
        name: alloc::format!(".bb{}", block.0),
        body: alloc::vec![inner],
    }
}

/// Lift a try/catch region starting at `block`.
fn lift_try_catch<I: Clone>(
    cfg: &Cfg<I>,
    dom: &DominatorTree,
    pdom: &DominatorTree,
    block: BlockId,
    visited: &mut BTreeSet<u32>,
    region_entries: &BTreeSet<u32>,
) -> Option<AstNode<I>> {
    let region = cfg.protecting_region(block)?;

    // Lift the try body: emit the current block's instructions, then
    // follow successors within the protected region. We do NOT
    // un-visit and re-enter lift_region because that would re-trigger
    // the region_entries check and cause infinite recursion.
    let mut try_body = Vec::new();
    let insts = cfg.block(block).instructions().to_vec();
    if !insts.is_empty() {
        try_body.push(AstNode::Block {
            id: block,
            instructions: insts,
        });
    }
    // Follow successors of the try entry within the protected region.
    for succ in cfg.successors(block) {
        if region.protected_blocks.contains(&succ) && !visited.contains(&succ.0) {
            try_body.extend(lift_region(cfg, dom, pdom, succ, visited, region_entries));
        }
    }

    // Lift handlers.
    let mut handlers = Vec::new();
    let mut finally_body = Vec::new();

    for handler in &region.handlers {
        let body = lift_region(cfg, dom, pdom, handler.entry, visited, region_entries);
        match handler.kind {
            HandlerKind::Finally => {
                finally_body = body;
            }
            _ => {
                handlers.push(CatchHandler {
                    entry: handler.entry,
                    body,
                });
            }
        }
    }

    Some(AstNode::TryCatch {
        try_body,
        handlers,
        finally_body,
    })
}

/// Get the post-dominator merge point if it hasn't been visited yet.
fn advance_merge(pdom: &DominatorTree, block: BlockId, visited: &BTreeSet<u32>) -> Option<BlockId> {
    pdom.idom(block).filter(|m| !visited.contains(&m.0))
}

/// Lift an if/else conditional starting at `block`.
fn lift_conditional<I: Clone>(
    cfg: &Cfg<I>,
    dom: &DominatorTree,
    pdom: &DominatorTree,
    block: BlockId,
    visited: &mut BTreeSet<u32>,
    region_entries: &BTreeSet<u32>,
) -> AstNode<I> {
    let mut true_target = None;
    let mut false_target = None;
    for &eid in cfg.successor_edges(block) {
        match cfg.edge(eid).kind() {
            EdgeKind::ConditionalTrue => true_target = Some(cfg.edge(eid).target()),
            EdgeKind::ConditionalFalse => false_target = Some(cfg.edge(eid).target()),
            _ => {}
        }
    }

    let merge = pdom.idom(block);

    let then_body = match true_target {
        Some(t) if merge.is_none_or(|m| t != m) => {
            lift_arm(cfg, dom, pdom, t, merge, visited, region_entries)
        }
        _ => Vec::new(),
    };
    let else_body = match false_target {
        Some(f) if merge.is_none_or(|m| f != m) => {
            lift_arm(cfg, dom, pdom, f, merge, visited, region_entries)
        }
        _ => Vec::new(),
    };

    AstNode::IfThenElse {
        condition: block,
        condition_instructions: cfg.block(block).instructions().to_vec(),
        then_body,
        else_body,
    }
}

/// Lift a switch starting at `block`.
fn lift_switch<I: Clone>(
    cfg: &Cfg<I>,
    dom: &DominatorTree,
    pdom: &DominatorTree,
    block: BlockId,
    visited: &mut BTreeSet<u32>,
    region_entries: &BTreeSet<u32>,
) -> AstNode<I> {
    let merge = pdom.idom(block);
    let mut cases = Vec::new();

    for &eid in cfg.successor_edges(block) {
        let edge = cfg.edge(eid);
        if edge.kind() == EdgeKind::SwitchCase {
            let cb = edge.target();
            visited.insert(cb.0);
            let header_insts = cfg.block(cb).instructions().to_vec();
            let body = lift_case_body(cfg, dom, pdom, cb, merge, visited, region_entries);
            cases.push(SwitchCase {
                id: cb,
                header_instructions: header_insts,
                body,
            });
        }
    }

    AstNode::Switch {
        condition: block,
        condition_instructions: cfg.block(block).instructions().to_vec(),
        cases,
    }
}

/// Lift a loop starting at `header`.
fn lift_loop<I: Clone>(
    cfg: &Cfg<I>,
    dom: &DominatorTree,
    pdom: &DominatorTree,
    header: BlockId,
    visited: &mut BTreeSet<u32>,
    region_entries: &BTreeSet<u32>,
) -> AstNode<I> {
    let mut body = Vec::new();

    let successor_edges = cfg.successor_edges(header);
    let is_conditional = has_edge_kind(cfg, successor_edges, EdgeKind::ConditionalTrue)
        && has_edge_kind(cfg, successor_edges, EdgeKind::ConditionalFalse);
    let has_switch = has_edge_kind(cfg, successor_edges, EdgeKind::SwitchCase);

    if is_conditional {
        let node = lift_conditional(cfg, dom, pdom, header, visited, region_entries);
        body.push(node);
        if let Some(merge) = pdom
            .idom(header)
            .filter(|merge| !visited.contains(&merge.0))
        {
            body.extend(lift_region(cfg, dom, pdom, merge, visited, region_entries));
        }
    } else if has_switch {
        let node = lift_switch(cfg, dom, pdom, header, visited, region_entries);
        body.push(node);
        if let Some(merge) = pdom
            .idom(header)
            .filter(|merge| !visited.contains(&merge.0))
        {
            body.extend(lift_region(cfg, dom, pdom, merge, visited, region_entries));
        }
    } else {
        let header_insts = cfg.block(header).instructions().to_vec();
        if !header_insts.is_empty() {
            body.push(AstNode::Block {
                id: header,
                instructions: header_insts,
            });
        }
        for &eid in successor_edges {
            let edge = cfg.edge(eid);
            if edge.kind() != EdgeKind::Back && !visited.contains(&edge.target().0) {
                body.extend(lift_region(
                    cfg,
                    dom,
                    pdom,
                    edge.target(),
                    visited,
                    region_entries,
                ));
            }
        }
    }

    AstNode::Loop { header, body }
}

/// Lift an arm (then/else) stopping at the merge point.
fn lift_arm<I: Clone>(
    cfg: &Cfg<I>,
    dom: &DominatorTree,
    pdom: &DominatorTree,
    start: BlockId,
    stop: Option<BlockId>,
    visited: &mut BTreeSet<u32>,
    region_entries: &BTreeSet<u32>,
) -> Vec<AstNode<I>> {
    if stop.is_some_and(|s| s == start) {
        return Vec::new();
    }
    lift_region(cfg, dom, pdom, start, visited, region_entries)
}

/// Lift the body of a switch case from its successors.
fn lift_case_body<I: Clone>(
    cfg: &Cfg<I>,
    dom: &DominatorTree,
    pdom: &DominatorTree,
    case_block: BlockId,
    stop: Option<BlockId>,
    visited: &mut BTreeSet<u32>,
    region_entries: &BTreeSet<u32>,
) -> Vec<AstNode<I>> {
    let mut body = Vec::new();
    for succ in cfg.successors(case_block) {
        if stop.is_none_or(|s| s != succ) && !visited.contains(&succ.0) {
            body.extend(lift_region(cfg, dom, pdom, succ, visited, region_entries));
        }
    }
    body
}

/// Find the exit of a loop (block reachable via break/conditional-break
/// from within the loop body that hasn't been visited yet).
///
/// Only considers edges whose source is inside the loop (visited) and
/// whose target is outside it (not visited), so nested loops don't
/// confuse the search.
///
/// Instead of scanning every edge in the CFG, this only examines the
/// successor edges of visited (in-loop) blocks, making it proportional
/// to the loop body size rather than the entire CFG.
fn find_loop_exit<I>(cfg: &Cfg<I>, header: BlockId, visited: &BTreeSet<u32>) -> Option<BlockId> {
    // First pass: look for exit edges from loop-body blocks (excluding
    // the header, which is checked separately below).
    for &block_raw in visited {
        let block = BlockId(block_raw);
        if block == header {
            continue;
        }
        for &eid in cfg.successor_edges(block) {
            let edge = cfg.edge(eid);
            let is_exit_edge = matches!(
                edge.kind(),
                EdgeKind::Unconditional | EdgeKind::ConditionalTrue | EdgeKind::ConditionalFalse
            );
            if is_exit_edge && !visited.contains(&edge.target().0) {
                return Some(edge.target());
            }
        }
    }
    // Also check edges directly from the header (e.g., conditional break
    // at the header level).
    for &eid in cfg.successor_edges(header) {
        let edge = cfg.edge(eid);
        if !visited.contains(&edge.target().0) && edge.kind() != EdgeKind::Back {
            return Some(edge.target());
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builder::CfgBuilder;
    use crate::flow::FlowEffect;
    use crate::test_util::{MockInst, df_ff, df_pred, ff};
    use alloc::vec;

    #[test]
    fn lift_predicated_regionises_same_predicate_runs() {
        let cfg = CfgBuilder::build(vec![
            df_ff("plain"),
            df_pred("guarded_a", 3, true),
            df_pred("guarded_b", 3, true),
            df_pred("negated", 3, false),
            df_ff("after"),
        ])
        .unwrap();

        let ast = lift_predicated(&cfg);
        let pseudo = ast.to_pseudocode();
        assert!(pseudo.contains("@guarded(guarded_a)"), "{pseudo}");
        assert!(pseudo.contains("@guarded(!negated)"), "{pseudo}");
        // Same-predicate instructions share one region.
        assert_eq!(pseudo.matches("@guarded(").count(), 2, "{pseudo}");
        // Unpredicated instructions stay outside regions.
        assert!(pseudo.starts_with("plain\n"), "{pseudo}");
    }

    /// Helper: build CFG then lift, return pseudocode.
    fn lift_pseudo(insts: Vec<MockInst>) -> alloc::string::String {
        let cfg = CfgBuilder::build(insts).unwrap();
        let ast = lift(&cfg);
        ast.to_pseudocode()
    }

    // ---- Linear / trivial ----

    #[test]
    fn lift_linear() {
        let p = lift_pseudo(vec![
            ff("a"),
            ff("b"),
            ff("c"),
            MockInst(FlowEffect::Return, "ret"),
        ]);
        assert!(p.contains('a'), "should contain instruction a: {p}");
        assert!(p.contains("ret"), "should contain ret: {p}");
        // No control flow keywords.
        assert!(!p.contains("if"), "no if expected: {p}");
        assert!(!p.contains("loop"), "no loop expected: {p}");
    }

    // ---- If/else ----

    #[test]
    fn lift_if_no_else() {
        let p = lift_pseudo(vec![
            ff("a"),
            MockInst(FlowEffect::ConditionalOpen, "if"),
            ff("b"),
            MockInst(FlowEffect::ConditionalClose, "endif"),
            ff("c"),
            MockInst(FlowEffect::Return, "ret"),
        ]);
        assert!(p.contains("if {"), "should have if: {p}");
        assert!(p.contains('b'), "then body should contain b: {p}");
        assert!(p.contains('c'), "post-merge should contain c: {p}");
    }

    #[test]
    fn lift_if_else() {
        let p = lift_pseudo(vec![
            ff("a"),
            MockInst(FlowEffect::ConditionalOpen, "if"),
            ff("then_inst"),
            MockInst(FlowEffect::ConditionalAlternate, "else"),
            ff("else_inst"),
            MockInst(FlowEffect::ConditionalClose, "endif"),
            ff("merge"),
            MockInst(FlowEffect::Return, "ret"),
        ]);
        assert!(p.contains("if {"), "should have if: {p}");
        assert!(p.contains("then_inst"), "then arm: {p}");
        // else arm or merge should appear
        assert!(
            p.contains("else_inst") || p.contains("} else {"),
            "else arm: {p}"
        );
    }

    // ---- Loop ----

    #[test]
    fn lift_simple_loop() {
        let p = lift_pseudo(vec![
            MockInst(FlowEffect::LoopOpen, "loop"),
            ff("body"),
            MockInst(FlowEffect::LoopClose, "endloop"),
            MockInst(FlowEffect::Return, "ret"),
        ]);
        assert!(p.contains("loop {"), "should have loop: {p}");
        assert!(p.contains("body"), "loop body: {p}");
    }

    #[test]
    fn lift_loop_with_break() {
        let p = lift_pseudo(vec![
            MockInst(FlowEffect::LoopOpen, "loop"),
            ff("a"),
            MockInst(FlowEffect::ConditionalBreak, "breakc"),
            ff("b"),
            MockInst(FlowEffect::LoopClose, "endloop"),
            MockInst(FlowEffect::Return, "ret"),
        ]);
        assert!(p.contains("loop {"), "should have loop: {p}");
        // The breakc creates a conditional inside the loop
        assert!(p.contains('a'), "should contain a: {p}");
    }

    // ---- Switch ----

    #[test]
    fn lift_switch() {
        let p = lift_pseudo(vec![
            MockInst(FlowEffect::SwitchOpen, "switch"),
            ff("dispatch"),
            MockInst(FlowEffect::SwitchCase, "case0"),
            ff("arm0"),
            MockInst(FlowEffect::SwitchCase, "case1"),
            ff("arm1"),
            MockInst(FlowEffect::SwitchClose, "endswitch"),
            ff("after"),
            MockInst(FlowEffect::Return, "ret"),
        ]);
        assert!(p.contains("switch {"), "should have switch: {p}");
        assert!(p.contains("case {"), "should have case: {p}");
    }

    // ---- Nested structures ----

    #[test]
    fn lift_if_in_loop() {
        let p = lift_pseudo(vec![
            MockInst(FlowEffect::LoopOpen, "loop"),
            MockInst(FlowEffect::ConditionalOpen, "if"),
            ff("then"),
            MockInst(FlowEffect::ConditionalAlternate, "else"),
            ff("else_body"),
            MockInst(FlowEffect::ConditionalClose, "endif"),
            MockInst(FlowEffect::LoopClose, "endloop"),
            MockInst(FlowEffect::Return, "ret"),
        ]);
        assert!(p.contains("loop {"), "should have loop: {p}");
        assert!(p.contains("if {"), "should have if inside loop: {p}");
    }

    #[test]
    fn lift_loop_in_if() {
        let p = lift_pseudo(vec![
            MockInst(FlowEffect::ConditionalOpen, "if"),
            MockInst(FlowEffect::LoopOpen, "loop"),
            ff("body"),
            MockInst(FlowEffect::ConditionalBreak, "breakc"),
            MockInst(FlowEffect::LoopClose, "endloop"),
            MockInst(FlowEffect::ConditionalClose, "endif"),
            MockInst(FlowEffect::Return, "ret"),
        ]);
        // Should have both if and loop structures
        let has_if = p.contains("if {");
        let has_loop = p.contains("loop {");
        assert!(has_if || has_loop, "should have nested structure: {p}");
    }

    // ---- AST node structure checks ----

    #[test]
    fn lift_returns_sequence_or_single() {
        let cfg = CfgBuilder::build(vec![ff("a"), MockInst(FlowEffect::Return, "ret")]).unwrap();
        let ast = lift(&cfg);
        // Should be a Block or Return, not an empty Sequence.
        assert!(!ast.is_empty(), "should not be empty");
    }

    #[test]
    fn lift_conditional_produces_if_node() {
        let cfg = CfgBuilder::build(vec![
            ff("a"),
            MockInst(FlowEffect::ConditionalOpen, "if"),
            ff("b"),
            MockInst(FlowEffect::ConditionalClose, "endif"),
            MockInst(FlowEffect::Return, "ret"),
        ])
        .unwrap();
        let ast = lift(&cfg);
        // Walk the AST to find an IfThenElse node.
        let found = has_node_kind(&ast, |n| matches!(n, AstNode::IfThenElse { .. }));
        assert!(found, "should contain IfThenElse node: {ast:?}");
    }

    #[test]
    fn lift_loop_produces_loop_node() {
        let cfg = CfgBuilder::build(vec![
            MockInst(FlowEffect::LoopOpen, "loop"),
            ff("x"),
            MockInst(FlowEffect::LoopClose, "endloop"),
            MockInst(FlowEffect::Return, "ret"),
        ])
        .unwrap();
        let ast = lift(&cfg);
        let found = has_node_kind(&ast, |n| matches!(n, AstNode::Loop { .. }));
        assert!(found, "should contain Loop node: {ast:?}");
    }

    #[test]
    fn lift_switch_produces_switch_node() {
        let cfg = CfgBuilder::build(vec![
            MockInst(FlowEffect::SwitchOpen, "switch"),
            ff("d"),
            MockInst(FlowEffect::SwitchCase, "c1"),
            ff("a1"),
            MockInst(FlowEffect::SwitchCase, "c2"),
            ff("a2"),
            MockInst(FlowEffect::SwitchClose, "endswitch"),
            MockInst(FlowEffect::Return, "ret"),
        ])
        .unwrap();
        let ast = lift(&cfg);
        let found = has_node_kind(&ast, |n| matches!(n, AstNode::Switch { .. }));
        assert!(found, "should contain Switch node: {ast:?}");
    }

    /// Recursively check if any node in the AST matches a predicate.
    fn has_node_kind<I>(node: &AstNode<I>, pred: fn(&AstNode<I>) -> bool) -> bool {
        if pred(node) {
            return true;
        }
        match node {
            AstNode::Sequence { body }
            | AstNode::Loop { body, .. }
            | AstNode::Label { body, .. } => body.iter().any(|c| has_node_kind(c, pred)),
            AstNode::IfThenElse {
                then_body,
                else_body,
                ..
            } => {
                then_body.iter().any(|c| has_node_kind(c, pred))
                    || else_body.iter().any(|c| has_node_kind(c, pred))
            }
            AstNode::Switch { cases, .. } => cases
                .iter()
                .any(|c| c.body.iter().any(|n| has_node_kind(n, pred))),
            AstNode::TryCatch {
                try_body,
                handlers,
                finally_body,
            } => {
                try_body.iter().any(|c| has_node_kind(c, pred))
                    || handlers
                        .iter()
                        .any(|h| h.body.iter().any(|n| has_node_kind(n, pred)))
                    || finally_body.iter().any(|c| has_node_kind(c, pred))
            }
            _ => false,
        }
    }

    // ---- TryCatch lifting ----

    #[test]
    fn lift_try_catch_produces_try_node() {
        use crate::region::{Handler, HandlerKind, Region, RegionId};
        use alloc::collections::BTreeSet;

        let mut cfg: Cfg<MockInst> = Cfg::new();
        // entry(0) → try_body(1) → after(3)
        //            try_body(1) --Exception--> handler(2) → after(3)
        let try_body = cfg.new_block(); // 1
        let handler_block = cfg.new_block(); // 2
        let after = cfg.new_block(); // 3

        cfg.block_mut(cfg.entry())
            .instructions_vec_mut()
            .push(ff("entry"));
        cfg.block_mut(try_body)
            .instructions_vec_mut()
            .push(ff("try_inst"));
        cfg.block_mut(handler_block)
            .instructions_vec_mut()
            .push(ff("catch_inst"));
        cfg.block_mut(after)
            .instructions_vec_mut()
            .push(ff("after"));

        cfg.add_edge(cfg.entry(), try_body, EdgeKind::Fallthrough);
        cfg.add_edge(try_body, after, EdgeKind::Fallthrough);
        cfg.add_edge(try_body, handler_block, EdgeKind::ExceptionHandler);
        cfg.add_edge(handler_block, after, EdgeKind::Fallthrough);

        let mut protected = BTreeSet::new();
        protected.insert(try_body);
        cfg.add_region(Region {
            id: RegionId(0),
            protected_blocks: protected,
            handlers: alloc::vec![Handler {
                entry: handler_block,
                body: {
                    let mut s = BTreeSet::new();
                    s.insert(handler_block);
                    s
                },
                kind: HandlerKind::Catch,
            }],
            parent: None,
        });

        let ast = lift(&cfg);
        let found = has_node_kind(&ast, |n| matches!(n, AstNode::TryCatch { .. }));
        assert!(found, "should contain TryCatch node: {ast:?}");
        let pseudo = ast.to_pseudocode();
        assert!(
            pseudo.contains("try"),
            "pseudocode should contain try: {pseudo}"
        );
    }

    // ---- Goto / Label lifting ----

    #[test]
    fn lift_jump_edge_produces_goto() {
        let mut cfg: Cfg<MockInst> = Cfg::new();
        // entry(0) --Jump--> target(1)
        let target = cfg.new_block();
        cfg.block_mut(cfg.entry())
            .instructions_vec_mut()
            .push(ff("src"));
        cfg.block_mut(target).instructions_vec_mut().push(ff("dst"));

        cfg.add_edge(cfg.entry(), target, EdgeKind::Jump);

        let ast = lift(&cfg);
        let found = has_node_kind(&ast, |n| matches!(n, AstNode::Goto { .. }));
        assert!(found, "should contain Goto node: {ast:?}");
        let pseudo = ast.to_pseudocode();
        assert!(
            pseudo.contains("goto"),
            "pseudocode should contain goto: {pseudo}"
        );
    }

    #[test]
    fn lift_jump_target_gets_label() {
        let mut cfg: Cfg<MockInst> = Cfg::new();
        // entry(0) --ConditionalTrue--> normal(1) --Fallthrough--> target(2) --Fallthrough--> end(3)
        // entry(0) --ConditionalFalse--> jumper(4) --Jump--> target(2)
        // target(2) has a Jump predecessor so it gets a Label wrapper.
        let normal = cfg.new_block(); // 1
        let target = cfg.new_block(); // 2
        let end = cfg.new_block(); // 3
        let jumper = cfg.new_block(); // 4
        cfg.block_mut(cfg.entry())
            .instructions_vec_mut()
            .push(ff("entry"));
        cfg.block_mut(normal)
            .instructions_vec_mut()
            .push(ff("normal"));
        cfg.block_mut(target).instructions_vec_mut().push(ff("dst"));
        cfg.block_mut(end).instructions_vec_mut().push(ff("end"));
        cfg.block_mut(jumper)
            .instructions_vec_mut()
            .push(ff("jumper"));

        cfg.add_edge(cfg.entry(), normal, EdgeKind::ConditionalTrue);
        cfg.add_edge(cfg.entry(), jumper, EdgeKind::ConditionalFalse);
        cfg.add_edge(normal, target, EdgeKind::Fallthrough);
        cfg.add_edge(jumper, target, EdgeKind::Jump);
        cfg.add_edge(target, end, EdgeKind::Fallthrough);

        let ast = lift(&cfg);
        let found = has_node_kind(&ast, |n| matches!(n, AstNode::Label { .. }));
        assert!(found, "should contain Label node: {ast:?}");
    }
}
