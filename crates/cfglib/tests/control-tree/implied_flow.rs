//! The flow a tree implies, compared with the graph's own edges.

use std::collections::{BTreeMap, BTreeSet};

use cfglib::{
    BlockId, Cfg, ControlBody, ControlNode, ControlNodeId, ControlTree, LiftReport, LoopKind,
};

/// The normal flow edges a tree implies, found by giving the tree a
/// continuation-passing reading that never consults the structurer: a body
/// flows into whatever follows it, a branch flows from its condition into both
/// arms, a loop's body flows back to its continue point, `break` flows to the
/// loop's follow, and a `goto` flows to its target.
pub struct ImpliedFlow {
    /// Edges the shape of the tree expresses.
    pub structural: BTreeSet<(BlockId, BlockId)>,
    /// Edges that exist only because a `Goto` node names their target.
    pub through_gotos: BTreeSet<(BlockId, BlockId)>,
}

#[derive(Clone, Copy)]
struct Entry {
    block: Option<BlockId>,
    via_goto: bool,
}

impl Entry {
    const NOWHERE: Self = Self {
        block: None,
        via_goto: false,
    };

    const fn at(block: BlockId) -> Self {
        Self {
            block: Some(block),
            via_goto: false,
        }
    }
}

struct LoopFrame {
    header: BlockId,
    continue_to: Entry,
    break_to: Entry,
}

struct Reader<'tree, T> {
    tree: &'tree ControlTree,
    /// Blocks that end the function: a `Block` leaf of one flows nowhere.
    ends_function: T,
    flow: ImpliedFlow,
    loops: Vec<LoopFrame>,
}

impl<T: Fn(BlockId) -> bool> Reader<'_, T> {
    fn add(&mut self, from: BlockId, to: Entry) {
        if let Some(target) = to.block {
            if to.via_goto {
                self.flow.through_gotos.insert((from, target));
            } else {
                self.flow.structural.insert((from, target));
            }
        }
    }

    /// Read a body whose normal completion continues at `next`; the result is
    /// where the body's execution begins.
    fn body(&mut self, body: ControlBody, next: Entry) -> Entry {
        let mut follow = next;
        for &id in self.tree.body(body).iter().rev() {
            follow = self.node(id, follow);
        }
        follow
    }

    fn frame(&self, label: Option<BlockId>) -> &LoopFrame {
        match label {
            None => self.loops.last(),
            Some(header) => self.loops.iter().rev().find(|frame| frame.header == header),
        }
        .expect("a break or continue names an enclosing loop")
    }

    fn node(&mut self, id: ControlNodeId, next: Entry) -> Entry {
        let tree = self.tree;
        match *tree.node(id) {
            ControlNode::Block(span) => {
                // A protected block that returns still has its exception edge,
                // so it is a `Block` leaf yet continues nowhere.
                if !(self.ends_function)(span.block()) {
                    self.add(span.block(), next);
                }
                Entry::at(span.block())
            }
            ControlNode::Return(span) => Entry::at(span.block()),
            ControlNode::Sequence(body)
            | ControlNode::Label { body, .. }
            | ControlNode::Guarded { body, .. } => self.body(body, next),
            ControlNode::IfThenElse {
                condition,
                then_body,
                else_body,
            } => {
                let test = condition.block();
                let then_entry = self.body(then_body, next);
                let else_entry = self.body(else_body, next);
                self.add(test, then_entry);
                self.add(test, else_entry);
                Entry::at(test)
            }
            ControlNode::Loop { header, kind, body } => self.loop_node(header, kind, body, next),
            ControlNode::Switch {
                condition,
                cases,
                default_body,
                default_edge,
            } => {
                let dispatch = condition.block();
                for case in tree.cases(cases) {
                    let entry = self.body(case.body, next);
                    assert_eq!(
                        entry.block,
                        Some(case.block),
                        "a case arm begins at its case block"
                    );
                    self.add(dispatch, entry);
                }
                if default_edge.is_some() {
                    let entry = self.body(default_body, next);
                    self.add(dispatch, entry);
                }
                Entry::at(dispatch)
            }
            ControlNode::Break { label } => self.frame(label).break_to,
            ControlNode::Continue { label } => self.frame(label).continue_to,
            ControlNode::Goto { target } => Entry {
                block: Some(target),
                via_goto: true,
            },
            ControlNode::TryCatch {
                try_body,
                handlers,
                finally_body,
            } => {
                // The graph enters a handler through exception edges only, and
                // the protected body and each handler end at the continuation.
                self.body(finally_body, next);
                for handler in tree.handlers(handlers) {
                    self.body(handler.body, next);
                }
                self.body(try_body, next)
            }
        }
    }

    fn loop_node(
        &mut self,
        header: BlockId,
        kind: LoopKind,
        body: ControlBody,
        next: Entry,
    ) -> Entry {
        let tree = self.tree;
        match kind {
            LoopKind::While { condition, .. } => {
                let chain: Vec<BlockId> = tree.spans(condition).iter().map(|s| s.block()).collect();
                assert_eq!(
                    chain.first(),
                    Some(&header),
                    "the chain starts at the header"
                );
                self.loops.push(LoopFrame {
                    header,
                    continue_to: Entry::at(header),
                    break_to: next,
                });
                let body_entry = self.body(body, Entry::at(header));
                self.loops.pop();
                for pair in chain.windows(2) {
                    self.add(pair[0], Entry::at(pair[1]));
                }
                let test = *chain.last().expect("a condition chain holds the header");
                self.add(test, body_entry);
                self.add(test, next);
            }
            LoopKind::DoWhile { latch, .. } => {
                let latch = latch.block();
                self.loops.push(LoopFrame {
                    header,
                    continue_to: Entry::at(latch),
                    break_to: next,
                });
                // A single-block loop is its own latch and has no body.
                let body_entry = self.body(body, Entry::at(latch));
                self.loops.pop();
                if !body.is_empty() {
                    assert_eq!(
                        body_entry.block,
                        Some(header),
                        "a body begins at its header"
                    );
                }
                self.add(latch, Entry::at(header));
                self.add(latch, next);
            }
            LoopKind::Endless => {
                self.loops.push(LoopFrame {
                    header,
                    continue_to: Entry::at(header),
                    break_to: next,
                });
                let body_entry = self.body(body, Entry::at(header));
                self.loops.pop();
                assert_eq!(
                    body_entry.block,
                    Some(header),
                    "a body begins at its header"
                );
            }
        }
        Entry::at(header)
    }
}

/// Reconstruct the normal edges a tree implies. `ends_function` names the blocks
/// with no normal successor, which a `Block` leaf cannot tell apart from one
/// that continues.
pub fn implied_flow(tree: &ControlTree, ends_function: impl Fn(BlockId) -> bool) -> ImpliedFlow {
    let mut reader = Reader {
        tree,
        ends_function,
        flow: ImpliedFlow {
            structural: BTreeSet::new(),
            through_gotos: BTreeSet::new(),
        },
        loops: Vec::new(),
    };
    reader.node(tree.root(), Entry::NOWHERE);
    reader.flow
}

/// The edges of `cfg` that ordinary control flow takes: every kind except the
/// exceptional ones.
pub fn normal_edges<I, E>(cfg: &Cfg<I, E>) -> BTreeSet<(BlockId, BlockId)> {
    cfg.edges()
        .filter(|edge| !edge.kind().is_exceptional())
        .map(|edge| (edge.source(), edge.target()))
        .collect()
}

/// Edges between blocks that hold something, with every block that holds
/// nothing made transparent: a path through hollow blocks becomes one edge.
fn contract_hollow_blocks(
    edges: &BTreeSet<(BlockId, BlockId)>,
    hollow: &impl Fn(BlockId) -> bool,
) -> BTreeSet<(BlockId, BlockId)> {
    let mut successors: BTreeMap<BlockId, Vec<BlockId>> = BTreeMap::new();
    for &(source, target) in edges {
        successors.entry(source).or_default().push(target);
    }
    let mut contracted = BTreeSet::new();
    for &(source, first) in edges {
        if hollow(source) {
            continue;
        }
        let mut seen = BTreeSet::new();
        let mut stack = vec![first];
        while let Some(block) = stack.pop() {
            if !hollow(block) {
                contracted.insert((source, block));
            } else if seen.insert(block) {
                stack.extend(successors.get(&block).into_iter().flatten().copied());
            }
        }
    }
    contracted
}

/// The flow the tree implies equals the graph's normal edges: the edges the
/// tree expresses by shape plus the ones it reports as gotos are exactly the
/// real ones, and a goto edge always lands on a target the report names.
///
/// A block that holds no instruction has no node, so both sides are compared
/// with `hollow` blocks made transparent.
pub fn assert_flow_matches_graph_with<I, E>(
    cfg: &Cfg<I, E>,
    tree: &ControlTree,
    report: &LiftReport,
    covered: &BTreeSet<BlockId>,
    hollow: &impl Fn(BlockId) -> bool,
) {
    let implied = implied_flow(tree, |block| {
        cfg.outgoing(block)
            .all(|edge| cfg.edge(edge).kind().is_exceptional())
    });
    // Code the tree drops, such as an unreachable block, takes its edges along.
    let real: BTreeSet<(BlockId, BlockId)> = normal_edges(cfg)
        .into_iter()
        .filter(|(source, _)| covered.contains(source))
        .collect();
    let goto_targets: BTreeSet<BlockId> = report.gotos.iter().map(|goto| goto.target).collect();
    for &(source, target) in &implied.through_gotos {
        assert!(
            goto_targets.contains(&target),
            "{source} -> {target} is a goto the report does not list"
        );
    }
    let all: BTreeSet<(BlockId, BlockId)> = implied
        .structural
        .union(&implied.through_gotos)
        .copied()
        .collect();
    assert_eq!(
        contract_hollow_blocks(&all, hollow),
        contract_hollow_blocks(&real, hollow),
        "the tree's implied flow and the graph's edges differ\n{tree:?}"
    );
}

/// [`assert_flow_matches_graph_with`] where a block is hollow when it holds no
/// instruction.
pub fn assert_flow_matches_graph<I, E>(
    cfg: &Cfg<I, E>,
    tree: &ControlTree,
    report: &LiftReport,
    covered: &BTreeSet<BlockId>,
) {
    assert_flow_matches_graph_with(cfg, tree, report, covered, &|block| {
        cfg.block(block).is_empty()
    });
}
