//! A control tree runs like the graph it was computed from.
//!
//! A small interpreter runs the graph and the tree under one stream of
//! decisions and records the instructions each executes. Every conditional
//! branch, dispatch, and loop test asks the stream for its outcome, so two
//! runs that agree so far ask the same questions in the same order. The trace
//! of the tree must equal the trace of the graph: a transfer the structurer
//! dropped, redirected, or resolved against the wrong loop shows up as a
//! diverging instruction.
//!
//! Some blocks raise: when a block that has an exception edge finishes its
//! instructions, a fixed function of the stream's seed and the block says
//! whether it takes the first such edge instead of its normal successors. The
//! graph then continues at the handler entry. The tree continues in the
//! handler arm of the try region that holds the entry, or under the label of a
//! pad the structurer swept. A pad that the tree has no place to enter ends the
//! run, which must then be a prefix of the graph's.

extern crate alloc;
use alloc::collections::BTreeSet;
use alloc::vec::Vec;

use super::exhaustive::{Terminator, graph, unpredicated};
use super::invariants::{check, handler_sets, reach, render};
use crate::block::BlockId;
use crate::cfg::Cfg;
use crate::edge::{EdgeId, EdgeKind};
use crate::graph::dominator::DominatorTree;
use crate::region::{Handler, HandlerBody, HandlerKind, Region, RegionId};
use crate::{
    BlockSpan, ControlBody, ControlNode, ControlNodeId, ControlTree, LiftReport, LoopKind,
};

/// How many streams of decisions the runs of one graph consume.
const STREAM_COUNT: u64 = 4;

/// How many streams a hand-built graph is compared under.
const MANY_STREAMS: u64 = 64;

/// How many instructions a run executes before it stops.
const INSTRUCTION_LIMIT: usize = 48;

/// How many steps a run takes before it stops, so a loop that executes
/// nothing ends too.
const STEP_LIMIT: usize = 2_000;

/// The decisions both runs consume, in order.
#[derive(Clone)]
struct Decisions(u64);

impl Decisions {
    fn choose(&mut self, options: usize) -> usize {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        let options = u64::try_from(options).expect("an option count fits in 64 bits");
        usize::try_from((self.0 >> 20) % options).expect("a value below a usize bound")
    }
}

/// The block `block` raises to once its instructions ran, when it raises: one
/// of its exception edges, chosen by the seed and the block. A block whose
/// only edges are exceptional always raises, since it has nowhere else to go.
/// A block without instructions never raises, because the tree has no node to
/// raise at.
fn raise_target<I, E>(cfg: &Cfg<I, E>, seed: u64, block: BlockId) -> Option<BlockId> {
    if cfg.block(block).instructions().is_empty() {
        return None;
    }
    let targets: Vec<BlockId> = cfg
        .outgoing(block)
        .map(|edge| cfg.edge(edge))
        .filter(|edge| edge.kind().is_exceptional())
        .map(|edge| edge.target())
        .collect();
    if targets.is_empty() {
        return None;
    }
    let has_normal_successor = cfg
        .outgoing(block)
        .any(|edge| !cfg.edge(edge).kind().is_exceptional());
    let mut mix =
        Decisions(seed ^ (u64::from(block.raw()) + 1).wrapping_mul(0xD1B5_4A32_D192_ED03));
    mix.choose(2);
    let raises = !has_normal_successor || mix.choose(2) == 1;
    raises.then(|| targets[mix.choose(targets.len())])
}

/// Whether every reachable edge tagged [`EdgeKind::Back`] goes to a block
/// that dominates its source, as dominance alone would find. A tag on any other
/// edge asserts a loop the graph does not hold (a forward edge, an edge into the
/// middle of a cycle), and the tree of such a graph has no graph to match.
fn back_tags_agree_with_dominance<I, E>(cfg: &Cfg<I, E>) -> bool {
    let dominators = DominatorTree::compute(cfg);
    cfg.edge_ids().all(|edge| {
        let edge = cfg.edge(edge);
        edge.kind() != EdgeKind::Back
            || !dominators.is_reachable(edge.source())
            || dominators.dominates(edge.target(), edge.source())
    })
}

/// Whether the tree of `cfg` promises the graph's behavior: the regions keep
/// their contract and no Back tag asserts a loop that dominance does not.
fn is_faithful<I, E>(cfg: &Cfg<I, E>) -> bool {
    back_tags_agree_with_dominance(cfg) && regions_are_well_formed(cfg)
}

/// Whether the regions keep the contract of
/// [`ControlTree::compute_with_report`]: a handler's blocks (its entry, its
/// filter funclet, its known extent) are not protected by its region, are not
/// the function entry, and belong to that handler alone. A region that breaks
/// this may hold a block in two places, so the tree has no single behavior to
/// match. With regions, every block also holds an instruction: an empty pad
/// that returns is a catch body that ends, which the tree cannot tell from
/// one that falls out into the code after it.
fn regions_are_well_formed<I, E>(cfg: &Cfg<I, E>) -> bool {
    if cfg.regions().is_empty() {
        return true;
    }
    if cfg
        .block_ids()
        .any(|block| cfg.block(block).instructions().is_empty())
    {
        return false;
    }
    let sets = handler_sets(cfg);
    let claimed: usize = sets.iter().map(BTreeSet::len).sum();
    let all: BTreeSet<BlockId> = sets.iter().flatten().copied().collect();
    all.len() == claimed
        && !all.contains(&cfg.entry())
        && cfg
            .regions()
            .iter()
            .all(|region| region.protected_blocks.is_disjoint(&all))
}

/// The instructions a run executed, as (block, position) in order.
type Trace = Vec<(BlockId, usize)>;

/// The run reached its instruction limit.
struct Stop;

fn emit(trace: &mut Trace, block: BlockId, positions: core::ops::Range<usize>) -> Result<(), Stop> {
    for position in positions {
        if trace.len() == INSTRUCTION_LIMIT {
            return Err(Stop);
        }
        trace.push((block, position));
    }
    Ok(())
}

/// The case edges of a dispatch block, in edge order, and its default edge.
fn dispatch<I, E>(cfg: &Cfg<I, E>, block: BlockId) -> (Vec<EdgeId>, Option<EdgeId>) {
    let mut cases = Vec::new();
    let mut default = None;
    for edge in cfg.outgoing(block) {
        match cfg.edge(edge).kind() {
            EdgeKind::SwitchCase => cases.push(edge),
            EdgeKind::Fallthrough | EdgeKind::Unconditional if default.is_none() => {
                default = Some(edge);
            }
            _ => {}
        }
    }
    (cases, default)
}

/// The edge of a dispatch block that the next decision takes.
fn dispatch_edge<I, E>(
    cfg: &Cfg<I, E>,
    block: BlockId,
    decisions: &mut Decisions,
) -> Option<EdgeId> {
    let (cases, default) = dispatch(cfg, block);
    let options = cases.len() + usize::from(default.is_some());
    // A dispatch with one way out asks nothing: the tree may hold it as a
    // straight line, with no node to ask.
    let picked = if options > 1 {
        decisions.choose(options)
    } else {
        0
    };
    cases.get(picked).copied().or(default)
}

/// The block the graph runs after `block`, asking for a decision at a branch
/// or dispatch.
fn successor<I, E>(cfg: &Cfg<I, E>, block: BlockId, decisions: &mut Decisions) -> Option<BlockId> {
    let flow: Vec<EdgeId> = cfg
        .outgoing(block)
        .filter(|&edge| !cfg.edge(edge).kind().is_exceptional())
        .collect();
    let target = |kind: EdgeKind| {
        flow.iter()
            .map(|&edge| cfg.edge(edge))
            .find(|edge| edge.kind() == kind)
            .map(|edge| edge.target())
    };
    if let (Some(on_true), Some(on_false)) = (
        target(EdgeKind::ConditionalTrue),
        target(EdgeKind::ConditionalFalse),
    ) {
        return Some(if decisions.choose(2) == 1 {
            on_true
        } else {
            on_false
        });
    }
    if target(EdgeKind::SwitchCase).is_some() {
        let edge = dispatch_edge(cfg, block, decisions)?;
        return Some(cfg.edge(edge).target());
    }
    flow.first().map(|&edge| cfg.edge(edge).target())
}

fn run_graph<I, E>(cfg: &Cfg<I, E>, seed: u64) -> Trace {
    let mut decisions = Decisions(seed);
    let mut trace = Trace::new();
    let mut block = cfg.entry();
    for _ in 0..STEP_LIMIT {
        let length = cfg.block(block).instructions().len();
        if emit(&mut trace, block, 0..length).is_err() {
            break;
        }
        let next = raise_target(cfg, seed, block).or_else(|| successor(cfg, block, &mut decisions));
        match next {
            Some(next) => block = next,
            None => break,
        }
    }
    trace
}

/// One level of the tree's execution stack.
enum Frame {
    /// Statements still to run, and the next one.
    Body {
        nodes: Vec<ControlNodeId>,
        next: usize,
    },
    /// A loop whose body, when it ends, starts the next iteration.
    Loop { node: ControlNodeId, entered: bool },
    /// A switch arm, which a plain break does not stop at.
    Arm,
}

/// Why a tree run ended.
#[derive(Debug, PartialEq, Eq)]
enum Ending {
    /// The tree returned or ran out of statements.
    Done,
    /// The run hit its instruction or step limit.
    Limit,
    /// A goto named a label the tree does not hold.
    Dangling,
    /// A raise reached a pad that the tree has no place to enter.
    Unmodeled,
}

/// What a transfer into the middle of the tree looks for.
#[derive(Clone, Copy)]
enum Wanted {
    /// The label of a block.
    Label(BlockId),
    /// The handler arm of a try region whose handler enters at a block.
    Handler(BlockId),
}

struct TreeRun<'a, I, E> {
    cfg: &'a Cfg<I, E>,
    tree: &'a ControlTree,
    seed: u64,
    /// The blocks the entry reaches over edges of every kind, sorted. The
    /// tree keeps other owned code (a handler's dead tail) after the live
    /// code of its body, where it would run by falling into it. The graph
    /// never runs it, so neither does this run: the first dead node ends its
    /// body.
    live: Vec<BlockId>,
    /// The statements of the function body, then each swept label alone: code
    /// the body does not fall into.
    units: Vec<Vec<ControlNodeId>>,
    decisions: Decisions,
    trace: Trace,
    frames: Vec<Frame>,
    ending: Ending,
}

enum Flow {
    Next,
    Exit,
    /// The node is dead code, and so is the rest of its body.
    Dead,
}

/// How the run ended, and what it executed.
fn run_tree<I, E>(
    cfg: &Cfg<I, E>,
    tree: &ControlTree,
    report: &LiftReport,
    seed: u64,
) -> (Trace, Ending) {
    let mut units = Vec::new();
    match *tree.node(tree.root()) {
        ControlNode::Sequence(body) => {
            let nodes = tree.body(body);
            let swept = |id: &ControlNodeId| {
                matches!(
                    *tree.node(*id),
                    ControlNode::Label { block, .. } if report.swept_blocks.contains(&block)
                )
            };
            let split = nodes
                .iter()
                .rposition(|id| !swept(id))
                .map_or(0, |at| at + 1);
            units.push(nodes[..split].to_vec());
            units.extend(nodes[split..].iter().map(|&id| alloc::vec![id]));
        }
        _ => units.push(alloc::vec![tree.root()]),
    }
    let first = units[0].clone();
    let mut run = TreeRun {
        cfg,
        tree,
        seed,
        live: reach(cfg, Vec::new()),
        units,
        decisions: Decisions(seed),
        trace: Trace::new(),
        frames: alloc::vec![Frame::Body {
            nodes: first,
            next: 0,
        }],
        ending: Ending::Done,
    };
    run.execute();
    (run.trace, run.ending)
}

impl<I, E> TreeRun<'_, I, E> {
    fn execute(&mut self) {
        for _ in 0..STEP_LIMIT {
            let Some(top) = self.frames.last_mut() else {
                return;
            };
            let outcome = match top {
                Frame::Body { nodes, next } => {
                    let Some(&id) = nodes.get(*next) else {
                        self.frames.pop();
                        continue;
                    };
                    *next += 1;
                    self.step(id)
                }
                Frame::Loop { node, entered } => {
                    let (node, was_entered) = (*node, *entered);
                    *entered = true;
                    self.iterate(node, was_entered)
                }
                Frame::Arm => {
                    self.frames.pop();
                    Ok(Flow::Next)
                }
            };
            match outcome {
                Ok(Flow::Next) => {}
                Ok(Flow::Dead) => {
                    self.frames.pop();
                }
                Ok(Flow::Exit) => return,
                Err(Stop) => {
                    if self.ending == Ending::Done {
                        self.ending = Ending::Limit;
                    }
                    return;
                }
            }
        }
        self.ending = Ending::Limit;
    }

    fn push_body(&mut self, body: ControlBody) {
        let nodes = self.tree.body(body).to_vec();
        self.frames.push(Frame::Body { nodes, next: 0 });
    }

    /// Execute `span`. The block raises when the span ends it: then the tree
    /// has moved to the handler, and the answer is `true`.
    fn emit_span(&mut self, span: BlockSpan) -> Result<bool, Stop> {
        emit(&mut self.trace, span.block(), span.range())?;
        let ends_block = span.end() == self.cfg.block(span.block()).instructions().len();
        let Some(target) = ends_block
            .then(|| raise_target(self.cfg, self.seed, span.block()))
            .flatten()
        else {
            return Ok(false);
        };
        self.jump(
            &[Wanted::Handler(target), Wanted::Label(target)],
            Ending::Unmodeled,
        )?;
        Ok(true)
    }

    /// The decision of a conditional: whether the true edge is taken.
    fn takes_true(&mut self) -> bool {
        self.decisions.choose(2) == 1
    }

    /// Run the loop test at the start of an iteration.
    fn iterate(&mut self, node: ControlNodeId, entered: bool) -> Result<Flow, Stop> {
        let ControlNode::Loop { kind, body, .. } = *self.tree.node(node) else {
            unreachable!("a loop frame names a loop");
        };
        match kind {
            LoopKind::Endless => self.push_body(body),
            LoopKind::While {
                condition,
                exit_on_true,
            } => {
                for &span in self.tree.spans(condition) {
                    if self.emit_span(span)? {
                        return Ok(Flow::Next);
                    }
                }
                if self.takes_true() == exit_on_true {
                    self.frames.pop();
                } else {
                    self.push_body(body);
                }
            }
            LoopKind::DoWhile {
                latch,
                continue_on_true,
            } => {
                if !entered {
                    self.push_body(body);
                } else if !self.emit_span(latch)? {
                    if self.takes_true() == continue_on_true {
                        self.push_body(body);
                    } else {
                        self.frames.pop();
                    }
                }
            }
        }
        Ok(Flow::Next)
    }

    /// The position of the loop frame a break or continue names.
    fn loop_frame(&self, label: Option<BlockId>) -> usize {
        self.frames
            .iter()
            .rposition(|frame| match (frame, label) {
                (Frame::Loop { .. }, None) => true,
                (Frame::Loop { node, .. }, Some(header)) => {
                    matches!(*self.tree.node(*node), ControlNode::Loop { header: h, .. } if h == header)
                }
                _ => false,
            })
            .expect("a break or continue sits inside the loop it names")
    }

    /// The block a node stands for, looking through the labels around it.
    fn anchor(&self, id: ControlNodeId) -> Option<BlockId> {
        match *self.tree.node(id) {
            ControlNode::Block(span) | ControlNode::Return(span) => Some(span.block()),
            ControlNode::IfThenElse { condition, .. } | ControlNode::Switch { condition, .. } => {
                Some(condition.block())
            }
            ControlNode::Loop { header, .. } => Some(header),
            ControlNode::Label { body, .. } => self
                .tree
                .body(body)
                .first()
                .and_then(|&only| self.anchor(only)),
            _ => None,
        }
    }

    /// Whether the graph can run `block`.
    fn is_live(&self, block: BlockId) -> bool {
        self.live.binary_search(&block).is_ok()
    }

    fn step(&mut self, id: ControlNodeId) -> Result<Flow, Stop> {
        let node = *self.tree.node(id);
        if self.anchor(id).is_some_and(|block| !self.is_live(block)) {
            return Ok(Flow::Dead);
        }
        match node {
            ControlNode::Block(span) => {
                self.emit_span(span)?;
            }
            ControlNode::Return(span) => {
                self.emit_span(span)?;
                return Ok(Flow::Exit);
            }
            ControlNode::Sequence(body)
            | ControlNode::Label { body, .. }
            | ControlNode::Guarded { body, .. }
            | ControlNode::TryCatch { try_body: body, .. } => self.push_body(body),
            ControlNode::IfThenElse {
                condition,
                then_body,
                else_body,
            } => {
                if !self.emit_span(condition)? {
                    let taken = if self.takes_true() {
                        then_body
                    } else {
                        else_body
                    };
                    self.push_body(taken);
                }
            }
            ControlNode::Loop { .. } => self.frames.push(Frame::Loop {
                node: id,
                entered: false,
            }),
            ControlNode::Switch {
                condition,
                cases,
                default_body,
                default_edge,
            } => {
                if !self.emit_span(condition)? {
                    let edge = dispatch_edge(self.cfg, condition.block(), &mut self.decisions);
                    let arm = if edge == default_edge {
                        default_body
                    } else {
                        self.tree
                            .cases(cases)
                            .iter()
                            .find(|case| {
                                edge.is_some_and(|edge| self.tree.edges(case.edges).contains(&edge))
                            })
                            .expect("a taken case edge belongs to a case")
                            .body
                    };
                    self.frames.push(Frame::Arm);
                    self.push_body(arm);
                }
            }
            ControlNode::Break { label } => {
                let at = self.loop_frame(label);
                self.frames.truncate(at);
            }
            ControlNode::Continue { label } => {
                let at = self.loop_frame(label);
                self.frames.truncate(at + 1);
                if let Some(Frame::Loop { entered, .. }) = self.frames.last_mut() {
                    *entered = true;
                }
            }
            ControlNode::Goto { target } => {
                self.jump(&[Wanted::Label(target)], Ending::Dangling)?;
            }
        }
        Ok(Flow::Next)
    }

    /// Replace the stack with the one that runs the first of `wanted` the tree
    /// holds, or stop with `missing` when it holds none.
    fn jump(&mut self, wanted: &[Wanted], missing: Ending) -> Result<(), Stop> {
        let units = self.units.clone();
        for &wanted in wanted {
            let mut frames = Vec::new();
            if units
                .iter()
                .any(|unit| self.locate(unit, wanted, &mut frames))
            {
                self.frames = frames;
                return Ok(());
            }
        }
        self.ending = missing;
        Err(Stop)
    }

    /// Rebuild in `frames` the stack that is running `wanted`, when `nodes`
    /// hold it: the stack of its ancestors, each resuming after the child that
    /// holds it.
    fn locate(&self, nodes: &[ControlNodeId], wanted: Wanted, frames: &mut Vec<Frame>) -> bool {
        for (position, &child) in nodes.iter().enumerate() {
            let node = *self.tree.node(child);
            if matches!(wanted, Wanted::Label(target)
                if matches!(node, ControlNode::Label { block, .. } if block == target))
            {
                frames.push(Frame::Body {
                    nodes: nodes.to_vec(),
                    next: position,
                });
                return true;
            }
            let mark = frames.len();
            frames.push(Frame::Body {
                nodes: nodes.to_vec(),
                next: position + 1,
            });
            if let (Wanted::Handler(entry), ControlNode::TryCatch { handlers, .. }) = (wanted, node)
                && let Some(handler) = self
                    .tree
                    .handlers(handlers)
                    .iter()
                    .find(|handler| handler.entry == entry)
            {
                frames.push(Frame::Body {
                    nodes: self.tree.body(handler.body).to_vec(),
                    next: 0,
                });
                return true;
            }
            for body in self.tree.child_bodies(child) {
                let inner = frames.len();
                match node {
                    ControlNode::Loop { .. } => frames.push(Frame::Loop {
                        node: child,
                        entered: true,
                    }),
                    ControlNode::Switch { .. } => frames.push(Frame::Arm),
                    _ => {}
                }
                if self.locate(self.tree.body(body), wanted, frames) {
                    return true;
                }
                frames.truncate(inner);
            }
            frames.truncate(mark);
        }
        false
    }
}

/// Run `cfg` and `tree` under several streams of decisions and require the
/// same instructions in the same order.
///
/// A goto whose label found no anchor, or a raise into a pad the tree cannot
/// enter, stops the tree's run, which must then be a prefix of the graph's. The
/// report lists a goto without a label.
pub(super) fn assert_runs_alike<I, E>(
    cfg: &Cfg<I, E>,
    tree: &ControlTree,
    report: &LiftReport,
    context: &str,
) {
    assert_streams_alike(cfg, tree, report, context, STREAM_COUNT);
}

/// Every block that raises into a handler is inside the try body of the region
/// that owns it: the tree's `try` covers what the exception edges say it
/// covers.
pub(super) fn assert_raisers_are_covered<I, E>(cfg: &Cfg<I, E>, tree: &ControlTree, context: &str) {
    if !is_faithful(cfg) {
        return;
    }
    let live = reach(cfg, Vec::new());
    for id in tree.preorder() {
        let ControlNode::TryCatch {
            try_body, handlers, ..
        } = *tree.node(id)
        else {
            continue;
        };
        let covered = blocks_of_body(tree, try_body);
        for handler in tree.handlers(handlers) {
            for &block in &live {
                let raises_here = !cfg.block(block).instructions().is_empty()
                    && cfg.outgoing(block).any(|edge| {
                        let edge = cfg.edge(edge);
                        edge.kind().is_exceptional() && edge.target() == handler.entry
                    });
                assert!(
                    !raises_here || covered.contains(&block),
                    "{block} raises into {} outside its try in {context}\n{}",
                    handler.entry,
                    render(tree)
                );
            }
        }
    }
}

/// The blocks whose instructions the statements of `body` hold.
fn blocks_of_body(tree: &ControlTree, body: ControlBody) -> BTreeSet<BlockId> {
    let mut blocks = BTreeSet::new();
    let mut stack: Vec<ControlNodeId> = tree.body(body).to_vec();
    while let Some(id) = stack.pop() {
        match *tree.node(id) {
            ControlNode::Block(span) | ControlNode::Return(span) => {
                blocks.insert(span.block());
            }
            ControlNode::IfThenElse { condition, .. } | ControlNode::Switch { condition, .. } => {
                blocks.insert(condition.block());
            }
            ControlNode::Loop { kind, .. } => match kind {
                LoopKind::Endless => {}
                LoopKind::While { condition, .. } => {
                    blocks.extend(tree.spans(condition).iter().map(|span| span.block()));
                }
                LoopKind::DoWhile { latch, .. } => {
                    blocks.insert(latch.block());
                }
            },
            _ => {}
        }
        for child in tree.child_bodies(id) {
            stack.extend_from_slice(tree.body(child));
        }
    }
    blocks
}

/// [`assert_runs_alike`] over `streams` streams of decisions, for a graph
/// whose divergence only some streams reach.
fn assert_streams_alike<I, E>(
    cfg: &Cfg<I, E>,
    tree: &ControlTree,
    report: &LiftReport,
    context: &str,
    streams: u64,
) {
    if !is_faithful(cfg) {
        return;
    }
    for stream in 1..=streams {
        let seed = stream.wrapping_mul(0x9E37_79B9_7F4A_7C15);
        let expected = run_graph(cfg, seed);
        let (actual, ending) = run_tree(cfg, tree, report, seed);
        match ending {
            Ending::Dangling | Ending::Unmodeled => {
                assert!(
                    ending == Ending::Unmodeled || !report.unresolved_labels.is_empty(),
                    "a goto without a label that the report does not list in {context}"
                );
                assert!(
                    expected.starts_with(&actual),
                    "the tree diverges before it stopped in {context} stream {stream}\n{}\n{report:?}",
                    render(tree)
                );
            }
            Ending::Done | Ending::Limit => assert_eq!(
                actual,
                expected,
                "the tree diverges from the graph in {context} stream {stream}\n{}\n{report:?}",
                render(tree)
            ),
        }
    }
}

/// Compute the tree of `cfg`, check every invariant, and compare many runs.
fn check_equivalent<I, E>(cfg: &Cfg<I, E>, context: &str) {
    let (tree, report) = ControlTree::compute_with_report(cfg);
    check(cfg, &tree, &report, context);
    assert_streams_alike(cfg, &tree, &report, context, MANY_STREAMS);
}

#[test]
fn an_inner_branch_that_merges_at_the_loop_header_continues_the_loop() {
    // bb2 branches to bb0 on both arms, so the loop header is its merge, but
    // it sits in an arm whose own merge is bb1: the transfer to bb0 must
    // survive as a continue instead of falling through into bb1.
    let cfg = graph(
        &[
            Terminator::Branch(1, 2),
            Terminator::Branch(0, 3),
            Terminator::Branch(0, 0),
            Terminator::Exit,
        ],
        &[1; 4],
        unpredicated,
    );
    check_equivalent(&cfg, "branch merging at the header inside an arm");
}

#[test]
fn a_swept_branch_that_merges_at_emitted_code_transfers_there() {
    // The switch's second arm is swept under a label, and its branch merges at
    // bb2, which the walk already emitted after the loop. The swept code must
    // go to bb2 instead of ending.
    let cfg = graph(
        &[
            Terminator::Switch {
                cases: 0b011,
                default: Some(2),
            },
            Terminator::Branch(2, 2),
            Terminator::Exit,
        ],
        &[1; 3],
        unpredicated,
    );
    check_equivalent(&cfg, "swept branch merging at emitted code");
}

#[test]
fn a_back_edge_from_dead_code_does_not_loop_the_live_program() {
    // bb1 is dead, so its back edge to bb0 never runs: bb0 returns, and the
    // tree must not wrap it in an endless loop.
    let cfg = graph(
        &[Terminator::Exit, Terminator::One(0, EdgeKind::Back)],
        &[1; 2],
        unpredicated,
    );
    let (tree, report) = ControlTree::compute_with_report(&cfg);
    check(&cfg, &tree, &report, "dead back edge");
    assert!(
        !tree
            .preorder()
            .any(|id| matches!(tree.node(id), ControlNode::Loop { .. })),
        "{}",
        render(&tree)
    );
}

#[test]
fn a_handler_entry_that_dominates_its_protected_blocks_does_not_cut_their_flow() {
    // b0 is the loop header and the catch-all pad. b1 and b2 are protected and
    // raise back to it, so dominance calls both exception edges back edges. b1
    // still falls through into b2: the tree must run b2 after b1 instead of
    // continuing the loop at b1.
    let mut cfg = graph(
        &[
            Terminator::One(1, EdgeKind::Fallthrough),
            Terminator::One(2, EdgeKind::Fallthrough),
            Terminator::Exit,
        ],
        &[1; 3],
        unpredicated,
    );
    let [b0, b1, b2] = [0, 1, 2].map(BlockId::from_index);
    cfg.add_edge(b1, b0, EdgeKind::ExceptionHandler);
    cfg.add_edge(b2, b0, EdgeKind::ExceptionHandler);
    cfg.add_region(Region {
        id: RegionId::from_raw(0),
        protected_blocks: [b1, b2].into_iter().collect(),
        handlers: alloc::vec![Handler {
            entry: b0,
            body: HandlerBody::known([b0]),
            kind: HandlerKind::CatchAll,
        }],
        parent: None,
    });
    check_equivalent(&cfg, "handler entry dominating its protected blocks");
}

#[test]
fn a_retry_loop_that_leaves_through_the_enclosing_loops_back_edge_continues_it() {
    // b0 is a while loop; its body b1 is protected, and its pad b2 retries b1,
    // so b1 heads an inner loop through the exception edge. b1's only normal
    // edge returns to b0: the inner loop's body must continue the outer loop
    // there instead of ending after b1.
    let mut cfg = graph(
        &[
            Terminator::Branch(1, 3),
            Terminator::One(0, EdgeKind::Fallthrough),
            Terminator::One(1, EdgeKind::Fallthrough),
            Terminator::Exit,
        ],
        &[1; 4],
        unpredicated,
    );
    let [b1, b2] = [1, 2].map(BlockId::from_index);
    cfg.add_edge(b1, b2, EdgeKind::ExceptionHandler);
    cfg.add_region(Region {
        id: RegionId::from_raw(0),
        protected_blocks: [b1].into_iter().collect(),
        handlers: alloc::vec![Handler {
            entry: b2,
            body: HandlerBody::known([b2]),
            kind: HandlerKind::Catch,
        }],
        parent: None,
    });
    check_equivalent(&cfg, "retry loop inside a while loop");
}
