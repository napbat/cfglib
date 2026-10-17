//! A control tree runs like the graph it was computed from.
//!
//! A small interpreter runs the graph and the tree under one stream of
//! decisions and records the instructions each executes. Every conditional
//! branch, dispatch, and loop test asks the stream for its outcome, so two
//! runs that agree so far ask the same questions in the same order. The trace
//! of the tree must equal the trace of the graph: a transfer the structurer
//! dropped, redirected, or resolved against the wrong loop shows up as a
//! diverging instruction. Exception edges are never taken and a try region
//! runs its protected body, so the runs cover the normal flow around regions.

extern crate alloc;
use alloc::vec::Vec;

use super::exhaustive::{Terminator, for_each_graph, graph, terminators, unpredicated};
use crate::block::BlockId;
use crate::cfg::Cfg;
use crate::edge::{EdgeId, EdgeKind};
use crate::{ControlNode, ControlNodeId, ControlTree, LiftReport, LoopKind};

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
    let picked = decisions.choose(cases.len() + usize::from(default.is_some()));
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

fn run_graph<I, E>(cfg: &Cfg<I, E>, mut decisions: Decisions) -> Trace {
    let mut trace = Trace::new();
    let mut block = cfg.entry();
    for _ in 0..STEP_LIMIT {
        let length = cfg.block(block).instructions().len();
        if emit(&mut trace, block, 0..length).is_err() {
            break;
        }
        match successor(cfg, block, &mut decisions) {
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
}

struct TreeRun<'a, I, E> {
    cfg: &'a Cfg<I, E>,
    tree: &'a ControlTree,
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
}

/// How the run ended, and what it executed.
fn run_tree<I, E>(
    cfg: &Cfg<I, E>,
    tree: &ControlTree,
    report: &LiftReport,
    decisions: Decisions,
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
        units,
        decisions,
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

    fn push_body(&mut self, body: crate::ControlBody) {
        let nodes = self.tree.body(body).to_vec();
        self.frames.push(Frame::Body { nodes, next: 0 });
    }

    fn emit_span(&mut self, span: crate::BlockSpan) -> Result<(), Stop> {
        emit(&mut self.trace, span.block(), span.range())
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
                    self.emit_span(span)?;
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
                } else {
                    self.emit_span(latch)?;
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

    fn step(&mut self, id: ControlNodeId) -> Result<Flow, Stop> {
        match *self.tree.node(id) {
            ControlNode::Block(span) => self.emit_span(span)?,
            ControlNode::Return(span) => {
                self.emit_span(span)?;
                return Ok(Flow::Exit);
            }
            ControlNode::Sequence(body) | ControlNode::Label { body, .. } => self.push_body(body),
            ControlNode::Guarded { body, .. } => self.push_body(body),
            ControlNode::TryCatch { try_body, .. } => self.push_body(try_body),
            ControlNode::IfThenElse {
                condition,
                then_body,
                else_body,
            } => {
                self.emit_span(condition)?;
                let taken = if self.takes_true() {
                    then_body
                } else {
                    else_body
                };
                self.push_body(taken);
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
                self.emit_span(condition)?;
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
                let mut frames = Vec::new();
                let units = self.units.clone();
                if !units
                    .iter()
                    .any(|unit| self.locate(unit, target, &mut frames))
                {
                    self.ending = Ending::Dangling;
                    return Err(Stop);
                }
                self.frames = frames;
            }
        }
        Ok(Flow::Next)
    }

    /// Rebuild in `frames` the stack that is running `target`'s label, when
    /// `nodes` hold it.
    fn locate(&self, nodes: &[ControlNodeId], target: BlockId, frames: &mut Vec<Frame>) -> bool {
        for (position, &child) in nodes.iter().enumerate() {
            if matches!(*self.tree.node(child), ControlNode::Label { block, .. } if block == target)
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
            for body in self.tree.child_bodies(child) {
                let inner = frames.len();
                match *self.tree.node(child) {
                    ControlNode::Loop { .. } => frames.push(Frame::Loop {
                        node: child,
                        entered: true,
                    }),
                    ControlNode::Switch { .. } => frames.push(Frame::Arm),
                    _ => {}
                }
                if self.locate(self.tree.body(body), target, frames) {
                    return true;
                }
                frames.truncate(inner);
            }
            frames.truncate(mark);
        }
        false
    }
}

/// Run `cfg` and its tree under several streams of decisions.
fn check_equivalent<I, E>(cfg: &Cfg<I, E>, context: &str) {
    let (tree, report) = ControlTree::compute_with_report(cfg);
    for seed in 1..=6_u64 {
        let decisions = Decisions(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15));
        let expected = run_graph(cfg, decisions.clone());
        let (actual, ending) = run_tree(cfg, &tree, &report, decisions);
        if ending == Ending::Dangling {
            // The report names the goto whose label has no anchor.
            assert!(
                !report.unresolved_labels.is_empty(),
                "a goto without a label that the report does not list in {context}"
            );
            assert!(expected.starts_with(&actual), "{context} stream {seed}");
        } else {
            assert_eq!(
                actual,
                expected,
                "the tree diverges from the graph in {context} stream {seed}\n{}\n{report:?}",
                tree.to_pseudocode(|sink, point| write!(
                    sink,
                    "b{}.{}",
                    point.block.index(),
                    point.inst_idx
                ))
                .unwrap_or_default()
            );
        }
    }
}

#[test]
fn every_graph_of_up_to_three_blocks_runs_like_its_tree() {
    for count in 1..=3 {
        let all = terminators(count, &[EdgeKind::Fallthrough, EdgeKind::Jump], true);
        for_each_graph(count, &all, |picked| {
            let cfg = graph(picked, &alloc::vec![1; count], unpredicated);
            check_equivalent(&cfg, &alloc::format!("{picked:?}"));
        });
    }
}

#[test]
fn every_graph_of_four_blocks_runs_like_its_tree() {
    let all = terminators(4, &[EdgeKind::Fallthrough, EdgeKind::Jump], false);
    for_each_graph(4, &all, |picked| {
        let cfg = graph(picked, &[1; 4], unpredicated);
        check_equivalent(&cfg, &alloc::format!("{picked:?}"));
    });
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
