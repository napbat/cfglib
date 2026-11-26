//! Building the decomposition from the tree, with the blocks it leaves out.
//!
//! One pass over the tree finds the obstacles and the blocks it has a leaf,
//! a test or a header for; a second builds the grammar's nodes on an explicit
//! stack. The builder tracks the edges that leave what it has built so far,
//! as a count and their common target. When the edges falling into the next
//! statement all end at a block the tree has no leaf for, and they are every
//! edge into it, that block comes first: an empty block the tree passes
//! through, or a jump trampoline a `break` or `continue` routes through.
//!
//! Such a block is placed as late as it can be. The edges falling out of an
//! arm wait for the parallel, which places the block after itself when they
//! are every edge into it, and at the end of the arm otherwise; the edges
//! falling out of the procedure end at its empty exit.

extern crate alloc;
use alloc::vec;
use alloc::vec::Vec;

use super::{SplDecomposition, SplError, SplNode, SplNodeId, SplRun};
use crate::analysis::control_tree::node::{ControlBody, ControlNode, ControlNodeId, LoopKind};
use crate::analysis::control_tree::tree::ControlTree;
use crate::block::BlockId;
use crate::cfg::Cfg;
use crate::edge::EdgeKind;
use crate::graph::traverse::reachable;
use crate::graph::view::Direction;
use fragment::{Arm, BodyFrame, BranchFrame, Built, Cursor, Exits, Frame, Piece, Stubs, Target};

mod fragment;

/// Build the decomposition of `cfg` from `tree`.
pub(super) fn build<I, E>(
    tree: &ControlTree,
    cfg: &Cfg<I, E>,
) -> Result<SplDecomposition, SplError> {
    let named = scan(tree, cfg)?;
    let reached = reachable(cfg, [cfg.entry()], Direction::Forward);
    if let Some(dead) = cfg.block_ids().find(|block| !reached[block.index()]) {
        return Err(SplError::UnplacedBlock(dead));
    }
    let exits: Vec<BlockId> = cfg
        .block_ids()
        .filter(|&block| cfg.outgoing(block).next().is_none())
        .collect();
    if exits.len() > 1 {
        let last = final_block(tree);
        let early = exits
            .iter()
            .copied()
            .find(|&exit| Some(exit) != last)
            .expect("of two exit blocks, one is not the last statement");
        return Err(SplError::EarlyReturn(early));
    }

    let mut builder = Builder::new(tree, cfg, named);
    let root = builder.run()?;
    if let Some(unplaced) = cfg.block_ids().find(|block| !builder.placed[block.index()]) {
        return Err(SplError::UnplacedBlock(unplaced));
    }
    let decomposition = builder.finish(root);
    if let Some(&exit) = exits.first()
        && decomposition.blocks.last() != Some(&exit)
    {
        return Err(SplError::EarlyReturn(exit));
    }
    Ok(decomposition)
}

/// Find the first construct in preorder with no counterpart in the grammar,
/// and mark the blocks the tree has a leaf, a test or a header for.
fn scan<I, E>(tree: &ControlTree, cfg: &Cfg<I, E>) -> Result<Vec<bool>, SplError> {
    let last = final_statement(tree);
    let mut named = vec![false; cfg.block_bound()];
    for id in tree.preorder() {
        match *tree.node(id) {
            ControlNode::Block(span) => named[span.block().index()] = true,
            ControlNode::Return(span) => {
                if Some(id) != last {
                    return Err(SplError::EarlyReturn(span.block()));
                }
                named[span.block().index()] = true;
            }
            ControlNode::IfThenElse { condition, .. } | ControlNode::Switch { condition, .. } => {
                named[condition.block().index()] = true;
            }
            ControlNode::Loop { header, kind, .. } => {
                named[header.index()] = true;
                match kind {
                    LoopKind::While { condition, .. } => {
                        for span in tree.spans(condition) {
                            named[span.block().index()] = true;
                        }
                    }
                    LoopKind::DoWhile { latch, .. } => named[latch.block().index()] = true,
                    LoopKind::Endless => {}
                }
            }
            ControlNode::Break { label: Some(_) } | ControlNode::Continue { label: Some(_) } => {
                return Err(SplError::MultiLevelJump(id));
            }
            ControlNode::Label { block, body } => {
                if loop_label(tree, block, body).is_none() {
                    return Err(SplError::Label(id));
                }
            }
            ControlNode::Goto { .. } => return Err(SplError::Goto(id)),
            ControlNode::TryCatch { .. } => return Err(SplError::TryCatch(id)),
            ControlNode::Sequence(_) | ControlNode::Break { .. } | ControlNode::Continue { .. } => {
            }
        }
    }
    Ok(named)
}

/// The loop a label holds, when the label holds just a loop it names: the
/// structurer labels a loop that a multi-level jump leaves, and the jump is
/// the obstacle.
fn loop_label(tree: &ControlTree, block: BlockId, body: ControlBody) -> Option<ControlNodeId> {
    match *tree.body(body) {
        [only] if matches!(*tree.node(only), ControlNode::Loop { header, .. } if header == block) => {
            Some(only)
        }
        _ => None,
    }
}

/// The last statement of the procedure.
fn final_statement(tree: &ControlTree) -> Option<ControlNodeId> {
    let root = tree.root();
    match *tree.node(root) {
        ControlNode::Sequence(body) => tree.body(body).last().copied(),
        _ => Some(root),
    }
}

/// The block of the last statement of the procedure, when it is a leaf.
fn final_block(tree: &ControlTree) -> Option<BlockId> {
    match *tree.node(final_statement(tree)?) {
        ControlNode::Block(span) | ControlNode::Return(span) => Some(span.block()),
        _ => None,
    }
}

/// Which blocks without a leaf may come before the next statement.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Settle {
    /// Any: a statement with a block of its own follows.
    Any,
    /// Any but an exit: a `break` or `continue` follows, and an exit is the
    /// last statement of the procedure, so it waits until after the loop.
    BeforeJump,
}

struct Builder<'a, I, E> {
    tree: &'a ControlTree,
    cfg: &'a Cfg<I, E>,
    nodes: Vec<SplNode>,
    children: Vec<SplNodeId>,
    /// The blocks the tree has a leaf, a test or a header for.
    named: Vec<bool>,
    /// The blocks with a leaf.
    placed: Vec<bool>,
    /// The number of edges into each block.
    in_count: Vec<u32>,
    /// Whether each enclosing loop is post-tested, innermost last: there, a
    /// `continue` goes to the latch, which follows the body's statements.
    post_tested: Vec<bool>,
}

impl<'a, I, E> Builder<'a, I, E> {
    fn new(tree: &'a ControlTree, cfg: &'a Cfg<I, E>, named: Vec<bool>) -> Self {
        let mut in_count = vec![0_u32; cfg.block_bound()];
        for edge in cfg.edges() {
            in_count[edge.target().index()] += 1;
        }
        Self {
            tree,
            cfg,
            nodes: Vec::new(),
            children: Vec::new(),
            named,
            placed: vec![false; cfg.block_bound()],
            in_count,
            post_tested: Vec::new(),
        }
    }

    /// Build the whole procedure and return its node.
    fn run(&mut self) -> Result<SplNodeId, SplError> {
        let tree = self.tree;
        let root = tree.root();
        let cursor = match *tree.node(root) {
            ControlNode::Sequence(body) => Cursor::Run(tree.body(body), 0),
            _ => Cursor::One(root),
        };
        let mut first = BodyFrame::new(Some(cursor), Stubs::NONE);
        let entry = self.cfg.entry();
        if !self.named[entry.index()] {
            self.leaf(&mut first.piece, entry)?;
        }
        let mut stack = vec![Frame::Body(first)];
        let mut built: Option<Built> = None;
        while let Some(frame) = stack.pop() {
            built = match frame {
                Frame::Body(mut body) => {
                    if let Some(Built::Node(node, exits)) = built.take() {
                        body.piece.statements.push(node);
                        body.piece.exits.falls = Stubs::NONE;
                        body.piece.exits.absorb(exits);
                    }
                    self.step_body(body, &mut stack)?
                }
                Frame::Branch(mut branch) => {
                    if let Some(Built::Piece(arm)) = built.take() {
                        branch.arms.push(arm);
                    }
                    self.step_branch(branch, &mut stack)?
                }
                Frame::Loop(header) => {
                    let Some(Built::Piece(body)) = built.take() else {
                        unreachable!("a loop resumes when its body is built");
                    };
                    Some(self.finish_loop(header, body)?)
                }
            };
        }
        let Some(Built::Piece(mut procedure)) = built else {
            unreachable!("the root body is built last");
        };
        // What falls out of the procedure ends at its exit.
        self.settle(&mut procedure, Settle::Any)?;
        Ok(self.seal(&procedure.statements))
    }

    /// Build the next statement of `body`, pushing it back with the frames
    /// the statement opens, or return the finished body.
    fn step_body<'t>(
        &mut self,
        mut body: BodyFrame<'t>,
        stack: &mut Vec<Frame<'t>>,
    ) -> Result<Option<Built>, SplError>
    where
        'a: 't,
    {
        let tree = self.tree;
        let Some(id) = body.next_statement() else {
            let Some((latch, continue_on_true)) = body.latch.take() else {
                return Ok(Some(Built::Piece(body.piece)));
            };
            self.settle(&mut body.piece, Settle::Any)?;
            self.leaf(&mut body.piece, latch)?;
            let continue_arm = (Arm::Body(None), self.branch_stubs(latch, continue_on_true));
            let exit_arm = (Arm::Break, self.branch_stubs(latch, !continue_on_true));
            let arms = if continue_on_true {
                [continue_arm, exit_arm]
            } else {
                [exit_arm, continue_arm]
            };
            stack.push(Frame::Body(body));
            stack.push(Frame::Branch(BranchFrame::new(latch, arms.into())));
            return Ok(None);
        };
        let node = *tree.node(id);
        let settle = match node {
            ControlNode::Sequence(run) => {
                body.cursors.push(Cursor::Run(tree.body(run), 0));
                stack.push(Frame::Body(body));
                return Ok(None);
            }
            ControlNode::Label {
                block,
                body: labeled,
            } => {
                let inner = loop_label(tree, block, labeled)
                    .expect("the scan rejects a label that is not a loop's");
                body.cursors.push(Cursor::One(inner));
                stack.push(Frame::Body(body));
                return Ok(None);
            }
            ControlNode::Break { .. } | ControlNode::Continue { .. } => Settle::BeforeJump,
            _ => Settle::Any,
        };
        self.settle(&mut body.piece, settle)?;
        let opened = self.statement(&mut body.piece, node)?;
        match opened {
            Opened::Nothing => stack.push(Frame::Body(body)),
            Opened::Branch(branch) => {
                stack.push(Frame::Body(body));
                stack.push(Frame::Branch(branch));
            }
            Opened::Loop(header, kind, loop_body) => {
                let entering = core::mem::replace(&mut body.piece.exits.falls, Stubs::NONE);
                stack.push(Frame::Body(body));
                self.open_loop(header, kind, loop_body, entering, stack)?;
            }
        }
        Ok(None)
    }

    /// Append a statement other than a nested body to `piece`, and return the
    /// construct it opens.
    fn statement(&mut self, piece: &mut Piece, node: ControlNode) -> Result<Opened, SplError> {
        let tree = self.tree;
        Ok(match node {
            ControlNode::Block(span) | ControlNode::Return(span) => {
                self.leaf(piece, span.block())?;
                Opened::Nothing
            }
            ControlNode::Break { .. } => {
                let node = self.push(SplNode::Break);
                piece.statements.push(node);
                piece.exits.breaks = piece.exits.breaks.join(piece.exits.falls);
                piece.exits.falls = Stubs::NONE;
                Opened::Nothing
            }
            ControlNode::Continue { .. } => {
                // In a post-tested loop the continue point is the latch, which
                // follows the statements: falling there is the continue.
                if self.post_tested.last() != Some(&true) {
                    let node = self.push(SplNode::Continue);
                    piece.statements.push(node);
                    piece.exits.continues = piece.exits.continues.join(piece.exits.falls);
                    piece.exits.falls = Stubs::NONE;
                }
                Opened::Nothing
            }
            ControlNode::IfThenElse {
                condition,
                then_body,
                else_body,
            } => {
                let branch = condition.block();
                self.leaf(piece, branch)?;
                let mut arms = vec![
                    (Arm::Body(Some(then_body)), self.branch_stubs(branch, true)),
                    (Arm::Body(Some(else_body)), self.branch_stubs(branch, false)),
                ];
                self.merge_jump_arms(&mut arms);
                Opened::Branch(BranchFrame::new(branch, arms))
            }
            ControlNode::Switch {
                condition,
                cases,
                default_body,
                default_edge,
            } => {
                let branch = condition.block();
                self.leaf(piece, branch)?;
                let mut arms: Vec<(Arm, Stubs)> = tree
                    .cases(cases)
                    .iter()
                    .map(|case| {
                        let edges = tree.edges(case.edges).iter().copied();
                        (Arm::Body(Some(case.body)), Stubs::of(self.cfg, edges))
                    })
                    .collect();
                if default_edge.is_some() || !default_body.is_empty() {
                    let stubs = Stubs::of(self.cfg, default_edge);
                    arms.push((Arm::Body(Some(default_body)), stubs));
                }
                self.merge_jump_arms(&mut arms);
                Opened::Branch(BranchFrame::new(branch, arms))
            }
            ControlNode::Loop { header, kind, body } => Opened::Loop(header, kind, body),
            ControlNode::Sequence(_)
            | ControlNode::Label { .. }
            | ControlNode::Goto { .. }
            | ControlNode::TryCatch { .. } => {
                unreachable!("nested bodies are read by the caller, and the scan rejects the rest")
            }
        })
    }

    /// Push the frames of a loop entered over `entering`: the loop, its body,
    /// and for a pre-tested loop the test at the end of the condition chain.
    fn open_loop<'t>(
        &mut self,
        header: BlockId,
        kind: LoopKind,
        loop_body: ControlBody,
        entering: Stubs,
        stack: &mut Vec<Frame<'t>>,
    ) -> Result<(), SplError>
    where
        'a: 't,
    {
        let tree = self.tree;
        self.post_tested
            .push(matches!(kind, LoopKind::DoWhile { .. }));
        stack.push(Frame::Loop(header));
        match kind {
            LoopKind::While {
                condition,
                exit_on_true,
            } => {
                let mut inner = BodyFrame::new(None, entering);
                for span in tree.spans(condition) {
                    self.leaf(&mut inner.piece, span.block())?;
                }
                let test = tree
                    .spans(condition)
                    .last()
                    .map_or(header, |span| span.block());
                let iterate = (
                    Arm::Body(Some(loop_body)),
                    self.branch_stubs(test, !exit_on_true),
                );
                let exit = (Arm::Break, self.branch_stubs(test, exit_on_true));
                let arms = if exit_on_true {
                    [exit, iterate]
                } else {
                    [iterate, exit]
                };
                stack.push(Frame::Body(inner));
                stack.push(Frame::Branch(BranchFrame::new(test, arms.into())));
            }
            LoopKind::DoWhile {
                latch,
                continue_on_true,
            } => {
                // A one-block post-tested loop has the header as its latch,
                // which the latch's test places.
                let latch = latch.block();
                let mut inner = self.header_body(header, loop_body, entering, latch == header)?;
                inner.latch = Some((latch, continue_on_true));
                stack.push(Frame::Body(inner));
            }
            LoopKind::Endless => {
                let inner = self.header_body(header, loop_body, entering, false)?;
                stack.push(Frame::Body(inner));
            }
        }
        Ok(())
    }

    /// The body of a post-tested or endless loop, which holds the header's
    /// own statement first. A header without code that does not branch has no
    /// statement in the tree, so its leaf comes first here, unless the header
    /// is the latch.
    fn header_body<'t>(
        &mut self,
        header: BlockId,
        loop_body: ControlBody,
        entering: Stubs,
        header_is_latch: bool,
    ) -> Result<BodyFrame<'t>, SplError>
    where
        'a: 't,
    {
        let tree = self.tree;
        let statements = tree.body(loop_body);
        let mut inner = BodyFrame::new(Some(Cursor::Run(statements, 0)), entering);
        let mut first = statements.first().copied();
        while let Some(ControlNode::Sequence(run)) = first.map(|id| *tree.node(id)) {
            first = tree.body(run).first().copied();
        }
        let leads = first.is_some_and(|id| match *tree.node(id) {
            ControlNode::Block(span) | ControlNode::Return(span) => span.block() == header,
            ControlNode::IfThenElse { condition, .. } | ControlNode::Switch { condition, .. } => {
                condition.block() == header
            }
            _ => false,
        });
        if !leads && !header_is_latch {
            self.leaf(&mut inner.piece, header)?;
        }
        Ok(inner)
    }

    /// The `Loop` of `header` around `body`, whose falling edges go back to
    /// the header and whose breaks leave the loop.
    fn finish_loop(&mut self, header: BlockId, mut body: Piece) -> Result<Built, SplError> {
        self.settle(&mut body, Settle::Any)?;
        self.post_tested.pop();
        let body_node = self.seal(&body.statements);
        let node = self.push(SplNode::Loop {
            header,
            body: body_node,
        });
        Ok(Built::Node(node, Exits::falling(body.exits.breaks)))
    }

    /// Build the next arm of `branch`, pushing it back with the arm's frame,
    /// or return the finished `Parallel`.
    fn step_branch<'t>(
        &mut self,
        mut branch: BranchFrame,
        stack: &mut Vec<Frame<'t>>,
    ) -> Result<Option<Built>, SplError>
    where
        'a: 't,
    {
        let tree = self.tree;
        match branch.pending.pop() {
            Some((Arm::Break, stubs)) => {
                // A test's way out may pass blocks without code on the way.
                let mut arm = Piece::entered(stubs);
                self.settle(&mut arm, Settle::BeforeJump)?;
                arm.statements.push(self.push(SplNode::Break));
                arm.exits.breaks = arm.exits.falls;
                arm.exits.falls = Stubs::NONE;
                branch.arms.push(arm);
                stack.push(Frame::Branch(branch));
                Ok(None)
            }
            Some((Arm::Body(arm), stubs)) => {
                let cursor = arm.map(|body| Cursor::Run(tree.body(body), 0));
                stack.push(Frame::Branch(branch));
                stack.push(Frame::Body(BodyFrame::new(cursor, stubs)));
                Ok(None)
            }
            None => self.finish_branch(branch).map(Some),
        }
    }

    /// The `Parallel` of the arms of `branch`.
    ///
    /// What falls out of the arms leaves the parallel when it all enters one
    /// block without a leaf, which then follows the parallel. Otherwise an arm
    /// ends with such a block when its falling edges alone enter it.
    fn finish_branch(&mut self, mut branch: BranchFrame) -> Result<Built, SplError> {
        let together = branch
            .arms
            .iter()
            .fold(Stubs::NONE, |falls, arm| falls.join(arm.exits.falls));
        if !self.enters_unplaced(together) {
            for arm in &mut branch.arms {
                self.settle(arm, Settle::Any)?;
            }
        }
        let mut exits = Exits::falling(Stubs::NONE);
        let mut nodes = Vec::with_capacity(branch.arms.len());
        for arm in &branch.arms {
            exits.absorb(arm.exits);
            nodes.push(self.seal(&arm.statements));
        }
        let arms = self.run_of(&nodes);
        let node = self.push(SplNode::Parallel {
            branch: branch.branch,
            arms,
        });
        Ok(Built::Node(node, exits))
    }

    /// Make one arm of the arms that do the same thing without a block of
    /// their own (fall through, `break`, or `continue`) and whose edges
    /// together, and only together, are every edge into a block without a
    /// leaf. That block then begins the one arm, and the grammar merges the
    /// parallel edges into it.
    fn merge_jump_arms(&self, arms: &mut Vec<(Arm, Stubs)>) {
        let tree = self.tree;
        let shape = |arm: &Arm| match *arm {
            Arm::Body(None) => Some(0),
            Arm::Body(Some(body)) => match *tree.body(body) {
                [] => Some(0),
                [only] => match *tree.node(only) {
                    ControlNode::Break { label: None } => Some(1),
                    ControlNode::Continue { label: None } => Some(2),
                    _ => None,
                },
                _ => None,
            },
            Arm::Break => None,
        };
        let key = |(arm, stubs): &(Arm, Stubs)| match stubs.target {
            Target::One(target) => shape(arm).map(|shape| (shape, target)),
            _ => None,
        };
        let mut position = 0;
        while position < arms.len() {
            let Some(group) = key(&arms[position]) else {
                position += 1;
                continue;
            };
            let together = arms
                .iter()
                .filter(|arm| key(arm) == Some(group))
                .fold(Stubs::NONE, |stubs, arm| stubs.join(arm.1));
            if together.count > arms[position].1.count && self.enters_unplaced(together) {
                arms[position].1 = together;
                let mut index = position + 1;
                while index < arms.len() {
                    if key(&arms[index]) == Some(group) {
                        arms.remove(index);
                    } else {
                        index += 1;
                    }
                }
            }
            position += 1;
        }
    }

    /// Append the leaf of `block` to `piece`: the edges falling into the
    /// piece end there, and the block's own edges fall out.
    fn leaf(&mut self, piece: &mut Piece, block: BlockId) -> Result<(), SplError> {
        let placed = &mut self.placed[block.index()];
        if *placed {
            return Err(SplError::UnplacedBlock(block));
        }
        *placed = true;
        let node = self.push(SplNode::Block(block));
        piece.statements.push(node);
        let cfg = self.cfg;
        let flow = cfg
            .outgoing(block)
            .filter(|&edge| !cfg.edge(edge).kind().is_exceptional());
        piece.exits.falls = Stubs::of(cfg, flow);
        Ok(())
    }

    /// Whether `stubs` are every edge into one block without a leaf.
    fn enters_unplaced(&self, stubs: Stubs) -> bool {
        let Target::One(block) = stubs.target else {
            return false;
        };
        let index = block.index();
        !self.named[index] && !self.placed[index] && stubs.count == self.in_count[index]
    }

    /// Append to `piece` the blocks without a leaf that flow enters from it
    /// alone, of the kind `settle` allows.
    fn settle(&mut self, piece: &mut Piece, settle: Settle) -> Result<(), SplError> {
        while self.enters_unplaced(piece.exits.falls) {
            let Target::One(next) = piece.exits.falls.target else {
                unreachable!("an unplaced block is entered at one target");
            };
            if settle == Settle::BeforeJump && self.cfg.outgoing(next).next().is_none() {
                break;
            }
            self.leaf(piece, next)?;
        }
        Ok(())
    }

    /// The edges of `block` of the given conditional kind.
    fn branch_stubs(&self, block: BlockId, on_true: bool) -> Stubs {
        let kind = if on_true {
            EdgeKind::ConditionalTrue
        } else {
            EdgeKind::ConditionalFalse
        };
        let cfg = self.cfg;
        Stubs::of(
            cfg,
            cfg.outgoing(block)
                .filter(|&edge| cfg.edge(edge).kind() == kind),
        )
    }

    fn push(&mut self, node: SplNode) -> SplNodeId {
        let id = SplNodeId::from_index(self.nodes.len());
        self.nodes.push(node);
        id
    }

    fn run_of(&mut self, nodes: &[SplNodeId]) -> SplRun {
        let run = SplRun {
            start: u32::try_from(self.children.len()).expect("SPL child count fits in u32"),
            len: u32::try_from(nodes.len()).expect("SPL child count fits in u32"),
        };
        self.children.extend_from_slice(nodes);
        run
    }

    /// The node of a body: its one statement, or a series of them.
    fn seal(&mut self, statements: &[SplNodeId]) -> SplNodeId {
        if let &[only] = statements {
            return only;
        }
        let children = self.run_of(statements);
        self.push(SplNode::Series { children })
    }

    /// The finished decomposition rooted at `root`, with its leaves in
    /// preorder and its nesting depth.
    fn finish(self, root: SplNodeId) -> SplDecomposition {
        let mut blocks = Vec::new();
        let mut nesting_depth = 0;
        let mut stack = vec![(root, 0_usize)];
        while let Some((id, depth)) = stack.pop() {
            let (children, inner) = match self.nodes[id.index()] {
                SplNode::Block(block) => {
                    blocks.push(block);
                    continue;
                }
                SplNode::Series { children } => (children, depth),
                SplNode::Parallel { arms, .. } => (arms, depth + 1),
                SplNode::Loop { body, .. } => {
                    nesting_depth = nesting_depth.max(depth + 1);
                    stack.push((body, depth + 1));
                    continue;
                }
                SplNode::Break | SplNode::Continue => continue,
            };
            nesting_depth = nesting_depth.max(inner);
            let start = children.start as usize;
            let run = &self.children[start..start + children.len()];
            stack.extend(run.iter().rev().map(|&child| (child, inner)));
        }
        SplDecomposition {
            nodes: self.nodes,
            children: self.children,
            root,
            blocks,
            nesting_depth,
        }
    }
}

/// The construct a statement opens.
enum Opened {
    Nothing,
    Branch(BranchFrame),
    Loop(BlockId, LoopKind, ControlBody),
}
