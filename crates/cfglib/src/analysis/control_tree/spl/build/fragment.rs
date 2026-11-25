//! The pieces the builder works with: the edges that leave what is built so
//! far, the statements of a body, and the frames of its explicit stack.

extern crate alloc;
use alloc::vec;
use alloc::vec::Vec;

use crate::analysis::control_tree::node::{ControlBody, ControlNodeId};
use crate::analysis::control_tree::spl::SplNodeId;
use crate::block::BlockId;
use crate::cfg::Cfg;
use crate::edge::EdgeId;

/// Edges that fall out of what is built so far, or break or continue: how
/// many, and the block they all end at.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct Stubs {
    pub(super) count: u32,
    pub(super) target: Target,
}

/// The block a set of edges ends at.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Target {
    /// There is no edge.
    Nothing,
    /// Every edge ends at this block.
    One(BlockId),
    /// The edges end at different blocks.
    Mixed,
}

impl Stubs {
    pub(super) const NONE: Self = Self {
        count: 0,
        target: Target::Nothing,
    };

    /// The edges of `edges`.
    pub(super) fn of<I, E>(cfg: &Cfg<I, E>, edges: impl IntoIterator<Item = EdgeId>) -> Self {
        edges.into_iter().fold(Self::NONE, |stubs, edge| {
            stubs.join(Self {
                count: 1,
                target: Target::One(cfg.edge(edge).target()),
            })
        })
    }

    /// The edges of both.
    pub(super) fn join(self, other: Self) -> Self {
        let target = match (self.target, other.target) {
            (Target::Nothing, target) | (target, Target::Nothing) => target,
            (Target::One(a), Target::One(b)) if a == b => Target::One(a),
            _ => Target::Mixed,
        };
        Self {
            count: self.count + other.count,
            target,
        }
    }
}

/// The edges a fragment leaves through.
#[derive(Debug, Clone, Copy)]
pub(super) struct Exits {
    pub(super) falls: Stubs,
    pub(super) breaks: Stubs,
    pub(super) continues: Stubs,
}

impl Exits {
    pub(super) const fn falling(falls: Stubs) -> Self {
        Self {
            falls,
            breaks: Stubs::NONE,
            continues: Stubs::NONE,
        }
    }

    /// Add what leaves a part that falls out where this one does.
    pub(super) fn absorb(&mut self, part: Self) {
        self.falls = self.falls.join(part.falls);
        self.breaks = self.breaks.join(part.breaks);
        self.continues = self.continues.join(part.continues);
    }
}

/// Statements built in order, and the edges that leave them.
pub(super) struct Piece {
    pub(super) statements: Vec<SplNodeId>,
    pub(super) exits: Exits,
}

impl Piece {
    /// No statement yet, entered over `falls`.
    pub(super) const fn entered(falls: Stubs) -> Self {
        Self {
            statements: Vec::new(),
            exits: Exits::falling(falls),
        }
    }
}

/// Where the statements of one arm come from.
#[derive(Debug, Clone, Copy)]
pub(super) enum Arm {
    /// The statements of a body of the tree, none for an empty arm.
    Body(Option<ControlBody>),
    /// A loop test's way out: the arm breaks.
    Break,
}

/// The statements a body reads.
pub(super) enum Cursor<'t> {
    Run(&'t [ControlNodeId], usize),
    One(ControlNodeId),
}

/// One body being built into a sequence.
pub(super) struct BodyFrame<'t> {
    /// The statement lists still being read, innermost (a nested `Sequence`)
    /// last.
    pub(super) cursors: Vec<Cursor<'t>>,
    pub(super) piece: Piece,
    /// A post-tested loop's latch and whether its true edge continues, tested
    /// after the statements.
    pub(super) latch: Option<(BlockId, bool)>,
}

impl<'t> BodyFrame<'t> {
    pub(super) fn new(cursor: Option<Cursor<'t>>, falls: Stubs) -> Self {
        Self {
            cursors: cursor.into_iter().collect(),
            piece: Piece::entered(falls),
            latch: None,
        }
    }

    /// The next statement of the tree, if any.
    pub(super) fn next_statement(&mut self) -> Option<ControlNodeId> {
        while let Some(cursor) = self.cursors.last_mut() {
            match cursor {
                Cursor::Run(run, position) => {
                    if let Some(&next) = run.get(*position) {
                        *position += 1;
                        return Some(next);
                    }
                }
                &mut Cursor::One(one) => {
                    self.cursors.pop();
                    return Some(one);
                }
            }
            self.cursors.pop();
        }
        None
    }
}

/// One `Parallel` being built, an arm at a time.
pub(super) struct BranchFrame {
    pub(super) branch: BlockId,
    /// The arms still to build, last first, with the edges into each.
    pub(super) pending: Vec<(Arm, Stubs)>,
    /// The arms built, not yet sealed into nodes.
    pub(super) arms: Vec<Piece>,
}

impl BranchFrame {
    /// A `Parallel` of `branch` to build from `arms`, given in order.
    pub(super) fn new(branch: BlockId, mut arms: Vec<(Arm, Stubs)>) -> Self {
        arms.reverse();
        Self {
            branch,
            pending: arms,
            arms: vec![],
        }
    }
}

/// One frame of the builder's stack.
pub(super) enum Frame<'t> {
    Body(BodyFrame<'t>),
    Branch(BranchFrame),
    /// A loop, by its header, whose body is being built.
    Loop(BlockId),
}

/// What a finished frame hands to the one below it.
pub(super) enum Built {
    /// A body's statements, not yet sealed into a node.
    Piece(Piece),
    /// A `Parallel` or `Loop` node.
    Node(SplNodeId, Exits),
}
