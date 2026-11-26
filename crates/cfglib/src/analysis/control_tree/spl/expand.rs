//! The grammar's CFG construction, run on a decomposition and compared with
//! the graph it came from.
//!
//! The construction follows [`SplDecomposition`]: every node
//! takes the edges falling into it as stubs, blocks whose edge has yet to
//! find its target, and connects them where the grammar says. The stubs live
//! in one arena as linked lists, so joining the stubs of two arms costs O(1),
//! and the nodes are constructed on an explicit stack. The edges made are
//! then grouped by source and compared with the graph's, one source at a
//! time, so the whole check is O(n + m).

extern crate alloc;
use alloc::vec;
use alloc::vec::Vec;

use super::{SplDecomposition, SplError, SplNode, SplNodeId};
use crate::block::BlockId;
use crate::cfg::Cfg;
use crate::graph::epoch::EpochMarks;
use crate::identity::{NO_INDEX, slot, word};

/// Check that the grammar's construction of `decomposition` gives exactly the
/// edges of `cfg`, parallel edges counted once, from the entry.
///
/// The first source in block order whose edges differ is reported: an edge
/// of the graph the construction does not make, or else, for an edge the
/// construction makes that the graph lacks, the block's first edge, or the
/// block as an early exit when it has none.
pub(super) fn verify<I, E>(
    decomposition: &SplDecomposition,
    cfg: &Cfg<I, E>,
) -> Result<(), SplError> {
    if decomposition.blocks.first() != Some(&cfg.entry()) {
        return Err(SplError::UnplacedBlock(cfg.entry()));
    }
    let made = Construction::default().run(decomposition)?;

    // The targets the construction makes, grouped by source.
    let bound = cfg.block_bound();
    let mut starts = vec![0_usize; bound + 1];
    for &(source, _) in &made {
        starts[source.index() + 1] += 1;
    }
    for index in 1..starts.len() {
        starts[index] += starts[index - 1];
    }
    let mut cursor = starts.clone();
    let mut targets = vec![cfg.entry(); made.len()];
    for &(source, target) in &made {
        targets[cursor[source.index()]] = target;
        cursor[source.index()] += 1;
    }

    // Each source marks its targets in the graph and in the construction.
    let mut in_graph = EpochMarks::new(bound);
    let mut in_made = EpochMarks::new(bound);
    for source in cfg.block_ids() {
        let at = source.index();
        in_graph.reset();
        in_made.reset();
        for &target in &targets[starts[at]..starts[at + 1]] {
            in_made.mark(target.index());
        }
        for edge in cfg.outgoing(source) {
            let target = cfg.edge(edge).target();
            if !in_made.is_marked(target.index()) {
                return Err(SplError::UnplacedEdge(edge));
            }
            in_graph.mark(target.index());
        }
        if targets[starts[at]..starts[at + 1]]
            .iter()
            .any(|target| !in_graph.is_marked(target.index()))
        {
            return Err(cfg
                .outgoing(source)
                .next()
                .map_or(SplError::EarlyReturn(source), SplError::UnplacedEdge));
        }
    }
    Ok(())
}

/// A list of stubs, as the first and last of the arena's links, both
/// [`NO_INDEX`] for the empty list.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Stubs {
    head: u32,
    tail: u32,
}

impl Stubs {
    const EMPTY: Self = Self {
        head: NO_INDEX,
        tail: NO_INDEX,
    };
}

/// What a fragment leaves through.
#[derive(Debug, Clone, Copy)]
struct Exits {
    falls: Stubs,
    breaks: Stubs,
    continues: Stubs,
}

impl Exits {
    const NONE: Self = Self {
        falls: Stubs::EMPTY,
        breaks: Stubs::EMPTY,
        continues: Stubs::EMPTY,
    };
}

/// A node whose children are being constructed.
#[derive(Clone, Copy)]
enum Frame<'d> {
    /// A series with the children after `position` still to construct, and
    /// what broke or continued in the ones before.
    Series {
        children: &'d [SplNodeId],
        position: usize,
        exits: Exits,
    },
    /// A parallel with the arms after `position` still to construct, and what
    /// left the ones before.
    Parallel {
        branch: BlockId,
        arms: &'d [SplNodeId],
        position: usize,
        exits: Exits,
    },
    /// A loop whose body is being constructed.
    Loop(BlockId),
}

/// The stub arena and the edges made so far.
#[derive(Default)]
struct Construction {
    /// The block each stub leaves.
    block: Vec<BlockId>,
    /// The stub after each one in its list.
    next: Vec<u32>,
    made: Vec<(BlockId, BlockId)>,
}

impl Construction {
    /// The edges the grammar makes for the whole decomposition.
    fn run(
        mut self,
        decomposition: &SplDecomposition,
    ) -> Result<Vec<(BlockId, BlockId)>, SplError> {
        let mut stack: Vec<Frame<'_>> = Vec::new();
        let (mut node, mut incoming) = (decomposition.root, Stubs::EMPTY);
        loop {
            let exits = match *decomposition.node(node) {
                SplNode::Loop { header, body } => {
                    stack.push(Frame::Loop(header));
                    node = body;
                    continue;
                }
                ref other => self.enter(decomposition, *other, incoming, &mut stack)?,
            };
            let Some(next) = self.resume(&mut stack, exits) else {
                return Ok(self.made);
            };
            (node, incoming) = next;
        }
    }

    /// Construct a node other than a loop from the stubs falling into it: a
    /// leaf at once, a series or parallel by pushing its frame. Return what
    /// leaves the leaf, or what to hand the new frame.
    fn enter<'d>(
        &mut self,
        decomposition: &'d SplDecomposition,
        node: SplNode,
        incoming: Stubs,
        stack: &mut Vec<Frame<'d>>,
    ) -> Result<Exits, SplError> {
        Ok(match node {
            SplNode::Block(block) => {
                self.connect(incoming, block);
                Exits {
                    falls: self.single(block),
                    ..Exits::NONE
                }
            }
            SplNode::Break => Exits {
                breaks: incoming,
                ..Exits::NONE
            },
            SplNode::Continue => Exits {
                continues: incoming,
                ..Exits::NONE
            },
            SplNode::Series { children } => {
                stack.push(Frame::Series {
                    children: decomposition.children(children),
                    position: 0,
                    exits: Exits::NONE,
                });
                // The series hands what falls into it to its first child.
                Exits {
                    falls: incoming,
                    ..Exits::NONE
                }
            }
            SplNode::Parallel { branch, arms } => {
                // A parallel takes the one edge of its branch block that
                // falls into it, and gives each arm one.
                let alone = incoming.head != NO_INDEX
                    && incoming.head == incoming.tail
                    && self.block[slot(incoming.head)] == branch;
                if !alone {
                    return Err(SplError::UnplacedBlock(branch));
                }
                stack.push(Frame::Parallel {
                    branch,
                    arms: decomposition.children(arms),
                    position: 0,
                    exits: Exits::NONE,
                });
                Exits::NONE
            }
            SplNode::Loop { .. } => unreachable!("a loop is entered through its body"),
        })
    }

    /// Hand `exits` down the stack until a frame has a child to construct,
    /// and return that child with the stubs falling into it; `None` once the
    /// stack is empty.
    fn resume(
        &mut self,
        stack: &mut Vec<Frame<'_>>,
        mut exits: Exits,
    ) -> Option<(SplNodeId, Stubs)> {
        loop {
            match stack.pop()? {
                Frame::Series {
                    children,
                    position,
                    exits: mut gathered,
                } => {
                    gathered.breaks = self.join(gathered.breaks, exits.breaks);
                    gathered.continues = self.join(gathered.continues, exits.continues);
                    if let Some(&child) = children.get(position) {
                        stack.push(Frame::Series {
                            children,
                            position: position + 1,
                            exits: gathered,
                        });
                        return Some((child, exits.falls));
                    }
                    exits.breaks = gathered.breaks;
                    exits.continues = gathered.continues;
                }
                Frame::Parallel {
                    branch,
                    arms,
                    position,
                    exits: mut gathered,
                } => {
                    gathered.falls = self.join(gathered.falls, exits.falls);
                    gathered.breaks = self.join(gathered.breaks, exits.breaks);
                    gathered.continues = self.join(gathered.continues, exits.continues);
                    if let Some(&arm) = arms.get(position) {
                        stack.push(Frame::Parallel {
                            branch,
                            arms,
                            position: position + 1,
                            exits: gathered,
                        });
                        return Some((arm, self.single(branch)));
                    }
                    exits = gathered;
                }
                Frame::Loop(header) => {
                    let back = self.join(exits.falls, exits.continues);
                    self.connect(back, header);
                    exits = Exits {
                        falls: exits.breaks,
                        ..Exits::NONE
                    };
                }
            }
        }
    }

    /// A list of one stub leaving `block`.
    fn single(&mut self, block: BlockId) -> Stubs {
        let stub = word(self.block.len());
        self.block.push(block);
        self.next.push(NO_INDEX);
        Stubs {
            head: stub,
            tail: stub,
        }
    }

    /// The stubs of both lists.
    fn join(&mut self, first: Stubs, second: Stubs) -> Stubs {
        if first.head == NO_INDEX {
            return second;
        }
        if second.head != NO_INDEX {
            self.next[slot(first.tail)] = second.head;
        }
        Stubs {
            head: first.head,
            tail: if second.head == NO_INDEX {
                first.tail
            } else {
                second.tail
            },
        }
    }

    /// Make an edge from every stub of `stubs` to `target`.
    fn connect(&mut self, stubs: Stubs, target: BlockId) {
        let mut stub = stubs.head;
        while stub != NO_INDEX {
            self.made.push((self.block[slot(stub)], target));
            stub = self.next[slot(stub)];
        }
    }
}
