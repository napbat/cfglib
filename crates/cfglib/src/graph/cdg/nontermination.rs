//! Nontermination-sensitive control dependence, one dependent at a time.
//!
//! A block `n` is nontermination-sensitively control dependent on a
//! predicate `p`, a block with at least two distinct successors, when `p`
//! has a successor from which every maximal path passes `n` and another from
//! which some maximal path does not (Ranganath et al.). A maximal path ends
//! at a block without successors or runs forever, so a loop that never
//! reaches `n` is a way around it: unlike post-dominance, the relation needs
//! no exit, and a loop's exit test controls what follows the loop.
//!
//! Chalupa, Klaška, Strejček and Tomovič ("Fast Computation of Strong
//! Control Dependencies", CAV 2021, algorithm 2) find the controllers of one
//! `n` by coloring backward from it: a block turns red when all its
//! successors are red, and it is then exactly a block every maximal path from
//! which passes `n`. A predicate is a controller when it has a red successor
//! and one that is not. Each edge is read at most twice, so one dependent
//! costs O(n + m), and the whole relation, which can be quadratic, O(n·(n +
//! m)).

extern crate alloc;
use alloc::vec::Vec;

use crate::graph::view::{DenseId, GraphView};

/// The buffers [`nontermination_sensitive_controllers`] fills, owned by the
/// caller.
///
/// Only the blocks a call colors or counts are touched, and each is restored
/// before the call returns, so a call that reaches a few blocks of a large
/// graph costs only those blocks once the tables have grown.
#[derive(Debug, Clone, Default)]
pub struct NonterminationScratch {
    /// How many successor edges of each counted block are not red yet.
    pending: Vec<u32>,
    /// Whether each block has been counted, and whether it is red.
    counted: Vec<bool>,
    red: Vec<bool>,
    /// `node`, then every block counted, each once, to read and restore.
    touched: Vec<usize>,
    /// Red blocks whose predecessors are yet to be counted down.
    stack: Vec<usize>,
}

impl NonterminationScratch {
    /// Scratch holding nothing yet.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }
}

/// Write to `out` the predicates that `node` is nontermination-sensitively
/// control dependent on, ascending by dense index.
///
/// A predicate `p` controls `node` when `p` has two distinct successors, one
/// from which every maximal path passes `node` and one from which some
/// maximal path does not. A maximal path ends at a block without successors
/// or runs forever. Any directed graph will do: there is no root, no exit and
/// no post-dominator tree, and `node` controls itself when it has a
/// successor that leads back only to it and another that need not.
///
/// The call colors backward from `node` (Chalupa et al., CAV 2021), so it
/// costs O(n + m) at most and only the part of the graph that leads to
/// `node`. The whole relation, one call per block, is O(n·(n + m)) and can
/// itself be quadratic, so there is deliberately no builder of it.
///
/// `out` is cleared first. Parallel edges and self-loops are read as they
/// are: a block whose only successor is itself never passes another block.
///
/// # Panics
///
/// Panics when `node`, or a block a successor or predecessor list names, has
/// an index beyond the graph's node bound.
///
/// # Examples
///
/// The head of a loop that may run forever decides whether its body runs on
/// either arm, so each arm depends on the head; nothing depends on the latch,
/// which has one successor:
///
/// ```
/// use cfglib::{Cfg, EdgeKind, NonterminationScratch, nontermination_sensitive_controllers};
///
/// let mut cfg = Cfg::<u32>::new();
/// let entry = cfg.entry();
/// let head = cfg.new_block();
/// let ok = cfg.new_block();
/// let err = cfg.new_block();
/// let latch = cfg.new_block();
/// cfg.add_edge(entry, head, EdgeKind::Unconditional);
/// cfg.add_edge(head, ok, EdgeKind::ConditionalTrue);
/// cfg.add_edge(head, err, EdgeKind::ConditionalFalse);
/// cfg.add_edge(ok, latch, EdgeKind::Unconditional);
/// cfg.add_edge(err, latch, EdgeKind::Unconditional);
/// cfg.add_edge(latch, head, EdgeKind::Back);
///
/// let mut scratch = NonterminationScratch::new();
/// let mut controllers = Vec::new();
/// nontermination_sensitive_controllers(&cfg, ok, &mut scratch, &mut controllers);
/// assert_eq!(controllers, [head]);
/// nontermination_sensitive_controllers(&cfg, latch, &mut scratch, &mut controllers);
/// assert!(controllers.is_empty());
/// ```
pub fn nontermination_sensitive_controllers<G: GraphView>(
    graph: &G,
    node: G::NodeId,
    scratch: &mut NonterminationScratch,
    out: &mut Vec<G::NodeId>,
) {
    out.clear();
    let bound = graph.node_bound();
    assert!(
        node.index() < bound,
        "the dependent is not a node of the graph"
    );
    let NonterminationScratch {
        pending,
        counted,
        red,
        touched,
        stack,
    } = scratch;
    if pending.len() < bound {
        pending.resize(bound, 0);
        counted.resize(bound, false);
        red.resize(bound, false);
    }

    let start = node.index();
    red[start] = true;
    touched.push(start);
    stack.push(start);
    while let Some(block) = stack.pop() {
        for before in graph.predecessors(G::NodeId::from_index(block)) {
            let index = before.index();
            if red[index] {
                continue;
            }
            if !counted[index] {
                counted[index] = true;
                touched.push(index);
                pending[index] = u32::try_from(graph.successors(before).count())
                    .expect("a block's successor count fits in u32");
            }
            pending[index] -= 1;
            if pending[index] == 0 {
                red[index] = true;
                stack.push(index);
            }
        }
    }

    // A block is counted only when a red successor counts it down, so one
    // that stays uncolored has a red successor and one that is not: it is a
    // predicate, and a controller. A red block has only red successors,
    // except `node`, which is red by definition and is read directly.
    let (reached, missed) = graph
        .successors(node)
        .fold((false, false), |(reached, missed), next| {
            (reached || red[next.index()], missed || !red[next.index()])
        });
    if reached && missed {
        out.push(node);
    }
    out.extend(
        touched[1..]
            .iter()
            .filter(|&&block| !red[block])
            .map(|&block| G::NodeId::from_index(block)),
    );
    out.sort_unstable_by_key(|block| block.index());

    for &block in touched.iter() {
        counted[block] = false;
        red[block] = false;
    }
    touched.clear();
}

#[cfg(test)]
mod tests;
