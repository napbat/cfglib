//! The checkers of a decomposition, written from the definitions alone.
//!
//! The validity checker expands the decomposition back into a graph through
//! the grammar's CFG construction, recursively and with plain vectors, and
//! compares the result with the graph: the same blocks, each in one leaf, and
//! the same edges, parallel edges counted once. The path checker reads the
//! definition of a path decomposition.

extern crate alloc;
use alloc::collections::BTreeSet;
use alloc::vec::Vec;

use crate::block::BlockId;
use crate::cfg::Cfg;
use crate::{SplDecomposition, SplNode, SplNodeId};

/// The edges falling, breaking and continuing out of a fragment, by the
/// blocks they leave.
type Exits = (Vec<BlockId>, Vec<BlockId>, Vec<BlockId>);

/// The grammar's CFG construction of one decomposition.
struct Expansion<'d> {
    spl: &'d SplDecomposition,
    edges: BTreeSet<(BlockId, BlockId)>,
    leaves: Vec<BlockId>,
}

impl Expansion<'_> {
    fn expand(&mut self, id: SplNodeId, incoming: Vec<BlockId>) -> Exits {
        match *self.spl.node(id) {
            SplNode::Block(block) => {
                for source in incoming {
                    self.edges.insert((source, block));
                }
                self.leaves.push(block);
                (alloc::vec![block], Vec::new(), Vec::new())
            }
            SplNode::Series { children } => {
                let (mut falls, mut breaks, mut continues) = (incoming, Vec::new(), Vec::new());
                for &child in self.spl.children(children) {
                    let (next, more_breaks, more_continues) = self.expand(child, falls);
                    falls = next;
                    breaks.extend(more_breaks);
                    continues.extend(more_continues);
                }
                (falls, breaks, continues)
            }
            SplNode::Parallel { branch, arms } => {
                assert_eq!(
                    incoming,
                    [branch],
                    "a parallel takes its branch block's edges"
                );
                assert_eq!(
                    self.leaves.last(),
                    Some(&branch),
                    "the branch leaf comes first"
                );
                let mut exits: Exits = (Vec::new(), Vec::new(), Vec::new());
                for &arm in self.spl.children(arms) {
                    let (falls, breaks, continues) = self.expand(arm, alloc::vec![branch]);
                    exits.0.extend(falls);
                    exits.1.extend(breaks);
                    exits.2.extend(continues);
                }
                exits
            }
            SplNode::Loop { header, body } => {
                let first = self.leaves.len();
                let (falls, breaks, continues) = self.expand(body, incoming);
                assert_eq!(
                    self.leaves.get(first),
                    Some(&header),
                    "a loop's body begins with its header"
                );
                for source in falls.into_iter().chain(continues) {
                    self.edges.insert((source, header));
                }
                (breaks, Vec::new(), Vec::new())
            }
            SplNode::Break => (Vec::new(), incoming, Vec::new()),
            SplNode::Continue => (Vec::new(), Vec::new(), incoming),
        }
    }
}

/// Check that the grammar's construction of `spl` is `cfg`: every block in
/// one leaf, in the order of [`SplDecomposition::blocks`], the entry first,
/// and the same edges. What falls out of the whole procedure leaves an exit,
/// and nothing breaks or continues out of it.
pub(super) fn check_decomposition<I>(spl: &SplDecomposition, cfg: &Cfg<I>, context: &str) {
    let mut expansion = Expansion {
        spl,
        edges: BTreeSet::new(),
        leaves: Vec::new(),
    };
    let (falls, breaks, continues) = expansion.expand(spl.root(), Vec::new());
    assert_eq!(
        expansion.leaves,
        spl.blocks(),
        "{context}: leaves in preorder"
    );
    assert_eq!(
        expansion.leaves.first(),
        Some(&cfg.entry()),
        "{context}: entry first"
    );
    let mut blocks = expansion.leaves.clone();
    blocks.sort_unstable();
    let live: Vec<BlockId> = cfg.block_ids().collect();
    assert_eq!(blocks, live, "{context}: every block in exactly one leaf");
    let wanted: BTreeSet<(BlockId, BlockId)> = cfg
        .edges()
        .map(|edge| (edge.source(), edge.target()))
        .collect();
    assert_eq!(expansion.edges, wanted, "{context}: the edges");
    assert!(
        breaks.is_empty() && continues.is_empty(),
        "{context}: closed"
    );
    for block in falls {
        assert!(
            cfg.outgoing(block).next().is_none(),
            "{context}: {block} ends the procedure"
        );
    }
}

/// Check the path decomposition of `spl` against its definition, and its
/// width against `2·d + 1`.
pub(super) fn check_path<I>(spl: &SplDecomposition, cfg: &Cfg<I>, context: &str) {
    let path = spl.path_decomposition();
    let bags: Vec<BTreeSet<BlockId>> = path
        .bags()
        .map(|bag| {
            let set: BTreeSet<BlockId> = bag.iter().copied().collect();
            assert_eq!(set.len(), bag.len(), "{context}: a bag lists a block once");
            set
        })
        .collect();
    for edge in cfg.edges() {
        assert!(
            bags.iter()
                .any(|bag| bag.contains(&edge.source()) && bag.contains(&edge.target())),
            "{context}: a bag holds both ends of {edge:?}"
        );
    }
    for block in cfg.block_ids() {
        let holding: Vec<usize> = (0..bags.len())
            .filter(|&bag| bags[bag].contains(&block))
            .collect();
        let (first, last) = (holding.first(), holding.last());
        assert!(first.is_some(), "{context}: a bag holds {block}");
        assert_eq!(
            last.zip(first).map(|(last, first)| last - first + 1),
            Some(holding.len()),
            "{context}: the bags holding {block} are consecutive"
        );
    }
    let width = bags.iter().map(BTreeSet::len).max().unwrap_or(1) - 1;
    assert_eq!(path.width(), width, "{context}: width");
    assert!(
        width <= 2 * spl.nesting_depth() + 1,
        "{context}: width {width} at nesting depth {}",
        spl.nesting_depth()
    );
}
