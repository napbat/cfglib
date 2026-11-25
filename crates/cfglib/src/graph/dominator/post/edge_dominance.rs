//! Edge post-dominance: whether every path from a block to the virtual exit
//! takes one edge.
//!
//! This is edge dominance on the reverse graph the tree is computed on, read
//! back on the original edges. The edge `from -> to` post-dominates `block`
//! exactly when `from` post-dominates `block` and the edge is the only way out
//! of `from` to a block `from` does not post-dominate. An exit and a cut root
//! have one more way out, their edge to the virtual exit, so no edge out of
//! them post-dominates anything.

use super::PostDominatorTree;
use crate::graph::view::{DenseId, GraphView};

impl<N: DenseId> PostDominatorTree<N> {
    /// Return whether every path from `block` to the virtual exit takes the
    /// edge `from -> to`.
    ///
    /// The answer is `false` when the graph has no such edge or has it more
    /// than once, when `from` or `block` is outside the tree, and when `from`
    /// is an exit or a [cut root](Self::cut_roots): its edge to the virtual
    /// exit is one more way out. A self-loop post-dominates nothing.
    ///
    /// On a block without [`reaches_exit`](Self::reaches_exit) the answer is
    /// relative to the cut, like every other post-dominance fact there.
    ///
    /// `graph` must be the graph the tree was computed from. The call reads
    /// the successors of `from` once and looks `from` up among the cut roots,
    /// so it costs O(outdegree(`from`) + log(cut roots)) and allocates
    /// nothing.
    ///
    /// # Panics
    ///
    /// Panics when `from` or `block`, or a successor of `from`, has an index
    /// beyond the node bound the tree was computed with.
    ///
    /// # Examples
    ///
    /// ```
    /// use cfglib::{Cfg, EdgeKind, PostDominatorTree};
    ///
    /// let mut cfg = Cfg::<u32>::new();
    /// let entry = cfg.entry();
    /// let check = cfg.new_block();
    /// let fail = cfg.new_block();
    /// let exit = cfg.new_block();
    /// cfg.add_edge(entry, check, EdgeKind::Unconditional);
    /// cfg.add_edge(check, fail, EdgeKind::ConditionalTrue);
    /// cfg.add_edge(check, exit, EdgeKind::ConditionalFalse);
    /// cfg.add_edge(fail, exit, EdgeKind::Unconditional);
    ///
    /// let post = PostDominatorTree::compute(&cfg);
    /// // Every way from the entry to the exit leaves through `entry -> check`.
    /// assert!(post.edge_post_dominates(&cfg, entry, check, entry));
    /// // The check can skip `fail`, so neither of its arms is on every path.
    /// assert!(!post.edge_post_dominates(&cfg, check, fail, check));
    /// assert!(post.edge_post_dominates(&cfg, fail, exit, fail));
    /// ```
    #[must_use]
    pub fn edge_post_dominates<G>(&self, graph: &G, from: N, to: N, block: N) -> bool
    where
        G: GraphView<NodeId = N>,
    {
        if !self.contains(from) || !self.contains(block) || !self.post_dominates(from, block) {
            return false;
        }
        if self
            .cut_roots
            .binary_search_by_key(&from.index(), |root| root.index())
            .is_ok()
        {
            return false;
        }
        let mut leaves = graph
            .successors(from)
            .filter(|&next| !self.post_dominates(from, next));
        leaves.next() == Some(to) && leaves.next().is_none()
    }
}
