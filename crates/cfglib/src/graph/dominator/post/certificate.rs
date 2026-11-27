//! Certification of a post-dominator tree by a low-high order.
//!
//! The tree is the dominator tree of the reverse graph with a virtual exit, so
//! the check is the one of `certificate` on that graph, read from the virtual
//! exit. The exits and the cut are recomputed from the graph first and
//! compared with the ones the tree keeps, because they define which graph
//! that is.

use super::{CutBuffers, PostDominatorTree};
use crate::graph::dominator::certificate::{Reading, check};
use crate::graph::dominator::{CertificateError, DominatorScratch, ExitPolicy, LowHighOrder};
use crate::graph::view::{DenseId, Induced, Reversed, RootedView, VirtualRoot};
use crate::identity::NO_INDEX;

impl<N: DenseId> PostDominatorTree<N> {
    /// Compute the post-dominator tree under [`ExitPolicy::CutLoops`]
    /// together with a low-high order that certifies it.
    ///
    /// The tree is the one [`compute`](Self::compute) builds. The order is a
    /// low-high order of the tree with the virtual exit at its root, as
    /// [`LowHighOrder`] describes, built from the by-products of the same
    /// construction in linear time.
    ///
    /// # Panics
    ///
    /// Panics under the conditions of [`compute`](Self::compute).
    ///
    /// # Examples
    ///
    /// ```
    /// use cfglib::{Cfg, DominatorScratch, EdgeKind, PostDominatorTree};
    ///
    /// let mut cfg = Cfg::<u32>::new();
    /// let entry = cfg.entry();
    /// let spin = cfg.new_block();
    /// let exit = cfg.new_block();
    /// cfg.add_edge(entry, spin, EdgeKind::ConditionalTrue);
    /// cfg.add_edge(entry, exit, EdgeKind::ConditionalFalse);
    /// cfg.add_edge(spin, spin, EdgeKind::Unconditional);
    ///
    /// let mut scratch = DominatorScratch::default();
    /// let (post, certificate) = PostDominatorTree::compute_certified_in(&mut scratch, &cfg);
    /// assert_eq!(post, PostDominatorTree::compute(&cfg));
    /// assert_eq!(post.verify_certificate(&cfg, &certificate), Ok(()));
    /// ```
    #[must_use]
    pub fn compute_certified_in<G>(
        scratch: &mut DominatorScratch,
        graph: &G,
    ) -> (Self, LowHighOrder<N>)
    where
        G: RootedView<NodeId = N>,
    {
        let mut certificate = LowHighOrder::empty();
        let tree = Self::construct(scratch, graph, ExitPolicy::CutLoops, Some(&mut certificate));
        (tree, certificate)
    }

    /// Check that this is the post-dominator tree of `graph`, given a
    /// low-high order of it.
    ///
    /// The exits, the cut and the blocks the entry reaches are recomputed
    /// from `graph` and must be the ones the tree keeps; every exit and cut
    /// root must hang from the virtual exit. Then the tree, with the virtual
    /// exit at its root, must pass the check of
    /// [`DominatorTree::verify_certificate`](crate::DominatorTree::verify_certificate)
    /// on the reverse graph that joins those blocks to the virtual exit. The
    /// cost is O(n + m), and the recomputed cut allocates its own tables.
    ///
    /// # Errors
    ///
    /// Returns the first failure found, as a [`CertificateError`]. An edge
    /// named by [`ParentProperty`](CertificateError::ParentProperty) is an
    /// edge of the reverse graph.
    pub fn verify_certificate<G>(
        &self,
        graph: &G,
        certificate: &LowHighOrder<N>,
    ) -> Result<(), CertificateError<N>>
    where
        G: RootedView<NodeId = N>,
    {
        let bound = graph.node_bound();
        if self.tree.idom.len() != bound || certificate.position.len() != bound {
            return Err(CertificateError::NodeBound {
                graph: bound,
                tree: self.tree.idom.len(),
                certificate: certificate.position.len(),
            });
        }
        let root = graph.root();
        if root.index() >= bound {
            return Err(CertificateError::Root(root));
        }
        let mut cut = CutBuffers::default();
        let structure = cut.analyze(graph, root);
        let hanging = structure
            .outlets
            .iter()
            .all(|&block| self.contains(block) && self.tree.idom(block).is_none());
        if !hanging
            || structure.reaches_exit != self.reaches_exit
            || structure.exits != self.exits
            || structure.cut_roots != self.cut_roots
        {
            return Err(CertificateError::ExitStructure);
        }

        let live = Induced::new(graph, &cut.reachable);
        let reverse = Reversed::new(&live);
        let augmented = VirtualRoot::masked(&reverse, &structure.outlets, &cut.exit_edge);
        check(
            &augmented,
            &Reading::below_exit(&self.tree, certificate, bound),
        )
    }
}

/// The certificate of the whole tree, `with_exit`, read without the virtual
/// exit at its root.
pub(super) fn without_virtual_exit<N: DenseId>(
    with_exit: &LowHighOrder<usize>,
    exit: usize,
) -> LowHighOrder<N> {
    LowHighOrder {
        order: with_exit.order[1..]
            .iter()
            .map(|&block| N::from_index(block))
            .collect(),
        position: with_exit.position[..exit]
            .iter()
            .map(|&position| {
                if position == NO_INDEX {
                    NO_INDEX
                } else {
                    position - 1
                }
            })
            .collect(),
    }
}
