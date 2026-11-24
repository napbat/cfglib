//! Certification of a post-dominator tree by a low-high order.
//!
//! The tree is the dominator tree of the reverse graph with a virtual exit, so
//! the check is the one of `certificate` on that graph, the
//! [`AugmentedReverse`] view, rooted at the virtual exit. The exits and the cut
//! are recomputed from the graph first and compared with the ones the tree
//! keeps, because they define which graph that is.

use super::augmented::AugmentedReverse;
use super::cut::CutBuffers;
use super::{PostDominatorScratch, PostDominatorTree};
use crate::graph::dominator::certificate::{Failure, Order, Tree, check};
use crate::graph::dominator::{ABSENT, CertificateError, DominatorTree, LowHighOrder, slot};
use crate::graph::view::{DenseId, RootedView};

impl<N: DenseId> PostDominatorTree<N> {
    /// Compute the post-dominator tree together with a low-high order that
    /// certifies it.
    ///
    /// The tree is the one [`compute_in`](Self::compute_in) builds. The order
    /// is a low-high order of the tree with the virtual exit at its root, as
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
    /// use cfglib::{Cfg, EdgeKind, PostDominatorScratch, PostDominatorTree};
    ///
    /// let mut cfg = Cfg::<u32>::new();
    /// let entry = cfg.entry();
    /// let spin = cfg.new_block();
    /// let exit = cfg.new_block();
    /// cfg.add_edge(entry, spin, EdgeKind::ConditionalTrue);
    /// cfg.add_edge(entry, exit, EdgeKind::ConditionalFalse);
    /// cfg.add_edge(spin, spin, EdgeKind::Unconditional);
    ///
    /// let mut scratch = PostDominatorScratch::new();
    /// let (post, certificate) = PostDominatorTree::compute_certified_in(&mut scratch, &cfg);
    /// assert_eq!(post, PostDominatorTree::compute(&cfg));
    /// assert_eq!(post.verify_certificate(&cfg, &certificate), Ok(()));
    /// ```
    #[must_use]
    pub fn compute_certified_in<G>(
        scratch: &mut PostDominatorScratch,
        graph: &G,
    ) -> (Self, LowHighOrder<N>)
    where
        G: RootedView<NodeId = N>,
    {
        let mut certificate = LowHighOrder::empty();
        let tree = Self::construct(scratch, graph, Some(&mut certificate));
        (tree, certificate)
    }

    /// Check that this is the post-dominator tree of `graph`, given a
    /// low-high order of it.
    ///
    /// The exits, the cut and the blocks the entry reaches are recomputed
    /// from `graph` and must be the ones the tree keeps; every exit and cut
    /// root must hang from the virtual exit. Then the tree, with the virtual
    /// exit at its root, must pass the check of
    /// [`DominatorTree::verify_certificate`] on the reverse graph that joins
    /// those blocks to the virtual exit. The cost is O(n + m), and the
    /// recomputed cut allocates its own tables.
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
            .exits
            .iter()
            .chain(&structure.cut_roots)
            .all(|&block| self.contains(block) && self.tree.idom(block).is_none());
        if !hanging
            || structure.reaches_exit != self.reaches_exit
            || structure.exits != self.exits
            || structure.cut_roots != self.cut_roots
        {
            return Err(CertificateError::ExitStructure);
        }

        let augmented = AugmentedReverse::new(graph, &cut, &structure);
        let tree = WithExit {
            tree: &self.tree,
            exit: bound,
        };
        let order = ExitFirst {
            order: certificate,
            exit: bound,
        };
        check(&augmented, &tree, &order).map_err(|failure| lift(failure, bound))
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
                if position == ABSENT {
                    ABSENT
                } else {
                    position - 1
                }
            })
            .collect(),
    }
}

/// A failure named in blocks. Every exit and cut root hangs from the virtual
/// exit by the time the check runs, so no failure names it; one that did
/// would be a wrong exit structure.
fn lift<N: DenseId>(failure: Failure, exit: usize) -> CertificateError<N> {
    let names_exit = match failure {
        Failure::Preorder(node) | Failure::LowHigh(node) => node == exit,
        Failure::ParentProperty(tail, head) => tail == exit || head == exit,
    };
    if names_exit {
        CertificateError::ExitStructure
    } else {
        failure.lift(N::from_index)
    }
}

/// The forest of a post-dominator tree with the virtual exit, the index
/// `exit`, joining its roots.
struct WithExit<'a, N> {
    tree: &'a DominatorTree<N>,
    exit: usize,
}

impl<N: DenseId> Tree for WithExit<'_, N> {
    fn root(&self) -> usize {
        self.exit
    }

    fn size(&self) -> usize {
        self.tree.order.len() + 1
    }

    fn contains(&self, node: usize) -> bool {
        node == self.exit || self.tree.pre.get(node).is_some_and(|&pre| pre != ABSENT)
    }

    fn parent(&self, node: usize) -> Option<usize> {
        if node == self.exit {
            return None;
        }
        let parent = self.tree.idom[node];
        Some(if parent == ABSENT {
            self.exit
        } else {
            slot(parent)
        })
    }

    fn dominates(&self, ancestor: usize, node: usize) -> bool {
        ancestor == self.exit
            || (node != self.exit
                && self
                    .tree
                    .dominates(N::from_index(ancestor), N::from_index(node)))
    }

    fn nodes(&self) -> impl Iterator<Item = usize> + '_ {
        core::iter::once(self.exit).chain(self.tree.order.iter().map(|node| node.index()))
    }
}

/// A certificate of a post-dominator tree with the virtual exit put back in
/// front.
struct ExitFirst<'a, N> {
    order: &'a LowHighOrder<N>,
    exit: usize,
}

impl<N: DenseId> Order for ExitFirst<'_, N> {
    fn len(&self) -> usize {
        self.order.order.len() + 1
    }

    fn at(&self, position: usize) -> usize {
        match position {
            0 => self.exit,
            _ => self.order.order[position - 1].index(),
        }
    }

    fn position(&self, node: usize) -> Option<usize> {
        if node == self.exit {
            return Some(0);
        }
        Order::position(self.order, node).map(|position| position + 1)
    }
}
