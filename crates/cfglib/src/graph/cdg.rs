//! Control-dependence analysis on the generic directed-graph substrate.
//!
//! Node `B` is control-dependent on node `A` when `A` selects whether `B`
//! executes. The resulting graph has one node per source node and an edge
//! `A -> B` for that relation; node payloads retain the original identity.
//!
//! The relation is the one of Pingali and Bilardi: `B` depends on an arm
//! `A -> T` when `B` post-dominates `T` and does not strictly post-dominate
//! `A`. A block that branches back to itself is therefore control dependent
//! on itself.
//!
//! In a region that no exit follows, post-dominance needs a convention, and
//! the relation follows the one its tree was built under (see
//! [`ExitPolicy`](crate::ExitPolicy)): relative to the cut by default, or
//! nontermination-insensitive with every exit-free sink as one class.
//! [`nontermination_sensitive_controllers`] gives the third reading, which
//! needs no post-dominator tree: a branch controls a block when one arm must
//! reach it and the other may avoid it, by leaving or by looping forever.

extern crate alloc;
use alloc::vec::Vec;

use crate::graph::dominator::PostDominatorTree;
use crate::graph::epoch::EpochMarks;
use crate::graph::store::Graph;
use crate::graph::view::{DenseId, GraphView};
pub use nontermination::{NonterminationScratch, nontermination_sensitive_controllers};

mod nontermination;

/// Compute control dependences from a graph view and its post-dominator
/// tree.
///
/// The graph has the edge `A -> B` exactly when `B` is control dependent on
/// an arm `A -> T` to a successor `T` of `A`, which is when `B`
/// post-dominates `T` and does not strictly post-dominate `A`. These are the
/// blocks of [`PostDominatorTree::control_dependents`], and an arm that
/// leads back to `A` itself gives the edge `A -> A`. Parallel records
/// contribute one edge, and a block the entry cannot reach has none.
///
/// The tree decides what an exit-free region gives. Under the default
/// [`ExitPolicy::CutLoops`](crate::ExitPolicy::CutLoops) the blocks that close
/// such a loop control it. Under
/// [`ExitPolicy::SinkComponents`](crate::ExitPolicy::SinkComponents) no arm
/// inside a sink class makes an edge, a class that depends contributes an
/// edge to each of its blocks, and the result is the nontermination-insensitive
/// control dependence of Hecker et al.: `A -> B` when `A` has one successor
/// that `B` post-dominates and one it does not, under sink post-dominance.
///
/// Returned nodes are allocated in source dense-index order, so a source node
/// and its graph node have the same dense index even when the view overrides
/// `node_ids()` iteration order. The node payload remains the authoritative
/// link back to the source graph.
///
/// # Examples
///
/// ```
/// use cfglib::{
///     Cfg, EdgeKind, NodeId, PostDominatorTree, control_dependence_graph,
/// };
///
/// let mut cfg = Cfg::<u32>::new();
/// let branch = cfg.entry();
/// let left = cfg.new_block();
/// let right = cfg.new_block();
/// let merge = cfg.new_block();
/// cfg.add_edge(branch, left, EdgeKind::ConditionalTrue);
/// cfg.add_edge(branch, right, EdgeKind::ConditionalFalse);
/// cfg.add_edge(left, merge, EdgeKind::Fallthrough);
/// cfg.add_edge(right, merge, EdgeKind::Fallthrough);
///
/// let post_dominators = PostDominatorTree::compute(&cfg);
/// let dependences = control_dependence_graph(&cfg, &post_dominators);
/// let branch_node = NodeId::from_index(branch.index());
/// let controlled: Vec<_> = dependences
///     .successors(branch_node)
///     .map(|node| dependences[node])
///     .collect();
/// assert!(controlled.contains(&left));
/// assert!(controlled.contains(&right));
/// ```
///
/// A block that branches back to itself decides whether it runs again, so it
/// is control dependent on itself and the graph has the edge from it to
/// itself:
///
/// ```
/// use cfglib::{
///     Cfg, EdgeKind, NodeId, PostDominatorTree, control_dependence_graph,
/// };
///
/// let mut cfg = Cfg::<u32>::new();
/// let head = cfg.entry();
/// let exit = cfg.new_block();
/// cfg.add_edge(head, head, EdgeKind::ConditionalTrue);
/// cfg.add_edge(head, exit, EdgeKind::ConditionalFalse);
///
/// let post_dominators = PostDominatorTree::compute(&cfg);
/// let dependences = control_dependence_graph(&cfg, &post_dominators);
/// let head_node = NodeId::from_index(head.index());
/// let controlled: Vec<_> = dependences
///     .successors(head_node)
///     .map(|node| dependences[node])
///     .collect();
/// assert_eq!(controlled, [head]);
/// ```
#[must_use]
pub fn control_dependence_graph<G: GraphView>(
    source: &G,
    post_dominators: &PostDominatorTree<G::NodeId>,
) -> Graph<G::NodeId, ()> {
    // A tree of the cut has no sink classes, so each arm is the plain walk;
    // choosing it once here keeps that walk inline in the loop.
    if post_dominators.has_sink_classes() {
        collect_dependences(source, post_dominators, |from, to| {
            post_dominators.control_dependents(from, to)
        })
    } else {
        collect_dependences(source, post_dominators, |from, to| {
            post_dominators.tree_dependents(from, to)
        })
    }
}

/// The control dependence graph of `source`, with `dependents_of` giving
/// the blocks that depend on each arm.
fn collect_dependences<G, I>(
    source: &G,
    post_dominators: &PostDominatorTree<G::NodeId>,
    dependents_of: impl Fn(G::NodeId, G::NodeId) -> I,
) -> Graph<G::NodeId, ()>
where
    G: GraphView,
    I: Iterator<Item = G::NodeId>,
{
    let mut graph = Graph::with_capacity(source.node_bound(), source.node_bound());
    let nodes: Vec<_> = (0..source.node_bound())
        .map(G::NodeId::from_index)
        .map(|node| graph.add_node(node))
        .collect();
    let mut seen_dependents = EpochMarks::new(source.node_bound());
    let mut dependents = Vec::new();
    let mut controllers: Vec<_> = source.node_ids().collect();
    // Stable edge IDs expose global `(controller, dependent)` ordering. Dense
    // index order need not agree with a consumer ID's `Ord`, so sort here.
    controllers.sort_unstable();

    for controller in controllers {
        // Control dependence is defined through post-dominance, and a block
        // the entry cannot reach has no post-dominance facts. Emitting its
        // edges would fabricate "A selects whether B executes" claims for
        // code that never runs. Every reachable block has facts, because the
        // tree gives each exit-free region a cut.
        if !post_dominators.contains(controller) {
            continue;
        }
        seen_dependents.reset();
        dependents.clear();
        for target in source.successors(controller) {
            for dependent in dependents_of(controller, target) {
                if !seen_dependents.is_marked(dependent.index()) {
                    seen_dependents.mark(dependent.index());
                    dependents.push(dependent);
                }
            }
        }
        dependents.sort_unstable();
        for &dependent in &dependents {
            graph.add_edge(nodes[controller.index()], nodes[dependent.index()], ());
        }
    }
    graph
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::block::BlockId;
    use crate::cfg::Cfg;
    use crate::edge::EdgeKind;
    use crate::graph::store::NodeId;
    use crate::graph::view::Rooted;
    use crate::test_util::ff;
    use alloc::collections::BTreeSet;
    use alloc::vec;
    use core::cmp::Ordering;

    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
    struct ReverseOrdNode(usize);

    impl PartialOrd for ReverseOrdNode {
        fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
            Some(self.cmp(other))
        }
    }

    impl Ord for ReverseOrdNode {
        fn cmp(&self, other: &Self) -> Ordering {
            other.0.cmp(&self.0)
        }
    }

    impl DenseId for ReverseOrdNode {
        fn from_index(index: usize) -> Self {
            Self(index)
        }

        fn index(self) -> usize {
            self.0
        }
    }

    struct ReverseOrdGraph;

    impl GraphView for ReverseOrdGraph {
        type NodeId = ReverseOrdNode;

        fn node_bound(&self) -> usize {
            6
        }

        fn node_ids(&self) -> impl Iterator<Item = Self::NodeId> + '_ {
            [3, 0, 5, 1, 4, 2].into_iter().map(ReverseOrdNode)
        }

        fn successors(&self, node: Self::NodeId) -> impl Iterator<Item = Self::NodeId> + '_ {
            const EMPTY: &[usize] = &[];
            const ZERO: &[usize] = &[1, 1, 2];
            const ONE: &[usize] = &[3, 4, 4];
            const EXIT: &[usize] = &[5];
            let successors = match node.0 {
                0 => ZERO,
                1 => ONE,
                2..=4 => EXIT,
                _ => EMPTY,
            };
            successors.iter().copied().map(ReverseOrdNode)
        }

        fn predecessors(&self, node: Self::NodeId) -> impl Iterator<Item = Self::NodeId> + '_ {
            const EMPTY: &[usize] = &[];
            const FROM_ZERO: &[usize] = &[0];
            const TWICE_FROM_ZERO: &[usize] = &[0, 0];
            const FROM_ONE: &[usize] = &[1];
            const TWICE_FROM_ONE: &[usize] = &[1, 1];
            const TO_EXIT: &[usize] = &[2, 3, 4];
            let predecessors = match node.0 {
                1 => TWICE_FROM_ZERO,
                2 => FROM_ZERO,
                3 => FROM_ONE,
                4 => TWICE_FROM_ONE,
                5 => TO_EXIT,
                _ => EMPTY,
            };
            predecessors.iter().copied().map(ReverseOrdNode)
        }
    }

    fn node(block: BlockId) -> NodeId {
        NodeId::from_index(block.index())
    }

    #[test]
    fn diamond_control_dependences_are_edges() {
        let mut cfg = Cfg::new();
        let left = cfg.new_block();
        let right = cfg.new_block();
        let merge = cfg.new_block();
        cfg.block_mut(cfg.entry()).push(ff("entry"));
        cfg.block_mut(left).push(ff("left"));
        cfg.block_mut(right).push(ff("right"));
        cfg.block_mut(merge).push(ff("merge"));
        cfg.add_edge(cfg.entry(), left, EdgeKind::ConditionalTrue);
        cfg.add_edge(cfg.entry(), right, EdgeKind::ConditionalFalse);
        cfg.add_edge(left, merge, EdgeKind::Fallthrough);
        cfg.add_edge(right, merge, EdgeKind::Fallthrough);

        let post_dominators = PostDominatorTree::compute(&cfg);
        let graph = control_dependence_graph(&cfg, &post_dominators);
        let controlled: BTreeSet<_> = graph
            .successors(node(cfg.entry()))
            .map(|dependent| *graph.node(dependent))
            .collect();

        assert_eq!(controlled, BTreeSet::from([left, right]));
        assert_eq!(graph.predecessors(node(merge)).count(), 0);
    }

    #[test]
    fn blocks_the_entry_cannot_reach_emit_no_dependences() {
        // `island` never runs, so its branch must not become a fabricated
        // control dependence of the live blocks it names.
        let mut cfg: Cfg<()> = Cfg::new();
        let live = cfg.new_block();
        let tail = cfg.new_block();
        let island = cfg.new_block();
        cfg.add_edge(cfg.entry(), live, EdgeKind::Fallthrough);
        cfg.add_edge(live, tail, EdgeKind::Fallthrough);
        cfg.add_edge(island, live, EdgeKind::ConditionalTrue);
        cfg.add_edge(island, tail, EdgeKind::ConditionalFalse);

        let post = PostDominatorTree::compute(&cfg);
        assert!(!post.contains(island));
        let cdg = control_dependence_graph(&cfg, &post);
        assert_eq!(cdg.edge_count(), 0);
    }

    fn dependences_of(cfg: &Cfg<()>) -> Vec<(BlockId, BlockId)> {
        let post = PostDominatorTree::compute(cfg);
        let graph = control_dependence_graph(cfg, &post);
        graph
            .edges()
            .map(|edge| (*graph.node(edge.source()), *graph.node(edge.target())))
            .collect()
    }

    #[test]
    fn a_branch_into_an_exit_free_loop_still_controls_both_arms() {
        // Without a cut the spinning arm has no post-dominance facts, so the
        // branch would control nothing and a dead-code pass would delete it.
        // The spin governs only itself, through the arm back into itself.
        let mut cfg: Cfg<()> = Cfg::new();
        let spin = cfg.new_block();
        let out = cfg.new_block();
        cfg.add_edge(cfg.entry(), spin, EdgeKind::ConditionalTrue);
        cfg.add_edge(cfg.entry(), out, EdgeKind::ConditionalFalse);
        cfg.add_edge(spin, spin, EdgeKind::Unconditional);

        let post = PostDominatorTree::compute(&cfg);
        let graph = control_dependence_graph(&cfg, &post);
        let controlled: BTreeSet<_> = graph
            .successors(node(cfg.entry()))
            .map(|dependent| *graph.node(dependent))
            .collect();
        let spin_controls: BTreeSet<_> = graph
            .successors(node(spin))
            .map(|dependent| *graph.node(dependent))
            .collect();

        assert_eq!(controlled, BTreeSet::from([spin, out]));
        assert_eq!(spin_controls, BTreeSet::from([spin]));
        assert_eq!(graph.edge_count(), 3);
    }

    #[test]
    fn a_block_that_loops_back_to_itself_is_control_dependent_on_itself() {
        let mut cfg: Cfg<()> = Cfg::new();
        let head = cfg.entry();
        let exit = cfg.new_block();
        cfg.add_edge(head, head, EdgeKind::ConditionalTrue);
        cfg.add_edge(head, exit, EdgeKind::ConditionalFalse);

        assert_eq!(dependences_of(&cfg), [(head, head)]);
    }

    #[test]
    fn parallel_self_loop_records_give_one_edge() {
        let mut cfg: Cfg<()> = Cfg::new();
        let head = cfg.entry();
        let exit = cfg.new_block();
        cfg.add_edge(head, head, EdgeKind::ConditionalTrue);
        cfg.add_edge(head, head, EdgeKind::ConditionalFalse);
        cfg.add_edge(head, exit, EdgeKind::Fallthrough);

        assert_eq!(dependences_of(&cfg), [(head, head)]);
    }

    #[test]
    fn a_self_loop_another_arm_already_governs_still_gives_one_edge() {
        // `head -> body -> head` reaches `head` through the arm into `body`
        // as well as through the arm into itself.
        let mut cfg: Cfg<()> = Cfg::new();
        let head = cfg.entry();
        let body = cfg.new_block();
        let exit = cfg.new_block();
        cfg.add_edge(head, head, EdgeKind::ConditionalTrue);
        cfg.add_edge(head, body, EdgeKind::ConditionalFalse);
        cfg.add_edge(head, exit, EdgeKind::Fallthrough);
        cfg.add_edge(body, head, EdgeKind::Back);

        assert_eq!(dependences_of(&cfg), [(head, head), (head, body)]);
    }

    #[test]
    fn a_self_loop_on_a_branch_arm_adds_only_its_own_edge() {
        let mut cfg: Cfg<()> = Cfg::new();
        let branch = cfg.entry();
        let spinner = cfg.new_block();
        let plain = cfg.new_block();
        let merge = cfg.new_block();
        cfg.add_edge(branch, spinner, EdgeKind::ConditionalTrue);
        cfg.add_edge(branch, plain, EdgeKind::ConditionalFalse);
        cfg.add_edge(spinner, spinner, EdgeKind::ConditionalTrue);
        cfg.add_edge(spinner, merge, EdgeKind::ConditionalFalse);
        cfg.add_edge(plain, merge, EdgeKind::Fallthrough);

        assert_eq!(
            dependences_of(&cfg),
            [(branch, spinner), (branch, plain), (spinner, spinner)]
        );
    }

    #[test]
    fn custom_id_order_retains_global_edge_identity_order() {
        let post = PostDominatorTree::compute(&Rooted::new(&ReverseOrdGraph, ReverseOrdNode(0)));
        let graph = control_dependence_graph(&ReverseOrdGraph, &post);
        for index in 0..ReverseOrdGraph.node_bound() {
            assert_eq!(
                *graph.node(NodeId::from_index(index)),
                ReverseOrdNode(index)
            );
        }
        let relations: Vec<_> = graph
            .edges()
            .map(|edge| (*graph.node(edge.source()), *graph.node(edge.target())))
            .collect();

        assert_eq!(
            relations,
            vec![
                (ReverseOrdNode(1), ReverseOrdNode(4)),
                (ReverseOrdNode(1), ReverseOrdNode(3)),
                (ReverseOrdNode(0), ReverseOrdNode(2)),
                (ReverseOrdNode(0), ReverseOrdNode(1)),
            ]
        );
    }

    #[test]
    fn linear_cfg_has_no_control_dependence_edges() {
        let mut cfg = Cfg::new();
        let next = cfg.new_block();
        cfg.block_mut(cfg.entry()).push(ff("entry"));
        cfg.block_mut(next).push(ff("next"));
        cfg.add_edge(cfg.entry(), next, EdgeKind::Fallthrough);

        let post_dominators = PostDominatorTree::compute(&cfg);
        let graph = control_dependence_graph(&cfg, &post_dominators);

        assert_eq!(graph.node_bound(), cfg.block_bound());
        assert_eq!(graph.edge_count(), 0);
    }
}
