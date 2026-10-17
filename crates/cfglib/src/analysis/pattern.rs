//! Pattern matching — idiom recognition within a graph.
//!
//! Identifies common structural patterns (if-then-else diamonds, chains,
//! self-loops, trampolines) for downstream consumers. [`detect_patterns`]
//! is topology-only and serves any graph view; [`detect_cfg_patterns`]
//! additionally orients diamond arms by edge kind and detects empty
//! trampoline blocks, which need [`CfgView`] instruction knowledge.

extern crate alloc;
use alloc::vec::Vec;
use smallvec::SmallVec;

use crate::block::BlockId;
use crate::dataflow::cfg_view::CfgView;
use crate::edge::EdgeKind;
use crate::graph::view::{DenseId, GraphView};

/// A recognized structural pattern, over node identity `N`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CfgPattern<N = BlockId> {
    /// Diamond: entry branches to two arms that reconverge at merge.
    ///
    /// The entry, both arms, and the merge are four distinct nodes: two
    /// edges to one arm are not two arms, and a path that returns to the
    /// entry or ends in an arm is a loop, not a merge.
    ///
    /// [`detect_patterns`] emits arms in successor order;
    /// [`detect_cfg_patterns`] orients `arms[0]` to the
    /// [`EdgeKind::ConditionalTrue`] side.
    Diamond {
        /// Node that branches.
        entry: N,
        /// The two branch arms.
        arms: [N; 2],
        /// Merge point.
        merge: N,
    },
    /// A single-entry, single-exit linear chain of nodes.
    Chain {
        /// Ordered list of nodes in the chain.
        blocks: Vec<N>,
    },
    /// An empty block (no instructions) acting as a trampoline.
    ///
    /// Only emitted by [`detect_cfg_patterns`] — emptiness is an
    /// instruction-level property.
    EmptyTrampoline {
        /// The empty block.
        block: N,
    },
    /// A self-loop (node has an edge to itself).
    SelfLoop {
        /// The looping node.
        block: N,
    },
}

/// The only item `items` yields, or `None` when it yields none or several.
fn sole<T>(mut items: impl Iterator<Item = T>) -> Option<T> {
    let first = items.next()?;
    items.next().is_none().then_some(first)
}

/// Scan a graph view for topology-level patterns.
///
/// # Examples
///
/// ```
/// use cfglib::{Cfg, CfgPattern, EdgeKind, detect_patterns};
///
/// let mut cfg = Cfg::<u32>::new();
/// let entry = cfg.entry();
/// let [left, right, merge] = std::array::from_fn(|_| cfg.new_block());
/// cfg.add_edge(entry, left, EdgeKind::ConditionalTrue);
/// cfg.add_edge(entry, right, EdgeKind::ConditionalFalse);
/// cfg.add_edge(left, merge, EdgeKind::Fallthrough);
/// cfg.add_edge(right, merge, EdgeKind::Fallthrough);
///
/// assert_eq!(
///     detect_patterns(&cfg),
///     [CfgPattern::Diamond { entry, arms: [left, right], merge }]
/// );
/// ```
#[must_use]
pub fn detect_patterns<G: GraphView>(graph: &G) -> Vec<CfgPattern<G::NodeId>> {
    let mut patterns = Vec::new();

    for bid in graph.node_ids() {
        let succs: SmallVec<[G::NodeId; 4]> = graph.successors(bid).collect();

        if succs.contains(&bid) {
            patterns.push(CfgPattern::SelfLoop { block: bid });
        }

        // Diamond detection: two distinct successors that share a single
        // successor, which is neither the branching node nor an arm.
        if let [a, b] = succs[..]
            && a != b
            && let Some(merge) = sole(graph.successors(a))
            && sole(graph.successors(b)) == Some(merge)
            && ![bid, a, b].contains(&merge)
        {
            patterns.push(CfgPattern::Diamond {
                entry: bid,
                arms: [a, b],
                merge,
            });
        }
    }

    // Chain detection: sequences of single-pred, single-succ nodes.
    let mut visited = alloc::vec![false; graph.node_bound()];
    let mut chain = Vec::new();
    // `walked[node] == search` once candidate number `search`'s backward walk
    // has passed `node`; a fresh number resets the whole table.
    let mut walked = alloc::vec![0_usize; graph.node_bound()];
    let mut search = 0;
    for bid in graph.node_ids() {
        if visited[bid.index()]
            || sole(graph.predecessors(bid)).is_none()
            || sole(graph.successors(bid)).is_none()
        {
            continue;
        }

        // Walk backward to find chain start. The walked set guards against
        // isolated cycles (every node one-pred/one-succ), which would
        // otherwise loop forever; a direct self-loop is the len-1 case.
        search += 1;
        walked[bid.index()] = search;
        let mut start = bid;
        while let Some(pred) = sole(graph.predecessors(start))
            && core::mem::replace(&mut walked[pred.index()], search) != search
            && sole(graph.successors(pred)).is_some()
        {
            start = pred;
        }
        chain.clear();
        chain.push(start);
        visited[start.index()] = true;
        let mut cur = start;
        while let Some(next) = sole(graph.successors(cur))
            && next != cur
            && !visited[next.index()]
            && sole(graph.predecessors(next)).is_some()
        {
            chain.push(next);
            visited[next.index()] = true;
            cur = next;
        }
        if chain.len() >= 2 {
            patterns.push(CfgPattern::Chain {
                blocks: chain.clone(),
            });
        }
    }

    patterns
}

/// Scan a [`CfgView`] for patterns, adding instruction-aware refinements.
///
/// Runs [`detect_patterns`], orients each diamond's `arms[0]` to the
/// [`EdgeKind::ConditionalTrue`] side, and additionally detects
/// [`CfgPattern::EmptyTrampoline`] blocks.
#[must_use]
pub fn detect_cfg_patterns<G: CfgView>(cfg: &G) -> Vec<CfgPattern> {
    let mut patterns = detect_patterns(cfg);

    // Orient diamond arms: arms[0] = ConditionalTrue side when tagged.
    for pattern in &mut patterns {
        if let CfgPattern::Diamond { entry, arms, .. } = pattern {
            let true_target = cfg.outgoing(*entry).find_map(|eid| {
                (cfg.edge_kind(eid) == EdgeKind::ConditionalTrue).then(|| cfg.edge(eid).target())
            });
            if let Some(target) = true_target
                && arms[1] == target
            {
                arms.swap(0, 1);
            }
        }
    }

    // Empty trampolines: no instructions, exactly one successor, not entry.
    for bid in cfg.node_ids() {
        if cfg.instructions(bid).is_empty() && cfg.successors(bid).count() == 1 && bid != cfg.root()
        {
            patterns.push(CfgPattern::EmptyTrampoline { block: bid });
        }
    }

    patterns
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cfg::Cfg;
    use crate::edge::EdgeKind;
    use crate::test_util::ff;

    #[test]
    fn detects_diamond_with_orientation() {
        let mut cfg = Cfg::new();
        let a = cfg.new_block();
        let b = cfg.new_block();
        let merge = cfg.new_block();
        cfg.block_mut(cfg.entry()).instructions_mut().push(ff("br"));
        cfg.block_mut(a).instructions_mut().push(ff("a"));
        cfg.block_mut(b).instructions_mut().push(ff("b"));
        cfg.block_mut(merge).instructions_mut().push(ff("m"));
        // Wire false first so orientation genuinely reorders.
        cfg.add_edge(cfg.entry(), b, EdgeKind::ConditionalFalse);
        cfg.add_edge(cfg.entry(), a, EdgeKind::ConditionalTrue);
        cfg.add_edge(a, merge, EdgeKind::Fallthrough);
        cfg.add_edge(b, merge, EdgeKind::Fallthrough);
        let pats = detect_cfg_patterns(&cfg);
        let diamond = pats
            .iter()
            .find(|p| matches!(p, CfgPattern::Diamond { .. }))
            .expect("diamond detected");
        if let CfgPattern::Diamond {
            arms: [first, second],
            ..
        } = diamond
        {
            assert_eq!(*first, a, "arms[0] is the ConditionalTrue side");
            assert_eq!(*second, b);
        }
    }

    #[test]
    fn isolated_cycle_terminates() {
        // Every node one-pred/one-succ around a cycle: the backward chain
        // walk must terminate (it previously looped forever here).
        use crate::graph::store::Graph;
        let mut graph: Graph<(), ()> = Graph::new();
        let a = graph.add_node(());
        let b = graph.add_node(());
        graph.add_edge(a, b, ());
        graph.add_edge(b, a, ());

        let patterns = detect_patterns(&graph);
        assert!(
            patterns
                .iter()
                .all(|p| !matches!(p, CfgPattern::SelfLoop { .. })),
            "a 2-cycle is not a self-loop: {patterns:?}"
        );
    }

    #[test]
    fn detects_self_loop() {
        let mut cfg = Cfg::new();
        cfg.block_mut(cfg.entry())
            .instructions_mut()
            .push(ff("loop"));
        cfg.add_edge(cfg.entry(), cfg.entry(), EdgeKind::Back);
        let pats = detect_patterns(&cfg);
        assert!(
            pats.iter()
                .any(|p| matches!(p, CfgPattern::SelfLoop { .. }))
        );
    }

    #[test]
    fn detect_patterns_reports_every_pattern_family_in_order() {
        use crate::graph::store::Graph;
        let mut graph: Graph<(), ()> = Graph::new();
        let n: Vec<_> = (0..18).map(|_| graph.add_node(())).collect();
        let edges = [
            // A run 0..=3 into a branch, whose arms rejoin at 6.
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 4),
            (3, 5),
            (4, 6),
            (5, 6),
            // A self-loop inside a tail that keeps going.
            (6, 6),
            (6, 7),
            (7, 8),
            (8, 9),
            // A bare two-cycle and a bare self-loop.
            (10, 11),
            (11, 10),
            (12, 12),
            // Parallel edges to the same arm.
            (13, 14),
            (13, 14),
            (14, 15),
            // An isolated tail whose head has two predecessors.
            (16, 17),
            (9, 17),
        ];
        for (source, target) in edges {
            graph.add_edge(n[source], n[target], ());
        }

        // Diamonds and self-loops come first in node order, then chains by
        // first member; a cycle's chain starts where the backward walk stops.
        assert_eq!(
            detect_patterns(&graph),
            alloc::vec![
                CfgPattern::Diamond {
                    entry: n[3],
                    arms: [n[4], n[5]],
                    merge: n[6]
                },
                CfgPattern::SelfLoop { block: n[6] },
                CfgPattern::SelfLoop { block: n[12] },
                CfgPattern::Chain {
                    blocks: alloc::vec![n[0], n[1], n[2], n[3]]
                },
                CfgPattern::Chain {
                    blocks: alloc::vec![n[7], n[8], n[9]]
                },
                CfgPattern::Chain {
                    blocks: alloc::vec![n[11], n[10]]
                },
            ]
        );
    }

    #[test]
    fn a_diamond_has_four_distinct_nodes() {
        use crate::graph::store::Graph;

        let cases: [(&str, &[(usize, usize)]); 3] = [
            // A single arm reached by two edges, not two arms.
            ("parallel edges to one arm", &[(0, 1), (0, 1), (1, 2)]),
            // Both arms return to the branching node: a loop, not a merge.
            (
                "arms return to the entry",
                &[(0, 1), (0, 2), (1, 0), (2, 0)],
            ),
            // One arm is itself the merge: it never reconverges with the other.
            ("an arm is the merge", &[(0, 1), (0, 2), (1, 1), (2, 1)]),
        ];
        for (name, edges) in cases {
            let mut graph: Graph<(), ()> = Graph::new();
            let nodes: Vec<_> = (0..3).map(|_| graph.add_node(())).collect();
            for &(source, target) in edges {
                graph.add_edge(nodes[source], nodes[target], ());
            }
            let patterns = detect_patterns(&graph);
            assert!(
                patterns
                    .iter()
                    .all(|pattern| !matches!(pattern, CfgPattern::Diamond { .. })),
                "{name}: no diamond expected in {patterns:?}"
            );
        }
    }
}
