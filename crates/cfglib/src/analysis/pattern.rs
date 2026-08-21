//! Pattern matching — idiom recognition within a graph.
//!
//! Identifies common structural patterns (if-then-else diamonds, chains,
//! self-loops, trampolines) for downstream consumers. [`detect_patterns`]
//! is topology-only and serves any graph view; [`detect_cfg_patterns`]
//! additionally orients diamond arms by edge kind and detects empty
//! trampoline blocks, which need [`Cfg`] instruction knowledge.

extern crate alloc;
use alloc::vec::Vec;

use crate::block::BlockId;
use crate::cfg::Cfg;
use crate::edge::EdgeKind;
use crate::graph::view::DirectedGraphView;

/// A recognised structural pattern, over node identity `N`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CfgPattern<N = BlockId> {
    /// Diamond: entry branches to two arms that reconverge at merge.
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

/// Scan a graph view for topology-level patterns.
///
/// # Examples
///
/// ```
/// use cfglib::{Cfg, EdgeKind, detect_patterns};
///
/// let mut cfg = Cfg::<u32>::new();
/// let b0 = cfg.entry();
/// let b1 = cfg.new_block();
/// cfg.add_edge(b0, b1, EdgeKind::ConditionalTrue);
/// cfg.add_edge(b0, b1, EdgeKind::ConditionalFalse);
///
/// let patterns = detect_patterns(&cfg);
/// // Detects structural patterns like diamond, self-loop, etc.
/// ```
#[must_use]
pub fn detect_patterns<G: DirectedGraphView>(graph: &G) -> Vec<CfgPattern<G::NodeId>> {
    let mut patterns = Vec::new();

    for bid in graph.node_ids() {
        let succs: Vec<G::NodeId> = graph.successors(bid).collect();

        // Self-loop detection.
        if succs.contains(&bid) {
            patterns.push(CfgPattern::SelfLoop { block: bid });
        }

        // Diamond detection: two successors that share a single successor.
        if succs.len() == 2 {
            let (a, b) = (succs[0], succs[1]);
            let a_succs: Vec<G::NodeId> = graph.successors(a).collect();
            let b_succs: Vec<G::NodeId> = graph.successors(b).collect();
            if a_succs.len() == 1 && b_succs.len() == 1 && a_succs[0] == b_succs[0] {
                patterns.push(CfgPattern::Diamond {
                    entry: bid,
                    arms: [a, b],
                    merge: a_succs[0],
                });
            }
        }
    }

    // Chain detection: sequences of single-pred, single-succ nodes.
    let mut visited = alloc::collections::BTreeSet::new();
    for bid in graph.node_ids() {
        if visited.contains(&bid) {
            continue;
        }
        let preds: Vec<G::NodeId> = graph.predecessors(bid).collect();
        if preds.len() != 1 {
            continue;
        }
        let succs: Vec<G::NodeId> = graph.successors(bid).collect();
        if succs.len() != 1 {
            continue;
        }

        // Walk backward to find chain start. The walked set guards against
        // isolated cycles (every node one-pred/one-succ), which would
        // otherwise loop forever; a direct self-loop is the len-1 case.
        let mut walked = alloc::collections::BTreeSet::new();
        walked.insert(bid);
        let mut start = bid;
        loop {
            let ps: Vec<G::NodeId> = graph.predecessors(start).collect();
            if ps.len() != 1 || !walked.insert(ps[0]) {
                break;
            }
            let ss: Vec<G::NodeId> = graph.successors(ps[0]).collect();
            if ss.len() != 1 {
                break;
            }
            start = ps[0];
        }
        // Walk forward to collect the chain.
        let mut chain = alloc::vec![start];
        visited.insert(start);
        let mut cur = start;
        loop {
            let ss: Vec<G::NodeId> = graph.successors(cur).collect();
            if ss.len() != 1 {
                break;
            }
            let next = ss[0];
            if next == cur || visited.contains(&next) {
                break;
            }
            let ps: Vec<G::NodeId> = graph.predecessors(next).collect();
            if ps.len() != 1 {
                break;
            }
            chain.push(next);
            visited.insert(next);
            cur = next;
        }
        if chain.len() >= 2 {
            patterns.push(CfgPattern::Chain { blocks: chain });
        }
    }

    patterns
}

/// Scan a [`Cfg`] for patterns, adding instruction-aware refinements.
///
/// Runs [`detect_patterns`], orients each diamond's `arms[0]` to the
/// [`EdgeKind::ConditionalTrue`] side, and additionally detects
/// [`CfgPattern::EmptyTrampoline`] blocks.
#[must_use]
pub fn detect_cfg_patterns<I>(cfg: &Cfg<I>) -> Vec<CfgPattern> {
    let mut patterns = detect_patterns(cfg);

    // Orient diamond arms: arms[0] = ConditionalTrue side when tagged.
    for pattern in &mut patterns {
        if let CfgPattern::Diamond { entry, arms, .. } = pattern {
            let true_target = cfg.successor_edges(*entry).iter().find_map(|&eid| {
                (cfg.edge(eid).kind() == EdgeKind::ConditionalTrue).then(|| cfg.edge(eid).target())
            });
            if true_target == Some(arms[1]) {
                arms.swap(0, 1);
            }
        }
    }

    // Empty trampolines: no instructions, exactly one successor, not entry.
    for block in cfg.blocks() {
        let bid = block.id();
        if block.instructions().is_empty() && cfg.successors(bid).len() == 1 && bid != cfg.entry() {
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
        cfg.block_mut(cfg.entry())
            .instructions_vec_mut()
            .push(ff("br"));
        cfg.block_mut(a).instructions_vec_mut().push(ff("a"));
        cfg.block_mut(b).instructions_vec_mut().push(ff("b"));
        cfg.block_mut(merge).instructions_vec_mut().push(ff("m"));
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
        use crate::graph::directed::DirectedGraph;
        let mut graph: DirectedGraph<(), ()> = DirectedGraph::new();
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
            .instructions_vec_mut()
            .push(ff("loop"));
        cfg.add_edge(cfg.entry(), cfg.entry(), EdgeKind::Back);
        let pats = detect_patterns(&cfg);
        assert!(
            pats.iter()
                .any(|p| matches!(p, CfgPattern::SelfLoop { .. }))
        );
    }
}
