//! Structural detection — natural loops, regions, and reducibility.
//!
//! Identifies loop structures and classifies a graph as reducible or
//! irreducible, building on the dominator tree and back-edge detection.
//!
//! The core detectors are generic over [`DirectedGraphView`] /
//! [`RootedGraphView`] and use dominance only. [`Cfg`] consumers whose
//! builders tag explicit [`EdgeKind::Back`] edges (structured `loop` /
//! `continue` markers, including on irreducible machine CFGs) can use the
//! `_tagged` variants, which union the tags with dominance-based detection.

extern crate alloc;
use alloc::collections::BTreeSet;
use alloc::vec::Vec;

use super::dominator::DominatorTree;
use super::view::{DenseNodeId, DirectedGraphView, RootedGraphView};
use crate::block::BlockId;
use crate::cfg::Cfg;
use crate::edge::EdgeKind;

/// A natural loop over node identity `N`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NaturalLoop<N = BlockId> {
    /// The loop header (entry point, dominates all other nodes).
    pub header: N,
    /// All nodes in the loop body (including the header).
    pub body: BTreeSet<N>,
    /// Back-edge tail nodes (nodes that jump back to the header).
    pub latches: BTreeSet<N>,
    /// Nesting depth (0 = outermost).
    pub depth: usize,
}

/// A back-edge: tail → header where header dominates tail.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct BackEdge<N = BlockId> {
    /// The node at the tail of the back-edge (the source of the jump).
    pub tail: N,
    /// The loop header node (the target of the back-edge).
    pub header: N,
}

/// Find all back-edges in a graph view (edges whose target dominates
/// their source).
///
/// The result is deduplicated: parallel edges between the same pair
/// appear once. For [`Cfg`]s whose builders also tag explicit
/// [`EdgeKind::Back`] edges, [`find_back_edges_tagged`] additionally
/// honours the tags.
///
/// # Examples
///
/// ```
/// use cfglib::{Cfg, EdgeKind, DominatorTree, find_back_edges};
///
/// let mut cfg = Cfg::<u32>::new();
/// let b0 = cfg.entry();
/// let b1 = cfg.new_block();
/// cfg.add_edge(b0, b1, EdgeKind::Fallthrough);
/// cfg.add_edge(b1, b0, EdgeKind::Back);
///
/// let dom = DominatorTree::compute(&cfg);
/// let backs = find_back_edges(&cfg, &dom);
/// assert_eq!(backs.len(), 1);
/// assert_eq!(backs[0].header, b0);
/// assert_eq!(backs[0].tail, b1);
/// ```
#[must_use]
pub fn find_back_edges<G: DirectedGraphView>(
    graph: &G,
    dom: &DominatorTree<G::NodeId>,
) -> Vec<BackEdge<G::NodeId>> {
    let mut backs = Vec::new();
    let depths = dom.analysis_depths();
    for node in graph.node_ids() {
        for successor in graph.successors(node) {
            if dom.dominates_with_analysis_depths(successor, node, &depths) {
                backs.push(BackEdge {
                    tail: node,
                    header: successor,
                });
            }
        }
    }
    backs.sort();
    backs.dedup();
    backs
}

/// Find back-edges in a [`Cfg`], honouring explicit [`EdgeKind::Back`] tags.
///
/// The union of [`find_back_edges`]'s dominance-based detection and the
/// builder's tags. The tags matter on irreducible machine CFGs where a
/// frontend knows an edge is a loop back-edge even though dominance cannot
/// prove it.
#[must_use]
pub fn find_back_edges_tagged<I>(cfg: &Cfg<I>, dom: &DominatorTree) -> Vec<BackEdge> {
    let mut backs = find_back_edges(cfg, dom);
    for edge in cfg.edges() {
        if edge.kind() == EdgeKind::Back {
            backs.push(BackEdge {
                tail: edge.source(),
                header: edge.target(),
            });
        }
    }
    backs.sort();
    backs.dedup();
    backs
}

/// Compute the merged natural loop body for every back-edge to one header.
///
/// The body is the set of nodes that can reach any latch without going
/// through `header`, plus `header` itself. A single multi-source reverse walk
/// computes the same union without revisiting a shared body for every latch.
fn loop_body_for<G: DirectedGraphView>(
    graph: &G,
    header: G::NodeId,
    latches: &[G::NodeId],
) -> BTreeSet<G::NodeId> {
    let mut body = BTreeSet::new();
    body.insert(header);
    let mut stack = Vec::new();
    for &latch in latches {
        if body.insert(latch) {
            stack.push(latch);
        }
    }
    while let Some(n) = stack.pop() {
        for p in graph.predecessors(n) {
            if !body.contains(&p) {
                body.insert(p);
                stack.push(p);
            }
        }
    }
    body
}

/// Build merged, depth-annotated loops from a set of back-edges.
fn loops_from_backs<G: DirectedGraphView>(
    graph: &G,
    backs: &[BackEdge<G::NodeId>],
) -> Vec<NaturalLoop<G::NodeId>> {
    if backs.is_empty() {
        return Vec::new();
    }

    // Group back-edges by header.
    let mut header_map: alloc::collections::BTreeMap<G::NodeId, Vec<G::NodeId>> =
        alloc::collections::BTreeMap::new();
    for be in backs {
        header_map.entry(be.header).or_default().push(be.tail);
    }

    let mut loops: Vec<NaturalLoop<G::NodeId>> = Vec::new();
    for (header, latches) in &header_map {
        let body = loop_body_for(graph, *header, latches);
        loops.push(NaturalLoop {
            header: *header,
            body,
            latches: latches.iter().copied().collect(),
            depth: 0, // filled in below
        });
    }

    // Compute nesting depth in O(L × max_body) instead of O(L²):
    // Build a map from node → number of loops containing it, then
    // each loop's depth = (count of its header) − 1 (itself).
    {
        let node_count = graph.node_count();
        let mut containing: Vec<u32> = alloc::vec![0; node_count];
        for lp in &loops {
            for &b in &lp.body {
                containing[b.index()] += 1;
            }
        }
        for lp in &mut loops {
            // Every loop's body includes its own header, so subtract 1.
            lp.depth = (containing[lp.header.index()] - 1) as usize;
        }
    }

    // Sort by depth (outermost first), then by header.
    loops.sort_by(|a, b| a.depth.cmp(&b.depth).then(a.header.cmp(&b.header)));
    loops
}

/// Detect all natural loops in a graph view.
///
/// Loops sharing the same header are merged into a single
/// [`NaturalLoop`] with multiple latches.
///
/// **Dominance-based only.** Unlike the pre-substrate function of the same
/// name, this does NOT honour explicit [`EdgeKind::Back`] tags — a tagged
/// back-edge whose target does not dominate its source (an irreducible
/// machine CFG) is invisible here. [`Cfg`] callers wanting tag recall use
/// [`detect_loops_tagged`], as [`cfg_metrics`](crate::cfg_metrics) and
/// [`cfg_block_nesting_depths`](crate::cfg_block_nesting_depths) do.
///
/// # Examples
///
/// ```
/// use cfglib::{Cfg, EdgeKind, DominatorTree, detect_loops};
///
/// let mut cfg = Cfg::<u32>::new();
/// let b0 = cfg.entry();
/// let b1 = cfg.new_block();
/// let b2 = cfg.new_block();
/// cfg.add_edge(b0, b1, EdgeKind::Fallthrough);
/// cfg.add_edge(b1, b0, EdgeKind::Back);
/// cfg.add_edge(b0, b2, EdgeKind::ConditionalTrue);
///
/// let dom = DominatorTree::compute(&cfg);
/// let loops = detect_loops(&cfg, &dom);
/// assert_eq!(loops.len(), 1);
/// assert_eq!(loops[0].header, b0);
/// assert!(loops[0].body.contains(&b1));
/// ```
#[must_use]
pub fn detect_loops<G: DirectedGraphView>(
    graph: &G,
    dom: &DominatorTree<G::NodeId>,
) -> Vec<NaturalLoop<G::NodeId>> {
    loops_from_backs(graph, &find_back_edges(graph, dom))
}

/// Detect natural loops in a [`Cfg`], honouring explicit
/// [`EdgeKind::Back`] tags (see [`find_back_edges_tagged`]).
#[must_use]
pub fn detect_loops_tagged<I>(cfg: &Cfg<I>, dom: &DominatorTree) -> Vec<NaturalLoop> {
    loops_from_backs(cfg, &find_back_edges_tagged(cfg, dom))
}

/// Whether the graph is reducible.
///
/// A graph is **reducible** if and only if every cycle contains a node
/// that dominates all other nodes in that cycle. Equivalently, every
/// retreating edge in a DFS is a back-edge (target dominates source).
///
/// The DFS runs from the view's root; any edge to a gray (in-progress)
/// node whose target does not dominate the source witnesses an
/// irreducible cycle.
#[must_use]
pub fn is_reducible<G: RootedGraphView>(graph: &G, dom: &DominatorTree<G::NodeId>) -> bool {
    find_irreducible_entry(graph, dom).is_none()
}

/// Return the first irreducible entry witnessed by the reducibility DFS.
///
/// Keeping target discovery and the Boolean query on one traversal prevents
/// transformations from choosing a different witness than [`is_reducible`]
/// used to reject the graph.
pub(crate) fn find_irreducible_entry<G: RootedGraphView>(
    graph: &G,
    dom: &DominatorTree<G::NodeId>,
) -> Option<G::NodeId> {
    const WHITE: u8 = 0;
    const GRAY: u8 = 1;
    const BLACK: u8 = 2;

    let n = graph.node_count();
    if n == 0 {
        return None;
    }

    let mut color = alloc::vec![WHITE; n];
    let mut stack: Vec<(G::NodeId, bool)> = alloc::vec![(graph.root(), false)];

    while let Some((node, processed)) = stack.pop() {
        if processed {
            color[node.index()] = BLACK;
            continue;
        }
        if color[node.index()] != WHITE {
            continue;
        }
        color[node.index()] = GRAY;
        stack.push((node, true));

        for succ in graph.successors(node) {
            match color[succ.index()] {
                WHITE => stack.push((succ, false)),
                GRAY
                    // Retreating edge — must be a natural back-edge.
                    if !dom.dominates(succ, node) => {
                        return Some(succ);
                    }
                _ => {} // Cross/forward edge — fine.
            }
        }
    }
    None
}

// ── Loop canonicalization ───────────────────────────────────────────

/// Information about a canonicalized loop.
#[derive(Debug, Clone)]
pub struct CanonicalLoop {
    /// The original natural loop.
    pub natural_loop: NaturalLoop,
    /// The preheader block (newly inserted).
    pub preheader: BlockId,
    /// Exit blocks — blocks outside the loop that are targets of edges
    /// from inside the loop.
    pub exits: BTreeSet<BlockId>,
}

/// Insert a dedicated **preheader** block for a natural loop.
///
/// A preheader is a single-successor block that becomes the sole
/// non-backedge predecessor of the loop header. This simplifies
/// many loop transformations (LICM, unrolling, etc.).
///
/// Returns the `BlockId` of the new preheader, or `None` if a
/// preheader was not needed (single non-backedge predecessor).
pub fn insert_preheader<I: Clone>(cfg: &mut Cfg<I>, lp: &NaturalLoop) -> Option<BlockId> {
    // Collect non-backedge predecessors of the header.
    let outside_preds: Vec<crate::edge::EdgeId> = cfg
        .predecessor_edges(lp.header)
        .iter()
        .copied()
        .filter(|&eid| {
            let src = cfg.edge(eid).source();
            !lp.body.contains(&src)
        })
        .collect();

    if outside_preds.len() <= 1 {
        return None; // already canonical
    }

    let preheader = cfg.new_block();

    // Redirect all outside predecessor edges to target the preheader.
    for eid in &outside_preds {
        let edge = cfg.edge(*eid);
        let src = edge.source();
        let kind = edge.kind();
        cfg.remove_edge(*eid);
        cfg.add_edge(src, preheader, kind);
    }

    // Add fallthrough from preheader to the header.
    cfg.add_edge(preheader, lp.header, crate::edge::EdgeKind::Fallthrough);

    Some(preheader)
}

/// Identify exit nodes of a natural loop.
///
/// An exit node is any node **outside** the loop body that has a
/// predecessor inside the loop body.
#[must_use]
pub fn loop_exit_blocks<G: DirectedGraphView>(
    graph: &G,
    lp: &NaturalLoop<G::NodeId>,
) -> BTreeSet<G::NodeId> {
    let mut exits = BTreeSet::new();
    for &b in &lp.body {
        for s in graph.successors(b) {
            if !lp.body.contains(&s) {
                exits.insert(s);
            }
        }
    }
    exits
}

/// Canonicalize all loops: insert preheaders and identify exits.
///
/// Uses [`detect_loops_tagged`], so explicit [`EdgeKind::Back`] tags are
/// honoured.
pub fn canonicalize_loops<I: Clone>(cfg: &mut Cfg<I>, dom: &DominatorTree) -> Vec<CanonicalLoop> {
    let loops = detect_loops_tagged(cfg, dom);
    let mut result = Vec::new();

    for lp in loops {
        let exits = loop_exit_blocks(cfg, &lp);
        let preheader = insert_preheader(cfg, &lp).unwrap_or_else(|| {
            // No new preheader needed; use the single outside pred.
            let outside: Vec<BlockId> = cfg
                .predecessors(lp.header)
                .filter(|p| !lp.body.contains(p))
                .collect();
            outside.into_iter().next().unwrap_or(lp.header)
        });

        result.push(CanonicalLoop {
            natural_loop: lp,
            preheader,
            exits,
        });
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builder::CfgBuilder;
    use crate::flow::FlowEffect;
    use crate::graph::dominator::DominatorTree;
    use crate::test_util::{MockInst, ff};
    use alloc::vec;

    #[test]
    fn no_loops_in_linear_cfg() {
        let cfg = CfgBuilder::build(vec![ff("a"), ff("b"), ff("c")]).unwrap();
        let dom = DominatorTree::compute(&cfg);
        let loops = detect_loops(&cfg, &dom);
        assert_eq!(loops.len(), 0);
    }

    #[test]
    fn simple_loop_detected() {
        let cfg = CfgBuilder::build(vec![
            MockInst(FlowEffect::LoopOpen, "loop"),
            ff("body"),
            MockInst(FlowEffect::LoopClose, "endloop"),
            MockInst(FlowEffect::Return, "ret"),
        ])
        .unwrap();
        let dom = DominatorTree::compute(&cfg);
        let loops = detect_loops(&cfg, &dom);
        assert_eq!(loops.len(), 1);
        assert!(!loops[0].body.is_empty(), "loop body is non-empty");
    }

    #[test]
    fn loops_with_one_header_merge_every_latch_body() {
        let mut cfg = Cfg::<MockInst>::new();
        let header = cfg.entry();
        let left_body = cfg.new_block();
        let left_latch = cfg.new_block();
        let right_body = cfg.new_block();
        let right_latch = cfg.new_block();
        let exit = cfg.new_block();

        cfg.add_edge(header, left_body, EdgeKind::ConditionalTrue);
        cfg.add_edge(header, right_body, EdgeKind::ConditionalFalse);
        cfg.add_edge(header, exit, EdgeKind::SwitchCase);
        cfg.add_edge(left_body, left_latch, EdgeKind::Fallthrough);
        cfg.add_edge(left_latch, header, EdgeKind::Back);
        cfg.add_edge(right_body, right_latch, EdgeKind::Fallthrough);
        cfg.add_edge(right_latch, header, EdgeKind::Back);

        let dom = DominatorTree::compute(&cfg);
        let loops = detect_loops(&cfg, &dom);

        assert_eq!(loops.len(), 1);
        assert_eq!(loops[0].header, header);
        assert_eq!(loops[0].latches, BTreeSet::from([left_latch, right_latch]));
        assert_eq!(
            loops[0].body,
            BTreeSet::from([header, left_body, left_latch, right_body, right_latch])
        );
        assert_eq!(loops[0].depth, 0);
    }

    #[test]
    fn nested_loops_have_depth() {
        let cfg = CfgBuilder::build(vec![
            MockInst(FlowEffect::LoopOpen, "outer"),
            MockInst(FlowEffect::LoopOpen, "inner"),
            ff("body"),
            MockInst(FlowEffect::ConditionalBreak, "breakc_inner"),
            MockInst(FlowEffect::LoopClose, "end_inner"),
            MockInst(FlowEffect::ConditionalBreak, "breakc_outer"),
            MockInst(FlowEffect::LoopClose, "end_outer"),
            MockInst(FlowEffect::Return, "ret"),
        ])
        .unwrap();
        let dom = DominatorTree::compute(&cfg);
        let loops = detect_loops(&cfg, &dom);
        assert_eq!(loops.len(), 2);
        // Outermost first (depth 0), inner second (depth 1).
        assert_eq!(loops[0].depth, 0);
        assert_eq!(loops[1].depth, 1);
    }

    #[test]
    fn loop_with_break_still_detected() {
        let cfg = CfgBuilder::build(vec![
            MockInst(FlowEffect::LoopOpen, "loop"),
            MockInst(FlowEffect::ConditionalBreak, "breakc"),
            ff("body"),
            MockInst(FlowEffect::LoopClose, "endloop"),
        ])
        .unwrap();
        let dom = DominatorTree::compute(&cfg);
        let loops = detect_loops(&cfg, &dom);
        assert_eq!(loops.len(), 1);
    }

    #[test]
    fn tagged_back_edge_survives_without_dominance() {
        // Two entries into a tagged cycle: dominance alone cannot prove
        // b1 → b2 is a loop, but the builder tag says it is.
        let mut cfg = Cfg::<MockInst>::new();
        let b1 = cfg.new_block();
        let b2 = cfg.new_block();
        cfg.add_edge(cfg.entry(), b1, EdgeKind::ConditionalTrue);
        cfg.add_edge(cfg.entry(), b2, EdgeKind::ConditionalFalse);
        cfg.add_edge(b1, b2, EdgeKind::Fallthrough);
        cfg.add_edge(b2, b1, EdgeKind::Back);

        let dom = DominatorTree::compute(&cfg);
        assert_eq!(find_back_edges(&cfg, &dom).len(), 0);
        let tagged = find_back_edges_tagged(&cfg, &dom);
        assert_eq!(tagged.len(), 1);
        assert_eq!(tagged[0].header, b1);
        let loops = detect_loops_tagged(&cfg, &dom);
        assert_eq!(loops.len(), 1);
        assert!(loops[0].body.contains(&b2));
    }

    #[test]
    fn linear_cfg_is_reducible() {
        let cfg = CfgBuilder::build(vec![ff("a"), ff("b")]).unwrap();
        let dom = DominatorTree::compute(&cfg);
        assert!(is_reducible(&cfg, &dom));
    }

    #[test]
    fn loop_cfg_is_reducible() {
        let cfg = CfgBuilder::build(vec![
            MockInst(FlowEffect::LoopOpen, "loop"),
            ff("body"),
            MockInst(FlowEffect::LoopClose, "endloop"),
        ])
        .unwrap();
        let dom = DominatorTree::compute(&cfg);
        assert!(is_reducible(&cfg, &dom));
    }

    #[test]
    fn if_else_no_loops() {
        let cfg = CfgBuilder::build(vec![
            MockInst(FlowEffect::ConditionalOpen, "if"),
            ff("then"),
            MockInst(FlowEffect::ConditionalAlternate, "else"),
            ff("else_body"),
            MockInst(FlowEffect::ConditionalClose, "endif"),
        ])
        .unwrap();
        let dom = DominatorTree::compute(&cfg);
        let loops = detect_loops(&cfg, &dom);
        assert_eq!(loops.len(), 0);
        assert!(is_reducible(&cfg, &dom));
    }

    #[test]
    fn loop_exit_blocks_found() {
        let cfg = CfgBuilder::build(vec![
            MockInst(FlowEffect::LoopOpen, "loop"),
            MockInst(FlowEffect::ConditionalBreak, "breakc"),
            ff("body"),
            MockInst(FlowEffect::LoopClose, "endloop"),
            MockInst(FlowEffect::Return, "ret"),
        ])
        .unwrap();
        let dom = DominatorTree::compute(&cfg);
        let loops = detect_loops(&cfg, &dom);
        assert_ne!(loops.len(), 0);
        let exits = loop_exit_blocks(&cfg, &loops[0]);
        assert!(
            !exits.is_empty(),
            "loop should have at least one exit block"
        );
    }

    #[test]
    fn canonicalize_loops_adds_exits() {
        let mut cfg = CfgBuilder::build(vec![
            MockInst(FlowEffect::LoopOpen, "loop"),
            MockInst(FlowEffect::ConditionalBreak, "breakc"),
            ff("body"),
            MockInst(FlowEffect::LoopClose, "endloop"),
            MockInst(FlowEffect::Return, "ret"),
        ])
        .unwrap();
        let dom = DominatorTree::compute(&cfg);
        let canonical = canonicalize_loops(&mut cfg, &dom);
        assert!(!canonical.is_empty());
        assert!(!canonical[0].exits.is_empty());
    }
}
