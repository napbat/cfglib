//! Structural detection — natural loops, regions, and reducibility.
//!
//! Identifies loop structures and classifies a graph as reducible or
//! irreducible, building on the dominator tree and back-edge detection.
//!
//! The core detectors are generic over [`GraphView`] / [`RootedView`] and use
//! dominance only. A consumer whose builder tags explicit
//! [`EdgeKind::Back`] edges (structured `loop` / `continue` markers, including
//! on irreducible machine CFGs) uses the `_tagged` variants, which union the
//! tags with dominance-based detection. Those are generic too: any
//! [`EdgeView`] whose edge data declares a [`KindedEdge`] kind participates,
//! so the tag-aware detectors are not a second copy written against [`Cfg`].

extern crate alloc;
use alloc::collections::BTreeSet;
use alloc::vec::Vec;
use core::ops::ControlFlow;

use super::dominator::DominatorTree;
use super::edge_view::EdgeView;
use super::loop_nest::nesting;
use super::search::{DfsEvent, depth_first_events};
use super::traverse::TraversalDirection;
use super::view::{GraphView, RootedView};
use crate::block::BlockId;
use crate::cfg::Cfg;
use crate::edge::{EdgeKind, KindedEdge};

/// A natural loop over node identity `N`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NaturalLoop<N = BlockId> {
    /// The loop header, the node the back-edges return to.
    ///
    /// Dominance-based detection gives a header that dominates the whole
    /// body. A tagged loop on an irreducible graph need not have one.
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
/// appear once. Only code the root reaches has back-edges: a block
/// dominates itself whether or not it runs, so unreachable code is skipped
/// rather than reported as loops. For [`Cfg`]s whose builders also tag
/// explicit [`EdgeKind::Back`] edges, [`find_back_edges_tagged`]
/// additionally honors the tags.
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
pub fn find_back_edges<G: GraphView>(
    graph: &G,
    dom: &DominatorTree<G::NodeId>,
) -> Vec<BackEdge<G::NodeId>> {
    let mut backs = Vec::new();
    for node in graph.node_ids().filter(|&node| dom.is_reachable(node)) {
        for successor in graph.successors(node) {
            if dom.dominates(successor, node) {
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

/// Find back-edges in an edge view, honoring explicit [`EdgeKind::Back`] tags.
///
/// The union of [`find_back_edges`]'s dominance-based detection and the
/// builder's tags. The tags matter on irreducible machine CFGs where a
/// frontend knows an edge is a loop back-edge even though dominance cannot
/// prove it. A tag on an edge out of unreachable code is ignored, as
/// dominance is.
#[must_use]
pub fn find_back_edges_tagged<G>(
    graph: &G,
    dom: &DominatorTree<G::NodeId>,
) -> Vec<BackEdge<G::NodeId>>
where
    G: EdgeView,
    G::EdgeData: KindedEdge,
{
    let mut backs = find_back_edges(graph, dom);
    for edge in graph.edge_ids() {
        let edge = graph.edge(edge);
        if edge.data().kind() == EdgeKind::Back && dom.is_reachable(edge.source()) {
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
/// The body is the set of reachable nodes that can reach any latch without
/// going through `header`, plus `header` itself. A single multi-source
/// reverse walk computes the same union without revisiting a shared body for
/// every latch.
///
/// When the header dominates every latch, that walk cannot leave the region
/// the header dominates. A tagged latch the header does not dominate (an
/// irreducible cycle) lets the walk run on to blocks that merely lead to the
/// cycle, such as the entry, so the body is cut down to the walked blocks the
/// header reaches: the blocks on a path from the header to a latch.
fn loop_body_for<G: GraphView>(
    graph: &G,
    dom: &DominatorTree<G::NodeId>,
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
        for p in graph.predecessors(n).filter(|&p| dom.is_reachable(p)) {
            if !body.contains(&p) {
                body.insert(p);
                stack.push(p);
            }
        }
    }

    if latches.iter().all(|&latch| dom.dominates(header, latch)) {
        return body;
    }
    let mut cycle = BTreeSet::from([header]);
    cycle.extend(latches.iter().copied());
    stack.push(header);
    while let Some(n) = stack.pop() {
        for s in graph.successors(n) {
            if body.contains(&s) && cycle.insert(s) {
                stack.push(s);
            }
        }
    }
    cycle
}

/// Build merged, depth-annotated loops from a set of back-edges.
fn loops_from_backs<G: GraphView>(
    graph: &G,
    dom: &DominatorTree<G::NodeId>,
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
        let body = loop_body_for(graph, dom, *header, latches);
        loops.push(NaturalLoop {
            header: *header,
            body,
            latches: latches.iter().copied().collect(),
            depth: 0, // filled in below
        });
    }

    // The nesting tree states the depth, so the flat list cannot disagree
    // with it.
    let depths = nesting(&loops).depths;
    for (lp, depth) in loops.iter_mut().zip(depths) {
        lp.depth = depth;
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
/// name, this does NOT honor explicit [`EdgeKind::Back`] tags — a tagged
/// back-edge whose target does not dominate its source (an irreducible
/// machine CFG) is invisible here. [`Cfg`] callers wanting tag recall use
/// [`detect_loops_tagged`], as [`CfgMetrics::compute`](crate::CfgMetrics::compute) and
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
pub fn detect_loops<G: GraphView>(
    graph: &G,
    dom: &DominatorTree<G::NodeId>,
) -> Vec<NaturalLoop<G::NodeId>> {
    loops_from_backs(graph, dom, &find_back_edges(graph, dom))
}

/// Detect natural loops in an edge view, honoring explicit
/// [`EdgeKind::Back`] tags (see [`find_back_edges_tagged`]).
#[must_use]
pub fn detect_loops_tagged<G>(
    graph: &G,
    dom: &DominatorTree<G::NodeId>,
) -> Vec<NaturalLoop<G::NodeId>>
where
    G: EdgeView,
    G::EdgeData: KindedEdge,
{
    loops_from_backs(graph, dom, &find_back_edges_tagged(graph, dom))
}

/// Whether the graph is reducible.
///
/// A graph is **reducible** if and only if every cycle contains a node
/// that dominates all other nodes in that cycle. Equivalently, every
/// retreating edge in a DFS is a back-edge (target dominates source).
///
/// The DFS runs from the view's root; any [`DfsEvent::BackEdge`] whose
/// target does not dominate its source witnesses an irreducible cycle.
#[must_use]
pub fn is_reducible<G: RootedView>(graph: &G, dom: &DominatorTree<G::NodeId>) -> bool {
    find_irreducible_entry(graph, dom).is_none()
}

/// Return the first irreducible entry witnessed by the reducibility DFS.
///
/// Keeping target discovery and the Boolean query on one traversal prevents
/// transformations from choosing a different witness than [`is_reducible`]
/// used to reject the graph.
pub(crate) fn find_irreducible_entry<G: RootedView>(
    graph: &G,
    dom: &DominatorTree<G::NodeId>,
) -> Option<G::NodeId> {
    if graph.node_bound() == 0 {
        return None;
    }
    depth_first_events(
        graph,
        graph.root(),
        TraversalDirection::Outgoing,
        |event| match event {
            DfsEvent::BackEdge(tail, header) if !dom.dominates(header, tail) => {
                ControlFlow::Break(header)
            }
            _ => ControlFlow::Continue(()),
        },
    )
}

/// Information about a canonicalized loop.
#[derive(Debug, Clone)]
pub struct CanonicalLoop {
    /// The natural loop. Its body includes the preheaders
    /// [`canonicalize_loops`] inserted for the loops nested in it.
    pub natural_loop: NaturalLoop,
    /// The loop's preheader: the one block outside the loop that enters the
    /// header, whose only successor is the header.
    ///
    /// `None` when the header has no predecessor outside the loop, which is
    /// the case for a loop headed by the entry block: the entry cannot gain
    /// a predecessor, so no preheader can be put before it.
    pub preheader: Option<BlockId>,
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
/// The header's edges from outside the loop are redirected to the preheader
/// in place: each keeps its identity, kind, weight and payload, and its
/// position among its source's outgoing edges, so a branch keeps its arm
/// order. The preheader reaches the header by a new fallthrough edge.
///
/// Returns the `BlockId` of the new preheader, or `None` if the header is
/// already canonical: it has no predecessor outside the loop, or exactly one
/// whose only successor is the header.
pub fn insert_preheader<I, E: Default>(cfg: &mut Cfg<I, E>, lp: &NaturalLoop) -> Option<BlockId> {
    // Collect non-backedge predecessors of the header.
    let outside_preds: Vec<crate::edge::EdgeId> = cfg
        .incoming(lp.header)
        .filter(|&eid| {
            let src = cfg.edge(eid).source();
            !lp.body.contains(&src)
        })
        .collect();

    match outside_preds.as_slice() {
        [] => return None,
        &[only] if cfg.outgoing(cfg.edge(only).source()).nth(1).is_none() => return None,
        _ => {}
    }

    let preheader = cfg.new_block();
    for eid in outside_preds {
        cfg.redirect_edge_target(eid, preheader);
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
pub fn loop_exit_blocks<G: GraphView>(
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
/// honored. Loops come outermost first, as detection orders them, each with
/// the preheader [`insert_preheader`] puts before it. A preheader that sits on
/// the paths through a loop nested in another is part of the outer loop's
/// body, and exits are read once every preheader is in, so an edge that leaves
/// one loop for the preheader of the next ends at the preheader.
///
/// `dom` is the dominator tree of `cfg` as it was passed in, and the inserted
/// blocks are not in it: compute it again before relying on it.
pub fn canonicalize_loops<I, E: Default>(
    cfg: &mut Cfg<I, E>,
    dom: &DominatorTree,
) -> Vec<CanonicalLoop> {
    let mut loops = detect_loops_tagged(cfg, dom);
    let mut preheaders = Vec::with_capacity(loops.len());

    for index in 0..loops.len() {
        let header = loops[index].header;
        let inserted = insert_preheader(cfg, &loops[index]);
        if let Some(preheader) = inserted {
            // The new block is reached from the sources of the redirected
            // edges and reaches the header, so it lies on a cycle of every
            // other loop that holds both the header and one of those sources.
            let sources: Vec<BlockId> = cfg.predecessors(preheader).collect();
            for (other_index, other) in loops.iter_mut().enumerate() {
                if other_index != index
                    && other.body.contains(&header)
                    && sources.iter().any(|source| other.body.contains(source))
                {
                    other.body.insert(preheader);
                }
            }
        }
        // Without an insertion the header has no outside predecessor, or one
        // that is already the preheader.
        preheaders.push(inserted.or_else(|| {
            cfg.predecessors(header)
                .find(|block| !loops[index].body.contains(block))
        }));
    }

    loops
        .into_iter()
        .zip(preheaders)
        .map(|(natural_loop, preheader)| CanonicalLoop {
            exits: loop_exit_blocks(cfg, &natural_loop),
            natural_loop,
            preheader,
        })
        .collect()
}

#[cfg(test)]
mod tests;
