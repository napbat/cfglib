//! Naive reference solvers: iterate every node in turn until nothing
//! changes. They read adjacency from the test's own `Shape` data, so they
//! share no scheduling, worklist, or graph-store code with the solvers under
//! test; only the lattice operations are shared, because those are the
//! problem being solved.

use cfglib::EdgeKind;

use crate::graphs::Shape;

/// What a node-granular round-robin solve produced, indexed by block.
pub struct NodeFacts<F> {
    /// The fact flowing into the node in the analysis direction.
    pub merged: Vec<F>,
    /// The fact flowing out of the node after its transfer.
    pub transferred: Vec<F>,
}

/// Iterate `domain` round-robin to a fixpoint. A node with no upstream in the
/// analysis direction takes `boundary`; otherwise it takes the meet of its
/// upstream nodes' transferred facts. Nodes outside `domain` stay at bottom.
pub fn round_robin<F: Clone + PartialEq>(
    shape: &Shape,
    domain: &[usize],
    forward: bool,
    bottom: &F,
    boundary: &F,
    meet: &dyn Fn(&F, &F) -> F,
    transfer: &dyn Fn(usize, &F) -> F,
) -> NodeFacts<F> {
    let mut merged = vec![bottom.clone(); shape.blocks];
    let mut transferred = vec![bottom.clone(); shape.blocks];
    loop {
        let mut changed = false;
        for &node in domain {
            let upstream = if forward {
                shape.predecessors(node)
            } else {
                shape.successors(node)
            };
            let merged_in = upstream
                .iter()
                .map(|from| transferred[*from].clone())
                .reduce(|left, right| meet(&left, &right))
                .unwrap_or_else(|| boundary.clone());
            let out = transfer(node, &merged_in);
            if merged_in != merged[node] || out != transferred[node] {
                changed = true;
                merged[node] = merged_in;
                transferred[node] = out;
            }
        }
        if !changed {
            return NodeFacts {
                merged,
                transferred,
            };
        }
    }
}

/// What an edge-sensitive round-robin solve produced.
pub struct EdgeSolution<F> {
    /// The physical pre-state of each node.
    pub input: Vec<F>,
    /// The physical post-state of each node.
    pub output: Vec<F>,
    /// The fact on each edge slot, `None` for a removed edge.
    pub edge: Vec<Option<F>>,
}

/// The edge-sensitive fixpoint, by iterating every live node in turn.
///
/// A node's flow fact is the meet, in edge order, of its boundary fact (when
/// it has one) and the facts on its upstream edges: incoming edges forward,
/// outgoing backward. Each edge on the downstream side is then recomputed
/// from the node's physical input and output.
pub fn edge_round_robin<F: Clone + PartialEq>(
    shape: &Shape,
    forward: bool,
    bottom: &F,
    boundary: &dyn Fn(usize) -> Option<F>,
    meet: &dyn Fn(&F, &F) -> F,
    transfer_node: &dyn Fn(usize, &F) -> F,
    transfer_edge: &dyn Fn(EdgeKind, &F, &F) -> F,
) -> EdgeSolution<F> {
    let live_edges = shape.live_edges();
    let mut input = vec![bottom.clone(); shape.blocks];
    let mut output = vec![bottom.clone(); shape.blocks];
    let mut edge: Vec<Option<F>> = vec![None; shape.edges.len()];
    for (id, _) in &live_edges {
        edge[*id] = Some(bottom.clone());
    }
    loop {
        let mut changed = false;
        for node in shape.live() {
            let upstream = live_edges.iter().filter(|(_, (source, target, _))| {
                if forward {
                    *target == node
                } else {
                    *source == node
                }
            });
            let mut flow = boundary(node);
            for (id, _) in upstream {
                let fact = edge[*id].clone().expect("a live edge has a fact");
                flow = Some(match flow {
                    Some(current) => meet(&current, &fact),
                    None => fact,
                });
            }
            let flow = flow.unwrap_or_else(|| bottom.clone());
            let result = transfer_node(node, &flow);
            let (node_in, node_out) = if forward {
                (flow, result)
            } else {
                (result, flow)
            };
            if node_in != input[node] || node_out != output[node] {
                changed = true;
                input[node] = node_in.clone();
                output[node] = node_out.clone();
            }
            for (id, (source, target, kind)) in &live_edges {
                let downstream = if forward {
                    *source == node
                } else {
                    *target == node
                };
                if downstream {
                    let fact = transfer_edge(*kind, &node_in, &node_out);
                    if edge[*id].as_ref() != Some(&fact) {
                        changed = true;
                        edge[*id] = Some(fact);
                    }
                }
            }
        }
        if !changed {
            return EdgeSolution {
                input,
                output,
                edge,
            };
        }
    }
}
