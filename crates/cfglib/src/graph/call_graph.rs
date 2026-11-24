//! Whole-program call-graph construction on [`Graph`].
//!
//! A call graph does not need its own storage abstraction. Functions are node
//! payloads, call details are edge payloads, and all traversal, SCC, dominance,
//! and topological algorithms operate on the returned generic graph directly.
//! Call targets come from the instructions themselves via
//! [`CallInfo`] — callee identities are consumer-typed (symbol ids, addresses,
//! names), never a library-owned string.

extern crate alloc;
use alloc::collections::{BTreeMap, BinaryHeap};
use alloc::vec;
use alloc::vec::Vec;
use core::cmp::Reverse;

use crate::cfg::Cfg;
use crate::flow::CallInfo;
use crate::graph::scc::tarjan_scc;
use crate::graph::store::{Graph, NodeId};

/// A function [`propagate_summaries`] has not yet placed in its component's
/// callee-first order.
const UNSEEN: usize = usize::MAX;
/// A function on the walk that places its component's members.
const OPEN: usize = usize::MAX - 1;

/// The graph [`call_graph`] builds: functions as nodes, calls as edges.
type FunctionGraph<C> = Graph<FunctionNode<C>, CallMetadata>;

/// Payload of a function node in a call graph.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FunctionNode<C> {
    /// Consumer-defined function identity (symbol id, address, name).
    pub id: C,
}

/// Payload attached to a call edge.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CallMetadata {
    /// Whether the call transfers control without returning to its caller.
    pub is_tail_call: bool,
}

/// The function nodes of a call graph by identity, for repeated lookups.
///
/// [`find_function`] scans the graph on every call. The index answers the
/// same question in O(log n) after one O(n log n) pass, and
/// [`call_graph_indexed`] returns the index that building the graph makes
/// anyway. The index is a snapshot: it does not see nodes added to or removed
/// from the graph after it was computed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FunctionIndex<C> {
    by_key: BTreeMap<C, NodeId>,
}

impl<C: Ord + Clone> FunctionIndex<C> {
    /// Index the function nodes of `graph` by identity, in O(n log n).
    ///
    /// When several nodes carry one identity, the node with the lowest index
    /// wins, as it does for [`find_function`]. A graph that [`call_graph`]
    /// builds has one node per key, at the key's first position.
    ///
    /// # Examples
    ///
    /// ```
    /// use cfglib::{CallMetadata, FunctionIndex, FunctionNode, Graph, find_function};
    ///
    /// let mut graph = Graph::<FunctionNode<&str>, CallMetadata>::new();
    /// let main = graph.add_node(FunctionNode { id: "main" });
    /// let helper = graph.add_node(FunctionNode { id: "helper" });
    /// graph.add_edge(main, helper, CallMetadata { is_tail_call: false });
    ///
    /// let index = FunctionIndex::compute(&graph);
    /// assert_eq!(index.get(&"helper"), Some(helper));
    /// assert_eq!(index.get(&"missing"), None);
    /// assert_eq!(index.get(&"main"), find_function(&graph, &"main"));
    /// ```
    #[must_use]
    pub fn compute(graph: &Graph<FunctionNode<C>, CallMetadata>) -> Self {
        let mut by_key = BTreeMap::new();
        for node in graph.node_ids() {
            by_key.entry(graph.node(node).id.clone()).or_insert(node);
        }
        Self { by_key }
    }
}

impl<C: Ord> FunctionIndex<C> {
    /// Return the node of the function `id`, or `None` when no function has
    /// that identity, in O(log n).
    #[must_use]
    pub fn get(&self, id: &C) -> Option<NodeId> {
        self.by_key.get(id).copied()
    }
}

/// Build a call graph by scanning the instructions of a set of keyed CFGs.
///
/// Nodes are emitted in input order. Every instruction whose
/// [`CallInfo::callee`] resolves to one of the input keys creates an
/// inter-procedural edge; calls to unknown targets remain represented in the
/// source CFG but do not create an edge.
///
/// A key is one function. When it is listed more than once it still has one
/// node, at its first position, and the calls of every CFG listed under it
/// are edges of that node.
///
/// [`call_graph_indexed`] also returns the index of the functions by key.
#[must_use]
pub fn call_graph<I: CallInfo>(
    functions: &[(I::Callee, &Cfg<I>)],
) -> Graph<FunctionNode<I::Callee>, CallMetadata> {
    call_graph_indexed(functions).0
}

/// Build a call graph as [`call_graph`] does, together with the index of its
/// functions by key.
///
/// Resolving the calls needs that index, so returning it costs nothing more.
/// It equals [`FunctionIndex::compute`] on the returned graph: a key listed
/// more than once maps to its one node, at its first position.
///
/// # Examples
///
/// ```
/// use cfglib::{CallInfo, Cfg, call_graph_indexed, recursive_functions};
///
/// /// A call to a named function, or plain work.
/// struct Op(Option<&'static str>);
///
/// impl CallInfo for Op {
///     type Callee = &'static str;
///
///     fn callee(&self) -> Option<&'static str> {
///         self.0
///     }
/// }
///
/// let calling = |callee| {
///     let mut cfg = Cfg::new();
///     cfg.block_mut(cfg.entry()).push(Op(callee));
///     cfg
/// };
/// let (main, even, odd) = (calling(Some("even")), calling(Some("odd")), calling(Some("even")));
///
/// let (graph, index) = call_graph_indexed(&[("main", &main), ("even", &even), ("odd", &odd)]);
/// let [main, even, odd] = ["main", "even", "odd"].map(|key| index.get(&key).unwrap());
/// assert!(graph.successors(main).eq([even]));
///
/// let recursive = recursive_functions(&graph);
/// assert!(!recursive[main.index()]);
/// assert!(recursive[even.index()] && recursive[odd.index()]);
/// ```
#[must_use]
pub fn call_graph_indexed<I: CallInfo>(
    functions: &[(I::Callee, &Cfg<I>)],
) -> (FunctionGraph<I::Callee>, FunctionIndex<I::Callee>) {
    let mut graph = Graph::with_capacity(functions.len(), functions.len());
    let mut key_to_id = BTreeMap::new();

    for (key, _) in functions {
        if key_to_id.contains_key(key) {
            continue;
        }
        let id = graph.add_node(FunctionNode { id: key.clone() });
        key_to_id.insert(key.clone(), id);
    }

    for (caller_key, cfg) in functions {
        let Some(&caller) = key_to_id.get(caller_key) else {
            continue;
        };
        for block in cfg.blocks() {
            for instruction in block.instructions() {
                if let Some(callee_key) = instruction.callee()
                    && let Some(&callee) = key_to_id.get(&callee_key)
                {
                    graph.add_edge(
                        caller,
                        callee,
                        CallMetadata {
                            is_tail_call: instruction.is_tail_call(),
                        },
                    );
                }
            }
        }
    }

    (graph, FunctionIndex { by_key: key_to_id })
}

/// Find a function node by its identity.
///
/// The answer is a scan of the nodes in index order, O(n) per call, that
/// returns the first node with identity `id`. A consumer that resolves many
/// keys should compute a [`FunctionIndex`] once, or keep the one
/// [`call_graph_indexed`] returns.
#[must_use]
pub fn find_function<C: PartialEq>(
    graph: &Graph<FunctionNode<C>, CallMetadata>,
    id: &C,
) -> Option<NodeId> {
    graph.node_ids().find(|&node| graph.node(node).id == *id)
}

/// Return whether a function is directly or mutually recursive.
///
/// A function is recursive when it calls itself or one of its callees reaches
/// it again. The answer is a walk from the function's callees that stops when
/// it returns to the function, so it costs the part of the call graph the
/// function reaches: O(n + m) per call when that is most of the graph. A
/// consumer that classifies every function should call
/// [`recursive_functions`] once, which answers for all of them in O(n + m).
///
/// # Panics
///
/// Panics when `function` is not a node of `graph`.
#[must_use]
pub fn is_recursive_function<C>(
    graph: &Graph<FunctionNode<C>, CallMetadata>,
    function: NodeId,
) -> bool {
    assert!(
        graph.contains_node(function),
        "node {} is not a node of the call graph",
        function.index()
    );
    let mut visited = vec![false; graph.node_bound()];
    let mut pending: Vec<NodeId> = graph.successors(function).collect();
    while let Some(node) = pending.pop() {
        if node == function {
            return true;
        }
        if !core::mem::replace(&mut visited[node.index()], true) {
            pending.extend(graph.successors(node));
        }
    }
    false
}

/// Return for every node whether it is directly or mutually recursive.
///
/// The answer for all nodes comes from one [`tarjan_scc`] pass, O(n + m): a
/// node is recursive when its strong component has several nodes or its one
/// node calls itself, so entry `f` equals [`is_recursive_function`] for every
/// live node `f`.
///
/// The vector is indexed by [`NodeId::index`] and sized by the graph's node
/// *bound*; a removed node's slot is `false`. Generic over any [`Graph`], like
/// [`propagate_summaries`].
#[must_use]
pub fn recursive_functions<N, E>(graph: &Graph<N, E>) -> Vec<bool> {
    let mut recursive = vec![false; graph.node_bound()];
    let components = tarjan_scc(graph);
    for component in components.components() {
        if components.component(component[0]).is_cyclic(graph) {
            for &node in component {
                recursive[node.index()] = true;
            }
        }
    }
    recursive
}

/// Compute a per-node summary over a dependency graph in callee-first
/// order, iterating cyclic components to a fixpoint.
///
/// The interprocedural scaffold: nodes are functions, edges point from
/// caller to callee, and `compute` derives one function's summary while
/// reading its callees' current summaries out of the slice (indexed by
/// [`NodeId::index`]). Acyclic call graphs get exactly one `compute` per
/// function with every callee already final; recursive components iterate
/// until their summaries stabilize, so `compute` must be monotone over a
/// finite-height summary domain for termination.
/// [`propagate_summaries_widened`] runs the same schedule with a widening
/// hook, for summary domains of infinite height.
///
/// Inside a recursive component, `compute` runs once for every member, callees
/// first along a depth-first postorder of the component, and then again only
/// for the callers of a member whose summary changed. A summary therefore
/// crosses one call edge per `compute`, not one per pass over the component.
///
/// The returned vector is indexed by [`NodeId::index`] and sized by the
/// graph's node *bound*, so a removed node keeps `bottom` in its slot.
///
/// Generic over any [`Graph`] — the same shape serves module
/// graphs, type-relation closures, or any callee-first aggregation.
#[must_use]
pub fn propagate_summaries<N, E, S: Clone + PartialEq>(
    graph: &Graph<N, E>,
    bottom: &S,
    compute: impl FnMut(&Graph<N, E>, NodeId, &[S]) -> S,
) -> Vec<S> {
    propagate(graph, bottom, compute, None::<fn(NodeId, &S, &S) -> S>)
}

/// Compute per-node summaries as [`propagate_summaries`] does, and widen at
/// the functions that cut the recursion, so that summary domains of infinite
/// height (intervals, stack depths, argument ranges) converge too.
///
/// The schedule is the one of [`propagate_summaries`]. Inside a recursive
/// component, a widening point is a member that reads a callee of its own
/// component that does not run before it in the callee-first order: a later
/// member, or itself through a self-call. A cycle cannot run strictly
/// callee-first all the way around, so every cycle of the component holds a
/// widening point: the points form a feedback vertex set of the component.
///
/// When `compute` changes the summary of a widening point, the summary stored
/// is `widen(node, previous, computed)`, not `computed`; `previous` is
/// `bottom` the first time. A summary that `widen` leaves at `previous` does
/// not count as changed. Every other function, and every function outside a
/// recursive component, stores what `compute` returns.
///
/// The iteration terminates when `widen` is a widening operator: whatever
/// summaries `compute` returns, the summaries stored at one widening point
/// stop changing after finitely many steps. When `widen` also returns an upper
/// bound of its two summaries and `compute` is monotone, every returned summary
/// is at least what `compute` derives from the returned summaries of the
/// function's callees: a sound over-approximation of the least fixpoint. A
/// `widen` that returns `computed` makes this function
/// [`propagate_summaries`], call for call.
///
/// # Examples
///
/// The deepest stack a call reaches is the function's frame plus the deepest
/// stack of its callees. Recursion makes that grow forever; the widening
/// sends a depth that grows again to `u32::MAX`, unbounded.
///
/// ```
/// use cfglib::{Graph, propagate_summaries_widened};
///
/// // Frame sizes: main → even ⇄ odd, and even → leaf.
/// let mut graph = Graph::<u32, ()>::new();
/// let [main, even, odd, leaf] = [16, 32, 32, 8].map(|frame| graph.add_node(frame));
/// for (caller, callee) in [(main, even), (even, odd), (odd, even), (even, leaf)] {
///     graph.add_edge(caller, callee, ());
/// }
///
/// let depths = propagate_summaries_widened(
///     &graph,
///     &None,
///     |graph, node, depths: &[Option<u32>]| {
///         let deepest = graph.successors(node).filter_map(|callee| depths[callee.index()]);
///         Some(graph.node(node).saturating_add(deepest.max().unwrap_or(0)))
///     },
///     |_, previous, computed| match (*previous, *computed) {
///         (Some(previous), Some(computed)) if computed > previous => Some(u32::MAX),
///         (previous, computed) => previous.max(computed),
///     },
/// );
/// assert_eq!(depths[leaf.index()], Some(8));
/// assert_eq!(depths[main.index()], Some(u32::MAX));
/// ```
#[must_use]
pub fn propagate_summaries_widened<N, E, S: Clone + PartialEq>(
    graph: &Graph<N, E>,
    bottom: &S,
    compute: impl FnMut(&Graph<N, E>, NodeId, &[S]) -> S,
    widen: impl FnMut(NodeId, &S, &S) -> S,
) -> Vec<S> {
    propagate(graph, bottom, compute, Some(widen))
}

/// The schedule of [`propagate_summaries`], with `widen` applied at the
/// widening points of [`propagate_summaries_widened`] when it is given.
fn propagate<N, E, S: Clone + PartialEq>(
    graph: &Graph<N, E>,
    bottom: &S,
    mut compute: impl FnMut(&Graph<N, E>, NodeId, &[S]) -> S,
    mut widen: Option<impl FnMut(NodeId, &S, &S) -> S>,
) -> Vec<S> {
    let mut summaries = alloc::vec![bottom.clone(); graph.node_bound()];
    let components = tarjan_scc(graph);
    // Each member's position in its component's callee-first order.
    let mut rank = alloc::vec![UNSEEN; graph.node_bound()];
    // Scratch reused by every recursive component.
    let mut order: Vec<NodeId> = Vec::new();
    let mut walk = Vec::new();
    let mut queued: Vec<bool> = Vec::new();
    let mut widens: Vec<bool> = Vec::new();
    let mut pending: BinaryHeap<Reverse<usize>> = BinaryHeap::new();
    // Components arrive in reverse topological order: callees first.
    for (component_index, component) in components.components().enumerate() {
        if !components.component(component[0]).is_cyclic(graph) {
            let node = component[0];
            summaries[node.index()] = compute(graph, node, &summaries);
            continue;
        }
        // Depth-first postorder over the component's own call edges, from
        // each member in ascending order: a callee comes before its callers
        // except along the edges that close a cycle.
        order.clear();
        for &start in component {
            if rank[start.index()] != UNSEEN {
                continue;
            }
            rank[start.index()] = OPEN;
            walk.push((start, graph.successors(start)));
            while let Some((node, callees)) = walk.last_mut() {
                let node = *node;
                match callees.next() {
                    Some(callee)
                        if rank[callee.index()] == UNSEEN
                            && components.component_index(callee) == component_index =>
                    {
                        rank[callee.index()] = OPEN;
                        walk.push((callee, graph.successors(callee)));
                    }
                    Some(_) => {}
                    None => {
                        rank[node.index()] = order.len();
                        order.push(node);
                        walk.pop();
                    }
                }
            }
        }
        // A member widens when it reads a member of the component ranked no
        // earlier than itself. Ranks cannot fall all the way around a cycle,
        // so every cycle has such a call edge.
        if widen.is_some() {
            widens.clear();
            widens.extend(order.iter().enumerate().map(|(position, &node)| {
                graph.successors(node).any(|callee| {
                    components.component_index(callee) == component_index
                        && rank[callee.index()] >= position
                })
            }));
        }
        // Every member starts queued; a change queues the member's callers
        // inside the component, and the lowest rank runs first.
        queued.clear();
        queued.resize(order.len(), true);
        pending.clear();
        pending.extend((0..order.len()).map(Reverse));
        while let Some(Reverse(position)) = pending.pop() {
            queued[position] = false;
            let node = order[position];
            let mut updated = compute(graph, node, &summaries);
            if updated == summaries[node.index()] {
                continue;
            }
            if let Some(widen) = widen.as_mut()
                && widens[position]
            {
                updated = widen(node, &summaries[node.index()], &updated);
                if updated == summaries[node.index()] {
                    continue;
                }
            }
            summaries[node.index()] = updated;
            for caller in graph.predecessors(node) {
                if components.component_index(caller) == component_index {
                    let caller_rank = rank[caller.index()];
                    if !core::mem::replace(&mut queued[caller_rank], true) {
                        pending.push(Reverse(caller_rank));
                    }
                }
            }
        }
    }
    summaries
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::traverse::topological_sort;
    use crate::test_util::rng::Rng;
    use crate::test_util::{DfInst, df_call, df_ff};
    use alloc::vec;

    #[test]
    fn build_from_cfgs_resolves_calls() {
        let mut main_cfg: Cfg<DfInst> = Cfg::new();
        main_cfg
            .block_mut(main_cfg.entry())
            .push(df_call("call", "helper", false));
        let mut helper_cfg: Cfg<DfInst> = Cfg::new();
        helper_cfg.block_mut(helper_cfg.entry()).push(df_ff("ret"));

        let graph = call_graph(&[("main", &main_cfg), ("helper", &helper_cfg)]);
        let main = find_function(&graph, &"main").unwrap();
        let helper = find_function(&graph, &"helper").unwrap();
        assert!(graph.successors(main).any(|callee| callee == helper));
        assert!(!graph.edges().next().unwrap().payload().is_tail_call);
    }

    #[test]
    fn a_key_listed_twice_is_one_function_calling_for_both_cfgs() {
        let calling = |callee: &'static str| {
            let mut cfg: Cfg<DfInst> = Cfg::new();
            cfg.block_mut(cfg.entry())
                .push(df_call("call", callee, false));
            cfg
        };
        let (first, second) = (calling("left"), calling("right"));
        let mut leaf: Cfg<DfInst> = Cfg::new();
        leaf.block_mut(leaf.entry()).push(df_ff("ret"));

        let graph = call_graph(&[
            ("main", &first),
            ("left", &leaf),
            ("main", &second),
            ("right", &leaf),
        ]);

        assert_eq!(graph.node_count(), 3, "one node per distinct key");
        let main = find_function(&graph, &"main").unwrap();
        let callees: Vec<_> = graph
            .successors(main)
            .map(|callee| graph.node(callee).id)
            .collect();
        assert_eq!(callees, ["left", "right"]);
    }

    #[test]
    fn topology_and_recursion_use_shared_algorithms() {
        let mut graph = Graph::new();
        let main = graph.add_node(FunctionNode { id: "main" });
        let helper = graph.add_node(FunctionNode { id: "helper" });
        graph.add_edge(
            main,
            helper,
            CallMetadata {
                is_tail_call: false,
            },
        );
        assert_eq!(topological_sort(&graph), Some(vec![main, helper]));
        assert_eq!(graph.predecessors(main).count(), 0);
        assert_eq!(graph.successors(helper).count(), 0);

        graph.add_edge(
            helper,
            main,
            CallMetadata {
                is_tail_call: false,
            },
        );
        assert!(topological_sort(&graph).is_none());
        assert!(is_recursive_function(&graph, main));
    }

    #[test]
    fn summaries_propagate_callee_first_and_stabilize_cycles() {
        // main calls a; a and b are mutually recursive; a calls leaf.
        let mut graph = Graph::new();
        let main = graph.add_node(FunctionNode { id: "main" });
        let a = graph.add_node(FunctionNode { id: "a" });
        let b = graph.add_node(FunctionNode { id: "b" });
        let leaf = graph.add_node(FunctionNode { id: "leaf" });
        let call = CallMetadata {
            is_tail_call: false,
        };
        graph.add_edge(main, a, call.clone());
        graph.add_edge(a, b, call.clone());
        graph.add_edge(b, a, call.clone());
        graph.add_edge(a, leaf, call);

        // Summary: the set of transitively reachable callees, by id.
        let summaries = propagate_summaries(
            &graph,
            &alloc::collections::BTreeSet::new(),
            |graph, node, summaries| {
                let mut reach = alloc::collections::BTreeSet::new();
                for callee in graph.successors(node) {
                    reach.insert(graph.node(callee).id);
                    reach.extend(summaries[callee.index()].iter().copied());
                }
                reach
            },
        );
        let names = |node: NodeId| summaries[node.index()].clone();
        assert!(names(leaf).is_empty());
        assert_eq!(names(main), ["a", "b", "leaf"].into_iter().collect());
        assert_eq!(names(a), ["a", "b", "leaf"].into_iter().collect());
        assert_eq!(names(b), ["a", "b", "leaf"].into_iter().collect());
    }

    /// The walk stops at the function, ignores cycles it never re-enters, and
    /// answers for every function what the function's component answers.
    #[test]
    fn recursion_matches_the_strong_components() {
        let mut graph = Graph::new();
        let call = || CallMetadata {
            is_tail_call: false,
        };
        let ids = ["main", "a", "b", "leaf", "loop", "self", "dead"];
        let nodes: Vec<_> = ids
            .iter()
            .map(|&id| graph.add_node(FunctionNode { id }))
            .collect();
        let [main, a, b, leaf, looping, selfish, dead] = nodes[..] else {
            unreachable!("seven functions");
        };
        for (caller, callee) in [
            (main, a),
            (a, b),
            (b, a),
            (a, leaf),
            (main, looping),
            (looping, selfish),
            (selfish, selfish),
            (dead, main),
        ] {
            graph.add_edge(caller, callee, call());
        }
        let components = tarjan_scc(&graph);
        for node in graph.node_ids() {
            assert_eq!(
                is_recursive_function(&graph, node),
                components.component(node).is_cyclic(&graph),
                "{}",
                graph.node(node).id
            );
        }
        assert!(!is_recursive_function(&graph, main));
        assert!(!is_recursive_function(&graph, looping));
        assert!(is_recursive_function(&graph, selfish));
    }

    #[test]
    #[should_panic(expected = "is not a node of the call graph")]
    fn a_removed_function_is_rejected() {
        let mut graph = Graph::<FunctionNode<&str>, CallMetadata>::new();
        let gone = graph.add_node(FunctionNode { id: "gone" });
        graph.remove_node(gone);
        let _ = is_recursive_function(&graph, gone);
    }

    /// A ring whose only seed sits at its last member, against the id order:
    /// the former round-robin schedule ran one pass over the ring per call
    /// edge the seed had to cross, about k² transfers.
    #[test]
    fn a_seed_crosses_a_recursive_ring_in_one_transfer_per_member() {
        const MEMBERS: usize = 1_000;
        let mut graph = Graph::<(), ()>::new();
        let members: Vec<NodeId> = (0..MEMBERS).map(|_| graph.add_node(())).collect();
        for (index, &caller) in members.iter().enumerate() {
            graph.add_edge(caller, members[(index + 1) % MEMBERS], ());
        }
        let mut calls = 0_usize;
        let summaries = propagate_summaries(&graph, &false, |graph, node, summaries| {
            calls += 1;
            node.index() + 1 == MEMBERS
                || graph
                    .successors(node)
                    .any(|callee| summaries[callee.index()])
        });
        assert!(summaries.iter().all(|&summary| summary));
        assert!(calls <= 3 * MEMBERS, "{calls} transfers");
    }

    /// The schedule `propagate_summaries` ran before the worklist: every
    /// member of a recursive component again, in ascending order, until a
    /// whole pass changes nothing.
    fn round_robin<S: Clone + PartialEq>(
        graph: &Graph<(), ()>,
        bottom: &S,
        mut compute: impl FnMut(&Graph<(), ()>, NodeId, &[S]) -> S,
    ) -> Vec<S> {
        let mut summaries = vec![bottom.clone(); graph.node_bound()];
        let components = tarjan_scc(graph);
        for component in components.components() {
            let cyclic = components.component(component[0]).is_cyclic(graph);
            loop {
                let mut changed = false;
                for &node in component {
                    let updated = compute(graph, node, &summaries);
                    if updated != summaries[node.index()] {
                        summaries[node.index()] = updated;
                        changed = true;
                    }
                }
                if !cyclic || !changed {
                    break;
                }
            }
        }
        summaries
    }

    /// A random graph of at most 24 nodes with parallel edges, self-calls,
    /// recursive clusters and removed slots.
    fn random_graph(rng: &mut Rng) -> Graph<(), ()> {
        let node_count = 1 + rng.below(24);
        let mut graph = Graph::<(), ()>::new();
        let nodes: Vec<NodeId> = (0..node_count).map(|_| graph.add_node(())).collect();
        for _ in 0..rng.below(3 * node_count + 1) {
            graph.add_edge(
                nodes[rng.below(node_count)],
                nodes[rng.below(node_count)],
                (),
            );
        }
        for &node in &nodes {
            if rng.one_in(8) {
                graph.remove_node(node);
            }
        }
        graph
    }

    /// The ids a function reaches, modulo 64: a monotone summary of finite
    /// height whose first `compute` always changes `0`.
    fn reached_ids(graph: &Graph<(), ()>, node: NodeId, summaries: &[u64]) -> u64 {
        graph
            .successors(node)
            .fold(1_u64 << (node.index() % 64), |mask, callee| {
                mask | summaries[callee.index()]
            })
    }

    /// Random graphs with self-calls, recursive clusters and removed slots,
    /// under a monotone finite-height summary (the reached ids modulo 64): the
    /// worklist reaches the round-robin fixpoint exactly.
    #[test]
    fn the_worklist_reaches_the_round_robin_fixpoint() {
        let mut rng = Rng::new(93);
        for _ in 0..300 {
            let graph = random_graph(&mut rng);
            assert_eq!(
                propagate_summaries(&graph, &0, reached_ids),
                round_robin(&graph, &0, reached_ids)
            );
        }
    }

    /// Every first `compute` changes a summary, so `widen` sees every widening
    /// point. The points are recursive functions and cut every cycle, and a
    /// `widen` that keeps the computed summary changes neither the result nor
    /// the `compute` calls of `propagate_summaries`.
    #[test]
    fn the_widening_points_cut_every_cycle() {
        let mut rng = Rng::new(9_393);
        for _ in 0..300 {
            let graph = random_graph(&mut rng);
            let mut plain_calls = 0_usize;
            let plain = propagate_summaries(&graph, &0, |graph, node, summaries| {
                plain_calls += 1;
                reached_ids(graph, node, summaries)
            });
            let mut widened_calls = 0_usize;
            let mut points = vec![false; graph.node_bound()];
            let widened = propagate_summaries_widened(
                &graph,
                &0,
                |graph, node, summaries| {
                    widened_calls += 1;
                    reached_ids(graph, node, summaries)
                },
                |node, previous, computed| {
                    assert_ne!(previous, computed, "`widen` runs on a change only");
                    points[node.index()] = true;
                    *computed
                },
            );
            assert_eq!(widened, plain);
            assert_eq!(widened_calls, plain_calls);

            let recursive = recursive_functions(&graph);
            let mut cut = graph.clone();
            for node in graph.node_ids() {
                if points[node.index()] {
                    assert!(recursive[node.index()], "a widening point is recursive");
                    cut.remove_node(node);
                }
            }
            assert!(
                !recursive_functions(&cut).contains(&true),
                "removing the widening points leaves no cycle"
            );
        }
    }

    /// The deepest stack over a ring of one-byte frames grows forever. The
    /// only widening point is the member whose call closes the ring in the
    /// callee-first order, and the depths settle in about two passes.
    #[test]
    fn a_recursive_ring_widens_at_one_member() {
        const MEMBERS: usize = 1_000;
        let mut graph = Graph::<(), ()>::new();
        let members: Vec<NodeId> = (0..MEMBERS).map(|_| graph.add_node(())).collect();
        for (index, &caller) in members.iter().enumerate() {
            graph.add_edge(caller, members[(index + 1) % MEMBERS], ());
        }
        let mut calls = 0_usize;
        let mut widened = Vec::new();
        let depths = propagate_summaries_widened(
            &graph,
            &None,
            |graph, node, depths: &[Option<u64>]| {
                calls += 1;
                let deepest = graph
                    .successors(node)
                    .filter_map(|callee| depths[callee.index()])
                    .max();
                Some(deepest.unwrap_or(0).saturating_add(1))
            },
            |node, previous, computed| {
                widened.push(node);
                match (*previous, *computed) {
                    (Some(previous), Some(computed)) if computed > previous => Some(u64::MAX),
                    (previous, computed) => previous.max(computed),
                }
            },
        );
        assert!(depths.iter().all(|&depth| depth == Some(u64::MAX)));
        assert_eq!(
            widened,
            [members[MEMBERS - 1]; 2],
            "set from bottom once, then sent to unbounded"
        );
        assert!(calls <= 3 * MEMBERS, "{calls} transfers");
    }
}
