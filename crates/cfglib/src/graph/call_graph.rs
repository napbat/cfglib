//! Whole-program call-graph construction on [`Graph`], and the callee-first
//! summary propagation over any [`GraphView`].
//!
//! A call graph does not need its own storage abstraction. Functions are node
//! payloads, call details are edge payloads, and all traversal, SCC, dominance,
//! and topological algorithms operate on the returned generic graph directly.
//! Call targets come from the instructions themselves via
//! [`CallInfo`] — callee identities are consumer-typed (symbol ids, addresses,
//! names), never a library-owned string.

extern crate alloc;
use alloc::collections::BTreeMap;
use alloc::vec;
use alloc::vec::Vec;

use crate::cfg::Cfg;
use crate::dataflow::worklist::Worklist;
use crate::flow::CallInfo;
use crate::graph::scc::tarjan_scc;
use crate::graph::store::{Graph, NodeId};
use crate::graph::view::{DenseId, GraphView, NodeView};
use crate::identity::{NO_INDEX, slot, word};

/// A function on the walk that places its component's members, in the rank
/// table of [`propagate_summaries`], where [`NO_INDEX`] marks a function not
/// yet placed in its component's callee-first order.
const OPEN: u32 = NO_INDEX - 1;

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
pub fn find_function<G, C>(graph: &G, id: &C) -> Option<G::NodeId>
where
    G: NodeView<NodeData = FunctionNode<C>>,
    C: PartialEq,
{
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
/// It takes the store rather than a view because it rejects a removed node,
/// which only the store can tell from a live one without a scan of the
/// nodes.
///
/// # Panics
///
/// Panics when `function` is not a node of `graph`.
#[must_use]
pub fn is_recursive_function<N, E>(graph: &Graph<N, E>, function: NodeId) -> bool {
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
/// The vector is indexed by [`DenseId::index`] and sized by the view's node
/// *bound*; a removed node's slot is `false`.
#[must_use]
pub fn recursive_functions<G: GraphView>(graph: &G) -> Vec<bool> {
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

/// A per-node summary over a dependency graph, which
/// [`propagate_summaries`] computes callee-first.
///
/// Nodes are functions, edges point from caller to callee, and
/// [`compute`](Self::compute) derives one function's summary while reading
/// its callees' current summaries. The widening and narrowing hooks take
/// the parameters of [`TryNodeProblem`](crate::TryNodeProblem)'s (the graph,
/// the node, the summary it held and the one now computed) and default to
/// changing nothing; [`SummaryConfig`] decides whether they run.
pub trait SummaryProblem<G: GraphView> {
    /// The summary of one function.
    type Summary: Clone + PartialEq;

    /// The summary every function starts with, and keeps when it is removed.
    fn bottom(&self, graph: &G) -> Self::Summary;

    /// The summary of `node`, from its callees' current summaries, indexed
    /// by [`DenseId::index`].
    fn compute(&self, graph: &G, node: G::NodeId, summaries: &[Self::Summary]) -> Self::Summary;

    /// Widens at the widening point `node`: combines the summary it held
    /// with the one `compute` returned.
    ///
    /// It must return a summary at least both arguments, and the summaries
    /// stored at one widening point must stop changing after finitely many
    /// steps whatever `compute` returns. The default returns `next`, which is
    /// no widening.
    fn widen(
        &self,
        _graph: &G,
        _node: G::NodeId,
        _previous: &Self::Summary,
        next: &Self::Summary,
    ) -> Self::Summary {
        next.clone()
    }

    /// Narrows at the widening point `node` during the descending pass:
    /// refines the summary it holds with the one `compute` returned.
    ///
    /// It should return a summary between `next` and `previous`, and the
    /// summaries it stores at one widening point must stop changing after
    /// finitely many steps. The default returns `next`, a plain descending
    /// step.
    fn narrow(
        &self,
        _graph: &G,
        _node: G::NodeId,
        _previous: &Self::Summary,
        next: &Self::Summary,
    ) -> Self::Summary {
        next.clone()
    }
}

/// Which hooks of a [`SummaryProblem`] [`propagate_summaries`] runs. The
/// default runs neither: a plain least fixpoint over a finite-height
/// summary domain.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct SummaryConfig {
    /// Widen at the widening points of each recursive component, so that
    /// summary domains of infinite height (intervals, stack depths, argument
    /// ranges) converge too.
    pub widening: bool,
    /// Once a recursive component is stable, run a descending pass over it
    /// that narrows at its widening points, before its callers read it.
    pub narrowing: bool,
}

/// Compute a per-node summary over a dependency graph in callee-first
/// order, iterating cyclic components to a fixpoint.
///
/// The interprocedural scaffold: acyclic call graphs get exactly one
/// `compute` per function with every callee already final; recursive
/// components iterate until their summaries stabilize, so without widening
/// `compute` must be monotone over a finite-height summary domain for
/// termination.
///
/// Inside a recursive component, `compute` runs once for every member,
/// callees first along a depth-first postorder of the component, and then
/// again only for the callers of a member whose summary changed. A summary
/// therefore crosses one call edge per `compute`, not one per pass over the
/// component. This schedule is its own, not a solve in a weak topological
/// order: callee-first along the call edges is what makes an acyclic call
/// graph cost one `compute` per function.
///
/// - **Widening** ([`SummaryConfig::widening`]). A widening point is a member
///   that reads a callee of its own component that does not run before it in
///   the callee-first order: a later member, or itself through a self-call. A
///   cycle cannot run strictly callee-first all the way around, so every
///   cycle of the component holds a widening point: the points form a
///   feedback vertex set of the component. When `compute` changes the
///   summary of a widening point, the summary stored is
///   [`widen`](SummaryProblem::widen)`(graph, node, previous, computed)`, not `computed`;
///   `previous` is `bottom` the first time. A summary that `widen` leaves at
///   `previous` does not count as changed. Every other function stores what
///   `compute` returns. The iteration terminates when `widen` is a widening
///   operator; when it also returns an upper bound of its two summaries and
///   `compute` is monotone, every returned summary is a sound
///   over-approximation of the least fixpoint.
/// - **Narrowing** ([`SummaryConfig::narrowing`]). Once a recursive component
///   has stabilized, a descending pass recomputes its members in the same
///   callee-first order: every member stores what `compute` returns, except a
///   widening point, which stores [`narrow`](SummaryProblem::narrow)`(graph, node, current,
///   computed)`. A changed summary queues its callers inside the component
///   again, and the pass ends when nothing changes. The callers outside the
///   component run only after that, so they read the narrowed summaries.
///
/// The returned vector is indexed by [`DenseId::index`] and sized by the
/// view's node *bound*, so a removed node keeps `bottom` in its slot. The
/// same shape serves module graphs, type-relation closures, or any
/// callee-first aggregation.
///
/// # Examples
///
/// A value that every call raises by one but saturates at 255. Recursion
/// makes it climb, the widening sends a climbing value to `u32::MAX`, and
/// narrowing brings it back down to the 255 the saturation gives.
///
/// ```
/// use cfglib::{Graph, NodeId, SummaryConfig, SummaryProblem, propagate_summaries};
///
/// /// Each node's payload says whether it saturates what it returns.
/// struct Bound;
///
/// impl SummaryProblem<Graph<bool, ()>> for Bound {
///     type Summary = u32;
///     fn bottom(&self, _: &Graph<bool, ()>) -> u32 {
///         0
///     }
///     fn compute(&self, graph: &Graph<bool, ()>, node: NodeId, bounds: &[u32]) -> u32 {
///         let highest = graph.successors(node).map(|callee| bounds[callee.index()]).max();
///         let bound = highest.map_or(0, |bound| bound.saturating_add(1));
///         if *graph.node(node) { bound.min(255) } else { bound }
///     }
///     fn widen(&self, _: &Graph<bool, ()>, _: NodeId, previous: &u32, computed: &u32) -> u32 {
///         if computed > previous { u32::MAX } else { *previous }
///     }
///     fn narrow(&self, _: &Graph<bool, ()>, _: NodeId, current: &u32, computed: &u32) -> u32 {
///         if *current == u32::MAX { *computed } else { *current }
///     }
/// }
///
/// // main → even ⇄ odd, and `odd` saturates what it returns at 255.
/// let mut graph = Graph::<bool, ()>::new();
/// let [main, even, odd] = [false, false, true].map(|saturates| graph.add_node(saturates));
/// for (caller, callee) in [(main, even), (even, odd), (odd, even)] {
///     graph.add_edge(caller, callee, ());
/// }
///
/// // Widening alone leaves the cycle, and so `main`, unbounded.
/// let widening = SummaryConfig { widening: true, narrowing: false };
/// let widened = propagate_summaries(&graph, &Bound, widening);
/// assert_eq!(widened[main.index()], u32::MAX);
/// // Narrowing recovers the saturation: odd 255, even 256, main 257.
/// let narrowing = SummaryConfig { widening: true, narrowing: true };
/// let narrowed = propagate_summaries(&graph, &Bound, narrowing);
/// assert_eq!([narrowed[odd.index()], narrowed[even.index()], narrowed[main.index()]], [255, 256, 257]);
/// ```
#[must_use]
pub fn propagate_summaries<G, P>(graph: &G, problem: &P, config: SummaryConfig) -> Vec<P::Summary>
where
    G: GraphView,
    P: SummaryProblem<G>,
{
    let mut summaries = vec![problem.bottom(graph); graph.node_bound()];
    let components = tarjan_scc(graph);
    // Each member's position in its component's callee-first order.
    let mut rank = vec![NO_INDEX; graph.node_bound()];
    // Scratch reused by every recursive component.
    let mut order: Vec<G::NodeId> = Vec::new();
    let mut walk = Vec::new();
    let mut widens: Vec<bool> = Vec::new();
    let mut pending = Worklist::new(0);
    // Components arrive in reverse topological order: callees first.
    for (component_index, component) in components.components().enumerate() {
        if !components.component(component[0]).is_cyclic(graph) {
            let node = component[0];
            summaries[node.index()] = problem.compute(graph, node, &summaries);
            continue;
        }
        // Depth-first postorder over the component's own call edges, from
        // each member in ascending order: a callee comes before its callers
        // except along the edges that close a cycle.
        order.clear();
        for &start in component {
            if rank[start.index()] != NO_INDEX {
                continue;
            }
            rank[start.index()] = OPEN;
            walk.push((start, graph.successors(start)));
            while let Some((node, callees)) = walk.last_mut() {
                let node = *node;
                match callees.next() {
                    Some(callee)
                        if rank[callee.index()] == NO_INDEX
                            && components.component_index(callee) == component_index =>
                    {
                        rank[callee.index()] = OPEN;
                        walk.push((callee, graph.successors(callee)));
                    }
                    Some(_) => {}
                    None => {
                        rank[node.index()] = word(order.len());
                        order.push(node);
                        walk.pop();
                    }
                }
            }
        }
        // A member widens when it reads a member of the component ranked no
        // earlier than itself. Ranks cannot fall all the way around a cycle,
        // so every cycle has such a call edge.
        if config.widening || config.narrowing {
            widens.clear();
            widens.extend(order.iter().enumerate().map(|(position, &node)| {
                graph.successors(node).any(|callee| {
                    components.component_index(callee) == component_index
                        && slot(rank[callee.index()]) >= position
                })
            }));
        }
        // Every member starts queued; a change queues the member's callers
        // inside the component, and the lowest rank runs first.
        pending.queue_all(order.len());
        while let Some(position) = pending.pop_first() {
            let node = order[position];
            let mut updated = problem.compute(graph, node, &summaries);
            if updated == summaries[node.index()] {
                continue;
            }
            if config.widening && widens[position] {
                updated = problem.widen(graph, node, &summaries[node.index()], &updated);
                if updated == summaries[node.index()] {
                    continue;
                }
            }
            summaries[node.index()] = updated;
            for caller in graph.predecessors(node) {
                if components.component_index(caller) == component_index {
                    pending.insert(slot(rank[caller.index()]));
                }
            }
        }
        if config.narrowing {
            let ranking = Ranking {
                order: &order,
                rank: &rank,
                widens: &widens,
            };
            descend(graph, problem, &mut summaries, &ranking, &mut pending);
        }
    }
    summaries
}

/// One recursive component in the callee-first order of
/// [`propagate_summaries`].
struct Ranking<'a, N> {
    /// The members in callee-first order.
    order: &'a [N],
    /// Each member's position in `order`, by node index.
    rank: &'a [u32],
    /// Whether the member at each position is a widening point.
    widens: &'a [bool],
}

/// The descending pass of [`propagate_summaries`] over a recursive component
/// that is stable under widening: recompute every member from those
/// summaries in the same order, narrowing at the same points, until nothing
/// changes, so that the component's callers read the narrowed summaries.
fn descend<G, P>(
    graph: &G,
    problem: &P,
    summaries: &mut [P::Summary],
    ranking: &Ranking<'_, G::NodeId>,
    pending: &mut Worklist,
) where
    G: GraphView,
    P: SummaryProblem<G>,
{
    let Ranking {
        order,
        rank,
        widens,
    } = *ranking;
    pending.queue_all(order.len());
    while let Some(position) = pending.pop_first() {
        let node = order[position];
        let mut updated = problem.compute(graph, node, summaries);
        if widens[position] {
            updated = problem.narrow(graph, node, &summaries[node.index()], &updated);
        }
        if updated == summaries[node.index()] {
            continue;
        }
        summaries[node.index()] = updated;
        for caller in graph.predecessors(node) {
            // A caller outside the component has no rank, or the rank of a
            // member of another component, which `order` does not hold.
            let caller_rank = slot(rank[caller.index()]);
            if order.get(caller_rank) == Some(&caller) {
                pending.insert(caller_rank);
            }
        }
    }
}

#[cfg(test)]
mod tests;
