//! Analyses that read only a rooted graph view, checked on a `Cfg` and on a
//! `Graph` rooted with `Rooted`, each with holes and compacted.

use std::collections::BTreeSet;

use cfglib::{
    DominatorTree, GraphMetrics, PostDominatorTree, PostNode, RootedView, Scc, SccDecomposition,
    condensation, condensation_of, control_dependence_graph, detect_loops, is_reducible,
    kosaraju_scc, tarjan_scc,
};

use crate::oracle::{PostReference, Reference};
use crate::topology::{EXIT_COMPLETE, GENERAL, Realizations, SEEDS, Slots, Topology, ViewCheck};

fn realize(seed: u64, shape: crate::topology::Shape) -> Realizations {
    Realizations::of_topology(Topology::random(seed, shape)).labeled(format!("seed {seed}"))
}

fn slot_set<G: RootedView>(nodes: &[G::NodeId], slots: &Slots<G::NodeId>) -> BTreeSet<usize> {
    nodes.iter().map(|&node| slots.slot_of(node)).collect()
}

struct DominatorFacts;

impl ViewCheck for DominatorFacts {
    fn check<G: RootedView>(
        &self,
        view: &G,
        slots: &Slots<G::NodeId>,
        reference: &Reference<'_>,
        label: &str,
    ) {
        let tree = DominatorTree::compute(view);
        let reachable = reference.reachable();
        for node in reference.live() {
            let id = slots.id(node);
            let is_reachable = reachable.contains(&node);
            assert_eq!(
                tree.is_reachable(id),
                is_reachable,
                "{label}: is_reachable({node})"
            );
            assert_eq!(
                tree.idom(id),
                reference
                    .immediate_dominator(node)
                    .map(|dominator| slots.id(dominator)),
                "{label}: idom({node})"
            );
            let strict_count = u32::try_from(reference.strict_dominators(node).len())
                .expect("a dozen slots fit in u32");
            assert_eq!(
                tree.level(id),
                is_reachable.then_some(strict_count),
                "{label}: level({node})"
            );
            for dominator in reference.live() {
                assert_eq!(
                    tree.dominates(slots.id(dominator), id),
                    reference.dominates(dominator, node),
                    "{label}: dominates({dominator}, {node})"
                );
            }
        }
        assert_eq!(
            slot_set::<G>(tree.preorder(), slots),
            reachable,
            "{label}: the preorder lists exactly the reachable live nodes"
        );
        assert_eq!(
            tree.preorder().len(),
            reachable.len(),
            "{label}: preorder repeats a node"
        );
    }
}

#[test]
fn dominators_match_removal_reachability_around_holes() {
    for seed in SEEDS {
        realize(seed, GENERAL).run_views(&DominatorFacts);
    }
}

struct PostDominatorFacts;

impl ViewCheck for PostDominatorFacts {
    fn check<G: RootedView>(
        &self,
        view: &G,
        slots: &Slots<G::NodeId>,
        reference: &Reference<'_>,
        label: &str,
    ) {
        let tree = PostDominatorTree::compute(view);
        let reachable = reference.reachable();
        for block in reference.live() {
            let id = slots.id(block);
            let is_reachable = reachable.contains(&block);
            assert_eq!(
                tree.contains(id),
                is_reachable,
                "{label}: contains({block})"
            );
            let expected = is_reachable.then(|| match reference.immediate_post_dominator(block) {
                PostReference::Block(other) => PostNode::Node(slots.id(other)),
                PostReference::Exit => PostNode::Exit,
            });
            assert_eq!(tree.ipdom(id), expected, "{label}: ipdom({block})");
            if is_reachable {
                assert!(tree.reaches_exit(id), "{label}: {block} reaches an exit");
            }
            for post_dominator in reference.live() {
                assert_eq!(
                    tree.post_dominates(slots.id(post_dominator), id),
                    reference.post_dominates(post_dominator, block),
                    "{label}: post_dominates({post_dominator}, {block})"
                );
            }
        }
        assert_eq!(
            slot_set::<G>(tree.exits(), slots),
            reference.exits(),
            "{label}: exits are the reachable blocks without successors"
        );
        assert!(
            tree.cut_roots().is_empty(),
            "{label}: no exit-free region exists"
        );
    }
}

#[test]
fn post_dominators_match_exit_path_removal_around_holes() {
    for seed in SEEDS {
        realize(seed, EXIT_COMPLETE).run_views(&PostDominatorFacts);
    }
}

struct ControlDependences;

impl ViewCheck for ControlDependences {
    fn check<G: RootedView>(
        &self,
        view: &G,
        slots: &Slots<G::NodeId>,
        reference: &Reference<'_>,
        label: &str,
    ) {
        let post_dominators = PostDominatorTree::compute(view);
        let dependences = control_dependence_graph(view, &post_dominators);
        let actual: BTreeSet<(usize, usize)> = dependences
            .edges()
            .map(|edge| {
                (
                    slots.slot_of(*dependences.node(edge.source())),
                    slots.slot_of(*dependences.node(edge.target())),
                )
            })
            .collect();
        let expected: BTreeSet<(usize, usize)> = reference
            .reachable()
            .into_iter()
            .flat_map(|controller| {
                reference
                    .control_dependents(controller)
                    .into_iter()
                    .map(move |dependent| (controller, dependent))
            })
            .collect();
        assert_eq!(actual, expected, "{label}: control dependences");
    }
}

#[test]
fn control_dependences_match_the_arm_definition_around_holes() {
    for seed in SEEDS {
        realize(seed, EXIT_COMPLETE).run_views(&ControlDependences);
    }
}

fn component_slots<G: RootedView>(
    decomposition: &SccDecomposition<G::NodeId>,
    slots: &Slots<G::NodeId>,
) -> Vec<BTreeSet<usize>> {
    decomposition
        .components()
        .map(|component| slot_set::<G>(component, slots))
        .collect()
}

struct StronglyConnectedComponents;

impl ViewCheck for StronglyConnectedComponents {
    fn check<G: RootedView>(
        &self,
        view: &G,
        slots: &Slots<G::NodeId>,
        reference: &Reference<'_>,
        label: &str,
    ) {
        let expected = reference.components();
        let condensation_edges = reference.condensation_edges();
        let acyclic =
            expected.iter().all(|component| component.len() == 1) && !reference.has_self_loop();
        let tarjan = tarjan_scc(view);
        let kosaraju = kosaraju_scc(view);
        for (name, decomposition, sources_first) in
            [("tarjan", &tarjan, false), ("kosaraju", &kosaraju, true)]
        {
            let found = component_slots::<G>(decomposition, slots);
            assert_eq!(
                found.len(),
                expected.len(),
                "{label}: {name} component count"
            );
            assert_eq!(
                found.iter().cloned().collect::<BTreeSet<_>>(),
                expected,
                "{label}: {name} partition of the live nodes"
            );
            let index_of = |component: &BTreeSet<usize>| {
                let member = *component.iter().next().expect("components are never empty");
                decomposition.component_index(slots.id(member))
            };
            for node in reference.live() {
                let component = decomposition.component(slots.id(node));
                let members = slot_set::<G>(component.nodes(), slots);
                assert!(
                    members.contains(&node),
                    "{label}: {name} component of {node}"
                );
                assert_eq!(
                    found[decomposition.component_index(slots.id(node))],
                    members,
                    "{label}: {name} component_index({node}) names its component"
                );
            }
            for (source, target) in &condensation_edges {
                let (source, target) = (index_of(source), index_of(target));
                let ordered = if sources_first {
                    source < target
                } else {
                    target < source
                };
                assert!(
                    ordered,
                    "{label}: {name} numbering across {source} -> {target}"
                );
            }
            assert_eq!(
                decomposition.is_dag(view),
                acyclic,
                "{label}: {name} is_dag"
            );
        }
    }
}

#[test]
fn components_partition_live_nodes_and_keep_their_numbering_contract() {
    for seed in SEEDS {
        realize(seed, GENERAL).run_views(&StronglyConnectedComponents);
    }
}

struct Condensations;

impl ViewCheck for Condensations {
    fn check<G: RootedView>(
        &self,
        view: &G,
        slots: &Slots<G::NodeId>,
        reference: &Reference<'_>,
        label: &str,
    ) {
        let expected_components = reference.components();
        let expected_edges = reference.condensation_edges();

        let graph = condensation(view);
        let members = |component: &Scc<G::NodeId>| slot_set::<G>(component.nodes(), slots);
        let found_components: Vec<BTreeSet<usize>> = graph
            .node_ids()
            .map(|node| members(graph.node(node)))
            .collect();
        assert_eq!(
            found_components.iter().cloned().collect::<BTreeSet<_>>(),
            expected_components,
            "{label}: condensation nodes are the components"
        );
        assert_eq!(
            found_components.len(),
            expected_components.len(),
            "{label}: node count"
        );
        let found_edges: BTreeSet<(BTreeSet<usize>, BTreeSet<usize>)> = graph
            .edges()
            .map(|edge| {
                (
                    members(graph.node(edge.source())),
                    members(graph.node(edge.target())),
                )
            })
            .collect();
        assert_eq!(found_edges, expected_edges, "{label}: condensation edges");
        assert!(
            tarjan_scc(&graph).is_dag(&graph),
            "{label}: a condensation is acyclic"
        );

        let decomposition = kosaraju_scc(view);
        let numbered = condensation_of(view, &decomposition);
        assert_eq!(
            numbered.node_ids().count(),
            decomposition.len(),
            "{label}: condensation_of has one node per component"
        );
        let index_of = |component: &BTreeSet<usize>| {
            let member = *component.iter().next().expect("components are never empty");
            decomposition.component_index(slots.id(member))
        };
        let expected_numbered: BTreeSet<(usize, usize)> = expected_edges
            .iter()
            .map(|(source, target)| (index_of(source), index_of(target)))
            .collect();
        let found_numbered: BTreeSet<(usize, usize)> = numbered
            .edges()
            .map(|edge| (edge.source().index(), edge.target().index()))
            .collect();
        assert_eq!(
            found_numbered, expected_numbered,
            "{label}: condensation_of edges"
        );
    }
}

#[test]
fn condensations_are_the_quotient_graph_around_holes() {
    for seed in SEEDS {
        realize(seed, GENERAL).run_views(&Condensations);
    }
}

struct LoopFacts;

impl ViewCheck for LoopFacts {
    fn check<G: RootedView>(
        &self,
        view: &G,
        slots: &Slots<G::NodeId>,
        reference: &Reference<'_>,
        label: &str,
    ) {
        let tree = DominatorTree::compute(view);
        let found = detect_loops(view, &tree);
        let expected = reference.loops();
        assert_eq!(found.len(), expected.len(), "{label}: loop count");
        let bodies: Vec<BTreeSet<usize>> = found
            .iter()
            .map(|natural| {
                natural
                    .body
                    .iter()
                    .map(|&node| slots.slot_of(node))
                    .collect()
            })
            .collect();
        for natural in &found {
            let header = slots.slot_of(natural.header);
            let expected = expected
                .get(&header)
                .unwrap_or_else(|| panic!("{label}: {header} is not a loop header"));
            let body: BTreeSet<usize> = natural
                .body
                .iter()
                .map(|&node| slots.slot_of(node))
                .collect();
            let latches: BTreeSet<usize> = natural
                .latches
                .iter()
                .map(|&node| slots.slot_of(node))
                .collect();
            assert_eq!(body, expected.body, "{label}: body of the loop at {header}");
            assert_eq!(
                latches, expected.latches,
                "{label}: latches of the loop at {header}"
            );

            let laminar = bodies.iter().all(|other| {
                other.is_disjoint(&body) || other.is_subset(&body) || body.is_subset(other)
            });
            if laminar {
                let enclosing = bodies
                    .iter()
                    .filter(|other| body.is_subset(other) && **other != body)
                    .count();
                assert_eq!(
                    natural.depth, enclosing,
                    "{label}: depth of the loop at {header}"
                );
            }
        }
        assert_eq!(
            is_reducible(view, &tree),
            reference.is_reducible(),
            "{label}: is_reducible"
        );
    }
}

#[test]
fn natural_loops_and_reducibility_match_the_back_edge_definition_around_holes() {
    for seed in SEEDS {
        realize(seed, GENERAL).run_views(&LoopFacts);
    }
}

struct Metrics;

impl ViewCheck for Metrics {
    fn check<G: RootedView>(
        &self,
        view: &G,
        _slots: &Slots<G::NodeId>,
        reference: &Reference<'_>,
        label: &str,
    ) {
        let metrics = GraphMetrics::compute(view);
        let expected = reference.metrics();
        assert_eq!(
            metrics.node_count, expected.node_count,
            "{label}: node_count"
        );
        assert_eq!(
            metrics.edge_count, expected.edge_count,
            "{label}: edge_count"
        );
        assert_eq!(
            metrics.cyclomatic_complexity, expected.cyclomatic_complexity,
            "{label}: cyclomatic_complexity"
        );
        assert_eq!(
            metrics.reachable_node_count, expected.reachable_node_count,
            "{label}: reachable_node_count"
        );
        assert_eq!(
            metrics.unreachable_node_count, expected.unreachable_node_count,
            "{label}: unreachable_node_count"
        );
        assert_eq!(
            metrics.exit_count, expected.exit_count,
            "{label}: exit_count"
        );
        assert_eq!(
            metrics.max_nesting_depth == 0,
            reference.loops().is_empty(),
            "{label}: a loop-free graph has nesting depth zero and only that one"
        );
    }
}

#[test]
fn graph_metrics_count_live_nodes_and_edges_around_holes() {
    for seed in SEEDS {
        realize(seed, GENERAL).run_views(&Metrics);
    }
}

/// Every view analysis that holds for any rooted graph, for a case built
/// elsewhere (for example after a transform).
pub fn check_general_views(realized: &Realizations) {
    realized.run_views(&DominatorFacts);
    realized.run_views(&StronglyConnectedComponents);
    realized.run_views(&Condensations);
    realized.run_views(&LoopFacts);
    realized.run_views(&Metrics);
}
