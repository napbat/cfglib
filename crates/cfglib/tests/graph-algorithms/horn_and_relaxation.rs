//! `HornClauses` against naive forward chaining, and `min_label_relaxation` against Bellman-Ford.

use cfglib::{
    EdgeId, EdgeRecord, Graph, HornClauses, NodeId, TraversalDirection, min_label_relaxation,
};

use crate::Lcg;

/// Repeat "derive every head whose body is derived" until nothing changes.
fn forward_chaining(fact_count: usize, clauses: &[(usize, Vec<usize>)]) -> Vec<bool> {
    let mut derived = vec![false; fact_count];
    loop {
        let mut changed = false;
        for (head, body) in clauses {
            if !derived[*head] && body.iter().all(|&fact| derived[fact]) {
                derived[*head] = true;
                changed = true;
            }
        }
        if !changed {
            return derived;
        }
    }
}

fn clause_set(fact_count: usize, clauses: &[(usize, Vec<usize>)]) -> HornClauses {
    let mut set = HornClauses::new(fact_count);
    for (head, body) in clauses {
        set.add_clause(*head, body);
    }
    set
}

/// Random clauses: a quarter are axioms, and bodies may repeat a fact.
fn random_clauses(rng: &mut Lcg, fact_count: usize, count: usize) -> Vec<(usize, Vec<usize>)> {
    (0..count)
        .map(|_| {
            let head = rng.below(fact_count);
            let length = rng.below(4);
            let body = (0..length).map(|_| rng.below(fact_count)).collect();
            (head, body)
        })
        .collect()
}

#[test]
fn derivability_matches_naive_forward_chaining_on_random_clause_sets() {
    let mut mixed = 0;
    for seed in 1..=40 {
        let mut rng = Lcg(seed);
        let facts = 30;
        let clauses = random_clauses(&mut rng, facts, 45);
        let derived = clause_set(facts, &clauses).derivable();
        assert_eq!(derived, forward_chaining(facts, &clauses), "seed {seed}");
        let count = derived.iter().filter(|&&yes| yes).count();
        if count > 0 && count < facts {
            mixed += 1;
        }
    }
    assert!(
        mixed > 15,
        "only {mixed} of 40 clause sets derive some facts but not all"
    );
}

#[test]
fn the_answer_does_not_depend_on_the_order_clauses_were_added() {
    for seed in 1..=10 {
        let mut rng = Lcg(seed + 100);
        let clauses = random_clauses(&mut rng, 25, 40);
        let expected = clause_set(25, &clauses).derivable();

        let mut shuffled = clauses.clone();
        for index in (1..shuffled.len()).rev() {
            shuffled.swap(index, rng.below(index + 1));
        }
        assert_eq!(
            clause_set(25, &shuffled).derivable(),
            expected,
            "seed {seed}"
        );
        let reversed: Vec<_> = clauses.iter().rev().cloned().collect();
        assert_eq!(
            clause_set(25, &reversed).derivable(),
            expected,
            "seed {seed}, reversed"
        );
    }
}

#[test]
fn adding_a_clause_only_grows_the_derivable_set_and_solving_changes_nothing() {
    let mut rng = Lcg(7);
    let clauses = random_clauses(&mut rng, 30, 40);
    let mut set = clause_set(30, &clauses);
    let before = set.derivable();
    assert_eq!(
        set.derivable(),
        before,
        "solving does not consume or mutate the set"
    );

    for round in 0..10 {
        set.add_clause(rng.below(30), &[rng.below(30), rng.below(30)]);
        let after = set.derivable();
        for (fact, (&was, &is)) in before.iter().zip(&after).enumerate() {
            assert!(
                !was || is,
                "round {round}: fact {fact} stopped being derivable"
            );
        }
    }
}

#[test]
fn grammar_nullability_of_a_small_language_is_derived_by_hand() {
    // Program -> Stmts; Stmts -> ε | Stmt Stmts; Stmt -> Opt Terminal; Opt -> ε; Block -> Stmts Opt.
    let (program, stmts, stmt, opt, terminal, block) = (0, 1, 2, 3, 4, 5);
    let mut grammar = HornClauses::new(6);
    grammar.add_clause(program, &[stmts]);
    grammar.add_clause(stmts, &[]);
    grammar.add_clause(stmts, &[stmt, stmts]);
    grammar.add_clause(stmt, &[opt, terminal]);
    grammar.add_clause(opt, &[]);
    grammar.add_clause(block, &[stmts, opt]);

    assert_eq!(grammar.derivable(), [true, true, false, true, false, true]);
}

#[test]
fn a_cycle_of_callers_without_a_grounded_member_is_never_derived_dead() {
    // main(0) is an entry, a(1) is called by main, b(2) and c(3) only call each other,
    // d(4) has no callers, e(5) is called only by d: a function is dead when all callers are.
    let mut dead = HornClauses::new(6);
    dead.add_clause(1, &[0]);
    dead.add_clause(2, &[3]);
    dead.add_clause(3, &[2]);
    dead.add_clause(4, &[]);
    dead.add_clause(5, &[4]);

    assert_eq!(
        dead.derivable(),
        [false, false, false, false, true, true],
        "the least fixpoint has no ground for b and c"
    );
    dead.add_clause(2, &[]);
    assert_eq!(
        dead.derivable(),
        [false, false, true, true, true, true],
        "grounding one derives both"
    );
}

#[test]
fn a_repeated_body_fact_counts_once_per_occurrence_and_one_derivation_discharges_all() {
    let mut set = HornClauses::new(3);
    set.add_clause(0, &[1, 1, 2]);
    assert_eq!(set.derivable(), [false, false, false]);
    set.add_clause(1, &[]);
    assert_eq!(
        set.derivable(),
        [false, true, false],
        "fact 2 is still missing"
    );
    set.add_clause(2, &[]);
    assert_eq!(set.derivable(), [true, true, true]);
}

#[test]
fn the_empty_fact_space_and_clones_are_ordinary() {
    let empty = HornClauses::new(0);
    assert_eq!(empty.derivable().len(), 0);

    let mut set = HornClauses::new(2);
    set.add_clause(0, &[]);
    let copy = set.clone();
    assert_eq!(copy, set);
    set.add_clause(1, &[0]);
    assert_ne!(copy, set);
    assert_eq!(
        copy.derivable(),
        [true, false],
        "the clone does not see the later clause"
    );
    assert_eq!(set.derivable(), [true, true]);
}

#[test]
#[should_panic(expected = "head fact is out of range")]
fn a_head_outside_the_fact_space_panics() {
    HornClauses::new(3).add_clause(3, &[]);
}

#[test]
#[should_panic(expected = "body fact is out of range")]
fn a_body_fact_outside_the_fact_space_panics() {
    HornClauses::new(3).add_clause(0, &[1, 3]);
}

/// An edge payload: a cost, or an edge the analysis refuses to cross.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Step {
    Cost(u32),
    Closed,
}

/// Fewest hops first, then least total cost. Deliberately neither `Clone` nor `Copy`.
#[derive(Debug, PartialEq, Eq, PartialOrd, Ord)]
struct Effort {
    hops: u32,
    units: u32,
}

fn advance(label: &Effort, step: Step) -> Option<Effort> {
    match step {
        Step::Cost(units) => Some(Effort {
            hops: label.hops + 1,
            units: label.units + units,
        }),
        Step::Closed => None,
    }
}

/// Relax every edge until nothing improves.
fn bellman_ford(
    nodes: usize,
    edges: &[(usize, usize, Step)],
    seeds: &[(usize, u32, u32)],
    forward: bool,
) -> Vec<Option<Effort>> {
    let mut labels: Vec<Option<Effort>> = (0..nodes).map(|_| None).collect();
    for &(node, hops, units) in seeds {
        let candidate = Effort { hops, units };
        if labels[node].as_ref().is_none_or(|known| candidate < *known) {
            labels[node] = Some(candidate);
        }
    }
    loop {
        let mut changed = false;
        for (source, target, step) in edges {
            let (near, far) = if forward {
                (*source, *target)
            } else {
                (*target, *source)
            };
            let Some(candidate) = labels[near]
                .as_ref()
                .and_then(|label| advance(label, *step))
            else {
                continue;
            };
            if labels[far].as_ref().is_none_or(|known| candidate < *known) {
                labels[far] = Some(candidate);
                changed = true;
            }
        }
        if !changed {
            return labels;
        }
    }
}

/// A model edge: source, target and payload.
type ModelEdge = (usize, usize, Step);

/// A weighted graph with removed nodes and edges, with the live edges in the model's terms.
fn weighted_scenario(seed: u64) -> (Graph<(), Step>, Vec<ModelEdge>) {
    let mut rng = Lcg(seed);
    let nodes = 24;
    let mut graph = Graph::new();
    let ids: Vec<NodeId> = (0..nodes).map(|_| graph.add_node(())).collect();
    let mut model: Vec<Option<ModelEdge>> = Vec::new();
    for _ in 0..70 {
        let (source, target) = (rng.below(nodes), rng.below(nodes));
        let step = if rng.below(6) == 0 {
            Step::Closed
        } else {
            Step::Cost(u32::try_from(rng.below(9)).expect("small") + 1)
        };
        graph.add_edge(ids[source], ids[target], step);
        model.push(Some((source, target, step)));
    }
    for (edge, entry) in model.iter_mut().enumerate() {
        if rng.below(12) == 0 {
            graph.remove_edge(EdgeId::from_index(edge));
            *entry = None;
        }
    }
    for _ in 0..2 {
        let node = rng.below(nodes);
        graph.remove_node(ids[node]);
        for entry in &mut model {
            if entry.is_some_and(|(source, target, _)| source == node || target == node) {
                *entry = None;
            }
        }
    }
    (graph, model.into_iter().flatten().collect())
}

fn relaxed(
    graph: &Graph<(), Step>,
    seeds: &[(usize, u32, u32)],
    direction: TraversalDirection,
) -> Vec<Option<Effort>> {
    min_label_relaxation(
        graph,
        seeds
            .iter()
            .map(|&(node, hops, units)| (NodeId::from_index(node), Effort { hops, units })),
        direction,
        |edge: &EdgeRecord<Step>, label: &Effort| advance(label, *edge.payload()),
    )
}

#[test]
fn relaxation_matches_bellman_ford_in_both_directions_on_graphs_with_removed_entities() {
    for seed in 1..=12 {
        let (graph, edges) = weighted_scenario(seed);
        // Seed live nodes only; the scenario removed two of 24.
        let live: Vec<usize> = graph.node_ids().map(NodeId::index).collect();
        let seeds = [
            (live[0], 0, 0),
            (live[5], 0, 3),
            (live[9], 1, 0),
            (live[5], 2, 1),
        ];
        for (direction, forward) in [
            (TraversalDirection::Outgoing, true),
            (TraversalDirection::Incoming, false),
        ] {
            let actual = relaxed(&graph, &seeds, direction);
            let expected = bellman_ford(24, &edges, &seeds, forward);
            assert_eq!(actual, expected, "seed {seed}, {direction:?}");
            assert_eq!(actual.len(), graph.node_bound(), "one label per node slot");
        }
    }
}

#[test]
fn the_scenarios_have_improvements_blocked_edges_and_unreachable_nodes() {
    let (graph, edges) = weighted_scenario(3);
    assert!(edges.iter().any(|(_, _, step)| *step == Step::Closed));
    let live: Vec<usize> = graph.node_ids().map(NodeId::index).collect();
    let labels = relaxed(&graph, &[(live[0], 0, 0)], TraversalDirection::Outgoing);
    let reached = labels.iter().flatten().count();
    assert!(
        reached > 3 && reached < graph.node_count(),
        "{reached} of {} reached",
        graph.node_count()
    );
    assert!(labels.iter().flatten().any(|label| label.hops >= 3));
}

fn diamond_with_shortcut() -> (Graph<&'static str, u32>, [NodeId; 4]) {
    // s -> a (1), s -> b (5), a -> b (1), b -> t (1), a -> t (10)
    let mut graph = Graph::new();
    let s = graph.add_node("s");
    let a = graph.add_node("a");
    let b = graph.add_node("b");
    let t = graph.add_node("t");
    graph.add_edge(s, a, 1);
    graph.add_edge(s, b, 5);
    graph.add_edge(a, b, 1);
    graph.add_edge(b, t, 1);
    graph.add_edge(a, t, 10);
    (graph, [s, a, b, t])
}

#[test]
fn a_smaller_label_reexpands_a_node_visited_earlier_with_a_larger_one() {
    let (graph, [s, a, b, t]) = diamond_with_shortcut();
    let distances = min_label_relaxation(
        &graph,
        [(s, 0_u32)],
        TraversalDirection::Outgoing,
        |edge, label| Some(label + edge.payload()),
    );
    assert_eq!(distances[s.index()], Some(0));
    assert_eq!(distances[a.index()], Some(1));
    assert_eq!(
        distances[b.index()],
        Some(2),
        "through a, not the direct edge of cost 5"
    );
    assert_eq!(
        distances[t.index()],
        Some(3),
        "improved after b was reached more cheaply"
    );
}

#[test]
fn incoming_relaxation_hands_the_stored_edge_to_the_transfer_and_expands_from_the_target() {
    let (graph, [s, a, b, t]) = diamond_with_shortcut();
    let mut first_call = None;
    let distances = min_label_relaxation(
        &graph,
        [(t, 0_u32)],
        TraversalDirection::Incoming,
        |edge, label| {
            first_call.get_or_insert((edge.source(), edge.target()));
            Some(label + edge.payload())
        },
    );
    assert_eq!(distances[t.index()], Some(0));
    assert_eq!(distances[b.index()], Some(1));
    assert_eq!(distances[a.index()], Some(2), "a -> b -> t beats a -> t");
    assert_eq!(distances[s.index()], Some(3));
    let (source, target) = first_call.expect("the transfer ran");
    assert_eq!(
        target, t,
        "the edge keeps its stored direction while the walk goes backwards"
    );
    assert!(source == b || source == a);
}

#[test]
fn parallel_edges_competing_seeds_and_blocked_edges_resolve_to_the_minimum() {
    let mut graph = Graph::new();
    let a = graph.add_node(());
    let b = graph.add_node(());
    let c = graph.add_node(());
    graph.add_edge(a, b, Some(7_u32));
    graph.add_edge(a, b, Some(2));
    graph.add_edge(a, a, Some(0));
    graph.add_edge(b, c, None);

    let labels = min_label_relaxation(
        &graph,
        [(a, 5_u32), (a, 1), (a, 9)],
        TraversalDirection::Outgoing,
        |edge, label| edge.payload().map(|cost| label + cost),
    );
    assert_eq!(
        labels,
        [Some(1), Some(3), None],
        "the smallest seed, the cheapest parallel edge, a blocked edge"
    );
}

#[test]
fn a_node_no_seed_reaches_has_no_label_and_two_seeds_label_two_regions() {
    let mut graph = Graph::new();
    let nodes: Vec<NodeId> = (0..6).map(|_| graph.add_node(())).collect();
    graph.add_edge(nodes[0], nodes[1], 1_u32);
    graph.add_edge(nodes[1], nodes[2], 1);
    graph.add_edge(nodes[3], nodes[4], 4);
    let labels = min_label_relaxation(
        &graph,
        [(nodes[0], 0_u32), (nodes[3], 10)],
        TraversalDirection::Outgoing,
        |edge, label| Some(label + edge.payload()),
    );
    assert_eq!(
        labels,
        [Some(0), Some(1), Some(2), Some(10), Some(14), None]
    );

    let none = min_label_relaxation(
        &graph,
        Vec::<(NodeId, u32)>::new(),
        TraversalDirection::Outgoing,
        |_, label| Some(*label),
    );
    assert!(none.iter().all(Option::is_none));
    assert_eq!(none.len(), 6);
}

#[test]
#[should_panic(expected = "seed node is out of range")]
fn a_seed_outside_the_graph_panics() {
    let graph = Graph::<(), ()>::new();
    let _ = min_label_relaxation(
        &graph,
        [(NodeId::from_index(0), 0_u32)],
        TraversalDirection::Outgoing,
        |_, label| Some(*label),
    );
}
