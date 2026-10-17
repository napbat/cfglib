//! Interference graphs and graph coloring: the register-allocation core a
//! back end builds on cfglib's liveness.

use std::collections::{BTreeMap, BTreeSet};

use cfglib::{
    BasicBlock, Cfg, ColorAssignment, Graph, GraphView, InstrInfo, Liveness, NodeId, color_graph,
    interference_graph,
};

use crate::oracles::{brute_force_live_in, brute_force_live_out};
use crate::programs::random_program;
use crate::toy::{Builder, INPUTS, Inst, Payload, Reg, Route, run};

const SEEDS: u64 = 80;

/// The number of colors an assignment reports having used. The one place the
/// result's count field is named.
fn color_count<N>(colors: &ColorAssignment<N>) -> usize {
    colors.num_colors
}

/// Pairs of registers a textbook interference graph connects: two registers
/// live at the same point, or a register written while another is live after
/// the write (even when the written value is itself never read).
fn brute_force_pairs<E>(cfg: &Cfg<Inst, E>) -> BTreeSet<(Reg, Reg)> {
    fn connect_all(live: &BTreeSet<Reg>, pairs: &mut BTreeSet<(Reg, Reg)>) {
        for &left in live {
            for &right in live.range(left + 1..) {
                pairs.insert((left, right));
            }
        }
    }
    let live_out = brute_force_live_out(cfg);
    let mut pairs = BTreeSet::new();
    for block in cfg.block_ids() {
        let mut live = live_out[&block].clone();
        connect_all(&live, &mut pairs);
        for inst in cfg.block(block).instructions().iter().rev() {
            for &def in inst.defs() {
                for &other in &live {
                    if other != def {
                        pairs.insert((def.min(other), def.max(other)));
                    }
                }
            }
            for def in inst.defs() {
                live.remove(def);
            }
            live.extend(inst.uses().iter().copied());
            connect_all(&live, &mut pairs);
        }
    }
    pairs
}

fn mentioned_registers<E>(cfg: &Cfg<Inst, E>) -> BTreeSet<Reg> {
    cfg.blocks()
        .flat_map(BasicBlock::instructions)
        .flat_map(|inst| inst.defs().iter().chain(inst.uses()))
        .copied()
        .collect()
}

fn registers_and_pairs(graph: &Graph<Reg, ()>) -> (BTreeSet<Reg>, Vec<(Reg, Reg)>) {
    let registers = graph.nodes().copied().collect();
    let mut pairs: Vec<(Reg, Reg)> = graph
        .edges()
        .map(|edge| {
            let (left, right) = (*graph.node(edge.source()), *graph.node(edge.target()));
            (left.min(right), left.max(right))
        })
        .collect();
    pairs.sort_unstable();
    (registers, pairs)
}

fn interference_case<E: Payload>(seed: u64) {
    let cfg = random_program::<E>(seed, true);
    let graph = interference_graph(&cfg, &Liveness::compute(&cfg));
    let (registers, pairs) = registers_and_pairs(&graph);
    let context = format!("seed {seed}");

    assert_eq!(
        registers,
        mentioned_registers(&cfg),
        "{context}: one node per register"
    );
    let distinct: BTreeSet<_> = pairs.iter().copied().collect();
    assert_eq!(
        distinct.len(),
        pairs.len(),
        "{context}: each relation is stored once"
    );
    assert!(
        pairs.iter().all(|(left, right)| left != right),
        "{context}: no self-conflicts"
    );
    assert_eq!(distinct, brute_force_pairs(&cfg), "{context}");
}

#[test]
fn interference_connects_exactly_the_registers_that_are_live_together() {
    for seed in 0..SEEDS {
        interference_case::<()>(seed);
        interference_case::<Route>(seed);
    }
}

#[test]
fn a_chain_of_two_operands_interferes_only_where_both_are_live() {
    let mut builder = Builder::<()>::new();
    let entry = builder.entry();
    for inst in [
        Inst::constant(10, 1),
        Inst::constant(11, 2),
        Inst::add(12, 10, 11),
        Inst::print(12),
    ] {
        builder.push(entry, inst);
    }
    let cfg = builder.finish();

    let graph = interference_graph(&cfg, &Liveness::compute(&cfg));
    let (registers, pairs) = registers_and_pairs(&graph);

    assert_eq!(registers, BTreeSet::from([10, 11, 12]));
    assert_eq!(
        pairs,
        [(10, 11)],
        "the sum may share a register with either operand"
    );
    let colors = color_graph(&graph);
    assert_eq!(color_count(&colors), 2);
}

fn assert_proper<G: GraphView>(graph: &G, colors: &ColorAssignment<G::NodeId>, context: &str) {
    for node in graph.node_ids() {
        assert!(
            colors.assignment.contains_key(&node),
            "{context}: {node:?} is uncolored"
        );
        for neighbor in graph.successors(node).chain(graph.predecessors(node)) {
            if neighbor != node {
                assert_ne!(
                    colors.assignment[&node], colors.assignment[&neighbor],
                    "{context}: adjacent nodes share a color"
                );
            }
        }
    }
    assert_eq!(
        colors.assignment.len(),
        graph.node_ids().count(),
        "{context}"
    );
    let used: BTreeSet<usize> = colors.assignment.values().copied().collect();
    assert_eq!(
        color_count(colors),
        used.len(),
        "{context}: the count is the colors used"
    );
    assert_eq!(
        used,
        (0..used.len()).collect::<BTreeSet<_>>(),
        "{context}: colors are dense from 0"
    );
    let max_degree = graph
        .node_ids()
        .map(|node| {
            graph
                .successors(node)
                .chain(graph.predecessors(node))
                .filter(|&other| other != node)
                .collect::<BTreeSet<_>>()
                .len()
        })
        .max()
        .unwrap_or(0);
    assert!(
        color_count(colors) <= max_degree + 1,
        "{context}: greedy needs at most degree + 1 colors"
    );
}

#[test]
fn a_coloring_of_an_interference_graph_gives_neighbors_different_colors() {
    for seed in 0..SEEDS {
        let cfg = random_program::<()>(seed, true);
        let graph = interference_graph(&cfg, &Liveness::compute(&cfg));
        assert_proper(&graph, &color_graph(&graph), &format!("seed {seed}"));
    }
}

fn rename_by_color(cfg: &Cfg<Inst>, colors: &BTreeMap<Reg, Reg>) -> Cfg<Inst> {
    let mut renamed = cfg.clone();
    for block in cfg.block_ids() {
        let instructions: Vec<Inst> = cfg
            .block(block)
            .instructions()
            .iter()
            .map(|inst| inst.renamed(|register| colors[&register]))
            .collect();
        *renamed.block_mut(block).instructions_mut() = instructions;
    }
    renamed
}

#[test]
fn allocating_one_register_per_color_preserves_the_program_output() {
    let mut shared = 0;
    for seed in 0..SEEDS {
        let cfg = random_program::<()>(seed, true);
        let graph = interference_graph(&cfg, &Liveness::compute(&cfg));
        let coloring = color_graph(&graph);
        let colors: BTreeMap<Reg, Reg> = graph
            .node_ids()
            .map(|node| {
                let color = u8::try_from(coloring.assignment[&node]).unwrap();
                (*graph.node(node), color)
            })
            .collect();
        shared += colors.len() - color_count(&coloring);
        let allocated = rename_by_color(&cfg, &colors);
        let read_before_written = &brute_force_live_in(&cfg)[&cfg.entry()];

        for inputs in INPUTS {
            let moved: Vec<(Reg, i64)> = inputs
                .iter()
                .filter(|(register, _)| read_before_written.contains(register))
                .map(|&(register, value)| (colors[&register], value))
                .collect();
            let expected = run(&cfg, &inputs, &[]).unwrap();
            assert_eq!(
                run(&allocated, &moved, &[]),
                Ok(expected),
                "seed {seed}: registers sharing a color must never be live together; {inputs:?}"
            );
        }
    }
    assert!(
        shared > 100,
        "the allocations never shared a register: {shared}"
    );
}

fn graph_of(nodes: usize, edges: &[(usize, usize)]) -> (Graph<usize, ()>, Vec<NodeId>) {
    let mut graph = Graph::new();
    let ids: Vec<NodeId> = (0..nodes).map(|payload| graph.add_node(payload)).collect();
    for &(source, target) in edges {
        graph.add_edge(ids[source], ids[target], ());
    }
    (graph, ids)
}

#[test]
fn colorings_of_known_graphs_use_the_forced_number_of_colors() {
    let complete: Vec<(usize, usize)> = (0..5)
        .flat_map(|left| (left + 1..5).map(move |right| (left, right)))
        .collect();
    let (clique, _) = graph_of(5, &complete);
    assert_eq!(
        color_count(&color_graph(&clique)),
        5,
        "K5 needs five colors"
    );

    let (independent, _) = graph_of(4, &[]);
    assert_eq!(color_count(&color_graph(&independent)), 1);

    let (empty, _) = graph_of(0, &[]);
    let nothing = color_graph(&empty);
    assert_eq!((color_count(&nothing), nothing.assignment.len()), (0, 0));

    let odd_cycle: Vec<(usize, usize)> = (0..5).map(|node| (node, (node + 1) % 5)).collect();
    let (pentagon, _) = graph_of(5, &odd_cycle);
    assert_eq!(
        color_count(&color_graph(&pentagon)),
        3,
        "an odd cycle needs exactly three"
    );
    assert_proper(&pentagon, &color_graph(&pentagon), "pentagon");
}

#[test]
fn edge_direction_parallel_edges_and_self_loops_do_not_change_the_conflicts() {
    let (graph, ids) = graph_of(3, &[(0, 1), (1, 0), (0, 1), (2, 2), (1, 2)]);
    let colors = color_graph(&graph);

    assert_ne!(colors.assignment[&ids[0]], colors.assignment[&ids[1]]);
    assert_ne!(colors.assignment[&ids[1]], colors.assignment[&ids[2]]);
    assert_eq!(
        color_count(&colors),
        2,
        "node 2 may reuse node 0's color despite its self-loop"
    );
    assert_proper(&graph, &colors, "mixed edges");
}

#[test]
fn removed_nodes_are_not_colored_and_their_edges_do_not_constrain() {
    let (mut graph, ids) = graph_of(4, &[(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)]);
    graph.remove_node(ids[2]);
    let colors = color_graph(&graph);

    assert!(!colors.assignment.contains_key(&ids[2]));
    assert_eq!(colors.assignment.len(), 3);
    assert_proper(&graph, &colors, "after node removal");
    assert_eq!(color_count(&colors), 2, "0-1 and 3-0 remain: a path");
}

#[test]
fn a_cfg_is_colorable_as_a_graph_view_of_its_own() {
    for seed in 0..20 {
        let cfg = random_program::<Route>(seed, true);
        assert_proper(&cfg, &color_graph(&cfg), &format!("seed {seed}"));
    }
}
