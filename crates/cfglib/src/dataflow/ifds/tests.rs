extern crate alloc;
use alloc::collections::BTreeSet;
use alloc::vec;
use alloc::vec::Vec;

use super::{CfgSupergraph, IfdsConfig, IfdsProblem, IfdsResults, Supergraph, solve_ifds};
use crate::cfg::Cfg;
use crate::dataflow::fixpoint::{SolveConfig, SolveError};
use crate::dataflow::program_point::ProgramPoint;
use crate::edge::EdgeKind;
use crate::flow::{Flow, FlowControl, FlowEffect};
use crate::test_util::rng::Rng;

/// A supergraph written out node by node: an exit is a node without
/// successors.
#[derive(Debug, Default)]
struct Toy {
    proc_of: Vec<usize>,
    starts: Vec<usize>,
    successors: Vec<Vec<usize>>,
    callees: Vec<Vec<usize>>,
}

impl Toy {
    /// Add a procedure of `nodes` nodes, the first its start.
    fn add_proc(&mut self, nodes: usize) -> Vec<usize> {
        let first = self.proc_of.len();
        let proc = self.starts.len();
        self.starts.push(first);
        for _ in 0..nodes {
            self.proc_of.push(proc);
            self.successors.push(Vec::new());
            self.callees.push(Vec::new());
        }
        (first..first + nodes).collect()
    }

    fn edge(&mut self, from: usize, to: usize) {
        self.successors[from].push(to);
    }

    fn call(&mut self, node: usize, callee: usize) {
        self.callees[node].push(callee);
    }
}

impl Supergraph for Toy {
    type Node = usize;
    type Proc = usize;

    fn node_bound(&self) -> usize {
        self.proc_of.len()
    }

    fn proc_bound(&self) -> usize {
        self.starts.len()
    }

    fn proc_of(&self, node: usize) -> usize {
        self.proc_of[node]
    }

    fn start(&self, proc: usize) -> usize {
        self.starts[proc]
    }

    fn is_exit(&self, node: usize) -> bool {
        self.successors[node].is_empty()
    }

    fn successors(&self, node: usize, out: &mut Vec<usize>) {
        out.extend_from_slice(&self.successors[node]);
    }

    fn callees(&self, node: usize, out: &mut Vec<usize>) {
        out.extend_from_slice(&self.callees[node]);
    }
}

/// One flow function application, as the problem sees it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Step {
    Normal {
        from: usize,
        to: usize,
    },
    Call {
        call: usize,
        callee: usize,
    },
    Return {
        call: usize,
        callee: usize,
        exit: usize,
        site: usize,
    },
    CallToReturn {
        call: usize,
        site: usize,
    },
}

/// A problem whose four flow functions are one closure over [`Step`].
struct Flows<F> {
    facts: usize,
    flow: F,
}

impl<F: Fn(Step, u32) -> Vec<u32>> IfdsProblem<Toy> for Flows<F> {
    fn fact_bound(&self) -> usize {
        self.facts
    }

    fn normal(&self, from: usize, to: usize, fact: u32, out: &mut Vec<u32>) {
        out.extend((self.flow)(Step::Normal { from, to }, fact));
    }

    fn call(&self, call: usize, callee: usize, fact: u32, out: &mut Vec<u32>) {
        out.extend((self.flow)(Step::Call { call, callee }, fact));
    }

    fn ret(
        &self,
        call: usize,
        callee: usize,
        exit: usize,
        site: usize,
        fact: u32,
        out: &mut Vec<u32>,
    ) {
        let step = Step::Return {
            call,
            callee,
            exit,
            site,
        };
        out.extend((self.flow)(step, fact));
    }

    fn call_to_return(&self, call: usize, site: usize, fact: u32, out: &mut Vec<u32>) {
        out.extend((self.flow)(Step::CallToReturn { call, site }, fact));
    }
}

/// Reachability over the exploded supergraph, whose states are a stack of
/// call nodes, a node and a fact.
///
/// With `matched`, a return pops the call that entered the procedure, so
/// only realizable paths count; it terminates when the call graph is
/// acyclic. Without, every state has an empty stack and an exit returns to
/// every call of its procedure, which is what a context-insensitive solve
/// computes. A seed holds wherever its procedure starts with the zero fact,
/// as well as on its own.
fn exploded_reachability<F: Fn(Step, u32) -> Vec<u32>>(
    graph: &Toy,
    problem: &Flows<F>,
    seeds: &[(usize, u32)],
    matched: bool,
) -> Vec<BTreeSet<u32>> {
    let flows = |step, fact| {
        let mut out = (problem.flow)(step, fact);
        if fact == 0 {
            out.push(0);
        }
        out
    };
    let mut holds = vec![BTreeSet::new(); graph.node_bound()];
    let mut seen = BTreeSet::new();
    let mut queue: Vec<(Vec<usize>, usize, u32)> = seeds
        .iter()
        .map(|&(node, fact)| (Vec::new(), node, fact))
        .collect();
    while let Some(state) = queue.pop() {
        if !seen.insert(state.clone()) {
            continue;
        }
        let (stack, node, fact) = state;
        holds[node].insert(fact);
        let proc = graph.proc_of[node];
        if node == graph.starts[proc] && fact == 0 {
            for &(seed, seed_fact) in seeds {
                if graph.proc_of[seed] == proc {
                    queue.push((stack.clone(), seed, seed_fact));
                }
            }
        }

        if graph.is_exit(node) {
            let returns: Vec<(Vec<usize>, usize)> = if matched {
                stack
                    .split_last()
                    .map(|(&call, rest)| (rest.to_vec(), call))
                    .into_iter()
                    .collect()
            } else {
                (0..graph.node_bound())
                    .filter(|&call| graph.callees[call].contains(&proc))
                    .map(|call| (Vec::new(), call))
                    .collect()
            };
            for (rest, call) in returns {
                for &site in &graph.successors[call] {
                    let step = Step::Return {
                        call,
                        callee: proc,
                        exit: node,
                        site,
                    };
                    for out in flows(step, fact) {
                        queue.push((rest.clone(), site, out));
                    }
                }
            }
        }
        if graph.callees[node].is_empty() {
            for &to in &graph.successors[node] {
                for out in flows(Step::Normal { from: node, to }, fact) {
                    queue.push((stack.clone(), to, out));
                }
            }
        } else {
            for &callee in &graph.callees[node] {
                let mut deeper = Vec::new();
                if matched {
                    deeper.clone_from(&stack);
                    deeper.push(node);
                }
                for out in flows(Step::Call { call: node, callee }, fact) {
                    queue.push((deeper.clone(), graph.starts[callee], out));
                }
            }
            for &site in &graph.successors[node] {
                for out in flows(Step::CallToReturn { call: node, site }, fact) {
                    queue.push((stack.clone(), site, out));
                }
            }
        }
    }
    holds
}

fn facts_of(results: &IfdsResults<Toy>, node: usize) -> Vec<u32> {
    results.facts_at(node).collect()
}

fn solve(graph: &Toy, problem: &impl IfdsProblem<Toy>, seeds: &[(usize, u32)]) -> IfdsResults<Toy> {
    solve_ifds(graph, problem, seeds, IfdsConfig::new()).expect("an unbounded solve")
}

#[test]
fn the_possibly_uninitialized_variables_of_reps_horwitz_and_sagiv() {
    // The running example of the POPL 1995 paper, with facts Λ, x, g, a:
    //
    //   declare g                      procedure P(value a)
    //   procedure main                   if a > 0 then
    //     declare x                        read(g)
    //     read(x)                          a := a - g
    //     call P(x)                        call P(a)
    //   end                                print(a, g)
    //                                    fi
    //                                  end
    let (lambda, x, g, a) = (0_u32, 1_u32, 2_u32, 3_u32);
    let mut graph = Toy::default();
    let main = graph.add_proc(5); // s_main, read(x), call P, return site, e_main
    let p = graph.add_proc(8); // s_p, if, read(g), a := a - g, call P, return site, print, e_p
    for pair in main.windows(2) {
        graph.edge(pair[0], pair[1]);
    }
    let [s_p, test, read_g, assign, call, site, print, e_p] = p.try_into().expect("eight nodes");
    graph.edge(s_p, test);
    graph.edge(test, read_g);
    graph.edge(test, e_p);
    graph.edge(read_g, assign);
    graph.edge(assign, call);
    graph.edge(call, site);
    graph.edge(site, print);
    graph.edge(print, e_p);
    graph.call(main[2], 1);
    graph.call(call, 1);

    let flow = move |step: Step, fact: u32| -> Vec<u32> {
        match step {
            // Every variable is uninitialized when main starts.
            Step::Normal { from: 0, .. } if fact == lambda => vec![x, g],
            Step::Normal { from: 1, .. } if fact == x => vec![],
            Step::Normal { from, .. } if from == read_g && fact == g => vec![],
            Step::Normal { from, .. } if from == assign && fact == g => vec![g, a],
            // The actual parameter becomes the formal; the global passes.
            Step::Call { call: 2, .. } if fact == x => vec![a],
            Step::Call { .. } if fact == g || fact == a => vec![fact],
            // Back from P only the global returns; the caller's locals go
            // alongside the call.
            Step::Return { .. } if fact == g => vec![g],
            Step::CallToReturn { .. } if fact == x || fact == a => vec![fact],
            Step::Call { .. } | Step::Return { .. } | Step::CallToReturn { .. } => vec![],
            Step::Normal { .. } => (fact != lambda).then_some(fact).into_iter().collect(),
        }
    };
    let problem = Flows { facts: 4, flow };
    let seeds = [(main[0], lambda)];
    let results = solve(&graph, &problem, &seeds);

    let expected: [(usize, &[u32]); 13] = [
        (main[0], &[lambda]),
        (main[1], &[lambda, x, g]),
        (main[2], &[lambda, g]),
        (main[3], &[lambda, g]),
        (main[4], &[lambda, g]),
        (s_p, &[lambda, g]),
        (test, &[lambda, g]),
        (read_g, &[lambda, g]),
        (assign, &[lambda]),
        (call, &[lambda]),
        (site, &[lambda]),
        (print, &[lambda]),
        (e_p, &[lambda, g]),
    ];
    for (node, facts) in expected {
        assert_eq!(facts_of(&results, node), facts, "node {node}");
    }
    assert_eq!(results.summaries(1, g).collect::<Vec<_>>(), [(e_p, g)]);
    assert_eq!(
        results.summaries(1, lambda).collect::<Vec<_>>(),
        [(e_p, lambda)]
    );
    assert_eq!(results.summaries(1, x).count(), 0);

    // The point of the example: returning from P to the recursive call's
    // return site after main entered it is not realizable, and a
    // context-insensitive solve takes that path and reports g at print.
    let merged = exploded_reachability(&graph, &problem, &seeds, false);
    assert!(merged[print].contains(&g));
    assert!(!results.holds(print, g));
    for (node, reached) in merged.iter().enumerate() {
        let precise: BTreeSet<u32> = results.facts_at(node).collect();
        assert!(precise.is_subset(reached), "node {node}");
    }
}

#[test]
fn two_callers_of_one_identity_procedure_keep_their_own_facts() {
    // caller1 taints 1 and calls id; caller2 taints 2 and calls id. Nothing
    // goes alongside the calls, so a return site sees only what id returns.
    let mut graph = Toy::default();
    let first = graph.add_proc(3);
    let second = graph.add_proc(3);
    let id = graph.add_proc(1);
    for caller in [&first, &second] {
        graph.edge(caller[0], caller[1]);
        graph.edge(caller[1], caller[2]);
        graph.call(caller[1], 2);
    }
    let (start1, start2) = (first[0], second[0]);
    let flow = move |step: Step, fact: u32| -> Vec<u32> {
        match step {
            Step::Normal { from, .. } if fact == 0 && from == start1 => vec![1],
            Step::Normal { from, .. } if fact == 0 && from == start2 => vec![2],
            Step::CallToReturn { .. } => vec![],
            _ => (fact != 0).then_some(fact).into_iter().collect(),
        }
    };
    let problem = Flows { facts: 3, flow };
    let seeds = [(start1, 0), (start2, 0)];
    let results = solve(&graph, &problem, &seeds);
    assert_eq!(facts_of(&results, first[2]), [0, 1]);
    assert_eq!(facts_of(&results, second[2]), [0, 2]);
    assert_eq!(facts_of(&results, id[0]), [0, 1, 2]);
    assert_eq!(results.summaries(2, 1).collect::<Vec<_>>(), [(id[0], 1)]);

    // A context-insensitive solve merges what the two callers pass in and
    // hands both facts back to both return sites.
    let merged = exploded_reachability(&graph, &problem, &seeds, false);
    assert_eq!(merged[first[2]], BTreeSet::from([0, 1, 2]));
    assert_eq!(merged[second[2]], BTreeSet::from([0, 1, 2]));
}

/// A deterministic random relation for each flow function: fact `f` is in
/// the image of `fact` under `step` when a hash of the three says so.
fn hashed_flows(seed: u64, facts: usize) -> impl Fn(Step, u32) -> Vec<u32> {
    move |step, fact| {
        let key: [u64; 5] = match step {
            Step::Normal { from, to } => [1, from as u64, to as u64, 0, 0],
            Step::Call { call, callee } => [2, call as u64, callee as u64, 0, 0],
            Step::Return {
                call,
                callee,
                exit,
                site,
            } => [3, call as u64, callee as u64, exit as u64, site as u64],
            Step::CallToReturn { call, site } => [4, call as u64, site as u64, 0, 0],
        };
        (1..facts)
            .map(|candidate| u32::try_from(candidate).expect("few facts"))
            .filter(|&candidate| {
                let mut state = seed ^ (u64::from(fact) << 32) ^ u64::from(candidate);
                for part in key {
                    state = (state ^ part).wrapping_mul(0x9E37_79B9_7F4A_7C15);
                    state ^= state >> 29;
                }
                state.is_multiple_of(3)
            })
            .collect()
    }
}

/// Two to five procedures of up to four nodes, calls only to later
/// procedures, intraprocedural loops and repeated edges allowed.
fn random_program(rng: &mut Rng) -> Toy {
    let procs = 2 + rng.below(4);
    let mut graph = Toy::default();
    let nodes: Vec<Vec<usize>> = (0..procs)
        .map(|_| graph.add_proc(1 + rng.below(4)))
        .collect();
    for (proc, members) in nodes.iter().enumerate() {
        for &node in members {
            for _ in 0..rng.below(3) {
                graph.edge(node, members[rng.below(members.len())]);
            }
            if !graph.successors[node].is_empty() && proc + 1 < procs && rng.chance(50) {
                for _ in 0..=rng.below(2) {
                    graph.call(node, proc + 1 + rng.below(procs - proc - 1));
                }
            }
        }
    }
    graph
}

#[test]
fn the_tabulation_equals_realizable_path_reachability_on_random_programs() {
    let mut sensitive = 0;
    for seed in 0..400 {
        let mut rng = Rng::new(seed);
        let graph = random_program(&mut rng);
        let facts = 1 + rng.below(4);
        let problem = Flows {
            facts,
            flow: hashed_flows(seed, facts),
        };
        let mut seeds = vec![(0, 0)];
        if rng.chance(30) {
            let fact = u32::try_from(rng.below(facts)).expect("few facts");
            seeds.push((rng.below(graph.node_bound()), fact));
        }

        let results = solve(&graph, &problem, &seeds);
        let oracle = exploded_reachability(&graph, &problem, &seeds, true);
        for (node, expected) in oracle.iter().enumerate() {
            let found: BTreeSet<u32> = results.facts_at(node).collect();
            assert_eq!(&found, expected, "seed {seed}, node {node}");
        }
        if exploded_reachability(&graph, &problem, &seeds, false) != oracle {
            sensitive += 1;
        }

        let collected = solve_ifds(
            &graph,
            &problem,
            &seeds,
            IfdsConfig::new().with_path_edge_collection(true),
        )
        .expect("an unbounded solve");
        assert_same_answer(&graph, facts, &results, &collected);
    }
    assert!(
        sensitive > 30,
        "only {sensitive} programs where context matters"
    );
}

fn assert_same_answer(
    graph: &Toy,
    facts: usize,
    kept: &IfdsResults<Toy>,
    collected: &IfdsResults<Toy>,
) {
    assert_eq!(kept.steps(), collected.steps());
    for node in 0..graph.node_bound() {
        assert_eq!(
            facts_of(kept, node),
            facts_of(collected, node),
            "node {node}"
        );
    }
    for proc in 0..graph.proc_bound() {
        for fact in 0..u32::try_from(facts).expect("few facts") {
            assert!(
                kept.summaries(proc, fact)
                    .eq(collected.summaries(proc, fact)),
                "proc {proc}, fact {fact}"
            );
        }
    }
}

#[test]
fn collecting_path_edges_lowers_the_peak_and_changes_no_answer() {
    // main calls forty leaf procedures in a row; each is a chain of ten
    // nodes that carries three facts. A finished leaf's path edges are dead.
    const LEAVES: usize = 40;
    let mut graph = Toy::default();
    let main = graph.add_proc(LEAVES + 1);
    for pair in main.windows(2) {
        graph.edge(pair[0], pair[1]);
    }
    for (leaf, &call) in main.iter().take(LEAVES).enumerate() {
        let chain = graph.add_proc(10);
        for pair in chain.windows(2) {
            graph.edge(pair[0], pair[1]);
        }
        graph.call(call, leaf + 1);
    }
    let flow = |step: Step, fact: u32| -> Vec<u32> {
        match step {
            Step::Call { .. } if fact == 0 => vec![1, 2, 3],
            Step::Return { .. } => vec![],
            _ => (fact != 0).then_some(fact).into_iter().collect(),
        }
    };
    let problem = Flows { facts: 4, flow };
    let seeds = [(main[0], 0)];
    let kept = solve(&graph, &problem, &seeds);
    let config = IfdsConfig::new().with_path_edge_collection(true);
    assert!(config.collects_path_edges());
    let collected = solve_ifds(&graph, &problem, &seeds, config).expect("an unbounded solve");
    assert_same_answer(&graph, 4, &kept, &collected);
    assert_eq!(kept.peak_path_edges(), kept.steps(), "nothing is dropped");
    assert!(
        2 * collected.peak_path_edges() < kept.peak_path_edges(),
        "peak {} with collection, {} without",
        collected.peak_path_edges(),
        kept.peak_path_edges()
    );
}

#[test]
fn a_call_into_a_collected_context_reuses_its_summaries() {
    // main calls leaf twice with the zero fact, far apart. The leaf's nodes
    // come first, so the worklist finishes it right after the first call and
    // a collection pass drops it before the second call, which still gets
    // the leaf's facts without analyzing the leaf again.
    let mut graph = Toy::default();
    let leaf = graph.add_proc(3);
    graph.edge(leaf[0], leaf[1]);
    graph.edge(leaf[1], leaf[2]);
    let main = graph.add_proc(200);
    for pair in main.windows(2) {
        graph.edge(pair[0], pair[1]);
    }
    graph.call(main[1], 0);
    graph.call(main[150], 0);
    let leaf_start = leaf[0];
    let flow = move |step: Step, fact: u32| -> Vec<u32> {
        match step {
            Step::Normal { from, .. } if from == leaf_start && fact == 0 => vec![1],
            Step::CallToReturn { .. } => vec![],
            _ => (fact != 0).then_some(fact).into_iter().collect(),
        }
    };
    let problem = Flows { facts: 2, flow };
    let seeds = [(main[0], 0)];
    let kept = solve(&graph, &problem, &seeds);
    let collected = solve_ifds(
        &graph,
        &problem,
        &seeds,
        IfdsConfig::new().with_path_edge_collection(true),
    )
    .expect("an unbounded solve");
    assert_same_answer(&graph, 2, &kept, &collected);
    assert_eq!(facts_of(&collected, main[151]), [0, 1]);
    assert_eq!(
        collected.summaries(0, 0).collect::<Vec<_>>(),
        [(leaf[2], 0), (leaf[2], 1)]
    );
    assert!(collected.peak_path_edges() < kept.peak_path_edges());
}

#[test]
fn the_step_limit_counts_processed_path_edges() {
    // A chain of four nodes whose start generates fact 1: seven path edges.
    let mut graph = Toy::default();
    let main = graph.add_proc(4);
    for pair in main.windows(2) {
        graph.edge(pair[0], pair[1]);
    }
    let problem = Flows {
        facts: 2,
        flow: |_: Step, _: u32| vec![1],
    };
    let seeds = [(main[0], 0)];
    let steps = solve(&graph, &problem, &seeds).steps();
    assert_eq!(steps, 7);
    let exact = IfdsConfig::new().with_solve_config(SolveConfig::with_step_limit(steps));
    assert_eq!(exact.solve_config().max_steps(), Some(steps));
    assert!(solve_ifds(&graph, &problem, &seeds, exact).is_ok());

    let short = IfdsConfig::new().with_solve_config(SolveConfig::with_step_limit(steps - 1));
    let error = solve_ifds(&graph, &problem, &seeds, short).expect_err("one path edge is left");
    let SolveError::StepLimitExceeded {
        limit,
        steps: done,
        pending_node,
    } = error;
    assert_eq!((limit, done, pending_node), (6, 6, main[3]));
}

#[test]
fn no_seed_reaches_nothing_and_a_mid_procedure_seed_holds_where_it_is() {
    let mut graph = Toy::default();
    let main = graph.add_proc(3);
    graph.edge(main[0], main[1]);
    graph.edge(main[1], main[2]);
    let problem = Flows {
        facts: 2,
        flow: |_: Step, fact: u32| (fact != 0).then_some(fact).into_iter().collect(),
    };
    let none = solve(&graph, &problem, &[]);
    assert_eq!(none.steps(), 0);
    assert!((0..3).all(|node| none.facts_at(node).next().is_none()));

    let seeded = solve(&graph, &problem, &[(main[1], 1)]);
    assert_eq!(facts_of(&seeded, main[0]), [] as [u32; 0]);
    assert_eq!(facts_of(&seeded, main[1]), [1]);
    assert_eq!(facts_of(&seeded, main[2]), [1]);
    assert_eq!(seeded.summaries(0, 0).collect::<Vec<_>>(), [(main[2], 1)]);
}

#[test]
#[should_panic(expected = "seed node is out of range")]
fn a_seed_past_the_nodes_panics() {
    let mut graph = Toy::default();
    graph.add_proc(1);
    let problem = Flows {
        facts: 1,
        flow: |_: Step, _: u32| Vec::new(),
    };
    let _ = solve_ifds(&graph, &problem, &[(1, 0)], IfdsConfig::new());
}

#[test]
#[should_panic(expected = "fact is out of range")]
fn a_flow_fact_past_the_bound_panics() {
    let mut graph = Toy::default();
    let main = graph.add_proc(2);
    graph.edge(main[0], main[1]);
    let problem = Flows {
        facts: 2,
        flow: |_: Step, _: u32| vec![2],
    };
    let _ = solve_ifds(&graph, &problem, &[(main[0], 0)], IfdsConfig::new());
}

/// A toy instruction: work, a taint source, or a call by key.
#[derive(Debug, Clone, PartialEq, Eq)]
enum Op {
    Work,
    Source,
    Call(u32),
}

impl FlowControl for Op {
    type Target = u32;
    type CaseKey = ();

    fn flow_effect(&self) -> FlowEffect<u32, ()> {
        FlowEffect::Flow(match self {
            Op::Call(key) => Flow::Call { target: *key },
            Op::Work | Op::Source => Flow::Next,
        })
    }
}

/// Taint over a [`CfgSupergraph`]: `Source` taints fact 1, and a call
/// passes it through the callee rather than alongside.
struct CfgTaint<'a, 'g>(&'a CfgSupergraph<'g, Cfg<Op>>);

impl<'g> IfdsProblem<CfgSupergraph<'g, Cfg<Op>>> for CfgTaint<'_, 'g> {
    fn fact_bound(&self) -> usize {
        2
    }

    fn normal(&self, from: usize, _: usize, fact: u32, out: &mut Vec<u32>) {
        if fact == 0 && self.0.instruction(from) == Some(&Op::Source) {
            out.push(1);
        } else if fact != 0 {
            out.push(fact);
        }
    }

    fn call(&self, _: usize, _: usize, fact: u32, out: &mut Vec<u32>) {
        out.extend((fact != 0).then_some(fact));
    }

    fn ret(&self, _: usize, _: usize, _: usize, _: usize, fact: u32, out: &mut Vec<u32>) {
        out.extend((fact != 0).then_some(fact));
    }

    fn call_to_return(&self, _: usize, _: usize, _: u32, _: &mut Vec<u32>) {}
}

#[test]
fn the_cfg_adapter_numbers_program_points_and_resolves_listed_callees() {
    // main (key 10): bb0 [Source, Call(20)] -> bb1 [Call(99), Work].
    // helper (key 20): bb0 [Work]. A second CFG under key 20 is never called.
    let mut main = Cfg::<Op>::new();
    let next = main.new_block();
    main.block_mut(main.entry()).push(Op::Source);
    main.block_mut(main.entry()).push(Op::Call(20));
    main.block_mut(next).push(Op::Call(99));
    main.block_mut(next).push(Op::Work);
    main.add_edge(main.entry(), next, EdgeKind::Fallthrough);
    let mut helper = Cfg::<Op>::new();
    helper.block_mut(helper.entry()).push(Op::Work);
    let shadow = Cfg::<Op>::new();
    let functions = [(10, &main), (20, &helper), (20, &shadow)];
    let graph = CfgSupergraph::new(&functions);

    // main: 0..3 in bb0, 3..6 in bb1; helper: 6, 7; the shadow: 8.
    assert_eq!(graph.node_bound(), 9);
    assert_eq!(graph.proc_bound(), 3);
    let entry = main.entry();
    assert_eq!(
        graph.point(1),
        (
            0,
            ProgramPoint {
                block: entry,
                inst_idx: 1
            }
        )
    );
    assert_eq!(
        graph.node(
            0,
            ProgramPoint {
                block: next,
                inst_idx: 2
            }
        ),
        5
    );
    assert_eq!(graph.instruction(3), Some(&Op::Call(99)));
    assert_eq!(graph.instruction(5), None);
    assert_eq!((graph.start(0), graph.start(1), graph.start(2)), (0, 6, 8));
    assert_eq!(graph.proc_of(7), 1);

    let mut out = Vec::new();
    graph.callees(1, &mut out);
    assert_eq!(out, [1], "key 20 resolves to its first procedure");
    out.clear();
    graph.callees(3, &mut out);
    assert!(out.is_empty(), "key 99 is not listed, so the call is plain");
    let mut next_nodes = Vec::new();
    graph.successors(2, &mut next_nodes);
    assert_eq!(next_nodes, [3], "the end of bb0 flows to bb1");
    assert!(!graph.is_exit(2));
    assert!(graph.is_exit(5) && graph.is_exit(7) && graph.is_exit(8));

    let results = solve_ifds(&graph, &CfgTaint(&graph), &[(0, 0)], IfdsConfig::new())
        .expect("an unbounded solve");
    assert_eq!(results.facts_at(1).collect::<Vec<_>>(), [0, 1]);
    assert_eq!(
        results.facts_at(6).collect::<Vec<_>>(),
        [0, 1],
        "the taint enters helper"
    );
    assert_eq!(
        results.facts_at(5).collect::<Vec<_>>(),
        [0, 1],
        "and comes back"
    );
    assert_eq!(results.facts_at(8).count(), 0, "the shadow is never called");
    assert_eq!(results.summaries(1, 1).collect::<Vec<_>>(), [(7, 1)]);
}

#[test]
#[should_panic(expected = "the block is not in the procedure")]
fn the_adapter_rejects_a_removed_block() {
    let mut cfg = Cfg::<Op>::new();
    let gone = cfg.new_block();
    cfg.remove_block(gone);
    let functions = [(0, &cfg)];
    let graph = CfgSupergraph::new(&functions);
    assert_eq!(graph.node_bound(), 1);
    let _ = graph.node(
        0,
        ProgramPoint {
            block: gone,
            inst_idx: 0,
        },
    );
}
