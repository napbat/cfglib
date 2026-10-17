//! `MemoryValueFlow`: the graph that connects ordinary values through memory.
//! Exact edge lists on tiny programs, the store-to-load paths memory carries
//! under each alias vocabulary, the documented errors, and a differential
//! check of memory-carried reachability against the reaching-store
//! exploration on random programs.

use std::collections::{BTreeMap, BTreeSet};

use cfglib::{
    BlockId, Cfg, ConservativeMemoryAlias, DominatorTree, EdgeKind, ExactMemoryAlias, MemoryAccess,
    MemoryEvent, MemoryEventSite, MemorySsa, MemoryValueEdge, MemoryValueFlow,
    MemoryValueFlowError, MemoryValueFlowScratch, MemoryValueNode, MemoryValueRole, NodeId,
    ProgramPoint, SsaForm, SsaValue,
};

use crate::machine::{Barrier, Inst, Program, Rng, chain, diamond};
use crate::oracle::{Reaching, alias_classes, explore};

type Memory = MemorySsa<u8, u8, Barrier>;
type Flow = MemoryValueFlow<u8>;
type Node = MemoryValueNode<u8>;
/// The events that store (or load) each ordinary value.
type ValueSites = BTreeMap<SsaValue<u8>, Vec<MemoryEventSite>>;
/// For each stored value, the loaded values one hop through memory reaches.
type Hops<'a> = BTreeMap<&'a SsaValue<u8>, BTreeSet<&'a SsaValue<u8>>>;

fn point(block: usize, inst_idx: usize) -> ProgramPoint {
    ProgramPoint {
        block: BlockId::from_index(block),
        inst_idx,
    }
}

fn site(block: usize, inst_idx: usize) -> MemoryEventSite {
    MemoryEventSite::new(point(block, inst_idx), 0)
}

fn analyze<A: cfglib::MemoryAlias<u8> + ?Sized>(
    cfg: &Cfg<Inst<u8>>,
    alias: &A,
) -> (SsaForm<u8>, Memory, Flow) {
    let ssa = SsaForm::compute(cfg, &DominatorTree::compute(cfg));
    let memory = MemorySsa::compute(cfg, alias);
    let flow = MemoryValueFlow::compute(&memory, &ssa).expect("a consistent program");
    (ssa, memory, flow)
}

/// Every edge as `(source node, target node, edge)`.
fn edges(flow: &Flow) -> BTreeSet<(Node, Node, MemoryValueEdge)> {
    flow.graph()
        .edges()
        .map(|record| {
            (
                flow.graph().node(record.source()).clone(),
                flow.graph().node(record.target()).clone(),
                *record.payload(),
            )
        })
        .collect()
}

fn value(variable: u8, version: usize) -> Node {
    MemoryValueNode::Value(SsaValue::new(variable, version))
}

/// Whether memory carries a value from `from` to `to`: a path that uses only
/// the edges stores, states, phis, reads, and loads make.
fn memory_carries(flow: &Flow, from: NodeId, to: NodeId) -> bool {
    let mut adjacency: BTreeMap<NodeId, Vec<NodeId>> = BTreeMap::new();
    for record in flow.graph().edges() {
        if matches!(
            record.payload(),
            MemoryValueEdge::Store { .. }
                | MemoryValueEdge::Write { .. }
                | MemoryValueEdge::MemoryPhi { .. }
                | MemoryValueEdge::MemoryPhiStart { .. }
                | MemoryValueEdge::Read { .. }
                | MemoryValueEdge::Load { .. }
        ) {
            adjacency
                .entry(record.source())
                .or_default()
                .push(record.target());
        }
    }
    let mut seen = BTreeSet::new();
    let mut stack = vec![from];
    while let Some(node) = stack.pop() {
        if node == to {
            return true;
        }
        if seen.insert(node) {
            stack.extend(adjacency.get(&node).into_iter().flatten().copied());
        }
    }
    false
}

fn carries(flow: &Flow, from: &SsaValue<u8>, to: &SsaValue<u8>) -> bool {
    let from = flow
        .value_node(from)
        .expect("the source value is in the graph");
    let to = flow
        .value_node(to)
        .expect("the target value is in the graph");
    memory_carries(flow, from, to)
}

#[test]
fn a_store_and_a_load_connect_the_stored_value_to_the_loaded_one_through_the_memory_state() {
    // v1 = ...; x = v1; v2 = x
    let cfg = chain(vec![vec![
        Inst::compute(1, &[]),
        Inst::store(0, 1),
        Inst::load(0, 2),
    ]]);
    let (_, memory, flow) = analyze(&cfg, &ExactMemoryAlias);
    let (store, load) = (site(0, 1), site(0, 2));
    let class = memory.class_of(&0).unwrap();
    let state = |version| MemoryValueNode::Memory(SsaValue::new(class, version));

    assert_eq!(flow.node_count(), 6);
    assert_eq!(
        edges(&flow),
        BTreeSet::from([
            (
                value(1, 1),
                MemoryValueNode::Event(store),
                MemoryValueEdge::Store {
                    site: store,
                    input_index: 0
                }
            ),
            (
                MemoryValueNode::Event(store),
                state(1),
                MemoryValueEdge::Write { site: store }
            ),
            (
                state(1),
                MemoryValueNode::Event(load),
                MemoryValueEdge::Read { site: load }
            ),
            (
                MemoryValueNode::Event(load),
                value(2, 1),
                MemoryValueEdge::Load {
                    site: load,
                    output_index: 0
                }
            ),
        ]),
        "a blind write has no edge from the state it replaces"
    );
    assert_eq!(flow.edge_count(), 4);
    assert!(carries(&flow, &SsaValue::new(1, 1), &SsaValue::new(2, 1)));
    // The nodes are findable by what they stand for.
    assert!(
        flow.memory_node(&SsaValue::live_in(class)).is_some(),
        "the initial state is a node too"
    );
    assert!(flow.event_node(store).is_some());
    assert!(flow.value_node(&SsaValue::new(7, 3)).is_none());
    assert!(
        flow.event_node(site(0, 0)).is_none(),
        "a pure instruction has no event node"
    );
}

#[test]
fn the_start_values_of_a_loop_headed_by_the_entry_enter_the_phis_over_start_edges() {
    // entry (header): v2 = x;   latch: v1 = ...; x = v1; back to the entry.
    // The entry merges v1, v2, and the state of x with what the function
    // starts with, which no CFG edge carries.
    let mut cfg = Cfg::new();
    let (latch, exit) = (cfg.new_block(), cfg.new_block());
    let entry = cfg.entry();
    cfg.add_edge(entry, latch, EdgeKind::ConditionalTrue);
    cfg.add_edge(entry, exit, EdgeKind::ConditionalFalse);
    cfg.add_edge(latch, entry, EdgeKind::Back);
    cfg.block_mut(entry).push(Inst::load(0, 2));
    cfg.block_mut(latch).push(Inst::compute(1, &[]));
    cfg.block_mut(latch).push(Inst::store(0, 1));
    let (_, memory, flow) = analyze(&cfg, &ExactMemoryAlias);
    let class = memory.class_of(&0).unwrap();
    let state = |version| MemoryValueNode::Memory(SsaValue::new(class, version));
    let (load, store) = (site(0, 0), site(1, 1));
    let end_of_latch = point(1, 2);
    let at_entry = BlockId::from_index(0);

    assert_eq!(
        edges(&flow),
        BTreeSet::from([
            (
                value(1, 0),
                value(1, 1),
                MemoryValueEdge::ValuePhiStart { block: at_entry }
            ),
            (
                value(2, 0),
                value(2, 1),
                MemoryValueEdge::ValuePhiStart { block: at_entry }
            ),
            (
                value(1, 2),
                value(1, 1),
                MemoryValueEdge::ValuePhi {
                    block: at_entry,
                    predecessor: end_of_latch
                }
            ),
            (
                value(2, 2),
                value(2, 1),
                MemoryValueEdge::ValuePhi {
                    block: at_entry,
                    predecessor: end_of_latch
                }
            ),
            (
                state(0),
                state(1),
                MemoryValueEdge::MemoryPhiStart { block: at_entry }
            ),
            (
                state(2),
                state(1),
                MemoryValueEdge::MemoryPhi {
                    block: at_entry,
                    predecessor: end_of_latch
                }
            ),
            (
                state(1),
                MemoryValueNode::Event(load),
                MemoryValueEdge::Read { site: load }
            ),
            (
                MemoryValueNode::Event(load),
                value(2, 2),
                MemoryValueEdge::Load {
                    site: load,
                    output_index: 0
                }
            ),
            (
                value(1, 2),
                MemoryValueNode::Event(store),
                MemoryValueEdge::Store {
                    site: store,
                    input_index: 0
                }
            ),
            (
                MemoryValueNode::Event(store),
                state(2),
                MemoryValueEdge::Write { site: store }
            ),
        ]),
        "the start values have their own edges, apart from the departures"
    );
    assert!(carries(&flow, &SsaValue::new(1, 2), &SsaValue::new(2, 2)));
}

#[test]
fn a_read_modify_write_has_both_a_read_edge_and_a_write_edge() {
    // old = exchange(x, new)
    let cfg = chain(vec![vec![Inst::exchange(0, 1, 2)]]);
    let (_, memory, flow) = analyze(&cfg, &ExactMemoryAlias);
    let exchange = site(0, 0);
    let class = memory.class_of(&0).unwrap();
    let state = |version| MemoryValueNode::Memory(SsaValue::new(class, version));
    assert_eq!(
        edges(&flow),
        BTreeSet::from([
            (
                value(1, 0),
                MemoryValueNode::Event(exchange),
                MemoryValueEdge::Store {
                    site: exchange,
                    input_index: 0
                }
            ),
            (
                state(0),
                MemoryValueNode::Event(exchange),
                MemoryValueEdge::Read { site: exchange }
            ),
            (
                MemoryValueNode::Event(exchange),
                state(1),
                MemoryValueEdge::Write { site: exchange }
            ),
            (
                MemoryValueNode::Event(exchange),
                value(2, 1),
                MemoryValueEdge::Load {
                    site: exchange,
                    output_index: 0
                }
            ),
            (
                value(1, 0),
                value(2, 1),
                MemoryValueEdge::Value { point: point(0, 0) }
            ),
        ]),
        "the ordinary use also feeds the ordinary definition of its instruction"
    );
}

#[test]
fn address_inputs_enter_an_event_by_their_own_edge_kind() {
    // table[v0][v1] = v2
    let cfg = chain(vec![vec![Inst::store_indexed(0, 0, 1, 2)]]);
    let (_, memory, flow) = analyze(&cfg, &ExactMemoryAlias);
    let store = site(0, 0);
    let class = memory.class_of(&0).unwrap();
    assert_eq!(
        edges(&flow),
        BTreeSet::from([
            (
                value(0, 0),
                MemoryValueNode::Event(store),
                MemoryValueEdge::Address {
                    site: store,
                    input_index: 0
                }
            ),
            (
                value(1, 0),
                MemoryValueNode::Event(store),
                MemoryValueEdge::Address {
                    site: store,
                    input_index: 1
                }
            ),
            (
                value(2, 0),
                MemoryValueNode::Event(store),
                MemoryValueEdge::Store {
                    site: store,
                    input_index: 0
                }
            ),
            (
                MemoryValueNode::Event(store),
                MemoryValueNode::Memory(SsaValue::new(class, 1)),
                MemoryValueEdge::Write { site: store },
            ),
        ])
    );
}

#[test]
fn a_fence_is_an_isolated_event_node() {
    let cfg = chain(vec![vec![Inst::fence(Barrier::Acquire)]]);
    let (_, _, flow) = analyze(&cfg, &ExactMemoryAlias);
    assert_eq!(flow.node_count(), 1);
    assert_eq!(flow.edge_count(), 0);
    assert!(flow.event_node(site(0, 0)).is_some());
}

#[test]
fn memory_phis_carry_each_arms_store_to_the_load_after_the_join() {
    let cfg = diamond(
        vec![],
        vec![Inst::store(0, 1)],
        vec![Inst::store(0, 2)],
        vec![Inst::load(0, 3)],
    );
    let (_, memory, flow) = analyze(&cfg, &ExactMemoryAlias);
    let phi_edges: Vec<_> = edges(&flow)
        .into_iter()
        .filter_map(|(_, _, edge)| match edge {
            MemoryValueEdge::MemoryPhi { block, predecessor } => Some((block, predecessor)),
            _ => None,
        })
        .collect();
    assert_eq!(
        phi_edges,
        [
            (BlockId::from_index(3), point(1, 1)),
            (BlockId::from_index(3), point(2, 1))
        ]
    );
    assert_eq!(memory.phis().len(), 1);
    let loaded = SsaValue::new(3, 1);
    assert!(carries(&flow, &SsaValue::new(1, 0), &loaded));
    assert!(carries(&flow, &SsaValue::new(2, 0), &loaded));
}

#[test]
fn a_blind_write_hides_the_earlier_value_and_a_read_modify_write_passes_it_on() {
    // x = v1; x = v2; v3 = x   -- only v2 reaches the load
    let blind = chain(vec![vec![
        Inst::store(0, 1),
        Inst::store(0, 2),
        Inst::load(0, 3),
    ]]);
    let (_, _, flow) = analyze(&blind, &ExactMemoryAlias);
    assert!(carries(&flow, &SsaValue::new(2, 0), &SsaValue::new(3, 1)));
    assert!(!carries(&flow, &SsaValue::new(1, 0), &SsaValue::new(3, 1)));

    // x = v1; v4 = exchange(x, v2); v3 = x   -- both reach the load
    let chained = chain(vec![vec![
        Inst::store(0, 1),
        Inst::exchange(0, 2, 4),
        Inst::load(0, 3),
    ]]);
    let (_, _, flow) = analyze(&chained, &ExactMemoryAlias);
    assert!(carries(&flow, &SsaValue::new(2, 0), &SsaValue::new(3, 1)));
    assert!(
        carries(&flow, &SsaValue::new(1, 0), &SsaValue::new(3, 1)),
        "through the exchange's own write"
    );
    assert!(
        carries(&flow, &SsaValue::new(1, 0), &SsaValue::new(4, 1)),
        "the exchange observes what was stored"
    );
}

#[test]
fn the_alias_vocabulary_decides_which_locations_carry_a_value() {
    // x = v1; v2 = y
    let cfg = chain(vec![vec![Inst::store(0, 1), Inst::load(1, 2)]]);
    let (_, _, exact) = analyze(&cfg, &ExactMemoryAlias);
    assert!(
        !carries(&exact, &SsaValue::new(1, 0), &SsaValue::new(2, 1)),
        "distinct locations are disjoint"
    );

    let (_, _, conservative) = analyze(&cfg, &ConservativeMemoryAlias);
    assert!(
        carries(&conservative, &SsaValue::new(1, 0), &SsaValue::new(2, 1)),
        "one class holds both"
    );

    let nearby = |left: &u8, right: &u8| left.abs_diff(*right) <= 1;
    let (_, _, custom) = analyze(&cfg, &nearby);
    assert!(carries(&custom, &SsaValue::new(1, 0), &SsaValue::new(2, 1)));
}

#[test]
fn a_declared_variable_the_instruction_does_not_expose_is_a_structured_error() {
    let access = |access: MemoryAccess<u8, u8>| vec![MemoryEvent::Access(access)];
    let analyze_one = |inst: Inst<u8>| {
        let cfg = chain(vec![vec![inst]]);
        let ssa = SsaForm::compute(&cfg, &DominatorTree::compute(&cfg));
        let memory: Memory = MemorySsa::compute(&cfg, &ExactMemoryAlias);
        MemoryValueFlow::compute(&memory, &ssa).map(|_| ())
    };
    let here = site(0, 0);

    // The address input is not among the instruction's uses.
    let address = Inst::new(
        &[1],
        &[],
        access(MemoryAccess::write(0, [1]).with_address_uses([9])),
    );
    assert_eq!(
        analyze_one(address),
        Err(MemoryValueFlowError::MissingVariable {
            site: here,
            variable: 9,
            role: MemoryValueRole::Address
        })
    );
    // The stored value is not among the instruction's uses.
    let stored = Inst::new(&[1], &[], access(MemoryAccess::write(0, [9])));
    assert_eq!(
        analyze_one(stored),
        Err(MemoryValueFlowError::MissingVariable {
            site: here,
            variable: 9,
            role: MemoryValueRole::Stored
        })
    );
    // The loaded value is not among the instruction's definitions, even
    // though it is among its uses.
    let loaded = Inst::new(&[9], &[2], access(MemoryAccess::read(0, [9])));
    assert_eq!(
        analyze_one(loaded),
        Err(MemoryValueFlowError::MissingVariable {
            site: here,
            variable: 9,
            role: MemoryValueRole::Loaded
        })
    );
    assert!(analyze_one(Inst::load(0, 2)).is_ok());
}

#[test]
fn an_ssa_form_that_does_not_describe_the_event_is_reported_by_event() {
    let big = chain(vec![vec![
        Inst::compute(1, &[]),
        Inst::store(0, 1),
        Inst::load(0, 2),
    ]]);
    let small = chain(vec![vec![Inst::<u8>::compute(1, &[])]]);
    let memory: Memory = MemorySsa::compute(&big, &ExactMemoryAlias);
    let ssa = SsaForm::compute(&small, &DominatorTree::compute(&small));
    assert_eq!(
        MemoryValueFlow::compute(&memory, &ssa).map(|_| ()),
        Err(MemoryValueFlowError::MissingInstruction { site: site(0, 1) })
    );
}

#[test]
fn flow_errors_are_ordinary_errors_and_a_scratch_survives_one() {
    let bad = Inst::new(
        &[1],
        &[],
        vec![MemoryEvent::Access(MemoryAccess::write(0, [9]))],
    );
    let broken = chain(vec![vec![bad]]);
    let ssa = SsaForm::compute(&broken, &DominatorTree::compute(&broken));
    let memory: Memory = MemorySsa::compute(&broken, &ExactMemoryAlias);
    let mut scratch = MemoryValueFlowScratch::new();
    let error = MemoryValueFlow::compute_in(&mut scratch, &memory, &ssa)
        .map(|_| ())
        .unwrap_err();
    assert_eq!(
        Err(error.clone()),
        MemoryValueFlow::compute(&memory, &ssa).map(|_| ())
    );
    let boxed: Box<dyn std::error::Error> = Box::new(error);
    assert_ne!(boxed.to_string(), "");
    for role in [
        MemoryValueRole::Address,
        MemoryValueRole::Stored,
        MemoryValueRole::Loaded,
    ] {
        assert_ne!(role.to_string(), "");
    }

    // After the failure the same scratch builds a valid graph.
    let good = chain(vec![vec![
        Inst::compute(1, &[]),
        Inst::store(0, 1),
        Inst::load(0, 2),
    ]]);
    let ssa = SsaForm::compute(&good, &DominatorTree::compute(&good));
    let memory: Memory = MemorySsa::compute(&good, &ExactMemoryAlias);
    let reused = MemoryValueFlow::compute_in(&mut scratch, &memory, &ssa).unwrap();
    assert_eq!(reused, MemoryValueFlow::compute(&memory, &ssa).unwrap());
}

#[test]
fn memory_carried_reachability_matches_the_reaching_store_exploration_on_random_programs() {
    let mut rng = Rng::new(71);
    let mut checked = 0;
    let (mut connected_pairs, mut disconnected_pairs) = (0, 0);
    while checked < 250 {
        let program = Program::random(&mut rng, true);
        if !program.every_live_block_is_reachable() {
            continue;
        }
        checked += 1;
        let cfg = program.cfg();
        let relation: &dyn Fn(&u8, &u8) -> bool = if checked % 2 == 0 {
            &|left, right| left == right
        } else {
            &|left, right| left.abs_diff(*right) <= 1
        };
        let (ssa, _, flow) = analyze(&cfg, &relation);

        let locations = program.accessed_locations();
        let reaching = explore(&program, &alias_classes(&locations, relation));

        let (stored, loaded) = stored_and_loaded_values(&program, &ssa);
        let hops = memory_hops(&stored, &loaded, &reaching);
        // A loaded value that is stored again carries the dependency on, so
        // reachability is the closure of the hops.
        for from in stored.keys() {
            let reached = reachable_through_hops(&hops, from);
            for to in loaded.keys() {
                if from != to {
                    let expected = reached.contains(to);
                    assert_eq!(
                        carries(&flow, from, to),
                        expected,
                        "{from:?} to {to:?} in round {checked}"
                    );
                    if expected {
                        connected_pairs += 1;
                    } else {
                        disconnected_pairs += 1;
                    }
                }
            }
        }
    }
    // Both outcomes must occur often, or the comparison proves nothing.
    assert!(
        connected_pairs >= 20,
        "pairs memory connects: {connected_pairs}"
    );
    assert!(
        disconnected_pairs >= 20,
        "pairs memory separates: {disconnected_pairs}"
    );
}

/// Which ordinary values each event stores and loads, as SSA values.
fn stored_and_loaded_values(program: &Program<u8>, ssa: &SsaForm<u8>) -> (ValueSites, ValueSites) {
    let mut stored = ValueSites::new();
    let mut loaded = ValueSites::new();
    for block in (0..program.blocks).filter(|block| program.is_live(*block)) {
        for (index, inst) in program.code[block].iter().enumerate() {
            let annotation = ssa.instruction(point(block, index)).unwrap();
            for (event_index, event) in inst.events.iter().enumerate() {
                let MemoryEvent::Access(access) = event else {
                    continue;
                };
                let here = MemoryEventSite::new(point(block, index), event_index);
                let find = |values: &[SsaValue<u8>], variable: u8| {
                    values
                        .iter()
                        .find(|candidate| candidate.variable == variable)
                        .cloned()
                        .unwrap()
                };
                for variable in access.value_uses() {
                    stored
                        .entry(find(annotation.uses(), *variable))
                        .or_default()
                        .push(here);
                }
                for variable in access.value_defs() {
                    loaded
                        .entry(find(annotation.defs(), *variable))
                        .or_default()
                        .push(here);
                }
            }
        }
    }
    (stored, loaded)
}

/// One hop through memory: a stored value reaches a loaded value when a read
/// of it is fed by the write, or when one exchange both stores the first and
/// loads the second (one event node joins its inputs to its outputs, although
/// the loaded value is the one observed before the store commits).
fn memory_hops<'a>(
    stored: &'a ValueSites,
    loaded: &'a ValueSites,
    reaching: &Reaching<u8>,
) -> Hops<'a> {
    let mut hops: Hops<'a> = BTreeMap::new();
    for (from, writes) in stored {
        for (to, reads) in loaded {
            let connected = writes.iter().any(|write| {
                reads.contains(write)
                    || reaching
                        .transitive_readers(*write)
                        .iter()
                        .any(|reader| reads.contains(reader))
            });
            if connected {
                hops.entry(from).or_default().insert(to);
            }
        }
    }
    hops
}

/// Every value reachable from `from` by following one or more hops.
fn reachable_through_hops<'a>(hops: &Hops<'a>, from: &SsaValue<u8>) -> BTreeSet<&'a SsaValue<u8>> {
    let mut reached = BTreeSet::new();
    let mut pending: Vec<&SsaValue<u8>> = hops.get(from).into_iter().flatten().copied().collect();
    while let Some(value) = pending.pop() {
        if reached.insert(value) {
            pending.extend(hops.get(value).into_iter().flatten().copied());
        }
    }
    reached
}

#[test]
fn a_loaded_value_stored_again_carries_the_first_store_through_a_chain_of_exchanges() {
    // v1 = ...; v2 = exchange(a, v1); v3 = exchange(b, v2); v4 = b
    // v1 enters location a and the exchange hands back the old value v2,
    // which the second exchange stores into b, where v4 reads it. The path
    // from v1 to v4 crosses the loaded value v2 and two memory states: a
    // reachability that looked only one store-to-load hop ahead would miss it.
    let cfg = chain(vec![vec![
        Inst::compute(1, &[]),
        Inst::exchange(0, 1, 2),
        Inst::exchange(1, 2, 3),
        Inst::load(1, 4),
    ]]);
    let (_, _, flow) = analyze(&cfg, &ExactMemoryAlias);
    let v = |variable, version| SsaValue::new(variable, version);
    assert!(
        carries(&flow, &v(1, 1), &v(2, 1)),
        "one exchange stores v1 and loads v2"
    );
    assert!(
        carries(&flow, &v(2, 1), &v(3, 1)),
        "the second exchange stores v2 and loads v3"
    );
    assert!(
        carries(&flow, &v(2, 1), &v(4, 1)),
        "v2 is stored into b, which v4 reads"
    );
    assert!(
        carries(&flow, &v(1, 1), &v(4, 1)),
        "v1 reaches v4 through the value v2"
    );
    // Nothing flows backward: v3 and v4 are only loaded, never stored.
    assert!(!carries(&flow, &v(3, 1), &v(4, 1)));
    assert!(!carries(&flow, &v(4, 1), &v(1, 1)));
    assert!(!carries(&flow, &v(2, 1), &v(1, 1)));
}
