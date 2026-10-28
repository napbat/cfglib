extern crate alloc;
extern crate std;

use crate::builder::address::AddressInstruction;
use alloc::collections::BTreeSet;
use alloc::vec;
use alloc::vec::Vec;

use super::{HopEnd, UnwindEnd, UnwindHop, UnwindPath, UnwindRoute, UnwindStep};
use crate::block::BlockId;
use crate::builder::address::tests::{Inst, inst, quiet_inst};
use crate::builder::address::{AddressCfgOptions, AddressGraph, build_address_cfg_with_routes};
use crate::edge::EdgeKind;
use crate::flow::{EdgeRole, Flow};

mod errors;
mod shapes;

/// `(source, target, is an unwind edge, route)`; the route is set on route
/// edges only.
pub(super) type Wire = (u32, u32, bool, Option<usize>);

pub(super) fn wire(info: crate::builder::address::AddressEdgeInfo<'_, u32, i32>) -> Wire {
    let route = match info.role {
        EdgeRole::Route { route } => Some(route),
        _ => None,
    };
    (
        info.source,
        info.target,
        info.kind == EdgeKind::ExceptionUnwind,
        route,
    )
}

/// A built graph with the hops of its routes.
pub(super) struct Built {
    pub(super) graph: AddressGraph<Inst, Wire>,
    pub(super) hops: Vec<UnwindHop>,
}

/// Consecutive instructions from `start`, one for each flow.
pub(super) fn body(start: u32, flows: &[Flow<u32, i32>]) -> Vec<Inst> {
    (start..)
        .zip(flows)
        .map(|(address, flow)| inst(address, flow.clone()))
        .collect()
}

/// The stream of the parts, in the order given.
pub(super) fn stream(parts: &[Vec<Inst>]) -> Vec<Inst> {
    parts.concat()
}

/// Returns, at the addresses `0..=count`, so each range `n..n + 1` of
/// `n < count` is a valid protected range.
pub(super) fn main_code(count: u32) -> Vec<Inst> {
    body(
        0,
        &vec![Flow::Return; usize::try_from(count + 1).expect("small")],
    )
}

pub(super) fn route(
    protected: core::ops::Range<u32>,
    paths: Vec<(Vec<UnwindStep<u32>>, UnwindEnd<u32>)>,
) -> UnwindRoute<u32> {
    UnwindRoute {
        protected,
        paths: paths
            .into_iter()
            .map(|(steps, end)| UnwindPath { steps, end })
            .collect(),
    }
}

pub(super) fn funclets(addresses: &[u32]) -> Vec<UnwindStep<u32>> {
    addresses.iter().map(|&a| UnwindStep::Funclet(a)).collect()
}

pub(super) fn try_build(
    instructions: Vec<Inst>,
    routes: &[UnwindRoute<u32>],
) -> Result<Built, crate::builder::address::AddressBuildError<u32>> {
    let mut graph = build_address_cfg_with_routes(
        instructions,
        &[],
        routes,
        AddressCfgOptions::default(),
        wire,
    )?;
    let hops = graph.add_unwind_routes(routes, wire)?;
    Ok(Built { graph, hops })
}

pub(super) fn build(instructions: Vec<Inst>, routes: &[UnwindRoute<u32>]) -> Built {
    try_build(instructions, routes).expect("the routes are valid")
}

impl Built {
    pub(super) fn first_address(&self, block: BlockId) -> u32 {
        self.graph.cfg.block(block).instructions()[0].address()
    }

    /// The one block that the only hop of `source` goes to.
    pub(super) fn next_block(&self, source: BlockId) -> BlockId {
        let hops = self.hops_of(source);
        assert_eq!(hops.len(), 1, "one hop leaves the block");
        match hops[0].end {
            HopEnd::Block(target) => target,
            HopEnd::Leave => panic!("the hop leaves the function"),
        }
    }

    pub(super) fn block_of(&self, address: u32) -> BlockId {
        self.graph.instruction_blocks[&address]
    }

    /// The route edges as address pairs.
    pub(super) fn route_edges(&self) -> BTreeSet<(u32, u32)> {
        self.graph
            .cfg
            .edges()
            .filter(|edge| edge.data().payload().3.is_some())
            .map(|edge| {
                let &(source, target, _, _) = edge.data().payload();
                (source, target)
            })
            .collect()
    }

    /// The route edges as block pairs.
    pub(super) fn route_edge_blocks(&self) -> BTreeSet<(BlockId, BlockId)> {
        self.graph
            .cfg
            .edges()
            .filter(|edge| edge.data().payload().3.is_some())
            .map(|edge| (edge.source(), edge.target()))
            .collect()
    }

    /// The blocks that start at `address`: the original and each copy.
    pub(super) fn instances_of(&self, address: u32) -> Vec<BlockId> {
        self.graph
            .cfg
            .block_ids()
            .filter(|&block| self.first_address(block) == address)
            .collect()
    }

    /// The blocks that control reaches from `entry` along the edges that are
    /// no unwind or call edges.
    pub(super) fn reach(&self, entry: BlockId) -> BTreeSet<BlockId> {
        let cfg = &self.graph.cfg;
        let mut seen = BTreeSet::from([entry]);
        let mut pending = vec![entry];
        while let Some(block) = pending.pop() {
            for edge in cfg.outgoing(block) {
                let edge = cfg.edge(edge);
                let followed = !matches!(
                    edge.data().kind(),
                    EdgeKind::ExceptionUnwind | EdgeKind::Call
                );
                if followed && seen.insert(edge.target()) {
                    pending.push(edge.target());
                }
            }
        }
        seen
    }

    /// The blocks that control reaches from `start` along every edge.
    pub(super) fn closure(&self, start: BlockId) -> BTreeSet<BlockId> {
        let cfg = &self.graph.cfg;
        let mut seen = BTreeSet::from([start]);
        let mut pending = vec![start];
        while let Some(block) = pending.pop() {
            for target in cfg.successors(block) {
                if seen.insert(target) {
                    pending.push(target);
                }
            }
        }
        seen
    }

    pub(super) fn hops_of(&self, source: BlockId) -> Vec<&UnwindHop> {
        self.hops
            .iter()
            .filter(|hop| hop.source == source)
            .collect()
    }

    /// Every path from `source`: the funclets that the runtime enters, and
    /// the end. A handler is a block that starts at an address of `handlers`.
    /// The depth bound turns a cycle of funclet instances into a failure.
    pub(super) fn paths_from(
        &self,
        source: BlockId,
        handlers: &[u32],
    ) -> BTreeSet<(Vec<u32>, UnwindEnd<u32>)> {
        let mut paths = BTreeSet::new();
        self.collect_paths(source, handlers, &mut Vec::new(), &mut paths);
        paths
    }

    fn collect_paths(
        &self,
        source: BlockId,
        handlers: &[u32],
        prefix: &mut Vec<u32>,
        paths: &mut BTreeSet<(Vec<u32>, UnwindEnd<u32>)>,
    ) {
        assert!(prefix.len() < 8, "the funclet instances form a cycle");
        for hop in self.hops_of(source) {
            match hop.end {
                HopEnd::Leave => {
                    paths.insert((prefix.clone(), UnwindEnd::Propagate));
                }
                HopEnd::Block(target) if target == source => {
                    paths.insert((prefix.clone(), UnwindEnd::Resume));
                }
                HopEnd::Block(target) => {
                    let address = self.first_address(target);
                    if handlers.contains(&address) {
                        paths.insert((prefix.clone(), UnwindEnd::Handler(address)));
                    } else {
                        prefix.push(address);
                        for block in self.reach(target) {
                            self.collect_paths(block, handlers, prefix, paths);
                        }
                        prefix.pop();
                    }
                }
            }
        }
    }

    /// Checks every rule that holds for every route set: the paths from each
    /// protected instruction are the paths of its route, each hop that goes to
    /// a block is an edge and each edge is a hop, and the hops are sorted.
    pub(super) fn assert_exact(&self, routes: &[UnwindRoute<u32>], handlers: &[u32]) {
        for route in routes {
            let expected: BTreeSet<_> = route
                .paths
                .iter()
                .map(|path| (funclet_addresses(&path.steps), path.end))
                .collect();
            for address in route.protected.clone() {
                assert_eq!(
                    self.paths_from(self.block_of(address), handlers),
                    expected,
                    "paths from the protected instruction at {address}"
                );
            }
        }
        let hop_edges: BTreeSet<_> = self
            .hops
            .iter()
            .filter_map(|hop| match hop.end {
                HopEnd::Block(target) => Some((hop.source, target)),
                HopEnd::Leave => None,
            })
            .collect();
        assert_eq!(hop_edges, self.route_edge_blocks());
        assert!(
            self.hops
                .windows(2)
                .all(|pair| (pair[0].source, pair[0].end) < (pair[1].source, pair[1].end)),
            "the hops are sorted and have one entry for each source and end"
        );
    }
}

pub(super) fn funclet_addresses(steps: &[UnwindStep<u32>]) -> Vec<u32> {
    steps
        .iter()
        .filter_map(|step| match step {
            UnwindStep::Funclet(address) => Some(*address),
            UnwindStep::Call(_) => None,
        })
        .collect()
}

pub(super) fn call(tag: u32) -> UnwindStep<u32> {
    UnwindStep::Call(tag)
}

const RET: Flow<u32, i32> = Flow::Return;
const NEXT: Flow<u32, i32> = Flow::Next;

#[test]
fn a_protected_instruction_has_an_edge_to_the_first_funclet_only() {
    let instructions = stream(&[
        body(0, &[NEXT, RET]),
        body(10, &[NEXT, RET]),
        body(20, &[RET]),
        body(30, &[RET]),
    ]);
    let routes = [route(
        0..1,
        vec![
            (funclets(&[10, 20]), UnwindEnd::Propagate),
            (funclets(&[10]), UnwindEnd::Handler(30)),
        ],
    )];
    let built = build(instructions, &routes);

    assert_eq!(
        built.route_edges(),
        BTreeSet::from([(0, 10), (11, 20), (11, 30)]),
        "the protected instruction skips no funclet, and only the return of the first one \
         enters the second funclet and the handler"
    );
    built.assert_exact(&routes, &[30]);
    assert_eq!(
        built
            .hops_of(built.block_of(20))
            .iter()
            .map(|hop| hop.end)
            .collect::<Vec<_>>(),
        [HopEnd::Leave],
        "the return of the last funclet propagates"
    );
}

#[test]
fn route_edges_are_exception_unwind_edges_with_the_edge_role_of_the_route() {
    let instructions = stream(&[main_code(2), body(10, &[RET])]);
    let routes = [
        route(0..1, vec![(funclets(&[10]), UnwindEnd::Propagate)]),
        route(1..2, vec![(funclets(&[10]), UnwindEnd::Propagate)]),
    ];
    let built = build(instructions, &routes);

    let edges: BTreeSet<_> = built
        .graph
        .cfg
        .edges()
        .map(|edge| *edge.data().payload())
        .collect();
    assert_eq!(
        edges,
        BTreeSet::from([(0, 10, true, Some(0)), (1, 10, true, Some(1)),])
    );
    assert_eq!(
        built.instances_of(10).len(),
        1,
        "equal remaining paths share the instance of both routes"
    );
}

#[test]
fn the_edge_of_a_shared_instance_names_the_first_route_that_states_it() {
    let instructions = stream(&[main_code(2), body(10, &[RET]), body(20, &[RET])]);
    let routes = [
        route(0..1, vec![(funclets(&[10]), UnwindEnd::Handler(20))]),
        route(1..2, vec![(funclets(&[10]), UnwindEnd::Handler(20))]),
    ];
    let built = build(instructions, &routes);

    let tagged: BTreeSet<_> = built
        .graph
        .cfg
        .edges()
        .map(|edge| *edge.data().payload())
        .collect();
    assert_eq!(
        tagged,
        BTreeSet::from([
            (0, 10, true, Some(0)),
            (1, 10, true, Some(1)),
            (10, 20, true, Some(0)),
        ]),
        "the return of the shared funclet has one edge, with the first route"
    );
}

#[test]
fn shared_funclets_in_opposite_order_do_not_lead_into_each_other() {
    // Both protected instructions return, so each context stands alone.
    let instructions = stream(&[main_code(2), body(10, &[RET]), body(20, &[RET])]);
    let routes = [
        route(0..1, vec![(funclets(&[10, 20]), UnwindEnd::Propagate)]),
        route(1..2, vec![(funclets(&[20, 10]), UnwindEnd::Propagate)]),
    ];
    let built = build(instructions, &routes);

    built.assert_exact(&routes, &[]);
    assert_eq!(
        built.paths_from(built.block_of(0), &[]),
        BTreeSet::from([(vec![10, 20], UnwindEnd::Propagate)]),
    );
    assert_eq!(
        built.paths_from(built.block_of(1), &[]),
        BTreeSet::from([(vec![20, 10], UnwindEnd::Propagate)]),
    );
    assert_eq!(built.instances_of(10).len(), 2);
    assert_eq!(built.instances_of(20).len(), 2);

    let from_a = built.closure(built.block_of(0));
    let from_b = built.closure(built.block_of(1));
    assert!(
        from_a.is_disjoint(&from_b),
        "no block of one context is reachable from the other context"
    );
    // The first funclet that context B enters ends the way: its return
    // leads nowhere.
    let first_of_b = built.next_block(built.block_of(1));
    assert_eq!(built.first_address(first_of_b), 20);
    let second_of_b = built.next_block(first_of_b);
    assert_eq!(built.first_address(second_of_b), 10);
    assert_eq!(
        built
            .hops_of(second_of_b)
            .iter()
            .map(|hop| hop.end)
            .collect::<Vec<_>>(),
        [HopEnd::Leave]
    );
}

#[test]
fn a_funclet_that_two_chains_share_is_never_entered_in_the_wrong_context() {
    // The cycle shape of the table-driven edges: g then h, and h then g.
    let instructions = stream(&[main_code(3), body(10, &[RET]), body(20, &[RET])]);
    let routes = [
        route(0..1, vec![(funclets(&[10, 20]), UnwindEnd::Propagate)]),
        route(1..2, vec![(funclets(&[20, 10]), UnwindEnd::Propagate)]),
        route(2..3, vec![(funclets(&[10]), UnwindEnd::Propagate)]),
    ];
    let built = build(instructions, &routes);

    built.assert_exact(&routes, &[]);
    assert_eq!(
        built.paths_from(built.block_of(2), &[]),
        BTreeSet::from([(vec![10], UnwindEnd::Propagate)]),
        "a context that enters the funclet and ends must not go on to the other funclet"
    );
}

#[test]
fn chains_with_equal_remaining_paths_share_their_instances() {
    // State 5 enters f5, f4, f3; state 4 enters f4, f3.
    let instructions = stream(&[
        main_code(2),
        body(30, &[RET]),
        body(40, &[RET]),
        body(50, &[RET]),
    ]);
    let routes = [
        route(0..1, vec![(funclets(&[50, 40, 30]), UnwindEnd::Propagate)]),
        route(1..2, vec![(funclets(&[40, 30]), UnwindEnd::Propagate)]),
    ];
    let built = build(instructions, &routes);

    built.assert_exact(&routes, &[]);
    for funclet in [30, 40, 50] {
        assert_eq!(built.instances_of(funclet).len(), 1, "funclet {funclet}");
    }
}

#[test]
fn chains_with_different_remaining_paths_do_not_share_an_instance() {
    let with_funclets = |funclets: &[u32]| {
        let mut parts = vec![main_code(2)];
        parts.extend(funclets.iter().map(|&address| body(address, &[RET])));
        stream(&parts)
    };

    // The remaining path differs: f4 then f3, against f4 alone.
    let routes = [
        route(0..1, vec![(funclets(&[40, 30]), UnwindEnd::Propagate)]),
        route(1..2, vec![(funclets(&[40]), UnwindEnd::Propagate)]),
    ];
    let built = build(with_funclets(&[30, 40]), &routes);
    built.assert_exact(&routes, &[]);
    assert_eq!(built.instances_of(40).len(), 2);
    assert_eq!(built.instances_of(30).len(), 1);

    // The calls differ.
    let routes = [
        route(
            0..1,
            vec![(
                vec![UnwindStep::Funclet(40), call(1), UnwindStep::Funclet(30)],
                UnwindEnd::Propagate,
            )],
        ),
        route(1..2, vec![(funclets(&[40, 30]), UnwindEnd::Propagate)]),
    ];
    let built = build(with_funclets(&[30, 40]), &routes);
    built.assert_exact(&routes, &[]);
    assert_eq!(built.instances_of(40).len(), 2);
    assert_eq!(built.instances_of(30).len(), 1);

    // The end differs.
    let routes = [
        route(0..1, vec![(funclets(&[40]), UnwindEnd::Handler(60))]),
        route(1..2, vec![(funclets(&[40]), UnwindEnd::Propagate)]),
    ];
    let built = build(with_funclets(&[40, 60]), &routes);
    built.assert_exact(&routes, &[60]);
    assert_eq!(built.instances_of(40).len(), 2);
}

#[test]
fn instances_of_one_funclet_share_no_block() {
    let instructions = stream(&[main_code(2), body(10, &[RET]), body(20, &[RET])]);
    let routes = [
        route(0..1, vec![(funclets(&[10, 20]), UnwindEnd::Propagate)]),
        route(1..2, vec![(funclets(&[10]), UnwindEnd::Propagate)]),
    ];
    let built = build(instructions, &routes);

    let instances = built.instances_of(10);
    assert_eq!(instances.len(), 2);
    assert!(
        built
            .reach(instances[0])
            .is_disjoint(&built.reach(instances[1]))
    );
    assert!(
        instances.contains(&built.block_of(10)),
        "one instance is the original block of the address"
    );
}

#[test]
fn hops_list_the_distinct_call_alternatives_of_each_way_sorted() {
    let instructions = stream(&[main_code(1), body(10, &[RET]), body(20, &[RET])]);
    let routes = [route(
        0..1,
        vec![
            (
                vec![call(2), UnwindStep::Funclet(10)],
                UnwindEnd::Handler(20),
            ),
            (
                vec![call(1), UnwindStep::Funclet(10)],
                UnwindEnd::Handler(20),
            ),
            (
                vec![call(1), UnwindStep::Funclet(10)],
                UnwindEnd::Handler(20),
            ),
            (vec![UnwindStep::Funclet(10)], UnwindEnd::Handler(20)),
            (vec![call(1), call(2)], UnwindEnd::Propagate),
            (vec![call(5)], UnwindEnd::Propagate),
        ],
    )];
    let built = build(instructions, &routes);

    assert_eq!(
        built.hops,
        vec![
            UnwindHop {
                source: built.block_of(0),
                end: HopEnd::Block(built.block_of(10)),
                calls: vec![vec![], vec![1], vec![2]],
            },
            UnwindHop {
                source: built.block_of(0),
                end: HopEnd::Leave,
                calls: vec![vec![1, 2], vec![5]],
            },
            UnwindHop {
                source: built.block_of(10),
                end: HopEnd::Block(built.block_of(20)),
                calls: vec![vec![]],
            },
        ]
    );
    assert_eq!(
        built.route_edges(),
        BTreeSet::from([(0, 10), (10, 20)]),
        "the alternatives share one edge, and a call is no edge"
    );
}

#[test]
fn a_resume_end_is_an_edge_from_the_protected_block_to_itself() {
    let instructions = stream(&[body(0, &[NEXT, RET]), body(10, &[RET])]);
    let routes = [route(
        0..1,
        vec![
            (vec![call(7)], UnwindEnd::Resume),
            (funclets(&[10]), UnwindEnd::Propagate),
        ],
    )];
    let built = build(instructions, &routes);

    assert_eq!(built.route_edges(), BTreeSet::from([(0, 0), (0, 10)]));
    let itself: Vec<_> = built
        .hops_of(built.block_of(0))
        .into_iter()
        .filter(|hop| hop.end == HopEnd::Block(built.block_of(0)))
        .collect();
    assert_eq!(itself.len(), 1);
    assert_eq!(itself[0].calls, vec![vec![7]]);
    built.assert_exact(&routes, &[]);
}

#[test]
fn a_path_that_propagates_adds_a_hop_and_no_edge() {
    let instructions = stream(&[body(0, &[RET])]);
    let routes = [route(0..1, vec![(vec![call(3)], UnwindEnd::Propagate)])];
    let built = build(instructions, &routes);

    assert!(built.route_edges().is_empty());
    assert_eq!(
        built.hops,
        vec![UnwindHop {
            source: built.block_of(0),
            end: HopEnd::Leave,
            calls: vec![vec![3]],
        }]
    );
}

#[test]
fn an_instruction_that_cannot_throw_has_no_way() {
    let mut instructions = stream(&[body(1, &[RET]), body(10, &[RET])]);
    instructions.insert(0, quiet_inst(0, NEXT));
    let routes = [route(0..10, vec![(funclets(&[10]), UnwindEnd::Propagate)])];
    let built = build(instructions, &routes);

    assert_eq!(built.route_edges(), BTreeSet::from([(1, 10)]));
    assert_eq!(built.hops_of(built.block_of(0)), Vec::<&UnwindHop>::new());
}

#[test]
fn a_route_that_has_no_path_adds_nothing() {
    let instructions = stream(&[body(0, &[RET])]);
    let routes = [route(0..1, vec![])];
    let built = build(instructions, &routes);

    assert!(built.route_edges().is_empty());
    assert_eq!(built.hops, Vec::<UnwindHop>::new());
}

#[test]
fn the_hops_are_in_block_order_and_the_result_is_deterministic() {
    let instructions = || {
        stream(&[
            main_code(2),
            body(10, &[RET]),
            body(20, &[RET]),
            body(30, &[RET]),
        ])
    };
    let routes = [
        route(1..2, vec![(funclets(&[30, 10]), UnwindEnd::Handler(20))]),
        route(0..1, vec![(funclets(&[10, 30]), UnwindEnd::Propagate)]),
    ];
    let first = build(instructions(), &routes);
    let second = build(instructions(), &routes);

    assert_eq!(first.hops, second.hops);
    assert_eq!(first.route_edge_blocks(), second.route_edge_blocks());
    first.assert_exact(&routes, &[20]);
}

#[test]
fn a_funclet_entry_that_is_split_after_the_build_still_has_its_instances() {
    // The funclet at 10 is a straight line of three instructions that a caller
    // splits in the middle after the build.
    let instructions = stream(&[main_code(2), body(10, &[NEXT, NEXT, RET]), body(20, &[RET])]);
    let routes = [
        route(0..1, vec![(funclets(&[10, 20]), UnwindEnd::Propagate)]),
        route(1..2, vec![(funclets(&[10]), UnwindEnd::Propagate)]),
    ];
    let mut graph = build_address_cfg_with_routes(
        instructions,
        &[],
        &routes,
        AddressCfgOptions::default(),
        wire,
    )
    .expect("valid");
    let head = graph.instruction_blocks[&10];
    let tail = graph
        .cfg
        .split_block_with_payload(head, 1, (10, 11, false, None));
    graph.instruction_blocks.insert(11, tail);
    let hops = graph.add_unwind_routes(&routes, wire).expect("valid");
    let built = Built { graph, hops };

    built.assert_exact(&routes, &[]);
    assert_eq!(built.instances_of(10).len(), 2);
    assert_eq!(built.instances_of(11).len(), 2);
}

#[test]
fn routes_that_protect_one_instruction_add_their_paths() {
    let instructions = stream(&[main_code(1), body(10, &[RET]), body(20, &[RET])]);
    let routes = [
        route(0..1, vec![(funclets(&[10]), UnwindEnd::Propagate)]),
        route(0..1, vec![(funclets(&[20]), UnwindEnd::Propagate)]),
    ];
    let built = build(instructions, &routes);

    assert_eq!(
        built.paths_from(built.block_of(0), &[]),
        BTreeSet::from([
            (vec![10], UnwindEnd::Propagate),
            (vec![20], UnwindEnd::Propagate)
        ])
    );
}
