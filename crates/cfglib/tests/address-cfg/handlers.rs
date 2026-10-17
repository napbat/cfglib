//! Exception tables: unwind edges, regions, handler references, handler
//! kinds, and what the exception model and extent recovery read from them.

use std::collections::BTreeSet;

use cfglib::{
    BlockId, ControlNode, ControlTree, DominatorTree, EhBlockKind, EhEdgeKind, EhModel, Handler,
    HandlerBody, HandlerKind, HandlerRef, RegionId, promote_handler_extents, verify,
};

use crate::harness::{
    FALL, NOT_TAKEN, Role, Settings, TAKEN, TryRange, UNWIND, Wire, assert_empty, build,
    build_with, catch,
};
use crate::isa::Dest::At;
use crate::isa::{Isa, Op};
use crate::oracles::reach;

fn blocks_of<I: Isa>(built: &crate::harness::Built<I>, positions: &[usize]) -> BTreeSet<BlockId> {
    positions.iter().map(|&p| built.block_of(p)).collect()
}

fn a_try_range_wires_unwind_edges_and_registers_one_region<I: Isa>() {
    let built = build_with::<I>(
        &[
            Op::Pure,
            Op::Work,
            Op::Work,
            Op::Return,
            Op::Work,
            Op::Return,
        ],
        vec![catch(1..3, 4)],
        Settings::default(),
    );
    // Each protected instruction leads its own block, and the range end and
    // the handler entry lead blocks too.
    assert_eq!(
        built.blocks(),
        vec![vec![0], vec![1], vec![2], vec![3], vec![4, 5]]
    );
    assert_eq!(
        built.successors(),
        vec![
            vec![(1, FALL)],
            vec![(2, FALL), (4, UNWIND)],
            vec![(3, FALL), (4, UNWIND)],
            vec![],
            vec![],
        ]
    );
    assert_eq!(
        built.roles()[1],
        vec![Role::Sequential, Role::Unwind { handler: 0 }]
    );
    let unwind = built.cfg().outgoing(built.block_of(1)).nth(1).unwrap();
    assert_eq!(
        *built.cfg().edge(unwind).payload(),
        Wire {
            source: built.raw(1),
            target: built.raw(4),
            kind: UNWIND,
            role: Role::Unwind { handler: 0 },
        }
    );

    let cfg = built.cfg();
    assert_eq!(cfg.regions().len(), 1);
    let region = &cfg.regions()[0];
    assert_eq!(region.id, RegionId::from_raw(0));
    assert_eq!(region.protected_blocks, blocks_of(&built, &[1, 2]));
    assert_eq!(region.parent, None);
    assert_eq!(
        region.handlers,
        [Handler {
            entry: built.block_of(4),
            body: HandlerBody::Unknown,
            kind: HandlerKind::Catch,
        }]
    );
    let handler = built.graph.handler_refs[0];
    assert_eq!(handler, HandlerRef::new(region.id, 0));
    assert_eq!(cfg.handler(handler), Some(&region.handlers[0]));
    assert_eq!(cfg.region(region.id), Some(region));
    let protecting = |position: usize| {
        cfg.protecting_region(built.block_of(position))
            .map(|r| r.id)
    };
    assert_eq!(protecting(1), Some(region.id));
    assert_eq!(protecting(2), Some(region.id));
    for outside in [0, 3, 4] {
        assert_eq!(
            protecting(outside),
            None,
            "instruction {outside} is outside"
        );
    }
    let dominators = DominatorTree::compute(cfg);
    assert_eq!(dominators.idom(built.block_of(4)), Some(built.block_of(1)));
}

fn a_non_throwing_protected_instruction_keeps_its_block_but_loses_its_unwind_edge<I: Isa>() {
    let built = build_with::<I>(
        &[
            Op::Work,
            Op::Pure,
            Op::Work,
            Op::Return,
            Op::Work,
            Op::Return,
        ],
        vec![catch(0..3, 4)],
        Settings::default(),
    );
    assert_eq!(
        built.blocks(),
        vec![vec![0], vec![1], vec![2], vec![3], vec![4, 5]]
    );
    assert_eq!(
        built.successors(),
        vec![
            vec![(1, FALL), (4, UNWIND)],
            vec![(2, FALL)],
            vec![(3, FALL), (4, UNWIND)],
            vec![],
            vec![],
        ]
    );
    // The pure block is still protected: pruning drops the edge, not the
    // coverage.
    assert_eq!(
        built.cfg().regions()[0].protected_blocks,
        blocks_of(&built, &[0, 1, 2])
    );
}

fn handlers_sharing_a_range_share_one_region_and_unwind_in_table_order<I: Isa>() {
    let built = build_with::<I>(
        &[
            Op::Work,
            Op::Work,
            Op::Return,
            Op::Work,
            Op::Return,
            Op::Work,
            Op::Return,
        ],
        vec![
            catch(0..2, 3),
            TryRange {
                protected: 0..2,
                entry: 5,
                kind: HandlerKind::CatchAll,
            },
        ],
        Settings::default(),
    );
    assert_eq!(
        built.blocks(),
        vec![vec![0], vec![1], vec![2], vec![3, 4], vec![5, 6]]
    );
    assert_eq!(
        built.successors(),
        vec![
            vec![(1, FALL), (3, UNWIND), (4, UNWIND)],
            vec![(2, FALL), (3, UNWIND), (4, UNWIND)],
            vec![],
            vec![],
            vec![],
        ]
    );
    assert_eq!(
        built.roles()[0],
        vec![
            Role::Sequential,
            Role::Unwind { handler: 0 },
            Role::Unwind { handler: 1 }
        ]
    );
    let regions = built.cfg().regions();
    assert_eq!(regions.len(), 1);
    assert_eq!(
        regions[0]
            .handlers
            .iter()
            .map(|h| h.kind)
            .collect::<Vec<_>>(),
        [HandlerKind::Catch, HandlerKind::CatchAll]
    );
    assert_eq!(
        built.graph.handler_refs,
        [
            HandlerRef::new(regions[0].id, 0),
            HandlerRef::new(regions[0].id, 1)
        ]
    );
}

fn a_range_ending_at_the_code_end_needs_no_end_instruction<I: Isa>() {
    let built = build_with::<I>(
        &[
            Op::BranchIf(At(3)),
            Op::Work,
            Op::Return,
            Op::Work,
            Op::Work,
        ],
        vec![catch(3..5, 1)],
        Settings::default(),
    );
    assert_eq!(built.blocks(), vec![vec![0], vec![1, 2], vec![3], vec![4]]);
    assert_eq!(
        built.successors(),
        vec![
            vec![(2, TAKEN), (1, NOT_TAKEN)],
            vec![],
            vec![(3, FALL), (1, UNWIND)],
            // The stream ends here, so only the unwind edge leaves.
            vec![(1, UNWIND)],
        ]
    );
    assert_eq!(
        built.cfg().regions()[0].protected_blocks,
        blocks_of(&built, &[3, 4])
    );
}

fn handler_kinds_pass_through_and_a_filter_is_patched_through_its_reference<I: Isa>() {
    let mut ops = vec![Op::Work; 4];
    ops.push(Op::Return);
    // Five handler bodies, then the filter predicate's own funclet at
    // instructions 15 and 16.
    for _ in 0..6 {
        ops.extend([Op::Work, Op::Return]);
    }
    let kinds = [
        (5, HandlerKind::Catch),
        (7, HandlerKind::CatchAll),
        (9, HandlerKind::Finally),
        (11, HandlerKind::Fault),
        // A filter names a block that does not exist at build time, so it is
        // registered as a catch and patched afterwards.
        (13, HandlerKind::Catch),
    ];
    let table = kinds
        .iter()
        .map(|&(entry, kind)| TryRange {
            protected: 0..4,
            entry,
            kind,
        })
        .collect();
    let mut built = build_with::<I>(&ops, table, Settings::default());
    assert_eq!(built.cfg().regions().len(), 1);
    let handlers = built.graph.handler_refs.clone();
    for (&(_, kind), &handler) in kinds.iter().zip(&handlers) {
        let stored = built.graph.cfg.handler(handler).unwrap();
        assert_eq!(stored.kind, kind);
        assert_eq!(stored.body, HandlerBody::Unknown);
        assert_eq!(stored.body.blocks(), None);
        assert!(!stored.body.is_known());
    }

    let filter_block = built.block_of(15);
    let patched = built.graph.cfg.handler_mut(handlers[4]).unwrap();
    patched.kind = HandlerKind::Filter { filter_block };
    assert_eq!(
        built.cfg().handler(handlers[4]).unwrap().kind,
        HandlerKind::Filter { filter_block }
    );

    let model = EhModel::compute(built.cfg());
    let entry = |position: usize| built.block_of(position);
    assert_eq!(model.landing_pads(), [entry(5), entry(7)]);
    assert_eq!(model.cleanup_blocks(), [entry(9), entry(11)]);
    assert_eq!(model.block_kind(entry(13)), EhBlockKind::CatchSwitch);
    assert_empty(&model.resume_blocks());
    assert_eq!(model.block_kind(built.block_of(0)), EhBlockKind::Normal);
    assert_eq!(model.block_bound(), built.cfg().block_bound());
    let region = built.cfg().regions()[0].id;
    for (slot, &(position, _)) in kinds.iter().enumerate() {
        assert_eq!(
            model.handlers(entry(position)),
            [HandlerRef::new(region, slot)]
        );
        assert_eq!(
            model.protected_by(entry(position)),
            &blocks_of(&built, &[0, 1, 2, 3])
        );
    }
}

fn the_exception_model_reads_one_unwind_edge_per_protected_instruction_and_handler<I: Isa>() {
    let built = build_with::<I>(
        &[
            Op::Work,
            Op::Work,
            Op::Return,
            Op::Work,
            Op::Return,
            Op::Work,
            Op::Return,
        ],
        vec![
            catch(0..2, 3),
            TryRange {
                protected: 0..2,
                entry: 5,
                kind: HandlerKind::Finally,
            },
        ],
        Settings::default(),
    );
    let cfg = built.cfg();
    let model = EhModel::compute(cfg);
    assert_eq!(model.eh_edges().len(), 4);
    for edge in model.eh_edges() {
        assert_eq!(edge.kind, EhEdgeKind::Unwind);
        assert!(edge.is_unwind);
        let stored = cfg.edge(edge.edge_id);
        assert_eq!((stored.source(), stored.target()), (edge.from, edge.to));
        assert_eq!(stored.kind(), UNWIND);
    }
    assert_eq!(model.block_kind(built.block_of(3)), EhBlockKind::LandingPad);
    // Region metadata is authoritative: the finally stays a cleanup although
    // an unwind edge also reaches it.
    assert_eq!(model.block_kind(built.block_of(5)), EhBlockKind::Cleanup);
    assert_eq!(
        model.protected_by(built.block_of(3)),
        &blocks_of(&built, &[0, 1])
    );
    assert_eq!(
        model.protected_by(built.block_of(5)),
        &blocks_of(&built, &[0, 1])
    );
}

fn a_probe_without_any_table_still_makes_its_alternate_target_a_landing_pad<I: Isa>() {
    let built = build::<I>(&[Op::Work, Op::Probe(At(3)), Op::Work, Op::Return]);
    assert_eq!(
        built.successors(),
        vec![
            vec![(1, FALL)],
            vec![(3, UNWIND), (2, FALL)],
            vec![(3, FALL)],
            vec![],
        ]
    );
    let model = EhModel::compute(built.cfg());
    assert_eq!(model.eh_edges().len(), 1);
    let edge = &model.eh_edges()[0];
    assert_eq!((edge.from, edge.to), (built.block_of(1), built.block_of(3)));
    assert_eq!(edge.kind, EhEdgeKind::Unwind);
    assert_eq!(model.block_kind(built.block_of(3)), EhBlockKind::LandingPad);
    assert_eq!(
        model.protected_by(built.block_of(3)),
        &blocks_of(&built, &[1])
    );
    assert_empty(model.handlers(built.block_of(3)));
    assert_empty(built.cfg().regions());
}

fn promoting_extents_claims_exactly_the_blocks_the_handler_entry_dominates<I: Isa>() {
    let mut built = build_with::<I>(
        &[
            Op::Work,
            Op::Return,
            Op::BranchIf(At(4)),
            Op::Work,
            Op::Return,
        ],
        vec![catch(0..1, 2)],
        Settings::default(),
    );
    assert_eq!(
        built.successors(),
        vec![
            vec![(1, FALL), (2, UNWIND)],
            vec![],
            vec![(4, TAKEN), (3, NOT_TAKEN)],
            vec![(4, FALL)],
            vec![],
        ]
    );
    let region = built.cfg().regions()[0].id;
    let (before, report) = ControlTree::compute_with_report(built.cfg());
    assert_eq!(report.unstructured_regions, [region]);
    assert!(
        !before
            .preorder()
            .any(|n| matches!(before.node(n), ControlNode::TryCatch { .. }))
    );

    assert_eq!(promote_handler_extents(&mut built.graph.cfg), 1);
    let handler = built.graph.handler_refs[0];
    let body = built.cfg().handler(handler).unwrap().body.clone();
    // Dominated by the entry: reachable, but not once the entry is removed.
    let entry = built.block_of(2);
    let reachable = reach(built.cfg(), &[built.cfg().entry()], None);
    let without = reach(built.cfg(), &[built.cfg().entry()], Some(entry));
    let dominated: BTreeSet<BlockId> = reachable.difference(&without).copied().collect();
    assert_eq!(body, HandlerBody::Known(dominated));
    assert_eq!(body.blocks().unwrap(), &blocks_of(&built, &[2, 3, 4]));
    assert!(verify(built.cfg()).is_ok());

    let (after, report) = ControlTree::compute_with_report(built.cfg());
    assert_empty(&report.unstructured_regions);
    let arms: Vec<_> = after
        .preorder()
        .filter_map(|n| match *after.node(n) {
            ControlNode::TryCatch { handlers, .. } => Some(after.handlers(handlers).to_vec()),
            _ => None,
        })
        .collect();
    assert_eq!(arms.len(), 1, "the region structures as one try");
    assert_eq!(arms[0].len(), 1);
    assert_eq!(arms[0][0].handler, handler);
    assert_eq!(arms[0][0].entry, entry);
    assert_eq!(arms[0][0].kind, HandlerKind::Catch);
}

fn an_unreachable_handler_entry_stays_unknown_when_extents_are_promoted<I: Isa>() {
    // The only protected instruction cannot fault, so nothing unwinds into
    // the handler and its entry is unreachable.
    let mut built = build_with::<I>(
        &[Op::Pure, Op::Return, Op::Work, Op::Return],
        vec![catch(0..1, 2)],
        Settings::default(),
    );
    assert_eq!(built.successors(), vec![vec![(1, FALL)], vec![], vec![]]);
    assert_eq!(promote_handler_extents(&mut built.graph.cfg), 0);
    let handler = built.graph.handler_refs[0];
    assert_eq!(
        built.cfg().handler(handler).unwrap().body,
        HandlerBody::Unknown
    );
}

over_both_isas!(
    a_try_range_wires_unwind_edges_and_registers_one_region,
    a_non_throwing_protected_instruction_keeps_its_block_but_loses_its_unwind_edge,
    handlers_sharing_a_range_share_one_region_and_unwind_in_table_order,
    a_range_ending_at_the_code_end_needs_no_end_instruction,
    handler_kinds_pass_through_and_a_filter_is_patched_through_its_reference,
    the_exception_model_reads_one_unwind_edge_per_protected_instruction_and_handler,
    a_probe_without_any_table_still_makes_its_alternate_target_a_landing_pad,
    promoting_extents_claims_exactly_the_blocks_the_handler_entry_dominates,
    an_unreachable_handler_entry_stays_unknown_when_extents_are_promoted,
);
