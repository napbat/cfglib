//! `eliminate_phis` and `copies_by_predecessor`: which copies a phi
//! produces, and how they group.

use std::collections::BTreeSet;

use cfglib::{
    BlockId, Cfg, EdgeKind, PhiCopy, PhiCopyOrigin, SsaValue, copies_by_predecessor,
    eliminate_phis, split_critical_edges,
};

use crate::machine::{Insn, Reg};
use crate::programs::{
    analyze, departure, departure_of, diamond, entry_swap_loop, loop_with_early_continue,
    lost_copy_loop, nested_loops, swap_loop, two_phi_diamond, wire,
};

#[test]
fn a_diamond_phi_lowers_to_one_copy_per_predecessor_from_that_arms_value() {
    let program = diamond();
    let ssa = analyze(&program.cfg);
    let merge = program.cfg.block_ids().last().expect("the merge block");
    let [phi] = ssa.block(merge).phis.as_slice() else {
        panic!("one register merges at the join");
    };
    let arm_value = |index: usize| {
        let arm = BlockId::from_index(index);
        ssa.block(arm).instructions[0].defs()[0].clone()
    };

    let copies = eliminate_phis(&ssa);
    let expected = |arm: usize| PhiCopy {
        from: PhiCopyOrigin::Departure(departure(BlockId::from_index(arm), 1)),
        to_block: merge,
        destination: phi.result.clone(),
        source: arm_value(arm),
    };
    assert_eq!(copies.len(), 2);
    assert!(copies.contains(&expected(1)));
    assert!(copies.contains(&expected(2)));
    assert_ne!(arm_value(1), arm_value(2));
}

#[test]
fn a_form_without_phis_needs_no_copies() {
    let mut cfg = Cfg::<Insn<Reg>>::new();
    let next = cfg.new_block();
    let entry = cfg.entry();
    wire(&mut cfg, entry, vec![Insn::li("x", 1)]);
    wire(&mut cfg, next, vec![Insn::ret("x")]);
    cfg.add_edge(cfg.entry(), next, EdgeKind::Fallthrough);
    let ssa = analyze(&cfg);
    assert_eq!(eliminate_phis(&ssa), Vec::new());
    assert_eq!(copies_by_predecessor::<Reg>(&[]), Vec::new());
}

#[test]
fn copies_of_one_departure_point_form_one_group_across_all_phis_of_the_join() {
    let program = two_phi_diamond();
    let ssa = analyze(&program.cfg);
    let copies = eliminate_phis(&ssa);
    assert_eq!(copies.len(), 4, "two registers over two predecessors");

    let groups = copies_by_predecessor(&copies);
    assert_eq!(groups.len(), 2);
    for edge in &groups {
        let (origin, group) = (edge.from, &edge.copies);
        assert_eq!(group.len(), 2, "both phis copy at {origin:?}");
        assert!(group.iter().all(|copy| copy.from == origin));
        let destinations: BTreeSet<_> =
            group.iter().map(|copy| copy.destination.variable).collect();
        assert_eq!(destinations, BTreeSet::from([Reg("x"), Reg("y")]));
    }
    // Grouping neither loses nor invents copies.
    let regrouped: usize = groups.iter().map(|group| group.copies.len()).sum();
    assert_eq!(regrouped, copies.len());
}

#[test]
fn every_copy_group_after_critical_edge_splitting_leaves_a_block_with_one_successor() {
    for program in [
        diamond(),
        two_phi_diamond(),
        swap_loop(),
        lost_copy_loop(),
        loop_with_early_continue(),
        nested_loops(),
    ] {
        let mut cfg = program.cfg;
        split_critical_edges(&mut cfg);
        let ssa = analyze(&cfg);
        for edge in copies_by_predecessor(&eliminate_phis(&ssa)) {
            let point = departure_of(edge.from);
            assert_eq!(
                cfg.successors(point.block).count(),
                1,
                "copies at the end of {:?} would run on a path that does not need them",
                point.block
            );
        }
    }
}

#[test]
fn a_critical_edge_into_a_phi_block_is_why_splitting_is_needed() {
    let mut cfg = loop_with_early_continue().cfg;
    let ssa = analyze(&cfg);
    let branching: Vec<BlockId> = copies_by_predecessor(&eliminate_phis(&ssa))
        .into_iter()
        .map(|edge| departure_of(edge.from).block)
        .filter(|block| cfg.successors(*block).count() > 1)
        .collect();
    assert_eq!(
        branching.len(),
        1,
        "the continue edge carries copies from a branching block"
    );
    assert_eq!(split_critical_edges(&mut cfg), 1);
}

/// The registers `entry_swap_loop` defines somewhere in its loop, so the
/// entry merges each of them.
fn registers_the_loop_defines() -> BTreeSet<Reg> {
    ["stop", "t", "x", "y", "one", "i"].map(Reg).into()
}

#[test]
fn a_loop_headed_by_the_entry_copies_each_live_in_once_at_the_start_and_each_carried_value_on_the_back_edge()
 {
    let program = entry_swap_loop();
    let ssa = analyze(&program.cfg);
    let entry = program.cfg.entry();
    let body = BlockId::from_index(1);
    let copies = eliminate_phis(&ssa);

    let start: Vec<_> = copies
        .iter()
        .filter(|copy| copy.from == PhiCopyOrigin::FunctionStart)
        .collect();
    let on_back_edge: Vec<_> = copies
        .iter()
        .filter(|copy| copy.from == PhiCopyOrigin::Departure(departure(body, 5)))
        .collect();
    assert_eq!(start.len() + on_back_edge.len(), copies.len());
    assert_eq!(start.len(), registers_the_loop_defines().len());
    assert_eq!(on_back_edge.len(), start.len());

    let merged: BTreeSet<_> = start.iter().map(|copy| copy.destination.variable).collect();
    assert_eq!(merged, registers_the_loop_defines());
    for copy in &start {
        assert_eq!(copy.to_block, entry);
        assert_eq!(
            copy.source,
            SsaValue::live_in(copy.destination.variable),
            "the function starts with each register's live-in"
        );
        assert!(
            ssa.block(entry)
                .phis
                .iter()
                .any(|phi| phi.result == copy.destination),
            "the copy defines a phi of the entry"
        );
    }
    for copy in &on_back_edge {
        assert_eq!(copy.to_block, entry);
        assert!(
            !copy.source.is_live_in(),
            "every register the loop defines has a definition by the end of the body"
        );
    }
}

#[test]
fn the_function_start_is_its_own_group_ahead_of_the_edge_groups() {
    let program = entry_swap_loop();
    let ssa = analyze(&program.cfg);
    let copies = eliminate_phis(&ssa);
    let groups = copies_by_predecessor(&copies);

    let origins: Vec<_> = groups.iter().map(|group| group.from).collect();
    assert_eq!(
        origins,
        [
            PhiCopyOrigin::FunctionStart,
            PhiCopyOrigin::Departure(departure(BlockId::from_index(1), 5)),
        ],
        "both groups enter the entry, and only the second is an edge"
    );
    for group in &groups {
        let destinations: BTreeSet<_> = group
            .copies
            .iter()
            .map(|copy| copy.destination.variable)
            .collect();
        assert_eq!(
            destinations,
            registers_the_loop_defines(),
            "a parallel assignment of every merged register: {:?}",
            group.from
        );
    }
    let start = &groups[0];
    assert!(
        start.copies.iter().all(|copy| start
            .copies
            .iter()
            .all(|other| other.source != copy.destination)),
        "no start copy reads another's destination, so no temporary is needed"
    );
}
