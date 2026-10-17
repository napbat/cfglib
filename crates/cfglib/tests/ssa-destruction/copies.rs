//! `eliminate_phis` and `copies_by_predecessor`: which copies a phi
//! produces, and how they group.

use std::collections::BTreeSet;

use cfglib::{
    BlockId, Cfg, EdgeKind, PhiCopy, copies_by_predecessor, eliminate_phis, split_critical_edges,
};

use crate::machine::{Insn, Reg};
use crate::programs::{
    analyze, departure, diamond, loop_with_early_continue, lost_copy_loop, nested_loops, swap_loop,
    two_phi_diamond, wire,
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
        from: departure(BlockId::from_index(arm), 1),
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
        let (point, group) = (edge.from, &edge.copies);
        assert_eq!(group.len(), 2, "both phis copy at {point}");
        assert!(group.iter().all(|copy| copy.from == point));
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
            let point = edge.from;
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
        .map(|edge| edge.from.block)
        .filter(|block| cfg.successors(*block).count() > 1)
        .collect();
    assert_eq!(
        branching.len(),
        1,
        "the continue edge carries copies from a branching block"
    );
    assert_eq!(split_critical_edges(&mut cfg), 1);
}
