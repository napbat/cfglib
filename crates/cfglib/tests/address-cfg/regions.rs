//! Region ordering and nesting: enclosing ranges register first, nested ones
//! name their innermost parent, and equal or overlapping ranges stay peers.

use std::collections::BTreeSet;

use cfglib::{BlockId, EhBlockKind, EhModel, HandlerKind, HandlerRef, RegionId};

use crate::harness::{Built, FALL, Settings, TryRange, UNWIND, build_with, catch};
use crate::isa::{Isa, Op};

/// `count` faulting operations, a return, then a faulting handler body that
/// returns, once per entry of `handler_bodies`.
fn guarded_code(count: usize, handler_bodies: usize) -> Vec<Op> {
    let mut ops = vec![Op::Work; count];
    ops.push(Op::Return);
    for _ in 0..handler_bodies {
        ops.extend([Op::Work, Op::Return]);
    }
    ops
}

fn blocks_of<I: Isa>(built: &Built<I>, positions: &[usize]) -> BTreeSet<BlockId> {
    positions.iter().map(|&p| built.block_of(p)).collect()
}

fn nested_ranges_register_the_enclosing_region_first_whatever_the_table_order<I: Isa>() {
    // Table order is inner, outer, middle; spans are 2, 6, and 4 operations.
    let built = build_with::<I>(
        &guarded_code(6, 3),
        vec![
            catch(2..4, 7),
            TryRange {
                protected: 0..6,
                entry: 9,
                kind: HandlerKind::CatchAll,
            },
            TryRange {
                protected: 1..5,
                entry: 11,
                kind: HandlerKind::Finally,
            },
        ],
        Settings::default(),
    );
    assert_eq!(
        built.blocks(),
        vec![
            vec![0],
            vec![1],
            vec![2],
            vec![3],
            vec![4],
            vec![5],
            vec![6],
            vec![7, 8],
            vec![9, 10],
            vec![11, 12],
        ]
    );
    let regions = built.cfg().regions();
    let [outer, middle, inner] = [0, 1, 2].map(RegionId::from_index);
    assert_eq!(
        regions.iter().map(|region| region.id).collect::<Vec<_>>(),
        [outer, middle, inner]
    );
    assert_eq!(
        regions
            .iter()
            .map(|region| region.parent)
            .collect::<Vec<_>>(),
        [None, Some(outer), Some(middle)]
    );
    // Handler references follow the table, not the region order.
    assert_eq!(
        built.graph.handler_refs,
        [
            HandlerRef::new(inner, 0),
            HandlerRef::new(outer, 0),
            HandlerRef::new(middle, 0)
        ]
    );
    let cfg = built.cfg();
    let innermost = |position: usize| {
        cfg.protecting_region(built.block_of(position))
            .map(|r| r.id)
    };
    assert_eq!(
        (0..7).map(innermost).collect::<Vec<_>>(),
        [
            Some(outer),
            Some(middle),
            Some(inner),
            Some(inner),
            Some(middle),
            Some(outer),
            None
        ]
    );
    // A block unwinds to the handlers of every range covering it, in table
    // order, after its normal edge.
    let successors = built.successors();
    assert_eq!(successors[0], [(1, FALL), (8, UNWIND)]);
    assert_eq!(successors[1], [(2, FALL), (8, UNWIND), (9, UNWIND)]);
    assert_eq!(
        successors[2],
        [(3, FALL), (7, UNWIND), (8, UNWIND), (9, UNWIND)]
    );
    assert_eq!(
        successors[3],
        [(4, FALL), (7, UNWIND), (8, UNWIND), (9, UNWIND)]
    );
    assert_eq!(successors[4], [(5, FALL), (8, UNWIND), (9, UNWIND)]);
    assert_eq!(successors[5], [(6, FALL), (8, UNWIND)]);

    let model = EhModel::compute(cfg);
    let entry = |position| built.block_of(position);
    assert_eq!(model.handlers(entry(7)), [HandlerRef::new(inner, 0)]);
    assert_eq!(model.handlers(entry(9)), [HandlerRef::new(outer, 0)]);
    assert_eq!(model.handlers(entry(11)), [HandlerRef::new(middle, 0)]);
    assert_eq!(model.protected_by(entry(7)), &blocks_of(&built, &[2, 3]));
    assert_eq!(
        model.protected_by(entry(9)),
        &blocks_of(&built, &[0, 1, 2, 3, 4, 5])
    );
    assert_eq!(
        model.protected_by(entry(11)),
        &blocks_of(&built, &[1, 2, 3, 4])
    );
    assert_eq!(model.block_kind(entry(11)), EhBlockKind::Cleanup);
}

fn ranges_of_equal_span_stay_peers_in_table_order<I: Isa>() {
    let built = build_with::<I>(
        &guarded_code(4, 2),
        vec![catch(2..4, 5), catch(0..2, 7)],
        Settings::default(),
    );
    let regions = built.cfg().regions();
    assert_eq!(regions.len(), 2);
    assert_eq!(regions[0].protected_blocks, blocks_of(&built, &[2, 3]));
    assert_eq!(regions[1].protected_blocks, blocks_of(&built, &[0, 1]));
    assert_eq!(regions[0].parent, None);
    assert_eq!(regions[1].parent, None);
    assert_eq!(
        built.graph.handler_refs,
        [
            HandlerRef::new(regions[0].id, 0),
            HandlerRef::new(regions[1].id, 0)
        ]
    );
}

fn partly_overlapping_ranges_are_peers_and_the_later_region_is_innermost<I: Isa>() {
    let built = build_with::<I>(
        &guarded_code(5, 2),
        vec![catch(0..4, 6), catch(2..5, 8)],
        Settings::default(),
    );
    let regions = built.cfg().regions();
    // Neither range contains the other, so neither is a parent.
    assert_eq!(
        regions[0].protected_blocks,
        blocks_of(&built, &[0, 1, 2, 3])
    );
    assert_eq!(regions[1].protected_blocks, blocks_of(&built, &[2, 3, 4]));
    assert_eq!(regions[1].parent, None);
    let cfg = built.cfg();
    let innermost = |position: usize| {
        cfg.protecting_region(built.block_of(position))
            .map(|r| r.id)
    };
    assert_eq!(innermost(0), Some(regions[0].id));
    assert_eq!(
        innermost(2),
        Some(regions[1].id),
        "the later region wins shared blocks"
    );
    assert_eq!(innermost(4), Some(regions[1].id));
    let successors = built.successors();
    assert_eq!(successors[0], [(1, FALL), (6, UNWIND)]);
    assert_eq!(successors[2], [(3, FALL), (6, UNWIND), (7, UNWIND)]);
    assert_eq!(successors[4], [(5, FALL), (7, UNWIND)]);
}

fn a_range_sharing_its_start_with_a_larger_one_nests_inside_it<I: Isa>() {
    let built = build_with::<I>(
        &guarded_code(4, 2),
        vec![catch(0..2, 5), catch(0..4, 7)],
        Settings::default(),
    );
    let regions = built.cfg().regions();
    assert_eq!(
        regions[0].protected_blocks,
        blocks_of(&built, &[0, 1, 2, 3])
    );
    assert_eq!(regions[1].protected_blocks, blocks_of(&built, &[0, 1]));
    assert_eq!(regions[0].parent, None);
    assert_eq!(regions[1].parent, Some(regions[0].id));
    assert_eq!(
        built.graph.handler_refs,
        [
            HandlerRef::new(regions[1].id, 0),
            HandlerRef::new(regions[0].id, 0)
        ]
    );
}

fn three_levels_of_nesting_name_each_innermost_enclosing_region<I: Isa>() {
    // A chain of strictly shrinking ranges registered innermost first.
    let built = build_with::<I>(
        &guarded_code(8, 4),
        vec![
            catch(3..5, 9),
            catch(2..6, 11),
            catch(1..7, 13),
            catch(0..8, 15),
        ],
        Settings::default(),
    );
    let regions = built.cfg().regions();
    assert_eq!(
        regions
            .iter()
            .map(|region| region.parent)
            .collect::<Vec<_>>(),
        [
            None,
            Some(regions[0].id),
            Some(regions[1].id),
            Some(regions[2].id)
        ]
    );
    assert_eq!(
        regions
            .iter()
            .map(|region| region.protected_blocks.len())
            .collect::<Vec<_>>(),
        [8, 6, 4, 2]
    );
}

over_both_isas!(
    nested_ranges_register_the_enclosing_region_first_whatever_the_table_order,
    ranges_of_equal_span_stay_peers_in_table_order,
    partly_overlapping_ranges_are_peers_and_the_later_region_is_innermost,
    a_range_sharing_its_start_with_a_larger_one_nests_inside_it,
    three_levels_of_nesting_name_each_innermost_enclosing_region,
);
