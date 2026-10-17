//! Integration tests: structural CFG comparison.

use cfglib::{BlockId, Cfg, CfgDiff, EdgeKind};

/// `entry -> head -> body -> latch`, with `latch` flowing back into `target`.
fn loop_into(target: fn(&[BlockId; 3]) -> BlockId) -> Cfg<u32> {
    let mut cfg = Cfg::new();
    let blocks = [cfg.new_block(), cfg.new_block(), cfg.new_block()];
    cfg.add_edge(cfg.entry(), blocks[0], EdgeKind::Fallthrough);
    cfg.add_edge(blocks[0], blocks[1], EdgeKind::Fallthrough);
    cfg.add_edge(blocks[1], blocks[2], EdgeKind::Fallthrough);
    cfg.add_edge(blocks[2], target(&blocks), EdgeKind::Back);
    cfg
}

#[test]
fn an_identical_wiring_is_identical() {
    let left = loop_into(|blocks| blocks[0]);
    let right = loop_into(|blocks| blocks[0]);

    let diff = CfgDiff::compute(&left, &right);
    assert!(diff.is_identical());
    assert!((diff.match_ratio() - 1.0).abs() < f64::EPSILON);
}

#[test]
fn matching_blocks_with_the_same_degrees_does_not_make_different_wiring_identical() {
    // Three-cycle `head -> body -> latch -> head` against a tail that ends in
    // the two-cycle `body <-> latch`. Every block has one successor, and the
    // in-degrees are {2, 1, 1} on both sides, so a fingerprint match alone
    // pairs the blocks up without a single left-only or right-only block.
    let left = loop_into(|blocks| blocks[0]);
    let right = loop_into(|blocks| blocks[1]);

    let diff = CfgDiff::compute(&left, &right);
    assert!(diff.left_only.is_empty() && diff.right_only.is_empty());
    assert!(
        !diff.is_identical(),
        "the loop closes on a different block: {diff:?}"
    );
}

#[test]
fn unreachable_blocks_are_compared_like_any_other_block() {
    let mut left = loop_into(|blocks| blocks[0]);
    let mut right = loop_into(|blocks| blocks[0]);
    let orphan = left.new_block();
    right.new_block();

    let diff = CfgDiff::compute(&left, &right);
    assert!(diff.is_identical());
    assert!(
        (diff.match_ratio() - 1.0).abs() < f64::EPSILON,
        "identical CFGs match completely: {diff:?}"
    );

    right.new_block();
    let diff = CfgDiff::compute(&left, &right);
    assert!(!diff.is_identical());
    assert_eq!(diff.right_only.len(), 1, "{diff:?}");
    assert!(diff.left_only.is_empty(), "{diff:?}");
    assert!(
        diff.matched.iter().any(|pair| pair.left == orphan),
        "the unreachable block still pairs with an unreachable block: {diff:?}"
    );
}

#[test]
fn a_changed_instruction_count_is_a_match_flagged_as_changed() {
    let left = loop_into(|blocks| blocks[0]);
    let mut right = loop_into(|blocks| blocks[0]);
    let body = BlockId::from_index(2);
    right.block_mut(body).instructions_mut().push(7);

    let diff = CfgDiff::compute(&left, &right);
    assert!(diff.left_only.is_empty() && diff.right_only.is_empty());
    let changed: Vec<_> = diff
        .matched
        .iter()
        .filter(|pair| pair.instruction_count_changed)
        .map(|pair| (pair.left, pair.right))
        .collect();
    assert_eq!(changed, [(body, body)]);
    assert!(!diff.is_identical());
}

#[test]
fn the_order_edges_were_added_in_does_not_matter() {
    // A conditional whose arms differ below the branch, with the arms added
    // in opposite orders. `then_tail` and `else_arm` are the same kind of
    // block (one predecessor, no successor), so only the branch kinds say
    // which of them belongs to which arm.
    let build = |true_first: bool| {
        let mut cfg = Cfg::<u32>::new();
        let then_arm = cfg.new_block();
        let then_tail = cfg.new_block();
        let else_arm = cfg.new_block();
        cfg.add_edge(then_arm, then_tail, EdgeKind::Fallthrough);
        let entry = cfg.entry();
        if true_first {
            cfg.add_edge(entry, then_arm, EdgeKind::ConditionalTrue);
            cfg.add_edge(entry, else_arm, EdgeKind::ConditionalFalse);
        } else {
            cfg.add_edge(entry, else_arm, EdgeKind::ConditionalFalse);
            cfg.add_edge(entry, then_arm, EdgeKind::ConditionalTrue);
        }
        cfg
    };

    let diff = CfgDiff::compute(&build(true), &build(false));
    assert!(diff.is_identical(), "{diff:?}");
}
