//! Block recovery from branch targets, terminators, and the entry address:
//! straight lines, conditionals, block splits, merges, loops, and entries.

use cfglib::{BlockId, ControlNode, ControlTree, DominatorTree, EdgeKind};

use crate::harness::{
    FALL, JUMP, NOT_TAKEN, Role, Settings, TAKEN, assert_empty, build, build_with,
};
use crate::isa::Dest::At;
use crate::isa::{Isa, Op};
fn straight_code_is_one_block_without_edges<I: Isa>() {
    let built = build::<I>(&[Op::Work, Op::Work, Op::Pure, Op::Return]);
    assert_eq!(built.blocks(), vec![vec![0, 1, 2, 3]]);
    assert_eq!(built.successors(), vec![vec![]]);
    assert_eq!(built.cfg().entry(), built.block_of(0));
    assert_eq!(
        built.cfg().exit_blocks().collect::<Vec<_>>(),
        [built.cfg().entry()]
    );
    assert_empty(&built.graph.unresolved_transfers);
    assert_empty(&built.graph.calls);
    assert_empty(&built.graph.handler_refs);
    assert_empty(built.cfg().regions());
}

fn a_taken_conditional_splits_the_run_into_three_blocks<I: Isa>() {
    let built = build::<I>(&[
        Op::Work,
        Op::BranchIf(At(4)),
        Op::Work,
        Op::Work,
        Op::Return,
    ]);
    assert_eq!(built.blocks(), vec![vec![0, 1], vec![2, 3], vec![4]]);
    // The taken edge precedes the not-taken edge; the middle block runs into
    // the target through a sequential edge, not a branch.
    assert_eq!(
        built.successors(),
        vec![vec![(2, TAKEN), (1, NOT_TAKEN)], vec![(2, FALL)], vec![]]
    );
    assert_eq!(
        built.roles(),
        vec![
            vec![Role::Taken, Role::FallThrough],
            vec![Role::Sequential],
            vec![]
        ]
    );
}

fn a_branch_into_the_middle_of_a_run_splits_the_block<I: Isa>() {
    let built = build::<I>(&[
        Op::Work,
        Op::Work,
        Op::Work,
        Op::Work,
        Op::BranchIf(At(2)),
        Op::Return,
    ]);
    // Instruction 2 is a leader only because a later branch targets it.
    assert_eq!(built.blocks(), vec![vec![0, 1], vec![2, 3, 4], vec![5]]);
    assert_eq!(
        built.successors(),
        vec![vec![(1, FALL)], vec![(1, TAKEN), (2, NOT_TAKEN)], vec![]]
    );
    let middle = built.block_of(2);
    assert_eq!(built.block_of(4), middle);
    assert!(built.cfg().successors(middle).any(|next| next == middle));
}

fn a_merge_block_is_entered_by_a_jump_and_by_fallthrough<I: Isa>() {
    let built = build::<I>(&[
        Op::BranchIf(At(3)),
        Op::Work,
        Op::Jump(At(4)),
        Op::Work,
        Op::Return,
    ]);
    assert_eq!(built.blocks(), vec![vec![0], vec![1, 2], vec![3], vec![4]]);
    assert_eq!(
        built.successors(),
        vec![
            vec![(2, TAKEN), (1, NOT_TAKEN)],
            vec![(3, JUMP)],
            vec![(3, FALL)],
            vec![]
        ]
    );
    let merge = built.block_of(4);
    let mut entering: Vec<EdgeKind> = built
        .cfg()
        .incoming(merge)
        .map(|edge| built.cfg().edge(edge).kind())
        .collect();
    entering.sort_by_key(|kind| kind.name());
    assert_eq!(entering, [FALL, JUMP]);
    let dominators = DominatorTree::compute(built.cfg());
    assert_eq!(dominators.idom(merge), Some(built.block_of(0)));
}

fn a_backward_jump_to_the_entry_is_a_self_loop<I: Isa>() {
    let built = build::<I>(&[Op::Work, Op::Jump(At(0))]);
    assert_eq!(built.blocks(), vec![vec![0, 1]]);
    assert_eq!(built.successors(), vec![vec![(0, JUMP)]]);
    let entry = built.cfg().entry();
    assert_eq!(built.cfg().predecessors(entry).collect::<Vec<_>>(), [entry]);
    assert_eq!(built.cfg().exit_blocks().count(), 0);
    let tree = ControlTree::compute(built.cfg());
    assert!(
        tree.preorder().any(
            |node| matches!(tree.node(node), ControlNode::Loop { header, .. } if *header == entry)
        ),
        "the backward jump structures as a loop on the entry block"
    );
}

fn a_conditional_to_its_own_continuation_keeps_parallel_edges<I: Isa>() {
    let built = build::<I>(&[Op::BranchIf(At(1)), Op::Return]);
    assert_eq!(built.blocks(), vec![vec![0], vec![1]]);
    assert_eq!(
        built.successors(),
        vec![vec![(1, TAKEN), (1, NOT_TAKEN)], vec![]]
    );
    assert_eq!(built.cfg().edge_count(), 2);
}

fn code_after_a_terminator_forms_an_unreachable_block<I: Isa>() {
    let built = build::<I>(&[Op::Return, Op::Work, Op::Return]);
    assert_eq!(built.blocks(), vec![vec![0], vec![1, 2]]);
    assert_eq!(built.cfg().edge_count(), 0);
    let dominators = DominatorTree::compute(built.cfg());
    assert!(dominators.is_reachable(built.block_of(0)));
    assert!(!dominators.is_reachable(built.block_of(1)));
    // The block is built, but structuring drops code nothing reaches.
    let tree = ControlTree::compute(built.cfg());
    let mut structured = 0;
    tree.for_each_instruction(built.cfg(), |_, _| structured += 1);
    assert_eq!(structured, 1);
}

fn continuation_past_the_last_instruction_adds_no_edge<I: Isa>() {
    let falling = build::<I>(&[Op::Work, Op::Work]);
    assert_eq!(falling.blocks(), vec![vec![0, 1]]);
    assert_eq!(falling.cfg().edge_count(), 0);
    assert_empty(&falling.graph.unresolved_transfers);

    let branching = build::<I>(&[Op::Work, Op::BranchIf(At(0))]);
    assert_eq!(branching.successors(), vec![vec![(0, TAKEN)]]);
    assert_empty(&branching.graph.unresolved_transfers);
}

fn an_explicit_entry_makes_lower_instructions_a_separate_function_part<I: Isa>() {
    let ops = [
        Op::Work,
        Op::Return,
        Op::Work,
        Op::BranchIf(At(0)),
        Op::Return,
    ];
    let default_entry = build::<I>(&ops);
    assert_eq!(default_entry.cfg().entry(), default_entry.block_of(0));

    let explicit = build_with::<I>(
        &ops,
        Vec::new(),
        Settings {
            entry: Some(2),
            ..Settings::default()
        },
    );
    // The entry changes which block starts the function, not the blocks.
    assert_eq!(explicit.blocks(), default_entry.blocks());
    assert_eq!(explicit.successors(), default_entry.successors());
    assert_eq!(explicit.blocks(), vec![vec![0, 1], vec![2, 3], vec![4]]);
    assert_eq!(
        explicit.successors(),
        vec![vec![], vec![(0, TAKEN), (2, NOT_TAKEN)], vec![]]
    );
    assert_eq!(explicit.cfg().entry(), explicit.block_of(2));
    assert_ne!(explicit.cfg().entry(), BlockId::from_index(0));
    // The backward target below the entry is reachable from it.
    let dominators = DominatorTree::compute(explicit.cfg());
    assert!(dominators.is_reachable(explicit.block_of(0)));
    // Without the explicit entry the same blocks are dead code.
    let dominators = DominatorTree::compute(default_entry.cfg());
    assert!(!dominators.is_reachable(default_entry.block_of(2)));
}

fn an_entry_that_is_already_a_leader_adds_no_block<I: Isa>() {
    let ops = [Op::Work, Op::Work, Op::Return];
    let at_start = build_with::<I>(
        &ops,
        Vec::new(),
        Settings {
            entry: Some(0),
            ..Settings::default()
        },
    );
    assert_eq!(at_start.blocks(), vec![vec![0, 1, 2]]);
    let mid_entry = build_with::<I>(
        &ops,
        Vec::new(),
        Settings {
            entry: Some(1),
            ..Settings::default()
        },
    );
    // An entry in the middle of a run still leads a block of its own.
    assert_eq!(mid_entry.blocks(), vec![vec![0], vec![1, 2]]);
    assert_eq!(mid_entry.successors(), vec![vec![(1, FALL)], vec![]]);
    assert_eq!(mid_entry.cfg().entry(), mid_entry.block_of(1));
}
over_both_isas!(
    straight_code_is_one_block_without_edges,
    a_taken_conditional_splits_the_run_into_three_blocks,
    a_branch_into_the_middle_of_a_run_splits_the_block,
    a_merge_block_is_entered_by_a_jump_and_by_fallthrough,
    a_backward_jump_to_the_entry_is_a_self_loop,
    a_conditional_to_its_own_continuation_keeps_parallel_edges,
    code_after_a_terminator_forms_an_unreachable_block,
    continuation_past_the_last_instruction_adds_no_edge,
    an_explicit_entry_makes_lower_instructions_a_separate_function_part,
    an_entry_that_is_already_a_leader_adds_no_block,
);
