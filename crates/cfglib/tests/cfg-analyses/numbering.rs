//! Value numbering: what counts as the same value, what an impure statement
//! forgets, and what an unwinding handler may reuse.

use cfglib::{BlockId, BlockValueNumbers, Cfg, DominatorTree, EdgeKind, ValueNumbering};

use crate::stmt::{Route, Stmt, block, fill, link};

fn single_block(statements: impl IntoIterator<Item = Stmt>) -> (Cfg<Stmt, Route>, BlockId) {
    let mut cfg = Cfg::<Stmt, Route>::with_edge_payload();
    let entry = cfg.entry();
    fill(&mut cfg, entry, statements);
    (cfg, entry)
}

fn local(cfg: &Cfg<Stmt, Route>, entry: BlockId) -> BlockValueNumbers {
    BlockValueNumbers::compute(cfg, entry, 0).0
}

#[test]
fn equal_operators_over_equal_operands_are_one_value_and_operand_order_matters() {
    let (cfg, entry) = single_block([
        Stmt::add("t1", "a", "b"),
        Stmt::add("t2", "a", "b"),
        Stmt::multiply("t3", "a", "b"),
        Stmt::add("t4", "b", "a"),
    ]);

    let numbers = local(&cfg, entry);

    assert_eq!(numbers.redundant, [1]);
    assert_eq!(numbers.inst_vn[0], numbers.inst_vn[1]);
    assert_ne!(numbers.inst_vn[0], numbers.inst_vn[2], "another operator");
    assert_ne!(
        numbers.inst_vn[0], numbers.inst_vn[3],
        "this operator does not commute"
    );
    assert!(numbers.inst_vn.iter().all(Option::is_some));
}

#[test]
fn constants_with_equal_values_match_and_different_ones_do_not() {
    let (cfg, entry) = single_block([
        Stmt::constant("c1", 5),
        Stmt::constant("c2", 5),
        Stmt::constant("c3", 6),
        Stmt::add("t", "c1", "c3"),
        Stmt::add("u", "c2", "c3"),
    ]);

    let numbers = local(&cfg, entry);

    assert_eq!(
        numbers.redundant,
        [1, 4],
        "c2 repeats c1, and u repeats t over equal operands"
    );
    assert_ne!(numbers.inst_vn[0], numbers.inst_vn[2]);
}

#[test]
fn impure_statements_are_not_numbered_but_their_definitions_invalidate_what_depends_on_them() {
    let (cfg, entry) = single_block([
        Stmt::add("t1", "a", "b"),
        Stmt::opaque("a"),
        Stmt::add("t2", "a", "b"),
        Stmt::effect(&[crate::stmt::Effect::Output], &["t2"]),
    ]);

    let numbers = local(&cfg, entry);

    assert_eq!(numbers.inst_vn[1], None);
    assert_eq!(numbers.inst_vn[3], None);
    assert!(
        numbers.redundant.is_empty(),
        "a was overwritten, so a + b is a new value"
    );
    assert_ne!(numbers.inst_vn[0], numbers.inst_vn[2]);
}

struct Diamond {
    cfg: Cfg<Stmt, Route>,
    entry: BlockId,
    left: BlockId,
    right: BlockId,
    merge: BlockId,
}

fn diamond() -> Diamond {
    let mut cfg = Cfg::<Stmt, Route>::with_edge_payload();
    let entry = cfg.entry();
    fill(&mut cfg, entry, [Stmt::add("t", "a", "b")]);
    let left = block(
        &mut cfg,
        [Stmt::add("u", "a", "b"), Stmt::multiply("m", "a", "b")],
    );
    let right = block(&mut cfg, [Stmt::multiply("n", "a", "b")]);
    let merge = block(&mut cfg, [Stmt::multiply("v", "a", "b")]);
    link(&mut cfg, entry, left, EdgeKind::ConditionalTrue);
    link(&mut cfg, entry, right, EdgeKind::ConditionalFalse);
    link(&mut cfg, left, merge, EdgeKind::Fallthrough);
    link(&mut cfg, right, merge, EdgeKind::Fallthrough);
    Diamond {
        cfg,
        entry,
        left,
        right,
        merge,
    }
}

#[test]
fn a_dominated_block_reuses_its_dominators_values_but_siblings_and_joins_do_not() {
    let Diamond {
        cfg,
        entry,
        left,
        right,
        merge,
    } = diamond();
    let numbering = ValueNumbering::compute(&cfg, &DominatorTree::compute(&cfg));

    assert_eq!(numbering.blocks[&entry].redundant, Vec::<usize>::new());
    assert_eq!(
        numbering.blocks[&left].redundant,
        [0],
        "u repeats the entry's t"
    );
    assert!(
        numbering.blocks[&right].redundant.is_empty(),
        "the left arm's multiply is not available in the right arm"
    );
    assert!(
        numbering.blocks[&merge].redundant.is_empty(),
        "neither arm dominates the join, so neither arm's multiply is available there"
    );
    assert_eq!(numbering.redundant_count(), 1);
    assert_eq!(
        numbering.blocks[&left].inst_vn[0], numbering.blocks[&entry].inst_vn[0],
        "a redundant statement carries the number of the value it repeats"
    );
    assert!(numbering.value_count > 0);
}

#[test]
fn a_redefinition_in_a_dominating_block_keeps_later_blocks_from_reusing_the_old_value() {
    let mut cfg = Cfg::<Stmt, Route>::with_edge_payload();
    let entry = cfg.entry();
    fill(
        &mut cfg,
        entry,
        [Stmt::add("t1", "a", "b"), Stmt::opaque("a")],
    );
    let next = block(&mut cfg, [Stmt::add("t2", "a", "b")]);
    link(&mut cfg, entry, next, EdgeKind::Fallthrough);

    let numbering = ValueNumbering::compute(&cfg, &DominatorTree::compute(&cfg));

    assert_eq!(numbering.blocks[&next].redundant, Vec::<usize>::new());
}

#[test]
fn a_handler_reuses_only_what_the_throwing_block_computed_before_it_threw() {
    let mut cfg = Cfg::<Stmt, Route>::with_edge_payload();
    let guarded = cfg.entry();
    fill(
        &mut cfg,
        guarded,
        [
            Stmt::add("before", "a", "b"),
            Stmt::call("risky", &[], &[]).throwing(),
            Stmt::multiply("after", "a", "b"),
        ],
    );
    let normal = block(
        &mut cfg,
        [
            Stmt::add("n1", "a", "b"),
            Stmt::multiply("n2", "a", "b"),
            Stmt::ret(),
        ],
    );
    let handler = block(
        &mut cfg,
        [
            Stmt::add("h1", "a", "b"),
            Stmt::multiply("h2", "a", "b"),
            Stmt::ret(),
        ],
    );
    link(&mut cfg, guarded, normal, EdgeKind::Fallthrough);
    link(&mut cfg, guarded, handler, EdgeKind::ExceptionUnwind);

    let numbering = ValueNumbering::compute(&cfg, &DominatorTree::compute(&cfg));

    assert_eq!(
        numbering.blocks[&normal].redundant,
        [0, 1],
        "normal flow sees both: the call completed"
    );
    assert_eq!(
        numbering.blocks[&handler].redundant,
        [0],
        "the handler saw the state before the call: `before` yes, `after` not yet computed"
    );
}
