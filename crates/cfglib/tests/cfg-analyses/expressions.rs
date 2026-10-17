//! Expression-tree recovery from flat statements.

use std::collections::BTreeMap;

use cfglib::{
    BlockExprTrees, BlockId, Cfg, EdgeKind, ExprInstr, ExprNode, InstrInfo,
    recover_block_expressions, recover_expressions,
};

use crate::stmt::{Local, Op, Route, Stmt, block, fill, link};

type Tree = ExprNode<Local, Op, i64>;

fn leaf(variable: Local) -> Tree {
    ExprNode::Leaf(variable)
}

fn op(operator: Op, operands: Vec<Tree>) -> Tree {
    ExprNode::Op { operator, operands }
}

fn roots(trees: &BlockExprTrees<Local, Op, i64>) -> BTreeMap<Local, Tree> {
    trees.roots.iter().cloned().collect()
}

fn single_block(statements: impl IntoIterator<Item = Stmt>) -> (Cfg<Stmt, Route>, BlockId) {
    let mut cfg = Cfg::<Stmt, Route>::with_edge_payload();
    let entry = cfg.entry();
    fill(&mut cfg, entry, statements);
    (cfg, entry)
}

#[test]
fn a_single_use_temporary_is_inlined_into_its_user() {
    let (cfg, entry) = single_block([Stmt::multiply("t0", "b", "c"), Stmt::add("t1", "a", "t0")]);

    let trees = recover_block_expressions(&cfg, entry);

    assert_eq!(trees.block, entry);
    assert_eq!(
        roots(&trees),
        BTreeMap::from([(
            "t1",
            op(
                Op::Add,
                vec![leaf("a"), op(Op::Multiply, vec![leaf("b"), leaf("c")])]
            )
        )]),
        "add(a, mul(b, c))"
    );
}

#[test]
fn a_temporary_used_twice_stays_a_root_and_is_referenced_by_name() {
    let (cfg, entry) = single_block([Stmt::multiply("t0", "b", "c"), Stmt::add("t1", "t0", "t0")]);

    let recovered = roots(&recover_block_expressions(&cfg, entry));

    assert_eq!(
        recovered["t0"],
        op(Op::Multiply, vec![leaf("b"), leaf("c")])
    );
    assert_eq!(recovered["t1"], op(Op::Add, vec![leaf("t0"), leaf("t0")]));
    assert_eq!(recovered.len(), 2);
}

#[test]
fn constants_and_unanalyzable_definitions_become_their_own_node_kinds() {
    let (cfg, entry) = single_block([
        Stmt::constant("five", 5),
        Stmt::opaque("input"),
        Stmt::add("sum", "five", "input"),
    ]);

    let recovered = roots(&recover_block_expressions(&cfg, entry));

    assert_eq!(
        recovered["sum"],
        op(
            Op::Add,
            vec![
                ExprNode::Const(5),
                ExprNode::Opaque {
                    block: entry,
                    inst_idx: 1
                }
            ]
        )
    );
    assert_eq!(
        recovered.len(),
        1,
        "both single-use definitions were inlined"
    );
}

#[test]
fn a_value_used_once_here_and_again_elsewhere_is_inlined_here_and_absent_from_the_roots() {
    let mut cfg = Cfg::<Stmt, Route>::with_edge_payload();
    let entry = cfg.entry();
    fill(
        &mut cfg,
        entry,
        [Stmt::multiply("t", "a", "b"), Stmt::add("u", "t", "c")],
    );
    let later = block(&mut cfg, [Stmt::add("w", "t", "c")]);
    link(&mut cfg, entry, later, EdgeKind::Fallthrough);

    let all = recover_expressions(&cfg);

    assert_eq!(
        all.iter().map(|trees| trees.block).collect::<Vec<_>>(),
        [entry, later]
    );
    assert_eq!(
        roots(&all[0]),
        BTreeMap::from([(
            "u",
            op(
                Op::Add,
                vec![op(Op::Multiply, vec![leaf("a"), leaf("b")]), leaf("c")]
            )
        )]),
        "per-block recovery counts uses in this block only"
    );
    assert_eq!(
        roots(&all[1]),
        BTreeMap::from([("w", op(Op::Add, vec![leaf("t"), leaf("c")]))])
    );
}

#[test]
fn the_operand_list_the_statement_names_decides_the_tree_and_other_reads_do_not_join_it() {
    let (cfg, entry) = single_block([
        Stmt::compute(Op::Compare, "flags", &["x", "y"]),
        Stmt::compute(Op::Negate, "r", &["a"]).also_reading("flags"),
    ]);

    let recovered = roots(&recover_block_expressions(&cfg, entry));

    assert_eq!(
        recovered["r"],
        op(Op::Negate, vec![leaf("a")]),
        "the operand list is [a]"
    );
    assert_eq!(
        recovered["flags"],
        op(Op::Compare, vec![leaf("x"), leaf("y")]),
        "flags is read but not an operand, so it is not consumed into r"
    );
}

#[test]
fn a_temporary_is_not_inlined_past_a_redefinition_of_its_operand() {
    let (cfg, entry) = single_block([
        Stmt::add("t", "a", "b"),
        Stmt::constant("a", 9),
        Stmt::multiply("r", "t", "c"),
    ]);

    let recovered = roots(&recover_block_expressions(&cfg, entry));

    assert_eq!(
        recovered["r"],
        op(Op::Multiply, vec![leaf("t"), leaf("c")]),
        "inlining add(a, b) here would read the new a"
    );
    assert_eq!(recovered["t"], op(Op::Add, vec![leaf("a"), leaf("b")]));
}

#[test]
fn recovery_visits_every_live_block_and_skips_removed_ones() {
    let mut cfg = Cfg::<Stmt, Route>::with_edge_payload();
    let entry = cfg.entry();
    fill(&mut cfg, entry, [Stmt::add("x", "a", "b")]);
    let dropped = block(&mut cfg, [Stmt::add("y", "a", "b")]);
    let kept = block(&mut cfg, []);
    link(&mut cfg, entry, kept, EdgeKind::Fallthrough);
    cfg.remove_block(dropped);

    let all = recover_expressions(&cfg);

    assert_eq!(
        all.iter().map(|trees| trees.block).collect::<Vec<_>>(),
        [entry, kept]
    );
    assert_eq!(all[1].roots, Vec::new());
}

/// Evaluates a tree: leaves read `values`, constants are themselves.
fn evaluate(tree: &Tree, values: &BTreeMap<Local, i64>) -> i64 {
    match tree {
        ExprNode::Leaf(variable) => values[variable],
        ExprNode::Const(value) => *value,
        ExprNode::Op { operator, operands } => {
            let inputs: Vec<i64> = operands
                .iter()
                .map(|operand| evaluate(operand, values))
                .collect();
            match operator {
                Op::Add => inputs[0].wrapping_add(inputs[1]),
                Op::Multiply => inputs[0].wrapping_mul(inputs[1]),
                Op::Compare => i64::from(inputs[0] < inputs[1]),
                Op::Negate => inputs[0].wrapping_neg(),
            }
        }
        ExprNode::Opaque { .. } => unreachable!("this program has no opaque definitions"),
    }
}

#[test]
fn every_recovered_tree_computes_what_the_flat_statements_computed() {
    use crate::random::Rng;

    let names: [Local; 8] = ["i0", "i1", "i2", "t0", "t1", "t2", "t3", "t4"];
    for seed in 0..120 {
        let mut rng = Rng::new(seed);
        let mut statements = Vec::new();
        let mut defined: Vec<Local> = vec!["i0", "i1", "i2"];
        for &name in &names[3..] {
            let (left, right) = (
                defined[rng.below(defined.len())],
                defined[rng.below(defined.len())],
            );
            statements.push(match rng.below(4) {
                0 => Stmt::constant(name, i64::try_from(rng.below(9)).unwrap() - 4),
                1 => Stmt::compute(Op::Negate, name, &[left]),
                2 => Stmt::multiply(name, left, right),
                _ => Stmt::add(name, left, right),
            });
            defined.push(name);
        }
        let (cfg, entry) = single_block(statements.clone());
        let inputs = BTreeMap::from([("i0", 3_i64), ("i1", -5), ("i2", 7)]);

        // The flat execution: each single-assignment statement in order.
        let mut flat = inputs.clone();
        for statement in &statements {
            let value = if let Some(constant) = statement.as_const() {
                constant
            } else {
                let (operator, operands) = statement.as_expr().unwrap();
                evaluate(
                    &op(
                        operator,
                        operands.iter().map(|&operand| leaf(operand)).collect(),
                    ),
                    &flat,
                )
            };
            flat.insert(statement.defs()[0], value);
        }

        // A tree's leaves are inputs or other roots; roots are evaluated
        // from the program's definitions in order, each reading the values
        // computed so far.
        let recovered = roots(&recover_block_expressions(&cfg, entry));
        let mut environment = inputs.clone();
        for statement in &statements {
            let name = statement.defs()[0];
            if let Some(tree) = recovered.get(name) {
                environment.insert(name, evaluate(tree, &environment));
                assert_eq!(environment[name], flat[name], "seed {seed}: {name}");
            }
        }
        for statement in &statements {
            let name = statement.defs()[0];
            let reads: usize = statements
                .iter()
                .map(|other| other.uses().iter().filter(|&&used| used == name).count())
                .sum();
            assert_eq!(
                recovered.contains_key(name),
                reads != 1,
                "seed {seed}: {name} is read {reads} times; only a single-use definition is inlined"
            );
        }
    }
}
