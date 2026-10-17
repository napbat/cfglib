//! `VariablePlacements`: each referenced variable is anchored before the
//! first statement, in the narrowest lexical body that holds all of its
//! occurrences, and a `for` clause is not a body.
//!
//! The oracle walks the function itself: it records the path of lexical
//! lists down to every statement that mentions a variable, and the anchor of
//! a variable is the first statement of the deepest list that all of its
//! paths pass through.

use std::collections::{BTreeMap, BTreeSet};

use cfglib::ir::hlil::VariablePlacements;

use crate::dialect::Calc;
use crate::pipeline::{lifted, lowered, recovered, structured};
use crate::programs;
use crate::source::{Program, S, add, emit, int, lt, ret, set, var};
use crate::structured::{ExprId, ExprNode, Function, StmtId, StmtNode, VarId};

fn expression_variables(function: &Function, expr: ExprId, found: &mut BTreeSet<VarId>) {
    match &function.exprs[expr.index()].node {
        ExprNode::Var(variable) => {
            found.insert(*variable);
        }
        ExprNode::Const(_) => {}
        ExprNode::Op { operands, .. } => {
            for &operand in operands {
                expression_variables(function, operand, found);
            }
        }
    }
}

/// Every variable mentioned by the statements of a `for` clause, nested
/// statements included.
fn clause_variables(function: &Function, list: &[StmtId], found: &mut BTreeSet<VarId>) {
    for &stmt in list {
        found.extend(own_variables(function, stmt));
        for child in crate::structured_run::bodies(function, stmt) {
            clause_variables(function, child, found);
        }
    }
}

/// The variables that `stmt` mentions itself: in its own expressions, and,
/// for a `for`, in its clauses, which are part of its syntax.
fn own_variables(function: &Function, stmt: StmtId) -> BTreeSet<VarId> {
    let mut found = BTreeSet::new();
    let roots: Vec<ExprId> = match &function.stmts[stmt.index()] {
        StmtNode::Expression(expr) => vec![*expr],
        StmtNode::Assign { target, value } => vec![*target, *value],
        StmtNode::If { condition, .. }
        | StmtNode::While { condition, .. }
        | StmtNode::DoWhile { condition, .. } => vec![*condition],
        StmtNode::Switch { scrutinee, .. } => vec![*scrutinee],
        StmtNode::Return(values) => values.clone(),
        StmtNode::Region { operands, .. } => operands.clone(),
        StmtNode::For {
            initializer,
            condition,
            update,
            ..
        } => {
            clause_variables(function, initializer, &mut found);
            clause_variables(function, update, &mut found);
            condition.iter().copied().collect()
        }
        _ => Vec::new(),
    };
    for root in roots {
        expression_variables(function, root, &mut found);
    }
    found
}

/// The lexical bodies of `stmt`: the child lists that open a scope.
fn lexical_bodies(function: &Function, stmt: StmtId) -> Vec<&[StmtId]> {
    match &function.stmts[stmt.index()] {
        StmtNode::For { body, .. } => vec![body],
        _ => crate::structured_run::bodies(function, stmt),
    }
}

#[derive(Default)]
struct Walk {
    lists: Vec<Vec<StmtId>>,
    paths: BTreeMap<VarId, Vec<Vec<(usize, usize)>>>,
}

impl Walk {
    fn list(&mut self, function: &Function, statements: &[StmtId], path: &mut Vec<(usize, usize)>) {
        let id = self.lists.len();
        self.lists.push(statements.to_vec());
        for (index, &stmt) in statements.iter().enumerate() {
            path.push((id, index));
            for variable in own_variables(function, stmt) {
                self.paths.entry(variable).or_default().push(path.clone());
            }
            for body in lexical_bodies(function, stmt) {
                self.list(function, body, path);
            }
            path.pop();
        }
    }

    /// The statement that `variable`'s declaration must immediately precede.
    fn anchor(&self, variable: VarId) -> Option<StmtId> {
        let paths = self.paths.get(&variable)?;
        let mut depth = 0;
        while paths.iter().all(|path| path.len() > depth + 1)
            && paths
                .iter()
                .all(|path| path[depth + 1].0 == paths[0][depth + 1].0)
        {
            depth += 1;
        }
        let list = paths[0][depth].0;
        let first = paths.iter().map(|path| path[depth].1).min()?;
        Some(self.lists[list][first])
    }
}

fn expected(function: &Function) -> (BTreeMap<VarId, StmtId>, BTreeMap<StmtId, Vec<VarId>>) {
    let mut walk = Walk::default();
    walk.list(function, &function.body, &mut Vec::new());
    let mut placements = BTreeMap::new();
    let mut before: BTreeMap<StmtId, Vec<VarId>> = BTreeMap::new();
    for index in 0..function.vars.len() {
        let variable = VarId::from_index(index);
        if let Some(anchor) = walk.anchor(variable) {
            placements.insert(variable, anchor);
            before.entry(anchor).or_default().push(variable);
        }
    }
    (placements, before)
}

fn assert_matches_the_oracle(function: &Function, name: &str) {
    let computed = VariablePlacements::compute::<Calc, _>(function);
    let (placements, before) = expected(function);
    for index in 0..function.vars.len() {
        let variable = VarId::from_index(index);
        assert_eq!(
            computed.placement(variable),
            placements.get(&variable).copied(),
            "{name}: variable v{index}"
        );
    }
    for index in 0..function.stmts.len() {
        let stmt = StmtId::from_index(index);
        let wanted: &[VarId] = before.get(&stmt).map_or(&[], Vec::as_slice);
        assert_eq!(computed.before(stmt), wanted, "{name}: before s{index}");
    }
}

#[test]
fn placements_anchor_at_the_narrowest_body_for_every_pipeline_function() {
    for case in programs::all() {
        let source = structured(&case.program);
        let lifted = lifted(&lowered(&source).function).function;
        let recovered = recovered(&lifted).function;
        assert_matches_the_oracle(&source, case.program.name);
        assert_matches_the_oracle(&lifted, case.program.name);
        assert_matches_the_oracle(&recovered, case.program.name);
    }
}

fn function_of(statements: Vec<S>) -> Function {
    structured(&Program::new(
        "placement",
        &["a"],
        &["t", "u", "i"],
        statements,
    ))
}

#[test]
fn a_variable_used_in_one_branch_is_declared_inside_it() {
    let function = function_of(vec![
        S::If(
            lt(var("a"), int(0)),
            vec![set("t", int(1)), emit(var("t"))],
            vec![],
        ),
        ret(var("a")),
    ]);
    let placements = VariablePlacements::compute::<Calc, _>(&function);
    let t = function.variable_named("t");
    let StmtNode::If { then_body, .. } = &function.stmts[function.body[0].index()] else {
        panic!("the first statement is the if");
    };
    assert_eq!(
        placements.placement(t),
        Some(then_body[0]),
        "before the first use in the branch"
    );
    assert_eq!(placements.before(then_body[0]), [t]);
    assert!(
        !placements.before(function.body[0]).contains(&t),
        "not hoisted before the if"
    );
}

#[test]
fn a_variable_used_in_both_branches_is_declared_before_the_if() {
    let function = function_of(vec![
        S::If(
            lt(var("a"), int(0)),
            vec![set("t", int(1))],
            vec![set("t", int(2))],
        ),
        ret(var("a")),
    ]);
    let placements = VariablePlacements::compute::<Calc, _>(&function);
    let t = function.variable_named("t");
    assert_eq!(placements.placement(t), Some(function.body[0]));
}

#[test]
fn a_variable_used_twice_in_one_body_is_declared_before_its_first_use() {
    let function = function_of(vec![
        emit(var("a")),
        set("u", int(1)),
        emit(var("t")),
        emit(var("u")),
        ret(var("a")),
    ]);
    let placements = VariablePlacements::compute::<Calc, _>(&function);
    assert_eq!(
        placements.placement(function.variable_named("u")),
        Some(function.body[1])
    );
    assert_eq!(
        placements.placement(function.variable_named("t")),
        Some(function.body[2])
    );
    assert_eq!(
        placements.placement(function.variable_named("a")),
        Some(function.body[0])
    );
}

#[test]
fn a_variable_that_nothing_mentions_has_no_declaration() {
    let function = function_of(vec![ret(var("a"))]);
    let placements = VariablePlacements::compute::<Calc, _>(&function);
    assert_eq!(placements.placement(function.variable_named("t")), None);
    assert_eq!(
        placements.placement(function.variable_named("a")),
        Some(function.body[0])
    );
}

#[test]
fn a_for_clause_is_not_a_body_so_its_variable_is_declared_before_the_for() {
    let function = function_of(vec![
        S::For(
            vec![set("i", int(0))],
            Some(lt(var("i"), var("a"))),
            vec![set("i", add(var("i"), int(1)))],
            vec![emit(var("a"))],
        ),
        ret(var("a")),
    ]);
    let placements = VariablePlacements::compute::<Calc, _>(&function);
    let i = function.variable_named("i");
    assert_eq!(placements.placement(i), Some(function.body[0]));
    let StmtNode::For { body, .. } = &function.stmts[function.body[0].index()] else {
        panic!("the first statement is the for");
    };
    assert_ne!(placements.placement(i), Some(body[0]));
    assert_matches_the_oracle(&function, "for clause");
}

#[test]
fn a_variable_used_only_in_a_loop_body_is_declared_in_the_body() {
    let function = function_of(vec![
        S::While(
            lt(var("a"), int(3)),
            vec![
                set("t", add(var("a"), int(1))),
                emit(var("t")),
                set("a", var("t")),
            ],
        ),
        ret(var("a")),
    ]);
    let placements = VariablePlacements::compute::<Calc, _>(&function);
    let StmtNode::While { body, .. } = &function.stmts[function.body[0].index()] else {
        panic!("the first statement is the loop");
    };
    assert_eq!(
        placements.placement(function.variable_named("t")),
        Some(body[0])
    );
}
