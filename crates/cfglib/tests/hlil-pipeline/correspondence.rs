//! The tables that relate flat instructions and variables to the structured
//! function: the rows that `lift` writes, and how `Correspondence::remap`
//! carries them across `recover_structure`.

use std::panic::{AssertUnwindSafe, catch_unwind};

use cfglib::ir::hlil::{Entity, LiftMetadata, lift};
use cfglib::{EffectInfo, ProgramPoint};

use crate::describe;
use crate::dialect::{Calc, Lit, Op};
use crate::flat::{Code, Form, Instr};
use crate::pipeline::{Lifted, lifted, lowered, recovered, structured};
use crate::programs;
use crate::structured::{Builder, ExprId, ExprNode, Function, StmtId, StmtNode, VarId};

type Row = (ProgramPoint, Entity<StmtId, ExprId, VarId>);

/// The operation that the lifted form of a flat instruction applies.
fn lifted_op(form: &Form) -> Option<Op> {
    match form {
        Form::Apply(op) => Some(*op),
        Form::Caught => Some(Op::Caught),
        Form::Enter(_) => Some(Op::Acquire),
        Form::Exit(_) => Some(Op::Release),
        _ => None,
    }
}

/// Whether lifting must keep a visible trace of the instruction: it ends a
/// block, writes memory, produces output, or can throw. A pure definition
/// may legitimately vanish into an inlined expression or a dead-value prune.
fn must_have_row(instruction: &Instr) -> bool {
    match &instruction.form {
        Form::Branch
        | Form::BranchOn(_)
        | Form::Switch
        | Form::Return
        | Form::Store(_)
        | Form::Enter(_)
        | Form::Exit(_) => true,
        Form::Apply(_) | Form::Caught => {
            instruction.defs.is_empty()
                || instruction.may_throw()
                || !instruction.effects().is_empty()
        }
        Form::Const(_) | Form::Copy | Form::Move | Form::Nop => false,
    }
}

fn find(code: &Code, wanted: impl Fn(&Instr) -> bool) -> Vec<ProgramPoint> {
    code.points()
        .into_iter()
        .filter(|&point| wanted(code.instruction(point)))
        .collect()
}

/// Checks that the entity of a row is what the instruction stands for.
fn assert_row_agrees(code: &Code, function: &Function, (point, entity): Row, name: &str) {
    let instruction = code.instruction(point);
    let stmt_kind = |stmt: StmtId| &function.stmts[stmt.index()];
    let is_value = |expr: ExprId, op: Op| matches!(&function.exprs[expr.index()].node, ExprNode::Op { op: applied, .. } if *applied == op);
    match (&instruction.form, entity) {
        (Form::Nop, _) => panic!("{name}: a pure transfer has a row at {point}"),
        (Form::Branch | Form::BranchOn(_), Entity::Statement(stmt)) => assert!(
            matches!(
                stmt_kind(stmt),
                StmtNode::If { .. }
                    | StmtNode::While { .. }
                    | StmtNode::DoWhile { .. }
                    | StmtNode::For { .. }
            ),
            "{name}: the branch at {point} is {:?}",
            stmt_kind(stmt)
        ),
        (Form::Switch, Entity::Statement(stmt)) => {
            assert!(matches!(stmt_kind(stmt), StmtNode::Switch { .. }), "{name}");
        }
        (Form::Return, Entity::Statement(stmt)) => {
            assert!(matches!(stmt_kind(stmt), StmtNode::Return(_)), "{name}");
        }
        (Form::Store(_), Entity::Statement(stmt)) => {
            let StmtNode::Assign { target, .. } = stmt_kind(stmt) else {
                panic!("{name}: a store at {point} is not an assignment");
            };
            assert!(is_value(*target, Op::Cell), "{name}: the place is a cell");
        }
        (form, Entity::Statement(stmt)) if instruction.defs.is_empty() => {
            let StmtNode::Expression(expr) = stmt_kind(stmt) else {
                panic!("{name}: the effect at {point} is {:?}", stmt_kind(stmt));
            };
            assert!(
                is_value(*expr, lifted_op(form).expect("an effect is an operation")),
                "{name}"
            );
        }
        (_, Entity::Statement(stmt)) => assert!(
            matches!(stmt_kind(stmt), StmtNode::Assign { .. }),
            "{name}: the definition at {point} is {:?}",
            stmt_kind(stmt)
        ),
        (form, Entity::Expression(expr)) => match (lifted_op(form), form) {
            (Some(op), _) => assert!(
                is_value(expr, op),
                "{name}: {form:?} inlined as another operation"
            ),
            (None, Form::Const(literal)) => assert!(
                matches!(&function.exprs[expr.index()].node, ExprNode::Const(made) if made == literal),
                "{name}"
            ),
            // A copy shares the expression of the value it copies.
            _ => {}
        },
        (_, Entity::Variable(_)) => panic!("{name}: no instruction defines a whole variable"),
    }
}

#[test]
fn lifted_rows_agree_with_the_instruction_each_one_stands_for() {
    let mut rows = 0;
    for case in programs::all() {
        let code = lowered(&structured(&case.program)).function;
        let lifted = lifted(&code);
        for &row in &lifted.correspondence.instructions {
            assert_row_agrees(&code, &lifted.function, row, case.program.name);
            rows += 1;
        }
        for point in code.points() {
            if must_have_row(code.instruction(point)) {
                assert!(
                    lifted
                        .correspondence
                        .instructions
                        .iter()
                        .any(|&(at, _)| at == point),
                    "{}: {point} ({:?}) has no row",
                    case.program.name,
                    code.instruction(point).form
                );
            }
        }
    }
    assert!(rows > 300, "only {rows} rows were checked");
}

#[test]
fn lifted_rows_are_sorted_by_point_with_at_most_one_row_per_point() {
    for case in programs::all() {
        let code = lowered(&structured(&case.program)).function;
        let rows = lifted(&code).correspondence.instructions;
        assert!(
            rows.windows(2).all(|pair| pair[0].0 < pair[1].0),
            "{} has unsorted or repeated rows",
            case.program.name
        );
    }
}

#[test]
fn lifted_row_entities_are_in_range_of_the_lifted_function() {
    for case in programs::all() {
        let code = lowered(&structured(&case.program)).function;
        let lifted = lifted(&code);
        for &(_, entity) in &lifted.correspondence.instructions {
            match entity {
                Entity::Statement(stmt) => assert!(stmt.index() < lifted.function.stmts.len()),
                Entity::Expression(expr) => assert!(expr.index() < lifted.function.exprs.len()),
                Entity::Variable(variable) => {
                    assert!(variable.index() < lifted.function.vars.len());
                }
            }
        }
    }
}

#[test]
fn a_builder_that_issues_handles_in_call_order_gets_the_identity_variable_table() {
    for case in programs::all() {
        let code = lowered(&structured(&case.program)).function;
        let lifted = lifted(&code);
        let expected: Vec<VarId> = (0..code.locals.len()).map(VarId::from_index).collect();
        assert_eq!(
            lifted.correspondence.variables, expected,
            "{}",
            case.program.name
        );
        for (flat, &variable) in lifted.correspondence.variables.iter().enumerate() {
            assert_eq!(
                &lifted.function.vars[variable.index()].local,
                &code.locals[flat]
            );
        }
    }
}

#[test]
fn omitting_metadata_drops_the_rows_and_keeps_the_function_and_variables() {
    for case in programs::all() {
        let code = lowered(&structured(&case.program)).function;
        let kept = lifted(&code);
        let omitted = lift::<Calc, _, _>(&code, Builder::new(), LiftMetadata::Omit)
            .unwrap_or_else(|error| panic!("{error}"));
        assert!(
            omitted.correspondence.instructions.is_empty(),
            "{}",
            case.program.name
        );
        assert_eq!(
            omitted.correspondence.variables,
            kept.correspondence.variables
        );
        assert_eq!(omitted.function, kept.function, "{}", case.program.name);
        assert_eq!(omitted.report, kept.report);
    }
}

/// What a source statement and its lifted carrier must share: the place an
/// assignment writes and the operation at the root of what it writes or
/// performs. The operands may differ, because lifting inlines a single-use
/// definition that the source named (a handler's bound exception), and it
/// leaves a throwing operand in a statement of its own when inlining it would
/// move it past another throwing operand.
fn comparable_texts(
    source: &Function,
    origin: Entity<StmtId, ExprId, VarId>,
    lifted: &Function,
    carrier: Entity<StmtId, ExprId, VarId>,
) -> Option<(String, String)> {
    let is_assign = |function: &Function, stmt: StmtId| {
        matches!(function.stmts[stmt.index()], StmtNode::Assign { .. })
    };
    let root = |text: String| text.split('(').next().unwrap_or_default().to_owned();
    match (origin, carrier) {
        (Entity::Statement(from), Entity::Statement(to))
            if is_assign(source, from) && is_assign(lifted, to) =>
        {
            Some((
                root(describe::head(source, from)),
                root(describe::head(lifted, to)),
            ))
        }
        (Entity::Expression(effect), Entity::Statement(to))
            if matches!(lifted.stmts[to.index()], StmtNode::Expression(_))
                && source
                    .stmts
                    .iter()
                    .any(|stmt| matches!(stmt, StmtNode::Expression(at) if *at == effect)) =>
        {
            Some((
                root(describe::expression(source, effect)),
                root(describe::head(lifted, to)),
            ))
        }
        _ => None,
    }
}

#[test]
fn lowering_then_lifting_keeps_the_target_and_operation_of_each_source_statement() {
    let mut compared = 0;
    for case in programs::all() {
        let source = structured(&case.program);
        let lowered = lowered(&source);
        let lifted = lifted(&lowered.function);
        for &(point, origin) in &lowered.origins {
            let Some(&(_, carrier)) = lifted
                .correspondence
                .instructions
                .iter()
                .find(|&&(at, _)| at == point)
            else {
                continue;
            };
            if let Some((before, after)) =
                comparable_texts(&source, origin, &lifted.function, carrier)
            {
                assert_eq!(before, after, "{} at {point}", case.program.name);
                compared += 1;
            }
        }
    }
    assert!(compared > 100, "only {compared} statements were compared");
}

/// The one flat instruction that satisfies `wanted`.
fn only(code: &Code, what: &str, wanted: impl Fn(&Instr) -> bool) -> ProgramPoint {
    let found = find(code, wanted);
    assert_eq!(
        found.len(),
        1,
        "{what}: expected one match, found {found:?}"
    );
    found[0]
}

fn row_at(rows: &[Row], point: ProgramPoint) -> Option<Entity<StmtId, ExprId, VarId>> {
    rows.iter()
        .find(|&&(at, _)| at == point)
        .map(|&(_, entity)| entity)
}

fn row_of(lifted: &Lifted, point: ProgramPoint) -> Entity<StmtId, ExprId, VarId> {
    let Some(entity) = row_at(&lifted.correspondence.instructions, point) else {
        panic!("no row at {point}");
    };
    entity
}

#[test]
fn remapping_follows_a_counting_loop_into_the_for_statement_that_recovery_builds() {
    let source = structured(&programs::counting_while().program);
    let code = lowered(&source).function;
    let lifted = lifted(&code);
    let branch = only(&code, "the loop test", |i| i.form == Form::Branch);
    let emit = only(&code, "the emit", |i| i.form == Form::Apply(Op::Emit));
    let update = only(&code, "the increment", |i| i.form == Form::Apply(Op::Add));
    let init = only(&code, "the initializer", |i| {
        i.form == Form::Const(Lit::Int(0))
    });
    let recovery = recovered(&lifted.function);
    assert_eq!(recovery.for_loops, 1);
    let rows = lifted.correspondence.clone().remap(&recovery);

    let function = &recovery.function;
    let StmtNode::For {
        initializer,
        update: updates,
        body,
        ..
    } = &function.stmts[function.body[0].index()]
    else {
        panic!("the first statement is the recovered for, which took the initializer");
    };
    let remapped = |point| {
        let Some(entity) = row_at(&rows.instructions, point) else {
            panic!("the row at {point} was dropped");
        };
        entity
    };
    assert_eq!(remapped(branch), Entity::Statement(function.body[0]));
    assert_eq!(remapped(init), Entity::Statement(initializer[0]));
    assert_eq!(remapped(update), Entity::Statement(updates[0]));
    assert_eq!(remapped(emit), Entity::Statement(body[0]));
    // Every statement that kept its meaning reads the same as before.
    for (point, kept) in [
        (init, initializer[0]),
        (update, updates[0]),
        (emit, body[0]),
    ] {
        let Entity::Statement(before) = row_of(&lifted, point) else {
            panic!("{point} is a statement before recovery");
        };
        assert_eq!(
            describe::head(&lifted.function, before),
            describe::head(function, kept)
        );
    }
}

#[test]
fn remapping_sends_every_statement_of_a_recovered_selection_to_the_one_assignment() {
    let source = structured(&programs::select_assign().program);
    let code = lowered(&source).function;
    let lifted = lifted(&code);
    let branch = only(&code, "the branch", |i| i.form == Form::Branch);
    let compare = only(&code, "the comparison", |i| i.form == Form::Apply(Op::Lt));
    let copies = find(&code, |i| i.form == Form::Copy);
    assert_eq!(copies.len(), 2);
    let recovery = recovered(&lifted.function);
    assert_eq!(recovery.selects, 1);
    let rows = lifted.correspondence.clone().remap(&recovery);
    let carriers: Vec<_> = [branch, copies[0], copies[1]]
        .into_iter()
        .map(|point| {
            rows.instructions
                .iter()
                .find(|&&(at, _)| at == point)
                .map(|&(_, entity)| entity)
                .expect("a selection keeps the rows of what it consumed")
        })
        .collect();
    assert_eq!(carriers[0], carriers[1]);
    assert_eq!(carriers[0], carriers[2]);
    assert_eq!(carriers[0], Entity::Statement(recovery.function.body[0]));
    let Entity::Expression(condition) = rows
        .instructions
        .iter()
        .find(|&&(at, _)| at == compare)
        .map(|&(_, entity)| entity)
        .expect("the condition keeps its row")
    else {
        panic!("the comparison was inlined into the condition");
    };
    assert_eq!(
        describe::expression(&recovery.function, condition),
        "lt(v0, v1)"
    );
}

#[test]
fn remapping_gives_a_recovered_region_the_rows_of_what_it_regenerates() {
    let source = structured(&programs::acquire_with_cleanup().program);
    let code = lowered(&source).function;
    let lifted = lifted(&code);
    let acquire = only(&code, "acquire", |i| i.form == Form::Apply(Op::Acquire));
    let releases = find(&code, |i| i.form == Form::Apply(Op::Release));
    let raise = only(&code, "raise", |i| i.form == Form::Apply(Op::Raise));
    let caught = only(&code, "caught", |i| i.form == Form::Apply(Op::Caught));
    let emit = only(&code, "emit", |i| i.form == Form::Apply(Op::Emit));
    let recovery = recovered(&lifted.function);
    assert_eq!(recovery.regions, 1);
    let rows = lifted.correspondence.clone().remap(&recovery);
    let function = &recovery.function;
    let region = function.body[0];
    let StmtNode::Region { body, .. } = &function.stmts[region.index()] else {
        panic!("the first statement is the recovered region");
    };
    let remapped = |point| {
        rows.instructions
            .iter()
            .find(|&&(at, _)| at == point)
            .map(|&(_, entity)| entity)
    };
    // The enter, the exit inside the body, and the whole cleanup belong to
    // the region now.
    for point in [acquire, releases[0], releases[1], raise] {
        assert_eq!(remapped(point), Some(Entity::Statement(region)), "{point}");
    }
    // The statement inside the body keeps its place, and the value that the
    // cleanup regenerates has no copy to follow.
    assert_eq!(remapped(emit), Some(Entity::Statement(body[0])));
    assert_eq!(remapped(caught), None, "the rethrown exception has no copy");
}

#[test]
fn remapping_keeps_rows_sorted_in_range_and_never_adds_one() {
    for case in programs::all() {
        let code = lowered(&structured(&case.program)).function;
        let lifted = lifted(&code);
        let recovery = recovered(&lifted.function);
        let before = lifted.correspondence.instructions.clone();
        let after = lifted.correspondence.remap(&recovery);
        assert!(
            after.instructions.len() <= before.len(),
            "{}",
            case.program.name
        );
        assert!(
            after
                .instructions
                .windows(2)
                .all(|pair| pair[0].0 < pair[1].0)
        );
        let function = &recovery.function;
        for &(point, entity) in &after.instructions {
            assert!(
                before.iter().any(|&(at, _)| at == point),
                "{}",
                case.program.name
            );
            match entity {
                Entity::Statement(stmt) => assert!(stmt.index() < function.stmts.len()),
                Entity::Expression(expr) => assert!(expr.index() < function.exprs.len()),
                Entity::Variable(variable) => assert!(variable.index() < function.vars.len()),
            }
        }
        assert_eq!(after.variables.len(), code.locals.len());
        for (flat, &variable) in after.variables.iter().enumerate() {
            assert_eq!(function.vars[variable.index()].local, code.locals[flat]);
        }
    }
}

#[test]
fn remapping_a_table_of_another_function_panics() {
    let bigger = lowered(&structured(&programs::nested_loops().program)).function;
    let table = lifted(&bigger).correspondence;
    let smaller = lifted(&lowered(&structured(&programs::straight_line().program)).function);
    let recovery = recovered(&smaller.function);
    let outcome = catch_unwind(AssertUnwindSafe(|| table.remap(&recovery)));
    assert!(
        outcome.is_err(),
        "a variable of another function has no copy"
    );
}
