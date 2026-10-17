//! `lower`: what it records about each flat instruction, the regions and
//! unwind edges it registers, and the programs it must refuse.

use std::collections::BTreeSet;

use cfglib::HandlerKind as RegionHandlerKind;
use cfglib::ir::hlil::{
    Entity, FunctionBuilder, FunctionView, Handler, HandlerKind, LowerError, lower, verify,
};
use cfglib::{BlockId, EdgeKind, ProgramPoint};

use crate::dialect::{Calc, Lit, Local, Op, Ty};
use crate::flat::{CodeBuilder, CodeError, Form};
use crate::flat_run::run_flat;
use crate::machine::{End, Event};
use crate::pipeline::{lifted, lowered, structured};
use crate::programs;
use crate::source::{Program, S, add, int, ret, set, var};
use crate::structured::{Builder, ExprId, ExprNode, Function, StmtId, StmtNode, VarId};
use crate::structured_run::{preorder, run_structured};

#[test]
fn origins_name_every_emitted_instruction_exactly_once() {
    for case in programs::all() {
        let lowered = lowered(&structured(&case.program));
        let rows: Vec<ProgramPoint> = lowered.origins.iter().map(|&(point, _)| point).collect();
        let unique: BTreeSet<ProgramPoint> = rows.iter().copied().collect();
        assert_eq!(
            unique.len(),
            rows.len(),
            "{} repeats a point",
            case.program.name
        );
        let emitted: BTreeSet<ProgramPoint> = lowered.function.points().into_iter().collect();
        assert_eq!(unique, emitted, "{}", case.program.name);
    }
}

#[test]
fn assignments_and_returns_appear_in_origins_in_source_order() {
    for case in programs::all() {
        let function = structured(&case.program);
        let expected: Vec<StmtId> = preorder(&function)
            .into_iter()
            .filter(|&stmt| {
                matches!(
                    function.stmts[stmt.index()],
                    StmtNode::Assign { .. } | StmtNode::Return(_)
                )
            })
            .collect();
        let emitted: Vec<StmtId> = lowered(&function)
            .origins
            .iter()
            .filter_map(|&(_, entity)| match entity {
                Entity::Statement(stmt) => Some(stmt),
                _ => None,
            })
            .filter(|&stmt| {
                matches!(
                    function.stmts[stmt.index()],
                    StmtNode::Assign { .. } | StmtNode::Return(_)
                )
            })
            .collect();
        assert_eq!(emitted, expected, "{}", case.program.name);
    }
}

/// The flat form that finishes a statement of this kind.
fn finishing_forms(statement: &StmtNode) -> &'static [&'static str] {
    match statement {
        StmtNode::Assign { .. } => &["const", "copy", "apply", "store"],
        StmtNode::If { .. }
        | StmtNode::While { .. }
        | StmtNode::DoWhile { .. }
        | StmtNode::For { .. } => &["branch"],
        StmtNode::Switch { .. } => &["switch"],
        StmtNode::Return(_) => &["return"],
        StmtNode::Try { .. } => &["caught"],
        StmtNode::Region { .. } => &["enter", "exit"],
        other => panic!("{other:?} owns no instruction"),
    }
}

fn form_name(form: &Form) -> &'static str {
    match form {
        Form::Const(_) => "const",
        Form::Copy => "copy",
        Form::Apply(_) => "apply",
        Form::Store(_) => "store",
        Form::Branch => "branch",
        Form::Switch => "switch",
        Form::Return => "return",
        Form::Caught => "caught",
        Form::Enter(_) => "enter",
        Form::Exit(_) => "exit",
        other => panic!("lowering emits no {other:?}"),
    }
}

/// Checks that the `Copy` at `point` is the snapshot of a region operand: the
/// operand expression belongs to a `Region` statement, the copy reads the
/// live variable into a temporary, and the region's exit releases the
/// temporary, not the live variable.
fn assert_region_operand_copy(
    function: &Function,
    lowered: &crate::pipeline::Lowered,
    point: ProgramPoint,
    operand: ExprId,
    variable: VarId,
) {
    let region = preorder(function)
        .into_iter()
        .find(|&stmt| {
            matches!(&function.stmts[stmt.index()],
                StmtNode::Region { operands, .. } if operands.contains(&operand))
        })
        .unwrap_or_else(|| panic!("the copy at {point} does not come from a region operand"));
    let copy = lowered.function.instruction(point);
    let temporary = copy.defs[0];
    assert!(lowered.function.locals[temporary.index()].temporary);
    assert_eq!(copy.uses, [lowered.variables[variable.index()]]);
    let exits: Vec<&crate::flat::Instr> = lowered
        .origins
        .iter()
        .filter(|&&(_, entity)| entity == Entity::Statement(region))
        .map(|&(at, _)| lowered.function.instruction(at))
        .filter(|instruction| matches!(instruction.form, Form::Exit(_)))
        .collect();
    assert_eq!(exits.len(), 1, "one region has one exit");
    assert_eq!(exits[0].uses, [temporary], "the exit releases the copy");
}

#[test]
fn a_statement_owns_the_instruction_that_finishes_it() {
    for case in programs::all() {
        let function = structured(&case.program);
        let lowered = lowered(&function);
        for &(point, entity) in &lowered.origins {
            let form = &lowered.function.instruction(point).form;
            match entity {
                Entity::Statement(stmt) => {
                    let statement = &function.stmts[stmt.index()];
                    assert!(
                        finishing_forms(statement).contains(&form_name(form)),
                        "{}: {statement:?} owns a {form:?}",
                        case.program.name
                    );
                }
                Entity::Expression(expr) => match (&function.exprs[expr.index()].node, form) {
                    (ExprNode::Op { op, .. }, Form::Apply(applied)) => assert_eq!(op, applied),
                    (ExprNode::Const(literal), Form::Const(made)) => assert_eq!(literal, made),
                    // A region's bare-variable operand is copied once so that
                    // the exit releases the object the enter acquired.
                    (ExprNode::Var(variable), Form::Copy) => {
                        assert_region_operand_copy(&function, &lowered, point, expr, *variable);
                    }
                    (node, form) => panic!("{}: {node:?} lowered to {form:?}", case.program.name),
                },
                Entity::Variable(_) => panic!("lowering never names a variable as an origin"),
            }
        }
    }
}

#[test]
fn a_region_statement_owns_its_enter_and_its_exit() {
    let function = structured(&programs::guarded_body().program);
    let lowered = lowered(&function);
    let region = preorder(&function)
        .into_iter()
        .find(|&stmt| matches!(function.stmts[stmt.index()], StmtNode::Region { .. }))
        .expect("the program has a region");
    let owned: Vec<&Form> = lowered
        .origins
        .iter()
        .filter(|&&(_, entity)| entity == Entity::Statement(region))
        .map(|&(point, _)| &lowered.function.instruction(point).form)
        .collect();
    assert_eq!(owned, [&Form::Enter(Op::Guard), &Form::Exit(Op::Guard)]);
}

#[test]
fn a_handler_binding_is_defined_by_an_instruction_the_try_owns() {
    let function = structured(&programs::division_guard().program);
    let lowered = lowered(&function);
    let try_statement = preorder(&function)
        .into_iter()
        .find(|&stmt| matches!(function.stmts[stmt.index()], StmtNode::Try { .. }))
        .expect("the program has a try");
    let bindings: Vec<ProgramPoint> = lowered
        .origins
        .iter()
        .filter(|&&(_, entity)| entity == Entity::Statement(try_statement))
        .map(|&(point, _)| point)
        .collect();
    assert_eq!(bindings.len(), 1, "one handler binds one variable");
    let instruction = lowered.function.instruction(bindings[0]);
    assert_eq!(instruction.form, Form::Caught);
    let binding = function.variable_named("e");
    assert_eq!(instruction.defs, [lowered.variables[binding.index()]]);
}

#[test]
fn source_variables_keep_their_order_and_temporaries_follow() {
    for case in programs::all() {
        let function = structured(&case.program);
        let lowered = lowered(&function);
        let count = function.variable_count();
        assert!(lowered.variables.len() >= count, "{}", case.program.name);
        for (index, &variable) in lowered.variables.iter().enumerate() {
            let local = &lowered.function.locals[variable.index()];
            assert_eq!(local.temporary, index >= count, "{}", case.program.name);
            if index < count {
                assert_eq!(local.name, function.vars[index].local.name);
            }
        }
        assert_eq!(lowered.function.locals.len(), lowered.variables.len());
    }
}

/// The handler entries of every region that protects `block`.
fn unwind_targets_expected(code: &crate::flat::Code, block: BlockId) -> BTreeSet<BlockId> {
    code.cfg
        .regions()
        .iter()
        .filter(|region| region.protected_blocks.contains(&block))
        .flat_map(|region| region.handlers.iter().map(|handler| handler.entry))
        .collect()
}

#[test]
fn a_throwing_block_unwinds_to_every_handler_of_every_region_that_protects_it() {
    let mut protected_throwers = 0;
    for case in programs::all() {
        let code = lowered(&structured(&case.program)).function;
        for block in code.cfg.block_ids() {
            let throws = code
                .cfg
                .block(block)
                .instructions()
                .iter()
                .any(crate::flat::Instr::may_throw);
            let unwinds: BTreeSet<BlockId> = code
                .cfg
                .outgoing(block)
                .map(|id| code.cfg.edge(id))
                .filter(|edge| edge.kind() == EdgeKind::ExceptionUnwind)
                .map(|edge| edge.target())
                .collect();
            let expected = if throws {
                unwind_targets_expected(&code, block)
            } else {
                BTreeSet::new()
            };
            assert_eq!(unwinds, expected, "{} {block}", case.program.name);
            protected_throwers += usize::from(!expected.is_empty());
        }
    }
    assert!(
        protected_throwers >= 8,
        "only {protected_throwers} blocks throw under a region"
    );
}

#[test]
fn a_try_registers_one_region_with_its_handlers_in_order() {
    let code = lowered(&structured(&programs::catch_and_finally().program)).function;
    let regions = code.cfg.regions();
    assert_eq!(regions.len(), 1);
    let kinds: Vec<_> = regions[0]
        .handlers
        .iter()
        .map(|handler| handler.kind)
        .collect();
    assert_eq!(
        kinds,
        [RegionHandlerKind::CatchAll, RegionHandlerKind::Finally]
    );
    let protected = &regions[0].protected_blocks;
    for handler in &regions[0].handlers {
        let known = handler
            .body
            .blocks()
            .expect("lowering knows every handler extent");
        assert!(known.contains(&handler.entry));
        assert!(
            known.is_disjoint(protected),
            "a handler body is not protected by its own region"
        );
    }
}

#[test]
fn nested_trys_register_the_inner_region_inside_the_outer_one() {
    let code = lowered(&structured(&programs::nested_try().program)).function;
    let mut regions: Vec<_> = code.cfg.regions().iter().collect();
    regions.sort_by_key(|region| region.protected_blocks.len());
    assert_eq!(regions.len(), 2);
    assert!(
        regions[0]
            .protected_blocks
            .is_subset(&regions[1].protected_blocks)
    );
    assert!(regions[0].protected_blocks.len() < regions[1].protected_blocks.len());
}

#[test]
fn lowering_refuses_control_that_falls_off_the_end() {
    let program = Program::new(
        "falls_off",
        &["a"],
        &[],
        vec![set("a", add(var("a"), int(1)))],
    );
    let error = lower::<Calc, _, _>(&structured(&program), CodeBuilder::new())
        .expect_err("a body with no return has no flat form");
    assert!(matches!(error, LowerError::Unsupported(_)), "{error}");
}

#[test]
fn an_endless_loop_lowers_even_without_a_statement() {
    let spin = Program::new(
        "spins",
        &["a"],
        &[],
        vec![S::Loop(vec![set("a", add(var("a"), int(1)))])],
    );
    let code = lower::<Calc, _, _>(&structured(&spin), CodeBuilder::new())
        .unwrap_or_else(|error| panic!("{error}"))
        .function;
    assert_eq!(run_flat(&code, &[0]).end, End::Diverged);

    // An empty `loop {}` lowers to a header that jumps to itself, so it
    // diverges like the loop with a body.
    let empty = Program::new("empty_spin", &[], &[], vec![S::Loop(vec![])]);
    let source = structured(&empty);
    let code = lower::<Calc, _, _>(&source, CodeBuilder::new())
        .unwrap_or_else(|error| panic!("{error}"))
        .function;
    assert_eq!(run_flat(&code, &[]).end, End::Diverged);
    assert_eq!(run_structured(&source, &[]).end, End::Diverged);

    // The empty loop lifts back to an endless loop with no statement, and
    // the lifted function verifies.
    let lifted = lifted(&code).function;
    assert!(verify::<Calc, _>(&lifted).is_ok());
    let [only] = lifted.body.as_slice() else {
        panic!("the function is one statement, got {:?}", lifted.body);
    };
    assert_eq!(
        lifted.stmts[only.index()],
        StmtNode::Loop { body: Vec::new() }
    );
    assert_eq!(run_structured(&lifted, &[]).end, End::Diverged);
}

/// Compiles `program` through an unchecked builder: a function that `verify`
/// refuses but a sloppy frontend could still hand to `lower`.
fn unchecked(program: &Program) -> Function {
    program
        .build(Builder::unchecked())
        .unwrap_or_else(|error| panic!("{error}"))
}

#[test]
fn lowering_refuses_a_transfer_with_no_target() {
    for stray in [S::Break(None), S::Continue(None), S::Goto("nowhere")] {
        let program = Program::new("stray_transfer", &[], &[], vec![stray, ret(int(0))]);
        let function = unchecked(&program);
        assert!(
            !verify::<Calc, _>(&function).is_ok(),
            "verify finds the stray transfer"
        );
        let error = lower::<Calc, _, _>(&function, CodeBuilder::new())
            .expect_err("nothing encloses the transfer");
        assert!(matches!(error, LowerError::Unsupported(_)), "{error}");
    }
}

#[test]
fn lowering_refuses_an_assignment_into_a_constant() {
    let mut builder = Builder::new();
    let ty = Ty::Int;
    let target = builder.constant(&Lit::Int(1), &ty);
    let value = builder.constant(&Lit::Int(2), &ty);
    let assign = builder.assign(target, value);
    let zero = builder.constant(&Lit::Int(0), &ty);
    let done = builder.return_(&[zero]);
    builder.set_signature(&[], &[Ty::Int]);
    let function = builder.seal(&[assign, done]);
    let error =
        lower::<Calc, _, _>(&function, CodeBuilder::new()).expect_err("a constant is not a place");
    assert!(matches!(error, LowerError::Unsupported(_)), "{error}");
}

#[test]
fn lowering_refuses_a_handler_binding_with_no_declared_type() {
    let mut builder = Builder::new();
    let bound = builder.declare_variable(&Local::named("e"), None);
    let zero = builder.constant(&Lit::Int(0), &Ty::Int);
    let raise = builder.apply(&Op::Raise, &[zero], &Ty::Unit);
    let throw = builder.expression_statement(raise);
    let one = builder.constant(&Lit::Int(1), &Ty::Int);
    let done = builder.return_(&[one]);
    let handler = Handler {
        kind: HandlerKind::CatchAll,
        binding: Some(bound),
        caught_types: &[],
        body: &[done],
    };
    let guarded = builder.try_(&[throw], &[handler], &[]);
    let zero = builder.constant(&Lit::Int(0), &Ty::Int);
    let last = builder.return_(&[zero]);
    builder.set_signature(&[], &[Ty::Int]);
    let function = builder.seal(&[guarded, last]);
    let error = lower::<Calc, _, _>(&function, CodeBuilder::new())
        .expect_err("the delivered exception has no type to carry");
    assert!(matches!(error, LowerError::Unsupported(_)), "{error}");
}

#[test]
fn lowering_refuses_a_filter_handler_with_an_empty_filter() {
    let mut builder = Builder::new();
    let zero = builder.constant(&Lit::Int(0), &Ty::Int);
    let raise = builder.apply(&Op::Raise, &[zero], &Ty::Unit);
    let throw = builder.expression_statement(raise);
    let one = builder.constant(&Lit::Int(1), &Ty::Int);
    let done = builder.return_(&[one]);
    let handler = Handler {
        kind: HandlerKind::Filter { filter_body: &[] },
        binding: None,
        caught_types: &[],
        body: &[done],
    };
    let guarded = builder.try_(&[throw], &[handler], &[]);
    let zero = builder.constant(&Lit::Int(0), &Ty::Int);
    let last = builder.return_(&[zero]);
    builder.set_signature(&[], &[Ty::Int]);
    let function = builder.seal(&[guarded, last]);
    let error = lower::<Calc, _, _>(&function, CodeBuilder::new())
        .expect_err("a filter needs a predicate to lower");
    assert!(matches!(error, LowerError::Unsupported(_)), "{error}");
}

#[test]
fn a_builder_without_a_region_protocol_makes_lowering_fail_with_its_error() {
    let function = structured(&programs::guarded_body().program);
    let error = lower::<Calc, _, _>(&function, CodeBuilder::new().refusing_regions())
        .expect_err("the bytecode has no region");
    match error {
        LowerError::Builder(CodeError::Refused(_)) => {}
        other => panic!("the builder's refusal must come through unchanged, got {other}"),
    }
    assert!(lower::<Calc, _, _>(&function, CodeBuilder::new()).is_ok());
}

#[test]
fn lowered_bytecode_keeps_loop_headers_that_test_their_condition_on_every_iteration() {
    // The condition `lt(i, n)` is recomputed in the header on each pass: a
    // header that evaluated it once would never stop.
    let code = lowered(&structured(&programs::sum_while().program)).function;
    let run = run_flat(&code, &[4]);
    assert_eq!(run.end, End::Returned(vec![crate::machine::Value::Int(6)]));
    let headers = code
        .cfg
        .block_ids()
        .filter(|&block| {
            code.cfg.incoming(block).count() == 2
                && code
                    .cfg
                    .block(block)
                    .instructions()
                    .last()
                    .is_some_and(|last| last.form == Form::Branch)
        })
        .count();
    assert_eq!(headers, 1, "one loop has one header with a back edge");
}

#[test]
fn a_region_exit_releases_what_the_enter_acquired_even_if_the_operand_changes() {
    // The body reassigns the variable that names the acquired object. The
    // object released at the exit is still the one entered.
    let program = Program::new(
        "reassigned_operand",
        &["x"],
        &["y"],
        vec![
            S::Region(
                Op::Guard,
                vec![var("x")],
                vec![set("x", add(var("x"), int(1))), set("y", int(0))],
            ),
            ret(var("x")),
        ],
    );
    let function = structured(&program);
    let expected = run_structured(&function, &[5]);
    assert_eq!(expected.events, [Event::Acquired(5), Event::Released(5)]);
    let code = lowered(&function).function;
    assert_eq!(
        run_flat(&code, &[5]),
        expected,
        "the exit released the new value"
    );
}

#[test]
fn lowering_is_deterministic() {
    for case in programs::all() {
        let function = structured(&case.program);
        let (first, second) = (lowered(&function), lowered(&function));
        assert_eq!(first.origins, second.origins, "{}", case.program.name);
        assert_eq!(first.variables, second.variables, "{}", case.program.name);
        assert_eq!(first.function.points(), second.function.points());
    }
}
