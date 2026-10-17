//! Lowering over the toy dialects.
//!
//! The flat text of each sketched shape is a golden that both toys share. The
//! goldens were checked against the output of the lowerer that this one
//! replaces; the one shape on which the two differ is named in
//! `a_do_while_is_entered_where_its_body_begins`.

extern crate alloc;

use alloc::collections::BTreeSet;
use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec;
use alloc::vec::Vec;

use crate::block::BlockId;
use crate::dataflow::ProgramPoint;
use crate::edge::{EdgeId, KindedEdge};
use crate::graph::view::DenseId;
use crate::ir::hlil::{
    Appended, Entity, FlatBuilder, FunctionBuilder, FunctionView, LowerEdge, LowerError, Lowered,
    LoweredFunction, Occurrence, StmtShape, VariableOrigin, VerificationIssue, VerificationReport,
    lower, verify,
};
use crate::region::{HandlerBody, HandlerKind as RegionHandlerKind, Region, RegionId};
use crate::test_util::golden::assert_golden;

use super::flat::{FlatKit, FlatVar, ToyEdge, ToyFlat, ToyFlatBuilder, ToyForm};
use super::sketch::{Class, Ex, Hn, Sketch, Sn, cases};
use super::{ExprId, Kit, Prim, StmtId, ToyHlil, ToyHlilBuilder, Ty, VarId};

over_toys!(
    every_case_lowers_to_its_golden,
    every_case_holds_the_lowering_invariants,
    a_while_loop_declares_and_relates_everything,
    a_do_while_is_entered_where_its_body_begins,
    a_region_exits_the_object_it_entered,
    a_try_registers_its_region_and_unwind_edges,
    nested_trys_unwind_to_every_enclosing_handler,
    a_filter_funclet_stays_out_of_the_handler_body,
    a_handler_binding_materializes_the_exception,
    a_deep_expression_lowers_without_recursion,
    control_falling_off_the_end_is_rejected,
    transfers_without_a_target_are_rejected,
    ill_formed_statements_are_rejected,
    a_builder_refusal_surfaces_as_a_builder_error,
    a_builder_that_fails_to_seal_surfaces_its_error,
);

type Lowering<D> = LoweredFunction<ToyFlat<D>, StmtId, ExprId, VarId, FlatVar>;

fn lowered<D: FlatKit>(function: &ToyHlil<D>) -> Lowering<D> {
    lower(function, ToyFlatBuilder::<D>::new()).expect("the function lowers")
}

/// The message of the error that lowering `function` reports.
fn failure<D: FlatKit>(function: &ToyHlil<D>) -> String {
    lower(function, ToyFlatBuilder::<D>::new())
        .expect_err("the function does not lower")
        .to_string()
}

/// A sketched function that the checked builder would refuse, lowered.
fn unchecked_failure<D: FlatKit>(body: Vec<Sn>) -> String {
    failure(&sketch(0, &[], body).build_unchecked::<D>())
}

fn sketch(variables: u8, parameters: &[u8], body: Vec<Sn>) -> Sketch {
    Sketch {
        variables,
        untyped: Vec::new(),
        parameters: parameters.to_vec(),
        returns: 0,
        body,
    }
}

fn case<D: FlatKit>(name: &str) -> (ToyHlil<D>, Lowering<D>) {
    let (_, sketch) = cases()
        .into_iter()
        .find(|(case, _)| *case == name)
        .expect("a named case exists");
    let function = sketch.build::<D>();
    let lowering = lowered(&function);
    (function, lowering)
}

fn every_case_lowers_to_its_golden<D: FlatKit>() {
    for (name, sketch) in cases() {
        let function = sketch.build::<D>();
        assert_golden(
            &format!("hlil/lower-{name}.flat"),
            &lowered(&function).function.dump(),
        );
    }
}

/// The relation between a structured function and its lowering that holds
/// for every shape.
fn every_case_holds_the_lowering_invariants<D: FlatKit>() {
    for (name, sketch) in cases() {
        let function = sketch.build::<D>();
        let lowering = lowered(&function);
        let flat = &lowering.function;

        // Source variables map one to one, with their metadata and declared
        // type; the temporaries follow.
        let sources = function.variable_count();
        assert_eq!(lowering.variables.len(), flat.variables.len(), "{name}");
        for (index, (&handle, variable)) in
            lowering.variables.iter().zip(&flat.variables).enumerate()
        {
            assert_eq!(handle, FlatVar::from_index(index), "{name}");
            assert_eq!(variable.temporary, index >= sources, "{name}");
            if index < sources {
                let source = VarId::from_index(index);
                assert_eq!(&variable.variable, function.variable(source), "{name}");
                assert_eq!(
                    variable.declared_type.as_ref(),
                    function.declared_type(source),
                    "{name}"
                );
            }
        }
        let parameters: Vec<_> = function
            .parameters()
            .iter()
            .map(|parameter| FlatVar::from_index(parameter.index()))
            .collect();
        assert_eq!(flat.parameters, parameters, "{name}");
        assert_eq!(flat.returns, function.returns(), "{name}");

        // Each instruction has one origin, and each temporary one definition.
        let mut defined = BTreeSet::new();
        let mut points = BTreeSet::new();
        for block in flat.cfg.block_ids() {
            for (inst_idx, instruction) in flat.cfg.block(block).instructions().iter().enumerate() {
                points.insert(ProgramPoint { block, inst_idx });
                for &def in &instruction.defs {
                    if flat.variables[def.index()].temporary {
                        assert!(defined.insert(def), "{name}: {def:?} is defined twice");
                    }
                }
            }
        }
        let origin_points: BTreeSet<_> = lowering.origins.iter().map(|(point, _)| *point).collect();
        assert_eq!(origin_points, points, "{name}");
        assert_eq!(lowering.origins.len(), points.len(), "{name}");
        for (_, origin) in &lowering.origins {
            match *origin {
                Entity::Statement(stmt) => assert!(stmt.index() < function.stmt_count(), "{name}"),
                Entity::Expression(expr) => assert!(expr.index() < function.expr_count(), "{name}"),
                Entity::Variable(_) => panic!("{name}: no instruction originates in a variable"),
            }
        }
        assert_eq!(defined.len(), flat.variables.len() - sources, "{name}");

        for edge in flat.cfg.edges() {
            assert_eq!(edge.kind(), edge.payload().kind(), "{name}");
        }
    }
}

/// `while (i < n) { if (i < 100) { break; } i = i + 1; } return i;`
fn counting_loop<D: Kit>() -> (ToyHlil<D>, [StmtId; 5]) {
    let mut builder = ToyHlilBuilder::<D>::new();
    let i = builder.int_variable(0);
    let n = builder.int_variable(1);
    builder.set_signature(&[i, n], &[D::ty(Ty::Integer)]);

    let read_i = builder.read(i);
    let limit = builder.number(100);
    let below_limit = builder.call(Prim::LessThan, &[read_i, limit], Ty::Boolean);
    let leave = builder.break_(None);
    let guard = builder.if_(below_limit, &[leave], &[]);

    let read_i = builder.read(i);
    let one = builder.number(1);
    let sum = builder.call(Prim::Add, &[read_i, one], Ty::Integer);
    let step = builder.store(i, sum);

    let read_i = builder.read(i);
    let read_n = builder.read(n);
    let condition = builder.call(Prim::LessThan, &[read_i, read_n], Ty::Boolean);
    let looping = builder.while_(condition, &[guard, step]);
    let result = builder.read(i);
    let done = builder.return_(&[result]);
    let function = builder
        .finish(&[looping, done])
        .expect("the function is valid");
    (function, [looping, guard, leave, step, done])
}

fn a_while_loop_declares_and_relates_everything<D: FlatKit>() {
    let (function, [looping, guard, leave, step, done]) = counting_loop::<D>();
    let lowering = lowered(&function);
    let flat = &lowering.function;
    assert!(flat.check().is_ok());
    assert_eq!(flat.parameters, [FlatVar(0), FlatVar(1)]);
    assert_eq!(flat.returns, [D::ty(Ty::Integer)]);

    // A header with the loop condition, a guard block, the step, the exit.
    assert_eq!(
        flat.dump(),
        "\
signature (v0, v1) returns 1
v0 source
v1 source
v2 temporary
v3 temporary
v4 temporary
v5 temporary
bb0
  -> bb1 Entry
bb1
  v2 = lt(v0, v1)
  branch(v2)
  -> bb2 True
  -> bb4 False
bb2
  v3 = const 100
  v4 = lt(v0, v3)
  branch(v4)
  -> bb3 False
  -> bb4 True
bb3
  v5 = const 1
  v0 = add(v0, v5)
  -> bb1 Jump
bb4
  return(v0)
"
    );

    let origin = |block: usize, inst_idx: usize| {
        let point = ProgramPoint {
            block: BlockId::from_index(block),
            inst_idx,
        };
        lowering
            .origins
            .iter()
            .find(|(candidate, _)| *candidate == point)
            .map(|(_, origin)| *origin)
    };
    // The branch of the loop belongs to the `while`, the branch of the guard
    // to the `if`, the write of `i = i + 1` to the assignment, and the
    // return to the return. The `break` emits nothing.
    assert_eq!(origin(1, 1), Some(Entity::Statement(looping)));
    assert_eq!(origin(2, 2), Some(Entity::Statement(guard)));
    assert_eq!(origin(3, 1), Some(Entity::Statement(step)));
    assert_eq!(origin(4, 0), Some(Entity::Statement(done)));
    assert!(
        lowering
            .origins
            .iter()
            .all(|(_, origin)| *origin != Entity::Statement(leave))
    );
    // An operation that evaluates an expression carries the expression.
    assert!(matches!(origin(1, 0), Some(Entity::Expression(_))));
    assert!(matches!(origin(2, 0), Some(Entity::Expression(_))));
}

/// `acquire(v0) { call(v0); v0 = v0 + 1; }`: the exit releases the object
/// that the enter acquired, so the variable that holds it is not written
/// between the two.
fn a_region_exits_the_object_it_entered<D: FlatKit>() {
    let (_, lowering) = case::<D>("acquire-region");
    let flat = &lowering.function;
    let instructions: Vec<_> = flat
        .cfg
        .block_ids()
        .flat_map(|block| flat.cfg.block(block).instructions().iter())
        .collect();
    let position = |wanted: fn(&ToyForm<D>) -> bool| {
        instructions
            .iter()
            .position(|instruction| wanted(&instruction.form))
            .expect("the region has both halves")
    };
    let enter = position(|form| matches!(form, ToyForm::RegionEnter(_)));
    let exit = position(|form| matches!(form, ToyForm::RegionExit(_)));
    assert_eq!(instructions[enter].uses, instructions[exit].uses);
    for instruction in &instructions[enter + 1..exit] {
        for entered in &instructions[enter].uses {
            assert!(
                !instruction.defs.contains(entered),
                "the body writes {entered:?}, which the exit releases"
            );
        }
    }
}

fn a_do_while_is_entered_where_its_body_begins<D: FlatKit>() {
    // The body of the outer loop is the inner loop, so the outer loop is
    // entered where the inner body begins. The replaced lowerer entered it at
    // the inner latch, which skipped the inner body on every repetition.
    let (_, lowering) = case::<D>("nested-do-while");
    let flat = &lowering.function;
    let add = D::op(Prim::Add);
    let body = flat
        .cfg
        .block_ids()
        .find(|&block| {
            flat.cfg
                .block(block)
                .instructions()
                .iter()
                .any(|instruction| instruction.form == ToyForm::Operation(add.clone()))
        })
        .expect("the inner body holds the increment");
    let re_entries = flat
        .cfg
        .edges()
        .filter(|edge| *edge.payload() == ToyEdge::True && edge.target() == body)
        .count();
    assert_eq!(re_entries, 2, "both latches re-enter the inner body");
}

/// The blocks that hold an instruction that may throw.
fn throwing_blocks<D: FlatKit>(flat: &ToyFlat<D>) -> Vec<BlockId> {
    flat.cfg
        .block_ids()
        .filter(|&block| {
            flat.cfg
                .block(block)
                .instructions()
                .iter()
                .any(|instruction| instruction.may_throw)
        })
        .collect()
}

/// The targets of the unwind edges that leave `from`, in creation order.
fn unwind_targets<D: FlatKit>(flat: &ToyFlat<D>, from: BlockId) -> Vec<BlockId> {
    flat.cfg
        .outgoing(from)
        .map(|edge| flat.cfg.edge(edge))
        .filter(|edge| *edge.payload() == ToyEdge::Except)
        .map(|edge| edge.target())
        .collect()
}

fn ends_in_return<D: FlatKit>(flat: &ToyFlat<D>, block: BlockId) -> bool {
    let code = flat.cfg.block(block).instructions();
    matches!(code.last().map(|last| &last.form), Some(ToyForm::Return))
}

fn a_try_registers_its_region_and_unwind_edges<D: FlatKit>() {
    let (_, lowering) = case::<D>("guarded-call");
    let flat = &lowering.function;
    let [region] = flat.cfg.regions() else {
        panic!("one region: {}", flat.dump());
    };
    assert_eq!(region.parent, None);
    let [handler] = &region.handlers[..] else {
        panic!("one handler: {}", flat.dump());
    };
    assert_eq!(handler.kind, RegionHandlerKind::CatchAll);
    assert!(
        handler
            .body
            .blocks()
            .is_some_and(|blocks| blocks.contains(&handler.entry))
    );

    // The call is the only thrower; it stands in a protected block, which
    // unwinds to the handler entry. The handler is no part of the try.
    let [thrower] = throwing_blocks(flat)[..] else {
        panic!("one throwing block: {}", flat.dump());
    };
    assert!(region.protected_blocks.contains(&thrower));
    assert_eq!(unwind_targets(flat, thrower), [handler.entry]);
    assert!(!region.protected_blocks.contains(&handler.entry));
    // The handler body is the `return 7`.
    assert!(ends_in_return(flat, handler.entry));

    // Handlers in declaration order, the finally clause last, all unwound to.
    let (_, lowering) = case::<D>("try-handlers-and-finally");
    let flat = &lowering.function;
    let [region] = flat.cfg.regions() else {
        panic!("one region: {}", flat.dump());
    };
    let kinds: Vec<_> = region.handlers.iter().map(|handler| handler.kind).collect();
    assert_eq!(
        kinds,
        [
            RegionHandlerKind::Catch,
            RegionHandlerKind::Fault,
            RegionHandlerKind::CatchAll,
            RegionHandlerKind::Finally
        ]
    );
    let [thrower] = throwing_blocks(flat)[..] else {
        panic!("one throwing block: {}", flat.dump());
    };
    assert!(region.protected_blocks.contains(&thrower));
    let entries: Vec<_> = region
        .handlers
        .iter()
        .map(|handler| handler.entry)
        .collect();
    assert_eq!(unwind_targets(flat, thrower), entries);
    for entry in entries {
        assert!(!region.protected_blocks.contains(&entry));
    }
}

fn nested_trys_unwind_to_every_enclosing_handler<D: FlatKit>() {
    let (_, lowering) = case::<D>("nested-try");
    let flat = &lowering.function;
    // The inner try registers first; the outer one follows its handlers.
    let [inner, outer] = flat.cfg.regions() else {
        panic!("two regions: {}", flat.dump());
    };
    assert!(inner.protected_blocks.is_subset(&outer.protected_blocks));
    assert!(inner.protected_blocks.len() < outer.protected_blocks.len());
    // The effect before the inner try unwinds into the outer handler only;
    // the call inside the inner try unwinds into both, inner first.
    let [outer_handler] = &outer.handlers[..] else {
        panic!("one outer handler: {}", flat.dump());
    };
    let inner_handler = inner.handlers[0].entry;
    let throwers = throwing_blocks(flat);
    assert_eq!(throwers.len(), 2, "{}", flat.dump());
    let targets: Vec<_> = throwers
        .iter()
        .map(|&block| unwind_targets(flat, block))
        .collect();
    assert!(targets.contains(&vec![outer_handler.entry]), "{targets:?}");
    assert!(
        targets.contains(&vec![inner_handler, outer_handler.entry]),
        "{targets:?}"
    );
}

fn a_filter_funclet_stays_out_of_the_handler_body<D: FlatKit>() {
    let (_, lowering) = case::<D>("filter-handler");
    let flat = &lowering.function;
    let [region] = flat.cfg.regions() else {
        panic!("one region: {}", flat.dump());
    };
    let handler = &region.handlers[0];
    let RegionHandlerKind::Filter { filter_block } = handler.kind else {
        panic!("a filter handler");
    };
    let HandlerBody::Known(blocks) = &handler.body else {
        panic!("a known body");
    };
    assert!(!blocks.contains(&filter_block));
    // The funclet is runtime-invoked: nothing flows into it, and it ends in
    // the return that the filter body states.
    assert_eq!(flat.cfg.incoming(filter_block).count(), 0);
    assert!(ends_in_return(flat, filter_block));
}

fn a_handler_binding_materializes_the_exception<D: FlatKit>() {
    let (function, lowering) = case::<D>("try-handlers-and-finally");
    let flat = &lowering.function;
    let mut caught = Vec::new();
    for block in flat.cfg.block_ids() {
        for (inst_idx, instruction) in flat.cfg.block(block).instructions().iter().enumerate() {
            if instruction.form == ToyForm::CaughtException {
                caught.push((ProgramPoint { block, inst_idx }, instruction));
            }
        }
    }
    let [(point, instruction)] = &caught[..] else {
        panic!("one binding: {}", flat.dump());
    };
    // The binding is the third variable and is typed by its declaration. The
    // instruction belongs to the `try` statement.
    assert_eq!(instruction.defs, [FlatVar(2)]);
    assert_eq!(instruction.def_types, [D::ty(Ty::Integer)]);
    assert_eq!(instruction.uses, Vec::<FlatVar>::new());
    let origin = lowering
        .origins
        .iter()
        .find(|(candidate, _)| candidate == point)
        .map(|(_, origin)| *origin);
    let Some(Entity::Statement(stmt)) = origin else {
        panic!("a statement origin: {origin:?}");
    };
    assert!(matches!(function.stmt(stmt), StmtShape::Try { .. }));
}

fn a_deep_expression_lowers_without_recursion<D: FlatKit>() {
    const DEPTH: usize = 50_000;
    let mut builder = ToyHlilBuilder::<D>::new();
    let x = builder.int_variable(0);
    builder.set_signature(&[x], &[D::ty(Ty::Integer)]);
    let mut value = builder.read(x);
    for _ in 0..DEPTH {
        let one = builder.number(1);
        value = builder.call(Prim::Add, &[value, one], Ty::Integer);
    }
    let done = builder.return_(&[value]);
    let function = builder.finish(&[done]).expect("the function is valid");

    // One constant and one addition per level, then the return.
    let flat = lowered(&function).function;
    let instructions: usize = flat
        .cfg
        .blocks()
        .map(|block| block.instructions().len())
        .sum();
    assert_eq!(instructions, 2 * DEPTH + 1);
}

fn control_falling_off_the_end_is_rejected<D: FlatKit>() {
    let condition = || Ex::Apply(Prim::LessThan, Ty::Boolean, vec![Ex::Var(0), Ex::Num(3)]);
    let rejected = [
        // The body ends without a return.
        sketch(1, &[0], vec![Sn::Assign(0, Ex::Num(1))]),
        // A loop that can exit leaves its exit open.
        sketch(
            1,
            &[0],
            vec![Sn::While(condition(), vec![Sn::Assign(0, Ex::Num(1))])],
        ),
        // So does a branch with one open arm.
        sketch(
            1,
            &[0],
            vec![Sn::If(
                condition(),
                vec![Sn::Return(Vec::new())],
                Vec::new(),
            )],
        ),
    ];
    for sketch in rejected {
        let error = failure(&sketch.build::<D>());
        assert!(error.contains("falls off the end"), "{error}");
    }
}

fn transfers_without_a_target_are_rejected<D: FlatKit>() {
    // `verify` and `lower` reject each statement in the same words, so the
    // checked builder refuses what lowering cannot translate.
    let cases: [(Vec<Sn>, &str); 6] = [
        (
            vec![Sn::Break(None)],
            "s0 breaks outside any loop or switch",
        ),
        (vec![Sn::Continue(None)], "s0 continues outside any loop"),
        (
            vec![Sn::Break(Some("nowhere"))],
            "s0 breaks unenclosing label nowhere",
        ),
        (
            vec![Sn::Continue(Some("nowhere"))],
            "s0 continues unenclosing label nowhere",
        ),
        // A labeled block is no continue target, and a plain break skips it.
        (
            vec![Sn::Labeled("block", vec![Sn::Continue(Some("block"))])],
            "s0 continues label block, which labels a block and not a loop",
        ),
        (
            vec![Sn::Labeled("block", vec![Sn::Break(None)])],
            "s0 breaks outside any loop or switch",
        ),
    ];
    for (body, expected) in cases {
        let function = sketch(0, &[], body).build_unchecked::<D>();
        let report = verify::<D, _>(&function);
        assert!(
            report.issues.iter().any(|issue| issue.message == expected),
            "{expected}: {report}"
        );
        let error = failure(&function);
        assert!(error.contains(expected), "{expected}: {error}");
    }
    let error = unchecked_failure::<D>(vec![Sn::Goto("nowhere")]);
    assert!(error.contains("goto targets the label nowhere"), "{error}");
}

fn ill_formed_statements_are_rejected<D: FlatKit>() {
    let call = || Ex::Apply(Prim::Call, Ty::Void, Vec::new());
    // `1 = 2` has no `Sn` spelling, so the toy builder is driven directly.
    let mut builder = ToyHlilBuilder::<D>::new();
    let target = builder.number(1);
    let value = builder.number(2);
    let assign = builder.assign(target, value);
    let error = failure(&builder.finish_unchecked(&[assign]));
    assert!(
        error.contains("assignment into a constant place"),
        "{error}"
    );

    let mut untyped = sketch(
        1,
        &[],
        vec![Sn::Try(
            vec![Sn::Eval(call())],
            vec![Hn {
                class: Class::CatchAll,
                binding: Some(0),
                body: vec![Sn::Return(Vec::new())],
            }],
            Vec::new(),
        )],
    );
    untyped.untyped = vec![0];
    let error = failure(&untyped.build::<D>());
    assert!(
        error.contains("handler binding v0 has no declared type"),
        "{error}"
    );

    let with_filter = |filter: Vec<Sn>| {
        sketch(
            0,
            &[],
            vec![Sn::Try(
                vec![Sn::Eval(call())],
                vec![Hn {
                    class: Class::Filter(filter),
                    binding: None,
                    body: vec![Sn::Return(Vec::new())],
                }],
                Vec::new(),
            )],
        )
    };
    let error = failure(&with_filter(Vec::new()).build::<D>());
    assert!(error.contains("non-empty filter body"), "{error}");
    let error = failure(&with_filter(vec![Sn::Eval(call())]).build::<D>());
    assert!(
        error.contains("filter body must end in a terminator"),
        "{error}"
    );
}

fn a_builder_refusal_surfaces_as_a_builder_error<D: FlatKit>() {
    let (function, _) = case::<D>("try-handlers-and-finally");
    let refused = lower(
        &function,
        ToyFlatBuilder::<D>::new().refusing_caught_exceptions(),
    );
    let Err(LowerError::Builder(report)) = refused else {
        panic!("the builder refuses the binding");
    };
    assert!(
        report.to_string().contains("a caught exception"),
        "{report}"
    );

    let (function, _) = case::<D>("acquire-region");
    let refused = lower(&function, ToyFlatBuilder::<D>::new().refusing_regions());
    let Err(LowerError::Builder(report)) = refused else {
        panic!("the builder refuses the region");
    };
    assert!(
        report.to_string().contains("a region enter or exit"),
        "{report}"
    );
}

/// A builder that does everything and refuses to seal.
struct Unsealable<D: FlatKit>(ToyFlatBuilder<D>);

impl<D: FlatKit> FlatBuilder<D> for Unsealable<D> {
    type Var = FlatVar;
    type Function = ToyFlat<D>;
    type Error = VerificationReport;

    fn entry(&self) -> BlockId {
        self.0.entry()
    }

    fn declare_variable(&mut self, origin: VariableOrigin<'_, D>) -> FlatVar {
        self.0.declare_variable(origin)
    }

    fn set_signature(&mut self, parameters: &[FlatVar], returns: &[D::Type]) {
        self.0.set_signature(parameters, returns);
    }

    fn new_block(&mut self, label: Option<&str>) -> BlockId {
        self.0.new_block(label)
    }

    fn append(
        &mut self,
        block: BlockId,
        lowered: Lowered<'_, D>,
        uses: &[Occurrence<FlatVar, D::Type>],
        defs: &[Occurrence<FlatVar, D::Type>],
    ) -> Result<Appended, VerificationReport> {
        self.0.append(block, lowered, uses, defs)
    }

    fn add_edge(
        &mut self,
        source: BlockId,
        target: BlockId,
        role: LowerEdge<'_, D::Constant>,
    ) -> EdgeId {
        self.0.add_edge(source, target, role)
    }

    fn add_region(&mut self, region: Region) -> RegionId {
        self.0.add_region(region)
    }

    fn finish(self) -> Result<ToyFlat<D>, VerificationReport> {
        Err(VerificationReport::new(vec![VerificationIssue::new(
            "the consumer will not seal this function",
        )]))
    }
}

fn a_builder_that_fails_to_seal_surfaces_its_error<D: FlatKit>() {
    let (function, _) = case::<D>("counting-loop");
    let Err(LowerError::Builder(report)) = lower(&function, Unsealable(ToyFlatBuilder::<D>::new()))
    else {
        panic!("the consumer refuses to seal");
    };
    assert!(report.to_string().contains("will not seal"), "{report}");
}
