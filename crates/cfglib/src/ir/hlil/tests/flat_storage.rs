//! Tests of the flat toy storage itself: what a hand-authored function
//! carries, what [`ToyFlat::check`] rejects, and what [`ToyFlat::dump`]
//! prints. The lowering and lifting tests rest on them.

extern crate alloc;

use alloc::vec;
use alloc::vec::Vec;

use crate::block::BlockId;
use crate::edge::{EdgeKind, KindedEdge};

use crate::ir::hlil::{FlatBuilder, Lowered};

use super::flat::{Access, FlatKit, FlatVar, ToyEdge, ToyFlatBuilder, ToyForm, ToyInst};
use super::flat_lift::int;
use super::{Imp, Prim, Sym, Ty};

over_toys!(
    every_form_is_carried_with_its_flags,
    a_valid_function_seals,
    a_root_with_an_instruction_is_rejected,
    a_root_without_an_entry_edge_is_rejected,
    a_branch_needs_both_arms,
    a_switch_needs_a_default,
    a_return_must_end_the_flow,
    a_terminator_must_end_its_block,
    an_undeclared_variable_is_rejected,
    an_empty_block_cannot_decide,
    an_unwind_edge_needs_a_thrower,
    a_builder_can_refuse_what_it_has_no_form_for,
);

fn instruction<D: FlatKit>(form: ToyForm<D>, uses: &[FlatVar], defs: &[FlatVar]) -> ToyInst<D> {
    let uses: Vec<_> = uses.iter().map(|&variable| int::<D>(variable)).collect();
    let defs: Vec<_> = defs.iter().map(|&variable| int::<D>(variable)).collect();
    ToyInst::new(form, &uses, &defs)
}

/// The synthetic root, an entry edge to a first block, and one declared
/// variable. The first block is empty and has no successor yet.
fn base<D: FlatKit>() -> (ToyFlatBuilder<D>, BlockId, FlatVar) {
    let mut builder = ToyFlatBuilder::<D>::new();
    let variable = builder.int_variable(0);
    let first = builder.new_block(None);
    builder.connect(builder.entry(), first, ToyEdge::Entry);
    (builder, first, variable)
}

fn rejection<D: FlatKit>(builder: ToyFlatBuilder<D>) -> alloc::string::String {
    builder
        .finish_unchecked()
        .check()
        .issues
        .into_iter()
        .map(|issue| issue.message)
        .collect::<Vec<_>>()
        .join("; ")
}

fn every_form_is_carried_with_its_flags<D: FlatKit>() {
    let (mut builder, first, v0) = base::<D>();
    let v1 = builder.int_variable(1);
    let add = D::op(Prim::Add);
    let call = D::op(Prim::Call);
    let forms = [
        (
            ToyForm::<D>::Constant(D::constant(7)),
            "const 7",
            vec![],
            vec![v0],
        ),
        (ToyForm::Copy, "copy", vec![v0], vec![v1]),
        (
            ToyForm::Operation(add.clone()),
            "add",
            vec![v0, v1],
            vec![v1],
        ),
        (
            ToyForm::Store(add.clone()),
            "store@add",
            vec![v0, v1],
            vec![],
        ),
        (ToyForm::Exchange, "exchange", vec![v0, v1], vec![v1, v0]),
        (ToyForm::Branch, "branch", vec![v0], vec![]),
        (
            ToyForm::BranchOperation(add.clone()),
            "branch@add",
            vec![v0, v1],
            vec![],
        ),
        (ToyForm::Switch, "switch", vec![v0], vec![]),
        (ToyForm::Jump, "jump", vec![], vec![]),
        (ToyForm::Return, "return", vec![v0], vec![]),
        (ToyForm::CaughtException, "caught", vec![], vec![v0]),
        (
            ToyForm::RegionEnter(add.clone()),
            "enter@add",
            vec![v0],
            vec![],
        ),
        (ToyForm::RegionExit(add), "exit@add", vec![v0], vec![]),
    ];
    for (form, mnemonic, uses, defs) in forms {
        let described = instruction(form, &uses, &defs).describe();
        assert!(described.contains(mnemonic), "{described}");
        assert_eq!(described.contains(" = "), !defs.is_empty(), "{described}");
    }

    let plain = instruction::<D>(ToyForm::Operation(call.clone()), &[v0], &[v1]);
    assert!(plain.may_throw);
    assert_eq!(plain.effects, [D::effect(Access::Write)]);
    assert_eq!(plain.use_types, [D::ty(Ty::Integer)]);
    assert_eq!(plain.def_types, [D::ty(Ty::Integer)]);
    assert!(plain.describe().ends_with(" !"));
    let store = instruction::<D>(ToyForm::Store(call), &[v0, v1], &[]);
    assert!(!store.may_throw);
    assert_eq!(store.effects, [D::effect(Access::Write)]);
    let pure = instruction::<D>(ToyForm::Copy, &[v0], &[v1]);
    assert!(pure.effects.is_empty() && !pure.may_throw);

    let adjusted = pure
        .with_effects(vec![D::effect(Access::Read)])
        .with_may_throw(true)
        .reading_previous_value(0)
        .materialized();
    assert_eq!(adjusted.effects, [D::effect(Access::Read)]);
    assert!(adjusted.may_throw && adjusted.materialize);
    assert_eq!(adjusted.previous_value, Some(0));

    builder.push(first, adjusted);
    assert_eq!(
        builder
            .finish_unchecked()
            .cfg
            .block(first)
            .instructions()
            .len(),
        1
    );
}

fn a_valid_function_seals<D: FlatKit>() {
    let (mut builder, first, v0) = base::<D>();
    let (left, right, join) = (
        builder.new_block(Some("left")),
        builder.new_block(None),
        builder.new_block(None),
    );
    builder.push(
        first,
        instruction(ToyForm::Constant(D::constant(1)), &[], &[v0]),
    );
    builder.push(first, instruction(ToyForm::Branch, &[v0], &[]));
    builder.connect(first, left, ToyEdge::True);
    builder.connect(first, right, ToyEdge::False);
    builder.connect(left, join, ToyEdge::Jump);
    builder.connect(right, join, ToyEdge::Fall);
    builder.push(join, instruction(ToyForm::Return, &[v0], &[]));
    builder.set_signature(&[v0], &[D::ty(Ty::Integer)]);

    let function = builder.finish().expect("the function is well formed");
    assert_eq!(
        function.dump(),
        "\
signature (v0) returns 1
v0 source
bb0
  -> bb1 Entry
bb1
  v0 = const 1
  branch(v0)
  -> bb2 True
  -> bb3 False
bb2 \"left\"
  -> bb4 Jump
bb3
  -> bb4 Fall
bb4
  return(v0)
"
    );
}

fn a_root_with_an_instruction_is_rejected<D: FlatKit>() {
    let (mut builder, _, _) = base::<D>();
    builder.push(builder.entry(), instruction(ToyForm::Jump, &[], &[]));
    assert!(rejection(builder).contains("the synthetic root holds an instruction"));
}

fn a_root_without_an_entry_edge_is_rejected<D: FlatKit>() {
    let mut builder = ToyFlatBuilder::<D>::new();
    let first = builder.new_block(None);
    builder.connect(builder.entry(), first, ToyEdge::Fall);
    assert!(rejection(builder).contains("exactly one entry edge"));
}

fn a_branch_needs_both_arms<D: FlatKit>() {
    let (mut builder, first, v0) = base::<D>();
    let target = builder.new_block(None);
    builder.push(first, instruction(ToyForm::Branch, &[v0], &[]));
    builder.connect(first, target, ToyEdge::True);
    assert!(rejection(builder).contains("branches without one true and one false edge"));
}

fn a_switch_needs_a_default<D: FlatKit>() {
    let (mut builder, first, v0) = base::<D>();
    let target = builder.new_block(None);
    builder.push(first, instruction(ToyForm::Switch, &[v0], &[]));
    builder.connect(first, target, ToyEdge::Case(vec![D::constant(1)]));
    assert!(rejection(builder).contains("dispatches without one default edge"));
}

fn a_return_must_end_the_flow<D: FlatKit>() {
    let (mut builder, first, v0) = base::<D>();
    let target = builder.new_block(None);
    builder.push(first, instruction(ToyForm::Return, &[v0], &[]));
    builder.connect(first, target, ToyEdge::Fall);
    assert!(rejection(builder).contains("returns and then continues"));
}

fn a_terminator_must_end_its_block<D: FlatKit>() {
    let (mut builder, first, v0) = base::<D>();
    builder.push(first, instruction(ToyForm::Return, &[v0], &[]));
    builder.push(first, instruction(ToyForm::Jump, &[], &[]));
    assert!(rejection(builder).contains("terminates before the block ends"));
}

fn an_undeclared_variable_is_rejected<D: FlatKit>() {
    let (mut builder, first, _) = base::<D>();
    let stray = FlatVar(9);
    builder.push(first, instruction(ToyForm::Return, &[stray], &[]));
    assert!(rejection(builder).contains("undeclared"));
}

fn an_empty_block_cannot_decide<D: FlatKit>() {
    let (mut builder, first, _) = base::<D>();
    let (left, right) = (builder.new_block(None), builder.new_block(None));
    builder.connect(first, left, ToyEdge::Fall);
    builder.connect(first, right, ToyEdge::Fall);
    assert!(rejection(builder).contains("decides between several edges"));
}

fn an_unwind_edge_needs_a_thrower<D: FlatKit>() {
    let (mut builder, first, v0) = base::<D>();
    let handler = builder.new_block(None);
    builder.push(first, instruction(ToyForm::Copy, &[v0], &[v0]));
    builder.connect(first, handler, ToyEdge::Except);
    assert!(rejection(builder).contains("exceptional edge and nothing that throws"));
}

fn a_builder_can_refuse_what_it_has_no_form_for<D: FlatKit>() {
    let (mut builder, first, v0) = base::<D>();
    let uses = [int::<D>(v0)];
    let defs = [int::<D>(v0)];
    let acquire = D::op(Prim::Acquire);
    let release = D::op(Prim::Release);
    assert!(
        builder
            .append(first, Lowered::CaughtException, &[], &defs)
            .is_ok()
    );
    assert!(
        builder
            .append(first, Lowered::RegionEnter(&acquire), &uses, &[])
            .is_ok()
    );
    // The toy protocol pairs an acquire with its release only.
    assert!(
        builder
            .append(first, Lowered::RegionEnter(&release), &uses, &[])
            .is_err()
    );

    let mut caught = ToyFlatBuilder::<D>::new().refusing_caught_exceptions();
    assert!(
        caught
            .append(first, Lowered::CaughtException, &[], &defs)
            .is_err()
    );
    let mut regions = ToyFlatBuilder::<D>::new().refusing_regions();
    assert!(
        regions
            .append(first, Lowered::RegionExit(&acquire), &uses, &[])
            .is_err()
    );
}

#[test]
fn an_edge_takes_its_kind_from_the_dialect() {
    assert_eq!(ToyEdge::<Imp>::Jump.kind(), EdgeKind::Jump);
    assert_eq!(ToyEdge::<Sym>::Jump.kind(), EdgeKind::Unconditional);
    assert_eq!(ToyEdge::<Imp>::Except.kind(), EdgeKind::ExceptionHandler);
    assert_eq!(ToyEdge::<Sym>::Except.kind(), EdgeKind::ExceptionUnwind);
    for (edge, kind) in [
        (ToyEdge::<Imp>::Entry, EdgeKind::Fallthrough),
        (ToyEdge::Fall, EdgeKind::Fallthrough),
        (ToyEdge::Default, EdgeKind::Fallthrough),
        (ToyEdge::True, EdgeKind::ConditionalTrue),
        (ToyEdge::False, EdgeKind::ConditionalFalse),
        (ToyEdge::Case(vec![1]), EdgeKind::SwitchCase),
    ] {
        assert_eq!(edge.kind(), kind, "{edge:?}");
    }
}
