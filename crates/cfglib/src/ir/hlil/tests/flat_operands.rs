//! The lift reads what an instruction reads, writes and does through the
//! operand context of its flat function, and names blocks through
//! [`FlatFunction::block_label`].
//!
//! [`Pooled`] is a toy flat function whose instructions are runs of its
//! pools and whose blocks store no label, the layout of a consumer that keeps
//! one operand pool per function and derives its block names. Lifting it
//! prints exactly what lifting the [`ToyFlat`] it was copied from prints.

extern crate alloc;

use alloc::borrow::Cow;
use alloc::format;
use alloc::vec::Vec;
use core::ops::Range;

use crate::block::BlockId;
use crate::cfg::Cfg;
use crate::flow::{FlowControl, FlowEffect};
use crate::ir::hlil::{
    FlatBuilder, FlatFunction, InstructionShape, LiftInstruction, LiftMetadata, LiftOperands, lift,
    pseudocode, verify,
};
use crate::{EffectOperands, InstrOperands};

use super::Prim;
use super::ToyHlilBuilder;
use super::flat::{FlatKit, FlatVar, ToyEdge, ToyFlat, ToyFlatBuilder, ToyForm};
use super::flat_lift::{boolean, int, lifted_text};

over_toys!(
    a_pooled_function_lifts_as_its_owning_copy_does,
    a_derived_block_name_labels_the_goto_target,
    an_absent_block_name_falls_back_to_the_block_index,
);

/// One instruction of a [`Pooled`] function: its form and flags, and runs of
/// the function's pools in place of its operands.
#[derive(Debug, Clone)]
struct Row<D: FlatKit> {
    form: ToyForm<D>,
    uses: Range<usize>,
    defs: Range<usize>,
    effects: Range<usize>,
    may_throw: bool,
    previous_value: Option<usize>,
    materialize: bool,
}

impl<D: FlatKit> FlowControl for Row<D> {
    fn flow_effect(&self) -> FlowEffect {
        match self.form {
            ToyForm::Branch | ToyForm::BranchOperation(_) => FlowEffect::ConditionalJump,
            ToyForm::Switch => FlowEffect::IndirectJump,
            ToyForm::Jump => FlowEffect::Jump,
            ToyForm::Return => FlowEffect::Return,
            _ => FlowEffect::Fallthrough,
        }
    }
}

impl<D: FlatKit> LiftInstruction<D> for Row<D> {
    fn lift(&self) -> InstructionShape<D::Op, D::Constant> {
        match &self.form {
            ToyForm::Constant(constant) => InstructionShape::Constant(constant.clone()),
            ToyForm::Copy => InstructionShape::Copy,
            ToyForm::Operation(op) => InstructionShape::Operation(op.clone()),
            ToyForm::Store(op) => InstructionShape::Store {
                location: op.clone(),
            },
            ToyForm::Exchange => InstructionShape::ParallelCopy,
            ToyForm::Branch => InstructionShape::Branch,
            ToyForm::BranchOperation(op) => InstructionShape::BranchOperation(op.clone()),
            ToyForm::Switch => InstructionShape::Switch,
            ToyForm::Jump => InstructionShape::ControlFlow,
            ToyForm::Return => InstructionShape::Return,
            ToyForm::CaughtException | ToyForm::RegionEnter(_) | ToyForm::RegionExit(_) => {
                unreachable!("the fixtures here hold no handler or region")
            }
        }
    }

    fn previous_value_operand(&self) -> Option<usize> {
        self.previous_value
    }

    fn materialize_value(&self) -> bool {
        self.materialize
    }
}

/// The operands of every instruction of one [`Pooled`] function.
#[derive(Debug)]
struct Pools<D: FlatKit> {
    variables: Vec<FlatVar>,
    /// Parallel to `variables`.
    types: Vec<D::Type>,
    effects: Vec<D::Effect>,
}

impl<D: FlatKit> InstrOperands<Row<D>> for Pools<D> {
    type Variable = FlatVar;

    fn uses<'a>(&'a self, instruction: &'a Row<D>) -> &'a [FlatVar] {
        &self.variables[instruction.uses.clone()]
    }

    fn defs<'a>(&'a self, instruction: &'a Row<D>) -> &'a [FlatVar] {
        &self.variables[instruction.defs.clone()]
    }

    fn may_unwind(&self, instruction: &Row<D>) -> bool {
        instruction.may_throw
    }
}

impl<D: FlatKit> EffectOperands<Row<D>> for Pools<D> {
    type Effect = D::Effect;

    fn effects<'a>(&'a self, instruction: &'a Row<D>) -> &'a [D::Effect] {
        &self.effects[instruction.effects.clone()]
    }
}

impl<D: FlatKit> LiftOperands<D, Row<D>> for Pools<D> {
    fn use_types<'a>(&'a self, instruction: &'a Row<D>) -> &'a [D::Type] {
        &self.types[instruction.uses.clone()]
    }

    fn def_types<'a>(&'a self, instruction: &'a Row<D>) -> &'a [D::Type] {
        &self.types[instruction.defs.clone()]
    }
}

/// How a [`Pooled`] function names its blocks.
#[derive(Clone, Copy)]
enum Names {
    /// The labels its source stored.
    Source,
    /// `b{index}`, derived and stored nowhere.
    Derived,
    /// No name, so the lift falls back to `.bb{index}`.
    Absent,
}

/// A flat function whose instructions own nothing and whose blocks store no
/// label, copied from a [`ToyFlat`].
struct Pooled<'a, D: FlatKit> {
    source: &'a ToyFlat<D>,
    cfg: Cfg<Row<D>, ToyEdge<D>>,
    pools: Pools<D>,
    names: Names,
}

/// Appends `items` to `pool` and returns the run they occupy.
fn run<T: Clone>(pool: &mut Vec<T>, items: &[T]) -> Range<usize> {
    let start = pool.len();
    pool.extend_from_slice(items);
    start..pool.len()
}

impl<'a, D: FlatKit> Pooled<'a, D> {
    /// Moves every operand of `source` into pools, keeping its blocks and
    /// edges and dropping its block labels.
    fn of(source: &'a ToyFlat<D>, names: Names) -> Self {
        let mut pools = Pools {
            variables: Vec::new(),
            types: Vec::new(),
            effects: Vec::new(),
        };
        let mut cfg = Cfg::with_edge_payload();
        for block in source.cfg.block_ids().skip(1) {
            assert_eq!(cfg.new_block(), block, "the fixture removes no block");
        }
        for block in source.cfg.block_ids() {
            for instruction in source.cfg.block(block).instructions() {
                let uses = run(&mut pools.variables, &instruction.uses);
                pools.types.extend_from_slice(&instruction.use_types);
                let defs = run(&mut pools.variables, &instruction.defs);
                pools.types.extend_from_slice(&instruction.def_types);
                let effects = run(&mut pools.effects, &instruction.effects);
                cfg.block_mut(block).push(Row {
                    form: instruction.form.clone(),
                    uses,
                    defs,
                    effects,
                    may_throw: instruction.may_throw,
                    previous_value: instruction.previous_value,
                    materialize: instruction.materialize,
                });
            }
            assert!(cfg.block(block).label().is_none());
        }
        for edge in source.cfg.edge_ids() {
            let edge = source.cfg.edge(edge);
            cfg.add_edge_with_payload(
                edge.source(),
                edge.target(),
                edge.kind(),
                edge.payload().clone(),
            );
        }
        Self {
            source,
            cfg,
            pools,
            names,
        }
    }
}

impl<D: FlatKit> FlatFunction<D> for Pooled<'_, D> {
    type Var = FlatVar;
    type Inst = Row<D>;
    type Edge = ToyEdge<D>;
    type Operands = Pools<D>;

    fn cfg(&self) -> &Cfg<Row<D>, ToyEdge<D>> {
        &self.cfg
    }

    fn operands(&self) -> &Pools<D> {
        &self.pools
    }

    fn variable_count(&self) -> usize {
        self.source.variable_count()
    }

    fn variable(&self, variable: FlatVar) -> &D::Variable {
        self.source.variable(variable)
    }

    fn parameters(&self) -> &[FlatVar] {
        self.source.parameters()
    }

    fn returns(&self) -> &[D::Type] {
        self.source.returns()
    }

    fn block_label(&self, block: BlockId) -> Option<Cow<'_, str>> {
        match self.names {
            Names::Source => self.source.cfg.block(block).label().map(Cow::Borrowed),
            Names::Derived => Some(Cow::Owned(format!("b{}", block.index()))),
            Names::Absent => None,
        }
    }
}

/// The pseudocode of the lifted `function`, after checking that it verifies.
fn pooled_text<D: FlatKit>(function: &Pooled<'_, D>) -> alloc::string::String {
    let lifted = lift::<D, _, _>(function, ToyHlilBuilder::<D>::new(), LiftMetadata::Preserve)
        .expect("the pooled function lifts");
    assert!(
        verify(&lifted.function).is_ok(),
        "{}",
        verify(&lifted.function)
    );
    pseudocode(&lifted.function)
}

/// A cycle entered at either `left` or `right`, so `right` keeps a label
/// and a `goto`. `left` and `right` call, which writes and throws.
fn two_entry_cycle<D: FlatKit>() -> ToyFlat<D> {
    let mut flat = ToyFlatBuilder::<D>::new();
    let start = flat.new_block(Some("start"));
    let left = flat.new_block(Some("left"));
    let right = flat.new_block(Some("right"));
    let exit = flat.new_block(Some("exit"));
    let [x, y] = [0, 0].map(|number| flat.int_variable(number));
    let [c1, c2] = [1, 1].map(|number| flat.int_variable(number));
    let [one, sum] = [0, 0].map(|number| flat.int_variable(number));
    flat.operate(
        start,
        Prim::LessThan,
        &[int::<D>(x), int::<D>(y)],
        &[boolean::<D>(c1)],
    );
    flat.branch(start, c1);
    flat.literal(left, 1, one);
    flat.operate(
        left,
        Prim::Add,
        &[int::<D>(x), int::<D>(one)],
        &[int::<D>(sum)],
    );
    flat.operate(left, Prim::Call, &[], &[]);
    flat.operate(
        left,
        Prim::LessThan,
        &[int::<D>(sum), int::<D>(y)],
        &[boolean::<D>(c2)],
    );
    flat.branch(left, c2);
    flat.operate(right, Prim::Call, &[], &[]);
    flat.give_back(exit, &[x]);
    flat.start(start);
    flat.connect(start, left, ToyEdge::True);
    flat.connect(start, right, ToyEdge::False);
    flat.connect(left, right, ToyEdge::True);
    flat.connect(left, exit, ToyEdge::False);
    flat.connect(right, left, ToyEdge::Fall);
    flat.takes(&[x, y]);
    flat.finish().expect("the cycle is well formed")
}

fn a_pooled_function_lifts_as_its_owning_copy_does<D: FlatKit>() {
    let source = two_entry_cycle::<D>();
    let expected = lifted_text(&source);
    assert!(expected.contains("goto right;"), "{expected}");
    assert_eq!(pooled_text(&Pooled::of(&source, Names::Source)), expected);
}

fn a_derived_block_name_labels_the_goto_target<D: FlatKit>() {
    let source = two_entry_cycle::<D>();
    let right = BlockId::from_index(3);
    let expected = lifted_text(&source).replace("right", &format!("b{}", right.index()));
    assert_eq!(pooled_text(&Pooled::of(&source, Names::Derived)), expected);
}

fn an_absent_block_name_falls_back_to_the_block_index<D: FlatKit>() {
    let source = two_entry_cycle::<D>();
    let right = BlockId::from_index(3);
    let expected = lifted_text(&source).replace("right", &format!(".bb{}", right.index()));
    assert_eq!(pooled_text(&Pooled::of(&source, Names::Absent)), expected);
}
