//! The lift side of the flat toy storage: the input traits for [`ToyFlat`],
//! the lift policy of each toy dialect, and shorthands that author a flat
//! function by hand.
//!
//! The two toys differ in what a policy can express. `Imp` has an effect
//! enum and the exact inversion of `below`. `Sym` names its effects with
//! strings and has no logical negation.

extern crate alloc;

use alloc::string::String;
use alloc::vec::Vec;

use crate::block::BlockId;
use crate::cfg::Cfg;
use crate::edge::EdgeId;
use crate::flow::{FlowControl, FlowEffect};
use crate::graph::view::DenseId;
use crate::ir::hlil::{
    FlatBuilder, FlatFunction, InstructionShape, LiftDialect, LiftEdge, LiftInstruction,
    LiftMetadata, LiftedFunction, Occurrence, lift, pseudocode, verify,
};
use crate::{EffectInfo, InstrInfo};

use super::flat::{Access, FlatKit, FlatVar, ToyEdge, ToyFlat, ToyFlatBuilder, ToyForm, ToyInst};
use super::imp::{ImpOp, ImpType};
use super::{ExprId, Imp, Kit, Prim, StmtId, Sym, ToyHlil, ToyHlilBuilder, Ty, VarId};

impl LiftDialect for Imp {
    type Effect = Access;

    fn void_type() -> ImpType {
        ImpType::Void
    }

    fn temporary_variable(_ty: &ImpType) -> Option<u8> {
        Some(1)
    }

    fn logical_not() -> Option<ImpOp> {
        Some(ImpOp::Not)
    }

    fn negate_operation(op: &ImpOp) -> Option<ImpOp> {
        match op {
            ImpOp::Below => Some(ImpOp::AtLeast),
            ImpOp::AtLeast => Some(ImpOp::Below),
            _ => None,
        }
    }

    /// Reads pass reads; nothing passes a write or a potential throw.
    fn evaluation_commutes(
        moved_effects: &[Access],
        moved_may_throw: bool,
        crossed_effects: &[Access],
        crossed_may_throw: bool,
    ) -> bool {
        !moved_may_throw
            && !crossed_may_throw
            && moved_effects.iter().all(|effect| *effect == Access::Read)
            && crossed_effects.iter().all(|effect| *effect == Access::Read)
    }
}

impl LiftDialect for Sym {
    type Effect = &'static str;

    fn void_type() {}

    fn temporary_variable(_ty: &()) -> Option<(u8, Option<u8>)> {
        Some((1, None))
    }

    fn negate_operation(op: &&'static str) -> Option<&'static str> {
        match *op {
            "below" => Some("at-least"),
            "at-least" => Some("below"),
            _ => None,
        }
    }

    /// Reads pass reads; nothing passes a write or a potential throw.
    fn evaluation_commutes(
        moved_effects: &[&'static str],
        moved_may_throw: bool,
        crossed_effects: &[&'static str],
        crossed_may_throw: bool,
    ) -> bool {
        !moved_may_throw
            && !crossed_may_throw
            && moved_effects.iter().all(|effect| *effect == "read")
            && crossed_effects.iter().all(|effect| *effect == "read")
    }
}

impl<D: LiftDialect> InstrInfo for ToyInst<D> {
    type Variable = FlatVar;

    fn uses(&self) -> &[FlatVar] {
        &self.uses
    }

    fn defs(&self) -> &[FlatVar] {
        &self.defs
    }

    fn may_unwind(&self) -> bool {
        self.may_throw
    }
}

impl<D: LiftDialect> EffectInfo for ToyInst<D> {
    type Effect = D::Effect;

    fn effects(&self) -> &[D::Effect] {
        &self.effects
    }
}

impl<D: LiftDialect> FlowControl for ToyInst<D> {
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

impl<D: FlatKit> LiftInstruction<D> for ToyInst<D> {
    fn lift(&self) -> InstructionShape<D::Op, D::Constant> {
        match &self.form {
            ToyForm::Constant(constant) => InstructionShape::Constant(constant.clone()),
            ToyForm::Copy => InstructionShape::Copy,
            ToyForm::Operation(op) | ToyForm::RegionEnter(op) => {
                InstructionShape::Operation(op.clone())
            }
            ToyForm::Store(op) => InstructionShape::Store {
                location: op.clone(),
            },
            ToyForm::Exchange => InstructionShape::ParallelCopy,
            ToyForm::Branch => InstructionShape::Branch,
            ToyForm::BranchOperation(op) => InstructionShape::BranchOperation(op.clone()),
            ToyForm::Switch => InstructionShape::Switch,
            ToyForm::Jump => InstructionShape::ControlFlow,
            ToyForm::Return => InstructionShape::Return,
            ToyForm::CaughtException => InstructionShape::Operation(D::op(Prim::Caught)),
            ToyForm::RegionExit(_) => InstructionShape::Operation(D::op(Prim::Release)),
        }
    }

    fn use_types(&self) -> &[D::Type] {
        &self.use_types
    }

    fn def_types(&self) -> &[D::Type] {
        &self.def_types
    }

    fn previous_value_operand(&self) -> Option<usize> {
        self.previous_value
    }

    fn materialize_value(&self) -> bool {
        self.materialize
    }
}

impl<D: FlatKit> LiftEdge<D> for ToyEdge<D> {
    fn case_values(&self) -> &[D::Constant] {
        match self {
            Self::Case(values) => values,
            _ => &[],
        }
    }
}

impl<D: FlatKit> FlatFunction<D> for ToyFlat<D> {
    type Var = FlatVar;
    type Inst = ToyInst<D>;
    type Edge = ToyEdge<D>;

    fn cfg(&self) -> &Cfg<ToyInst<D>, ToyEdge<D>> {
        &self.cfg
    }

    fn variable_count(&self) -> usize {
        self.variables.len()
    }

    fn variable(&self, variable: FlatVar) -> &D::Variable {
        &self.variables[variable.index()].variable
    }

    fn parameters(&self) -> &[FlatVar] {
        &self.parameters
    }

    fn returns(&self) -> &[D::Type] {
        &self.returns
    }
}

/// A function lifted into the toy structured storage.
pub(crate) type Lifted<D> = LiftedFunction<ToyHlil<D>, StmtId, ExprId, VarId>;

/// Lifts `source` and keeps the correspondence.
pub(crate) fn lifted<D: FlatKit>(source: &ToyFlat<D>) -> Lifted<D> {
    lift::<D, _, _>(source, ToyHlilBuilder::<D>::new(), LiftMetadata::Preserve)
        .expect("the flat function lifts")
}

/// The pseudocode of the lifted `source`, after checking that the lifted
/// function verifies.
pub(crate) fn lifted_text<D: FlatKit>(source: &ToyFlat<D>) -> String {
    let function = lifted(source).function;
    assert!(verify(&function).is_ok(), "{}", verify(&function));
    pseudocode(&function)
}

/// A function under construction that has one block, `body`, entered from the
/// root.
pub(crate) fn one_block<D: FlatKit>() -> (ToyFlatBuilder<D>, BlockId) {
    let mut flat = ToyFlatBuilder::<D>::new();
    let block = flat.new_block(Some("body"));
    flat.start(block);
    (flat, block)
}

/// An integer-typed occurrence of `variable`.
pub(crate) fn int<D: Kit>(variable: FlatVar) -> Occurrence<FlatVar, D::Type> {
    Occurrence {
        variable,
        ty: D::ty(Ty::Integer),
    }
}

/// A boolean-typed occurrence of `variable`.
pub(crate) fn boolean<D: Kit>(variable: FlatVar) -> Occurrence<FlatVar, D::Type> {
    Occurrence {
        variable,
        ty: D::ty(Ty::Boolean),
    }
}

/// Shorthands that author a flat function block by block. Each appends one
/// instruction to `block`, and the flat form of an operation follows from its
/// [`Prim`].
impl<D: FlatKit> ToyFlatBuilder<D> {
    /// Appends an instruction of `form`.
    pub(crate) fn emit(
        &mut self,
        block: BlockId,
        form: ToyForm<D>,
        uses: &[Occurrence<FlatVar, D::Type>],
        defs: &[Occurrence<FlatVar, D::Type>],
    ) {
        self.push(block, ToyInst::new(form, uses, defs));
    }

    /// Appends the operation `prim`.
    pub(crate) fn operate(
        &mut self,
        block: BlockId,
        prim: Prim,
        uses: &[Occurrence<FlatVar, D::Type>],
        defs: &[Occurrence<FlatVar, D::Type>],
    ) {
        self.emit(block, ToyForm::Operation(D::op(prim)), uses, defs);
    }

    /// Appends `definition = value`.
    pub(crate) fn literal(&mut self, block: BlockId, value: i64, definition: FlatVar) {
        self.emit(
            block,
            ToyForm::Constant(D::constant(value)),
            &[],
            &[int::<D>(definition)],
        );
    }

    /// Appends `definition = copy(source)`.
    pub(crate) fn copy(&mut self, block: BlockId, source: FlatVar, definition: FlatVar) {
        self.emit(
            block,
            ToyForm::Copy,
            &[int::<D>(source)],
            &[int::<D>(definition)],
        );
    }

    /// Appends a branch on the boolean `condition`.
    pub(crate) fn branch(&mut self, block: BlockId, condition: FlatVar) {
        self.emit(block, ToyForm::Branch, &[boolean::<D>(condition)], &[]);
    }

    /// Appends a pure jump instruction.
    pub(crate) fn jump(&mut self, block: BlockId) {
        self.emit(block, ToyForm::Jump, &[], &[]);
    }

    /// Appends a return of the integers `values`.
    pub(crate) fn give_back(&mut self, block: BlockId, values: &[FlatVar]) {
        let uses: Vec<_> = values.iter().map(|&value| int::<D>(value)).collect();
        self.emit(block, ToyForm::Return, &uses, &[]);
    }

    /// Adds the entry edge from the synthetic root to `first`.
    pub(crate) fn start(&mut self, first: BlockId) -> EdgeId {
        let root = self.entry();
        self.connect(root, first, ToyEdge::Entry)
    }

    /// Declares the integer `parameters` as the signature, returning one
    /// integer.
    pub(crate) fn takes(&mut self, parameters: &[FlatVar]) {
        self.set_signature(parameters, &[D::ty(Ty::Integer)]);
    }
}
