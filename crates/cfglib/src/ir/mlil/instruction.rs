//! Generic typed MLIL instructions and cfglib analysis adapters.

extern crate alloc;

use alloc::borrow::Cow;
use alloc::collections::BTreeMap;
use alloc::format;
use alloc::vec::Vec;
use core::fmt;

use crate::{
    AliasPairs, CallInfo, ConstantFolder, CopySource, DisplayInstr, EffectInfo, ExprInstr,
    FlowControl, FlowEffect, InstrInfo, MemoryEvent, MemoryEventInfo,
};

use super::{AnalysisDialect, Dialect, InstructionId, MemoryDialect, TypedVariable, VariableId};

/// One typed semantic MLIL instruction.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Instruction<D: Dialect> {
    id: InstructionId,
    operation: D::Operation,
    uses: Vec<VariableId>,
    use_types: Vec<D::ValueType>,
    defs: Vec<VariableId>,
    def_types: Vec<D::ValueType>,
    effects: Vec<D::Effect>,
    flow: FlowEffect,
    may_throw: bool,
}

impl<D: Dialect> Instruction<D> {
    pub(super) fn new(
        id: InstructionId,
        operation: D::Operation,
        uses: Vec<TypedVariable<D>>,
        defs: Vec<TypedVariable<D>>,
        may_throw: bool,
    ) -> Self {
        let (uses, use_types) = split_typed(uses);
        let (defs, def_types) = split_typed(defs);
        let mut metadata = D::instruction_metadata(&operation, may_throw);
        metadata.effects.sort_unstable();
        metadata.effects.dedup();
        Self {
            id,
            operation,
            uses,
            use_types,
            defs,
            def_types,
            effects: metadata.effects,
            flow: metadata.flow,
            may_throw: metadata.may_throw,
        }
    }

    /// Returns the stable instruction identity.
    #[must_use]
    pub const fn id(&self) -> InstructionId {
        self.id
    }

    /// Returns the dialect-defined semantic operation.
    #[must_use]
    pub const fn operation(&self) -> &D::Operation {
        &self.operation
    }

    /// Returns variable uses in semantic operand order.
    #[must_use]
    pub fn uses(&self) -> &[VariableId] {
        &self.uses
    }

    /// Returns the type of every variable use in matching order.
    #[must_use]
    pub fn use_types(&self) -> &[D::ValueType] {
        &self.use_types
    }

    /// Returns variable definitions in semantic result order.
    #[must_use]
    pub fn defs(&self) -> &[VariableId] {
        &self.defs
    }

    /// Returns the type of every variable definition in matching order.
    #[must_use]
    pub fn def_types(&self) -> &[D::ValueType] {
        &self.def_types
    }

    /// Returns sorted, deduplicated observable effects.
    #[must_use]
    pub fn effects(&self) -> &[D::Effect] {
        &self.effects
    }

    /// Returns whether execution may transfer through an exceptional edge.
    #[must_use]
    pub const fn may_throw(&self) -> bool {
        self.may_throw
    }

    pub(super) fn rewrite_use(&mut self, old: VariableId, new: VariableId) {
        for variable in &mut self.uses {
            if *variable == old {
                *variable = new;
            }
        }
    }
}

impl<D: Dialect> InstrInfo for Instruction<D> {
    type Variable = VariableId;

    fn uses(&self) -> &[Self::Variable] {
        &self.uses
    }

    fn defs(&self) -> &[Self::Variable] {
        &self.defs
    }
}

impl<D: MemoryDialect> MemoryEventInfo for Instruction<D> {
    type Location = D::MemoryLocation;
    type Fence = D::MemoryFence;

    fn memory_events(
        &self,
    ) -> impl Iterator<Item = MemoryEvent<Self::Location, Self::Variable, Self::Fence>> {
        D::memory_events(self)
    }
}

impl<D: Dialect> EffectInfo for Instruction<D> {
    type Effect = D::Effect;

    fn effects(&self) -> &[Self::Effect] {
        &self.effects
    }
}

impl<D: Dialect> FlowControl for Instruction<D> {
    fn flow_effect(&self) -> FlowEffect {
        self.flow
    }
}

impl<D: AnalysisDialect> CallInfo for Instruction<D> {
    type Callee = D::Callee;

    fn callee(&self) -> Option<Self::Callee> {
        D::callee(&self.operation)
    }
}

impl<D: AnalysisDialect> CopySource for Instruction<D> {
    fn as_copy(&self) -> Option<(Self::Variable, Self::Variable)> {
        (D::is_copy(&self.operation)
            && self.effects.is_empty()
            && !self.may_throw
            && self.defs.len() == 1
            && self.uses.len() == 1)
            .then(|| (self.defs[0], self.uses[0]))
    }

    fn as_aliases(&self) -> Option<AliasPairs<'_, Self::Variable>> {
        (D::is_value_alias(&self.operation)
            && self.effects.is_empty()
            && !self.may_throw
            && !self.defs.is_empty()
            && self.defs.len() == self.uses.len())
        .then_some((&self.defs, &self.uses))
    }

    fn rewrite_use(&mut self, old: &Self::Variable, new: &Self::Variable) {
        self.rewrite_use(*old, *new);
    }
}

impl<D: AnalysisDialect> ConstantFolder for Instruction<D> {
    type Const = D::Constant;

    fn fold_constant(
        &self,
        known: &BTreeMap<Self::Variable, Self::Const>,
    ) -> Option<(Self::Variable, Self::Const)> {
        D::fold_constant(self, known)
    }

    fn fold_branch(&self, known: &BTreeMap<Self::Variable, Self::Const>) -> Option<bool> {
        D::fold_branch(self, known)
    }

    fn meet_constants(a: &Self::Const, b: &Self::Const) -> Option<Self::Const> {
        D::meet_constants(a, b)
    }
}

impl<D: AnalysisDialect> ExprInstr for Instruction<D> {
    type Operator = D::ExpressionOperator;
    type Const = D::Constant;

    fn as_expr(&self) -> Option<(Self::Operator, &[Self::Variable])> {
        let operator = D::expression_operator(&self.operation)?;
        (self.effects.is_empty() && self.defs.len() == 1).then_some((operator, self.uses()))
    }

    fn as_const(&self) -> Option<Self::Const> {
        D::constant(&self.operation)
    }
}

impl<D: Dialect> fmt::Display for Instruction<D> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        if !self.defs.is_empty() {
            for (position, variable) in self.defs.iter().enumerate() {
                if position != 0 {
                    formatter.write_str(", ")?;
                }
                variable.fmt(formatter)?;
            }
            formatter.write_str(" = ")?;
        }
        formatter.write_str(D::mnemonic(&self.operation))?;
        for (position, variable) in self.uses.iter().enumerate() {
            if position == 0 {
                formatter.write_str(" ")?;
            } else {
                formatter.write_str(", ")?;
            }
            variable.fmt(formatter)?;
        }
        Ok(())
    }
}

impl<D: Dialect> DisplayInstr for Instruction<D> {
    fn mnemonic(&self) -> Cow<'_, str> {
        Cow::Owned(format!("{self}"))
    }
}

fn split_typed<D: Dialect>(values: Vec<TypedVariable<D>>) -> (Vec<VariableId>, Vec<D::ValueType>) {
    values
        .into_iter()
        .map(|value| (value.variable, value.value_type))
        .unzip()
}
