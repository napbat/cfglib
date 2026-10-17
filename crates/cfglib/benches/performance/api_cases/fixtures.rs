use std::borrow::Cow;
use std::collections::BTreeMap;

use cfglib::{
    CallInfo, Cfg, ConstantFolder, CopySource, DisplayInstr, EdgeKind, EffectInfo, ExprInstr,
    FlowControl, FlowEffect, InstrInfo, JumpTargets, Predicated, SwitchSource, ValueNumberInfo,
};

#[derive(Clone, Debug)]
pub(super) struct ApiInst {
    pub(super) flow: FlowEffect,
    pub(super) uses: Vec<u32>,
    pub(super) defs: Vec<u32>,
    pub(super) effects: Vec<u8>,
    pub(super) constant: Option<u64>,
    pub(super) operator: u8,
    pub(super) copy: bool,
    pub(super) predicate: Option<(u32, bool)>,
    pub(super) callee: Option<u32>,
    pub(super) tail_call: bool,
    pub(super) jump_target: Option<u32>,
    pub(super) label: Option<u32>,
    pub(super) switch_targets: Option<(Vec<u32>, Option<u32>)>,
}

impl ApiInst {
    pub(super) fn pure(definition: u32, uses: Vec<u32>) -> Self {
        Self {
            flow: FlowEffect::Fallthrough,
            uses,
            defs: vec![definition],
            effects: Vec::new(),
            constant: None,
            operator: 1,
            copy: false,
            predicate: None,
            callee: None,
            tail_call: false,
            jump_target: None,
            label: None,
            switch_targets: None,
        }
    }

    pub(super) fn constant(definition: u32, value: u64) -> Self {
        Self {
            constant: Some(value),
            operator: 0,
            ..Self::pure(definition, Vec::new())
        }
    }

    pub(super) fn copy(destination: u32, source: u32) -> Self {
        Self {
            copy: true,
            ..Self::pure(destination, vec![source])
        }
    }

    pub(super) fn control(flow: FlowEffect) -> Self {
        Self {
            flow,
            defs: Vec::new(),
            ..Self::pure(0, Vec::new())
        }
    }
}

impl InstrInfo for ApiInst {
    type Variable = u32;

    fn uses(&self) -> &[Self::Variable] {
        &self.uses
    }

    fn defs(&self) -> &[Self::Variable] {
        &self.defs
    }

    fn may_unwind(&self) -> bool {
        self.flow == FlowEffect::MayThrow
    }
}

impl EffectInfo for ApiInst {
    type Effect = u8;

    fn effects(&self) -> &[Self::Effect] {
        &self.effects
    }
}

impl Predicated for ApiInst {
    fn predicate(&self) -> Option<(Self::Variable, bool)> {
        self.predicate
    }
}

impl CopySource for ApiInst {
    fn as_copy(&self) -> Option<(Self::Variable, Self::Variable)> {
        (self.copy && self.defs.len() == 1 && self.uses.len() == 1)
            .then(|| (self.defs[0], self.uses[0]))
    }

    fn rewrite_use(&mut self, old: &Self::Variable, new: &Self::Variable) {
        for value in &mut self.uses {
            if value == old {
                *value = *new;
            }
        }
    }
}

impl ConstantFolder for ApiInst {
    type Const = u64;

    fn fold_constant(
        &self,
        _known: &BTreeMap<Self::Variable, Self::Const>,
    ) -> Option<(Self::Variable, Self::Const)> {
        self.defs.first().copied().zip(self.constant)
    }
}

impl ExprInstr for ApiInst {
    type Operator = u8;
    type Const = u64;

    fn as_expr(&self) -> Option<(Self::Operator, &[Self::Variable])> {
        (self.constant.is_none() && !self.defs.is_empty() && self.effects.is_empty())
            .then_some((self.operator, &self.uses))
    }

    fn as_const(&self) -> Option<Self::Const> {
        self.constant
    }
}

impl ValueNumberInfo for ApiInst {
    type Operator = u8;

    fn operator(&self) -> Self::Operator {
        self.operator
    }

    fn is_pure(&self) -> bool {
        self.effects.is_empty()
    }
}

impl DisplayInstr for ApiInst {
    fn mnemonic(&self) -> Cow<'_, str> {
        Cow::Borrowed("api-inst")
    }
}

impl FlowControl for ApiInst {
    fn flow_effect(&self) -> FlowEffect {
        self.flow
    }
}

impl CallInfo for ApiInst {
    type Callee = u32;

    fn callee(&self) -> Option<Self::Callee> {
        self.callee
    }

    fn is_tail_call(&self) -> bool {
        self.tail_call
    }
}

impl JumpTargets for ApiInst {
    type Target = u32;

    fn jump_target(&self) -> Option<Self::Target> {
        self.jump_target
    }

    fn label(&self) -> Option<Self::Target> {
        self.label
    }
}

impl SwitchSource for ApiInst {
    type Target = u32;

    fn switch_targets(&self) -> Option<(Vec<Self::Target>, Option<Self::Target>)> {
        self.switch_targets.clone()
    }
}

pub(super) fn dataflow_cfg(block_count: usize, instructions_per_block: usize) -> Cfg<ApiInst> {
    assert!(block_count > 0);
    let mut cfg = Cfg::new();
    let mut block = cfg.entry();
    let mut variable = 0_u32;
    for block_index in 0..block_count {
        for instruction_index in 0..instructions_per_block {
            let mut instruction = if instruction_index == 0 {
                ApiInst::constant(variable, u64::from(variable))
            } else {
                ApiInst::pure(variable, vec![variable.saturating_sub(1)])
            };
            if instruction_index + 1 == instructions_per_block && block_index % 8 == 0 {
                instruction.effects.push(1);
            }
            cfg.block_mut(block).push(instruction);
            variable = variable.wrapping_add(1);
        }
        if block_index + 1 < block_count {
            let next = cfg.new_block();
            cfg.add_edge(block, next, EdgeKind::Fallthrough);
            block = next;
        }
    }
    cfg
}
