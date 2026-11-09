use std::borrow::Cow;

use cfglib::{
    Cfg, DisplayInstr, EdgeKind, EffectInfo, FlowControl, FlowEffect, InstrInfo, JumpTargets,
};

#[derive(Clone, Debug)]
pub(super) struct ApiInst {
    pub(super) flow: FlowEffect,
    pub(super) uses: Vec<u32>,
    pub(super) defs: Vec<u32>,
    pub(super) effects: Vec<u8>,
    pub(super) jump_target: Option<u32>,
    pub(super) label: Option<u32>,
}

impl ApiInst {
    pub(super) fn pure(definition: u32, uses: Vec<u32>) -> Self {
        Self {
            flow: FlowEffect::Fallthrough,
            uses,
            defs: vec![definition],
            effects: Vec::new(),
            jump_target: None,
            label: None,
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

impl JumpTargets for ApiInst {
    type Target = u32;

    fn jump_target(&self) -> Option<Self::Target> {
        self.jump_target
    }

    fn label(&self) -> Option<Self::Target> {
        self.label
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
                ApiInst::pure(variable, Vec::new())
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
