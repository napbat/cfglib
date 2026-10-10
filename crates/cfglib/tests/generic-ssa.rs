//! Integration tests: generic ssa.

use cfglib::{Cfg, DominatorTree, EdgeKind, InstrInfo, SsaForm, SsaValue};

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
enum X86Register {
    Rax,
    Rbx,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
enum X86Flag {
    Zero,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
enum X86Variable {
    Register(X86Register),
    Flag(X86Flag),
    StackSlot {
        base: X86Register,
        displacement: i32,
    },
}

#[derive(Debug, Clone)]
struct X86Instruction {
    uses: Vec<X86Variable>,
    defs: Vec<X86Variable>,
}

impl InstrInfo for X86Instruction {
    type Variable = X86Variable;

    fn uses(&self) -> &[Self::Variable] {
        &self.uses
    }

    fn defs(&self) -> &[Self::Variable] {
        &self.defs
    }
}

fn x86_instruction(uses: Vec<X86Variable>, defs: Vec<X86Variable>) -> X86Instruction {
    X86Instruction { uses, defs }
}

#[test]
fn x86_registers_flags_and_stack_slots_keep_native_identities() {
    let rax = X86Variable::Register(X86Register::Rax);
    let rbx = X86Variable::Register(X86Register::Rbx);
    let zero = X86Variable::Flag(X86Flag::Zero);
    let stack = X86Variable::StackSlot {
        base: X86Register::Rbx,
        displacement: -16,
    };

    let mut cfg = Cfg::<X86Instruction>::new();
    let taken = cfg.new_block();
    let fallthrough = cfg.new_block();
    let merge = cfg.new_block();
    cfg.add_edge(cfg.entry(), taken, EdgeKind::ConditionalTrue);
    cfg.add_edge(cfg.entry(), fallthrough, EdgeKind::ConditionalFalse);
    cfg.add_edge(taken, merge, EdgeKind::Fallthrough);
    cfg.add_edge(fallthrough, merge, EdgeKind::Fallthrough);

    cfg.block_mut(cfg.entry()).push(x86_instruction(
        vec![stack.clone()],
        vec![rbx, zero.clone()],
    ));
    cfg.block_mut(taken)
        .push(x86_instruction(vec![], vec![rax.clone()]));
    cfg.block_mut(fallthrough)
        .push(x86_instruction(vec![zero], vec![rax.clone()]));
    cfg.block_mut(merge)
        .push(x86_instruction(vec![rax.clone()], vec![]));

    let dominators = DominatorTree::compute(&cfg);
    let ssa = SsaForm::compute(&cfg, &dominators);
    let phi = &ssa.block(merge).phis[0];

    assert_eq!(phi.result.variable, rax);
    assert_eq!(phi.operands.len(), 2);
    assert_ne!(phi.operands[0].1, phi.operands[1].1);
    assert_eq!(ssa.block(merge).instructions[0].uses[0], phi.result);
    assert_eq!(
        ssa.block(cfg.entry()).instructions[0].uses[0],
        SsaValue::live_in(stack)
    );
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
enum ShaderRegisterFile {
    Temporary,
    Input,
    Output,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
enum Component {
    X,
    Y,
    Z,
    W,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct ShaderVariable {
    file: ShaderRegisterFile,
    register: u16,
    component: Component,
}

#[derive(Debug, Clone)]
struct ShaderInstruction {
    sources: Vec<ShaderVariable>,
    destinations: Vec<ShaderVariable>,
}

impl InstrInfo for ShaderInstruction {
    type Variable = ShaderVariable;

    fn uses(&self) -> &[Self::Variable] {
        &self.sources
    }

    fn defs(&self) -> &[Self::Variable] {
        &self.destinations
    }
}

fn shader_variable(
    file: ShaderRegisterFile,
    register: u16,
    component: Component,
) -> ShaderVariable {
    ShaderVariable {
        file,
        register,
        component,
    }
}

#[test]
fn shader_register_components_are_independent_ssa_variables() {
    let input_x = shader_variable(ShaderRegisterFile::Input, 0, Component::X);
    let input_y = shader_variable(ShaderRegisterFile::Input, 0, Component::Y);
    let temporary_x = shader_variable(ShaderRegisterFile::Temporary, 2, Component::X);
    let temporary_y = shader_variable(ShaderRegisterFile::Temporary, 2, Component::Y);
    let output_x = shader_variable(ShaderRegisterFile::Output, 0, Component::X);
    let output_z = shader_variable(ShaderRegisterFile::Output, 0, Component::Z);
    let output_w = shader_variable(ShaderRegisterFile::Output, 0, Component::W);

    let mut cfg = Cfg::<ShaderInstruction>::new();
    cfg.block_mut(cfg.entry()).instructions_mut().extend([
        ShaderInstruction {
            sources: vec![input_x.clone()],
            destinations: vec![temporary_x.clone()],
        },
        ShaderInstruction {
            sources: vec![input_y],
            destinations: vec![temporary_y.clone()],
        },
        ShaderInstruction {
            sources: vec![temporary_x.clone(), temporary_y.clone()],
            destinations: vec![output_x, output_z, output_w],
        },
    ]);

    let dominators = DominatorTree::compute(&cfg);
    let ssa = SsaForm::compute(&cfg, &dominators);
    let instructions = &ssa.block(cfg.entry()).instructions;

    assert_eq!(instructions[0].uses[0], SsaValue::live_in(input_x));
    assert_eq!(instructions[2].uses[0], instructions[0].defs[0]);
    assert_eq!(instructions[2].uses[1], instructions[1].defs[0]);
    assert_eq!(instructions[2].defs.len(), 3);
    assert_eq!(ssa.max_version(&temporary_x), 1);
    assert_eq!(ssa.max_version(&temporary_y), 1);
}
