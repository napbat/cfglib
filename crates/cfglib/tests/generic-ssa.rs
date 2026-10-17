//! Integration tests: generic ssa.

use std::collections::{BTreeMap, BTreeSet};

use cfglib::{
    Cfg, DominatorTree, EdgeKind, InstrInfo, PhiPlacements, ProgramPoint, SsaForm, SsaPhi, SsaValue,
};

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
    assert_eq!(phi.operands().len(), 2);
    assert_ne!(phi.operands()[0].1, phi.operands()[1].1);
    assert_eq!(ssa.block(merge).instructions[0].uses()[0], phi.result);
    assert_eq!(
        ssa.block(cfg.entry()).instructions[0].uses()[0],
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

    assert_eq!(instructions[0].uses()[0], SsaValue::live_in(input_x));
    assert_eq!(instructions[2].uses()[0], instructions[0].defs()[0]);
    assert_eq!(instructions[2].uses()[1], instructions[1].defs()[0]);
    assert_eq!(instructions[2].defs().len(), 3);
    assert_eq!(ssa.max_version(&temporary_x), 1);
    assert_eq!(ssa.max_version(&temporary_y), 1);
}

#[test]
fn a_function_whose_first_instruction_is_a_branch_target_merges_its_live_in_values() {
    // `loop: dec rax; jnz loop`, with the loop at the function's first
    // instruction. The entry block is its own predecessor, so rax and the
    // zero flag each take a phi there, and each merges the value the function
    // starts with as well as the value the back edge carries.
    let rax = X86Variable::Register(X86Register::Rax);
    let zero = X86Variable::Flag(X86Flag::Zero);
    let mut cfg = Cfg::<X86Instruction>::new();
    let exit = cfg.new_block();
    let entry = cfg.entry();
    cfg.add_edge(entry, entry, EdgeKind::ConditionalTrue);
    cfg.add_edge(entry, exit, EdgeKind::ConditionalFalse);
    cfg.block_mut(entry).push(x86_instruction(
        vec![rax.clone()],
        vec![rax.clone(), zero.clone()],
    ));
    cfg.block_mut(exit)
        .push(x86_instruction(vec![rax.clone(), zero.clone()], vec![]));

    let ssa = SsaForm::compute(&cfg, &DominatorTree::compute(&cfg));
    let decrement = &ssa.block(entry).instructions[0];
    let end_of_entry = ProgramPoint {
        block: entry,
        inst_idx: 1,
    };

    assert_eq!(ssa.block(entry).phis.len(), 2);
    for variable in [&rax, &zero] {
        let phi = ssa
            .block(entry)
            .phis
            .iter()
            .find(|phi| phi.result.variable == *variable)
            .expect("the entry merges every variable it defines");
        assert_eq!(
            phi.start_value(),
            Some(SsaValue::live_in(variable.clone())),
            "the value the function starts with joins the phi"
        );
        assert_eq!(
            phi.operands(),
            [(end_of_entry, decrement.def_of(variable).unwrap().clone())],
            "the back edge is the only operand with a departure point"
        );
    }
    let rax_phi = ssa
        .block(entry)
        .phis
        .iter()
        .find(|phi| phi.result.variable == rax)
        .unwrap();
    assert_eq!(decrement.use_of(&rax), Some(&rax_phi.result));
    assert_eq!(
        ssa.block(exit).instructions[0].use_of(&rax),
        decrement.def_of(&rax),
        "the exit reads what the last iteration left behind"
    );
    assert_eq!(
        ssa.phis()
            .filter(|(_, phi)| phi.merges_start_value())
            .count(),
        2,
        "no other block has a phi that merges the start"
    );
}

#[derive(Debug, Clone)]
struct Step {
    uses: Vec<u8>,
    defs: Vec<u8>,
}

impl InstrInfo for Step {
    type Variable = u8;

    fn uses(&self) -> &[u8] {
        &self.uses
    }

    fn defs(&self) -> &[u8] {
        &self.defs
    }
}

/// The variable every block reads and may define.
const X: u8 = 0;
/// A variable no block defines, so every read of it is a live-in.
const UNDEFINED: u8 = 1;

/// A block that reads `X`, then either defines it or reads `UNDEFINED`, then
/// reads `X` again: three instructions, whose uses cover a phi, a definition
/// that reaches only the block's end, and a live-in.
fn block_steps(defines: bool) -> [Step; 3] {
    [
        Step {
            uses: vec![X],
            defs: vec![],
        },
        if defines {
            Step {
                uses: vec![],
                defs: vec![X],
            }
        } else {
            Step {
                uses: vec![UNDEFINED],
                defs: vec![],
            }
        },
        Step {
            uses: vec![X],
            defs: vec![],
        },
    ]
}

/// The edges of every digraph on up to four nodes in which only the entry may
/// loop back onto itself, with the node count. The entry may be the target of
/// any edge, which is the shape under test.
fn small_digraphs() -> impl Iterator<Item = (usize, Vec<(usize, usize)>)> {
    (1..=4_usize).flat_map(|count| {
        let pairs: Vec<(usize, usize)> = (0..count)
            .flat_map(|source| (0..count).map(move |target| (source, target)))
            .filter(|&(source, target)| source != target || source == 0)
            .collect();
        (0..1_u32 << pairs.len()).map(move |mask| {
            let edges = pairs
                .iter()
                .enumerate()
                .filter(|&(bit, _)| mask & (1 << bit) != 0)
                .map(|(_, &edge)| edge)
                .collect();
            (count, edges)
        })
    })
}

fn every_block_is_reachable(count: usize, edges: &[(usize, usize)]) -> bool {
    let mut reached = BTreeSet::from([0]);
    let mut pending = vec![0];
    while let Some(block) = pending.pop() {
        for &(source, target) in edges {
            if source == block && reached.insert(target) {
                pending.push(target);
            }
        }
    }
    reached.len() == count
}

/// A definition of `X` that can reach a point: the value at function start,
/// or the definition in one block.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Site {
    Start,
    Definition(usize),
}

/// The definitions of `X` that reach the start of each block, by iterating
/// over every edge until nothing changes. The function starts at block 0.
fn reaching_the_block_start(
    count: usize,
    edges: &[(usize, usize)],
    defines: &[bool],
) -> Vec<BTreeSet<Site>> {
    let mut incoming = vec![BTreeSet::new(); count];
    incoming[0].insert(Site::Start);
    loop {
        let mut changed = false;
        for &(source, target) in edges {
            let leaving = if defines[source] {
                BTreeSet::from([Site::Definition(source)])
            } else {
                incoming[source].clone()
            };
            for site in leaving {
                changed |= incoming[target].insert(site);
            }
        }
        if !changed {
            return incoming;
        }
    }
}

enum Producer<'a> {
    Definition(usize),
    Phi(&'a SsaPhi<u8>),
}

/// The definitions of `X` that a value stands for: itself, or those of every
/// input of the phi that produced it.
fn sites_of(
    value: &SsaValue<u8>,
    producers: &BTreeMap<SsaValue<u8>, Producer<'_>>,
) -> BTreeSet<Site> {
    let mut sites = BTreeSet::new();
    let mut visited = BTreeSet::new();
    let mut pending = vec![value.clone()];
    while let Some(current) = pending.pop() {
        if !visited.insert(current.clone()) {
            continue;
        }
        if current.is_live_in() {
            sites.insert(Site::Start);
            continue;
        }
        match &producers[&current] {
            Producer::Definition(block) => {
                sites.insert(Site::Definition(*block));
            }
            Producer::Phi(phi) => {
                pending.extend(phi.operands().iter().map(|(_, operand)| operand.clone()));
                pending.extend(phi.start_value());
            }
        }
    }
    sites
}

fn assert_phis_are_well_formed(
    ssa: &SsaForm<u8>,
    edges: &[(usize, usize)],
    producers: &BTreeMap<SsaValue<u8>, Producer<'_>>,
    label: &str,
) {
    for (block, phi) in ssa.phis() {
        let predecessors: Vec<usize> = edges
            .iter()
            .filter(|&&(_, target)| target == block.index())
            .map(|&(source, _)| source)
            .collect();
        let departures: Vec<usize> = phi
            .operands()
            .iter()
            .map(|(point, _)| point.block.index())
            .collect();
        let sorted = |mut list: Vec<usize>| {
            list.sort_unstable();
            list
        };
        assert_eq!(
            sorted(departures),
            sorted(predecessors),
            "one operand per edge: {label}"
        );
        assert!(
            phi.operands().iter().all(|(point, _)| point.inst_idx == 3),
            "every edge leaves at the end of its block: {label}"
        );
        let inputs = phi
            .operands()
            .iter()
            .map(|(_, operand)| operand.clone())
            .chain(phi.start_value());
        for input in inputs {
            assert!(
                input.is_live_in() || producers.contains_key(&input),
                "every phi input is defined: {input:?}, {label}"
            );
        }
        assert_eq!(
            phi.start_value().is_some(),
            block.index() == 0,
            "exactly the entry's phis merge the start value: {label}"
        );
        assert_eq!(phi.merges_start_value(), phi.start_value().is_some());
    }
}

/// Every value the form defines, with its producer, each defined once and
/// with dense versions from one.
fn producers_of<'a>(ssa: &'a SsaForm<u8>, label: &str) -> BTreeMap<SsaValue<u8>, Producer<'a>> {
    let mut producers = BTreeMap::new();
    for block in ssa.blocks() {
        for phi in &block.phis {
            let previous = producers.insert(phi.result.clone(), Producer::Phi(phi));
            assert!(previous.is_none(), "defined once: {label}");
        }
        for instruction in &block.instructions {
            for definition in instruction.defs() {
                let producer = Producer::Definition(block.block.index());
                let previous = producers.insert(definition.clone(), producer);
                assert!(previous.is_none(), "defined once: {label}");
            }
        }
    }
    let versions: BTreeSet<usize> = producers.keys().map(|value| value.version).collect();
    assert_eq!(
        versions,
        (1..=ssa.max_version(&X)).collect(),
        "versions are dense and none is zero: {label}"
    );
    assert!(producers.keys().all(|value| value.variable == X));
    producers
}

/// Every read of `X` names the definitions that reach it in the program.
fn assert_reads_match_the_reaching_definitions(
    ssa: &SsaForm<u8>,
    producers: &BTreeMap<SsaValue<u8>, Producer<'_>>,
    edges: &[(usize, usize)],
    defines: &[bool],
    label: &str,
) {
    let at_start = reaching_the_block_start(defines.len(), edges, defines);
    for (block, &defined) in defines.iter().enumerate() {
        let instructions = &ssa.blocks()[block].instructions;
        let last = if defined {
            BTreeSet::from([Site::Definition(block)])
        } else {
            at_start[block].clone()
        };
        assert_eq!(
            sites_of(&instructions[0].uses()[0], producers),
            at_start[block],
            "first read of block {block}: {label}"
        );
        assert_eq!(
            sites_of(&instructions[2].uses()[0], producers),
            last,
            "last read of block {block}: {label}"
        );
        if !defined {
            assert_eq!(instructions[1].uses(), [SsaValue::live_in(UNDEFINED)]);
        }
    }
}

/// Placement alone agrees with the renamed form about every phi.
fn assert_placements_match(ssa: &SsaForm<u8>, placements: &PhiPlacements<u8>, label: &str) {
    assert_eq!(placements.len(), ssa.phis().count(), "{label}");
    for (block, phi) in ssa.phis() {
        let placement = placements
            .at(block)
            .iter()
            .find(|placement| placement.variable == phi.result.variable)
            .expect("every phi has its placement");
        let departures: Vec<_> = phi.operands().iter().map(|(point, _)| *point).collect();
        assert_eq!(placement.predecessors(), departures, "{label}");
        assert_eq!(
            placement.merges_start_value(),
            phi.merges_start_value(),
            "{label}"
        );
    }
}

/// The program of `edges` over `count` blocks whose block `i` defines `X`
/// when `defines[i]`.
fn small_program(count: usize, edges: &[(usize, usize)], defines: &[bool]) -> Cfg<Step> {
    let mut cfg = Cfg::<Step>::new();
    let mut blocks = vec![cfg.entry()];
    blocks.extend((1..count).map(|_| cfg.new_block()));
    for (block, &defined) in blocks.iter().zip(defines) {
        cfg.block_mut(*block)
            .instructions_mut()
            .extend(block_steps(defined));
    }
    for &(source, target) in edges {
        cfg.add_edge(blocks[source], blocks[target], EdgeKind::Fallthrough);
    }
    cfg
}

/// Every digraph on up to four nodes in which every block is reachable, with
/// the entry free to be a branch target: every read names the definitions the
/// program actually delivers, every value is defined exactly once, and the
/// phi inputs are the edges plus the start for the entry alone.
#[test]
fn ssa_over_every_small_digraph_names_exactly_the_definitions_that_reach_each_read() {
    let mut entries_that_are_branch_targets = 0;
    for (count, edges) in
        small_digraphs().filter(|(count, edges)| every_block_is_reachable(*count, edges))
    {
        let masks = 1_usize << count;
        let seed = edges
            .iter()
            .fold(0, |sum, &(source, target)| sum * 7 + source * 4 + target);
        for defines_mask in [seed % masks, (seed / masks) % masks] {
            let defines: Vec<bool> = (0..count)
                .map(|block| defines_mask & (1 << block) != 0)
                .collect();
            let label = format!("{count} blocks, {edges:?}, defines {defines:?}");
            let cfg = small_program(count, &edges, &defines);
            entries_that_are_branch_targets +=
                usize::from(edges.iter().any(|&(_, target)| target == 0));

            let dominators = DominatorTree::compute(&cfg);
            let ssa = SsaForm::compute(&cfg, &dominators);
            let producers = producers_of(&ssa, &label);
            assert_phis_are_well_formed(&ssa, &edges, &producers, &label);
            assert_reads_match_the_reaching_definitions(&ssa, &producers, &edges, &defines, &label);
            assert_placements_match(&ssa, &PhiPlacements::compute(&cfg, &dominators), &label);
        }
    }
    assert!(
        entries_that_are_branch_targets > 1000,
        "the enumeration must reach the shape under test"
    );
}
