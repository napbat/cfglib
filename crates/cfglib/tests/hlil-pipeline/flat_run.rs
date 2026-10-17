//! The reference interpreter of flat bytecode.
//!
//! It follows edges and regions the way the documentation of the flat
//! builder describes them: a branch takes its `Yes` or `No` edge, a switch
//! its matching `Case` edge or the default, and an instruction that throws
//! transfers to a handler of the innermost region that protects its block.
//! The first handler that is not a `finally` is the one that catches, and a
//! region that has only a `finally` hands the exception to it.

use cfglib::{BlockId, Cfg, EdgeKind, HandlerKind};

use crate::dialect::{Lit, Ty};
use crate::flat::{Code, Form, Instr, Link};
use crate::machine::{End, Machine, Outcome, Thrown, Value};

/// Runs `code` on integer `inputs` and reports what it observes.
pub(crate) fn run_flat(code: &Code, inputs: &[i64]) -> Outcome {
    let mut registers: Vec<Value> = code
        .declared
        .iter()
        .map(|ty| match ty {
            Some(Ty::Bool) => Value::Bool(false),
            _ => Value::Int(0),
        })
        .collect();
    for (&parameter, &input) in code.parameters.iter().zip(inputs) {
        registers[parameter.index()] = Value::Int(input);
    }
    let mut machine = Machine::new();
    let mut block = code.first_block();
    let end = loop {
        // An empty block costs a step too, so a block that jumps to itself
        // runs out of fuel.
        if !machine.tick() {
            break End::Diverged;
        }
        match run_block(code, block, &mut registers, &mut machine) {
            Ok(Exit::Next(next)) => block = next,
            Ok(Exit::Return(values)) => break End::Returned(values),
            Ok(Exit::FellOff) => break End::FellOff,
            Ok(Exit::Halted) => break End::Diverged,
            Err(Thrown(thrown)) => match handler_entry(code, block) {
                Some(entry) => {
                    machine.exception = thrown;
                    block = entry;
                }
                None => break End::Uncaught(thrown),
            },
        }
    };
    machine.finish(end)
}

enum Exit {
    Next(BlockId),
    Return(Vec<Value>),
    FellOff,
    Halted,
}

/// Runs the instructions of `block`. An exception leaves through `Err`.
fn run_block(
    code: &Code,
    block: BlockId,
    registers: &mut [Value],
    machine: &mut Machine,
) -> Result<Exit, Thrown> {
    let cfg = &code.cfg;
    let mut next = None;
    for instruction in cfg.block(block).instructions() {
        if !machine.tick() {
            return Ok(Exit::Halted);
        }
        let args: Vec<Value> = instruction
            .uses
            .iter()
            .map(|register| registers[register.index()])
            .collect();
        match &instruction.form {
            Form::Const(literal) => registers[instruction.defs[0].index()] = Value::of(*literal),
            Form::Copy => registers[instruction.defs[0].index()] = args[0],
            Form::Apply(op) => {
                let value = machine.apply(*op, &args)?;
                if let Some(definition) = instruction.defs.first() {
                    registers[definition.index()] = value;
                }
            }
            Form::Store(_) => machine.store(args[0].int(), args[args.len() - 1].int()),
            Form::Move => {
                for (definition, value) in instruction.defs.iter().zip(&args) {
                    registers[definition.index()] = *value;
                }
            }
            Form::Caught => registers[instruction.defs[0].index()] = Value::Int(machine.exception),
            Form::Enter(_) => machine.acquire(args[0].int()),
            Form::Exit(_) => machine.release(args[0].int()),
            Form::Nop => {}
            Form::Return => return Ok(Exit::Return(args)),
            Form::Branch => next = Some(branch(cfg, block, args[0].truth())),
            Form::BranchOn(op) => {
                let condition = machine.apply(*op, &args)?.truth();
                next = Some(branch(cfg, block, condition));
            }
            Form::Switch => next = Some(dispatch(cfg, block, args[0].int())),
        }
    }
    Ok(next
        .or_else(|| fall_through(cfg, block))
        .map_or(Exit::FellOff, Exit::Next))
}

fn target_of(
    cfg: &Cfg<Instr, Link>,
    block: BlockId,
    wanted: impl Fn(&Link) -> bool,
) -> Option<BlockId> {
    cfg.outgoing(block)
        .map(|id| cfg.edge(id))
        .find(|edge| wanted(edge.payload()))
        .map(|edge| edge.target())
}

fn branch(cfg: &Cfg<Instr, Link>, block: BlockId, condition: bool) -> BlockId {
    let wanted = if condition { Link::Yes } else { Link::No };
    target_of(cfg, block, |link| *link == wanted).expect("a branch has both edges")
}

fn dispatch(cfg: &Cfg<Instr, Link>, block: BlockId, scrutinee: i64) -> BlockId {
    let value = Lit::Int(scrutinee);
    target_of(
        cfg,
        block,
        |link| matches!(link, Link::Case(values) if values.contains(&value)),
    )
    .or_else(|| target_of(cfg, block, |link| *link == Link::Otherwise))
    .expect("a dispatch has a default edge")
}

fn fall_through(cfg: &Cfg<Instr, Link>, block: BlockId) -> Option<BlockId> {
    target_of(cfg, block, |link| *link != Link::Unwind)
}

/// The entry of the handler that catches an exception thrown in `block`,
/// checked against the unwind edges that lowering must have placed.
fn handler_entry(code: &Code, block: BlockId) -> Option<BlockId> {
    let region = code
        .cfg
        .regions()
        .iter()
        .filter(|region| region.protected_blocks.contains(&block))
        .min_by_key(|region| region.protected_blocks.len())?;
    let handler = region
        .handlers
        .iter()
        .find(|handler| !matches!(handler.kind, HandlerKind::Finally))
        .or_else(|| region.handlers.first())?;
    let unwinds = code.cfg.outgoing(block).any(|id| {
        let edge = code.cfg.edge(id);
        edge.kind() == EdgeKind::ExceptionUnwind && edge.target() == handler.entry
    });
    assert!(
        unwinds,
        "{block} throws into a region but has no unwind edge to {}",
        handler.entry
    );
    Some(handler.entry)
}
