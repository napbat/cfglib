//! A stack-machine bytecode in a 32-bit address space: a one-byte opcode
//! followed by fixed-width operands, so instruction widths vary from one
//! byte to a whole dispatch table. Whether an opcode may throw is a property
//! of the opcode.

use std::borrow::Cow;

use cfglib::{AddressInstruction, DisplayInstr, Flow};

use crate::isa::{Dest, Isa, Op};

/// Where the toy code is loaded.
const ORIGIN: u32 = 0x0040_0000;

/// One bytecode opcode with its operands.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Opcode {
    /// Pushes a constant: five bytes.
    Push(i32),
    /// Loads a local slot, which traps on a bad slot: three bytes.
    Load(u16),
    /// Unconditional branch: five bytes.
    Br(u32),
    /// Pops and branches when true: five bytes.
    BrTrue(u32),
    /// Direct call: five bytes.
    Call(u32),
    /// Virtual call through the stack top: one byte.
    CallVirtual,
    /// Branch to the address on the stack top: one byte.
    BrComputed,
    /// Dispatch table: opcode, default, and a key and target per case.
    Table {
        /// Target when no key matches.
        default: u32,
        /// Key and target of every case.
        cases: Vec<(i16, u32)>,
    },
    /// Returns: one byte.
    Ret,
    /// Throws the exception on the stack top: one byte.
    Throw,
    /// Starts a transaction whose abort resumes at the operand: five bytes.
    TxBegin(u32),
}

impl Opcode {
    fn width(&self) -> u32 {
        match self {
            Self::CallVirtual | Self::BrComputed | Self::Ret | Self::Throw => 1,
            Self::Load(_) => 3,
            Self::Push(_) | Self::Br(_) | Self::BrTrue(_) | Self::Call(_) | Self::TxBegin(_) => 5,
            Self::Table { cases, .. } => {
                5 + 6 * u32::try_from(cases.len()).expect("a toy table is small")
            }
        }
    }

    fn name(&self) -> &'static str {
        match self {
            Self::Push(_) => "push",
            Self::Load(_) => "load",
            Self::Br(_) => "br",
            Self::BrTrue(_) => "brtrue",
            Self::Call(_) => "call",
            Self::CallVirtual => "callvirt",
            Self::BrComputed => "brcomputed",
            Self::Table { .. } => "table",
            Self::Ret => "ret",
            Self::Throw => "throw",
            Self::TxBegin(_) => "txbegin",
        }
    }
}

/// One bytecode instruction at its byte offset.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Bytecode {
    /// The offset of the opcode byte.
    pub offset: u32,
    /// The decoded opcode.
    pub opcode: Opcode,
}

impl AddressInstruction for Bytecode {
    type Address = u32;
    type CaseKey = i16;

    fn address(&self) -> u32 {
        self.offset
    }

    fn end_address(&self) -> Option<u32> {
        self.offset.checked_add(self.opcode.width())
    }

    fn flow(&self) -> Flow<u32, i16> {
        match &self.opcode {
            Opcode::Push(_) | Opcode::Load(_) => Flow::Next,
            Opcode::Br(target) => Flow::Jump { target: *target },
            Opcode::BrTrue(target) => Flow::Conditional { target: *target },
            Opcode::Call(target) => Flow::Call { target: *target },
            Opcode::CallVirtual => Flow::IndirectCall,
            Opcode::BrComputed => Flow::Indirect,
            Opcode::Table { default, cases } => Flow::Switch {
                default: *default,
                cases: cases.clone(),
            },
            Opcode::Ret => Flow::Return,
            Opcode::Throw => Flow::Throw,
            Opcode::TxBegin(target) => Flow::Exceptional { target: *target },
        }
    }

    fn retains_exception_edge(&self) -> bool {
        !matches!(
            self.opcode,
            Opcode::Push(_) | Opcode::Br(_) | Opcode::BrTrue(_) | Opcode::Ret
        )
    }
}

impl DisplayInstr for Bytecode {
    fn mnemonic(&self) -> Cow<'_, str> {
        Cow::Borrowed(self.opcode.name())
    }
}

/// The bytecode ISA: 32-bit addresses, 16-bit case keys.
pub struct Isa32;

impl Isa for Isa32 {
    type Addr = u32;
    type Key = i16;
    type Insn = Bytecode;

    fn at(offset: u64) -> u32 {
        ORIGIN + u32::try_from(offset).expect("toy code fits the address space")
    }

    fn encode(address: u32, op: &Op, resolve: &dyn Fn(Dest) -> u32) -> Bytecode {
        let opcode = match op {
            Op::Work => Opcode::Load(0),
            Op::Pure => Opcode::Push(0),
            Op::Jump(dest) => Opcode::Br(resolve(*dest)),
            Op::BranchIf(dest) => Opcode::BrTrue(resolve(*dest)),
            Op::Call(dest) => Opcode::Call(resolve(*dest)),
            Op::CallIndirect => Opcode::CallVirtual,
            Op::JumpIndirect => Opcode::BrComputed,
            Op::Dispatch { default, cases } => Opcode::Table {
                default: resolve(*default),
                cases: cases
                    .iter()
                    .map(|&(key, dest)| (key, resolve(dest)))
                    .collect(),
            },
            Op::Return => Opcode::Ret,
            Op::Raise => Opcode::Throw,
            Op::Probe(dest) => Opcode::TxBegin(resolve(*dest)),
        };
        Bytecode {
            offset: address,
            opcode,
        }
    }
}
