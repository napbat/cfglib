//! A variable-length machine ISA in a 64-bit address space, shaped like a
//! disassembler's output: every decoded instruction carries its own byte
//! length and an explicit "may fault" annotation, and the code is loaded
//! above the 32-bit range so no address fits a `u32`.

use std::borrow::Cow;

use cfglib::{AddressInstruction, DisplayInstr, Flow};

use crate::isa::{Dest, Isa, Op};

/// Where the toy image is mapped: above `u32::MAX`.
const IMAGE_BASE: u64 = 0x0000_7ff6_1000_0000;

/// A decoded mnemonic with its direct operands.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Mnemonic {
    /// Register move.
    Mov,
    /// No operation.
    Nop,
    /// Relative jump.
    Jmp(u64),
    /// Conditional relative jump.
    Jcc(u64),
    /// Relative call.
    CallRel(u64),
    /// Call through a register.
    CallReg,
    /// Jump through a register.
    JmpReg,
    /// Jump through a table: default and keyed targets.
    JmpTable {
        /// Target when no key matches.
        default: u64,
        /// Key and target of every case.
        cases: Vec<(i8, u64)>,
    },
    /// Return to the caller.
    Ret,
    /// Undefined instruction: always raises.
    Ud2,
    /// Starts a transaction whose abort falls back to the operand.
    XBegin(u64),
}

impl Mnemonic {
    fn name(&self) -> &'static str {
        match self {
            Self::Mov => "mov",
            Self::Nop => "nop",
            Self::Jmp(_) => "jmp",
            Self::Jcc(_) => "jcc",
            Self::CallRel(_) => "call",
            Self::CallReg => "call_reg",
            Self::JmpReg => "jmp_reg",
            Self::JmpTable { .. } => "jmp_table",
            Self::Ret => "ret",
            Self::Ud2 => "ud2",
            Self::XBegin(_) => "xbegin",
        }
    }
}

/// One decoded machine instruction.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Machine {
    /// The virtual address of the first byte.
    pub address: u64,
    /// The encoded length in bytes.
    pub length: u8,
    /// The decoded mnemonic.
    pub mnemonic: Mnemonic,
    /// Whether the decoder marked the instruction as able to fault.
    pub faults: bool,
}

impl AddressInstruction for Machine {
    type Address = u64;
    type CaseKey = i8;

    fn address(&self) -> u64 {
        self.address
    }

    fn end_address(&self) -> Option<u64> {
        self.address.checked_add(u64::from(self.length))
    }

    fn flow(&self) -> Flow<u64, i8> {
        match &self.mnemonic {
            Mnemonic::Mov | Mnemonic::Nop => Flow::Next,
            Mnemonic::Jmp(target) => Flow::Jump { target: *target },
            Mnemonic::Jcc(target) => Flow::Conditional { target: *target },
            Mnemonic::CallRel(target) => Flow::Call { target: *target },
            Mnemonic::CallReg => Flow::IndirectCall,
            Mnemonic::JmpReg => Flow::Indirect,
            Mnemonic::JmpTable { default, cases } => Flow::Switch {
                default: *default,
                cases: cases.clone(),
            },
            Mnemonic::Ret => Flow::Return,
            Mnemonic::Ud2 => Flow::Throw,
            Mnemonic::XBegin(target) => Flow::Exceptional { target: *target },
        }
    }

    fn retains_exception_edge(&self) -> bool {
        self.faults
    }
}

impl DisplayInstr for Machine {
    fn mnemonic(&self) -> Cow<'_, str> {
        Cow::Borrowed(self.mnemonic.name())
    }
}

/// The machine ISA: 64-bit addresses, 8-bit case keys.
pub struct Isa64;

impl Isa for Isa64 {
    type Addr = u64;
    type Key = i8;
    type Insn = Machine;

    fn at(offset: u64) -> u64 {
        IMAGE_BASE + offset
    }

    fn encode(address: u64, op: &Op, resolve: &dyn Fn(Dest) -> u64) -> Machine {
        let (mnemonic, length, faults) = match op {
            Op::Work => (Mnemonic::Mov, 3, true),
            Op::Pure => (Mnemonic::Nop, 1, false),
            Op::Jump(dest) => (Mnemonic::Jmp(resolve(*dest)), 5, false),
            Op::BranchIf(dest) => (Mnemonic::Jcc(resolve(*dest)), 6, false),
            Op::Call(dest) => (Mnemonic::CallRel(resolve(*dest)), 5, true),
            Op::CallIndirect => (Mnemonic::CallReg, 2, true),
            Op::JumpIndirect => (Mnemonic::JmpReg, 2, false),
            Op::Dispatch { default, cases } => (
                Mnemonic::JmpTable {
                    default: resolve(*default),
                    cases: cases
                        .iter()
                        .map(|&(key, dest)| {
                            (
                                i8::try_from(key).expect("toy keys are small"),
                                resolve(dest),
                            )
                        })
                        .collect(),
                },
                7,
                false,
            ),
            Op::Return => (Mnemonic::Ret, 1, false),
            Op::Raise => (Mnemonic::Ud2, 2, true),
            Op::Probe(dest) => (Mnemonic::XBegin(resolve(*dest)), 6, false),
        };
        Machine {
            address,
            length,
            mnemonic,
            faults,
        }
    }
}
