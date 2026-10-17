//! The toy-ISA seam: a position-indexed program is laid out by an ISA into
//! addressed instructions, so one scenario can run over every address space.
//!
//! Scenarios spell control flow in program positions (`Dest::At(3)` is "the
//! fourth instruction"). Each toy ISA encodes the same abstract operation
//! into its own opcode vocabulary and sizes, and the assembler resolves
//! positions into that ISA's addresses. Expectations are stated in positions
//! too, so they hold for every address space and every instruction width.

use core::fmt::{Debug, Display};

use cfglib::{AddressInstruction, AddressSpace, DisplayInstr};

/// A branch destination, spelled in program positions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dest {
    /// The first byte of the instruction at this position.
    At(usize),
    /// One byte into the instruction at this position, which must be wider
    /// than one byte: an address that is not an instruction start.
    Inside(usize),
    /// This many bytes past the end of the code: outside the recovered
    /// stream.
    Beyond(u64),
}

/// One abstract operation of a toy program.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Op {
    /// Straight-line work that can fault.
    Work,
    /// Straight-line work that provably cannot fault.
    Pure,
    /// Unconditional direct jump.
    Jump(Dest),
    /// Conditional direct branch that falls through when not taken.
    BranchIf(Dest),
    /// Direct call that returns to the next instruction.
    Call(Dest),
    /// Call through a register or vtable slot.
    CallIndirect,
    /// Computed jump with no named target.
    JumpIndirect,
    /// Table dispatch: keyed cases in dispatch order and a default.
    Dispatch {
        /// Destination when no case matches.
        default: Dest,
        /// Keyed destinations in dispatch order.
        cases: Vec<(i16, Dest)>,
    },
    /// Leaves the function.
    Return,
    /// Raises an exception out of the function.
    Raise,
    /// Continues at the next instruction but may take an alternate path to
    /// the destination, as a transactional-abort fallback does.
    Probe(Dest),
}

/// A toy instruction set: how it encodes an [`Op`] and which address space it
/// lives in.
pub trait Isa {
    /// The address type of the ISA's address space.
    type Addr: AddressSpace + Debug + Display + Into<u64>;
    /// The ISA's switch case key.
    type Key: Copy + Into<i64> + Debug;
    /// The ISA's instruction.
    type Insn: AddressInstruction<Address = Self::Addr, CaseKey = Self::Key>
        + Clone
        + Debug
        + PartialEq
        + DisplayInstr;

    /// The address `offset` bytes above the start of the code.
    fn at(offset: u64) -> Self::Addr;

    /// Encodes `op` at `address`, spelling every destination through
    /// `resolve`. The encoded width must not depend on the destinations.
    fn encode(address: Self::Addr, op: &Op, resolve: &dyn Fn(Dest) -> Self::Addr) -> Self::Insn;
}

/// A program laid out in one ISA's address space.
pub struct Layout<I: Isa> {
    instructions: Vec<I::Insn>,
    /// The byte offset of every instruction, then of the end of the code.
    offsets: Vec<u64>,
}

impl<I: Isa> Layout<I> {
    /// Lays `ops` out back to back.
    ///
    /// # Panics
    ///
    /// Panics when a destination names a position outside the program or an
    /// `Inside` destination names a one-byte instruction.
    pub fn assemble(ops: &[Op]) -> Self {
        // Encoded widths do not depend on destinations, so a first pass with
        // placeholder destinations measures every instruction.
        let placeholder = |_: Dest| I::at(0);
        let mut offsets = vec![0_u64];
        for op in ops {
            let offset = *offsets.last().expect("offsets start with 0");
            let address = I::at(offset);
            let encoded = I::encode(address, op, &placeholder);
            let end = encoded.end_address().expect("toy code fits its space");
            let width = end.distance_from(address);
            offsets.push(offset + width);
        }
        let code_end = *offsets.last().expect("offsets start with 0");
        let resolve = |dest: Dest| match dest {
            Dest::At(position) => I::at(offsets[position]),
            Dest::Inside(position) => {
                assert!(
                    offsets[position + 1] - offsets[position] >= 2,
                    "instruction {position} is one byte wide"
                );
                I::at(offsets[position] + 1)
            }
            Dest::Beyond(bytes) => I::at(code_end + bytes),
        };
        let instructions = ops
            .iter()
            .enumerate()
            .map(|(position, op)| I::encode(I::at(offsets[position]), op, &resolve))
            .collect();
        Self {
            instructions,
            offsets,
        }
    }

    /// The instructions in address order.
    pub fn instructions(&self) -> &[I::Insn] {
        &self.instructions
    }

    /// The number of instructions.
    pub fn len(&self) -> usize {
        self.instructions.len()
    }

    /// The address of the instruction at `position`.
    pub fn address(&self, position: usize) -> I::Addr {
        I::at(self.offsets[position])
    }

    /// The address one byte into the instruction at `position`.
    pub fn inside(&self, position: usize) -> I::Addr {
        I::at(self.offsets[position] + 1)
    }

    /// The address `bytes` past the end of the code.
    pub fn beyond(&self, bytes: u64) -> I::Addr {
        I::at(self.offsets[self.instructions.len()] + bytes)
    }

    /// The position of the instruction that starts at `address`.
    pub fn position_of(&self, address: I::Addr) -> Option<usize> {
        let offset: u64 = address.into();
        let origin: u64 = I::at(0).into();
        let relative = offset.checked_sub(origin)?;
        self.offsets[..self.instructions.len()]
            .binary_search(&relative)
            .ok()
    }

    /// The number of bytes the instructions in `positions` occupy.
    pub fn byte_span(&self, positions: core::ops::Range<usize>) -> u64 {
        self.offsets[positions.end] - self.offsets[positions.start]
    }
}
