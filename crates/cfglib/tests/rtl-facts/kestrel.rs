//! Kestrel: a flag-setting register ISA with partial-width registers.
//!
//! Four 32-bit registers (their low and high halves are bit slices of one
//! root), four one-bit status flags, one 130-bit accumulator that spans three
//! words, and a bank of model-specific registers.

use cfglib::ir::rtl::{Bits, ConstBits, Dialect, Fold};

use crate::bits;
use crate::model::Target;

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct Kestrel;

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub enum Reg {
    R0,
    R1,
    R2,
    R3,
    Zero,
    Neg,
    Carry,
    Over,
    Wide,
    Csr,
}

impl Reg {
    pub const ALL: [Self; 10] = [
        Self::R0,
        Self::R1,
        Self::R2,
        Self::R3,
        Self::Zero,
        Self::Neg,
        Self::Carry,
        Self::Over,
        Self::Wide,
        Self::Csr,
    ];
}

/// The pure operators. Operand widths come from the operands themselves.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub enum KOp {
    /// Wrapping sum of two values of one width.
    Add,
    /// The wrapping sum, then the carry out of the top bit.
    AddCarry,
    /// Wrapping difference.
    Sub,
    And,
    Or,
    Xor,
    Not,
    /// Shift left by the second operand; a shift of at least the width gives zero.
    Shl,
    /// Logical shift right by the second operand.
    Lsr,
    Eq,
    /// Unsigned less-than.
    Ult,
    Zext,
    Trunc,
    /// `args[0]` is the high part, `args[1]` the low part.
    Concat,
    /// `args[0]` is a one-bit condition: its value picks `args[1]` or `args[2]`.
    Pick,
    /// An operator that has no constant evaluation.
    Opaque,
}

#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub enum KEffect {
    Load,
    Store,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Policy {
    Zeroing,
    Merging,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Fault {
    PageFault,
    DivideError,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Undef {
    AllFlags,
    Scratch,
}

impl Dialect for Kestrel {
    type Root = Reg;
    type Operator = KOp;
    type EffectOp = KEffect;
    type WritePolicy = Policy;
    type Outcome = Fault;
    type NondetDomain = Undef;

    fn root_width(root: Reg) -> Bits {
        match root {
            Reg::R0 | Reg::R1 | Reg::R2 | Reg::R3 | Reg::Csr => bits(32),
            Reg::Zero | Reg::Neg | Reg::Carry | Reg::Over => Bits::BOOL,
            Reg::Wide => bits(130),
        }
    }
}

impl Target for Kestrel {
    fn root_of(id: u8) -> Reg {
        Reg::ALL[usize::from(id)]
    }

    fn id_of(root: Reg) -> u8 {
        u8::try_from(
            Reg::ALL
                .iter()
                .position(|&candidate| candidate == root)
                .unwrap(),
        )
        .expect("ten roots fit in a byte")
    }

    fn operator() -> KOp {
        KOp::Xor
    }

    fn effect() -> KEffect {
        KEffect::Load
    }

    fn policy() -> Policy {
        Policy::Merging
    }

    fn outcome() -> Fault {
        Fault::PageFault
    }

    fn domain() -> Undef {
        Undef::AllFlags
    }

    fn add() -> KOp {
        KOp::Add
    }

    fn sub() -> KOp {
        KOp::Sub
    }

    fn and() -> KOp {
        KOp::And
    }

    fn xor() -> KOp {
        KOp::Xor
    }

    fn equal() -> KOp {
        KOp::Eq
    }

    fn opaque() -> KOp {
        KOp::Opaque
    }
}

/// The low 64 bits of `value`.
fn low_word(value: u128) -> u64 {
    u64::try_from(value & u128::from(u64::MAX)).expect("masked to 64 bits")
}

/// A mask of the low `width` bits.
fn mask(width: Bits) -> u128 {
    if width.width() == 128 {
        u128::MAX
    } else {
        (1_u128 << width.width()) - 1
    }
}

/// The value of an operand of at most 128 bits.
fn small(arg: &ConstBits<'_>) -> Option<u128> {
    if arg.width.width() > 128 {
        return None;
    }
    let low = u128::from(arg.words[0]);
    let high = arg.words.get(1).copied().map_or(0, u128::from);
    Some(low | (high << 64))
}

/// Where result `index` starts in the output buffer.
fn offset(results: &[Bits], index: usize) -> usize {
    results[..index].iter().map(|width| width.words()).sum()
}

fn put_small(out: &mut [u64], results: &[Bits], index: usize, value: u128) {
    let start = offset(results, index);
    let value = value & mask(results[index]);
    out[start] = low_word(value);
    if results[index].words() > 1 {
        out[start + 1] = low_word(value >> 64);
    }
}

/// Clears the bits of the last word that lie past `width`.
fn trim(words: &mut [u64], width: Bits) {
    let used = usize::from(width.width() % 64);
    if used != 0
        && let Some(last) = words.last_mut()
    {
        *last &= (1_u64 << used) - 1;
    }
}

/// A shift amount, or `None` when it does not fit in 64 bits.
fn amount(arg: &ConstBits<'_>) -> Option<u64> {
    let (low, rest) = arg.words.split_first()?;
    rest.iter().all(|&word| word == 0).then_some(*low)
}

impl Fold for Kestrel {
    fn fold(operator: &KOp, args: &[ConstBits<'_>], results: &[Bits], out: &mut [u64]) -> bool {
        match operator {
            KOp::And | KOp::Or | KOp::Xor => {
                let words = results[0].words();
                for (position, slot) in out[..words].iter_mut().enumerate() {
                    let (left, right) = (args[0].words[position], args[1].words[position]);
                    *slot = match operator {
                        KOp::And => left & right,
                        KOp::Or => left | right,
                        _ => left ^ right,
                    };
                }
                true
            }
            KOp::Not => {
                let words = results[0].words();
                for (slot, &word) in out.iter_mut().zip(args[0].words) {
                    *slot = !word;
                }
                trim(&mut out[..words], results[0]);
                true
            }
            KOp::Eq => {
                out[0] = u64::from(args[0].words == args[1].words);
                true
            }
            KOp::Ult => {
                let less = args[0].words.iter().rev().lt(args[1].words.iter().rev());
                out[0] = u64::from(less);
                true
            }
            KOp::Zext | KOp::Trunc => {
                let words = results[0].words();
                for (position, slot) in out[..words].iter_mut().enumerate() {
                    *slot = args[0].words.get(position).copied().unwrap_or(0);
                }
                trim(&mut out[..words], results[0]);
                true
            }
            KOp::Pick => {
                let chosen = if args[0].words[0] == 1 {
                    args[1]
                } else {
                    args[2]
                };
                out[..results[0].words()].copy_from_slice(chosen.words);
                true
            }
            KOp::Opaque => false,
            KOp::Add | KOp::AddCarry | KOp::Sub | KOp::Shl | KOp::Lsr | KOp::Concat => {
                fold_small(operator, args, results, out)
            }
        }
    }
}

/// The operators that read operands of at most 128 bits as one number.
fn fold_small(operator: &KOp, args: &[ConstBits<'_>], results: &[Bits], out: &mut [u64]) -> bool {
    let Some(left) = small(&args[0]) else {
        return false;
    };
    let Some(right) = small(&args[1]) else {
        return false;
    };
    let width = results[0];
    match operator {
        KOp::Add => put_small(out, results, 0, left.wrapping_add(right)),
        KOp::Sub => put_small(out, results, 0, left.wrapping_sub(right)),
        KOp::AddCarry => {
            let (sum, wrapped) = left.overflowing_add(right);
            let carry = if width.width() == 128 {
                wrapped
            } else {
                (sum >> width.width()) & 1 == 1
            };
            put_small(out, results, 0, sum);
            put_small(out, results, 1, u128::from(carry));
        }
        KOp::Shl | KOp::Lsr => {
            let Some(by) = amount(&args[1]) else {
                put_small(out, results, 0, 0);
                return true;
            };
            let shifted = match u32::try_from(by) {
                Ok(by) if by < width.width().into() && matches!(operator, KOp::Shl) => left << by,
                Ok(by) if by < width.width().into() => left >> by,
                _ => 0,
            };
            put_small(out, results, 0, shifted);
        }
        KOp::Concat => {
            let low_width = u32::from(args[1].width.width());
            put_small(out, results, 0, (left << low_width) | right);
        }
        _ => unreachable!("only the numeric operators reach this function"),
    }
    true
}
