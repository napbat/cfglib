//! The instruction types a consumer prints into the text form and parses back.

use std::borrow::Cow;

use cfglib::DisplayInstr;

/// A source statement kept as its own text, the way a source frontend stores
/// one.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Line(pub String);

impl Line {
    pub fn new(text: &str) -> Self {
        Self(text.to_owned())
    }
}

impl DisplayInstr for Line {
    fn mnemonic(&self) -> Cow<'_, str> {
        Cow::Borrowed(&self.0)
    }
}

/// A register machine instruction with a syntax of its own: the text form
/// hands the consumer one unescaped line and the consumer owns the grammar.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Isa {
    Load {
        destination: u8,
        offset: i32,
    },
    Add {
        destination: u8,
        left: u8,
        right: u8,
    },
    Compare {
        left: u8,
        right: u8,
    },
    Return,
}

impl DisplayInstr for Isa {
    fn mnemonic(&self) -> Cow<'_, str> {
        Cow::Owned(match *self {
            Self::Load {
                destination,
                offset,
            } => format!("load r{destination}, [{offset}]"),
            Self::Add {
                destination,
                left,
                right,
            } => format!("add r{destination}, r{left}, r{right}"),
            Self::Compare { left, right } => format!("cmp r{left}, r{right}"),
            Self::Return => "ret".to_owned(),
        })
    }
}

/// Read one instruction line back, rejecting anything that is not the
/// machine's syntax.
pub fn parse_isa(line: &str) -> Result<Isa, String> {
    let (opcode, operands) = line.split_once(' ').unwrap_or((line, ""));
    let operands: Vec<&str> = if operands.is_empty() {
        Vec::new()
    } else {
        operands.split(", ").collect()
    };
    match (opcode, operands.as_slice()) {
        ("ret", []) => Ok(Isa::Return),
        ("cmp", [left, right]) => Ok(Isa::Compare {
            left: register(left)?,
            right: register(right)?,
        }),
        ("add", [destination, left, right]) => Ok(Isa::Add {
            destination: register(destination)?,
            left: register(left)?,
            right: register(right)?,
        }),
        ("load", [destination, address]) => {
            let offset = address
                .strip_prefix('[')
                .and_then(|rest| rest.strip_suffix(']'))
                .and_then(|digits| digits.parse().ok())
                .ok_or_else(|| format!("bad address `{address}`"))?;
            Ok(Isa::Load {
                destination: register(destination)?,
                offset,
            })
        }
        _ => Err(format!("unknown instruction `{line}`")),
    }
}

fn register(token: &str) -> Result<u8, String> {
    token
        .strip_prefix('r')
        .and_then(|digits| digits.parse().ok())
        .ok_or_else(|| format!("bad register `{token}`"))
}
