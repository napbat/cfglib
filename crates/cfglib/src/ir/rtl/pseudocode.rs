//! Indented pseudocode rendering for RTL functions.
//!
//! RTL is the level where storage is still raw, so the printer says exactly
//! that: a transfer names the lanes it writes and the value it writes into
//! them, a parallel transfer is bracketed so its simultaneity is visible,
//! and a block ends with the edges leaving it and the kind of each.
//!
//! ```text
//! fn "shade"(0[0], 1[0]) -> F32
//! bb0 root:
//!     -> bb1 fallthrough [Entry]
//! bb1 body:
//!     0[0]:F32 <- add(0[0]:F32, 0x3f800000:F32)
//!     branch less(0[0]:F32, 0x0:F32)
//!     -> bb2 true [True]
//!     -> bb3 false [False]
//! ```
//!
//! Every dialect-owned name — storage, operators, constraints, edge
//! metadata — is rendered by the dialect's own mnemonic or [`Debug`], so the
//! printer stays language-neutral.

extern crate alloc;
use alloc::string::String;
use core::fmt::{self, Write as _};

use crate::display::IndentedWriter;
use crate::ir::dialect::Vocabulary;

use super::dialect::Dialect;
use super::expr::{Expr, Place};
use super::function::Function;
use super::statement::{Statement, StatementNode};
use super::types::Shape;

impl<D: Dialect> Function<D> {
    /// Render this function as indented pseudocode.
    ///
    /// # Panics
    ///
    /// Panics only if writing to an in-memory [`String`] unexpectedly fails.
    #[must_use]
    pub fn to_pseudocode(&self) -> String {
        let mut out = String::new();
        self.write_pseudocode(&mut out)
            .expect("writing pseudocode to a String cannot fail");
        out
    }

    /// Write this function as indented pseudocode into any sink.
    ///
    /// # Errors
    ///
    /// Returns the sink's formatting error if a write fails.
    pub fn write_pseudocode(&self, sink: &mut dyn fmt::Write) -> fmt::Result {
        let mut printer = IndentedWriter::new(sink);
        self.write_signature(&mut printer)?;
        for id in self.cfg().block_ids() {
            let block = self.cfg().block(id);
            write!(printer, "{id}")?;
            if let Some(label) = block.label() {
                write!(printer, " {label}")?;
            }
            printer.write_str(":")?;
            printer.end_line()?;
            printer.nested(|printer| {
                for node in block.instructions() {
                    write_statement::<D>(printer, node)?;
                }
                for edge in self.cfg().outgoing(id) {
                    let edge = self.cfg().edge(edge);
                    printer.indent()?;
                    write!(
                        printer,
                        "-> {} {} [{:?}]",
                        edge.target(),
                        D::edge_kind(edge.payload()).name(),
                        edge.payload()
                    )?;
                    printer.end_line()?;
                }
                Ok(())
            })?;
        }
        Ok(())
    }

    fn write_signature(&self, printer: &mut IndentedWriter<'_>) -> fmt::Result {
        write!(printer, "fn {:?}(", self.source())?;
        for (position, place) in self.signature().parameters.iter().enumerate() {
            if position != 0 {
                printer.write_str(", ")?;
            }
            write_place::<D>(printer, place)?;
        }
        printer.write_str(")")?;
        for (position, value_type) in self.signature().returns.iter().enumerate() {
            printer.write_str(if position == 0 { " -> " } else { ", " })?;
            write!(printer, "{value_type:?}")?;
        }
        printer.end_line()
    }
}

fn write_statement<D: Dialect>(
    printer: &mut IndentedWriter<'_>,
    node: &StatementNode<D>,
) -> fmt::Result {
    match node.statement() {
        Statement::Transfer {
            assignments,
            effects,
            may_throw,
        } => {
            if let [(place, value)] = assignments.as_slice() {
                printer.indent()?;
                write_place::<D>(printer, place)?;
                printer.write_str(" <- ")?;
                write_expr::<D>(printer, value)?;
                write_effects::<D>(printer, effects, *may_throw)?;
                return printer.end_line();
            }
            printer.indent()?;
            printer.write_str("par")?;
            write_effects::<D>(printer, effects, *may_throw)?;
            printer.open()?;
            for (place, value) in assignments {
                printer.indent()?;
                write_place::<D>(printer, place)?;
                printer.write_str(" <- ")?;
                write_expr::<D>(printer, value)?;
                printer.end_line()?;
            }
            printer.close()
        }
        Statement::Effect {
            operation,
            operands,
            effects,
            may_throw,
        } => {
            printer.indent()?;
            printer.write_str(D::effect_mnemonic(operation))?;
            write_operands::<D>(printer, operands)?;
            write_effects::<D>(printer, effects, *may_throw)?;
            printer.end_line()
        }
        Statement::Raise {
            operation,
            operands,
            effects,
        } => {
            printer.indent()?;
            printer.write_str("raise ")?;
            printer.write_str(D::effect_mnemonic(operation))?;
            write_operands::<D>(printer, operands)?;
            write_effects::<D>(printer, effects, false)?;
            printer.end_line()
        }
        Statement::Branch { condition } => write_keyed::<D>(printer, "branch ", condition),
        Statement::Dispatch { scrutinee } => write_keyed::<D>(printer, "dispatch ", scrutinee),
        Statement::Return { values } => {
            printer.indent()?;
            printer.write_str("return")?;
            for (position, value) in values.iter().enumerate() {
                printer.write_str(if position == 0 { " " } else { ", " })?;
                write_expr::<D>(printer, value)?;
            }
            printer.end_line()
        }
    }
}

fn write_keyed<D: Dialect>(
    printer: &mut IndentedWriter<'_>,
    keyword: &str,
    value: &Expr<D>,
) -> fmt::Result {
    printer.indent()?;
    printer.write_str(keyword)?;
    write_expr::<D>(printer, value)?;
    printer.end_line()
}

fn write_operands<D: Dialect>(
    printer: &mut IndentedWriter<'_>,
    operands: &[Expr<D>],
) -> fmt::Result {
    printer.write_str("(")?;
    for (position, operand) in operands.iter().enumerate() {
        if position != 0 {
            printer.write_str(", ")?;
        }
        write_expr::<D>(printer, operand)?;
    }
    printer.write_str(")")
}

/// Observable effects and exceptional reach, as a suffix on the statement.
fn write_effects<D: Dialect>(
    printer: &mut IndentedWriter<'_>,
    effects: &[<D as Vocabulary>::Effect],
    may_throw: bool,
) -> fmt::Result {
    for (position, effect) in effects.iter().enumerate() {
        printer.write_str(if position == 0 { " ! " } else { ", " })?;
        write!(printer, "{effect:?}")?;
    }
    if may_throw {
        printer.write_str(" throws")?;
    }
    Ok(())
}

fn write_place<D: Dialect>(printer: &mut IndentedWriter<'_>, place: &Place<D>) -> fmt::Result {
    write!(printer, "{:?}", place.storage)?;
    write_lanes(printer, &place.lanes)
}

fn write_lanes(printer: &mut IndentedWriter<'_>, lanes: &[u8]) -> fmt::Result {
    printer.write_str("[")?;
    for (position, lane) in lanes.iter().enumerate() {
        if position != 0 {
            printer.write_str(",")?;
        }
        write!(printer, "{lane}")?;
    }
    printer.write_str("]")
}

fn write_expr<D: Dialect>(printer: &mut IndentedWriter<'_>, expr: &Expr<D>) -> fmt::Result {
    match expr {
        Expr::Read {
            storage,
            lanes,
            scalar,
        } => {
            write!(printer, "{storage:?}")?;
            write_lanes(printer, lanes)?;
            printer.write_str(":")?;
            write!(printer, "{scalar:?}")
        }
        Expr::Const { bits, shape } => {
            if bits.len() == 1 {
                write!(printer, "{:#x}", bits[0])?;
            } else {
                printer.write_str("{")?;
                for (position, word) in bits.iter().enumerate() {
                    if position != 0 {
                        printer.write_str(", ")?;
                    }
                    write!(printer, "{word:#x}")?;
                }
                printer.write_str("}")?;
            }
            printer.write_str(":")?;
            write_shape::<D>(printer, shape)
        }
        Expr::Apply {
            operator, operands, ..
        } => {
            printer.write_str(D::mnemonic(operator))?;
            write_operands::<D>(printer, operands)
        }
        Expr::Reinterpret { operand, shape } => {
            printer.write_str("bitcast<")?;
            write_shape::<D>(printer, shape)?;
            printer.write_str(">(")?;
            write_expr::<D>(printer, operand)?;
            printer.write_str(")")
        }
    }
}

fn write_shape<D: Dialect>(
    printer: &mut IndentedWriter<'_>,
    shape: &Shape<D::Constraint>,
) -> fmt::Result {
    write!(printer, "{:?}", shape.scalar)?;
    if shape.lanes != 1 {
        write!(printer, "x{}", shape.lanes)?;
    }
    Ok(())
}
