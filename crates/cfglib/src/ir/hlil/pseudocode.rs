//! Indented pseudocode for a structured function.

extern crate alloc;

use alloc::string::String;
use core::fmt::{self, Write as _};
use core::marker::PhantomData;

use crate::display::IndentedWriter;

use super::dialect::Pseudocode;
use super::name::Name;
use super::shape::{ExprShape, HandlerKind, StmtShape};
use super::view::FunctionView;

/// Returns the body of `function` as indented pseudocode.
///
/// Variables print as `v{index}`. Operations print as
/// `mnemonic(operand, ...)`, with the mnemonic and the constants spelled by
/// the [`Pseudocode`] hooks of the dialect.
///
/// # Panics
///
/// Panics only if writing to an in-memory [`String`] unexpectedly fails.
#[must_use]
pub fn pseudocode<H, F>(function: &F) -> String
where
    H: Pseudocode,
    F: FunctionView<H>,
{
    let mut out = String::new();
    write_pseudocode(function, &mut out).expect("writing pseudocode to a String cannot fail");
    out
}

/// Writes the body of `function` as indented pseudocode into any sink.
///
/// # Errors
///
/// Returns the sink's formatting error if a write fails.
pub fn write_pseudocode<H, F>(function: &F, sink: &mut dyn fmt::Write) -> fmt::Result
where
    H: Pseudocode,
    F: FunctionView<H>,
{
    let printer = Printer {
        function,
        dialect: PhantomData,
    };
    printer.write_body(&mut IndentedWriter::new(sink), function.body())
}

struct ConstantDisplay<'a, H: Pseudocode>(&'a H::Constant);

impl<H: Pseudocode> fmt::Display for ConstantDisplay<'_, H> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        H::fmt_constant(formatter, self.0)
    }
}

struct Printer<'a, H: Pseudocode, F: FunctionView<H>> {
    function: &'a F,
    dialect: PhantomData<H>,
}

impl<H: Pseudocode, F: FunctionView<H>> Printer<'_, H, F> {
    fn write_expression(&self, sink: &mut dyn fmt::Write, expr: F::Expr) -> fmt::Result {
        match self.function.expr(expr) {
            ExprShape::Variable(variable) => write!(sink, "{}", Name::variable(variable)),
            ExprShape::Constant(constant) => write!(sink, "{}", ConstantDisplay::<H>(constant)),
            ExprShape::Operation { op, operands } => {
                sink.write_str(H::mnemonic(op))?;
                self.write_operands(sink, operands)
            }
        }
    }

    /// Writes `(a, b, c)`.
    fn write_operands(&self, sink: &mut dyn fmt::Write, operands: &[F::Expr]) -> fmt::Result {
        sink.write_str("(")?;
        for (position, &operand) in operands.iter().enumerate() {
            if position != 0 {
                sink.write_str(", ")?;
            }
            self.write_expression(sink, operand)?;
        }
        sink.write_str(")")
    }

    fn write_body(&self, printer: &mut IndentedWriter<'_>, body: &[F::Stmt]) -> fmt::Result {
        for &statement in body {
            self.write_statement(printer, statement)?;
        }
        Ok(())
    }

    /// Opens a block whose header is `keyword (expression)`.
    fn open_conditional(
        &self,
        printer: &mut IndentedWriter<'_>,
        keyword: &str,
        condition: F::Expr,
    ) -> fmt::Result {
        printer.write_str(keyword)?;
        printer.write_str(" (")?;
        self.write_expression(printer, condition)?;
        printer.write_str(")")?;
        printer.open()
    }

    fn write_statement(&self, printer: &mut IndentedWriter<'_>, stmt: F::Stmt) -> fmt::Result {
        printer.indent()?;
        match self.function.stmt(stmt) {
            StmtShape::Expression(expression) => {
                self.write_expression(printer, expression)?;
                printer.write_str(";")?;
                printer.end_line()
            }
            StmtShape::Assign { target, value } => {
                self.write_expression(printer, target)?;
                printer.write_str(" = ")?;
                self.write_expression(printer, value)?;
                printer.write_str(";")?;
                printer.end_line()
            }
            StmtShape::If {
                condition,
                then_body,
                else_body,
            } => self.write_if(printer, condition, then_body, else_body),
            StmtShape::While { condition, body } => {
                self.open_conditional(printer, "while", condition)?;
                self.write_body(printer, body)?;
                printer.close()
            }
            StmtShape::DoWhile { body, condition } => self.write_do_while(printer, body, condition),
            StmtShape::Loop { body } => {
                printer.write_str("loop")?;
                printer.open()?;
                self.write_body(printer, body)?;
                printer.close()
            }
            StmtShape::For {
                initializer,
                condition,
                update,
                body,
            } => self.write_for(printer, initializer, condition, update, body),
            StmtShape::Switch {
                scrutinee,
                default_body,
            } => self.write_switch(printer, stmt, scrutinee, default_body),
            StmtShape::Break { label } => write_transfer(printer, "break", label),
            StmtShape::Continue { label } => write_transfer(printer, "continue", label),
            StmtShape::Return { values } => {
                printer.write_str("return")?;
                for (position, &value) in values.iter().enumerate() {
                    printer.write_str(if position == 0 { " " } else { ", " })?;
                    self.write_expression(printer, value)?;
                }
                printer.write_str(";")?;
                printer.end_line()
            }
            StmtShape::Labeled { label, body } => {
                printer.write_str(label)?;
                printer.write_str(":")?;
                printer.open()?;
                self.write_body(printer, body)?;
                printer.close()
            }
            StmtShape::Goto { label } => {
                printer.write_str("goto ")?;
                printer.write_str(label)?;
                printer.write_str(";")?;
                printer.end_line()
            }
            StmtShape::Try { body, finally_body } => {
                self.write_try(printer, stmt, body, finally_body)
            }
            StmtShape::Region { op, operands, body } => {
                printer.write_str("@")?;
                printer.write_str(H::mnemonic(op))?;
                self.write_operands(printer, operands)?;
                printer.open()?;
                self.write_body(printer, body)?;
                printer.close()
            }
        }
    }

    fn write_if(
        &self,
        printer: &mut IndentedWriter<'_>,
        condition: F::Expr,
        then_body: &[F::Stmt],
        else_body: &[F::Stmt],
    ) -> fmt::Result {
        self.open_conditional(printer, "if", condition)?;
        self.write_body(printer, then_body)?;
        if !else_body.is_empty() {
            printer.close_inline()?;
            printer.write_str(" else")?;
            printer.open()?;
            self.write_body(printer, else_body)?;
        }
        printer.close()
    }

    fn write_do_while(
        &self,
        printer: &mut IndentedWriter<'_>,
        body: &[F::Stmt],
        condition: F::Expr,
    ) -> fmt::Result {
        printer.write_str("do")?;
        printer.open()?;
        self.write_body(printer, body)?;
        printer.close_inline()?;
        printer.write_str(" while (")?;
        self.write_expression(printer, condition)?;
        printer.write_str(");")?;
        printer.end_line()
    }

    fn write_for(
        &self,
        printer: &mut IndentedWriter<'_>,
        initializer: &[F::Stmt],
        condition: Option<F::Expr>,
        update: &[F::Stmt],
        body: &[F::Stmt],
    ) -> fmt::Result {
        printer.write_str("for (")?;
        printer.end_line()?;
        printer.nested(|printer| {
            self.write_body(printer, initializer)?;
            printer.indent()?;
            printer.write_str("; ")?;
            if let Some(condition) = condition {
                self.write_expression(printer, condition)?;
            }
            printer.write_str(" ;")?;
            printer.end_line()?;
            self.write_body(printer, update)
        })?;
        printer.indent()?;
        printer.write_str(")")?;
        printer.open()?;
        self.write_body(printer, body)?;
        printer.close()
    }

    fn write_switch(
        &self,
        printer: &mut IndentedWriter<'_>,
        stmt: F::Stmt,
        scrutinee: F::Expr,
        default_body: &[F::Stmt],
    ) -> fmt::Result {
        self.open_conditional(printer, "switch", scrutinee)?;
        for arm in self.function.switch_arms(stmt) {
            printer.indent()?;
            printer.write_str("case ")?;
            for (position, value) in arm.values.iter().enumerate() {
                if position != 0 {
                    printer.write_str(", ")?;
                }
                write!(printer, "{}", ConstantDisplay::<H>(value))?;
            }
            printer.write_str(":")?;
            printer.open()?;
            self.write_body(printer, arm.body)?;
            printer.close()?;
        }
        if !default_body.is_empty() {
            printer.indent()?;
            printer.write_str("default:")?;
            printer.open()?;
            self.write_body(printer, default_body)?;
            printer.close()?;
        }
        printer.close()
    }

    fn write_try(
        &self,
        printer: &mut IndentedWriter<'_>,
        stmt: F::Stmt,
        body: &[F::Stmt],
        finally_body: &[F::Stmt],
    ) -> fmt::Result {
        printer.write_str("try")?;
        printer.open()?;
        self.write_body(printer, body)?;
        for handler in self.function.handlers(stmt) {
            printer.close_inline()?;
            printer.write_str(match handler.kind {
                HandlerKind::Catch => " catch (",
                HandlerKind::CatchAll => " catch (...",
                HandlerKind::Fault => " fault (",
                HandlerKind::Filter { .. } => " filter (",
            })?;
            for (position, caught) in handler.caught_types.iter().enumerate() {
                if position != 0 {
                    printer.write_str(" | ")?;
                }
                write!(printer, "{caught:?}")?;
            }
            printer.write_str(")")?;
            if let Some(binding) = handler.binding {
                write!(printer, " {}", Name::variable(binding))?;
            }
            printer.open()?;
            if let HandlerKind::Filter { filter_body } = handler.kind {
                printer.indent()?;
                printer.write_str("when")?;
                printer.open()?;
                self.write_body(printer, filter_body)?;
                printer.close()?;
            }
            self.write_body(printer, handler.body)?;
        }
        if !finally_body.is_empty() {
            printer.close_inline()?;
            printer.write_str(" finally")?;
            printer.open()?;
            self.write_body(printer, finally_body)?;
        }
        printer.close()
    }
}

fn write_transfer(
    printer: &mut IndentedWriter<'_>,
    keyword: &str,
    label: Option<&str>,
) -> fmt::Result {
    printer.write_str(keyword)?;
    if let Some(label) = label {
        printer.write_str(" ")?;
        printer.write_str(label)?;
    }
    printer.write_str(";")?;
    printer.end_line()
}
