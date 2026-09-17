//! Indented pseudocode rendering for HLIL functions.

extern crate alloc;

use alloc::string::String;
use core::fmt::{self, Write as _};

use crate::display::IndentedWriter;

use super::statement::HandlerKind;
use super::{Dialect, ExpressionId, ExpressionKind, Function, StatementId, StatementKind};

struct ConstantDisplay<'a, D: Dialect>(&'a D::Constant);

impl<D: Dialect> fmt::Display for ConstantDisplay<'_, D> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        D::fmt_constant(formatter, self.0)
    }
}

impl<D: Dialect> Function<D> {
    /// Render this function's body as indented pseudocode.
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

    /// Write this function's body as indented pseudocode into any sink.
    ///
    /// # Errors
    ///
    /// Returns the sink's formatting error if a write fails.
    pub fn write_pseudocode(&self, sink: &mut dyn fmt::Write) -> fmt::Result {
        self.write_body(&mut IndentedWriter::new(sink), &self.body)
    }

    fn write_expression(&self, sink: &mut dyn fmt::Write, id: ExpressionId) -> fmt::Result {
        let Some(expression) = self.expression(id) else {
            return write!(sink, "<missing {id}>");
        };
        match expression.kind() {
            ExpressionKind::Variable(variable) => write!(sink, "{variable}"),
            ExpressionKind::Constant(constant) => {
                write!(sink, "{}", ConstantDisplay::<D>(constant))
            }
            ExpressionKind::Operation {
                operation,
                operands,
            } => {
                sink.write_str(D::mnemonic(operation))?;
                sink.write_str("(")?;
                for (position, &operand) in operands.iter().enumerate() {
                    if position != 0 {
                        sink.write_str(", ")?;
                    }
                    self.write_expression(sink, operand)?;
                }
                sink.write_str(")")
            }
        }
    }

    fn write_body(&self, printer: &mut IndentedWriter<'_>, body: &[StatementId]) -> fmt::Result {
        for &statement in body {
            self.write_statement(printer, statement)?;
        }
        Ok(())
    }

    /// Open a block whose header is `keyword (expression)`.
    fn open_conditional(
        &self,
        printer: &mut IndentedWriter<'_>,
        keyword: &str,
        condition: ExpressionId,
    ) -> fmt::Result {
        printer.write_str(keyword)?;
        printer.write_str(" (")?;
        self.write_expression(printer, condition)?;
        printer.write_str(")")?;
        printer.open()
    }

    fn write_statement(&self, printer: &mut IndentedWriter<'_>, id: StatementId) -> fmt::Result {
        let Some(statement) = self.statement(id) else {
            return printer.optional_line(|line| write!(line, "<missing {id}>"));
        };
        printer.indent()?;
        match statement.kind() {
            StatementKind::Expression(expression) => {
                self.write_expression(printer, *expression)?;
                printer.write_str(";")?;
                printer.end_line()
            }
            StatementKind::Assign { target, value } => {
                self.write_expression(printer, *target)?;
                printer.write_str(" = ")?;
                self.write_expression(printer, *value)?;
                printer.write_str(";")?;
                printer.end_line()
            }
            StatementKind::If {
                condition,
                then_body,
                else_body,
            } => self.write_if(printer, *condition, then_body, else_body),
            StatementKind::While { condition, body } => {
                self.open_conditional(printer, "while", *condition)?;
                self.write_body(printer, body)?;
                printer.close()
            }
            StatementKind::DoWhile { body, condition } => {
                self.write_do_while(printer, body, *condition)
            }
            StatementKind::Loop { body } => {
                printer.write_str("loop")?;
                printer.open()?;
                self.write_body(printer, body)?;
                printer.close()
            }
            StatementKind::For {
                initializer,
                condition,
                update,
                body,
            } => self.write_for(printer, initializer, *condition, update, body),
            StatementKind::Switch {
                scrutinee,
                cases,
                default_body,
            } => self.write_switch(printer, *scrutinee, cases, default_body),
            StatementKind::Break { label } => write_transfer(printer, "break", label.as_deref()),
            StatementKind::Continue { label } => {
                write_transfer(printer, "continue", label.as_deref())
            }
            StatementKind::Return { values } => {
                printer.write_str("return")?;
                for (position, &value) in values.iter().enumerate() {
                    printer.write_str(if position == 0 { " " } else { ", " })?;
                    self.write_expression(printer, value)?;
                }
                printer.write_str(";")?;
                printer.end_line()
            }
            StatementKind::Labeled { label, body } => {
                printer.write_str(label)?;
                printer.write_str(":")?;
                printer.open()?;
                self.write_body(printer, body)?;
                printer.close()
            }
            StatementKind::Goto { label } => {
                printer.write_str("goto ")?;
                printer.write_str(label)?;
                printer.write_str(";")?;
                printer.end_line()
            }
            StatementKind::Try {
                body,
                handlers,
                finally_body,
            } => self.write_try(printer, body, handlers, finally_body),
            StatementKind::Region {
                operation,
                operands,
                body,
            } => {
                printer.write_str("@")?;
                printer.write_str(D::mnemonic(operation))?;
                printer.write_str("(")?;
                for (position, &operand) in operands.iter().enumerate() {
                    if position != 0 {
                        printer.write_str(", ")?;
                    }
                    self.write_expression(printer, operand)?;
                }
                printer.write_str(")")?;
                printer.open()?;
                self.write_body(printer, body)?;
                printer.close()
            }
        }
    }

    fn write_if(
        &self,
        printer: &mut IndentedWriter<'_>,
        condition: ExpressionId,
        then_body: &[StatementId],
        else_body: &[StatementId],
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
        body: &[StatementId],
        condition: ExpressionId,
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
        initializer: &[StatementId],
        condition: Option<ExpressionId>,
        update: &[StatementId],
        body: &[StatementId],
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
        scrutinee: ExpressionId,
        cases: &[super::statement::SwitchArm<D>],
        default_body: &[StatementId],
    ) -> fmt::Result {
        self.open_conditional(printer, "switch", scrutinee)?;
        for case in cases {
            printer.indent()?;
            printer.write_str("case ")?;
            for (position, value) in case.values.iter().enumerate() {
                if position != 0 {
                    printer.write_str(", ")?;
                }
                write!(printer, "{}", ConstantDisplay::<D>(value))?;
            }
            printer.write_str(":")?;
            printer.open()?;
            self.write_body(printer, &case.body)?;
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
        body: &[StatementId],
        handlers: &[super::statement::Handler<D>],
        finally_body: &[StatementId],
    ) -> fmt::Result {
        printer.write_str("try")?;
        printer.open()?;
        self.write_body(printer, body)?;
        for handler in handlers {
            printer.close_inline()?;
            printer.write_str(match &handler.kind {
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
                write!(printer, " {binding}")?;
            }
            printer.open()?;
            if let HandlerKind::Filter { filter_body } = &handler.kind {
                printer.indent()?;
                printer.write_str("when")?;
                printer.open()?;
                self.write_body(printer, filter_body)?;
                printer.close()?;
            }
            self.write_body(printer, &handler.body)?;
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
