//! Indented pseudocode for a structured function.

extern crate alloc;

use alloc::string::String;
use alloc::vec;
use alloc::vec::Vec;
use core::fmt::{self, Write as _};
use core::marker::PhantomData;
use core::slice;

use crate::display::IndentedWriter;

use super::dialect::Pseudocode;
use super::name::Name;
use super::shape::{ExprShape, Handler, HandlerKind, StmtShape};
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

/// The operand lists of the operations being written, innermost last, each
/// with the position of its next operand.
type OpenOperands<'a, E> = Vec<(&'a [E], usize)>;

/// What the printer has still to write: the rest of a body, or the text that
/// follows a child body or sits between two.
enum Pending<'a, H: Pseudocode, F: FunctionView<H>> {
    /// The rest of a body.
    Body(slice::Iter<'a, F::Stmt>),
    /// `}` and the end of the line.
    Close,
    /// `} else {`
    Otherwise,
    /// `} while (condition);`
    DoWhileEnd(F::Expr),
    /// The line of a `for` between its two clauses.
    ForCondition(Option<F::Expr>),
    /// The end of the clauses of a `for`, and the opening of its body.
    ForBody,
    /// The header of a `case` arm.
    Case(&'a [H::Constant]),
    /// The header of the `default` arm.
    Default,
    /// A handler of a `try`.
    Handler(Handler<'a, H, F::Stmt, F::Var>),
    /// `} finally {`
    Finally,
}

/// Schedules `steps`, given in the order they run: the first is on top of
/// `pending`, so it runs next.
fn schedule<T>(pending: &mut Vec<T>, steps: impl IntoIterator<Item = T>) {
    let first = pending.len();
    pending.extend(steps);
    pending[first..].reverse();
}

impl<'a, H: Pseudocode, F: FunctionView<H>> Printer<'a, H, F> {
    fn write_expression(&self, sink: &mut dyn fmt::Write, expr: F::Expr) -> fmt::Result {
        let mut open = Vec::new();
        self.begin_expression(sink, expr, &mut open)?;
        self.write_open_operands(sink, open)
    }

    /// Writes `(a, b, c)`.
    fn write_operands(&self, sink: &mut dyn fmt::Write, operands: &'a [F::Expr]) -> fmt::Result {
        sink.write_str("(")?;
        self.write_open_operands(sink, vec![(operands, 0)])
    }

    /// Writes a variable or a constant whole, or the mnemonic and the opening
    /// parenthesis of an operation, which leaves its operands in `open`.
    fn begin_expression(
        &self,
        sink: &mut dyn fmt::Write,
        expr: F::Expr,
        open: &mut OpenOperands<'a, F::Expr>,
    ) -> fmt::Result {
        match self.function.expr(expr) {
            ExprShape::Variable(variable) => write!(sink, "{}", Name::variable(variable)),
            ExprShape::Constant(constant) => write!(sink, "{}", ConstantDisplay::<H>(constant)),
            ExprShape::Operation { op, operands } => {
                sink.write_str(H::mnemonic(op))?;
                sink.write_str("(")?;
                open.push((operands, 0));
                Ok(())
            }
        }
    }

    /// Writes the rest of every open operand list, with its closing
    /// parenthesis.
    ///
    /// The depth of an expression costs no call frames: the lists wait in
    /// `open` instead.
    fn write_open_operands(
        &self,
        sink: &mut dyn fmt::Write,
        mut open: OpenOperands<'a, F::Expr>,
    ) -> fmt::Result {
        while let Some((operands, next)) = open.last_mut() {
            let Some(&operand) = operands.get(*next) else {
                open.pop();
                sink.write_str(")")?;
                continue;
            };
            if *next != 0 {
                sink.write_str(", ")?;
            }
            *next += 1;
            self.begin_expression(sink, operand, &mut open)?;
        }
        Ok(())
    }

    /// Writes `body` and everything nested in it.
    ///
    /// The depth of the nesting costs no call frames: what a statement has
    /// still to write after its bodies, or between them, waits in `pending`
    /// instead.
    fn write_body(&self, printer: &mut IndentedWriter<'_>, body: &'a [F::Stmt]) -> fmt::Result {
        let mut pending = vec![Pending::Body(body.iter())];
        while let Some(step) = pending.pop() {
            match step {
                Pending::Body(mut statements) => {
                    if let Some(&statement) = statements.next() {
                        pending.push(Pending::Body(statements));
                        self.write_statement(printer, statement, &mut pending)?;
                    }
                }
                Pending::Close => printer.close()?,
                Pending::Otherwise => {
                    printer.close_inline()?;
                    printer.write_str(" else")?;
                    printer.open()?;
                }
                Pending::DoWhileEnd(condition) => {
                    printer.close_inline()?;
                    printer.write_str(" while (")?;
                    self.write_expression(printer, condition)?;
                    printer.write_str(");")?;
                    printer.end_line()?;
                }
                Pending::ForCondition(condition) => {
                    printer.indent()?;
                    printer.write_str("; ")?;
                    if let Some(condition) = condition {
                        self.write_expression(printer, condition)?;
                    }
                    printer.write_str(" ;")?;
                    printer.end_line()?;
                }
                Pending::ForBody => {
                    printer.unnest();
                    printer.indent()?;
                    printer.write_str(")")?;
                    printer.open()?;
                }
                Pending::Case(values) => {
                    printer.indent()?;
                    printer.write_str("case ")?;
                    for (position, value) in values.iter().enumerate() {
                        if position != 0 {
                            printer.write_str(", ")?;
                        }
                        write!(printer, "{}", ConstantDisplay::<H>(value))?;
                    }
                    printer.write_str(":")?;
                    printer.open()?;
                }
                Pending::Default => {
                    printer.indent()?;
                    printer.write_str("default:")?;
                    printer.open()?;
                }
                Pending::Handler(handler) => Self::write_handler(printer, &handler, &mut pending)?,
                Pending::Finally => {
                    printer.close_inline()?;
                    printer.write_str(" finally")?;
                    printer.open()?;
                }
            }
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

    /// Opens the block of a statement with no header but `text`, and
    /// schedules its body and the end of the block.
    fn open_body(
        printer: &mut IndentedWriter<'_>,
        text: &str,
        body: &'a [F::Stmt],
        pending: &mut Vec<Pending<'a, H, F>>,
    ) -> fmt::Result {
        printer.write_str(text)?;
        printer.open()?;
        schedule(pending, [Pending::Body(body.iter()), Pending::Close]);
        Ok(())
    }

    fn write_if(
        &self,
        printer: &mut IndentedWriter<'_>,
        condition: F::Expr,
        then_body: &'a [F::Stmt],
        else_body: &'a [F::Stmt],
        pending: &mut Vec<Pending<'a, H, F>>,
    ) -> fmt::Result {
        self.open_conditional(printer, "if", condition)?;
        let otherwise = (!else_body.is_empty())
            .then(|| [Pending::Otherwise, Pending::Body(else_body.iter())])
            .into_iter()
            .flatten();
        schedule(
            pending,
            [Pending::Body(then_body.iter())]
                .into_iter()
                .chain(otherwise)
                .chain([Pending::Close]),
        );
        Ok(())
    }

    fn write_for(
        printer: &mut IndentedWriter<'_>,
        initializer: &'a [F::Stmt],
        condition: Option<F::Expr>,
        update: &'a [F::Stmt],
        body: &'a [F::Stmt],
        pending: &mut Vec<Pending<'a, H, F>>,
    ) -> fmt::Result {
        printer.write_str("for (")?;
        printer.end_line()?;
        printer.nest();
        schedule(
            pending,
            [
                Pending::Body(initializer.iter()),
                Pending::ForCondition(condition),
                Pending::Body(update.iter()),
                Pending::ForBody,
                Pending::Body(body.iter()),
                Pending::Close,
            ],
        );
        Ok(())
    }

    fn write_switch(
        &self,
        printer: &mut IndentedWriter<'_>,
        stmt: F::Stmt,
        scrutinee: F::Expr,
        default_body: &'a [F::Stmt],
        pending: &mut Vec<Pending<'a, H, F>>,
    ) -> fmt::Result {
        self.open_conditional(printer, "switch", scrutinee)?;
        let arms = self.function.switch_arms(stmt).flat_map(|arm| {
            [
                Pending::Case(arm.values),
                Pending::Body(arm.body.iter()),
                Pending::Close,
            ]
        });
        let default = (!default_body.is_empty())
            .then(|| {
                [
                    Pending::Default,
                    Pending::Body(default_body.iter()),
                    Pending::Close,
                ]
            })
            .into_iter()
            .flatten();
        schedule(pending, arms.chain(default).chain([Pending::Close]));
        Ok(())
    }

    fn write_try(
        &self,
        printer: &mut IndentedWriter<'_>,
        stmt: F::Stmt,
        body: &'a [F::Stmt],
        finally_body: &'a [F::Stmt],
        pending: &mut Vec<Pending<'a, H, F>>,
    ) -> fmt::Result {
        printer.write_str("try")?;
        printer.open()?;
        let handlers = self.function.handlers(stmt).map(Pending::Handler);
        let finally = (!finally_body.is_empty())
            .then(|| [Pending::Finally, Pending::Body(finally_body.iter())])
            .into_iter()
            .flatten();
        schedule(
            pending,
            [Pending::Body(body.iter())]
                .into_iter()
                .chain(handlers)
                .chain(finally)
                .chain([Pending::Close]),
        );
        Ok(())
    }

    /// Writes the first line of `stmt`, up to what its child bodies write,
    /// and schedules the rest.
    fn write_statement(
        &self,
        printer: &mut IndentedWriter<'_>,
        stmt: F::Stmt,
        pending: &mut Vec<Pending<'a, H, F>>,
    ) -> fmt::Result {
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
            } => self.write_if(printer, condition, then_body, else_body, pending),
            StmtShape::While { condition, body } => {
                self.open_conditional(printer, "while", condition)?;
                schedule(pending, [Pending::Body(body.iter()), Pending::Close]);
                Ok(())
            }
            StmtShape::DoWhile { body, condition } => {
                printer.write_str("do")?;
                printer.open()?;
                schedule(
                    pending,
                    [Pending::Body(body.iter()), Pending::DoWhileEnd(condition)],
                );
                Ok(())
            }
            StmtShape::Loop { body } => Self::open_body(printer, "loop", body, pending),
            StmtShape::For {
                initializer,
                condition,
                update,
                body,
            } => Self::write_for(printer, initializer, condition, update, body, pending),
            StmtShape::Switch {
                scrutinee,
                default_body,
            } => self.write_switch(printer, stmt, scrutinee, default_body, pending),
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
                Self::open_body(printer, ":", body, pending)
            }
            StmtShape::Goto { label } => {
                printer.write_str("goto ")?;
                printer.write_str(label)?;
                printer.write_str(";")?;
                printer.end_line()
            }
            StmtShape::Try { body, finally_body } => {
                self.write_try(printer, stmt, body, finally_body, pending)
            }
            StmtShape::Region { op, operands, body } => {
                printer.write_str("@")?;
                printer.write_str(H::mnemonic(op))?;
                self.write_operands(printer, operands)?;
                printer.open()?;
                schedule(pending, [Pending::Body(body.iter()), Pending::Close]);
                Ok(())
            }
        }
    }

    /// Writes the header of `handler` and schedules its bodies.
    fn write_handler(
        printer: &mut IndentedWriter<'_>,
        handler: &Handler<'a, H, F::Stmt, F::Var>,
        pending: &mut Vec<Pending<'a, H, F>>,
    ) -> fmt::Result {
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
            schedule(
                pending,
                [
                    Pending::Body(filter_body.iter()),
                    Pending::Close,
                    Pending::Body(handler.body.iter()),
                ],
            );
        } else {
            schedule(pending, [Pending::Body(handler.body.iter())]);
        }
        Ok(())
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
