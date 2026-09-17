//! Indented pseudocode rendering for lifted ASTs.

extern crate alloc;
use alloc::string::String;
use core::fmt::{self, Write as _};

use crate::display::{DisplayInstr, IndentedWriter};
use crate::region::HandlerKind;

use super::node::{AstNode, CatchHandler, LoopKind, SwitchCase};

impl<I: DisplayInstr> AstNode<I> {
    /// Render this AST as indented pseudocode.
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

    /// Write this AST as indented pseudocode into any sink.
    ///
    /// # Errors
    ///
    /// Returns the sink's formatting error if a write fails.
    pub fn write_pseudocode(&self, sink: &mut dyn fmt::Write) -> fmt::Result {
        write_node(&mut IndentedWriter::new(sink), self)
    }
}

fn write_instructions<I: DisplayInstr>(
    printer: &mut IndentedWriter<'_>,
    instructions: &[I],
) -> fmt::Result {
    for instruction in instructions {
        printer.optional_line(|line| instruction.write_mnemonic(line))?;
    }
    Ok(())
}

fn write_nodes<I: DisplayInstr>(
    printer: &mut IndentedWriter<'_>,
    nodes: &[AstNode<I>],
) -> fmt::Result {
    for node in nodes {
        write_node(printer, node)?;
    }
    Ok(())
}

/// Write the instructions that compute a condition, keeping the last one —
/// the test itself — for the construct's own header.
fn write_condition_prefix<I: DisplayInstr>(
    printer: &mut IndentedWriter<'_>,
    condition: &[I],
) -> fmt::Result {
    match condition.split_last() {
        Some((_, leading)) if !leading.is_empty() => write_instructions(printer, leading),
        _ => Ok(()),
    }
}

fn write_if_then_else<I: DisplayInstr>(
    printer: &mut IndentedWriter<'_>,
    condition_instructions: &[I],
    then_body: &[AstNode<I>],
    else_body: &[AstNode<I>],
) -> fmt::Result {
    write_condition_prefix(printer, condition_instructions)?;
    printer.indent()?;
    printer.write_str("if")?;
    printer.open()?;
    write_nodes(printer, then_body)?;
    if !else_body.is_empty() {
        printer.close_inline()?;
        printer.write_str(" else")?;
        printer.open()?;
        write_nodes(printer, else_body)?;
    }
    printer.close()
}

fn write_loop<I: DisplayInstr>(
    printer: &mut IndentedWriter<'_>,
    kind: &LoopKind<I>,
    body: &[AstNode<I>],
) -> fmt::Result {
    match kind {
        LoopKind::Endless => {
            printer.indent()?;
            printer.write_str("loop")?;
            printer.open()?;
            write_nodes(printer, body)?;
            printer.close()
        }
        LoopKind::While {
            condition,
            exit_on_true,
            ..
        } => {
            printer.indent()?;
            printer.write_str("while")?;
            printer.open()?;
            write_instructions(printer, condition)?;
            printer.line(if *exit_on_true {
                "break if cond;"
            } else {
                "break if !cond;"
            })?;
            write_nodes(printer, body)?;
            printer.close()
        }
        LoopKind::DoWhile {
            condition,
            continue_on_true,
            ..
        } => {
            printer.indent()?;
            printer.write_str("do")?;
            printer.open()?;
            write_nodes(printer, body)?;
            write_instructions(printer, condition)?;
            printer.close_inline()?;
            printer.write_str(if *continue_on_true {
                " while (cond)"
            } else {
                " while (!cond)"
            })?;
            printer.end_line()
        }
    }
}

fn write_switch<I: DisplayInstr>(
    printer: &mut IndentedWriter<'_>,
    condition_instructions: &[I],
    cases: &[SwitchCase<I>],
    default_body: &[AstNode<I>],
) -> fmt::Result {
    write_condition_prefix(printer, condition_instructions)?;
    printer.indent()?;
    printer.write_str("switch")?;
    printer.open()?;
    for case in cases {
        printer.indent()?;
        printer.write_str("case")?;
        printer.open()?;
        write_nodes(printer, &case.body)?;
        printer.close()?;
    }
    if !default_body.is_empty() {
        printer.indent()?;
        printer.write_str("default")?;
        printer.open()?;
        write_nodes(printer, default_body)?;
        printer.close()?;
    }
    printer.close()
}

fn write_try_catch<I: DisplayInstr>(
    printer: &mut IndentedWriter<'_>,
    try_body: &[AstNode<I>],
    handlers: &[CatchHandler<I>],
    finally_body: &[AstNode<I>],
) -> fmt::Result {
    printer.indent()?;
    printer.write_str("try")?;
    printer.open()?;
    write_nodes(printer, try_body)?;
    for handler in handlers {
        printer.close_inline()?;
        match handler.kind {
            HandlerKind::Catch => printer.write_str(" catch")?,
            HandlerKind::CatchAll => printer.write_str(" catch (...)")?,
            HandlerKind::Finally => printer.write_str(" finally")?,
            HandlerKind::Fault => printer.write_str(" fault")?,
            HandlerKind::Filter { filter_block } => {
                write!(printer, " filter (.{filter_block})")?;
            }
        }
        printer.open()?;
        write_nodes(printer, &handler.body)?;
    }
    if !finally_body.is_empty() {
        printer.close_inline()?;
        printer.write_str(" finally")?;
        printer.open()?;
        write_nodes(printer, finally_body)?;
    }
    printer.close()
}

fn write_transfer(
    printer: &mut IndentedWriter<'_>,
    keyword: &str,
    label: Option<&String>,
) -> fmt::Result {
    printer.indent()?;
    printer.write_str(keyword)?;
    if let Some(label) = label {
        printer.write_str(" ")?;
        printer.write_str(label)?;
    }
    printer.write_str(";")?;
    printer.end_line()
}

fn write_node<I: DisplayInstr>(printer: &mut IndentedWriter<'_>, node: &AstNode<I>) -> fmt::Result {
    match node {
        AstNode::Block { instructions, .. } | AstNode::Return { instructions, .. } => {
            write_instructions(printer, instructions)
        }
        AstNode::Sequence { body } => write_nodes(printer, body),
        AstNode::IfThenElse {
            condition_instructions,
            then_body,
            else_body,
            ..
        } => write_if_then_else(printer, condition_instructions, then_body, else_body),
        AstNode::Loop { kind, body, .. } => write_loop(printer, kind, body),
        AstNode::Switch {
            condition_instructions,
            cases,
            default_body,
            ..
        } => write_switch(printer, condition_instructions, cases, default_body),
        AstNode::Break { label } => write_transfer(printer, "break", label.as_ref()),
        AstNode::Continue { label } => write_transfer(printer, "continue", label.as_ref()),
        AstNode::Label { name, body } => {
            printer.indent()?;
            printer.write_str(name)?;
            printer.write_str(":")?;
            printer.end_line()?;
            printer.nested(|printer| write_nodes(printer, body))
        }
        AstNode::Goto { target } => {
            printer.indent()?;
            printer.write_str("goto ")?;
            printer.write_str(target)?;
            printer.write_str(";")?;
            printer.end_line()
        }
        AstNode::TryCatch {
            try_body,
            handlers,
            finally_body,
        } => write_try_catch(printer, try_body, handlers, finally_body),
        AstNode::Guarded {
            predicate,
            when_true,
            body,
        } => {
            printer.indent()?;
            printer.write_str("@guarded(")?;
            if !when_true {
                printer.write_str("!")?;
            }
            predicate.write_mnemonic(printer)?;
            printer.write_str(")")?;
            printer.open()?;
            write_nodes(printer, body)?;
            printer.close()
        }
    }
}

impl<I: DisplayInstr> fmt::Display for AstNode<I> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.write_pseudocode(formatter)
    }
}
