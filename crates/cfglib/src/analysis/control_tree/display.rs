//! Indented pseudocode rendering for control trees.

extern crate alloc;
use alloc::string::String;
use core::fmt::{self, Write as _};

use super::node::{BlockSpan, ControlBody, ControlNode, ControlNodeId, LoopKind};
use super::tree::ControlTree;
use crate::block::BlockId;
use crate::dataflow::ProgramPoint;
use crate::display::IndentedWriter;
use crate::region::HandlerKind;

impl ControlTree {
    /// Render this tree as indented pseudocode.
    ///
    /// The tree holds no instruction, so the caller prints them:
    /// `write_instruction` writes the text of the instruction at a
    /// [`ProgramPoint`] into the sink it is given, and writes nothing for an
    /// instruction that has no text. Each non-empty text is one line.
    ///
    /// # Errors
    ///
    /// Returns an error if `write_instruction` fails, or if writing to the
    /// in-memory [`String`] fails.
    ///
    /// # Examples
    ///
    /// ```
    /// use cfglib::{Cfg, ControlTree};
    ///
    /// let mut cfg = Cfg::<&str>::new();
    /// cfg.block_mut(cfg.entry()).push("a");
    /// cfg.block_mut(cfg.entry()).push("b");
    ///
    /// let tree = ControlTree::compute(&cfg);
    /// let text = tree
    ///     .to_pseudocode(|sink, point| {
    ///         sink.write_str(cfg.block(point.block).instructions()[point.inst_idx])
    ///     })
    ///     .unwrap();
    /// assert_eq!(text, "a\nb\n");
    /// ```
    pub fn to_pseudocode(
        &self,
        write_instruction: impl FnMut(&mut dyn fmt::Write, ProgramPoint) -> fmt::Result,
    ) -> Result<String, fmt::Error> {
        let mut out = String::new();
        self.write_pseudocode(&mut out, write_instruction)?;
        Ok(out)
    }

    /// Write this tree as indented pseudocode into any sink, with the
    /// instruction text supplied by `write_instruction`.
    ///
    /// See [`to_pseudocode`](Self::to_pseudocode).
    ///
    /// # Errors
    ///
    /// Returns an error if `write_instruction` fails, or if a write to the
    /// sink fails.
    pub fn write_pseudocode(
        &self,
        sink: &mut dyn fmt::Write,
        write_instruction: impl FnMut(&mut dyn fmt::Write, ProgramPoint) -> fmt::Result,
    ) -> fmt::Result {
        Printer {
            tree: self,
            write_instruction,
        }
        .write_node(&mut IndentedWriter::new(sink), self.root())
    }
}

struct Printer<'tree, F> {
    tree: &'tree ControlTree,
    write_instruction: F,
}

impl<F: FnMut(&mut dyn fmt::Write, ProgramPoint) -> fmt::Result> Printer<'_, F> {
    fn write_span(&mut self, out: &mut IndentedWriter<'_>, span: BlockSpan) -> fmt::Result {
        for point in span.points() {
            let write_instruction = &mut self.write_instruction;
            out.optional_line(|line| write_instruction(line, point))?;
        }
        Ok(())
    }

    fn write_nodes(&mut self, out: &mut IndentedWriter<'_>, body: ControlBody) -> fmt::Result {
        for &id in self.tree.body(body) {
            self.write_node(out, id)?;
        }
        Ok(())
    }

    /// Write the instructions that compute a condition, keeping the last one —
    /// the test itself — for the construct's own header.
    fn write_condition_prefix(
        &mut self,
        out: &mut IndentedWriter<'_>,
        condition: BlockSpan,
    ) -> fmt::Result {
        if condition.len() > 1 {
            self.write_span(
                out,
                BlockSpan::new(condition.block(), condition.start(), condition.end() - 1),
            )?;
        }
        Ok(())
    }

    fn write_if_then_else(
        &mut self,
        out: &mut IndentedWriter<'_>,
        condition: BlockSpan,
        then_body: ControlBody,
        else_body: ControlBody,
    ) -> fmt::Result {
        self.write_condition_prefix(out, condition)?;
        out.indent()?;
        out.write_str("if")?;
        out.open()?;
        self.write_nodes(out, then_body)?;
        if !else_body.is_empty() {
            out.close_inline()?;
            out.write_str(" else")?;
            out.open()?;
            self.write_nodes(out, else_body)?;
        }
        out.close()
    }

    fn write_loop(
        &mut self,
        out: &mut IndentedWriter<'_>,
        kind: LoopKind,
        body: ControlBody,
    ) -> fmt::Result {
        match kind {
            LoopKind::Endless => {
                out.indent()?;
                out.write_str("loop")?;
                out.open()?;
                self.write_nodes(out, body)?;
                out.close()
            }
            LoopKind::While {
                condition,
                exit_on_true,
            } => {
                out.indent()?;
                out.write_str("while")?;
                out.open()?;
                for &span in self.tree.spans(condition) {
                    self.write_span(out, span)?;
                }
                out.line(if exit_on_true {
                    "break if cond;"
                } else {
                    "break if !cond;"
                })?;
                self.write_nodes(out, body)?;
                out.close()
            }
            LoopKind::DoWhile {
                latch,
                continue_on_true,
            } => {
                out.indent()?;
                out.write_str("do")?;
                out.open()?;
                self.write_nodes(out, body)?;
                self.write_span(out, latch)?;
                out.close_inline()?;
                out.write_str(if continue_on_true {
                    " while (cond)"
                } else {
                    " while (!cond)"
                })?;
                out.end_line()
            }
        }
    }

    fn write_switch(
        &mut self,
        out: &mut IndentedWriter<'_>,
        condition: BlockSpan,
        cases: &[super::node::SwitchCase],
        default_body: ControlBody,
    ) -> fmt::Result {
        self.write_condition_prefix(out, condition)?;
        out.indent()?;
        out.write_str("switch")?;
        out.open()?;
        for case in cases {
            out.indent()?;
            out.write_str("case")?;
            out.open()?;
            self.write_nodes(out, case.body)?;
            out.close()?;
        }
        if !default_body.is_empty() {
            out.indent()?;
            out.write_str("default")?;
            out.open()?;
            self.write_nodes(out, default_body)?;
            out.close()?;
        }
        out.close()
    }

    fn write_try_catch(
        &mut self,
        out: &mut IndentedWriter<'_>,
        try_body: ControlBody,
        handlers: &[super::node::CatchHandler],
        finally_body: ControlBody,
    ) -> fmt::Result {
        out.indent()?;
        out.write_str("try")?;
        out.open()?;
        self.write_nodes(out, try_body)?;
        for handler in handlers {
            out.close_inline()?;
            match handler.kind {
                HandlerKind::Catch => out.write_str(" catch")?,
                HandlerKind::CatchAll => out.write_str(" catch (...)")?,
                HandlerKind::Finally => out.write_str(" finally")?,
                HandlerKind::Fault => out.write_str(" fault")?,
                HandlerKind::Filter { filter_block } => {
                    write!(out, " filter ({})", Label(filter_block))?;
                }
            }
            out.open()?;
            self.write_nodes(out, handler.body)?;
        }
        if !finally_body.is_empty() {
            out.close_inline()?;
            out.write_str(" finally")?;
            out.open()?;
            self.write_nodes(out, finally_body)?;
        }
        out.close()
    }

    fn write_node(&mut self, out: &mut IndentedWriter<'_>, id: ControlNodeId) -> fmt::Result {
        match *self.tree.node(id) {
            ControlNode::Block(span) | ControlNode::Return(span) => self.write_span(out, span),
            ControlNode::Sequence(body) => self.write_nodes(out, body),
            ControlNode::IfThenElse {
                condition,
                then_body,
                else_body,
            } => self.write_if_then_else(out, condition, then_body, else_body),
            ControlNode::Loop { kind, body, .. } => self.write_loop(out, kind, body),
            ControlNode::Switch {
                condition,
                cases,
                default_body,
                ..
            } => {
                let tree = self.tree;
                self.write_switch(out, condition, tree.cases(cases), default_body)
            }
            ControlNode::Break { label } => write_transfer(out, "break", label),
            ControlNode::Continue { label } => write_transfer(out, "continue", label),
            ControlNode::Label { block, body } => {
                out.indent()?;
                write!(out, "{}:", Label(block))?;
                out.end_line()?;
                out.nested(|out| self.write_nodes(out, body))
            }
            ControlNode::Goto { target } => {
                out.indent()?;
                write!(out, "goto {};", Label(target))?;
                out.end_line()
            }
            ControlNode::TryCatch {
                try_body,
                handlers,
                finally_body,
            } => {
                let tree = self.tree;
                self.write_try_catch(out, try_body, tree.handlers(handlers), finally_body)
            }
            ControlNode::Guarded {
                predicate,
                when_true,
                body,
            } => {
                out.indent()?;
                out.write_str("@guarded(")?;
                if !when_true {
                    out.write_str("!")?;
                }
                (self.write_instruction)(out, predicate)?;
                out.write_str(")")?;
                out.open()?;
                self.write_nodes(out, body)?;
                out.close()
            }
        }
    }
}

/// The text of a block's label: the block's identity, which is all a tree
/// knows of it.
struct Label(BlockId);

impl fmt::Display for Label {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, ".{}", self.0)
    }
}

fn write_transfer(
    out: &mut IndentedWriter<'_>,
    keyword: &str,
    label: Option<BlockId>,
) -> fmt::Result {
    out.indent()?;
    out.write_str(keyword)?;
    if let Some(label) = label {
        write!(out, " {}", Label(label))?;
    }
    out.write_str(";")?;
    out.end_line()
}
