//! Indented pseudocode rendering for control trees.

extern crate alloc;
use alloc::string::String;
use alloc::vec::Vec;
use core::fmt::{self, Write as _};

use super::node::{BlockSpan, ControlBody, ControlNode, ControlNodeId, LoopKind};
use super::tree::ControlTree;
use crate::ProgramPoint;
use crate::block::BlockId;
use crate::display::IndentedWriter;
use crate::region::HandlerKind;

impl ControlTree {
    /// Render this tree as indented pseudocode.
    ///
    /// The tree holds no instruction, so the caller prints them:
    /// `write_instruction` writes the text of the instruction at a
    /// [`ProgramPoint`] into the sink it is given, and writes nothing for an
    /// instruction that has no text. Each non-empty text is one line, except
    /// that the last instruction of a branch or dispatch block, its test, is
    /// the condition in the header of the `if` or `switch` it opens
    /// (`if (test) {`); a test with no text leaves the header bare.
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
        .write_tree(&mut IndentedWriter::new(sink), self.root())
    }
}

struct Printer<'tree, F> {
    tree: &'tree ControlTree,
    write_instruction: F,
}

/// What the printer still has to write.
enum Print {
    /// A statement.
    Node(ControlNodeId),
    /// The `}` that ends a block.
    Close,
    /// The `} else {` between the arms of an `if`.
    Else,
    /// The `} catch {` (or `fault`, `filter`) that opens a handler's body.
    Handler(HandlerKind),
    /// The `} finally {` that opens a `try`'s `finally` body.
    Finally,
    /// The `case {` or `default {` that opens a switch arm.
    Arm(&'static str),
    /// The latch of a post-tested loop, and the `} while (...)` that ends it.
    LoopTest {
        latch: BlockSpan,
        continue_on_true: bool,
    },
    /// The end of the statements a label indents.
    Unnest,
}

impl<F: FnMut(&mut dyn fmt::Write, ProgramPoint) -> fmt::Result> Printer<'_, F> {
    fn write_span(&mut self, out: &mut IndentedWriter<'_>, span: BlockSpan) -> fmt::Result {
        for point in span.points() {
            let write_instruction = &mut self.write_instruction;
            out.optional_line(|line| write_instruction(line, point))?;
        }
        Ok(())
    }

    /// Write the line that opens a branching construct: the instructions that
    /// compute the condition ahead of it, then `keyword` followed by the text
    /// of the last instruction — the test itself — in parentheses. An empty
    /// test text leaves the keyword bare.
    fn write_branch_header(
        &mut self,
        out: &mut IndentedWriter<'_>,
        keyword: &str,
        condition: BlockSpan,
    ) -> fmt::Result {
        let Some(test_point) = condition.points().last() else {
            out.indent()?;
            return out.write_str(keyword);
        };
        self.write_span(
            out,
            BlockSpan::new(condition.block(), condition.start(), condition.end() - 1),
        )?;
        let mut test = String::new();
        (self.write_instruction)(&mut test, test_point)?;
        out.indent()?;
        out.write_str(keyword)?;
        if !test.is_empty() {
            write!(out, " ({test})")?;
        }
        Ok(())
    }

    /// Write the tree below `root`. Nesting is as deep as the tree, so the
    /// printer keeps a stack of what is left to write instead of recursing.
    fn write_tree(&mut self, out: &mut IndentedWriter<'_>, root: ControlNodeId) -> fmt::Result {
        let mut pending = alloc::vec![Print::Node(root)];
        while let Some(action) = pending.pop() {
            match action {
                Print::Node(id) => self.write_node(out, id, &mut pending)?,
                Print::Close => out.close()?,
                Print::Else => {
                    out.close_inline()?;
                    out.write_str(" else")?;
                    out.open()?;
                }
                Print::Handler(kind) => {
                    out.close_inline()?;
                    match kind {
                        HandlerKind::Catch => out.write_str(" catch")?,
                        HandlerKind::CatchAll => out.write_str(" catch (...)")?,
                        HandlerKind::Finally => out.write_str(" finally")?,
                        HandlerKind::Fault => out.write_str(" fault")?,
                        HandlerKind::Filter { filter_block } => {
                            write!(out, " filter ({})", Label(filter_block))?;
                        }
                    }
                    out.open()?;
                }
                Print::Finally => {
                    out.close_inline()?;
                    out.write_str(" finally")?;
                    out.open()?;
                }
                Print::Arm(keyword) => {
                    out.indent()?;
                    out.write_str(keyword)?;
                    out.open()?;
                }
                Print::LoopTest {
                    latch,
                    continue_on_true,
                } => {
                    self.write_span(out, latch)?;
                    out.close_inline()?;
                    out.write_str(if continue_on_true {
                        " while (cond)"
                    } else {
                        " while (!cond)"
                    })?;
                    out.end_line()?;
                }
                Print::Unnest => out.unnest(),
            }
        }
        Ok(())
    }

    /// Queue the statements of `body`, the first to be written first.
    fn queue_body(&self, pending: &mut Vec<Print>, body: ControlBody) {
        pending.extend(self.tree.body(body).iter().rev().map(|&id| Print::Node(id)));
    }

    /// Write what opens the node `id`, and queue the rest of it: its child
    /// bodies and the text between and after them. A queued entry is written
    /// after everything queued above it.
    fn write_node(
        &mut self,
        out: &mut IndentedWriter<'_>,
        id: ControlNodeId,
        pending: &mut Vec<Print>,
    ) -> fmt::Result {
        let tree = self.tree;
        match *tree.node(id) {
            ControlNode::Block(span) | ControlNode::Return(span) => self.write_span(out, span),
            ControlNode::Sequence(body) => {
                self.queue_body(pending, body);
                Ok(())
            }
            ControlNode::IfThenElse {
                condition,
                then_body,
                else_body,
            } => {
                self.write_branch_header(out, "if", condition)?;
                out.open()?;
                pending.push(Print::Close);
                if !else_body.is_empty() {
                    self.queue_body(pending, else_body);
                    pending.push(Print::Else);
                }
                self.queue_body(pending, then_body);
                Ok(())
            }
            ControlNode::Loop { kind, body, .. } => self.write_loop(out, kind, body, pending),
            ControlNode::Switch {
                condition,
                cases,
                default_body,
                ..
            } => {
                self.write_branch_header(out, "switch", condition)?;
                out.open()?;
                pending.push(Print::Close);
                if !default_body.is_empty() {
                    pending.push(Print::Close);
                    self.queue_body(pending, default_body);
                    pending.push(Print::Arm("default"));
                }
                for case in tree.cases(cases).iter().rev() {
                    pending.push(Print::Close);
                    self.queue_body(pending, case.body);
                    pending.push(Print::Arm("case"));
                }
                Ok(())
            }
            ControlNode::Break { label } => write_transfer(out, "break", label),
            ControlNode::Continue { label } => write_transfer(out, "continue", label),
            ControlNode::Label { block, body } => {
                out.indent()?;
                write!(out, "{}:", Label(block))?;
                out.end_line()?;
                out.nest();
                pending.push(Print::Unnest);
                self.queue_body(pending, body);
                Ok(())
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
                out.indent()?;
                out.write_str("try")?;
                out.open()?;
                pending.push(Print::Close);
                if !finally_body.is_empty() {
                    self.queue_body(pending, finally_body);
                    pending.push(Print::Finally);
                }
                for handler in tree.handlers(handlers).iter().rev() {
                    self.queue_body(pending, handler.body);
                    pending.push(Print::Handler(handler.kind));
                }
                self.queue_body(pending, try_body);
                Ok(())
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
                pending.push(Print::Close);
                self.queue_body(pending, body);
                Ok(())
            }
        }
    }

    fn write_loop(
        &mut self,
        out: &mut IndentedWriter<'_>,
        kind: LoopKind,
        body: ControlBody,
        pending: &mut Vec<Print>,
    ) -> fmt::Result {
        match kind {
            LoopKind::Endless => {
                out.indent()?;
                out.write_str("loop")?;
                out.open()?;
                pending.push(Print::Close);
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
                pending.push(Print::Close);
            }
            LoopKind::DoWhile {
                latch,
                continue_on_true,
            } => {
                out.indent()?;
                out.write_str("do")?;
                out.open()?;
                pending.push(Print::LoopTest {
                    latch,
                    continue_on_true,
                });
            }
        }
        self.queue_body(pending, body);
        Ok(())
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
