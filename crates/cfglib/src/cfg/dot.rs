//! The DOT form of a control-flow graph.
//!
//! A [`Cfg`] is one more style over the writer of
//! [`graph::dot`](crate::graph::dot): a block is labeled with its own name,
//! its identity, and one line per instruction, and an edge is drawn by
//! [`control_flow_edge_attributes`], which colors it by its kind.

extern crate alloc;
use alloc::borrow::Cow;
use alloc::string::String;
use core::fmt::{self, Write as _};

use crate::block::{BasicBlock, BlockId};
use crate::display::DisplayInstr;
use crate::graph::dot::{DotStyle, control_flow_edge_attributes, write_dot};

use super::Cfg;

impl<I, E> Cfg<I, E> {
    /// Write the CFG in DOT format using a caller-supplied instruction label.
    ///
    /// This is the bound-free escape hatch: rendering needs no trait on `I`
    /// at all. Labels are escaped before embedding, so raw program text is
    /// safe to return.
    ///
    /// # Errors
    ///
    /// Returns the sink's formatting error if a write fails.
    pub fn write_dot_with(
        &self,
        sink: &mut dyn fmt::Write,
        label: impl Fn(&I) -> Cow<'_, str>,
    ) -> fmt::Result {
        let style = DotStyle::new(
            |id, block: &BasicBlock<I>| Cow::Owned(block_label(id, block, &label)),
            control_flow_edge_attributes,
        )
        .named("cfg")
        .node_prefix("bb");
        write_dot(self, sink, &style)
    }

    /// Produce the DOT representation as a [`String`] using a caller-supplied
    /// instruction label.
    ///
    /// # Panics
    ///
    /// Panics only if writing to an in-memory [`String`] unexpectedly fails.
    #[must_use]
    pub fn to_dot_with(&self, label: impl Fn(&I) -> Cow<'_, str>) -> String {
        let mut out = String::new();
        self.write_dot_with(&mut out, label)
            .expect("writing DOT to a String cannot fail");
        out
    }
}

impl<I: DisplayInstr, E> Cfg<I, E> {
    /// Write the CFG in DOT format to any `fmt::Write` sink.
    ///
    /// # Errors
    ///
    /// Returns the sink's formatting error if a write fails.
    pub fn write_dot(&self, sink: &mut dyn fmt::Write) -> fmt::Result {
        self.write_dot_with(sink, DisplayInstr::mnemonic)
    }

    /// Produce the DOT representation as a [`String`].
    ///
    /// # Panics
    ///
    /// Panics only if writing to an in-memory [`String`] unexpectedly fails.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::borrow::Cow;
    /// use cfglib::{Cfg, DisplayInstr, EdgeKind};
    ///
    /// #[derive(Debug, Clone)]
    /// struct Inst(&'static str);
    /// impl DisplayInstr for Inst {
    ///     fn mnemonic(&self) -> Cow<'_, str> { Cow::Borrowed(self.0) }
    /// }
    ///
    /// let mut cfg = Cfg::<Inst>::new();
    /// cfg.block_mut(cfg.entry()).push(Inst("nop"));
    /// let dot = cfg.to_dot();
    /// assert!(dot.contains("digraph cfg"));
    /// assert!(dot.contains("nop"));
    /// ```
    #[must_use]
    pub fn to_dot(&self) -> String {
        self.to_dot_with(DisplayInstr::mnemonic)
    }
}

/// The node label of a control-flow graph: the block's own name, its
/// identity, and one line per instruction.
fn block_label<I>(
    id: BlockId,
    block: &BasicBlock<I>,
    label: &impl Fn(&I) -> Cow<'_, str>,
) -> String {
    let mut text = String::new();
    if let Some(name) = block.label() {
        text.push_str(name);
        text.push_str(":\n");
    }
    let _ = write!(text, "{id}");
    let mut empty = true;
    for instruction in block.instructions() {
        let rendered = label(instruction);
        if !rendered.is_empty() {
            text.push('\n');
            text.push_str(&rendered);
            empty = false;
        }
    }
    if empty {
        text.push_str("\n(empty)");
    }
    text
}

#[cfg(test)]
mod tests {
    use crate::cfg::Cfg;
    use crate::edge::EdgeKind;
    use crate::test_util::golden::assert_golden;
    use crate::test_util::{MockInst, ff};

    #[test]
    fn a_cfg_renders_one_line_per_instruction() {
        let mut cfg = Cfg::new();
        let second = cfg.new_block();
        cfg.block_mut(cfg.entry())
            .instructions_mut()
            .push(ff("entry_inst"));
        cfg.block_mut(second)
            .instructions_mut()
            .push(ff("second_inst"));
        cfg.add_edge(cfg.entry(), second, EdgeKind::Fallthrough);

        assert_golden("dot/cfg-linear.dot", &cfg.to_dot());
    }

    #[test]
    fn conditional_arms_are_colored_and_labeled_by_kind() {
        let mut cfg = Cfg::new();
        let taken = cfg.new_block();
        let fallen = cfg.new_block();
        cfg.block_mut(cfg.entry()).instructions_mut().push(ff("br"));
        cfg.add_edge(cfg.entry(), taken, EdgeKind::ConditionalTrue);
        cfg.add_edge(cfg.entry(), fallen, EdgeKind::ConditionalFalse);

        assert_golden("dot/cfg-conditional.dot", &cfg.to_dot());
    }

    #[test]
    fn an_empty_block_says_so() {
        let cfg: Cfg<MockInst> = Cfg::new();

        assert_golden("dot/cfg-empty.dot", &cfg.to_dot());
    }

    #[test]
    fn a_weighted_edge_is_labeled_and_drawn_thicker() {
        let mut cfg = Cfg::new();
        let target = cfg.new_block();
        cfg.block_mut(cfg.entry()).instructions_mut().push(ff("a"));
        let edge = cfg.add_edge(cfg.entry(), target, EdgeKind::Fallthrough);
        cfg.edge_mut(edge).set_weight(Some(0.75));

        assert_golden("dot/cfg-weighted.dot", &cfg.to_dot());
    }
}
