//! The DOT form of a control-flow graph.
//!
//! A [`Cfg`] is one more style over the writer of
//! [`graph::dot`](crate::graph::dot): a block is labeled with its own name,
//! its identity, and one line per instruction, and an edge is drawn by
//! [`control_flow_edge_attributes`], which colors it by its kind.

extern crate alloc;
use alloc::borrow::Cow;
use alloc::format;
use alloc::string::String;
use core::fmt::{self, Write as _};

use crate::block::{BasicBlock, BlockId};
use crate::display::{DisplayInstr, render};
use crate::edge::{Edge, EdgeId, EdgeKind};
use crate::graph::dot::{DotEdgeAttributes, DotStyle, write_dot};

use super::Cfg;

impl<I, E> Cfg<I, E> {
    /// Write the CFG in DOT format, labeling each instruction with `label`.
    ///
    /// Rendering needs no trait on `I` at all. Labels are escaped before
    /// embedding, so raw program text is safe to return. Use
    /// [`render`] for the text as a [`String`], and
    /// [`to_dot`](Self::to_dot) when `I` implements [`DisplayInstr`].
    ///
    /// # Errors
    ///
    /// Returns the sink's formatting error if a write fails.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::borrow::Cow;
    /// use cfglib::{Cfg, render};
    ///
    /// let mut cfg = Cfg::<u32>::new();
    /// cfg.block_mut(cfg.entry()).push(7);
    /// let dot = render(|sink| cfg.write_dot(sink, |value| Cow::Owned(format!("push {value}"))));
    /// assert!(dot.contains("push 7"));
    /// ```
    pub fn write_dot(
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
}

impl<I: DisplayInstr, E> Cfg<I, E> {
    /// Produce the DOT representation as a [`String`], labeling each
    /// instruction with its [`mnemonic`](DisplayInstr::mnemonic).
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
        render(|sink| self.write_dot(sink, DisplayInstr::mnemonic))
    }
}

/// The color, line style, label, and thickness of one control-flow edge.
///
/// This is the hook [`Cfg::write_dot`] installs: every [`EdgeKind`] has a
/// fixed color and line style, and a weighted edge is labeled with its
/// probability and drawn proportionally thicker. A [`DotStyle`] for a
/// consumer's own control-flow view can install it too.
#[must_use]
pub fn control_flow_edge_attributes<E>(_edge: EdgeId, data: &Edge<E>) -> DotEdgeAttributes<'_> {
    let (color, style, kind_label) = match data.kind() {
        EdgeKind::Fallthrough => ("black", "solid", ""),
        EdgeKind::ConditionalTrue => ("green4", "solid", "T"),
        EdgeKind::ConditionalFalse => ("red", "solid", "F"),
        EdgeKind::Unconditional => ("blue", "solid", ""),
        EdgeKind::Back => ("blue", "dashed", "back"),
        EdgeKind::Call => ("purple", "solid", "call"),
        EdgeKind::CallReturn => ("purple", "dashed", "ret"),
        EdgeKind::SwitchCase => ("orange", "dotted", "case"),
        EdgeKind::Jump => ("blue", "bold", "jmp"),
        EdgeKind::IndirectJump => ("blue", "dotted", "ijmp"),
        EdgeKind::IndirectCall => ("purple", "dotted", "icall"),
        EdgeKind::ExceptionHandler => ("darkred", "solid", "handler"),
        EdgeKind::ExceptionUnwind => ("darkred", "dashed", "unwind"),
        EdgeKind::ExceptionLeave => ("darkred", "dotted", "leave"),
        EdgeKind::ExceptionResume => ("darkred", "bold", "resume"),
        EdgeKind::ExceptionContinue => ("darkgreen", "dashed", "continue"),
    };
    let label = match data.weight() {
        Some(weight) if kind_label.is_empty() => Cow::Owned(format!("({weight:.2})")),
        Some(weight) => Cow::Owned(format!("{kind_label} ({weight:.2})")),
        None => Cow::Borrowed(kind_label),
    };
    DotEdgeAttributes {
        label,
        color: Some(color),
        style: Some(style),
        // 1.0 to 4.0 points, so a hot edge reads as the spine of the drawing.
        penwidth: data.weight().map(|weight| 1.0 + weight * 3.0),
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
