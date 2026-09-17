//! DOT (Graphviz) export for control-flow graphs.

extern crate alloc;
use alloc::borrow::Cow;
use alloc::string::String;
use core::fmt;

use crate::cfg::Cfg;
use crate::display::DisplayInstr;
use crate::edge::EdgeKind;
use crate::graph::view::{DenseId, GraphView};

/// Escape label text for safe embedding in a double-quoted DOT string.
///
/// Backslashes and quotes are escaped, and raw line breaks become DOT `\n`
/// escape sequences, so labels sourced from real program text (statements,
/// string literals) cannot break out of the attribute.
fn escape_label(label: &str) -> String {
    let mut escaped = String::with_capacity(label.len());
    for ch in label.chars() {
        match ch {
            '\\' => escaped.push_str("\\\\"),
            '"' => escaped.push_str("\\\""),
            '\n' => escaped.push_str("\\n"),
            '\r' => {}
            _ => escaped.push(ch),
        }
    }
    escaped
}

/// Write any graph view in DOT format with consumer-provided node labels.
///
/// Nodes are named `n{index}`; the label callback supplies display text,
/// which is escaped before embedding. This is the topology-only counterpart
/// of [`Cfg::write_dot`] for value-flow, call, and type-relation graphs.
///
/// # Errors
///
/// Returns the sink's formatting error if a write fails.
pub fn write_view_dot<G: GraphView>(
    graph: &G,
    w: &mut dyn fmt::Write,
    mut node_label: impl FnMut(G::NodeId) -> String,
) -> fmt::Result {
    writeln!(w, "digraph view {{")?;
    writeln!(
        w,
        "    node [shape=box fontname=\"monospace\" fontsize=10];"
    )?;
    for node in graph.node_ids() {
        let label = escape_label(&node_label(node));
        writeln!(w, "    n{} [label=\"{label}\"];", node.index())?;
    }
    for node in graph.node_ids() {
        for successor in graph.successors(node) {
            writeln!(w, "    n{} -> n{};", node.index(), successor.index())?;
        }
    }
    writeln!(w, "}}")
}

/// Render any graph view in DOT format with consumer-provided node labels.
///
/// The allocating counterpart of [`write_view_dot`], matching
/// [`Cfg::to_dot_with`].
///
/// # Panics
///
/// Panics only if writing to an in-memory [`String`] unexpectedly fails.
#[must_use]
pub fn to_view_dot<G: GraphView>(graph: &G, node_label: impl FnMut(G::NodeId) -> String) -> String {
    let mut out = String::new();
    write_view_dot(graph, &mut out, node_label).expect("writing DOT to a String cannot fail");
    out
}

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
        w: &mut dyn fmt::Write,
        mut label: impl FnMut(&I) -> Cow<'_, str>,
    ) -> fmt::Result {
        writeln!(w, "digraph cfg {{")?;
        writeln!(
            w,
            "    node [shape=box fontname=\"monospace\" fontsize=10];"
        )?;
        writeln!(w, "    edge [fontname=\"monospace\" fontsize=9];")?;

        for id in self.block_ids() {
            let block = self.block(id);
            let label_prefix = block
                .label()
                .map(|l| alloc::format!("{}:\\n", escape_label(l)))
                .unwrap_or_default();

            // Build a label with the mnemonic of each instruction.
            let mut body = String::new();
            for inst in block.instructions() {
                let m = label(inst);
                if !m.is_empty() {
                    if !body.is_empty() {
                        body.push_str("\\l");
                    }
                    body.push_str(&escape_label(&m));
                }
            }

            if body.is_empty() {
                body.push_str("(empty)");
            }

            body.push_str("\\l");

            writeln!(w, "    {id} [label=\"{label_prefix}{id}\\n{body}\"];")?;
        }

        for edge in self.edges() {
            let (color, style, lbl) = match edge.kind() {
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
            write!(
                w,
                "    {} -> {} [color={color} style={style}",
                edge.source(),
                edge.target(),
            )?;

            // Show weight in the label if present.
            let weight_str = edge
                .weight()
                .map(|w| alloc::format!(" ({w:.2})"))
                .unwrap_or_default();
            let full_label = alloc::format!("{lbl}{weight_str}");
            if !full_label.is_empty() {
                write!(w, " label=\"{full_label}\"")?;
            }

            // Thicker line for high-probability edges.
            if let Some(wt) = edge.weight() {
                let penwidth = 1.0 + wt * 3.0; // 1.0–4.0
                write!(w, " penwidth={penwidth:.1}")?;
            }

            writeln!(w, "];")?;
        }

        writeln!(w, "}}")
    }

    /// Produce the DOT representation as a [`String`] using a caller-supplied
    /// instruction label.
    ///
    /// # Panics
    ///
    /// Panics only if writing to an in-memory [`String`] unexpectedly fails.
    #[must_use]
    pub fn to_dot_with(&self, label: impl FnMut(&I) -> Cow<'_, str>) -> String {
        let mut s = String::new();
        self.write_dot_with(&mut s, label)
            .expect("fmt::Write to String cannot fail");
        s
    }
}

impl<I: DisplayInstr, E> Cfg<I, E> {
    /// Write the CFG in DOT format to any `fmt::Write` sink.
    ///
    /// # Errors
    ///
    /// Returns the sink's formatting error if a write fails.
    pub fn write_dot(&self, w: &mut dyn fmt::Write) -> fmt::Result {
        self.write_dot_with(w, DisplayInstr::mnemonic)
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
        let mut s = String::new();
        self.write_dot(&mut s)
            .expect("fmt::Write to String cannot fail");
        s
    }
}

#[cfg(test)]
mod tests {
    use super::{String, write_view_dot};
    use crate::cfg::Cfg;
    use crate::edge::EdgeKind;
    use crate::graph::store::Graph;
    use crate::test_util::{MockInst, ff};

    #[test]
    fn view_dot_renders_nodes_edges_and_escapes_labels() {
        let mut graph = Graph::new();
        let a = graph.add_node("say \"hi\"\nback\\slash");
        let b = graph.add_node("plain");
        graph.add_edge(a, b, ());

        let mut out = String::new();
        write_view_dot(&graph, &mut out, |node| String::from(*graph.node(node))).unwrap();
        assert!(out.starts_with("digraph view {"));
        assert!(out.contains("n0 -> n1;"));
        assert!(out.contains("say \\\"hi\\\"\\nback\\\\slash"));
        assert!(out.contains("[label=\"plain\"]"));
    }

    #[test]
    fn to_dot_contains_digraph_wrapper() {
        let mut cfg = Cfg::new();
        cfg.block_mut(cfg.entry())
            .instructions_mut()
            .push(ff("nop"));
        let dot = cfg.to_dot();
        assert!(dot.starts_with("digraph cfg {"));
        assert!(dot.trim_end().ends_with('}'));
    }

    #[test]
    fn to_dot_contains_block_labels() {
        let mut cfg = Cfg::new();
        let b = cfg.new_block();
        cfg.block_mut(cfg.entry())
            .instructions_mut()
            .push(ff("entry_inst"));
        cfg.block_mut(b).instructions_mut().push(ff("second_inst"));
        cfg.add_edge(cfg.entry(), b, EdgeKind::Fallthrough);
        let dot = cfg.to_dot();
        assert!(dot.contains("entry_inst"), "should contain mnemonic");
        assert!(dot.contains("second_inst"), "should contain mnemonic");
    }

    #[test]
    fn to_dot_conditional_edge_colors() {
        let mut cfg = Cfg::new();
        let a = cfg.new_block();
        let b = cfg.new_block();
        cfg.block_mut(cfg.entry()).instructions_mut().push(ff("br"));
        cfg.add_edge(cfg.entry(), a, EdgeKind::ConditionalTrue);
        cfg.add_edge(cfg.entry(), b, EdgeKind::ConditionalFalse);
        let dot = cfg.to_dot();
        assert!(dot.contains("green4"), "true edge should be green");
        assert!(dot.contains("red"), "false edge should be red");
        assert!(dot.contains("\"T\""), "true edge label");
        assert!(dot.contains("\"F\""), "false edge label");
    }

    #[test]
    fn to_dot_empty_block_shows_empty_label() {
        let cfg: Cfg<MockInst> = Cfg::new();
        let dot = cfg.to_dot();
        assert!(dot.contains("(empty)"), "empty block should say (empty)");
    }

    #[test]
    fn to_dot_edge_weight_shows_penwidth() {
        let mut cfg = Cfg::new();
        let b = cfg.new_block();
        cfg.block_mut(cfg.entry()).instructions_mut().push(ff("a"));
        let eid = cfg.add_edge(cfg.entry(), b, EdgeKind::Fallthrough);
        cfg.edge_mut(eid).set_weight(Some(0.75));
        let dot = cfg.to_dot();
        assert!(
            dot.contains("penwidth="),
            "weighted edge should have penwidth"
        );
        assert!(dot.contains("0.75"), "weight value should appear in label");
    }
}
