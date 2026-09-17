//! DOT (Graphviz) export for every graph in the crate.
//!
//! One writer serves all of them. [`write_dot`] walks any view that exposes
//! node payloads ([`NodeView`]) and edge identity ([`EdgeView`]) and asks a
//! [`DotStyle`] what to print: how to name the graph, what text labels a
//! node, and which Graphviz attributes an edge carries. [`Cfg`] and
//! [`Graph`] are two styles over that one writer rather than two writers.
//!
//! # Label text
//!
//! A node label is ordinary text with real line breaks. The writer escapes
//! it — backslashes and quotes cannot break out of the attribute — and ends
//! every line with Graphviz's `\l`, which left-justifies it. Program text is
//! therefore safe to hand back from a label hook exactly as it reads.

extern crate alloc;
use alloc::borrow::Cow;
use alloc::format;
use alloc::string::String;
use core::fmt::{self, Write as _};

use crate::block::{BasicBlock, BlockId};
use crate::cfg::Cfg;
use crate::display::DisplayInstr;
use crate::edge::{Edge, EdgeId, EdgeKind};
use crate::graph::edge_view::EdgeView;
use crate::graph::label::{Label, bind_label, display_label, no_label};
use crate::graph::store::{Graph, Id, IdTag};
use crate::graph::view::{DenseId, NodeView};

/// The direction successive ranks of a drawing flow in.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DotRankDir {
    /// Top to bottom — Graphviz's own default, and the usual reading order
    /// for a control-flow graph.
    #[default]
    TopToBottom,
    /// Left to right.
    LeftToRight,
    /// Bottom to top.
    BottomToTop,
    /// Right to left.
    RightToLeft,
}

impl DotRankDir {
    /// The Graphviz `rankdir` attribute value.
    #[must_use]
    pub const fn attribute(self) -> &'static str {
        match self {
            Self::TopToBottom => "TB",
            Self::LeftToRight => "LR",
            Self::BottomToTop => "BT",
            Self::RightToLeft => "RL",
        }
    }
}

/// The Graphviz attributes one edge carries.
///
/// Every field is optional; an edge that sets none of them is drawn with the
/// graph's `edge` defaults and printed without an attribute list at all.
#[derive(Debug, Clone, Default)]
pub struct DotEdgeAttributes<'a> {
    /// Text drawn beside the edge, escaped by the writer.
    pub label: Cow<'a, str>,
    /// Graphviz color name, such as `green4`.
    pub color: Option<&'a str>,
    /// Graphviz line style, such as `dashed`.
    pub style: Option<&'a str>,
    /// Line thickness in points.
    pub penwidth: Option<f64>,
}

impl DotEdgeAttributes<'_> {
    /// Attributes that add nothing to the graph's edge defaults.
    #[must_use]
    pub const fn plain() -> Self {
        Self {
            label: Cow::Borrowed(""),
            color: None,
            style: None,
            penwidth: None,
        }
    }

    /// Whether the edge adds nothing to the graph's edge defaults.
    #[must_use]
    pub fn is_plain(&self) -> bool {
        self.label.is_empty()
            && self.color.is_none()
            && self.style.is_none()
            && self.penwidth.is_none()
    }
}

/// An edge-attribute hook as a plain function pointer.
pub type DotEdgeStyle<Edge, Data> = for<'d> fn(Edge, &'d Data) -> DotEdgeAttributes<'d>;

/// How one graph is rendered: its name, its identifiers, and its hooks.
///
/// The two hooks are the whole policy. `node_label` turns a node and its
/// payload into label text, `edge_attributes` turns an edge and its payload
/// into [`DotEdgeAttributes`]. Both are ordinary closures or function items,
/// so a consumer renders its own payloads without implementing a trait.
#[derive(Debug, Clone, Copy)]
pub struct DotStyle<NodeLabel, EdgeAttributes> {
    graph_name: &'static str,
    node_prefix: &'static str,
    rankdir: DotRankDir,
    node_label: NodeLabel,
    edge_attributes: EdgeAttributes,
}

impl<NodeLabel, EdgeAttributes> DotStyle<NodeLabel, EdgeAttributes> {
    /// A style over the two hooks, named `view`, numbering nodes `n0`, `n1`,
    /// and laid out top to bottom.
    pub const fn new(node_label: NodeLabel, edge_attributes: EdgeAttributes) -> Self {
        Self {
            graph_name: "view",
            node_prefix: "n",
            rankdir: DotRankDir::TopToBottom,
            node_label,
            edge_attributes,
        }
    }

    /// Name the `digraph`. The name must be a DOT identifier.
    #[must_use]
    pub const fn named(mut self, graph_name: &'static str) -> Self {
        self.graph_name = graph_name;
        self
    }

    /// Set the prefix every node identifier is written with.
    #[must_use]
    pub const fn node_prefix(mut self, node_prefix: &'static str) -> Self {
        self.node_prefix = node_prefix;
        self
    }

    /// Set the layout direction.
    #[must_use]
    pub const fn rankdir(mut self, rankdir: DotRankDir) -> Self {
        self.rankdir = rankdir;
        self
    }
}

impl<Node, NodeData: ?Sized, Edge, EdgeData: ?Sized>
    DotStyle<Label<Node, NodeData>, DotEdgeStyle<Edge, EdgeData>>
{
    /// Topology only: unlabeled nodes and undecorated edges.
    #[must_use]
    pub fn plain() -> Self {
        Self::new(
            no_label as Label<Node, NodeData>,
            plain_edge_attributes as DotEdgeStyle<Edge, EdgeData>,
        )
    }
}

/// Bind an edge-attribute closure to the higher-ranked signature
/// [`write_dot`] requires, as
/// [`bind_label`] does for node labels.
#[must_use]
pub fn bind_edge_attributes<Edge, Data, Hook>(hook: Hook) -> Hook
where
    Data: ?Sized,
    Hook: for<'d> Fn(Edge, &'d Data) -> DotEdgeAttributes<'d>,
{
    hook
}

/// An edge-attribute hook that decorates nothing.
#[must_use]
pub fn plain_edge_attributes<Edge, Data: ?Sized>(
    _edge: Edge,
    _data: &Data,
) -> DotEdgeAttributes<'_> {
    DotEdgeAttributes::plain()
}

/// The color, line style, label, and thickness of one control-flow edge.
///
/// This is the hook [`Cfg::write_dot`] installs: every [`EdgeKind`] has a
/// fixed color and line style, and a weighted edge is labeled with its
/// probability and drawn proportionally thicker.
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

/// Write any node- and edge-bearing view in DOT format.
///
/// # Errors
///
/// Returns the sink's formatting error if a write fails.
pub fn write_dot<G, NodeLabel, EdgeAttributes>(
    view: &G,
    sink: &mut dyn fmt::Write,
    style: &DotStyle<NodeLabel, EdgeAttributes>,
) -> fmt::Result
where
    G: NodeView + EdgeView,
    NodeLabel: for<'d> Fn(G::NodeId, &'d G::NodeData) -> Cow<'d, str>,
    EdgeAttributes: for<'d> Fn(G::EdgeId, &'d G::EdgeData) -> DotEdgeAttributes<'d>,
{
    let prefix = style.node_prefix;
    writeln!(sink, "digraph {} {{", style.graph_name)?;
    writeln!(sink, "    rankdir={};", style.rankdir.attribute())?;
    writeln!(
        sink,
        "    node [shape=box fontname=\"monospace\" fontsize=10];"
    )?;
    writeln!(sink, "    edge [fontname=\"monospace\" fontsize=9];")?;

    for node in view.node_ids() {
        write!(sink, "    {prefix}{}", node.index())?;
        let label = (style.node_label)(node, view.node(node));
        if !label.is_empty() {
            sink.write_str(" [label=\"")?;
            write_escaped_lines(sink, &label)?;
            sink.write_str("\"]")?;
        }
        writeln!(sink, ";")?;
    }

    for id in view.edge_ids() {
        let edge = view.edge(id);
        write!(
            sink,
            "    {prefix}{} -> {prefix}{}",
            edge.source().index(),
            edge.target().index()
        )?;
        write_edge_attributes(sink, &(style.edge_attributes)(id, edge.data()))?;
        writeln!(sink, ";")?;
    }

    writeln!(sink, "}}")
}

/// Render any node- and edge-bearing view in DOT format.
///
/// The allocating counterpart of [`write_dot`].
///
/// # Panics
///
/// Panics only if writing to an in-memory [`String`] unexpectedly fails.
#[must_use]
pub fn to_dot<G, NodeLabel, EdgeAttributes>(
    view: &G,
    style: &DotStyle<NodeLabel, EdgeAttributes>,
) -> String
where
    G: NodeView + EdgeView,
    NodeLabel: for<'d> Fn(G::NodeId, &'d G::NodeData) -> Cow<'d, str>,
    EdgeAttributes: for<'d> Fn(G::EdgeId, &'d G::EdgeData) -> DotEdgeAttributes<'d>,
{
    let mut out = String::new();
    write_dot(view, &mut out, style).expect("writing DOT to a String cannot fail");
    out
}

/// Escape label text and terminate every line with Graphviz's `\l`.
fn write_escaped_lines(sink: &mut dyn fmt::Write, label: &str) -> fmt::Result {
    for line in label.split('\n') {
        write_escaped_label(sink, line)?;
        sink.write_str("\\l")?;
    }
    Ok(())
}

fn write_edge_attributes(
    sink: &mut dyn fmt::Write,
    attributes: &DotEdgeAttributes<'_>,
) -> fmt::Result {
    if attributes.is_plain() {
        return Ok(());
    }
    sink.write_str(" [")?;
    let mut separator = "";
    if let Some(color) = attributes.color {
        write!(sink, "{separator}color={color}")?;
        separator = " ";
    }
    if let Some(style) = attributes.style {
        write!(sink, "{separator}style={style}")?;
        separator = " ";
    }
    if !attributes.label.is_empty() {
        write!(sink, "{separator}label=\"")?;
        write_escaped_label(sink, &attributes.label)?;
        sink.write_char('"')?;
        separator = " ";
    }
    if let Some(penwidth) = attributes.penwidth {
        write!(sink, "{separator}penwidth={penwidth:.1}")?;
    }
    sink.write_char(']')
}

/// Escape one line of attribute text, mapping a line break to DOT's `\n`.
fn write_escaped_label(sink: &mut dyn fmt::Write, label: &str) -> fmt::Result {
    for ch in label.chars() {
        match ch {
            '\\' => sink.write_str("\\\\")?,
            '"' => sink.write_str("\\\"")?,
            '\n' => sink.write_str("\\n")?,
            '\r' => {}
            _ => sink.write_char(ch)?,
        }
    }
    Ok(())
}

impl<N: fmt::Display, E, NT: IdTag, ET: IdTag> Graph<N, E, NT, ET> {
    /// Write the store in DOT format, labeling each node with its payload.
    ///
    /// Nodes are named by their tag and dense index — `n0`, `n1`, … for the
    /// default tag. Edges are drawn undecorated; [`write_dot`] with a
    /// [`DotStyle`] of your own renders edge payloads too.
    ///
    /// # Errors
    ///
    /// Returns the sink's formatting error if a write fails.
    pub fn write_dot(&self, sink: &mut dyn fmt::Write) -> fmt::Result {
        write_dot(self, sink, &store_style::<N, E, NT, ET>())
    }

    /// Produce the DOT representation of the store as a [`String`].
    ///
    /// # Panics
    ///
    /// Panics only if writing to an in-memory [`String`] unexpectedly fails.
    ///
    /// # Examples
    ///
    /// ```
    /// use cfglib::Graph;
    ///
    /// let mut graph = Graph::new();
    /// let definition = graph.add_node("definition");
    /// let call = graph.add_node("call");
    /// graph.add_edge(definition, call, ());
    ///
    /// let dot = graph.to_dot();
    /// assert!(dot.contains("n0 [label=\"definition\\l\"];"));
    /// assert!(dot.contains("n0 -> n1;"));
    /// ```
    #[must_use]
    pub fn to_dot(&self) -> String {
        to_dot(self, &store_style::<N, E, NT, ET>())
    }
}

fn store_style<N: fmt::Display, E, NT: IdTag, ET: IdTag>()
-> DotStyle<Label<Id<NT>, N>, DotEdgeStyle<Id<ET>, E>> {
    DotStyle::new(
        display_label as Label<Id<NT>, N>,
        plain_edge_attributes as DotEdgeStyle<Id<ET>, E>,
    )
    .node_prefix(NT::PREFIX)
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
        sink: &mut dyn fmt::Write,
        label: impl Fn(&I) -> Cow<'_, str>,
    ) -> fmt::Result {
        let style = DotStyle::new(
            block_label(label),
            control_flow_edge_attributes as DotEdgeStyle<EdgeId, Edge<E>>,
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

/// The node label of a control-flow graph: the block's own name, its
/// identity, and one line per instruction.
fn block_label<I>(
    label: impl Fn(&I) -> Cow<'_, str>,
) -> impl for<'b> Fn(BlockId, &'b BasicBlock<I>) -> Cow<'b, str> {
    bind_label::<BlockId, BasicBlock<I>, _>(move |id, block| {
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
        Cow::Owned(text)
    })
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

#[cfg(test)]
mod tests;
