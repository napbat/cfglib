//! DOT (Graphviz) export for every graph in the crate.
//!
//! One writer serves all of them. [`write_dot`] walks any view that exposes
//! node payloads ([`NodeView`]) and edge identity ([`EdgeView`]) and asks a
//! [`DotStyle`] what to print: how to name the graph, what text labels a
//! node, and which Graphviz attributes an edge carries. [`Cfg`](crate::Cfg)
//! and [`Graph`] are two styles over that one writer rather than two writers.
//!
//! # Label text
//!
//! A node label is ordinary text with real line breaks. The writer escapes
//! it — backslashes and quotes cannot break out of the attribute — and ends
//! every line with Graphviz's `\l`, which left-justifies it. Program text is
//! therefore safe to hand back from a label hook exactly as it reads.
//!
//! # Identifiers and attribute values
//!
//! Every other piece of text the writer is handed (the graph's name, a node
//! prefix, an edge's color and line style) is a DOT *ID*, and the writer
//! spells it the way DOT reads it: bare when it is an identifier or a numeral,
//! and as a quoted string otherwise (`#ff0000`, `/blues9/3`, `dashed,bold`, a
//! word DOT reserves, the empty string). A number DOT has no numeral for, such
//! as a NaN pen width, is not written at all.

extern crate alloc;
use alloc::borrow::Cow;
use alloc::format;
use alloc::string::String;
use core::fmt;

use crate::graph::edge_view::EdgeView;
use crate::graph::label::{Label, display_label, no_label};
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
    /// Graphviz color, such as `green4` or `#ff0000`; the writer quotes it
    /// when DOT requires.
    pub color: Option<&'a str>,
    /// Graphviz line style, such as `dashed` or `dashed,bold`; the writer
    /// quotes it when DOT requires.
    pub style: Option<&'a str>,
    /// Line thickness in points. A NaN or infinite value has no DOT spelling,
    /// so such an edge is drawn at the graph's default width.
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
            && self.drawn_penwidth().is_none()
    }

    /// The pen width the writer draws: the one DOT can spell.
    fn drawn_penwidth(&self) -> Option<f64> {
        self.penwidth.filter(|width| width.is_finite())
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
    ///
    /// The bounds are the ones [`write_dot`] places on the hooks, stated here
    /// so a closure passed straight in is given the higher-ranked signature:
    /// the label text and attributes it returns may borrow the payload it was
    /// handed.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::borrow::Cow;
    /// use cfglib::{DotEdgeAttributes, DotStyle, Graph, display_label, to_dot};
    ///
    /// let mut graph = Graph::<&str, String>::new();
    /// let caller = graph.add_node("main");
    /// let callee = graph.add_node("load");
    /// graph.add_edge(caller, callee, "via vtable".to_owned());
    ///
    /// let style = DotStyle::new(display_label, |_, relation: &String| DotEdgeAttributes {
    ///     label: Cow::Borrowed(relation),
    ///     ..DotEdgeAttributes::plain()
    /// });
    /// assert!(to_dot(&graph, &style).contains("n0 -> n1 [label=\"via vtable\"];"));
    /// ```
    pub fn new<Node, NodeData, Edge, EdgeData>(
        node_label: NodeLabel,
        edge_attributes: EdgeAttributes,
    ) -> Self
    where
        NodeData: ?Sized,
        EdgeData: ?Sized,
        NodeLabel: for<'d> Fn(Node, &'d NodeData) -> Cow<'d, str>,
        EdgeAttributes: for<'d> Fn(Edge, &'d EdgeData) -> DotEdgeAttributes<'d>,
    {
        Self {
            graph_name: "view",
            node_prefix: "n",
            rankdir: DotRankDir::TopToBottom,
            node_label,
            edge_attributes,
        }
    }

    /// Name the `digraph`. The writer quotes the name when DOT requires.
    #[must_use]
    pub const fn named(mut self, graph_name: &'static str) -> Self {
        self.graph_name = graph_name;
        self
    }

    /// Set the prefix every node identifier is written with. The writer
    /// quotes the identifiers when DOT requires, so the prefix may be any text.
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

/// An edge-attribute hook that decorates nothing.
#[must_use]
pub fn plain_edge_attributes<Edge, Data: ?Sized>(
    _edge: Edge,
    _data: &Data,
) -> DotEdgeAttributes<'_> {
    DotEdgeAttributes::plain()
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
    // A prefix followed by digits is a bare ID for every index or for none.
    let quote_nodes = !is_bare_id(&format!("{prefix}0"));
    sink.write_str("digraph ")?;
    write_id(sink, style.graph_name)?;
    writeln!(sink, " {{")?;
    writeln!(sink, "    rankdir={};", style.rankdir.attribute())?;
    writeln!(
        sink,
        "    node [shape=box fontname=\"monospace\" fontsize=10];"
    )?;
    writeln!(sink, "    edge [fontname=\"monospace\" fontsize=9];")?;

    for node in view.node_ids() {
        sink.write_str("    ")?;
        write_node_id(sink, prefix, quote_nodes, node.index())?;
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
        sink.write_str("    ")?;
        write_node_id(sink, prefix, quote_nodes, edge.source().index())?;
        sink.write_str(" -> ")?;
        write_node_id(sink, prefix, quote_nodes, edge.target().index())?;
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
    // A direct call on a local `String`, not `render(|sink| write_dot(..))`:
    // the closure kept `write_dot` out of line behind a virtual sink, and
    // `api_to_dot` took 1.19x its time on main; this form takes 0.99x.
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
        write!(sink, "{separator}color=")?;
        write_id(sink, color)?;
        separator = " ";
    }
    if let Some(style) = attributes.style {
        write!(sink, "{separator}style=")?;
        write_id(sink, style)?;
        separator = " ";
    }
    if !attributes.label.is_empty() {
        write!(sink, "{separator}label=\"")?;
        write_escaped_label(sink, &attributes.label)?;
        sink.write_char('"')?;
        separator = " ";
    }
    if let Some(penwidth) = attributes.drawn_penwidth() {
        write!(sink, "{separator}penwidth={penwidth:.1}")?;
    }
    sink.write_char(']')
}

/// Write a node identifier: the prefix and the index, quoted when `quoted`.
fn write_node_id(
    sink: &mut dyn fmt::Write,
    prefix: &str,
    quoted: bool,
    index: usize,
) -> fmt::Result {
    if !quoted {
        return write!(sink, "{prefix}{index}");
    }
    sink.write_char('"')?;
    write_escaped_label(sink, prefix)?;
    write!(sink, "{index}\"")
}

/// Write `text` as a DOT ID: bare when DOT reads it as an identifier or a
/// numeral, a quoted string otherwise.
fn write_id(sink: &mut dyn fmt::Write, text: &str) -> fmt::Result {
    if is_bare_id(text) {
        return sink.write_str(text);
    }
    sink.write_char('"')?;
    write_escaped_label(sink, text)?;
    sink.write_char('"')
}

/// Whether DOT reads `text` as one ID without quotes: an identifier that is
/// not a reserved word, or a numeral.
fn is_bare_id(text: &str) -> bool {
    const RESERVED: [&str; 6] = ["node", "edge", "graph", "digraph", "subgraph", "strict"];
    let is_word_character = |ch: char| ch.is_ascii_alphanumeric() || ch == '_';
    let mut characters = text.chars();
    let identifier = characters
        .next()
        .is_some_and(|first| first.is_ascii_alphabetic() || first == '_')
        && characters.all(is_word_character);
    if identifier {
        return !RESERVED.iter().any(|word| word.eq_ignore_ascii_case(text));
    }
    // A numeral is `-?(.[0-9]+ | [0-9]+(.[0-9]*)?)`.
    let unsigned = text.strip_prefix('-').unwrap_or(text);
    let (whole, fraction) = match unsigned.split_once('.') {
        Some((whole, fraction)) => (whole, Some(fraction)),
        None => (unsigned, None),
    };
    let digits = |part: &str| part.bytes().all(|byte| byte.is_ascii_digit());
    match fraction {
        None => !whole.is_empty() && digits(whole),
        Some(fraction) => {
            digits(whole) && digits(fraction) && !(whole.is_empty() && fraction.is_empty())
        }
    }
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

#[cfg(test)]
mod tests;
