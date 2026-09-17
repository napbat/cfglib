//! A line-oriented text form of a graph, and the parser that reads it back.
//!
//! DOT draws a graph; this prints one. Every line is one fact, ids are the
//! dense indices the store already uses, and nothing is nested, so a diff
//! over two of these reads as a diff over the graph. [`parse_text`] turns
//! the text back into a plain [`Graph`] whose payloads are the label text,
//! which is what makes the form testable: a graph written through any view
//! is read back into one comparable value.
//!
//! # Grammar
//!
//! ```text
//! # a comment, and blank lines, are ignored
//! n0 the label of node 0
//! n1
//! n0 -> n1 the label of that edge
//! ```
//!
//! Node lines declare dense indices in ascending order; a gap is a node the
//! store has removed, and the parser reproduces the hole so indices keep
//! their meaning. Edge lines follow the store's insertion order, which is
//! the order the parser replays them in.
//!
//! A label is escaped so it cannot be mistaken for structure: `\\` is a
//! backslash, `\n` a line break, and `\-` a leading hyphen — the one
//! character that could otherwise let a node label read as an arrow.
//!
//! # What it is not
//!
//! This is a projection, not a snapshot: it carries identity, topology, and
//! whatever the label hooks render. Consumer payloads that are not text
//! belong in the `serde` representation of the store, which is the fidelity
//! path.

extern crate alloc;
use alloc::borrow::Cow;
use alloc::string::String;
use alloc::vec::Vec;
use core::fmt::{self, Write as _};

use crate::graph::edge_view::EdgeView;
use crate::graph::label::{Label, display_label, no_label};
use crate::graph::store::{Graph, Id, IdTag};
use crate::graph::view::{DenseId, NodeView};

mod parse;

pub use parse::parse_text;

/// Why a text graph could not be read.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TextError {
    /// The one-based line the failure was found on.
    pub line: usize,
    /// What was wrong with it.
    pub kind: TextErrorKind,
}

impl TextError {
    pub(crate) const fn new(line: usize, kind: TextErrorKind) -> Self {
        Self { line, kind }
    }
}

/// What was wrong with one line of a text graph.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum TextErrorKind {
    /// The line matched no form of the grammar.
    Unrecognized,
    /// An identifier was not the expected prefix followed by a dense index.
    Identifier,
    /// Node lines must declare dense indices in ascending order.
    NodeOrder {
        /// The lowest index the line could have declared.
        expected: usize,
    },
    /// A line named a node that no earlier line declared.
    UnknownNode {
        /// The dense index that was named.
        index: usize,
    },
    /// An edge named a control-flow kind that has no name in the vocabulary.
    EdgeKind,
    /// An escape sequence was unknown or unterminated.
    Escape,
    /// An exception-region line was malformed.
    Region,
    /// The caller's instruction parser rejected the line.
    Instruction(String),
}

impl fmt::Display for TextError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "line {}: ", self.line)?;
        match &self.kind {
            TextErrorKind::Unrecognized => formatter.write_str("unrecognized line"),
            TextErrorKind::Identifier => formatter.write_str("malformed identifier"),
            TextErrorKind::NodeOrder { expected } => {
                write!(formatter, "node index must be at least {expected}")
            }
            TextErrorKind::UnknownNode { index } => {
                write!(formatter, "node {index} has not been declared")
            }
            TextErrorKind::EdgeKind => formatter.write_str("unknown edge kind"),
            TextErrorKind::Escape => formatter.write_str("malformed escape sequence"),
            TextErrorKind::Region => formatter.write_str("malformed region"),
            TextErrorKind::Instruction(message) => {
                write!(formatter, "instruction: {message}")
            }
        }
    }
}

impl core::error::Error for TextError {}

/// How one graph is written as text: what labels a node, what labels an edge.
///
/// The store has no distinguished entry, so its text form has no `entry`
/// header; [`Cfg::to_text`](crate::Cfg::to_text) writes one because a CFG
/// does have one.
#[derive(Debug, Clone, Copy)]
pub struct TextStyle<NodeLabel, EdgeLabel> {
    node_label: NodeLabel,
    edge_label: EdgeLabel,
}

impl<NodeLabel, EdgeLabel> TextStyle<NodeLabel, EdgeLabel> {
    /// A style over the two label hooks.
    pub const fn new(node_label: NodeLabel, edge_label: EdgeLabel) -> Self {
        Self {
            node_label,
            edge_label,
        }
    }
}

impl<Node, NodeData: ?Sized, Edge, EdgeData: ?Sized>
    TextStyle<Label<Node, NodeData>, Label<Edge, EdgeData>>
{
    /// Topology only: neither nodes nor edges carry a label.
    #[must_use]
    pub fn plain() -> Self {
        Self::new(
            no_label as Label<Node, NodeData>,
            no_label as Label<Edge, EdgeData>,
        )
    }
}

/// Write any node- and edge-bearing view in the line-oriented text form.
///
/// # Errors
///
/// Returns the sink's formatting error if a write fails.
pub fn write_text<G, NodeLabel, EdgeLabel>(
    view: &G,
    sink: &mut dyn fmt::Write,
    style: &TextStyle<NodeLabel, EdgeLabel>,
) -> fmt::Result
where
    G: NodeView + EdgeView,
    NodeLabel: for<'d> Fn(G::NodeId, &'d G::NodeData) -> Cow<'d, str>,
    EdgeLabel: for<'d> Fn(G::EdgeId, &'d G::EdgeData) -> Cow<'d, str>,
{
    for node in view.node_ids() {
        write!(sink, "n{}", node.index())?;
        write_label(sink, &(style.node_label)(node, view.node(node)))?;
        sink.write_char('\n')?;
    }
    for id in view.edge_ids() {
        let edge = view.edge(id);
        write!(
            sink,
            "n{} -> n{}",
            edge.source().index(),
            edge.target().index()
        )?;
        write_label(sink, &(style.edge_label)(id, edge.data()))?;
        sink.write_char('\n')?;
    }
    Ok(())
}

/// Render any node- and edge-bearing view in the line-oriented text form.
///
/// The allocating counterpart of [`write_text`].
///
/// # Panics
///
/// Panics only if writing to an in-memory [`String`] unexpectedly fails.
#[must_use]
pub fn to_text<G, NodeLabel, EdgeLabel>(view: &G, style: &TextStyle<NodeLabel, EdgeLabel>) -> String
where
    G: NodeView + EdgeView,
    NodeLabel: for<'d> Fn(G::NodeId, &'d G::NodeData) -> Cow<'d, str>,
    EdgeLabel: for<'d> Fn(G::EdgeId, &'d G::EdgeData) -> Cow<'d, str>,
{
    let mut out = String::new();
    write_text(view, &mut out, style).expect("writing text to a String cannot fail");
    out
}

/// Write a label after its subject, escaped, or nothing when it is empty.
pub(crate) fn write_label(sink: &mut dyn fmt::Write, label: &str) -> fmt::Result {
    if label.is_empty() {
        return Ok(());
    }
    sink.write_char(' ')?;
    write_escaped(sink, label)
}

/// Escape one label so no line of the grammar can be misread.
pub(crate) fn write_escaped(sink: &mut dyn fmt::Write, label: &str) -> fmt::Result {
    EscapingSink::new(sink).write_str(label)
}

/// A sink that escapes everything written through it.
///
/// This is what lets an instruction render itself straight into a text
/// graph: the escape rule sees the characters as they are produced, so no
/// intermediate [`String`] stands between the instruction and the file.
pub(crate) struct EscapingSink<'sink> {
    sink: &'sink mut dyn fmt::Write,
    at_start: bool,
}

impl<'sink> EscapingSink<'sink> {
    pub(crate) fn new(sink: &'sink mut dyn fmt::Write) -> Self {
        Self {
            sink,
            at_start: true,
        }
    }
}

impl fmt::Write for EscapingSink<'_> {
    fn write_str(&mut self, text: &str) -> fmt::Result {
        for ch in text.chars() {
            match ch {
                '\\' => self.sink.write_str("\\\\")?,
                '\n' => self.sink.write_str("\\n")?,
                '\r' => self.sink.write_str("\\r")?,
                '-' if self.at_start => self.sink.write_str("\\-")?,
                _ => self.sink.write_char(ch)?,
            }
            self.at_start = false;
        }
        Ok(())
    }
}

/// Reverse [`write_escaped`].
pub(crate) fn unescape(label: &str, line: usize) -> Result<String, TextError> {
    let mut out = String::with_capacity(label.len());
    let mut characters = label.chars();
    while let Some(ch) = characters.next() {
        if ch != '\\' {
            out.push(ch);
            continue;
        }
        match characters.next() {
            Some('\\') => out.push('\\'),
            Some('n') => out.push('\n'),
            Some('r') => out.push('\r'),
            Some('-') => out.push('-'),
            _ => return Err(TextError::new(line, TextErrorKind::Escape)),
        }
    }
    Ok(out)
}

/// The dense index named by `token`, which must be `prefix` then digits.
pub(crate) fn parse_index(token: &str, prefix: &str, line: usize) -> Result<usize, TextError> {
    token
        .strip_prefix(prefix)
        .filter(|rest| !rest.is_empty() && rest.bytes().all(|byte| byte.is_ascii_digit()))
        .and_then(|rest| rest.parse().ok())
        .ok_or_else(|| TextError::new(line, TextErrorKind::Identifier))
}

/// One significant line of a text graph: its one-based number and content.
///
/// Blank lines, comment lines, and a trailing carriage return are the
/// scanner's business, so no reader has to repeat them.
pub(crate) fn significant_lines(text: &str) -> impl Iterator<Item = (usize, &str)> {
    text.lines().enumerate().filter_map(|(index, raw)| {
        let content = raw.strip_suffix('\r').unwrap_or(raw);
        let trimmed = content.trim_start();
        (!trimmed.is_empty() && !trimmed.starts_with('#')).then_some((index + 1, content))
    })
}

/// Split a line into its leading whitespace-free tokens and the rest.
///
/// The rest is the label: everything after the `count`-th token and the one
/// space that separates it, kept verbatim.
pub(crate) fn split_tokens(line: &str, count: usize) -> Option<(Vec<&str>, &str)> {
    let mut tokens = Vec::with_capacity(count);
    let mut rest = line;
    for _ in 0..count {
        rest = rest.trim_start();
        let end = rest.find(' ').unwrap_or(rest.len());
        let (token, remainder) = rest.split_at(end);
        if token.is_empty() {
            return None;
        }
        tokens.push(token);
        rest = remainder;
    }
    Some((tokens, rest.strip_prefix(' ').unwrap_or(rest)))
}

impl<N: fmt::Display, E: fmt::Display, NT: IdTag, ET: IdTag> Graph<N, E, NT, ET> {
    /// Render the store in the line-oriented text form, labeling nodes and
    /// edges with their payloads.
    ///
    /// Nodes are named `n0`, `n1`, … whatever the store's tag, because
    /// [`parse_text`] reads them back into a plain [`Graph`].
    ///
    /// # Panics
    ///
    /// Panics only if writing to an in-memory [`String`] unexpectedly fails.
    #[must_use]
    pub fn to_text(&self) -> String {
        to_text(
            self,
            &TextStyle::new(
                display_label as Label<Id<NT>, N>,
                display_label as Label<Id<ET>, E>,
            ),
        )
    }
}

#[cfg(test)]
mod tests;
