//! Reading the line-oriented text form back into a plain store.

extern crate alloc;
use alloc::string::String;

use crate::graph::store::{Graph, NodeId};

use super::{
    Slots, TextError, TextErrorKind, parse_index, significant_lines, split_tokens, unescape,
};

/// Read a text graph into a store whose payloads are the label text.
///
/// Node indices are reproduced exactly, holes included, so the store round
/// trips: writing the result again yields the input. At most 2²⁰ holes are
/// recreated; a document that leaves more is rejected rather than allocated.
///
/// # Errors
///
/// Returns the first malformed line, with its number and what was wrong.
pub fn parse_text(text: &str) -> Result<Graph<String, String>, TextError> {
    let mut graph = Graph::new();
    let mut slots = Slots::new();

    for (line, content) in significant_lines(text) {
        let (tokens, tail) = split_tokens(content, 1)
            .ok_or_else(|| TextError::new(line, TextErrorKind::Unrecognized))?;
        let index = parse_index(tokens[0], "n", line)?;

        let Some(arrow_tail) = tail.strip_prefix("-> ") else {
            let label = unescape(tail, line)?;
            let before = slots.declare(index, line)?;
            for _ in before..index {
                graph.add_node(String::new());
            }
            graph.add_node(label);
            continue;
        };

        let (target_tokens, label) = split_tokens(arrow_tail, 1)
            .ok_or_else(|| TextError::new(line, TextErrorKind::Unrecognized))?;
        let target = parse_index(target_tokens[0], "n", line)?;
        let source = live_node(&slots, index, line)?;
        let target = live_node(&slots, target, line)?;
        graph.add_edge(source, target, unescape(label, line)?);
    }

    for index in slots.holes() {
        graph.remove_node(NodeId::from_index(index));
    }
    Ok(graph)
}

fn live_node(slots: &Slots, index: usize, line: usize) -> Result<NodeId, TextError> {
    slots.check_live(index, line)?;
    Ok(NodeId::from_index(index))
}
