//! Reading the line-oriented text form back into a plain store.

extern crate alloc;
use alloc::string::String;
use alloc::vec::Vec;

use crate::graph::store::{Graph, NodeId};

use super::{TextError, TextErrorKind, parse_index, significant_lines, split_tokens, unescape};

/// Read a text graph into a store whose payloads are the label text.
///
/// Node indices are reproduced exactly, holes included, so the store round
/// trips: writing the result again yields the input.
///
/// # Errors
///
/// Returns the first malformed line, with its number and what was wrong.
pub fn parse_text(text: &str) -> Result<Graph<String, String>, TextError> {
    let mut graph = Graph::new();
    let mut declared: Vec<bool> = Vec::new();

    for (line, content) in significant_lines(text) {
        let (tokens, tail) = split_tokens(content, 1)
            .ok_or_else(|| TextError::new(line, TextErrorKind::Unrecognized))?;
        let index = parse_index(tokens[0], "n", line)?;

        let Some(arrow_tail) = tail.strip_prefix("-> ") else {
            declare_node(
                &mut graph,
                &mut declared,
                index,
                unescape(tail, line)?,
                line,
            )?;
            continue;
        };

        let (target_tokens, label) = split_tokens(arrow_tail, 1)
            .ok_or_else(|| TextError::new(line, TextErrorKind::Unrecognized))?;
        let target = parse_index(target_tokens[0], "n", line)?;
        let source = live_node(&declared, index, line)?;
        let target = live_node(&declared, target, line)?;
        graph.add_edge(source, target, unescape(label, line)?);
    }

    for (index, declared) in declared.iter().enumerate() {
        if !declared {
            graph.remove_node(NodeId::from_index(index));
        }
    }
    Ok(graph)
}

/// Add the node at `index`, first filling any gap with removed placeholders.
fn declare_node(
    graph: &mut Graph<String, String>,
    declared: &mut Vec<bool>,
    index: usize,
    label: String,
    line: usize,
) -> Result<(), TextError> {
    if index < declared.len() {
        return Err(TextError::new(
            line,
            TextErrorKind::NodeOrder {
                expected: declared.len(),
            },
        ));
    }
    while declared.len() < index {
        graph.add_node(String::new());
        declared.push(false);
    }
    graph.add_node(label);
    declared.push(true);
    Ok(())
}

fn live_node(declared: &[bool], index: usize, line: usize) -> Result<NodeId, TextError> {
    if declared.get(index).copied() == Some(true) {
        return Ok(NodeId::from_index(index));
    }
    Err(TextError::new(line, TextErrorKind::UnknownNode { index }))
}
