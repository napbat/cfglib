//! A small DOT scanner for the subset the writer emits, written without
//! looking at the writer: it recovers the graph's name, identifiers, edges and
//! attributes from the text so the tests compare them with the graph the text
//! was made from.
//!
//! The scanner is deliberately strict. A quote that was not escaped ends the
//! string early and the statements after it no longer parse, and an escape
//! other than `\\`, `\"`, `\n` and `\l` is rejected, so a label that leaks
//! Graphviz syntax fails here instead of being drawn wrongly.

use std::collections::BTreeMap;

/// One attribute value: a bare word, or a quoted string kept raw so its
/// escapes are decoded by whoever knows what the string means.
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    Bare(String),
    Quoted(String),
}

pub type Attributes = BTreeMap<String, Value>;

#[derive(Debug)]
pub struct Node {
    pub id: String,
    pub attributes: Attributes,
}

#[derive(Debug)]
pub struct Edge {
    pub source: String,
    pub target: String,
    pub attributes: Attributes,
}

#[derive(Debug)]
pub struct Dot {
    pub name: String,
    pub graph_attributes: BTreeMap<String, String>,
    pub nodes: Vec<Node>,
    pub edges: Vec<Edge>,
}

impl Dot {
    pub fn node_ids(&self) -> Vec<&str> {
        self.nodes.iter().map(|node| node.id.as_str()).collect()
    }

    pub fn edge_endpoints(&self) -> Vec<(&str, &str)> {
        self.edges
            .iter()
            .map(|edge| (edge.source.as_str(), edge.target.as_str()))
            .collect()
    }
}

impl Node {
    /// The label as the lines Graphviz draws, or `None` for an unlabeled
    /// node. Every line must end with `\l`.
    pub fn label_lines(&self) -> Option<Vec<String>> {
        let Value::Quoted(raw) = self.attributes.get("label")? else {
            panic!("a label is a quoted string");
        };
        let mut lines = Vec::new();
        let mut current = String::new();
        for piece in pieces(raw) {
            match piece {
                Piece::Character(ch) => current.push(ch),
                Piece::LeftJustifiedLineEnd => lines.push(std::mem::take(&mut current)),
            }
        }
        assert!(
            current.is_empty(),
            "label {raw:?} ends in an unterminated line"
        );
        Some(lines)
    }
}

impl Edge {
    /// The label with `\n` read as a line break, or `None` when absent.
    pub fn label(&self) -> Option<String> {
        let Value::Quoted(raw) = self.attributes.get("label")? else {
            panic!("a label is a quoted string");
        };
        Some(
            pieces(raw)
                .into_iter()
                .map(|piece| match piece {
                    Piece::Character(ch) => ch,
                    Piece::LeftJustifiedLineEnd => {
                        panic!("an edge label has no `\\l` line end: {raw:?}")
                    }
                })
                .collect(),
        )
    }

    pub fn bare(&self, key: &str) -> Option<&str> {
        match self.attributes.get(key)? {
            Value::Bare(word) => Some(word),
            Value::Quoted(raw) => panic!("{key} is written bare, not quoted as {raw:?}"),
        }
    }
}

enum Piece {
    Character(char),
    LeftJustifiedLineEnd,
}

fn pieces(raw: &str) -> Vec<Piece> {
    let mut decoded = Vec::new();
    let mut characters = raw.chars();
    while let Some(ch) = characters.next() {
        decoded.push(if ch == '\\' {
            match characters.next() {
                Some('\\') => Piece::Character('\\'),
                Some('"') => Piece::Character('"'),
                Some('n') => Piece::Character('\n'),
                Some('l') => Piece::LeftJustifiedLineEnd,
                other => panic!("escape {other:?} in {raw:?} is not one the writer may emit"),
            }
        } else {
            Piece::Character(ch)
        });
    }
    decoded
}

#[derive(Debug, Clone, PartialEq)]
enum Token {
    Word(String),
    Quoted(String),
    Arrow,
    Open,
    Close,
    OpenBracket,
    CloseBracket,
    Equals,
    Semicolon,
}

fn tokenize(text: &str) -> Vec<Token> {
    let mut tokens = Vec::new();
    let mut characters = text.chars().peekable();
    while let Some(ch) = characters.next() {
        match ch {
            ch if ch.is_whitespace() => {}
            '{' => tokens.push(Token::Open),
            '}' => tokens.push(Token::Close),
            '[' => tokens.push(Token::OpenBracket),
            ']' => tokens.push(Token::CloseBracket),
            '=' => tokens.push(Token::Equals),
            ';' => tokens.push(Token::Semicolon),
            '-' => {
                assert_eq!(
                    characters.next(),
                    Some('>'),
                    "a hyphen outside a string must start `->`"
                );
                tokens.push(Token::Arrow);
            }
            '"' => {
                let mut raw = String::new();
                loop {
                    match characters.next().expect("a string was never closed") {
                        '\\' => {
                            raw.push('\\');
                            raw.push(characters.next().expect("an escape ends the text"));
                        }
                        '"' => break,
                        other => raw.push(other),
                    }
                }
                tokens.push(Token::Quoted(raw));
            }
            ch if ch.is_ascii_alphanumeric() || ch == '_' || ch == '.' => {
                let mut word = String::from(ch);
                while let Some(&next) = characters.peek() {
                    if next.is_ascii_alphanumeric() || next == '_' || next == '.' {
                        word.push(next);
                        characters.next();
                    } else {
                        break;
                    }
                }
                tokens.push(Token::Word(word));
            }
            other => panic!("unexpected character {other:?} outside a string"),
        }
    }
    tokens
}

fn attributes(tokens: &[Token]) -> Attributes {
    let mut attributes = Attributes::new();
    for assignment in tokens.chunks(3) {
        let [Token::Word(key), Token::Equals, value] = assignment else {
            panic!("an attribute is `key=value`, got {assignment:?}");
        };
        let value = match value {
            Token::Word(word) => Value::Bare(word.clone()),
            Token::Quoted(raw) => Value::Quoted(raw.clone()),
            other => panic!("{other:?} is not an attribute value"),
        };
        assert!(
            attributes.insert(key.clone(), value).is_none(),
            "{key} is set twice"
        );
    }
    attributes
}

/// The attribute list that follows a statement's head, `[…]`, or nothing.
fn bracketed(tail: &[Token]) -> Attributes {
    match tail {
        [] => Attributes::new(),
        [Token::OpenBracket, inner @ .., Token::CloseBracket] => attributes(inner),
        other => panic!("expected an attribute list, got {other:?}"),
    }
}

pub fn scan(text: &str) -> Dot {
    let tokens = tokenize(text);
    let [
        Token::Word(keyword),
        Token::Word(name),
        Token::Open,
        body @ ..,
        Token::Close,
    ] = tokens.as_slice()
    else {
        panic!("not a `digraph name {{ … }}`:\n{text}");
    };
    assert_eq!(keyword, "digraph");

    let mut dot = Dot {
        name: name.clone(),
        graph_attributes: BTreeMap::new(),
        nodes: Vec::new(),
        edges: Vec::new(),
    };
    let statements: Vec<&[Token]> = body.split(|token| *token == Token::Semicolon).collect();
    let (rest, statements) = statements
        .split_last()
        .expect("split yields at least one piece");
    assert!(rest.is_empty(), "the last statement must end in `;`");
    for statement in statements {
        match statement {
            [Token::Word(key), Token::Equals, Token::Word(value)] => {
                dot.graph_attributes.insert(key.clone(), value.clone());
            }
            [Token::Word(scope), tail @ ..] if scope == "node" || scope == "edge" => {
                // Defaults for every node or edge: a style, not a fact about the graph.
                bracketed(tail);
            }
            [
                Token::Word(source),
                Token::Arrow,
                Token::Word(target),
                tail @ ..,
            ] => {
                dot.edges.push(Edge {
                    source: source.clone(),
                    target: target.clone(),
                    attributes: bracketed(tail),
                });
            }
            [Token::Word(id), tail @ ..] => dot.nodes.push(Node {
                id: id.clone(),
                attributes: bracketed(tail),
            }),
            other => panic!("unrecognized statement {other:?}"),
        }
    }
    dot
}
