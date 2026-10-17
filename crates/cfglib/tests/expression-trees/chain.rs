//! One block of chained temporaries, each read once, recovered to one deep tree.

use std::fmt::{self, Write};
use std::thread;

use cfglib::{BlockExprTrees, Cfg, ExprNode, recover_block_expressions};

use crate::instruction::{Inst, Operator, Tree};

/// The chain's length: far past what a recursive walk survives on a small stack.
const LENGTH: u32 = 100_000;

/// The variable the first temporary reads. No instruction defines it, nor `OTHER_INPUT`.
const INPUT: u32 = 0;
const OTHER_INPUT: u32 = LENGTH + 1;

/// The smallest stack a consumer thread is likely to have.
const STACK_BYTES: usize = 256 * 1024;

/// Runs `body` on a thread whose stack a recursion `LENGTH` deep cannot fit in.
fn on_small_stack<T: Send + 'static>(body: impl FnOnce() -> T + Send + 'static) -> T {
    thread::Builder::new()
        .stack_size(STACK_BYTES)
        .spawn(body)
        .expect("the thread starts")
        .join()
        .expect("the body completes without overflowing the stack")
}

/// Recovers `t1 = negate(input); t2 = negate(t1); ... t{length} = negate(t{length - 1})`
/// through the public entry point.
fn recovered_chain(length: u32, input: u32) -> BlockExprTrees<u32, Operator, i64> {
    let mut cfg = Cfg::<Inst>::new();
    let entry = cfg.entry();
    let temporaries = (1..=length).map(|temporary| {
        let operand = if temporary == 1 { input } else { temporary - 1 };
        Inst::negate(temporary, operand)
    });
    cfg.block_mut(entry).instructions_mut().extend(temporaries);

    recover_block_expressions(&cfg, entry)
}

/// The chain's one root: `length` operators over `input`.
fn chain(length: u32, input: u32) -> Tree {
    let mut trees = recovered_chain(length, input);

    assert_eq!(trees.roots.len(), 1, "only the last temporary is not read");
    let (root, tree) = trees.roots.pop().unwrap();
    assert_eq!(root, length);
    tree
}

/// The chain of `LENGTH` operators over `input`.
fn chain_over(input: u32) -> Tree {
    chain(LENGTH, input)
}

/// The number of operators above the leaf, counted without recursion.
fn spine_length(tree: &Tree) -> usize {
    let mut length = 0;
    let mut node = tree;
    while let ExprNode::Op { operands, .. } = node {
        assert_eq!(operands.len(), 1);
        node = &operands[0];
        length += 1;
    }
    assert!(matches!(node, ExprNode::Leaf(_)));
    length
}

/// The depth at which a derived pretty `Debug` overflows a `STACK_BYTES` stack
/// many times over. The text grows with its square, so it is shallower than
/// `LENGTH`.
const PRETTY_LENGTH: u32 = 1_000;

/// Checks the `{:?}` text of a chain as it arrives, so the output of a deep
/// tree is never held whole twice.
struct ExpectedText<'a> {
    expected: &'a str,
    written: usize,
}

impl Write for ExpectedText<'_> {
    fn write_str(&mut self, chunk: &str) -> fmt::Result {
        assert!(
            self.expected[self.written..].starts_with(chunk),
            "the text diverges at byte {}",
            self.written
        );
        self.written += chunk.len();
        Ok(())
    }
}

#[test]
fn a_deep_tree_clones_without_recursion() {
    on_small_stack(|| {
        let tree = chain_over(INPUT);

        let copy = tree.clone();

        assert_eq!(spine_length(&copy), LENGTH as usize);
        assert!(matches!(
            copy,
            ExprNode::Op {
                operator: Operator::Negate,
                ..
            }
        ));
    });
}

#[test]
fn deep_trees_compare_without_recursion() {
    on_small_stack(|| {
        let tree = chain_over(INPUT);

        assert_eq!(tree, chain_over(INPUT));
        assert_ne!(
            tree,
            chain_over(OTHER_INPUT),
            "the trees differ only at the leaf"
        );
    });
}

#[test]
fn a_deep_tree_prints_without_recursion() {
    on_small_stack(|| {
        let tree = chain_over(OTHER_INPUT);
        let expected = format!(
            "{}Leaf({}){}",
            "Op { operator: Negate, operands: [".repeat(LENGTH as usize),
            OTHER_INPUT,
            "] }".repeat(LENGTH as usize),
        );
        let mut sink = ExpectedText {
            expected: &expected,
            written: 0,
        };

        write!(sink, "{tree:?}").unwrap();

        assert_eq!(sink.written, expected.len());
    });
}

#[test]
fn a_deep_tree_pretty_prints_without_recursion() {
    on_small_stack(|| {
        // Each level nests an operator's field and its operand list: two
        // levels of four spaces.
        let indent = |level: u32| " ".repeat(4 * level as usize);
        let mut expected = String::new();
        for level in 0..PRETTY_LENGTH {
            let outer = indent(2 * level);
            let inner = indent(2 * level + 1);
            write!(
                expected,
                "{outer}Op {{\n{inner}operator: Negate,\n{inner}operands: [\n"
            )
            .unwrap();
        }
        let leaf = indent(2 * PRETTY_LENGTH);
        write!(expected, "{leaf}Leaf(\n{leaf}    {OTHER_INPUT},\n{leaf})").unwrap();
        for level in (0..PRETTY_LENGTH).rev() {
            let outer = indent(2 * level);
            write!(expected, ",\n{outer}    ],\n{outer}}}").unwrap();
        }
        let tree = chain(PRETTY_LENGTH, OTHER_INPUT);
        let mut sink = ExpectedText {
            expected: &expected,
            written: 0,
        };

        write!(sink, "{tree:#?}").unwrap();

        assert_eq!(sink.written, expected.len());
    });
}

#[test]
fn a_deep_tree_reports_its_size_without_recursion() {
    on_small_stack(|| {
        let tree = chain_over(INPUT);

        assert_eq!(tree.depth(), LENGTH as usize + 1);
        assert_eq!(tree.node_count(), LENGTH as usize + 1);
    });
}

#[test]
fn a_deep_tree_drops_without_recursion() {
    on_small_stack(|| {
        let trees = recovered_chain(LENGTH, INPUT);
        let tree = chain_over(INPUT);

        drop(trees);
        drop(tree);
    });
}
