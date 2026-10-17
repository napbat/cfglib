//! The expression tree node and its stack-safe operations.
//!
//! Recovery inlines every single-use temporary, so a block of `N` chained
//! instructions recovers to one tree of depth `N`. Every operation here
//! therefore keeps its own stack on the heap instead of recursing once per
//! level: the drop, `Clone`, `PartialEq`, and `Debug` that a derive would
//! generate recurse, and so would a hand-written `depth` or `node_count`.

extern crate alloc;
use alloc::vec::Vec;
use core::fmt::{self, Debug, Write};
use core::slice;

use crate::block::BlockId;

/// A node in an expression tree over variables `V`, operators `Op`, and
/// constants `C` — all consumer-typed.
///
/// Dropping, cloning, comparing, and formatting a tree use no recursion, so
/// a tree as deep as a block is long is safe on any stack. A node
/// implements [`Drop`] to make its own drop iterative, which means its
/// fields cannot be moved out of an owned node by a pattern: match on a
/// reference and clone, or take the operands with
/// [`core::mem::take`] on `&mut` access.
///
/// [`Debug`] prints what a derived implementation would, for `{:?}` and for
/// `{:#?}`. A `{:#?}` print formats `V`, `Op`, and `C` values with the
/// alternate flag only; a derive would also pass on the other flags of the
/// formatter, such as `x?` and the width.
pub enum ExprNode<V, Op, C> {
    /// A leaf source-IR variable.
    Leaf(V),
    /// An operation with a consumer operator identity and operand
    /// sub-expressions.
    Op {
        /// The operator (a mnemonic enum, an opcode, an interned symbol).
        operator: Op,
        /// The operands, each an expression sub-tree.
        operands: Vec<ExprNode<V, Op, C>>,
    },
    /// A constant value.
    Const(C),
    /// An opaque instruction that couldn't be decomposed further.
    /// Contains the instruction index within the block.
    Opaque {
        /// Block containing the instruction.
        block: BlockId,
        /// Index of the instruction.
        inst_idx: usize,
    },
}

impl<V, Op, C> ExprNode<V, Op, C> {
    /// Whether this is a leaf node.
    #[must_use]
    pub fn is_leaf(&self) -> bool {
        matches!(self, ExprNode::Leaf(_))
    }

    /// Whether this is a compound operation.
    #[must_use]
    pub fn is_op(&self) -> bool {
        matches!(self, ExprNode::Op { .. })
    }

    /// Depth of the expression tree.
    #[must_use]
    pub fn depth(&self) -> usize {
        Preorder::new(self).fold(0, |deepest, (depth, _)| deepest.max(depth))
    }

    /// Count total nodes in the expression tree.
    #[must_use]
    pub fn node_count(&self) -> usize {
        Preorder::new(self).count()
    }
}

/// Every node of a tree with its depth (the root's is 1): a parent before
/// its operands, operands left to right.
struct Preorder<'a, V, Op, C> {
    /// The root, until it has been returned.
    root: Option<&'a ExprNode<V, Op, C>>,
    /// The operands not yet returned, one list per ancestor of the node
    /// returned last that still has any.
    pending: Vec<slice::Iter<'a, ExprNode<V, Op, C>>>,
}

impl<'a, V, Op, C> Preorder<'a, V, Op, C> {
    fn new(root: &'a ExprNode<V, Op, C>) -> Self {
        Self {
            root: Some(root),
            pending: Vec::new(),
        }
    }
}

impl<'a, V, Op, C> Iterator for Preorder<'a, V, Op, C> {
    type Item = (usize, &'a ExprNode<V, Op, C>);

    fn next(&mut self) -> Option<Self::Item> {
        let (depth, node) = match self.root.take() {
            Some(root) => (1, root),
            None => loop {
                let siblings = self.pending.last_mut()?;
                if let Some(operand) = siblings.next() {
                    break (self.pending.len() + 1, operand);
                }
                self.pending.pop();
            },
        };
        if let ExprNode::Op { operands, .. } = node
            && !operands.is_empty()
        {
            self.pending.push(operands.iter());
        }
        Some((depth, node))
    }
}

impl<V, Op, C> Drop for ExprNode<V, Op, C> {
    /// Drops the subtrees on one explicit stack, reusing the operand vector
    /// of this node as its storage. The compiler's drop glue for a node
    /// drops each operand in turn, one stack frame per level.
    fn drop(&mut self) {
        let Self::Op { operands, .. } = self else {
            return;
        };
        let mut pending = core::mem::take(operands);
        while let Some(mut node) = pending.pop() {
            if let Self::Op { operands, .. } = &mut node {
                pending.append(operands);
            }
            // `node` drops here with no operands left, so its own `drop`
            // finds nothing to do.
        }
    }
}

impl<V: Clone, Op: Clone, C: Clone> Clone for ExprNode<V, Op, C> {
    fn clone(&self) -> Self {
        /// An operator whose operands are being cloned.
        struct Open<'a, V, Op, C> {
            operator: &'a Op,
            /// The operands whose clones have not been made yet.
            rest: slice::Iter<'a, ExprNode<V, Op, C>>,
            /// The clones of the operands before them.
            cloned: Vec<ExprNode<V, Op, C>>,
        }

        let mut open: Vec<Open<'_, V, Op, C>> = Vec::new();
        let mut source = self;
        loop {
            // Descend through first operands to an atom, or to an operator
            // with no operands, and clone it.
            let mut cloned = loop {
                match source {
                    Self::Leaf(variable) => break Self::Leaf(variable.clone()),
                    Self::Const(constant) => break Self::Const(constant.clone()),
                    Self::Opaque { block, inst_idx } => {
                        break Self::Opaque {
                            block: *block,
                            inst_idx: *inst_idx,
                        };
                    }
                    Self::Op { operator, operands } => {
                        let mut rest = operands.iter();
                        let Some(first) = rest.next() else {
                            break Self::Op {
                                operator: operator.clone(),
                                operands: Vec::new(),
                            };
                        };
                        open.push(Open {
                            operator,
                            rest,
                            cloned: Vec::with_capacity(operands.len()),
                        });
                        source = first;
                    }
                }
            };
            // Hand the clone to its operator, completing each operator whose
            // operands are now all cloned, until one has an operand left.
            loop {
                let Some(parent) = open.last_mut() else {
                    return cloned;
                };
                parent.cloned.push(cloned);
                if let Some(next) = parent.rest.next() {
                    source = next;
                    break;
                }
                let complete = open
                    .pop()
                    .expect("the operator that took the clone is still open");
                cloned = Self::Op {
                    operator: complete.operator.clone(),
                    operands: complete.cloned,
                };
            }
        }
    }
}

impl<V: PartialEq, Op: PartialEq, C: PartialEq> PartialEq for ExprNode<V, Op, C> {
    /// Compares the trees node by node in pre-order, operator before operands
    /// and operands left to right, stopping at the first difference.
    fn eq(&self, other: &Self) -> bool {
        let mut pending = Vec::new();
        let mut pair = (self, other);
        loop {
            let same_node = match pair {
                (Self::Leaf(left), Self::Leaf(right)) => left == right,
                (Self::Const(left), Self::Const(right)) => left == right,
                (
                    Self::Opaque {
                        block: left_block,
                        inst_idx: left_index,
                    },
                    Self::Opaque {
                        block: right_block,
                        inst_idx: right_index,
                    },
                ) => left_block == right_block && left_index == right_index,
                (
                    Self::Op {
                        operator: left_operator,
                        operands: left_operands,
                    },
                    Self::Op {
                        operator: right_operator,
                        operands: right_operands,
                    },
                ) => {
                    let same = left_operator == right_operator
                        && left_operands.len() == right_operands.len();
                    if same {
                        pending.extend(left_operands.iter().zip(right_operands).rev());
                    }
                    same
                }
                _ => false,
            };
            if !same_node {
                return false;
            }
            let Some(next) = pending.pop() else {
                return true;
            };
            pair = next;
        }
    }
}

impl<V: Eq, Op: Eq, C: Eq> Eq for ExprNode<V, Op, C> {}

impl<V: Debug, Op: Debug, C: Debug> Debug for ExprNode<V, Op, C> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut printer = Printer {
            pretty: formatter.alternate(),
            out: formatter,
            level: 0,
            at_line_start: false,
        };
        let mut pending = alloc::vec![Print::Node(self)];
        while let Some(step) = pending.pop() {
            match step {
                Print::Node(node) => printer.expand(node, &mut pending),
                Print::Text(text) => printer.write_str(text)?,
                Print::Value(value) => printer.value(value)?,
                Print::Indent => printer.level += 1,
                Print::Dedent => printer.level -= 1,
            }
        }
        Ok(())
    }
}

/// One step of printing a tree: what a derived `Debug` writes, as data, so
/// that the walk over the tree needs no stack frame per level.
enum Print<'a, V, Op, C> {
    /// Print this node, which expands into the steps that follow.
    Node(&'a ExprNode<V, Op, C>),
    /// Write this text, indented to the current level if it starts a line.
    Text(&'static str),
    /// Format this field value.
    Value(&'a dyn Debug),
    /// The text that follows is one level deeper.
    Indent,
    /// The text that follows is one level shallower.
    Dedent,
}

/// Writes a tree to a formatter in the layout of a derived `Debug`: all on
/// one line, or one field per line and four spaces per level for `{:#?}`.
struct Printer<'a, 'b> {
    out: &'a mut fmt::Formatter<'b>,
    /// Whether the formatter asked for `{:#?}`.
    pretty: bool,
    /// How many levels of fields enclose the text being written. It stays
    /// zero unless `pretty`.
    level: usize,
    /// Whether the last character written ended a line.
    at_line_start: bool,
}

impl Printer<'_, '_> {
    /// Replaces `node` by the steps that print it, pushed so that the first
    /// step to run is on top of `pending`.
    fn expand<'n, V: Debug, Op: Debug, C: Debug>(
        &self,
        node: &'n ExprNode<V, Op, C>,
        pending: &mut Vec<Print<'n, V, Op, C>>,
    ) {
        use Print::{Dedent, Indent, Text, Value};

        fn push_in_order<'n, V, Op, C, const N: usize>(
            pending: &mut Vec<Print<'n, V, Op, C>>,
            steps: [Print<'n, V, Op, C>; N],
        ) {
            pending.extend(steps.into_iter().rev());
        }

        match (node, self.pretty) {
            (ExprNode::Leaf(variable), false) => {
                push_in_order(pending, [Text("Leaf("), Value(variable), Text(")")]);
            }
            (ExprNode::Leaf(variable), true) => push_in_order(
                pending,
                [
                    Text("Leaf(\n"),
                    Indent,
                    Value(variable),
                    Text(",\n"),
                    Dedent,
                    Text(")"),
                ],
            ),
            (ExprNode::Const(constant), false) => {
                push_in_order(pending, [Text("Const("), Value(constant), Text(")")]);
            }
            (ExprNode::Const(constant), true) => push_in_order(
                pending,
                [
                    Text("Const(\n"),
                    Indent,
                    Value(constant),
                    Text(",\n"),
                    Dedent,
                    Text(")"),
                ],
            ),
            (ExprNode::Opaque { block, inst_idx }, false) => push_in_order(
                pending,
                [
                    Text("Opaque { block: "),
                    Value(block),
                    Text(", inst_idx: "),
                    Value(inst_idx),
                    Text(" }"),
                ],
            ),
            (ExprNode::Opaque { block, inst_idx }, true) => push_in_order(
                pending,
                [
                    Text("Opaque {\n"),
                    Indent,
                    Text("block: "),
                    Value(block),
                    Text(",\ninst_idx: "),
                    Value(inst_idx),
                    Text(",\n"),
                    Dedent,
                    Text("}"),
                ],
            ),
            (ExprNode::Op { operator, operands }, false) => {
                pending.push(Text("] }"));
                for (position, operand) in operands.iter().enumerate().rev() {
                    pending.push(Print::Node(operand));
                    if position > 0 {
                        pending.push(Text(", "));
                    }
                }
                push_in_order(
                    pending,
                    [
                        Text("Op { operator: "),
                        Value(operator),
                        Text(", operands: ["),
                    ],
                );
            }
            (ExprNode::Op { operator, operands }, true) => {
                pending.extend([Dedent, Text("}")].into_iter().rev());
                pending.push(Text("],\n"));
                if !operands.is_empty() {
                    pending.push(Dedent);
                    for operand in operands.iter().rev() {
                        pending.push(Text(",\n"));
                        pending.push(Print::Node(operand));
                    }
                    push_in_order(pending, [Text("\n"), Indent]);
                }
                push_in_order(
                    pending,
                    [
                        Text("Op {\n"),
                        Indent,
                        Text("operator: "),
                        Value(operator),
                        Text(",\noperands: ["),
                    ],
                );
            }
        }
    }

    /// Formats one field value as the derived implementation would in the
    /// current mode.
    fn value(&mut self, value: &dyn Debug) -> fmt::Result {
        if self.pretty {
            fmt::write(self, format_args!("{value:#?}"))
        } else {
            value.fmt(self.out)
        }
    }

    /// Writes `level` indents of four spaces.
    fn indent(&mut self) -> fmt::Result {
        const SPACES: &str = "                ";
        let mut remaining = self.level * 4;
        while remaining > 0 {
            let width = remaining.min(SPACES.len());
            self.out.write_str(&SPACES[..width])?;
            remaining -= width;
        }
        Ok(())
    }
}

impl Write for Printer<'_, '_> {
    /// Writes `text`, indenting each line that it starts.
    fn write_str(&mut self, text: &str) -> fmt::Result {
        for line in text.split_inclusive('\n') {
            if self.at_line_start {
                self.indent()?;
            }
            self.at_line_start = line.ends_with('\n');
            self.out.write_str(line)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;
    use std::format;
    use std::rc::Rc;

    use super::*;
    use crate::test_util::rng::Rng;

    type Tree = ExprNode<u32, &'static str, i64>;

    /// The tree as `#[derive(Debug, Clone, PartialEq, Eq)]` defines it: the
    /// implementations that the hand-written ones replaced, kept as the
    /// oracle for every tree shallow enough to recurse over.
    #[derive(Debug, Clone, PartialEq, Eq)]
    enum Derived {
        Leaf(u32),
        Op {
            operator: &'static str,
            operands: Vec<Derived>,
        },
        Const(i64),
        Opaque {
            block: BlockId,
            inst_idx: usize,
        },
    }

    impl Derived {
        fn into_node(self) -> Tree {
            match self {
                Self::Leaf(variable) => ExprNode::Leaf(variable),
                Self::Const(constant) => ExprNode::Const(constant),
                Self::Opaque { block, inst_idx } => ExprNode::Opaque { block, inst_idx },
                Self::Op { operator, operands } => ExprNode::Op {
                    operator,
                    operands: operands.into_iter().map(Self::into_node).collect(),
                },
            }
        }

        fn depth(&self) -> usize {
            match self {
                Self::Op { operands, .. } => {
                    1 + operands.iter().map(Self::depth).max().unwrap_or(0)
                }
                _ => 1,
            }
        }

        fn node_count(&self) -> usize {
            match self {
                Self::Op { operands, .. } => {
                    1 + operands.iter().map(Self::node_count).sum::<usize>()
                }
                _ => 1,
            }
        }
    }

    fn derived_from(node: &Tree) -> Derived {
        match node {
            ExprNode::Leaf(variable) => Derived::Leaf(*variable),
            ExprNode::Const(constant) => Derived::Const(*constant),
            ExprNode::Opaque { block, inst_idx } => Derived::Opaque {
                block: *block,
                inst_idx: *inst_idx,
            },
            ExprNode::Op { operator, operands } => Derived::Op {
                operator,
                operands: operands.iter().map(derived_from).collect(),
            },
        }
    }

    fn random_atom(rng: &mut Rng) -> Derived {
        match rng.below(3) {
            0 => Derived::Leaf(u32::try_from(rng.below(300)).unwrap()),
            1 => Derived::Const(i64::try_from(rng.below(2_000)).unwrap() - 1_000),
            _ => Derived::Opaque {
                block: BlockId::from_index(rng.below(4)),
                inst_idx: rng.below(50),
            },
        }
    }

    /// A tree of at most `levels` operator levels, with up to three operands
    /// per operator (none sometimes).
    fn random_tree(rng: &mut Rng, levels: usize) -> Derived {
        if levels == 0 || rng.chance(30) {
            return random_atom(rng);
        }
        Derived::Op {
            operator: ["add", "mul", "neg"][rng.below(3)],
            operands: (0..rng.below(4))
                .map(|_| random_tree(rng, levels - 1))
                .collect(),
        }
    }

    /// Replaces some nodes of `tree` by atoms, so a pair is often equal but
    /// for one node.
    fn perturb(tree: &mut Derived, rng: &mut Rng) {
        if rng.one_in(8) {
            *tree = random_atom(rng);
        } else if let Derived::Op { operands, .. } = tree {
            for operand in operands {
                perturb(operand, rng);
            }
        }
    }

    const SEEDS: u64 = 400;

    #[test]
    fn printing_matches_the_derived_format() {
        let mut printed_operations = 0_u64;
        for seed in 0..SEEDS {
            let mut rng = Rng::new(seed);
            let derived = random_tree(&mut rng, 5);
            let node = derived.clone().into_node();
            printed_operations += u64::from(node.is_op());

            assert_eq!(format!("{node:?}"), format!("{derived:?}"), "seed {seed}");
            assert_eq!(format!("{node:#?}"), format!("{derived:#?}"), "seed {seed}");
            assert_eq!(format!("{node:x?}"), format!("{derived:x?}"), "seed {seed}");
            assert_eq!(format!("{node:5?}"), format!("{derived:5?}"), "seed {seed}");
        }
        assert!(printed_operations > SEEDS / 2, "operators are covered");
    }

    #[test]
    fn printing_nests_inside_another_pretty_print() {
        // The whole tree sits two levels deep, under a tuple's field and a
        // list's element, and every line of it picks up both indents.
        let mut rng = Rng::new(3);
        let derived = random_tree(&mut rng, 4);
        let node = derived.clone().into_node();

        assert_eq!(
            format!("{:#?}", [(1, node)]),
            format!("{:#?}", [(1, derived)])
        );
    }

    #[test]
    fn printing_an_empty_operator_and_each_atom_matches_the_derived_format() {
        let atoms = [
            Derived::Op {
                operator: "nop",
                operands: Vec::new(),
            },
            Derived::Leaf(0),
            Derived::Const(-1),
            Derived::Opaque {
                block: BlockId::from_index(2),
                inst_idx: 9,
            },
        ];
        for derived in atoms {
            let node = derived.clone().into_node();
            assert_eq!(format!("{node:?}"), format!("{derived:?}"));
            assert_eq!(format!("{node:#?}"), format!("{derived:#?}"));
        }
    }

    #[test]
    fn equality_matches_the_derived_comparison() {
        let (mut equal, mut different) = (0, 0);
        for seed in 0..SEEDS {
            let mut rng = Rng::new(seed);
            let left = random_tree(&mut rng, 5);
            let mut right = left.clone();
            perturb(&mut right, &mut rng);
            let unrelated = random_tree(&mut rng, 5);

            for other in [&right, &unrelated, &left] {
                let expected = left == *other;
                assert_eq!(
                    left.clone().into_node() == other.clone().into_node(),
                    expected,
                    "seed {seed}"
                );
                if expected {
                    equal += 1;
                } else {
                    different += 1;
                }
            }
        }
        assert!(
            equal >= SEEDS && different >= SEEDS,
            "both outcomes are covered"
        );
    }

    #[test]
    fn clones_and_sizes_match_the_derived_behavior() {
        for seed in 0..SEEDS {
            let mut rng = Rng::new(seed);
            let derived = random_tree(&mut rng, 5);
            let node = derived.clone().into_node();

            let copy = node.clone();

            assert_eq!(derived_from(&copy), derived, "seed {seed}");
            assert_eq!(copy, node, "seed {seed}");
            assert_eq!(node.depth(), derived.depth(), "seed {seed}");
            assert_eq!(node.node_count(), derived.node_count(), "seed {seed}");
        }
    }

    /// A value that records its identity when dropped, so a drop that skips
    /// or repeats a node shows in the log.
    #[derive(Clone)]
    struct Tracked {
        id: u32,
        dropped: Rc<RefCell<Vec<u32>>>,
    }

    impl Drop for Tracked {
        fn drop(&mut self) {
            self.dropped.borrow_mut().push(self.id);
        }
    }

    /// The tree of `tree`'s shape whose values are `Tracked`, numbered in
    /// pre-order from `*next`, which ends past the last number used.
    fn tracked_from(
        tree: &Derived,
        dropped: &Rc<RefCell<Vec<u32>>>,
        next: &mut u32,
    ) -> ExprNode<Tracked, Tracked, Tracked> {
        let mut track = || {
            *next += 1;
            Tracked {
                id: *next,
                dropped: Rc::clone(dropped),
            }
        };
        match tree {
            Derived::Leaf(_) => ExprNode::Leaf(track()),
            Derived::Const(_) => ExprNode::Const(track()),
            Derived::Opaque { block, inst_idx } => ExprNode::Opaque {
                block: *block,
                inst_idx: *inst_idx,
            },
            Derived::Op { operands, .. } => ExprNode::Op {
                operator: track(),
                operands: operands
                    .iter()
                    .map(|operand| tracked_from(operand, dropped, next))
                    .collect(),
            },
        }
    }

    #[test]
    fn dropping_a_tree_and_its_clone_drops_every_value_exactly_once() {
        for seed in 0..SEEDS {
            let mut rng = Rng::new(seed);
            let derived = random_tree(&mut rng, 5);
            let dropped = Rc::new(RefCell::new(Vec::new()));
            let mut count = 0;
            let tree = tracked_from(&derived, &dropped, &mut count);
            let copy = tree.clone();

            drop(tree);
            assert_eq!(dropped.borrow().len(), count as usize, "seed {seed}");
            drop(copy);

            let mut log = dropped.borrow().clone();
            log.sort_unstable();
            let twice: Vec<u32> = (1..=count).flat_map(|id| [id, id]).collect();
            assert_eq!(log, twice, "seed {seed}");
        }
    }

    #[test]
    fn the_operands_of_an_owned_node_can_be_taken_through_mutable_access() {
        let mut node = Derived::Op {
            operator: "add",
            operands: alloc::vec![Derived::Leaf(1), Derived::Const(2)],
        }
        .into_node();

        let ExprNode::Op { operands, .. } = &mut node else {
            unreachable!("the node is an operation");
        };
        let taken = core::mem::take(operands);

        assert_eq!(taken, [ExprNode::Leaf(1), ExprNode::Const(2)]);
        assert_eq!(node.node_count(), 1);
    }
}
