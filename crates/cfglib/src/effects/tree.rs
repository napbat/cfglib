//! Expression trees over the statements of one instruction, for a readable
//! and exact view.

extern crate alloc;

use alloc::vec::Vec;

use crate::graph::view::DenseId;
use crate::ir::rtl::{
    Action, BitSlice, Bits, Control, Dialect, Place, RecordView, Selection, Statement, ValueRange,
    WriteTarget,
};

use super::defs::Defs;
use super::dialect::DialectEffects;

/// The identity of one node of a [`StatementTree`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TreeNodeId(u32);

impl TreeNodeId {
    /// Returns the dense index of the node in its tree.
    #[must_use]
    pub const fn index(self) -> usize {
        self.0 as usize
    }
}

impl DenseId for TreeNodeId {
    fn from_index(index: usize) -> Self {
        Self(u32::try_from(index).expect("a tree holds at most u32::MAX nodes"))
    }

    fn index(self) -> usize {
        self.0 as usize
    }
}

/// One value of the instruction, as [`StatementTree::node`] lends it.
///
/// Each value of the statements is one node, and an operand is a node.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TreeNode<'t, D: Dialect> {
    /// A read of the state before the instruction.
    Read(Selection<'t, D, TreeNodeId>),
    /// A constant.
    Const {
        /// The little-endian 64-bit words of the value.
        words: &'t [u64],
        /// The type of the value.
        width: Bits,
    },
    /// One result of a pure operator.
    Apply {
        /// The operator.
        operator: &'t D::Operator,
        /// The operands, in operator order.
        args: &'t [TreeNodeId],
        /// The position of this value among the results of the operator.
        result: u32,
        /// The type of the value.
        width: Bits,
    },
    /// A nondeterministic choice.
    Nondet {
        /// The type of the value.
        width: Bits,
        /// The domain of the choice.
        domain: D::NondetDomain,
    },
    /// One result of an ordered effect. The [`TreeItem::Effect`] of the
    /// effect names the node.
    Effect {
        /// The position of the effect among the effects of the instruction,
        /// from zero.
        ordinal: u32,
        /// The position of this value among the results of the effect.
        result: u32,
        /// The type of the value.
        width: Bits,
    },
}

/// What one span of a written root receives, with select-of-self recognized.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TreeSpanValue<D: Dialect> {
    /// The span keeps its value from before the instruction.
    Keep,
    /// The span receives bits of `value` from bit `offset`.
    Assign {
        /// The value.
        value: TreeNodeId,
        /// The first bit of `value` that the span receives.
        offset: u16,
    },
    /// The span receives bits of `value` from bit `offset` when `guard` is
    /// one, and keeps its value otherwise.
    ///
    /// The statements hold a select of `guard`, `value`, and a read of the
    /// bits of the root that the span receives: the read starts `offset`
    /// bits below the span and has the width of `value`.
    Conditional {
        /// The one-bit guard.
        guard: TreeNodeId,
        /// The value for a guard of one.
        value: TreeNodeId,
        /// The first bit of the value that the span receives.
        offset: u16,
    },
    /// The span receives a fresh nondeterministic choice.
    Undefined(D::NondetDomain),
}

/// One span of a written root.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TreeSpan<D: Dialect> {
    /// The bits of the root.
    pub span: BitSlice,
    /// What the bits receive.
    pub value: TreeSpanValue<D>,
}

/// One line of a [`StatementTree`], as [`StatementTree::items`] lends it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TreeItem<'t, D: Dialect> {
    /// Names a shared value before the first item that uses it.
    Let(TreeNodeId),
    /// An ordered effect.
    Effect {
        /// The effect.
        operation: &'t D::EffectOp,
        /// The operands, in effect order.
        args: &'t [TreeNodeId],
        /// The results, in effect order. Each one is a
        /// [`TreeNode::Effect`].
        results: &'t [TreeNodeId],
        /// The outcomes that the environment can raise at the effect.
        outcomes: &'t [D::Outcome],
    },
    /// A raise.
    Raise {
        /// The kind of outcome.
        outcome: D::Outcome,
        /// The one-bit guard, or `None` for an unconditional raise.
        guard: Option<TreeNodeId>,
        /// The values that the outcome carries.
        payload: &'t [TreeNodeId],
        /// The roots that the outcome publishes, with their values.
        binds: &'t [(D::Root, TreeNodeId)],
    },
    /// A write of one root, with its complete coverage.
    Write {
        /// The root.
        root: D::Root,
        /// The spans in ascending order.
        spans: &'t [TreeSpan<D>],
        /// The policy that the producer resolved into the spans.
        policy: D::WritePolicy,
    },
    /// A write of the candidate that `index` selects.
    FiniteWrite {
        /// The selector.
        index: TreeNodeId,
        /// The candidates, in index order.
        candidates: &'t [Place<D>],
        /// The written value.
        value: TreeNodeId,
        /// The policy of the write.
        policy: D::WritePolicy,
    },
    /// A write of an element of a sparse bank.
    SparseWrite {
        /// The bank.
        bank: D::Root,
        /// The element key.
        key: TreeNodeId,
        /// The written bits of the element.
        slice: BitSlice,
        /// The written value.
        value: TreeNodeId,
        /// The policy of the write.
        policy: D::WritePolicy,
    },
    /// The control boundary, which is always the last item.
    Control(Control<TreeNodeId>),
}

/// A run of one pool of a tree.
#[derive(Debug, Clone, Copy)]
struct Run {
    start: u32,
    len: u32,
}

impl Run {
    fn push<T: Copy>(pool: &mut Vec<T>, items: impl IntoIterator<Item = T>) -> Self {
        let start = pool.len();
        pool.extend(items);
        let len = pool.len() - start;
        Self {
            start: u32::try_from(start).expect("a tree pool holds at most u32::MAX items"),
            len: u32::try_from(len).expect("a tree pool holds at most u32::MAX items"),
        }
    }

    fn get<T>(self, pool: &[T]) -> &[T] {
        let start = self.start as usize;
        &pool[start..start + self.len as usize]
    }
}

#[derive(Debug)]
enum NodeData<'a, D: Dialect> {
    Read(Selection<'a, D, TreeNodeId>),
    Const {
        words: &'a [u64],
        width: Bits,
    },
    /// An operator that the dialect knows as a number
    /// ([`DialectEffects::constant`]): the number is `numbers[at]`.
    Number {
        at: u32,
        width: Bits,
    },
    Apply {
        operator: &'a D::Operator,
        args: Run,
        result: u32,
        width: Bits,
    },
    Nondet {
        width: Bits,
        domain: D::NondetDomain,
    },
    Effect {
        ordinal: u32,
        result: u32,
        width: Bits,
    },
}

#[expect(
    clippy::expl_impl_clone_on_copy,
    reason = "a derive bounds the dialect types that the fields only borrow"
)]
impl<D: Dialect> Clone for NodeData<'_, D> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<D: Dialect> Copy for NodeData<'_, D> {}

#[derive(Debug)]
enum ItemData<'a, D: Dialect> {
    Let(TreeNodeId),
    Effect {
        operation: &'a D::EffectOp,
        args: Run,
        results: Run,
        outcomes: &'a [D::Outcome],
    },
    Raise {
        outcome: D::Outcome,
        guard: Option<TreeNodeId>,
        payload: Run,
        binds: Run,
    },
    Write {
        root: D::Root,
        spans: Run,
        policy: D::WritePolicy,
    },
    FiniteWrite {
        index: TreeNodeId,
        candidates: &'a [Place<D>],
        value: TreeNodeId,
        policy: D::WritePolicy,
    },
    SparseWrite {
        bank: D::Root,
        key: TreeNodeId,
        slice: BitSlice,
        value: TreeNodeId,
        policy: D::WritePolicy,
    },
    Control(Control<TreeNodeId>),
}

#[expect(
    clippy::expl_impl_clone_on_copy,
    reason = "a derive bounds the dialect types that the fields only borrow"
)]
impl<D: Dialect> Clone for ItemData<'_, D> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<D: Dialect> Copy for ItemData<'_, D> {}

/// The statements of one instruction as expression trees.
///
/// The tree is the generic part of a readable view of instruction semantics.
/// Each ordered statement is one item: an effect, a raise, a write, and at
/// the end the control boundary. Each operand of an item is the root of an
/// expression tree over the pure values. A renderer of a dialect turns the
/// trees into text, and it can match named operations there.
///
/// The tree drops no meaning of the statements:
///
/// - A value with one use stays inline. An operator result or a
///   nondeterministic choice with more than one use is shared: a
///   [`TreeItem::Let`] names it before the first item that uses it, and every
///   use refers to that name. A read and a constant always stay inline,
///   because every read sees the state before the instruction.
/// - A span whose value is a select of the old value of its own bits is a
///   [`TreeSpanValue::Conditional`]. The guard of the select is the guard of
///   the assignment, and the select and the read of the old bits are not uses.
/// - The result of an effect is named by its [`TreeItem::Effect`].
///
/// A pure value that no item uses is not in any tree.
#[derive(Debug, Clone)]
pub struct StatementTree<'a, D: Dialect> {
    nodes: Vec<NodeData<'a, D>>,
    refs: Vec<TreeNodeId>,
    /// The numbers of the [`NodeData::Number`] nodes.
    numbers: Vec<u64>,
    binds: Vec<(D::Root, TreeNodeId)>,
    spans: Vec<TreeSpan<D>>,
    items: Vec<ItemData<'a, D>>,
    uses: Vec<u32>,
    bound: Vec<bool>,
}

impl<'a, D: Dialect> StatementTree<'a, D> {
    /// Builds the tree of one sealed record.
    ///
    /// # Panics
    ///
    /// Panics as [`Self::compute`] does.
    #[must_use]
    pub fn compute_record<C: DialectEffects<D>, R: RecordView<D>>(
        dialect: &C,
        record: &'a R,
    ) -> Self {
        Self::compute(dialect, record.statements(), record.control(), |value| {
            record.width(value)
        })
    }

    /// Builds the tree of the statements of one instruction and its control
    /// boundary.
    ///
    /// `dialect` recognizes the select operators. `width` gives the type of
    /// each value that the statements define.
    ///
    /// # Panics
    ///
    /// Panics when an operand is not a value that an earlier statement
    /// defines, or when [`DialectEffects::select`] names an operand position
    /// past the operands.
    #[must_use]
    pub fn compute<C: DialectEffects<D>, V: DenseId + 'a>(
        dialect: &C,
        statements: impl IntoIterator<Item = Statement<'a, D, V>>,
        control: Control<V>,
        width: impl Fn(V) -> Bits,
    ) -> Self {
        let mut build = Build {
            defs: Defs::new(),
            tree: Self {
                nodes: Vec::new(),
                refs: Vec::new(),
                numbers: Vec::new(),
                binds: Vec::new(),
                spans: Vec::new(),
                items: Vec::new(),
                uses: Vec::new(),
                bound: Vec::new(),
            },
        };
        for statement in statements {
            build.statement(dialect, &statement, &width);
        }
        let control = build.control(control);
        build.tree.items.push(ItemData::Control(control));
        let mut tree = build.tree;
        tree.count_uses();
        tree.place_lets();
        tree
    }

    /// Returns how many nodes the tree holds.
    #[must_use]
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Returns the node `id`.
    ///
    /// # Panics
    ///
    /// Panics when `id` is not a node of this tree.
    #[must_use]
    pub fn node(&self, id: TreeNodeId) -> TreeNode<'_, D> {
        match self.nodes[id.index()] {
            NodeData::Read(selection) => TreeNode::Read(selection),
            NodeData::Const { words, width } => TreeNode::Const { words, width },
            NodeData::Number { at, width } => TreeNode::Const {
                words: core::slice::from_ref(&self.numbers[at as usize]),
                width,
            },
            NodeData::Apply {
                operator,
                args,
                result,
                width,
            } => TreeNode::Apply {
                operator,
                args: args.get(&self.refs),
                result,
                width,
            },
            NodeData::Nondet { width, domain } => TreeNode::Nondet { width, domain },
            NodeData::Effect {
                ordinal,
                result,
                width,
            } => TreeNode::Effect {
                ordinal,
                result,
                width,
            },
        }
    }

    /// Returns how often the items and the trees refer to node `id`. A node
    /// inside a shared node counts once for that node.
    ///
    /// # Panics
    ///
    /// Panics when `id` is not a node of this tree.
    #[must_use]
    pub fn use_count(&self, id: TreeNodeId) -> u32 {
        self.uses[id.index()]
    }

    /// Returns whether node `id` has a name: a [`TreeItem::Let`] names it, or
    /// it is the result of an effect. A renderer writes the name of a named
    /// node and writes any other node inline.
    ///
    /// # Panics
    ///
    /// Panics when `id` is not a node of this tree.
    #[must_use]
    pub fn is_named(&self, id: TreeNodeId) -> bool {
        self.bound[id.index()] || matches!(self.nodes[id.index()], NodeData::Effect { .. })
    }

    /// Iterates over the items in order. The last item is the control
    /// boundary.
    pub fn items(&self) -> impl ExactSizeIterator<Item = TreeItem<'_, D>> + '_ {
        self.items.iter().map(|item| self.item(item))
    }

    fn item(&self, item: &ItemData<'a, D>) -> TreeItem<'_, D> {
        match *item {
            ItemData::Let(node) => TreeItem::Let(node),
            ItemData::Effect {
                operation,
                args,
                results,
                outcomes,
            } => TreeItem::Effect {
                operation,
                args: args.get(&self.refs),
                results: results.get(&self.refs),
                outcomes,
            },
            ItemData::Raise {
                outcome,
                guard,
                payload,
                binds,
            } => TreeItem::Raise {
                outcome,
                guard,
                payload: payload.get(&self.refs),
                binds: binds.get(&self.binds),
            },
            ItemData::Write {
                root,
                spans,
                policy,
            } => TreeItem::Write {
                root,
                spans: spans.get(&self.spans),
                policy,
            },
            ItemData::FiniteWrite {
                index,
                candidates,
                value,
                policy,
            } => TreeItem::FiniteWrite {
                index,
                candidates,
                value,
                policy,
            },
            ItemData::SparseWrite {
                bank,
                key,
                slice,
                value,
                policy,
            } => TreeItem::SparseWrite {
                bank,
                key,
                slice,
                value,
                policy,
            },
            ItemData::Control(control) => TreeItem::Control(control),
        }
    }

    /// Calls `visit` for each node that `item` refers to, in operand order.
    fn item_operands(&self, item: &ItemData<'a, D>, mut visit: impl FnMut(TreeNodeId)) {
        match *item {
            ItemData::Let(_) => {}
            ItemData::Effect { args, .. } => args.get(&self.refs).iter().copied().for_each(visit),
            ItemData::Raise {
                guard,
                payload,
                binds,
                ..
            } => {
                guard.into_iter().for_each(&mut visit);
                payload.get(&self.refs).iter().copied().for_each(&mut visit);
                binds
                    .get(&self.binds)
                    .iter()
                    .for_each(|&(_, node)| visit(node));
            }
            ItemData::Write { spans, .. } => {
                for span in spans.get(&self.spans) {
                    match span.value {
                        TreeSpanValue::Keep | TreeSpanValue::Undefined(_) => {}
                        TreeSpanValue::Assign { value, .. } => visit(value),
                        TreeSpanValue::Conditional { guard, value, .. } => {
                            visit(guard);
                            visit(value);
                        }
                    }
                }
            }
            ItemData::FiniteWrite { index, value, .. } => {
                visit(index);
                visit(value);
            }
            ItemData::SparseWrite { key, value, .. } => {
                visit(key);
                visit(value);
            }
            ItemData::Control(control) => match control {
                Control::Advance | Control::Terminating => {}
                Control::Jump { target }
                | Control::Call { target }
                | Control::Return { target } => {
                    visit(target);
                }
                Control::Branch {
                    condition,
                    taken,
                    not_taken,
                } => {
                    visit(condition);
                    visit(taken);
                    not_taken.into_iter().for_each(visit);
                }
                Control::Dispatch { scrutinee } => visit(scrutinee),
            },
        }
    }

    /// Calls `visit` for each operand of node `id`, in operand order.
    fn node_operands(&self, id: TreeNodeId, mut visit: impl FnMut(TreeNodeId)) {
        match self.nodes[id.index()] {
            NodeData::Read(Selection::FiniteIndexed { index, .. }) => visit(index),
            NodeData::Read(Selection::SparseIndexed { key, .. }) => visit(key),
            NodeData::Apply { args, .. } => args.get(&self.refs).iter().copied().for_each(visit),
            NodeData::Read(Selection::Direct(_))
            | NodeData::Const { .. }
            | NodeData::Number { .. }
            | NodeData::Nondet { .. }
            | NodeData::Effect { .. } => {}
        }
    }

    /// Counts the references of the items, and of every node that an item
    /// reaches, to each node. The operands of a node count once, at the
    /// first reference to the node.
    fn count_uses(&mut self) {
        let mut uses = alloc::vec![0_u32; self.nodes.len()];
        let mut stack = Vec::new();
        for item in &self.items {
            self.item_operands(item, |node| stack.push(node));
            while let Some(node) = stack.pop() {
                uses[node.index()] += 1;
                if uses[node.index()] == 1 {
                    self.node_operands(node, |operand| stack.push(operand));
                }
            }
        }
        self.uses = uses;
    }

    /// Returns whether a node with these uses gets a [`TreeItem::Let`].
    fn shares(&self, node: TreeNodeId) -> bool {
        self.uses[node.index()] > 1
            && matches!(
                self.nodes[node.index()],
                NodeData::Apply { .. } | NodeData::Nondet { .. }
            )
    }

    /// Inserts a [`TreeItem::Let`] for each shared node before the first
    /// item that uses it. A shared node comes after the shared nodes inside
    /// it.
    fn place_lets(&mut self) {
        let mut bound = alloc::vec![false; self.nodes.len()];
        let mut visited = alloc::vec![false; self.nodes.len()];
        let mut items = Vec::with_capacity(self.items.len());
        let mut stack: Vec<(TreeNodeId, bool)> = Vec::new();
        let mut roots = Vec::new();
        for item in &self.items {
            roots.clear();
            self.item_operands(item, |node| roots.push(node));
            for &root in roots.iter().rev() {
                stack.push((root, false));
            }
            while let Some((node, expanded)) = stack.pop() {
                if visited[node.index()] {
                    continue;
                }
                if expanded {
                    visited[node.index()] = true;
                    if self.shares(node) {
                        bound[node.index()] = true;
                        items.push(ItemData::Let(node));
                    }
                    continue;
                }
                stack.push((node, true));
                let start = stack.len();
                self.node_operands(node, |operand| stack.push((operand, false)));
                stack[start..].reverse();
            }
            items.push(*item);
        }
        self.items = items;
        self.bound = bound;
    }
}

/// The state of one build.
struct Build<'a, D: Dialect, V> {
    defs: Defs<'a, D, V, TreeNodeId>,
    tree: StatementTree<'a, D>,
}

impl<'a, D: Dialect, V: DenseId> Build<'a, D, V> {
    fn node(&self, value: V) -> TreeNodeId {
        self.defs.slot(value).extra
    }

    fn statement<C: DialectEffects<D>>(
        &mut self,
        dialect: &C,
        statement: &Statement<'a, D, V>,
        width: &impl Fn(V) -> Bits,
    ) {
        self.defs
            .define(statement, width, |_, _| TreeNodeId(u32::MAX));
        match *statement {
            Statement::Read { result, source } => {
                let source = match source {
                    Selection::Direct(place) => Selection::Direct(place),
                    Selection::FiniteIndexed { index, candidates } => Selection::FiniteIndexed {
                        index: self.node(index),
                        candidates,
                    },
                    Selection::SparseIndexed { bank, key, slice } => Selection::SparseIndexed {
                        bank,
                        key: self.node(key),
                        slice,
                    },
                };
                self.add_node(result, NodeData::Read(source));
            }
            Statement::Constant {
                result,
                words,
                width,
            } => self.add_node(result, NodeData::Const { words, width }),
            Statement::Apply {
                results:
                    ValueRange {
                        first: value,
                        len: 1,
                    },
                operator,
                args: [],
            } if let Some(number) = dialect.constant(operator) => {
                let at = u32::try_from(self.tree.numbers.len())
                    .expect("a tree holds fewer than 2^32 numbers");
                self.tree.numbers.push(number);
                let width = self.defs.slot(value).width;
                self.add_node(value, NodeData::Number { at, width });
            }
            Statement::Apply {
                results,
                operator,
                args,
            } => {
                let args = self.refs(args.iter().copied());
                for (result, value) in (0..).zip(results.iter()) {
                    let width = self.defs.slot(value).width;
                    self.add_node(
                        value,
                        NodeData::Apply {
                            operator,
                            args,
                            result,
                            width,
                        },
                    );
                }
            }
            Statement::Nondet {
                result,
                width,
                domain,
            } => self.add_node(result, NodeData::Nondet { width, domain }),
            Statement::Effect {
                results,
                operation,
                args,
                outcomes,
            } => self.effect(results, operation, args, outcomes),
            Statement::Raise(raise) => {
                let guard = raise.guard.map(|guard| self.node(guard));
                let payload = self.refs(raise.payload.iter().copied());
                let binds = Run::push(
                    &mut self.tree.binds,
                    raise
                        .binds
                        .iter()
                        .map(|&(root, value)| (root, self.defs.slot(value).extra)),
                );
                self.tree.items.push(ItemData::Raise {
                    outcome: raise.outcome,
                    guard,
                    payload,
                    binds,
                });
            }
            Statement::Write { target, policy } => {
                let item = self.write(dialect, target, policy);
                self.tree.items.push(item);
            }
        }
    }

    /// Adds the nodes of the results of one effect and its item.
    fn effect(
        &mut self,
        results: ValueRange<V>,
        operation: &'a D::EffectOp,
        args: &'a [V],
        outcomes: &'a [D::Outcome],
    ) {
        let ordinal = self.defs.last_effect();
        for (result, value) in (0..).zip(results.iter()) {
            let width = self.defs.slot(value).width;
            self.add_node(
                value,
                NodeData::Effect {
                    ordinal,
                    result,
                    width,
                },
            );
        }
        let args = self.refs(args.iter().copied());
        let results = self.refs(results.iter());
        self.tree.items.push(ItemData::Effect {
            operation,
            args,
            results,
            outcomes,
        });
    }

    /// Adds the nodes of `values` to the reference pool.
    fn refs(&mut self, values: impl IntoIterator<Item = V>) -> Run {
        let defs = &self.defs;
        Run::push(
            &mut self.tree.refs,
            values.into_iter().map(|value| defs.slot(value).extra),
        )
    }

    fn add_node(&mut self, value: V, node: NodeData<'a, D>) {
        let id = TreeNodeId::from_index(self.tree.nodes.len());
        self.tree.nodes.push(node);
        self.defs.slot_mut(value).extra = id;
    }

    fn write<C: DialectEffects<D>>(
        &mut self,
        dialect: &C,
        target: WriteTarget<'a, D, V>,
        policy: D::WritePolicy,
    ) -> ItemData<'a, D> {
        match target {
            WriteTarget::Root { root, plan } => {
                let start = self.tree.spans.len();
                for span in plan {
                    let value = match span.action {
                        Action::PreservePreState => TreeSpanValue::Keep,
                        Action::Nondeterministic(domain) => TreeSpanValue::Undefined(domain),
                        Action::TakeInput { value, offset } => {
                            match self.defs.conditional(
                                dialect,
                                root,
                                span.span.offset,
                                value,
                                offset,
                            ) {
                                Some((guard, taken)) => TreeSpanValue::Conditional {
                                    guard: self.node(guard),
                                    value: self.node(taken),
                                    offset,
                                },
                                None => TreeSpanValue::Assign {
                                    value: self.node(value),
                                    offset,
                                },
                            }
                        }
                    };
                    self.tree.spans.push(TreeSpan {
                        span: span.span,
                        value,
                    });
                }
                let spans = Run {
                    start: u32::try_from(start).expect("a tree pool holds at most u32::MAX items"),
                    len: u32::try_from(plan.len())
                        .expect("a tree pool holds at most u32::MAX items"),
                };
                ItemData::Write {
                    root,
                    spans,
                    policy,
                }
            }
            WriteTarget::FiniteIndexed {
                index,
                candidates,
                value,
            } => ItemData::FiniteWrite {
                index: self.node(index),
                candidates,
                value: self.node(value),
                policy,
            },
            WriteTarget::SparseIndexed {
                bank,
                key,
                slice,
                value,
            } => ItemData::SparseWrite {
                bank,
                key: self.node(key),
                slice,
                value: self.node(value),
                policy,
            },
        }
    }

    fn control(&self, control: Control<V>) -> Control<TreeNodeId> {
        match control {
            Control::Advance => Control::Advance,
            Control::Terminating => Control::Terminating,
            Control::Jump { target } => Control::Jump {
                target: self.node(target),
            },
            Control::Call { target } => Control::Call {
                target: self.node(target),
            },
            Control::Return { target } => Control::Return {
                target: self.node(target),
            },
            Control::Branch {
                condition,
                taken,
                not_taken,
            } => Control::Branch {
                condition: self.node(condition),
                taken: self.node(taken),
                not_taken: not_taken.map(|value| self.node(value)),
            },
            Control::Dispatch { scrutinee } => Control::Dispatch {
                scrutinee: self.node(scrutinee),
            },
        }
    }
}
