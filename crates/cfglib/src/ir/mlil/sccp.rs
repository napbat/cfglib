//! Sparse conditional constant propagation.

extern crate alloc;

use alloc::vec;
use alloc::vec::Vec;

use super::uses::{Use, UseIndex};
use crate::DenseBits;
use crate::graph::view::DenseId;
use crate::ir::mlil::{AnalysisDialect, FunctionView, Lattice};

/// What sparse conditional constant propagation learned about a function.
///
/// The algorithm of Wegman and Zadeck keeps the value lattice and the
/// executable control flow together. A block is executable when an executable
/// edge reaches it or when it is the entry block. A branch that
/// [`AnalysisDialect::fold_branch`] decides makes only its taken edge
/// executable. A branch that it does not decide, or decides with a position
/// that is not a successor record, makes every edge executable. A block
/// parameter takes the meet of the arguments on its executable incoming
/// edges.
///
/// A value that no executable code defines keeps [`Lattice::Undefined`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SccpFacts<C> {
    /// The lattice value of each value index.
    values: Vec<Lattice<C>>,
    /// The executable blocks, by block index.
    blocks: DenseBits,
    /// Where the edges of each block start in `edges`. One longer than the
    /// node bound.
    edge_starts: Vec<usize>,
    /// The executable edges: one flag for each successor record of each
    /// block, in block order.
    edges: DenseBits,
}

impl<C: Clone + Eq> SccpFacts<C> {
    /// Propagates constants through `function`.
    ///
    /// The parameters of the entry block are the parameters of the function,
    /// so they start as [`Lattice::Varying`]. `uses` is the use index of
    /// `function`.
    ///
    /// # Panics
    ///
    /// Panics when `uses` was not built from `function`.
    #[must_use]
    pub fn compute<D, F>(function: &F, uses: &UseIndex<F::Inst, F::Block>) -> Self
    where
        D: AnalysisDialect<Constant = C>,
        F: FunctionView<D>,
    {
        let bound = function.node_bound();
        let mut edge_starts = vec![0; bound + 1];
        for block in function.node_ids() {
            edge_starts[block.index() + 1] = FunctionView::<D>::successors(function, block).len();
        }
        for index in 0..bound {
            edge_starts[index + 1] += edge_starts[index];
        }
        let edge_count = edge_starts[bound];

        let mut solver = Solver::<D, F> {
            function,
            uses,
            facts: Self {
                values: vec![Lattice::Undefined; function.value_bound()],
                blocks: DenseBits::new(bound),
                edge_starts,
                edges: DenseBits::new(edge_count),
            },
            work: Vec::new(),
            args: Vec::new(),
            results: Vec::new(),
        };
        solver.run();
        solver.facts
    }

    /// Returns the lattice value of `value`.
    ///
    /// # Panics
    ///
    /// Panics when `value` is not below the value bound of the view that
    /// computed the facts.
    #[must_use]
    pub fn value<V: DenseId>(&self, value: V) -> &Lattice<C> {
        &self.values[value.index()]
    }

    /// Returns whether some execution can reach `block`.
    ///
    /// # Panics
    ///
    /// Panics when `block` is not below the node bound of the view that
    /// computed the facts.
    #[must_use]
    pub fn is_block_executable<B: DenseId>(&self, block: B) -> bool {
        self.blocks.get(block.index())
    }

    /// Returns the executable blocks in ascending order.
    pub fn executable_blocks<B: DenseId>(&self) -> impl Iterator<Item = B> {
        self.blocks.ones().map(B::from_index)
    }

    /// Returns whether some execution can take successor record `successor`
    /// of the terminator of `block`.
    ///
    /// # Panics
    ///
    /// Panics when `block` is not below the node bound of the view that
    /// computed the facts, or when `successor` is not a successor record of
    /// `block`.
    #[must_use]
    pub fn is_edge_executable<B: DenseId>(&self, block: B, successor: usize) -> bool {
        let start = self.edge_starts[block.index()];
        assert!(
            start + successor < self.edge_starts[block.index() + 1],
            "the block has no such successor record"
        );
        self.edges.get(start + successor)
    }
}

/// One unit of pending work.
enum Work<I, B> {
    /// Evaluate an instruction again.
    Inst(I),
    /// Pass every argument of an edge that just became executable, and mark
    /// its target executable.
    Edge {
        /// The block whose terminator holds the successor record.
        block: B,
        /// The position of the successor record.
        successor: usize,
    },
    /// Pass one argument of an executable edge again.
    Arg {
        /// The block whose terminator holds the successor record.
        block: B,
        /// The position of the successor record.
        successor: usize,
        /// The position of the argument.
        position: usize,
    },
}

/// The state of one propagation.
struct Solver<'a, D: AnalysisDialect, F: FunctionView<D>> {
    /// The function.
    function: &'a F,
    /// The uses of the values of the function.
    uses: &'a UseIndex<F::Inst, F::Block>,
    /// The facts that the solver fills.
    facts: SccpFacts<D::Constant>,
    /// The pending work.
    work: Vec<Work<F::Inst, F::Block>>,
    /// The lattice values of the operands of the instruction in hand.
    args: Vec<Lattice<D::Constant>>,
    /// The lattice values of the results of the instruction in hand.
    results: Vec<Lattice<D::Constant>>,
}

impl<D: AnalysisDialect, F: FunctionView<D>> Solver<'_, D, F> {
    /// Runs the propagation to its fixpoint.
    ///
    /// A value moves down the lattice at most twice, from undefined to a
    /// constant to varying, so the work is finite.
    fn run(&mut self) {
        let function = self.function;
        let entry = function.root();
        for &param in function.params(entry) {
            self.lower(param, &Lattice::Varying);
        }
        self.enter(entry);
        while let Some(work) = self.work.pop() {
            match work {
                Work::Inst(inst) => self.visit(inst),
                Work::Edge { block, successor } => self.pass_edge(block, successor),
                Work::Arg {
                    block,
                    successor,
                    position,
                } => self.pass_arg(block, successor, position),
            }
        }
    }

    /// Makes `block` executable and queues its instructions in order.
    fn enter(&mut self, block: F::Block) {
        if self.facts.blocks.insert(block.index()) {
            for &inst in self.function.insts(block).iter().rev() {
                self.work.push(Work::Inst(inst));
            }
        }
    }

    /// Evaluates `inst`, and decides the branch when it is a terminator.
    fn visit(&mut self, inst: F::Inst) {
        let function = self.function;
        let block = function.block_of(inst);
        if !self.facts.blocks.get(block.index()) {
            return;
        }
        let op = function.op(inst);
        self.args.clear();
        let values = &self.facts.values;
        self.args.extend(
            function
                .operands(inst)
                .iter()
                .map(|operand| values[operand.index()].clone()),
        );

        let outputs = function.results(inst);
        if !outputs.is_empty() {
            let mut results = core::mem::take(&mut self.results);
            results.clear();
            results.resize(outputs.len(), Lattice::Varying);
            D::fold(op, &self.args, &mut results);
            for (&output, folded) in outputs.iter().zip(&results) {
                self.lower(output, folded);
            }
            self.results = results;
        }

        if function.insts(block).last() == Some(&inst) {
            self.decide(block, op);
        }
    }

    /// Makes the taken successor records of the terminator `op` of `block`
    /// executable. The operands of `op` are in `self.args`.
    fn decide(&mut self, block: F::Block, op: &D::Op) {
        let function = self.function;
        let records = FunctionView::<D>::successors(function, block);
        let count = records.len();
        if count == 0 {
            return;
        }
        let taken = D::fold_branch(op, &self.args, records.map(|record| record.edge));
        match taken {
            Some(successor) if successor < count => self.take(block, successor),
            _ => {
                for successor in 0..count {
                    self.take(block, successor);
                }
            }
        }
    }

    /// Makes successor record `successor` of `block` executable.
    fn take(&mut self, block: F::Block, successor: usize) {
        let slot = self.facts.edge_starts[block.index()] + successor;
        if self.facts.edges.insert(slot) {
            self.work.push(Work::Edge { block, successor });
        }
    }

    /// Passes the arguments of a new executable edge, then enters its target.
    fn pass_edge(&mut self, block: F::Block, successor: usize) {
        let function = self.function;
        let record = FunctionView::<D>::successors(function, block)
            .nth(successor)
            .expect("an executable edge is a successor record");
        for (position, &arg) in record.args.iter().enumerate() {
            self.pass(function.params(record.target)[position], arg);
        }
        self.enter(record.target);
    }

    /// Passes one argument of an executable edge again.
    fn pass_arg(&mut self, block: F::Block, successor: usize, position: usize) {
        let function = self.function;
        let record = FunctionView::<D>::successors(function, block)
            .nth(successor)
            .expect("an executable edge is a successor record");
        self.pass(
            function.params(record.target)[position],
            record.args[position],
        );
    }

    /// Meets the lattice value of `arg` into the parameter `param`.
    fn pass(&mut self, param: F::Value, arg: F::Value) {
        let incoming = self.facts.values[arg.index()].clone();
        self.lower(param, &incoming);
    }

    /// Meets `incoming` into `value`, and queues the users of `value` when it
    /// changed.
    fn lower(&mut self, value: F::Value, incoming: &Lattice<D::Constant>) {
        let Some(merged) = meet(&self.facts.values[value.index()], incoming) else {
            return;
        };
        self.facts.values[value.index()] = merged;
        for &value_use in self.uses.uses(value) {
            match value_use {
                Use::Operand { inst, .. } => self.work.push(Work::Inst(inst)),
                Use::Edge {
                    block,
                    successor,
                    position,
                } => {
                    let successor = usize::try_from(successor).expect("a position fits in usize");
                    let slot = self.facts.edge_starts[block.index()] + successor;
                    if self.facts.edges.get(slot) {
                        self.work.push(Work::Arg {
                            block,
                            successor,
                            position: usize::try_from(position).expect("a position fits in usize"),
                        });
                    }
                }
            }
        }
    }
}

/// Returns the meet of `old` and `incoming` when it differs from `old`.
///
/// The order of the lattice is undefined above every constant above varying.
/// A value only moves down, so a dialect whose rules are not monotone still
/// reaches a fixpoint.
fn meet<C: Clone + Eq>(old: &Lattice<C>, incoming: &Lattice<C>) -> Option<Lattice<C>> {
    match (old, incoming) {
        (Lattice::Varying, _) | (_, Lattice::Undefined) => None,
        (Lattice::Undefined, other) => Some(other.clone()),
        (Lattice::Constant(left), Lattice::Constant(right)) => {
            (left != right).then_some(Lattice::Varying)
        }
        (Lattice::Constant(_), Lattice::Varying) => Some(Lattice::Varying),
    }
}
