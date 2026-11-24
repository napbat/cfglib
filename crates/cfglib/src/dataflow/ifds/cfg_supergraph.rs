//! A [`Supergraph`] over keyed CFGs whose calls come from [`CallInfo`].

extern crate alloc;
use alloc::collections::BTreeMap;
use alloc::vec::Vec;

use super::Supergraph;
use crate::block::BlockId;
use crate::dataflow::cfg_view::CfgView;
use crate::dataflow::program_point::ProgramPoint;
use crate::flow::CallInfo;
use crate::graph::view::GraphView;

/// An unset slot: a removed block, or an instruction that calls no listed
/// function.
const NONE: usize = usize::MAX;

/// The supergraph of a set of keyed CFGs, the input of
/// [`call_graph`](crate::call_graph), with one node per program point.
///
/// Each listed CFG is a procedure, numbered in input order. A block with `k`
/// instructions has the `k + 1` nodes `(block, 0)` to `(block, k)`: node
/// `(block, i)` is the point before instruction `i`, and the edge to
/// `(block, i + 1)` executes it, so a flow function reads the instruction
/// of its `from` node with [`instruction`](Self::instruction). The node at a
/// block's end flows to the start of each successor block.
///
/// - The start of a procedure is the start of its entry block.
/// - An exit is the end of a block without successors.
/// - A call node is a point whose instruction's [`CallInfo::callee`] names a
///   listed key; its return site is the next point. A call to an unknown or
///   unresolved callee is an ordinary instruction. A key listed more than
///   once resolves to its first procedure.
/// - A tail call returns to the point after it, which is normally the end
///   of a block without successors and so an exit: the callee's facts flow
///   on to the caller's caller.
///
/// # Examples
///
/// ```
/// use cfglib::{CallInfo, Cfg, CfgSupergraph, Supergraph};
///
/// #[derive(Debug)]
/// enum Op {
///     Work,
///     Call(&'static str),
/// }
/// impl CallInfo for Op {
///     type Callee = &'static str;
///     fn callee(&self) -> Option<&'static str> {
///         match self {
///             Op::Call(name) => Some(name),
///             Op::Work => None,
///         }
///     }
/// }
///
/// let mut main = Cfg::<Op>::new();
/// main.block_mut(main.entry()).push(Op::Call("helper"));
/// let mut helper = Cfg::<Op>::new();
/// helper.block_mut(helper.entry()).push(Op::Work);
///
/// let functions = [("main", &main), ("helper", &helper)];
/// let graph = CfgSupergraph::new(&functions);
/// // main: before the call, after it; helper: before the work, after it.
/// assert_eq!(graph.node_bound(), 4);
/// let mut callees = Vec::new();
/// graph.callees(graph.start(0), &mut callees);
/// assert_eq!(callees, [1]);
/// assert!(graph.is_exit(3));
/// ```
#[derive(Debug, Clone)]
pub struct CfgSupergraph<'g, V> {
    views: Vec<&'g V>,
    /// Per procedure, where its block table starts in `block_first`.
    block_base: Vec<usize>,
    /// Per procedure and block slot, the block's first node.
    block_first: Vec<usize>,
    /// Per node, its procedure, block and position.
    points: Vec<(usize, BlockId, usize)>,
    /// Per node, the procedure its instruction calls.
    calls: Vec<usize>,
}

impl<'g, V> CfgSupergraph<'g, V>
where
    V: CfgView,
    V::Inst: CallInfo,
{
    /// The supergraph of `functions`, each a callee key and its CFG.
    #[must_use]
    pub fn new(functions: &[(<V::Inst as CallInfo>::Callee, &'g V)]) -> Self {
        let mut procs_by_key = BTreeMap::new();
        for (proc, (key, _)) in functions.iter().enumerate() {
            procs_by_key.entry(key).or_insert(proc);
        }
        let mut graph = Self {
            views: Vec::with_capacity(functions.len()),
            block_base: Vec::with_capacity(functions.len()),
            block_first: Vec::new(),
            points: Vec::new(),
            calls: Vec::new(),
        };
        for (proc, &(_, view)) in functions.iter().enumerate() {
            graph.views.push(view);
            let base = graph.block_first.len();
            graph.block_base.push(base);
            graph
                .block_first
                .resize(base + GraphView::node_bound(view), NONE);
            for block in view.node_ids() {
                graph.block_first[base + block.index()] = graph.points.len();
                let instructions = view.instructions(block);
                for (position, instruction) in instructions.iter().enumerate() {
                    graph.points.push((proc, block, position));
                    let callee = instruction
                        .callee()
                        .and_then(|callee| procs_by_key.get(&callee).copied());
                    graph.calls.push(callee.unwrap_or(NONE));
                }
                graph.points.push((proc, block, instructions.len()));
                graph.calls.push(NONE);
            }
        }
        graph
    }
}

impl<'g, V: CfgView> CfgSupergraph<'g, V> {
    /// The procedure of `node` and its program point; an instruction index
    /// equal to the block's instruction count is the end of the block.
    ///
    /// # Panics
    ///
    /// Panics when `node` is not below the node bound.
    #[must_use]
    pub fn point(&self, node: usize) -> (usize, ProgramPoint) {
        let (proc, block, inst_idx) = self.points[node];
        (proc, ProgramPoint { block, inst_idx })
    }

    /// The node of `point` in procedure `proc`, the inverse of
    /// [`point`](Self::point).
    ///
    /// # Panics
    ///
    /// Panics when `proc` is out of range, or `point` names a block the
    /// procedure does not have or a position past the block's end.
    #[must_use]
    pub fn node(&self, proc: usize, point: ProgramPoint) -> usize {
        let first = self.block_first[self.block_base[proc] + point.block.index()];
        assert!(first != NONE, "the block is not in the procedure");
        assert!(
            point.inst_idx <= self.views[proc].instructions(point.block).len(),
            "the position is past the end of the block"
        );
        first + point.inst_idx
    }

    /// The instruction that the edges leaving `node` execute; `None` at the
    /// end of a block.
    ///
    /// # Panics
    ///
    /// Panics when `node` is not below the node bound.
    #[must_use]
    pub fn instruction(&self, node: usize) -> Option<&'g V::Inst> {
        let (proc, block, position) = self.points[node];
        self.views[proc].instructions(block).get(position)
    }

    /// The first node of `block` in procedure `proc`.
    fn first(&self, proc: usize, block: BlockId) -> usize {
        self.block_first[self.block_base[proc] + block.index()]
    }
}

impl<V: CfgView> Supergraph for CfgSupergraph<'_, V> {
    type Node = usize;
    type Proc = usize;

    fn node_bound(&self) -> usize {
        self.points.len()
    }

    fn proc_bound(&self) -> usize {
        self.views.len()
    }

    fn proc_of(&self, node: usize) -> usize {
        self.points[node].0
    }

    fn start(&self, proc: usize) -> usize {
        self.first(proc, self.views[proc].root())
    }

    fn is_exit(&self, node: usize) -> bool {
        let (proc, block, position) = self.points[node];
        let view = self.views[proc];
        position == view.instructions(block).len() && view.successors(block).next().is_none()
    }

    fn successors(&self, node: usize, out: &mut Vec<usize>) {
        let (proc, block, position) = self.points[node];
        let view = self.views[proc];
        if position < view.instructions(block).len() {
            out.push(node + 1);
        } else {
            out.extend(view.successors(block).map(|next| self.first(proc, next)));
        }
    }

    fn callees(&self, node: usize, out: &mut Vec<usize>) {
        if self.calls[node] != NONE {
            out.push(self.calls[node]);
        }
    }
}
