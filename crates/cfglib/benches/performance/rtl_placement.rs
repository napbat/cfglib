//! A synthetic register-transfer function and the oracle of its parameter
//! placement.

use cfglib::ir::rtl::{
    Action, BitSlice, Bits, Control, Dialect, FunctionView, ParamPlacement, Place, RecordView,
    RootLiveness, RootTable, Selection, SpanAction, Statement, WriteTarget,
};
use cfglib::{DenseId, DominatorTree, GraphView, RootedView};

/// How many roots the synthetic dialect names.
const ROOT_COUNT: u8 = 16;

/// The marker of the synthetic dialect: sixteen 64-bit roots and nothing else.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(super) struct Synthetic;

impl Dialect for Synthetic {
    type Root = u8;
    type Operator = ();
    type EffectOp = ();
    type WritePolicy = ();
    type Outcome = ();
    type NondetDomain = ();

    fn root_width(_root: u8) -> Bits {
        word()
    }
}

fn word() -> Bits {
    Bits::new(64).expect("64 bits is a legal width")
}

/// One block's lone instruction: read one root, then overwrite another.
pub(super) struct SyntheticRecord {
    read: u8,
    written: u8,
    plan: [SpanAction<Synthetic, u32>; 1],
}

impl SyntheticRecord {
    fn new(block: usize) -> Self {
        let root = |value: usize| {
            u8::try_from(value % usize::from(ROOT_COUNT)).expect("below the root count")
        };
        Self {
            read: root(block * 3 + 1),
            written: root(block + block / 2),
            plan: [SpanAction {
                span: BitSlice::whole(word()),
                action: Action::TakeInput {
                    value: 0,
                    offset: 0,
                },
            }],
        }
    }
}

impl RecordView<Synthetic> for &SyntheticRecord {
    type Value = u32;

    fn statements(&self) -> impl ExactSizeIterator<Item = Statement<'_, Synthetic, u32>> + '_ {
        [
            Statement::Read {
                result: 0,
                source: Selection::Direct(Place::whole(self.read)),
            },
            Statement::Write {
                target: WriteTarget::Root {
                    root: self.written,
                    plan: &self.plan,
                },
                policy: (),
            },
        ]
        .into_iter()
    }

    fn control(&self) -> Control<u32> {
        Control::Advance
    }

    fn width(&self, _value: u32) -> Bits {
        word()
    }
}

/// A function of one-instruction blocks whose flow is the branchy fixture of
/// the directed cases: a fallthrough edge, a skip edge from every even block,
/// and a back edge every thirty-two blocks.
pub(super) struct SyntheticFunction {
    successors: Vec<Vec<u32>>,
    predecessors: Vec<Vec<u32>>,
    /// The lone instruction of each block, which shares the block's index.
    instructions: Vec<u32>,
    records: Vec<SyntheticRecord>,
}

impl SyntheticFunction {
    pub(super) fn new(block_count: usize) -> Self {
        assert!(block_count > 2);
        let mut successors = vec![Vec::new(); block_count];
        let mut predecessors = vec![Vec::new(); block_count];
        let mut flow = |from: usize, to: usize| {
            successors[from].push(fixture_id(to));
            predecessors[to].push(fixture_id(from));
        };
        for block in 0..block_count - 1 {
            flow(block, block + 1);
        }
        for block in (0..block_count - 2).step_by(2) {
            flow(block, block + 2);
        }
        for block in (32..block_count).step_by(32) {
            flow(block, block - 16);
        }
        Self {
            successors,
            predecessors,
            instructions: (0..block_count).map(fixture_id).collect(),
            records: (0..block_count).map(SyntheticRecord::new).collect(),
        }
    }
}

fn fixture_id(index: usize) -> u32 {
    u32::try_from(index).expect("fixture index fits in u32")
}

impl GraphView for SyntheticFunction {
    type NodeId = u32;

    fn node_bound(&self) -> usize {
        self.successors.len()
    }

    fn node_ids(&self) -> impl Iterator<Item = u32> + '_ {
        0..fixture_id(self.successors.len())
    }

    fn successors(&self, node: u32) -> impl Iterator<Item = u32> + '_ {
        self.successors[node.index()].iter().copied()
    }

    fn predecessors(&self, node: u32) -> impl Iterator<Item = u32> + '_ {
        self.predecessors[node.index()].iter().copied()
    }
}

impl RootedView for SyntheticFunction {
    fn root(&self) -> u32 {
        0
    }
}

impl FunctionView<Synthetic> for SyntheticFunction {
    type Inst = u32;
    type Value = u32;
    type Record<'a> = &'a SyntheticRecord;

    fn inst_bound(&self) -> usize {
        self.records.len()
    }

    fn value_bound(&self) -> usize {
        1
    }

    fn instructions(&self, block: u32) -> &[u32] {
        core::slice::from_ref(&self.instructions[block.index()])
    }

    fn record(&self, inst: u32) -> &SyntheticRecord {
        &self.records[inst.index()]
    }
}

/// Checks `placement` against the iterated frontier that the dominator tree
/// states block by block, pruned by liveness.
pub(super) fn assert_param_placement(
    placement: &ParamPlacement<Synthetic>,
    function: &SyntheticFunction,
    dominators: &DominatorTree<u32>,
    roots: &RootTable<Synthetic>,
    liveness: &RootLiveness,
) {
    let block_count = function.node_bound();
    let mut expected = vec![Vec::new(); block_count];
    let mut frontier = Vec::new();
    for index in 0..roots.len() {
        let root = roots.root(index);
        let mut queued = vec![false; block_count];
        let mut needs_param = vec![false; block_count];
        let mut work = vec![function.root()];
        queued[function.root().index()] = true;
        for block in function.node_ids() {
            let writes = function.records[block.index()].written == root;
            if writes && dominators.is_reachable(block) && !queued[block.index()] {
                queued[block.index()] = true;
                work.push(block);
            }
        }
        while let Some(block) = work.pop() {
            dominators.frontier(function, block, &mut frontier);
            for &join in &frontier {
                needs_param[join.index()] = true;
                if !queued[join.index()] {
                    queued[join.index()] = true;
                    work.push(join);
                }
            }
        }
        for block in function.node_ids() {
            if needs_param[block.index()] && liveness.live_in(block).get(index) {
                expected[block.index()].push(root);
            }
        }
    }
    let mut param_count = 0;
    for block in function.node_ids() {
        assert_eq!(placement.roots_at(block), expected[block.index()]);
        param_count += expected[block.index()].len();
    }
    assert_eq!(placement.param_count(), param_count);
    assert!(param_count > block_count / 8, "the fixture must place work");
}
