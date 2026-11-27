//! Structuring a CFG into a control tree, and modeling its exception
//! handling.

use cfglib::{
    BlockId, Cfg, ControlTree, EdgeKind, EhBlockKind, EhModel, Handler, HandlerBody, HandlerKind,
    Region,
};

use crate::harness::{BenchmarkSuite, benchmark_case};

/// The try/catch constructs of the chain the cases read.
const TRY_COUNT: usize = 512;

/// `count` try/catch constructs in sequence. Each protects a diamond; its
/// catch pad runs one more block, and both rejoin before the next construct.
fn try_chain(count: usize) -> Cfg<u32> {
    let mut cfg = Cfg::new();
    let mut previous = cfg.entry();
    cfg.block_mut(previous).push(0);
    for index in 0..count {
        let blocks: [BlockId; 7] = std::array::from_fn(|_| cfg.new_block());
        let [head, then_arm, else_arm, merge, pad, pad_tail, join] = blocks;
        for block in blocks {
            cfg.block_mut(block)
                .push(u32::try_from(index).expect("the chain is short"));
        }
        cfg.add_edge(previous, head, EdgeKind::Fallthrough);
        cfg.add_edge(head, then_arm, EdgeKind::ConditionalTrue);
        cfg.add_edge(head, else_arm, EdgeKind::ConditionalFalse);
        cfg.add_edge(then_arm, merge, EdgeKind::Fallthrough);
        cfg.add_edge(else_arm, merge, EdgeKind::Fallthrough);
        cfg.add_edge(merge, join, EdgeKind::Fallthrough);
        cfg.add_edge(head, pad, EdgeKind::ExceptionUnwind);
        cfg.add_edge(pad, pad_tail, EdgeKind::Fallthrough);
        cfg.add_edge(pad_tail, join, EdgeKind::Fallthrough);
        cfg.add_region(Region {
            protected_blocks: [head, then_arm, else_arm, merge].into_iter().collect(),
            handlers: vec![Handler {
                entry: pad,
                body: HandlerBody::known([pad, pad_tail]),
                kind: HandlerKind::Catch,
            }],
            parent: None,
        })
        .expect("the chain's handler blocks are live");
        previous = join;
    }
    cfg
}

pub(super) fn register(suite: &mut BenchmarkSuite<'_>, cfg: &Cfg<u32>) {
    let chain = try_chain(TRY_COUNT);
    benchmark_case!(
        suite,
        "control_tree_compute_branchy",
        || ControlTree::compute(cfg),
        |tree: &ControlTree| assert!(tree.node_count() > 0)
    );
    benchmark_case!(
        suite,
        "control_tree_compute_try_chain",
        || ControlTree::compute(&chain),
        |tree: &ControlTree| assert!(tree.node_count() > TRY_COUNT)
    );
    benchmark_case!(
        suite,
        "eh_model_compute_try_chain",
        || EhModel::compute(&chain),
        |model: &EhModel| {
            assert_eq!(model.blocks_of(EhBlockKind::LandingPad).count(), TRY_COUNT);
        }
    );
}
