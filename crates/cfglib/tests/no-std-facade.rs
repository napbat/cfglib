//! A consumer on a target without `std`: a firmware analyzer or a wasm
//! module that links `cfglib` for control-flow analysis and nothing else.
//!
//! This target is `#![no_std]` and names only `core` and `alloc` types, so it
//! compiles only if every facade signature it touches does too. The test
//! harness links `std` regardless, which is why the assertions below check
//! behavior rather than the absence of `std`: the compiler enforces that.

#![no_std]

extern crate alloc;

use alloc::borrow::Cow;
use alloc::boxed::Box;
use alloc::collections::{BTreeMap, BTreeSet};
use alloc::string::{String, ToString};
use alloc::vec;
use alloc::vec::Vec;
use core::fmt::Write as _;

use cfglib::{
    BlockId, BuildError, Cfg, CfgBuilder, ClrExceptionRegion, ClrHandler, ClrHandlerKind,
    ControlNode, ControlTree, DenseBits, DenseFanout, DisplayInstr, DominatorTree, EdgeKind,
    EhModel, Fanout, FlowControl, FlowEffect, Graph, Handler, HandlerBody, HandlerKind,
    HandlerTypes, HornClauses, InstrInfo, KeyedGraph, Liveness, NodeId, PostDominatorTree,
    SortedMap, SsaForm, StackGraphError, TextError, TraversalDirection, VectoredExceptionModel,
    VectoredHandlerKind, VectoredHandlerOrder, breadth_first, install_clr_region, parse_cfg_text,
    parse_text, remove_unreachable, shortest_path, simplify, tarjan_scc, topological_sort, verify,
};

/// An instruction a consumer owns: its text, and the variables it reads and
/// writes, named by owned strings.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Statement {
    text: String,
    defs: Vec<String>,
    uses: Vec<String>,
}

impl Statement {
    fn new(text: &str, defs: &[&str], uses: &[&str]) -> Self {
        let own = |names: &[&str]| names.iter().map(|name| (*name).to_string()).collect();
        Self {
            text: text.to_string(),
            defs: own(defs),
            uses: own(uses),
        }
    }
}

impl InstrInfo for Statement {
    type Variable = String;

    fn uses(&self) -> &[String] {
        &self.uses
    }

    fn defs(&self) -> &[String] {
        &self.defs
    }
}

impl DisplayInstr for Statement {
    fn mnemonic(&self) -> Cow<'_, str> {
        Cow::Borrowed(&self.text)
    }
}

/// `if c { x = 1 } else { x = 2 }; use x`
fn diamond() -> (Cfg<Statement>, [BlockId; 4]) {
    let mut cfg = Cfg::<Statement>::new();
    let entry = cfg.entry();
    let [then_block, else_block, merge] = [(); 3].map(|()| cfg.new_block());
    cfg.block_mut(entry)
        .push(Statement::new("if c", &[], &["c"]));
    cfg.block_mut(then_block)
        .push(Statement::new("x = 1", &["x"], &[]));
    cfg.block_mut(else_block)
        .push(Statement::new("x = 2", &["x"], &[]));
    cfg.block_mut(merge)
        .push(Statement::new("use x", &[], &["x"]));
    cfg.add_edge(entry, then_block, EdgeKind::ConditionalTrue);
    cfg.add_edge(entry, else_block, EdgeKind::ConditionalFalse);
    cfg.add_edge(then_block, merge, EdgeKind::Fallthrough);
    cfg.add_edge(else_block, merge, EdgeKind::Fallthrough);
    (cfg, [entry, then_block, else_block, merge])
}

#[test]
fn a_cfg_is_built_verified_and_dominated_with_alloc_types() {
    let (cfg, [entry, then_block, else_block, merge]) = diamond();
    assert!(verify(&cfg).is_ok());
    assert_eq!(cfg.block_count(), 4);
    assert_eq!(cfg.edge_count(), 4);
    assert_eq!(cfg.exit_blocks().collect::<Vec<_>>(), [merge]);

    let dominators = DominatorTree::compute(&cfg);
    assert_eq!(dominators.idom(merge), Some(entry));
    assert!(dominators.dominates(entry, then_block));
    assert!(!dominators.dominates(then_block, merge));
    assert_eq!(
        dominators.nearest_common_dominator(then_block, else_block),
        Some(entry)
    );

    let post = PostDominatorTree::compute(&cfg);
    assert!(post.post_dominates(merge, entry));
    assert!(!post.post_dominates(then_block, entry));
}

#[test]
fn the_text_form_is_an_alloc_string_that_parses_back() {
    let (cfg, _) = diamond();
    let text: String = cfg.to_text();
    assert_eq!(
        text,
        "entry bb0\nbb0:\n    if c\nbb1:\n    x = 1\nbb2:\n    x = 2\nbb3:\n    use x\n\
         bb0 -> bb1 true\nbb0 -> bb2 false\nbb1 -> bb3 fallthrough\nbb2 -> bb3 fallthrough\n"
    );

    let parsed = parse_cfg_text(&text, |line| Ok(Statement::new(line, &[], &[])))
        .expect("the text is the writer's own");
    assert_eq!(parsed.to_text(), text);
    assert_eq!(parsed.block_count(), cfg.block_count());

    let mut streamed = String::new();
    cfg.write_text(&mut streamed).expect("a String never fails");
    assert_eq!(streamed, text);
    assert!(cfg.to_dot().contains("digraph cfg"));
}

#[test]
fn ssa_and_liveness_run_over_owned_string_variables() {
    let (cfg, [entry, then_block, else_block, merge]) = diamond();
    let dominators = DominatorTree::compute(&cfg);
    let ssa = SsaForm::compute(&cfg, &dominators);

    let phi = &ssa.block(merge).phis[0];
    assert_eq!(phi.result.variable, "x");
    assert_eq!(phi.operands().len(), 2);
    assert_ne!(phi.operands()[0].1, phi.operands()[1].1);

    let liveness = Liveness::compute(&cfg);
    let at_merge: &BTreeSet<String> = liveness.live_in(merge);
    assert_eq!(
        at_merge.iter().map(String::as_str).collect::<Vec<_>>(),
        ["x"]
    );
    assert!(liveness.is_live_out(&"x".to_string(), then_block));
    assert!(liveness.is_live_in(&"c".to_string(), entry));
    assert!(!liveness.is_live_in(&"x".to_string(), else_block));
}

#[test]
fn a_loop_is_structured_into_a_control_tree_and_printed_into_a_string() {
    let mut cfg = Cfg::<&'static str>::new();
    let header = cfg.new_block();
    let odd = cfg.new_block();
    let add = cfg.new_block();
    let even = cfg.new_block();
    let step = cfg.new_block();
    let exit = cfg.new_block();
    let statements = [
        (cfg.entry(), "total = 0"),
        (header, "i < n"),
        (odd, "odd(i)"),
        (add, "total += i"),
        (even, "total -= 1"),
        (step, "i += 1"),
        (exit, "return total"),
    ];
    for (block, statement) in statements {
        cfg.block_mut(block).push(statement);
    }
    cfg.add_edge(cfg.entry(), header, EdgeKind::Fallthrough);
    cfg.add_edge(header, odd, EdgeKind::ConditionalTrue);
    cfg.add_edge(header, exit, EdgeKind::ConditionalFalse);
    cfg.add_edge(odd, add, EdgeKind::ConditionalTrue);
    cfg.add_edge(add, step, EdgeKind::Fallthrough);
    cfg.add_edge(odd, even, EdgeKind::ConditionalFalse);
    cfg.add_edge(even, step, EdgeKind::Fallthrough);
    cfg.add_edge(step, header, EdgeKind::Jump);

    let (tree, report) = ControlTree::compute_with_report(&cfg);
    assert!(report.is_fully_structured(), "{report:?}");
    let pseudocode: String = tree
        .to_pseudocode(|sink, point| {
            sink.write_str(cfg.block(point.block).instructions()[point.inst_idx])
        })
        .expect("writing to a String cannot fail");
    // The branch test of an `if` is the construct's own header, so only the
    // straight-line statements and the loop condition are listed as text.
    for (block, statement) in statements {
        if block != odd {
            assert!(
                pseudocode.contains(statement),
                "{statement:?} is missing from\n{pseudocode}"
            );
        }
    }
    assert!(
        tree.preorder()
            .any(|id| matches!(tree.node(id), ControlNode::Return(_))),
        "the function returns"
    );
}

#[test]
fn a_graph_store_runs_traversals_and_components_on_owned_payloads() {
    let mut graph = Graph::<String, String>::new();
    let names = ["parse", "check", "lower", "emit"];
    let ids: Vec<NodeId> = names
        .iter()
        .map(|name| graph.add_node((*name).to_string()))
        .collect();
    for window in ids.windows(2) {
        graph.add_edge(window[0], window[1], "next".to_string());
    }
    assert_eq!(topological_sort(&graph), Some(ids.clone()));
    assert_eq!(
        shortest_path(&graph, ids[0], ids[3], TraversalDirection::Outgoing),
        Some(ids.clone())
    );
    assert_eq!(
        breadth_first(&graph, ids[3], TraversalDirection::Incoming),
        [ids[3], ids[2], ids[1], ids[0]]
    );

    graph.add_edge(ids[3], ids[1], "feedback".to_string());
    assert_eq!(
        topological_sort(&graph),
        None,
        "emit now feeds back into check"
    );
    let components = tarjan_scc(&graph);
    assert_eq!(components.len(), 2, "parse, and the check-lower-emit cycle");

    graph.remove_node(ids[0]);
    let renumbering = graph.compact();
    assert_eq!(renumbering.node(ids[0]), None);
    assert_eq!(graph.node_count(), 3);
    let text: String = graph.to_text();
    let parsed = parse_text(&text).expect("the writer emits its own grammar");
    assert_eq!(parsed.to_text(), text);
}

#[test]
fn keyed_graphs_fanouts_and_clause_sets_use_alloc_collections() {
    let mut graph = KeyedGraph::<String, String, u32>::new();
    let a = graph.intern(&"pkg::a".to_string());
    let b = graph.intern(&"pkg::b".to_string());
    graph.add_edge(a, b, 1);
    assert_eq!(graph.node_id(&"pkg::b".to_string()), Some(b));
    let keys: BTreeMap<&String, NodeId> = graph.keys().collect();
    assert_eq!(keys.len(), 2);

    let by_importer = Fanout::from_pairs(vec![("b", "a"), ("c", "a"), ("b", "d")]);
    assert_eq!(by_importer.get(&"b"), ["a", "d"]);
    let dense = DenseFanout::from_pairs(
        3,
        [(0_usize, 'x'), (2, 'y'), (0, 'z')].map(|(key, value)| (NodeId::from_index(key), value)),
    );
    assert_eq!(dense.get(NodeId::from_index(0)), ['x', 'z']);
    let counts = SortedMap::from_pairs(vec![("a", 1_u32), ("b", 2), ("a", 3)], |total, next| {
        *total += next;
    });
    assert_eq!(counts.get(&"a"), Some(&4));

    let mut bits = DenseBits::new(70);
    bits.insert(3);
    bits.insert(69);
    assert_eq!(bits.ones().collect::<Vec<_>>(), [3, 69]);

    let mut clauses = HornClauses::new(3);
    clauses.add_clause(0, &[1]);
    clauses.add_clause(1, &[]);
    assert_eq!(clauses.derivable(), [true, true, false]);
}

/// A structured instruction stream a consumer's decoder produces.
#[derive(Debug, Clone, PartialEq, Eq)]
enum Step {
    Plain(&'static str),
    If,
    Else,
    EndIf,
    Return,
}

impl FlowControl for Step {
    fn flow_effect(&self) -> FlowEffect {
        match self {
            Self::Plain(_) => FlowEffect::Fallthrough,
            Self::If => FlowEffect::ConditionalOpen,
            Self::Else => FlowEffect::ConditionalAlternate,
            Self::EndIf => FlowEffect::ConditionalClose,
            Self::Return => FlowEffect::Return,
        }
    }
}

#[test]
fn a_structured_stream_builds_and_the_cleanup_passes_report_alloc_results() {
    let mut cfg = CfgBuilder::build(vec![
        Step::Plain("a"),
        Step::If,
        Step::Plain("b"),
        Step::Else,
        Step::Plain("c"),
        Step::EndIf,
        Step::Return,
        Step::Plain("unreachable"),
    ])
    .expect("the stream is balanced");
    assert!(verify(&cfg).is_ok());

    let before = cfg.block_count();
    let removed: usize = remove_unreachable(&mut cfg);
    assert_eq!(
        removed, 1,
        "the code after the return is the one unreachable block"
    );
    assert_eq!(cfg.block_count(), before - 1);
    let _merged: usize = simplify(&mut cfg);
    cfg.compact();
    assert!(cfg.is_compact());
    assert!(verify(&cfg).is_ok());
    assert!(
        cfg.blocks()
            .all(|block| !block.instructions().contains(&Step::Plain("unreachable")))
    );

    let unbalanced: Result<Cfg<Step>, BuildError> = CfgBuilder::build(vec![Step::EndIf]);
    assert!(unbalanced.is_err());
}

#[test]
fn exception_regions_and_models_are_described_with_alloc_collections() {
    let (mut cfg, [entry, then_block, else_block, merge]) = diamond();
    let mut types = HandlerTypes::<String>::new();
    let region = install_clr_region(
        &mut cfg,
        &mut types,
        ClrExceptionRegion {
            protected_blocks: BTreeSet::from([entry, then_block]),
            handlers: vec![ClrHandler {
                entry: else_block,
                body: HandlerBody::known([else_block]),
                kind: ClrHandlerKind::Catch {
                    ty: "System.IO.IOException".to_string(),
                },
            }],
            parent: None,
        },
    );
    assert_eq!(
        cfg.protecting_region(then_block).map(|found| found.id),
        Some(region)
    );
    assert_eq!(cfg.protecting_region(merge).map(|found| found.id), None);
    assert_eq!(types.len(), 1);

    cfg.add_edge(then_block, else_block, EdgeKind::ExceptionUnwind);
    let model = EhModel::compute(&cfg);
    assert_eq!(model.landing_pads(), [else_block]);
    assert_eq!(
        model
            .protected_by(else_block)
            .iter()
            .copied()
            .collect::<Vec<_>>(),
        [entry, then_block]
    );
    let handlers: Vec<&Handler> = cfg.regions()[0].handlers.iter().collect();
    assert!(matches!(handlers[0].kind, HandlerKind::Catch));

    let mut vectored = VectoredExceptionModel::<String>::new();
    vectored.register(
        VectoredHandlerKind::Exception,
        VectoredHandlerOrder::First,
        "veh".to_string(),
    );
    assert_eq!(vectored.len(), 1);
}

/// Every error type of the facade implements `core::error::Error`, not the
/// `std` trait, so a `no_std` consumer can box and chain them.
fn assert_core_error<E: core::error::Error + 'static>(error: E) -> Box<dyn core::error::Error> {
    Box::new(error)
}

#[test]
fn facade_errors_implement_the_core_error_trait() {
    let text_error: TextError = parse_text("n0 a\\q\n").expect_err("an unknown escape");
    let boxed = assert_core_error(text_error);
    assert!(boxed.to_string().starts_with("line 1"), "{boxed}");

    let build_error = CfgBuilder::build(vec![Step::EndIf]).expect_err("an unbalanced stream");
    let _ = assert_core_error(build_error);
    let _ = assert_core_error(StackGraphError::EdgeFromJumpToScope);
    let split = Cfg::<Statement>::new().split_block_at_points(BlockId::from_index(0), &[5]);
    let _ = assert_core_error(split.expect_err("a point past the block"));
}

#[test]
fn rtl_value_types_are_plain_core_types() {
    use cfglib::ir::rtl::Bits;

    let word = Bits::new(32).expect("32 bits is a valid width");
    assert_eq!(word.width(), 32);
    assert_eq!(word.words(), 1);
    assert_eq!(Bits::new(0), None);
    assert_eq!(Bits::new(Bits::MAX_WIDTH + 1), None);
    let mut description = String::new();
    write!(description, "{word:?}").expect("a String never fails");
    assert!(description.contains("32"), "{description}");
}
