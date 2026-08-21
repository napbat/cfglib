//! Executable specification: a source-language frontend (symtree-shaped)
//! drives the whole stack with consumer-typed axes everywhere.
//!
//! Variables are interned symbol names (non-`Copy`, non-numeric), constants
//! are source literals (strings, bools, ints), operators are an enum, branch
//! targets are CST-node tokens, and rendering needs only [`DisplayInstr`].

use std::borrow::Cow;
use std::collections::BTreeMap;

use cfglib::{
    Cfg, CfgBuilder, ConstValue, ConstantFolder, DisplayInstr, DominatorTree, EdgeKind, ExprInstr,
    ExprNode, FlowControl, FlowEffect, InstrInfo, JumpTable, JumpTargets, Liveness, ReachingDefs,
    build_ssa, constant_propagation, detect_switch_tables, recover_block_expressions,
    recover_switch_tables, resolve_jump_edges, verify,
};

/// A source symbol identity — deliberately non-`Copy`, non-numeric.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct SymbolId(String);

fn sym(name: &str) -> SymbolId {
    SymbolId(String::from(name))
}

/// A source literal — impossible to express with `i64`-only constants.
#[derive(Debug, Clone, PartialEq, Eq)]
enum SourceConst {
    Int(i64),
    Str(String),
    Bool(bool),
}

/// A source operator identity.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SourceOp {
    Concat,
}

/// One statement sliced out of a CST.
#[derive(Debug, Clone)]
struct Stmt {
    /// Pre-order CST node id (anchor back into the syntax tree).
    node: u32,
    text: &'static str,
    effect: FlowEffect,
    defs: Vec<SymbolId>,
    uses: Vec<SymbolId>,
    constant: Option<SourceConst>,
    op: Option<SourceOp>,
    jump_target: Option<String>,
    label: Option<String>,
}

impl Stmt {
    fn new(node: u32, text: &'static str) -> Self {
        Stmt {
            node,
            text,
            effect: FlowEffect::Fallthrough,
            defs: Vec::new(),
            uses: Vec::new(),
            constant: None,
            op: None,
            jump_target: None,
            label: None,
        }
    }

    fn defines(mut self, name: &str) -> Self {
        self.defs.push(sym(name));
        self
    }

    fn reads(mut self, name: &str) -> Self {
        self.uses.push(sym(name));
        self
    }

    fn constant(mut self, value: SourceConst) -> Self {
        self.constant = Some(value);
        self
    }

    fn operating(mut self, op: SourceOp) -> Self {
        self.op = Some(op);
        self
    }
}

impl FlowControl for Stmt {
    fn flow_effect(&self) -> FlowEffect {
        self.effect
    }
}

impl DisplayInstr for Stmt {
    fn mnemonic(&self) -> Cow<'_, str> {
        Cow::Borrowed(self.text)
    }
}

impl InstrInfo for Stmt {
    type Variable = SymbolId;

    fn uses(&self) -> &[SymbolId] {
        &self.uses
    }

    fn defs(&self) -> &[SymbolId] {
        &self.defs
    }
}

impl ConstantFolder for Stmt {
    type Const = SourceConst;

    fn fold_constant(
        &self,
        _known: &BTreeMap<SymbolId, SourceConst>,
    ) -> Option<(SymbolId, SourceConst)> {
        let value = self.constant.clone()?;
        Some((self.defs.first()?.clone(), value))
    }
}

impl ExprInstr for Stmt {
    type Operator = SourceOp;
    type Const = SourceConst;

    fn as_expr(&self) -> Option<(SourceOp, &[SymbolId])> {
        self.op.map(|op| (op, self.uses.as_slice()))
    }

    fn as_const(&self) -> Option<SourceConst> {
        self.constant.clone()
    }
}

impl JumpTargets for Stmt {
    type Target = String;

    fn jump_target(&self) -> Option<String> {
        self.jump_target.clone()
    }

    fn label(&self) -> Option<String> {
        self.label.clone()
    }
}

#[test]
fn direct_construction_dominators_reaching_and_liveness_over_symbols() {
    // fn f(a) { x := a; x := 7; return x }  — built structurally, no builder.
    let mut cfg = Cfg::<Stmt>::new();
    let ret = cfg.new_block();
    cfg.block_mut(cfg.entry())
        .push(Stmt::new(1, "x := a").defines("x").reads("a"));
    cfg.block_mut(cfg.entry()).push(
        Stmt::new(2, "x := 7")
            .defines("x")
            .constant(SourceConst::Int(7)),
    );
    cfg.block_mut(ret).push(Stmt::new(3, "return x").reads("x"));
    cfg.add_edge(cfg.entry(), ret, EdgeKind::Fallthrough);

    assert!(verify(&cfg).is_ok());

    let dominators = DominatorTree::compute(&cfg);
    assert!(dominators.dominates(cfg.entry(), ret));

    // Flow-sensitive: only the second write of `x` reaches the return.
    let reaching = ReachingDefs::compute(&cfg);
    let x_defs = reaching.defs_of_at_entry(&sym("x"), ret);
    assert_eq!(x_defs.len(), 1, "dead first write must not reach");
    assert_eq!(x_defs[0].inst_idx, 1);

    // Liveness over non-Copy symbol identities.
    let liveness = Liveness::compute(&cfg);
    assert!(liveness.live_in(ret).contains(&sym("x")));
    assert!(!liveness.live_out(ret).contains(&sym("x")));

    // SSA over symbol identities.
    let ssa = build_ssa(&cfg, &dominators);
    assert_eq!(ssa.block(cfg.entry()).instructions.len(), 2);
}

#[test]
fn string_and_bool_constants_propagate() {
    let cfg = CfgBuilder::build(vec![
        Stmt::new(1, "s := \"hi\"")
            .defines("s")
            .constant(SourceConst::Str(String::from("hi"))),
        Stmt::new(2, "b := true")
            .defines("b")
            .constant(SourceConst::Bool(true)),
    ])
    .unwrap();

    let result = constant_propagation(&cfg);
    let out = result.fact_out(cfg.entry());
    assert_eq!(
        out.get(&sym("s")),
        Some(&ConstValue::Const(SourceConst::Str(String::from("hi"))))
    );
    assert_eq!(
        out.get(&sym("b")),
        Some(&ConstValue::Const(SourceConst::Bool(true)))
    );
}

#[test]
fn expression_trees_use_source_operators_and_constants() {
    let cfg = CfgBuilder::build(vec![
        Stmt::new(1, "t := \"a\"")
            .defines("t")
            .constant(SourceConst::Str(String::from("a"))),
        Stmt::new(2, "u := t + name")
            .defines("u")
            .reads("t")
            .reads("name")
            .operating(SourceOp::Concat),
    ])
    .unwrap();

    let trees = recover_block_expressions(&cfg, cfg.entry());
    assert_eq!(trees.roots.len(), 1);
    let (root, expr) = &trees.roots[0];
    assert_eq!(root, &sym("u"));
    match expr {
        ExprNode::Op { operator, operands } => {
            assert_eq!(*operator, SourceOp::Concat);
            assert_eq!(
                operands[0],
                ExprNode::Const(SourceConst::Str(String::from("a")))
            );
            assert_eq!(operands[1], ExprNode::Leaf(sym("name")));
        }
        other => panic!("expected Op at root, got {other:?}"),
    }
}

#[test]
fn dot_rendering_needs_only_display_instr_and_escapes_source_text() {
    // DisplayInstr alone — Stmt's FlowControl is irrelevant here, and the
    // bound-free escape hatch needs no trait at all.
    let mut cfg = Cfg::<Stmt>::new();
    cfg.block_mut(cfg.entry())
        .push(Stmt::new(1, "print(\"quoted \\ text\")"));

    let dot = cfg.to_dot();
    assert!(dot.contains("digraph cfg"));
    assert!(
        dot.contains("print(\\\"quoted \\\\ text\\\")"),
        "source text is escaped: {dot}"
    );

    let with = cfg.to_dot_with(|stmt| Cow::Owned(format!("node {}", stmt.node)));
    assert!(with.contains("node 1"));
}

#[test]
fn goto_wiring_uses_consumer_string_targets() {
    let mut retry = Stmt::new(10, "retry:");
    retry.effect = FlowEffect::Label;
    retry.label = Some(String::from("retry"));
    let mut goto_retry = Stmt::new(11, "goto retry");
    goto_retry.effect = FlowEffect::Jump;
    goto_retry.jump_target = Some(String::from("retry"));

    let mut cfg = CfgBuilder::build(vec![retry, Stmt::new(12, "work()"), goto_retry]).unwrap();

    let resolution = resolve_jump_edges(&mut cfg);
    assert_eq!(resolution.resolved, 1);
    assert_eq!(resolution.unresolved.len(), 0);
    let back = cfg
        .edges()
        .find(|edge| edge.kind() == EdgeKind::Jump)
        .expect("goto wired");
    let target_block = cfg.block(back.target());
    assert_eq!(target_block.instructions()[0].text, "retry:");
    // The label, the work, and the goto share one block, so the wired
    // backward goto is a self-loop on the label block.
    assert_eq!(back.source(), back.target());
}

#[test]
fn switch_recovery_over_cst_node_tokens() {
    /// A dispatch statement whose targets are CST node ids, not addresses.
    struct Dispatch {
        case_nodes: Vec<u32>,
    }

    impl cfglib::SwitchSource for Dispatch {
        type Target = u32;

        fn switch_targets(&self) -> Option<(Vec<u32>, Option<u32>)> {
            Some((self.case_nodes.clone(), None))
        }
    }

    let mut cfg = Cfg::<Dispatch>::new();
    let arm_a = cfg.new_block();
    let arm_b = cfg.new_block();
    cfg.add_edge(cfg.entry(), arm_a, EdgeKind::IndirectJump);
    cfg.block_mut(cfg.entry()).push(Dispatch {
        case_nodes: vec![100, 200],
    });

    let tables: Vec<JumpTable<u32>> = detect_switch_tables(&cfg);
    let node_to_block: BTreeMap<u32, cfglib::BlockId> =
        [(100, arm_a), (200, arm_b)].into_iter().collect();
    let recovered =
        recover_switch_tables(&mut cfg, &tables, |node| node_to_block.get(node).copied());
    assert_eq!(recovered[0].num_cases, 2);
    assert!(cfg.edges().all(|e| e.kind() != EdgeKind::IndirectJump));
}
