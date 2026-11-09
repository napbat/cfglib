//! Executable specification: a source-language frontend (symtree-shaped)
//! drives the whole stack with consumer-typed axes everywhere.
//!
//! Variables are interned symbol names (non-`Copy`, non-numeric), branch
//! targets are source labels, and rendering needs only [`DisplayInstr`].

use std::borrow::Cow;

/// The package's golden-file comparison, shared with the unit tests.
#[path = "../src/test_util/golden.rs"]
mod golden;

use golden::assert_golden;

use cfglib::{
    Cfg, CfgBuilder, DisplayInstr, DominatorTree, EdgeKind, FlowControl, FlowEffect, InstrInfo,
    JumpTargets, resolve_jump_edges, verify,
};

/// A source symbol identity — deliberately non-`Copy`, non-numeric.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct SymbolId(String);

fn sym(name: &str) -> SymbolId {
    SymbolId(String::from(name))
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
fn direct_construction_and_dominators_over_symbols() {
    // fn f(a) { x := a; x := 7; return x }  — built structurally, no builder.
    let mut cfg = Cfg::<Stmt>::new();
    let ret = cfg.new_block();
    cfg.block_mut(cfg.entry())
        .push(Stmt::new(1, "x := a").defines("x").reads("a"));
    cfg.block_mut(cfg.entry())
        .push(Stmt::new(2, "x := 7").defines("x"));
    cfg.block_mut(ret).push(Stmt::new(3, "return x").reads("x"));
    cfg.add_edge(cfg.entry(), ret, EdgeKind::Fallthrough);

    assert!(verify(&cfg).is_ok());

    let dominators = DominatorTree::compute(&cfg);
    assert!(dominators.dominates(cfg.entry(), ret));
}

#[test]
fn rendering_needs_only_display_instr_and_escapes_source_text() {
    // DisplayInstr alone — Stmt's FlowControl is irrelevant here, and the
    // bound-free escape hatch needs no trait at all.
    let mut cfg = Cfg::<Stmt>::new();
    cfg.block_mut(cfg.entry())
        .push(Stmt::new(1, "print(\"quoted \\ text\")"));
    let tail = cfg.new_block();
    cfg.block_mut(tail).push(Stmt::new(2, "return"));
    cfg.add_edge(cfg.entry(), tail, EdgeKind::Fallthrough);

    assert_golden("dot/source-statements.dot", &cfg.to_dot());
    assert_golden("text/source-statements.txt", &cfg.to_text());
    assert_golden(
        "dot/source-node-ids.dot",
        &cfg.to_dot_with(|stmt| Cow::Owned(format!("node {}", stmt.node))),
    );
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
    assert_eq!(
        resolution.unresolved,
        [] as [(cfglib::Id<cfglib::BlockTag>, std::string::String); 0]
    );
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
