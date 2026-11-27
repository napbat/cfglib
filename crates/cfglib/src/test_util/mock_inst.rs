//! Minimal mock instruction for graph and transform tests.

extern crate alloc;

use alloc::borrow::Cow;

use crate::display::DisplayInstr;
use crate::flow::{Flow, FlowControl, FlowEffect};

/// A minimal mock instruction carrying only flow-effect and mnemonic.
#[derive(Debug, Clone)]
pub struct MockInst(pub FlowEffect<&'static str, ()>, pub &'static str);

impl FlowControl for MockInst {
    type Target = &'static str;
    type CaseKey = ();

    fn flow_effect(&self) -> FlowEffect<&'static str, ()> {
        self.0.clone()
    }
}

impl DisplayInstr for MockInst {
    fn mnemonic(&self) -> Cow<'_, str> {
        Cow::Borrowed(self.1)
    }
}

/// Shorthand for a [`MockInst`] with [`Flow::Next`].
pub fn ff(name: &'static str) -> MockInst {
    MockInst(FlowEffect::Flow(Flow::Next), name)
}

/// Shorthand for a [`MockInst`] with [`Flow::Return`].
pub fn ret(name: &'static str) -> MockInst {
    MockInst(FlowEffect::Flow(Flow::Return), name)
}

/// Build a diamond CFG: entry → A, entry → B, A → merge, B → merge.
pub fn diamond_cfg() -> crate::cfg::Cfg<MockInst> {
    use crate::edge::EdgeKind;

    let mut cfg = crate::cfg::Cfg::new();
    let a = cfg.new_block();
    let b = cfg.new_block();
    let merge = cfg.new_block();
    cfg.block_mut(cfg.entry()).push(ff("entry"));
    cfg.block_mut(a).push(ff("a"));
    cfg.block_mut(b).push(ff("b"));
    cfg.block_mut(merge).push(ff("merge"));
    cfg.add_edge(cfg.entry(), a, EdgeKind::ConditionalTrue);
    cfg.add_edge(cfg.entry(), b, EdgeKind::ConditionalFalse);
    cfg.add_edge(a, merge, EdgeKind::Fallthrough);
    cfg.add_edge(b, merge, EdgeKind::Fallthrough);
    cfg
}
