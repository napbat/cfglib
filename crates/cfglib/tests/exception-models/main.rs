//! The elevator vocabulary: how a Windows/.NET lifter installs CLR and SEH
//! exception metadata into a `Cfg`, models dynamic SEH registration and
//! process-wide vectored handlers beside it, and annotates edges with
//! `ExceptionFlow`.
//!
//! Expectations are derived by hand from the documented clause kinds, chain
//! orders, and dispatch phases, then compared with the installed regions,
//! handlers, edges, and `EhModel` classification. Reachability over the
//! normal-only and full-flow views is checked against dominance by node
//! removal, which shares no code with `DominatorTree`.

use std::collections::BTreeSet;

use cfglib::{BlockId, Cfg, ExceptionFlow};

mod clr;
mod dispatch;
mod dispatch_order;
mod exception_flow;
mod seh;
mod vectored;

/// The control-flow graph a lifter builds: statement labels as instructions,
/// exception semantics on the edges.
type LiftedCfg = Cfg<&'static str, ExceptionFlow<&'static str>>;

fn set(blocks: &[BlockId]) -> BTreeSet<BlockId> {
    blocks.iter().copied().collect()
}
