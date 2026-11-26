//! The elevator vocabulary: how a Windows/.NET lifter records CLR clause
//! tables and SEH scope tables as `Cfg` regions, keeps catch-type tokens in a
//! `HandlerMetadata` side table, and annotates edges with `ExceptionFlow`.
//!
//! Expectations are derived by hand from the documented clause kinds and
//! dispatch phases, then compared with the registered regions, handlers,
//! edges, and `EhModel` classification. Reachability over the normal-only and
//! full-flow views is checked against dominance by node removal, which shares
//! no code with `DominatorTree`.

use std::collections::BTreeSet;

use cfglib::{BlockId, Cfg, ExceptionFlow};

mod clr;
mod dispatch;
mod exception_flow;
mod seh;

/// The control-flow graph a lifter builds: statement labels as instructions,
/// exception semantics on the edges.
type LiftedCfg = Cfg<&'static str, ExceptionFlow<&'static str>>;

fn set(blocks: &[BlockId]) -> BTreeSet<BlockId> {
    blocks.iter().copied().collect()
}
