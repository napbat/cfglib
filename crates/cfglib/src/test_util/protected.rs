//! The smallest exceptional-flow shape: one block whose instructions can
//! unwind into a handler.

use crate::block::BlockId;
use crate::cfg::Cfg;
use crate::edge::EdgeKind;

/// `entry -> protected -> normal`, with `protected` also unwinding to
/// `handler`.
///
/// `instructions` fill `protected`; the entry, normal, and handler blocks start
/// empty, so a test adds there whatever it needs to observe on each path.
/// Returns the graph and `[protected, normal, handler]`.
pub(crate) fn protected_region<I>(
    instructions: impl IntoIterator<Item = I>,
) -> (Cfg<I>, [BlockId; 3]) {
    let mut cfg = Cfg::new();
    let protected = cfg.new_block();
    let normal = cfg.new_block();
    let handler = cfg.new_block();
    cfg.add_edge(cfg.entry(), protected, EdgeKind::Fallthrough);
    cfg.add_edge(protected, normal, EdgeKind::Fallthrough);
    cfg.add_edge(protected, handler, EdgeKind::ExceptionUnwind);
    cfg.block_mut(protected)
        .instructions_mut()
        .extend(instructions);
    (cfg, [protected, normal, handler])
}

#[cfg(test)]
mod tests {
    extern crate alloc;

    use alloc::vec::Vec;

    use crate::edge::EdgeKind;
    use crate::flow::{FlowControl, FlowEffect};
    use crate::test_util::{df_ff, df_throwing, protected_region};

    #[test]
    fn only_the_protected_block_reaches_the_handler_and_it_unwinds_there() {
        let (cfg, [protected, normal, handler]) =
            protected_region([df_throwing(df_ff("call")), df_ff("after")]);

        let leaving: Vec<_> = cfg
            .outgoing(protected)
            .map(|edge| (cfg.edge(edge).target(), cfg.edge(edge).kind()))
            .collect();
        assert_eq!(
            leaving,
            [
                (normal, EdgeKind::Fallthrough),
                (handler, EdgeKind::ExceptionUnwind)
            ]
        );
        assert_eq!(cfg.incoming(handler).count(), 1);
        assert_eq!(cfg.incoming(normal).count(), 1);
        assert_eq!(cfg.outgoing(normal).count(), 0);
        assert_eq!(cfg.outgoing(handler).count(), 0);
    }

    #[test]
    fn the_instructions_land_in_the_protected_block_alone() {
        let (cfg, [protected, normal, handler]) =
            protected_region([df_throwing(df_ff("call")), df_ff("after")]);

        let effects: Vec<_> = cfg
            .block(protected)
            .instructions()
            .iter()
            .map(FlowControl::flow_effect)
            .collect();
        assert_eq!(effects, [FlowEffect::MayThrow, FlowEffect::Fallthrough]);
        for empty in [cfg.entry(), normal, handler] {
            assert!(cfg.block(empty).instructions().is_empty());
        }
    }
}
