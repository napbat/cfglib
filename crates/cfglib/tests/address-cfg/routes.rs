//! Exact unwind routes added to a built graph.

use cfglib::{EdgeKind, EhBlockKind, EhModel, HopEnd, UnwindEnd, UnwindPath, UnwindRoute};

use crate::harness::{Role, build, wire};
use crate::isa::Dest::At;
use crate::isa::{Isa, Op};

/// A filter that answers "continue execution" makes the runtime resume the
/// protected instruction itself (`EXCEPTION_CONTINUE_EXECUTION`). The way
/// back is an `ExceptionContinue` edge, so the exception model sees a
/// resume point and no landing pad that protects itself.
fn a_resume_route_continues_execution_at_the_protected_instruction<I: Isa>() {
    // The branch back to position 1 makes the faulting instruction at 0 a
    // block of its own.
    let mut built = build::<I>(&[Op::Work, Op::Work, Op::BranchIf(At(1)), Op::Return]);
    let protected = built.block_of(0);
    let routes = [UnwindRoute {
        protected: built.layout.address(0)..built.layout.address(1),
        paths: vec![UnwindPath {
            steps: Vec::new(),
            end: UnwindEnd::Resume,
        }],
    }];

    let hops = built
        .graph
        .add_unwind_routes(&routes, wire)
        .expect("the route is valid");

    assert_eq!(hops.len(), 1);
    assert_eq!(hops[0].end, HopEnd::Block(protected));
    let cfg = built.cfg();
    let exceptional: Vec<_> = cfg
        .outgoing(protected)
        .map(|edge| cfg.edge(edge))
        .filter(|edge| edge.kind().is_exceptional())
        .map(|edge| (edge.target(), edge.kind(), edge.payload().role))
        .collect();
    assert_eq!(
        exceptional,
        [(
            protected,
            EdgeKind::ExceptionContinue,
            Role::Continue { route: 0 }
        )]
    );
    let model = EhModel::compute(cfg);
    assert!(
        model
            .block_kinds()
            .all(|(_, kind)| kind != EhBlockKind::LandingPad),
        "resuming execution enters no landing pad"
    );
}

over_both_isas!(a_resume_route_continues_execution_at_the_protected_instruction);
