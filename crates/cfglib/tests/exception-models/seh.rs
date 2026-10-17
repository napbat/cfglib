//! SEH scope tables and the dynamic registration chain.

use cfglib::{
    BlockId, Cfg, EhBlockKind, EhModel, Handler, HandlerBody, HandlerKind, SehExceptionRegion,
    SehHandler, SehHandlerKind, SehRegistration, SehRegistrationChain, install_seh_region,
};

use crate::set;

#[test]
fn seh_scope_handlers_become_filters_and_finally_handlers_in_dispatch_order() {
    let mut cfg = Cfg::<&str>::new();
    let guarded = cfg.entry();
    let filter_a = cfg.new_block();
    let handler_a = cfg.new_block();
    let termination = cfg.new_block();
    let outer_guarded = cfg.new_block();
    let outer_handler = cfg.new_block();

    let outer = install_seh_region(
        &mut cfg,
        SehExceptionRegion {
            protected_blocks: set(&[guarded, outer_guarded]),
            handlers: vec![SehHandler {
                entry: outer_handler,
                body: HandlerBody::Unknown,
                kind: SehHandlerKind::Finally,
            }],
            parent: None,
        },
    );
    let inner = install_seh_region(
        &mut cfg,
        SehExceptionRegion {
            protected_blocks: set(&[guarded]),
            handlers: vec![
                SehHandler {
                    entry: handler_a,
                    body: HandlerBody::known([handler_a]),
                    kind: SehHandlerKind::Except {
                        filter_block: filter_a,
                    },
                },
                SehHandler {
                    entry: termination,
                    body: HandlerBody::known([termination]),
                    kind: SehHandlerKind::Finally,
                },
            ],
            parent: Some(outer),
        },
    );

    assert_eq!((outer.index(), inner.index()), (0, 1));
    let regions = cfg.regions();
    assert_eq!(regions[0].handlers.len(), 1);
    assert_eq!(regions[0].handlers[0].kind, HandlerKind::Finally);
    assert_eq!(regions[0].handlers[0].body, HandlerBody::Unknown);
    assert_eq!(regions[1].parent, Some(outer));
    // Scope-table order is dispatch order and survives installation.
    assert_eq!(
        regions[1].handlers,
        [
            Handler {
                entry: handler_a,
                body: HandlerBody::known([handler_a]),
                kind: HandlerKind::Filter {
                    filter_block: filter_a,
                },
            },
            Handler {
                entry: termination,
                body: HandlerBody::known([termination]),
                kind: HandlerKind::Finally,
            },
        ]
    );
    assert_eq!(
        cfg.protecting_region(guarded).map(|region| region.id),
        Some(inner)
    );
    assert_eq!(
        cfg.protecting_region(outer_guarded).map(|region| region.id),
        Some(outer)
    );
    assert_eq!(cfg.block_count(), 6);
    assert_eq!(cfg.edge_count(), 0);

    let model = EhModel::compute(&cfg);
    assert_eq!(model.block_kind(handler_a), EhBlockKind::CatchSwitch);
    assert_eq!(model.block_kind(termination), EhBlockKind::Cleanup);
    assert_eq!(model.block_kind(outer_handler), EhBlockKind::Cleanup);
    assert_eq!(model.protected_by(termination), &set(&[guarded]));
    assert_eq!(
        model.protected_by(outer_handler),
        &set(&[guarded, outer_guarded])
    );
}

#[test]
fn registration_chains_dispatch_newest_first_and_a_pop_removes_the_newest() {
    let mut cfg = Cfg::<&str>::new();
    let outer_try = cfg.entry();
    let inner_try = cfg.new_block();
    let handler = cfg.new_block();
    let region = |cfg: &mut Cfg<&str>, protected: BlockId| {
        install_seh_region(
            cfg,
            SehExceptionRegion {
                protected_blocks: set(&[protected]),
                handlers: vec![SehHandler {
                    entry: handler,
                    body: HandlerBody::Unknown,
                    kind: SehHandlerKind::Finally,
                }],
                parent: None,
            },
        )
    };
    let outer_region = region(&mut cfg, outer_try);
    let inner_region = region(&mut cfg, inner_try);

    let mut chain = SehRegistrationChain::new();
    assert!(chain.is_empty());
    assert_eq!(chain.len(), 0);
    assert!(chain.pop().is_none());
    assert_eq!(chain.dispatch_order().count(), 0);

    chain.push(SehRegistration {
        frame: "outer".to_string(),
        handler: 0x0040_1000_u32,
        regions: vec![outer_region],
    });
    chain.push(SehRegistration {
        frame: "inner".to_string(),
        handler: 0x0040_2000,
        // One registration can cover several lexical regions of a scope
        // table.
        regions: vec![inner_region, outer_region],
    });
    assert_eq!(chain.len(), 2);
    assert!(!chain.is_empty());

    let frames = |chain: &SehRegistrationChain<String, u32>| -> Vec<String> {
        chain
            .dispatch_order()
            .map(|entry| entry.frame.clone())
            .collect()
    };
    assert_eq!(frames(&chain), ["inner", "outer"]);
    assert_eq!(
        chain
            .dispatch_order()
            .rev()
            .map(|entry| entry.handler)
            .collect::<Vec<_>>(),
        [0x0040_1000, 0x0040_2000],
        "the iterator is double-ended: oldest first when reversed"
    );
    let newest = chain.dispatch_order().next().unwrap();
    for id in &newest.regions {
        assert!(
            cfg.region(*id).is_some(),
            "registrations name installed regions"
        );
    }

    let popped = chain.pop().expect("the newest frame unregisters");
    assert_eq!(popped.frame, "inner");
    assert_eq!(popped.regions, [inner_region, outer_region]);
    assert_eq!(frames(&chain), ["outer"]);
    assert_eq!(chain.len(), 1);

    let default: SehRegistrationChain<String, u32> = SehRegistrationChain::default();
    assert!(default.is_empty());
}

#[test]
fn a_registration_chain_matches_a_stack_under_random_pushes_and_pops() {
    let mut chain = SehRegistrationChain::<u32, u32>::new();
    let mut stack: Vec<u32> = Vec::new();
    let mut state = 0x2545_f491_4f6c_dd1d_u64;
    for step in 0..400_u32 {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        if state.is_multiple_of(3) {
            assert_eq!(chain.pop().map(|entry| entry.frame), stack.pop());
        } else {
            chain.push(SehRegistration {
                frame: step,
                handler: step ^ 0xff,
                regions: Vec::new(),
            });
            stack.push(step);
        }
        let dispatch: Vec<u32> = chain.dispatch_order().map(|entry| entry.frame).collect();
        let expected: Vec<u32> = stack.iter().rev().copied().collect();
        assert_eq!(dispatch, expected, "after step {step}");
        assert_eq!(chain.len(), stack.len());
    }
}
