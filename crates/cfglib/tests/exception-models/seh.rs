//! SEH scope tables registered as regions: an `__except` is a
//! [`HandlerKind::Filter`] and a `__finally` a [`HandlerKind::Finally`].

use cfglib::{Cfg, EhBlockKind, EhModel, Handler, HandlerBody, HandlerKind, Region, RegionIndex};

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

    let outer = cfg
        .add_region(Region {
            protected_blocks: set(&[guarded, outer_guarded]),
            handlers: vec![Handler {
                entry: outer_handler,
                body: HandlerBody::Unknown,
                kind: HandlerKind::Finally,
            }],
            parent: None,
        })
        .unwrap();
    // Scope-table order is dispatch order.
    let scope_table = vec![
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
    ];
    let inner = cfg
        .add_region(Region {
            protected_blocks: set(&[guarded]),
            handlers: scope_table.clone(),
            parent: Some(outer),
        })
        .unwrap();

    assert_eq!((outer.index(), inner.index()), (0, 1));
    let outer_region = cfg.region(outer);
    assert_eq!(outer_region.handlers.len(), 1);
    assert_eq!(outer_region.handlers[0].kind, HandlerKind::Finally);
    assert_eq!(outer_region.handlers[0].body, HandlerBody::Unknown);
    assert_eq!(cfg.region(inner).parent, Some(outer));
    // Dispatch order survives registration.
    assert_eq!(cfg.region(inner).handlers, scope_table);
    let index = RegionIndex::compute(&cfg);
    assert_eq!(index.protecting(guarded), Some(inner));
    assert_eq!(index.protecting(outer_guarded), Some(outer));
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
