extern crate alloc;

use alloc::string::ToString;
use alloc::vec;
use alloc::vec::Vec;

use crate::ir::mlil::FunctionBuilder;

use super::{Edge, Operation, ToyDialect};

/// A region names blocks by identity, so retiring an unrelated low-indexed
/// block must not make a high-indexed handler entry look nonexistent.
#[test]
fn a_region_survives_the_removal_of_an_unrelated_block() {
    let mut builder = FunctionBuilder::<ToyDialect>::new("toy::removal".to_string());
    // An unreachable forwarding block: empty, so removing it leaves no
    // dangling instruction identity behind.
    let dead = builder.new_block("dead");
    let body = builder.new_block("body");
    let pad = builder.new_block("pad");
    builder
        .append_instruction(body, Operation::Return, Vec::new(), Vec::new(), false, None)
        .unwrap();
    builder
        .append_instruction(pad, Operation::Return, Vec::new(), Vec::new(), false, None)
        .unwrap();
    builder
        .add_edge(builder.entry(), body, Edge::Entry, None)
        .unwrap();
    builder.add_edge(dead, body, Edge::Next, None).unwrap();
    builder
        .add_region(crate::Region {
            id: crate::RegionId::from_raw(0),
            protected_blocks: [body].into_iter().collect(),
            handlers: vec![crate::Handler {
                entry: pad,
                body: crate::HandlerBody::known([pad]),
                kind: crate::HandlerKind::CatchAll,
            }],
            parent: None,
        })
        .unwrap();

    let mut function = builder.finish().unwrap();
    assert!(function.verify().is_ok());

    assert!(function.cfg.remove_block(dead));
    assert_eq!(function.cfg.block_count(), 3);
    assert_eq!(function.cfg.block_bound(), 4);
    assert_eq!(
        pad.index(),
        3,
        "the handler entry must sit above the live block count"
    );

    let report = function.verify();
    assert!(report.is_ok(), "{:?}", report.issues);
}
