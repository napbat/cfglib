use cfglib::{
    AddressCfgOptions, AddressEdgeInfo, AddressGraph, AddressHandler, AddressInstruction, Flow,
    HandlerKind, build_address_cfg,
};

use super::BenchmarkSuite;
use crate::harness::benchmark_case;

const INSTRUCTION_COUNT: usize = 4_096;
const HANDLER_COUNT: usize = 256;

#[derive(Clone)]
struct Inst {
    address: u32,
    flow: Flow<u32, u32>,
}

impl AddressInstruction for Inst {
    type Address = u32;
    type CaseKey = u32;

    fn address(&self) -> u32 {
        self.address
    }

    fn end_address(&self) -> Option<u32> {
        self.address.checked_add(1)
    }

    fn flow(&self) -> Flow<u32, u32> {
        self.flow.clone()
    }

    fn retains_exception_edge(&self) -> bool {
        self.address % 3 != 0
    }
}

fn fixture() -> (Vec<Inst>, Vec<AddressHandler<u32>>) {
    let instructions = (0..INSTRUCTION_COUNT)
        .map(|position| Inst {
            address: u32::try_from(position).expect("fixture address fits"),
            flow: if position + 1 == INSTRUCTION_COUNT {
                Flow::Return
            } else {
                Flow::Next
            },
        })
        .collect();
    let handlers = (0..HANDLER_COUNT)
        .map(|position| {
            let start = u32::try_from(position * 12).expect("fixture range fits");
            AddressHandler {
                protected: start..start + 8,
                entry: u32::try_from(INSTRUCTION_COUNT - 1).expect("fixture entry fits"),
                kind: HandlerKind::CatchAll,
            }
        })
        .collect();
    (instructions, handlers)
}

fn payload(_: AddressEdgeInfo<'_, u32, u32>) {}

fn assert_graph(graph: &AddressGraph<Inst, ()>) {
    assert_eq!(graph.instruction_blocks.len(), INSTRUCTION_COUNT);
    assert_eq!(graph.handler_refs.len(), HANDLER_COUNT);
    assert!(graph.unresolved_transfers.is_empty());
}

pub(super) fn register(suite: &mut BenchmarkSuite<'_>) {
    let (instructions, handlers) = fixture();
    benchmark_case!(
        suite,
        "address_cfg_handler_heavy",
        covers[build_address_cfg],
        || build_address_cfg(
            instructions.clone(),
            &handlers,
            AddressCfgOptions::default(),
            payload
        )
        .unwrap(),
        assert_graph,
    );
    let entry = u32::try_from(INSTRUCTION_COUNT / 2).expect("fixture entry fits");
    benchmark_case!(
        suite,
        "address_cfg_explicit_entry",
        covers[build_address_cfg],
        || {
            let options = AddressCfgOptions {
                entry: Some(entry),
                ..AddressCfgOptions::default()
            };
            build_address_cfg(instructions.clone(), &handlers, options, payload).unwrap()
        },
        |graph: &AddressGraph<Inst, ()>| {
            assert_graph(graph);
            assert_eq!(graph.cfg.entry(), graph.instruction_blocks[&entry]);
        },
    );
}
