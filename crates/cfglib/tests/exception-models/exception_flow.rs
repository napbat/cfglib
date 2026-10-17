//! `ExceptionFlow`, `ExceptionPhase`, and `ExceptionDisposition` as edge
//! annotations.

use std::collections::HashSet;

use cfglib::{ExceptionDisposition, ExceptionFlow, ExceptionPhase};

#[test]
fn exception_flow_reads_back_what_it_was_built_with() {
    let normal = ExceptionFlow::normal("edge");
    assert_eq!(normal.phase(), None);
    assert_eq!(normal.disposition(), None);
    assert_eq!(*normal.metadata(), "edge");

    let phases = [ExceptionPhase::Search, ExceptionPhase::Unwind];
    let dispositions = [
        None,
        Some(ExceptionDisposition::ExecuteHandler),
        Some(ExceptionDisposition::ContinueSearch),
        Some(ExceptionDisposition::ContinueExecution),
    ];
    let mut distinct = Vec::new();
    for phase in phases {
        for disposition in dispositions {
            let flow = ExceptionFlow::exceptional(phase, disposition, 0xC000_0005_u32);
            assert_eq!(flow.phase(), Some(phase));
            assert_eq!(flow.disposition(), disposition);
            assert_eq!(*flow.metadata(), 0xC000_0005);
            assert!(
                !distinct.contains(&flow),
                "every phase and disposition pair differs"
            );
            distinct.push(flow);
        }
    }
    assert_eq!(distinct.len(), 8);
    assert_ne!(distinct[0], ExceptionFlow::normal(0xC000_0005));
    assert_eq!(
        phases.iter().collect::<HashSet<_>>().len()
            + dispositions.iter().flatten().collect::<HashSet<_>>().len(),
        5,
        "phases and dispositions are hashable and distinct"
    );
}

#[test]
fn exception_flow_metadata_is_editable_and_movable_without_changing_semantics() {
    let mut flow = ExceptionFlow::exceptional(
        ExceptionPhase::Unwind,
        Some(ExceptionDisposition::ExecuteHandler),
        String::from("scope#1"),
    );
    flow.metadata_mut().push_str("/clause#2");
    assert_eq!(flow.metadata(), "scope#1/clause#2");
    assert_eq!(flow.phase(), Some(ExceptionPhase::Unwind));
    assert_eq!(
        flow.disposition(),
        Some(ExceptionDisposition::ExecuteHandler)
    );
    let cloned = flow.clone();
    assert_eq!(cloned, flow);
    assert_eq!(flow.into_metadata(), "scope#1/clause#2");

    let mut plain = ExceptionFlow::normal(1_u8);
    *plain.metadata_mut() += 1;
    assert_eq!(plain.into_metadata(), 2);
    let default: ExceptionFlow<u8> = ExceptionFlow::default();
    assert_eq!(default, ExceptionFlow::normal(0));
    assert_eq!(default.phase(), None);
}
