//! `ir::llil`: the semantics of one instruction, emitted and captured.

use cfglib::ir::llil::{Instruction, Semantics, TranslateError};
use cfglib::ir::rtl::{Control, InstructionBuilder, Place, Selection, Statement, WriteTarget};
use cfglib::{AddressInstruction, Captured, CapturedValue, DenseId, Flow};

use crate::wren::{Policy, Reg, Wren, take};

/// A two-byte instruction that copies `A` to `B`, or one that the semantics
/// do not model.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Move {
    address: u32,
    bytes: [u8; 2],
    modeled: bool,
}

impl AddressInstruction for Move {
    type Address = u32;
    type CaseKey = u64;

    fn address(&self) -> u32 {
        self.address
    }

    fn end_address(&self) -> Option<u32> {
        self.address.checked_add(2)
    }

    fn flow(&self) -> Flow<u32, u64> {
        Flow::Next
    }
}

impl Instruction for Move {
    fn length(&self) -> usize {
        self.bytes.len()
    }

    fn bytes(&self) -> &[u8] {
        &self.bytes
    }
}

struct MoveSemantics;

impl Semantics for MoveSemantics {
    type Instruction = Move;
    type Rtl = Wren;

    fn translate<B: InstructionBuilder<Wren>>(
        &self,
        instruction: &Move,
        mut builder: B,
    ) -> Result<(), TranslateError<B::Error>> {
        if !instruction.modeled {
            return Err(TranslateError::Unsupported);
        }
        let value = builder.read(Selection::Direct(Place::whole(Reg::A)));
        builder.write(
            WriteTarget::Root {
                root: Reg::B,
                plan: &[take(0, 32, value, 0)],
            },
            Policy::Merge,
        );
        builder
            .finish(Control::Advance)
            .map_err(TranslateError::Seal)
    }
}

#[test]
fn an_instruction_states_its_encoding_and_its_flow() {
    let instruction = Move {
        address: 0x10,
        bytes: [0x8b, 0xd8],
        modeled: true,
    };
    assert_eq!(instruction.length(), 2);
    assert_eq!(instruction.bytes(), &[0x8b, 0xd8]);
    assert_eq!(instruction.end_address(), Some(0x12));
    assert_eq!(instruction.flow(), Flow::Next);
}

#[test]
fn the_default_capture_holds_what_translate_emits() {
    let instruction = Move {
        address: 0x10,
        bytes: [0x8b, 0xd8],
        modeled: true,
    };
    let mut captured = Captured::<Wren>::new();
    MoveSemantics
        .capture(&instruction, &mut captured)
        .expect("the move has semantics");

    let value = CapturedValue::from_index(0);
    let plan = [take(0, 32, value, 0)];
    let expected = [
        Statement::Read {
            result: value,
            source: Selection::Direct(Place::whole(Reg::A)),
        },
        Statement::Write {
            target: WriteTarget::Root {
                root: Reg::B,
                plan: &plan,
            },
            policy: Policy::Merge,
        },
    ];
    assert!(captured.statements().eq(expected));
    assert_eq!(captured.finished_control(), Some(Control::Advance));
}

#[test]
fn a_capture_reports_an_unmodeled_instruction() {
    let instruction = Move {
        address: 0x10,
        bytes: [0x0f, 0x0b],
        modeled: false,
    };
    let mut captured = Captured::<Wren>::new();
    assert_eq!(
        MoveSemantics.capture(&instruction, &mut captured),
        Err(TranslateError::Unsupported)
    );
}
