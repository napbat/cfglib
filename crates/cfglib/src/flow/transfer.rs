//! How one operation leaves its position, and the edges that describes.
//!
//! [`Flow`] is the one vocabulary for an operation's control transfer. A
//! machine instruction, a bytecode operation, and a lowered source statement
//! all state the same thing: whether control continues at the next
//! position, which targets it names, and how it leaves. Both builders read
//! it: [`build_address_cfg`](crate::build_address_cfg) through
//! [`AddressInstruction::flow`](crate::AddressInstruction::flow), and
//! [`CfgBuilder`](crate::CfgBuilder) through
//! [`FlowEffect::Flow`](crate::FlowEffect::Flow). Everything that turns a
//! stream of operations into a graph reads that statement through
//! [`Flow::transfers`], so the mapping from a flow to its successors is
//! written once here rather than in every builder and every walker.
//!
//! Each transfer carries an [`EdgeRole`]. The role says why the edge exists,
//! and [`EdgeRole::kind`] is the one place a role turns into the structural
//! [`EdgeKind`] the graph stores. A transfer whose target is outside the
//! recovered operations becomes an [`UnresolvedTransfer`], which keeps the
//! named target and its role instead of dropping the edge silently.

extern crate alloc;

use alloc::vec::Vec;

use crate::edge::EdgeKind;

/// One operation's control transfer.
///
/// `T` is the consumer's target identity: a code address for a machine
/// stream, a label or block identity for a lowered source stream. `K` is
/// the consumer's switch case key.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Flow<T, K> {
    /// Continues at the next position.
    Next,
    /// Leaves the function normally.
    Return,
    /// Leaves the function normally when its condition holds. Otherwise,
    /// continues at the next position.
    ConditionalReturn,
    /// Leaves the function exceptionally, through the enclosing regions.
    Throw,
    /// Transfers to a target the operation does not name.
    Indirect,
    /// Transfers to a target the operation does not name when taken.
    /// Otherwise, continues at the next position.
    ConditionalIndirect,
    /// Calls a target the operation does not name, then continues at the
    /// next position.
    IndirectCall,
    /// Always transfers to `target`.
    Jump {
        /// Branch target.
        target: T,
    },
    /// Transfers to `target` when taken. Otherwise, continues at the next
    /// position.
    Conditional {
        /// Taken-path target.
        target: T,
    },
    /// Calls `target`, then continues at the next position.
    Call {
        /// Call target.
        target: T,
    },
    /// Calls `target` in place of returning: the callee returns to this
    /// function's caller, so control does not continue here.
    ///
    /// A computed tail call names no target; it is an
    /// [`Indirect`](Self::Indirect) jump.
    TailCall {
        /// Call target.
        target: T,
    },
    /// Continues at the next position and may transfer to `target` through
    /// an exceptional or alternate path.
    ///
    /// This covers operations such as a transactional-abort fallback, whose
    /// alternate path is neither a conditional branch nor an unwind through
    /// a handler table. An operation with this flow leads its own block in
    /// a built graph, so the exceptional edge is exact to that operation.
    Exceptional {
        /// Exceptional or alternate-path target.
        target: T,
    },
    /// Raises a trap: a fault, an interrupt, a system call, or a
    /// breakpoint.
    ///
    /// The operation does not state where control goes. The platform does
    /// (an interrupt vector, a system-call handler, an exception
    /// dispatcher), and so does whether control comes back. The flow
    /// therefore has no continuation; its one transfer is computed.
    Trap,
    /// Raises a trap when its condition holds. Otherwise, continues at the
    /// next position.
    ///
    /// The trap's destination is a platform fact, as for
    /// [`Trap`](Self::Trap); the continuation is a fact of the operation.
    ConditionalTrap,
    /// Multi-way dispatch over keyed cases with a default.
    Switch {
        /// Target when no case matches.
        default: T,
        /// Keyed case targets in dispatch order.
        cases: Vec<(K, T)>,
    },
}

impl<T, K> Flow<T, K> {
    /// Whether the operation ends its basic block.
    ///
    /// Every transfer except a plain [`Next`](Self::Next) does. A call ends
    /// its block too: its call and continuation edges leave the block's last
    /// operation, so the return site leads a block of its own.
    #[must_use]
    pub const fn ends_block(&self) -> bool {
        !matches!(self, Self::Next)
    }

    /// Whether control can continue at the next position.
    #[must_use]
    pub const fn continues(&self) -> bool {
        matches!(
            self,
            Self::Next
                | Self::ConditionalReturn
                | Self::IndirectCall
                | Self::Conditional { .. }
                | Self::ConditionalIndirect
                | Self::Call { .. }
                | Self::Exceptional { .. }
                | Self::ConditionalTrap
        )
    }

    /// Whether the operation calls another function.
    #[must_use]
    pub const fn is_call(&self) -> bool {
        matches!(
            self,
            Self::Call { .. } | Self::TailCall { .. } | Self::IndirectCall
        )
    }

    /// The single target the operation names directly, if it names one.
    ///
    /// A jump, a conditional branch, a call, a tail call, and an exceptional
    /// transfer name one target. A switch names several, and the rest name
    /// none.
    #[must_use]
    pub fn direct_target(&self) -> Option<T>
    where
        T: Copy,
    {
        match self {
            Self::Jump { target }
            | Self::Conditional { target }
            | Self::Call { target }
            | Self::TailCall { target }
            | Self::Exceptional { target } => Some(*target),
            Self::Next
            | Self::Return
            | Self::ConditionalReturn
            | Self::Throw
            | Self::Indirect
            | Self::ConditionalIndirect
            | Self::IndirectCall
            | Self::Trap
            | Self::ConditionalTrap
            | Self::Switch { .. } => None,
        }
    }

    /// Every transfer this flow makes from `source`.
    ///
    /// `next` is the position control continues at, when the stream has
    /// one. A transfer with no target is a computed one; a transfer with a
    /// target is an edge the graph can carry when the target is recovered.
    pub fn transfers(&self, source: T, next: Option<T>) -> Transfers<'_, T, K>
    where
        T: Clone,
    {
        Transfers {
            flow: self,
            continuation: next.map(|next| (source, next)),
            index: 0,
        }
    }

    /// Every transfer this flow makes except its continuation.
    ///
    /// A structured stream has no positions that a target can name (its
    /// targets are labels), so it asks only for the transfers that leave.
    pub(crate) fn departures(&self) -> Transfers<'_, T, K>
    where
        T: Clone,
    {
        Transfers {
            flow: self,
            continuation: None,
            index: 0,
        }
    }
}

/// Why one edge exists.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EdgeRole<'k, T, K> {
    /// Sequential continuation at the next position.
    Sequential,
    /// Taken path of a conditional branch.
    ConditionalTaken,
    /// Not-taken path of a conditional branch.
    ConditionalFallThrough,
    /// Direct unconditional jump.
    Jump,
    /// Switch dispatch when no case matches.
    SwitchDefault,
    /// Switch dispatch for one keyed case.
    SwitchCase {
        /// Zero-based case position in dispatch order.
        index: usize,
        /// The consumer's case key.
        key: &'k K,
    },
    /// Direct call transfer, of a call or of a tail call.
    Call,
    /// Continuation after a call returns.
    CallContinuation {
        /// Position of the calling operation.
        call_site: T,
    },
    /// Exceptional or alternate transfer the operation states: a named
    /// target, or a trap whose destination the platform states.
    Exceptional,
    /// Exceptional transfer from a protected operation to a handler.
    Unwind {
        /// Index of the handler-table entry selecting this edge.
        handler: usize,
    },
    /// Exceptional transfer on one way that the runtime takes after a
    /// protected operation throws: an edge from the protected operation, or
    /// from the return of a funclet, to the next funclet instance or to a
    /// handler block.
    Route {
        /// Index of the first route that states the edge.
        route: usize,
    },
    /// The way back to the protected operation itself, on which the runtime
    /// resumes execution after a filter handled the exception in place
    /// (`EXCEPTION_CONTINUE_EXECUTION`).
    Continue {
        /// Index of the first route that states the edge.
        route: usize,
    },
}

impl<T, K> EdgeRole<'_, T, K> {
    /// The structural kind the graph stores for this role.
    #[must_use]
    pub const fn kind(&self) -> EdgeKind {
        match self {
            Self::Sequential => EdgeKind::Fallthrough,
            Self::ConditionalTaken => EdgeKind::ConditionalTrue,
            Self::ConditionalFallThrough => EdgeKind::ConditionalFalse,
            Self::Jump => EdgeKind::Jump,
            Self::SwitchDefault | Self::SwitchCase { .. } => EdgeKind::SwitchCase,
            Self::Call => EdgeKind::Call,
            Self::CallContinuation { .. } => EdgeKind::CallReturn,
            Self::Exceptional | Self::Unwind { .. } | Self::Route { .. } => {
                EdgeKind::ExceptionUnwind
            }
            Self::Continue { .. } => EdgeKind::ExceptionContinue,
        }
    }

    /// Whether this edge leaves the operation's own function.
    #[must_use]
    pub const fn is_call(&self) -> bool {
        matches!(self, Self::Call)
    }

    /// The role a transfer reports when its target is not recovered, with
    /// the case key dropped so the report outlives the flow.
    ///
    /// A continuation never reports one: a continuation whose next position
    /// is not a recovered operation is dropped without a report. Nor do the
    /// exception-model roles, which the handler table and the unwind routes
    /// add, not an operation's flow.
    #[must_use]
    pub const fn unresolved_role(&self) -> Option<UnresolvedRole> {
        let role = match self {
            Self::Jump => UnresolvedRole::Jump,
            Self::ConditionalTaken => UnresolvedRole::ConditionalTaken,
            Self::SwitchDefault => UnresolvedRole::SwitchDefault,
            Self::SwitchCase { index, .. } => UnresolvedRole::SwitchCase { index: *index },
            Self::Call => UnresolvedRole::Call,
            Self::Exceptional => UnresolvedRole::Exceptional,
            Self::Sequential
            | Self::ConditionalFallThrough
            | Self::CallContinuation { .. }
            | Self::Unwind { .. }
            | Self::Route { .. }
            | Self::Continue { .. } => return None,
        };
        Some(role)
    }
}

/// One transfer a flow makes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Transfer<'k, T, K> {
    /// The named target, or `None` for a computed transfer.
    pub target: Option<T>,
    /// Why the transfer exists.
    pub role: EdgeRole<'k, T, K>,
}

/// The transfers one flow makes, in dispatch order.
#[derive(Debug, Clone)]
pub struct Transfers<'f, T, K> {
    flow: &'f Flow<T, K>,
    /// The position of the operation and the position control continues
    /// at, when the caller asks for the continuation and the stream has one.
    continuation: Option<(T, T)>,
    index: usize,
}

impl<'f, T: Clone, K> Transfers<'f, T, K> {
    fn continuation(&self, role: EdgeRole<'f, T, K>) -> Option<Transfer<'f, T, K>> {
        self.continuation.as_ref().map(|(_, next)| Transfer {
            target: Some(next.clone()),
            role,
        })
    }

    fn call_continuation(&self) -> Option<Transfer<'f, T, K>> {
        self.continuation.as_ref().map(|(source, next)| Transfer {
            target: Some(next.clone()),
            role: EdgeRole::CallContinuation {
                call_site: source.clone(),
            },
        })
    }

    fn named(target: &T, role: EdgeRole<'f, T, K>) -> Transfer<'f, T, K> {
        Transfer {
            target: Some(target.clone()),
            role,
        }
    }

    /// The transfer in slot `index`, or `None` when the slot is empty.
    ///
    /// Slot 0 is the named or computed transfer; slot 1 is the continuation
    /// of a flow that has one; a switch uses slot 0 for its default and one
    /// slot per case after it.
    fn slot(&self, index: usize) -> Option<Transfer<'f, T, K>> {
        let computed = |role| Transfer { target: None, role };
        match (self.flow, index) {
            (Flow::Next, 0) | (Flow::Exceptional { .. } | Flow::ConditionalTrap, 1) => {
                self.continuation(EdgeRole::Sequential)
            }
            (Flow::Indirect, 0) => Some(computed(EdgeRole::Jump)),
            (Flow::IndirectCall, 0) => Some(computed(EdgeRole::Call)),
            (Flow::IndirectCall | Flow::Call { .. }, 1) => self.call_continuation(),
            (Flow::Jump { target }, 0) => Some(Self::named(target, EdgeRole::Jump)),
            (Flow::Conditional { target }, 0) => {
                Some(Self::named(target, EdgeRole::ConditionalTaken))
            }
            (Flow::ConditionalIndirect, 0) => Some(computed(EdgeRole::ConditionalTaken)),
            (Flow::Conditional { .. } | Flow::ConditionalIndirect | Flow::ConditionalReturn, 1) => {
                self.continuation(EdgeRole::ConditionalFallThrough)
            }
            (Flow::Call { target } | Flow::TailCall { target }, 0) => {
                Some(Self::named(target, EdgeRole::Call))
            }
            (Flow::Exceptional { target }, 0) => Some(Self::named(target, EdgeRole::Exceptional)),
            (Flow::Trap | Flow::ConditionalTrap, 0) => Some(computed(EdgeRole::Exceptional)),
            (Flow::Switch { default, .. }, 0) => {
                Some(Self::named(default, EdgeRole::SwitchDefault))
            }
            (Flow::Switch { cases, .. }, index) => {
                let (key, target) = cases.get(index - 1)?;
                Some(Self::named(
                    target,
                    EdgeRole::SwitchCase {
                        index: index - 1,
                        key,
                    },
                ))
            }
            _ => None,
        }
    }
}

impl<'f, T: Clone, K> Iterator for Transfers<'f, T, K> {
    type Item = Transfer<'f, T, K>;

    fn next(&mut self) -> Option<Self::Item> {
        // A flow whose continuation overflowed the position space skips the
        // continuation slot but still yields the transfers after it.
        while self.index < self.slot_count() {
            let item = self.slot(self.index);
            self.index += 1;
            if item.is_some() {
                return item;
            }
        }
        None
    }
}

impl<T, K> Transfers<'_, T, K> {
    fn slot_count(&self) -> usize {
        match self.flow {
            Flow::Return | Flow::Throw => 0,
            Flow::Next
            | Flow::Indirect
            | Flow::Jump { .. }
            | Flow::TailCall { .. }
            | Flow::Trap => 1,
            Flow::ConditionalReturn
            | Flow::IndirectCall
            | Flow::Conditional { .. }
            | Flow::ConditionalIndirect
            | Flow::Call { .. }
            | Flow::Exceptional { .. }
            | Flow::ConditionalTrap => 2,
            Flow::Switch { cases, .. } => cases.len() + 1,
        }
    }
}

/// Why one transfer could not be connected to a recovered operation.
///
/// This is the [`EdgeRole`] of the transfer without its borrowed case key.
/// Whether the transfer was computed or named a target that is not
/// recovered is the `target` of its [`UnresolvedTransfer`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnresolvedRole {
    /// Unconditional jump.
    Jump,
    /// Taken path of a conditional branch.
    ConditionalTaken,
    /// Switch default destination.
    SwitchDefault,
    /// One switch case destination, by its dispatch-order position.
    SwitchCase {
        /// Zero-based case position in [`Flow::Switch`].
        index: usize,
    },
    /// Call target, of a call or of a tail call.
    Call,
    /// Exceptional or alternate target, or the platform destination of a
    /// trap.
    Exceptional,
}

/// One transfer that leaves the recovered operations.
///
/// A named but unavailable destination keeps its identity in `target`; a
/// computed transfer has `None`. The local successors of the same operation
/// stay in the graph.
///
/// `S` identifies the transferring operation: its position for
/// [`build_address_cfg`](crate::build_address_cfg), its block for
/// [`resolve_jump_edges`](crate::resolve_jump_edges).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UnresolvedTransfer<T, S = T> {
    /// The transferring operation.
    pub source: S,
    /// Named destination outside the recovered operations, if any.
    pub target: Option<T>,
    /// Semantic role of the missing successor.
    pub role: UnresolvedRole,
}

/// One call site an intraprocedural graph flattened.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CallSite<T> {
    /// Position of the calling operation.
    pub source: T,
    /// The callee the operation names, or `None` for a computed call.
    pub target: Option<T>,
}

#[cfg(test)]
mod tests {
    extern crate alloc;

    use alloc::string::String;
    use alloc::vec;
    use alloc::vec::Vec;

    use super::{EdgeRole, Flow, Transfer, UnresolvedRole};
    use crate::edge::EdgeKind;

    fn transfers(flow: &Flow<u32, char>) -> Vec<Transfer<'_, u32, char>> {
        flow.transfers(10, Some(11)).collect()
    }

    /// `continues`, `is_call`, `direct_target` and `ends_block` are
    /// hand-written lists over the variants. The transfers an operation makes
    /// are the statement of what it does, so every list must agree with them.
    #[test]
    fn the_flow_predicates_agree_with_the_transfers_of_every_variant() {
        let flows: [Flow<u32, char>; 16] = [
            Flow::Next,
            Flow::Return,
            Flow::ConditionalReturn,
            Flow::Throw,
            Flow::Indirect,
            Flow::IndirectCall,
            Flow::Jump { target: 1 },
            Flow::Conditional { target: 1 },
            Flow::Call { target: 1 },
            Flow::TailCall { target: 1 },
            Flow::Exceptional { target: 1 },
            Flow::ConditionalIndirect,
            Flow::Trap,
            Flow::ConditionalTrap,
            Flow::Switch {
                default: 1,
                cases: vec![],
            },
            Flow::Switch {
                default: 1,
                cases: vec![('a', 2), ('b', 3)],
            },
        ];
        let is_continuation = |role: &EdgeRole<'_, u32, char>| {
            matches!(
                role,
                EdgeRole::Sequential
                    | EdgeRole::CallContinuation { .. }
                    | EdgeRole::ConditionalFallThrough
            )
        };
        for flow in &flows {
            let made = transfers(flow);
            assert_eq!(
                flow.continues(),
                made.iter().any(|transfer| is_continuation(&transfer.role)),
                "{flow:?} continues"
            );
            assert_eq!(
                flow.is_call(),
                made.iter().any(|transfer| transfer.role.is_call()),
                "{flow:?} is a call"
            );
            let named = made
                .iter()
                .filter(|transfer| transfer.target.is_some() && !is_continuation(&transfer.role))
                .count();
            assert_eq!(
                flow.direct_target().is_some(),
                named == 1 && !matches!(flow, Flow::Switch { .. }),
                "{flow:?} names one target"
            );
            assert_eq!(
                flow.ends_block(),
                !matches!(flow, Flow::Next),
                "{flow:?} ends its block"
            );
            let departing: Vec<_> = flow.departures().collect();
            let leaving: Vec<_> = made
                .iter()
                .copied()
                .filter(|transfer| !is_continuation(&transfer.role))
                .collect();
            assert_eq!(departing, leaving, "{flow:?} departs as it transfers");
        }
    }

    #[test]
    fn a_continuation_is_the_only_transfer_of_a_plain_flow() {
        let flow: Flow<u32, char> = Flow::Next;
        assert_eq!(
            transfers(&flow),
            vec![Transfer {
                target: Some(11),
                role: EdgeRole::Sequential
            }]
        );
        assert!(!flow.ends_block());
        assert!(flow.continues());
    }

    #[test]
    fn leaving_flows_make_no_transfer() {
        for flow in [Flow::<u32, char>::Return, Flow::Throw] {
            assert_eq!(
                transfers(&flow),
                [] as [crate::flow::transfer::Transfer<'_, u32, char>; 0]
            );
            assert!(flow.ends_block());
            assert!(!flow.continues());
        }
    }

    #[test]
    fn a_conditional_return_leaves_by_no_edge_and_falls_through_otherwise() {
        let flow: Flow<u32, char> = Flow::ConditionalReturn;
        assert_eq!(
            transfers(&flow),
            vec![Transfer {
                target: Some(11),
                role: EdgeRole::ConditionalFallThrough
            }]
        );
        assert_eq!(flow.departures().count(), 0);
    }

    #[test]
    fn a_tail_call_calls_its_target_and_does_not_continue() {
        let flow: Flow<u32, char> = Flow::TailCall { target: 40 };
        assert_eq!(
            transfers(&flow),
            vec![Transfer {
                target: Some(40),
                role: EdgeRole::Call
            }]
        );
        assert!(flow.is_call());
        assert!(!flow.continues());
        assert_eq!(flow.direct_target(), Some(40));
    }

    #[test]
    fn computed_transfers_have_no_target() {
        let jump: Flow<u32, char> = Flow::Indirect;
        assert_eq!(transfers(&jump)[0].target, None);
        assert_eq!(transfers(&jump)[0].role, EdgeRole::Jump);
        assert_eq!(
            transfers(&jump)[0].role.unresolved_role(),
            Some(UnresolvedRole::Jump)
        );

        let call: Flow<u32, char> = Flow::IndirectCall;
        let all = transfers(&call);
        assert_eq!(all[0].target, None);
        assert!(all[0].role.is_call());
        assert_eq!(all[0].role.unresolved_role(), Some(UnresolvedRole::Call));
        assert_eq!(all[1].role, EdgeRole::CallContinuation { call_site: 10 });
        assert_eq!(all[1].role.unresolved_role(), None);
        assert!(call.is_call());
    }

    #[test]
    fn a_conditional_computed_transfer_keeps_its_fall_through() {
        let flow: Flow<u32, char> = Flow::ConditionalIndirect;
        let all = transfers(&flow);
        assert_eq!(all.len(), 2);
        assert_eq!(all[0].target, None);
        assert_eq!(all[0].role, EdgeRole::ConditionalTaken);
        assert_eq!(
            all[0].role.unresolved_role(),
            Some(UnresolvedRole::ConditionalTaken)
        );
        assert_eq!(all[1].target, Some(11));
        assert_eq!(all[1].role, EdgeRole::ConditionalFallThrough);
        assert_eq!(all[1].role.unresolved_role(), None);
    }

    #[test]
    fn a_trap_reports_its_platform_destination_and_continues_only_when_conditional() {
        let trap: Flow<u32, char> = Flow::Trap;
        assert_eq!(
            transfers(&trap),
            vec![Transfer {
                target: None,
                role: EdgeRole::Exceptional
            }]
        );
        assert_eq!(
            transfers(&trap)[0].role.unresolved_role(),
            Some(UnresolvedRole::Exceptional)
        );
        assert_eq!(transfers(&trap)[0].role.kind(), EdgeKind::ExceptionUnwind);

        let conditional: Flow<u32, char> = Flow::ConditionalTrap;
        let all = transfers(&conditional);
        assert_eq!(all.len(), 2);
        assert_eq!(all[0].target, None);
        assert_eq!(
            all[0].role.unresolved_role(),
            Some(UnresolvedRole::Exceptional)
        );
        assert_eq!(all[1].target, Some(11));
        assert_eq!(all[1].role, EdgeRole::Sequential);
    }

    #[test]
    fn named_transfers_come_before_their_continuation() {
        let flow: Flow<u32, char> = Flow::Conditional { target: 40 };
        let all = transfers(&flow);
        assert_eq!(all[0].target, Some(40));
        assert_eq!(all[0].role, EdgeRole::ConditionalTaken);
        assert_eq!(all[1].target, Some(11));
        assert_eq!(all[1].role, EdgeRole::ConditionalFallThrough);
        assert_eq!(flow.direct_target(), Some(40));

        let flow: Flow<u32, char> = Flow::Exceptional { target: 40 };
        let all = transfers(&flow);
        assert_eq!(all[0].role, EdgeRole::Exceptional);
        assert_eq!(all[0].role.kind(), EdgeKind::ExceptionUnwind);
        assert_eq!(all[1].role, EdgeRole::Sequential);
    }

    #[test]
    fn an_overflowing_continuation_is_skipped_not_invented() {
        let flow: Flow<u32, char> = Flow::Call { target: 40 };
        let all: Vec<_> = flow.transfers(10, None).collect();
        assert_eq!(all.len(), 1);
        assert_eq!(all[0].role, EdgeRole::Call);
    }

    #[test]
    fn switch_cases_keep_their_dispatch_order_and_keys() {
        let flow = Flow::Switch {
            default: 50,
            cases: vec![('a', 60), ('b', 70)],
        };
        let all = transfers(&flow);
        assert_eq!(all.len(), 3);
        assert_eq!(all[0].role, EdgeRole::SwitchDefault);
        assert_eq!(
            all[2].role,
            EdgeRole::SwitchCase {
                index: 1,
                key: &'b'
            }
        );
        assert_eq!(all[2].target, Some(70));
        assert_eq!(
            all[2].role.unresolved_role(),
            Some(UnresolvedRole::SwitchCase { index: 1 })
        );
        assert_eq!(flow.direct_target(), None);
    }

    /// A label token of a structured stream need not be `Copy`; the
    /// transfers hand out clones of the targets the flow owns.
    #[test]
    fn owned_targets_are_cloned_into_the_transfers() {
        let flow: Flow<String, char> = Flow::Conditional {
            target: String::from("retry"),
        };
        let all: Vec<_> = flow
            .transfers(String::from("here"), Some(String::from("next")))
            .map(|transfer| (transfer.target, transfer.role.kind()))
            .collect();
        assert_eq!(
            all,
            [
                (Some(String::from("retry")), EdgeKind::ConditionalTrue),
                (Some(String::from("next")), EdgeKind::ConditionalFalse),
            ]
        );
    }

    #[test]
    fn every_role_names_one_structural_kind() {
        let roles: [EdgeRole<'_, u32, char>; 7] = [
            EdgeRole::Sequential,
            EdgeRole::Jump,
            EdgeRole::Call,
            EdgeRole::CallContinuation { call_site: 1 },
            EdgeRole::Unwind { handler: 0 },
            EdgeRole::Route { route: 0 },
            EdgeRole::Continue { route: 0 },
        ];
        let kinds: Vec<EdgeKind> = roles.iter().map(EdgeRole::kind).collect();
        assert_eq!(
            kinds,
            [
                EdgeKind::Fallthrough,
                EdgeKind::Jump,
                EdgeKind::Call,
                EdgeKind::CallReturn,
                EdgeKind::ExceptionUnwind,
                EdgeKind::ExceptionUnwind,
                EdgeKind::ExceptionContinue,
            ]
        );
    }
}
