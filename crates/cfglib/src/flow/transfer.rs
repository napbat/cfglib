//! How one operation leaves its position, and the edges that describes.
//!
//! [`Flow`] is the one vocabulary for an operation's control transfer. A
//! machine instruction, a bytecode operation, and a lowered source statement
//! all state the same thing: whether control continues at the next
//! position, which targets it names, and how it leaves. Everything that
//! turns a stream of operations into a graph reads that statement through
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
    /// Leaves the function exceptionally, through the enclosing regions.
    Throw,
    /// Transfers to a target the operation does not name.
    Indirect,
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
                | Self::IndirectCall
                | Self::Conditional { .. }
                | Self::Call { .. }
                | Self::Exceptional { .. }
        )
    }

    /// Whether the operation calls another function.
    #[must_use]
    pub const fn is_call(&self) -> bool {
        matches!(self, Self::Call { .. } | Self::IndirectCall)
    }

    /// The single target the operation names directly, if it names one.
    ///
    /// A jump, a conditional branch, a call, and an exceptional transfer
    /// name one target. A switch names several, and the rest name none.
    #[must_use]
    pub fn direct_target(&self) -> Option<T>
    where
        T: Copy,
    {
        match self {
            Self::Jump { target }
            | Self::Conditional { target }
            | Self::Call { target }
            | Self::Exceptional { target } => Some(*target),
            Self::Next
            | Self::Return
            | Self::Throw
            | Self::Indirect
            | Self::IndirectCall
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
        T: Copy,
    {
        Transfers {
            flow: self,
            source,
            next,
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
    /// Direct call transfer.
    Call,
    /// Continuation after a call returns.
    CallContinuation {
        /// Position of the calling operation.
        call_site: T,
    },
    /// Exceptional or alternate transfer the operation names directly.
    Exceptional,
    /// Exceptional transfer from a protected operation to a handler.
    Unwind {
        /// Index of the handler-table entry selecting this edge.
        handler: usize,
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
            Self::Exceptional | Self::Unwind { .. } => EdgeKind::ExceptionUnwind,
        }
    }

    /// Whether this edge leaves the operation's own function.
    #[must_use]
    pub const fn is_call(&self) -> bool {
        matches!(self, Self::Call)
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
    source: T,
    next: Option<T>,
    index: usize,
}

impl<'f, T: Copy, K> Transfers<'f, T, K> {
    fn continuation(&self, role: EdgeRole<'f, T, K>) -> Option<Transfer<'f, T, K>> {
        self.next.map(|target| Transfer {
            target: Some(target),
            role,
        })
    }

    fn named(target: T, role: EdgeRole<'f, T, K>) -> Transfer<'f, T, K> {
        Transfer {
            target: Some(target),
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
        let call_continuation = EdgeRole::CallContinuation {
            call_site: self.source,
        };
        match (self.flow, index) {
            (Flow::Next, 0) | (Flow::Exceptional { .. }, 1) => {
                self.continuation(EdgeRole::Sequential)
            }
            (Flow::Indirect, 0) => Some(computed(EdgeRole::Jump)),
            (Flow::IndirectCall, 0) => Some(computed(EdgeRole::Call)),
            (Flow::IndirectCall | Flow::Call { .. }, 1) => self.continuation(call_continuation),
            (Flow::Jump { target }, 0) => Some(Self::named(*target, EdgeRole::Jump)),
            (Flow::Conditional { target }, 0) => {
                Some(Self::named(*target, EdgeRole::ConditionalTaken))
            }
            (Flow::Conditional { .. }, 1) => self.continuation(EdgeRole::ConditionalFallThrough),
            (Flow::Call { target }, 0) => Some(Self::named(*target, EdgeRole::Call)),
            (Flow::Exceptional { target }, 0) => Some(Self::named(*target, EdgeRole::Exceptional)),
            (Flow::Switch { default, .. }, 0) => {
                Some(Self::named(*default, EdgeRole::SwitchDefault))
            }
            (Flow::Switch { cases, .. }, index) => {
                let (key, target) = cases.get(index - 1)?;
                Some(Self::named(
                    *target,
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

impl<'f, T: Copy, K> Iterator for Transfers<'f, T, K> {
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
            Flow::Next | Flow::Indirect | Flow::Jump { .. } => 1,
            Flow::IndirectCall
            | Flow::Conditional { .. }
            | Flow::Call { .. }
            | Flow::Exceptional { .. } => 2,
            Flow::Switch { cases, .. } => cases.len() + 1,
        }
    }
}

/// Why one transfer could not be connected to a recovered operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnresolvedRole {
    /// A computed jump with no statically named target.
    Indirect,
    /// A computed call with no statically named target.
    IndirectCall,
    /// Taken path of a conditional branch.
    ConditionalTaken,
    /// Direct unconditional jump.
    Jump,
    /// Switch default destination.
    SwitchDefault,
    /// One switch case destination, by its dispatch-order position.
    SwitchCase {
        /// Zero-based case position in [`Flow::Switch`].
        index: usize,
    },
    /// Direct call target.
    Call,
    /// Exceptional or alternate target the operation names.
    Exceptional,
}

impl<T, K> Transfer<'_, T, K> {
    /// The role this transfer reports when its target is not recovered.
    ///
    /// A continuation and an unwind always target a recovered operation,
    /// so they have no unresolved form and return `None`.
    #[must_use]
    pub const fn unresolved_role(&self) -> Option<UnresolvedRole> {
        let role = match (&self.role, self.target.is_some()) {
            (EdgeRole::Jump, false) => UnresolvedRole::Indirect,
            (EdgeRole::Call, false) => UnresolvedRole::IndirectCall,
            (EdgeRole::ConditionalTaken, true) => UnresolvedRole::ConditionalTaken,
            (EdgeRole::Jump, true) => UnresolvedRole::Jump,
            (EdgeRole::SwitchDefault, true) => UnresolvedRole::SwitchDefault,
            (EdgeRole::SwitchCase { index, .. }, true) => {
                UnresolvedRole::SwitchCase { index: *index }
            }
            (EdgeRole::Call, true) => UnresolvedRole::Call,
            (EdgeRole::Exceptional, true) => UnresolvedRole::Exceptional,
            _ => return None,
        };
        Some(role)
    }
}

/// One transfer that leaves the recovered operations.
///
/// A named but unavailable destination keeps its identity in `target`; a
/// computed transfer has `None`. The local successors of the same operation
/// stay in the graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UnresolvedTransfer<T> {
    /// Position of the transferring operation.
    pub source: T,
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

    use alloc::vec;
    use alloc::vec::Vec;

    use super::{EdgeRole, Flow, Transfer, UnresolvedRole};
    use crate::edge::EdgeKind;

    fn transfers(flow: &Flow<u32, char>) -> Vec<Transfer<'_, u32, char>> {
        flow.transfers(10, Some(11)).collect()
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
            assert!(transfers(&flow).is_empty());
            assert!(flow.ends_block());
            assert!(!flow.continues());
        }
    }

    #[test]
    fn computed_transfers_have_no_target() {
        let jump: Flow<u32, char> = Flow::Indirect;
        assert_eq!(transfers(&jump)[0].target, None);
        assert_eq!(transfers(&jump)[0].role, EdgeRole::Jump);
        assert_eq!(
            transfers(&jump)[0].unresolved_role(),
            Some(UnresolvedRole::Indirect)
        );

        let call: Flow<u32, char> = Flow::IndirectCall;
        let all = transfers(&call);
        assert_eq!(all[0].target, None);
        assert!(all[0].role.is_call());
        assert_eq!(all[0].unresolved_role(), Some(UnresolvedRole::IndirectCall));
        assert_eq!(all[1].role, EdgeRole::CallContinuation { call_site: 10 });
        assert_eq!(all[1].unresolved_role(), None);
        assert!(call.is_call());
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
            all[2].unresolved_role(),
            Some(UnresolvedRole::SwitchCase { index: 1 })
        );
        assert_eq!(flow.direct_target(), None);
    }

    #[test]
    fn every_role_names_one_structural_kind() {
        let roles: [EdgeRole<'_, u32, char>; 5] = [
            EdgeRole::Sequential,
            EdgeRole::Jump,
            EdgeRole::Call,
            EdgeRole::CallContinuation { call_site: 1 },
            EdgeRole::Unwind { handler: 0 },
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
            ]
        );
    }
}
