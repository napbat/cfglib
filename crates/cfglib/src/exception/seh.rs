//! Normalized Windows frame-based SEH metadata.

extern crate alloc;

use alloc::collections::BTreeSet;
use alloc::vec::Vec;

use crate::block::BlockId;
use crate::cfg::Cfg;
use crate::region::{Handler, HandlerBody, HandlerKind, Region, RegionId};

/// Windows SEH scope-handler kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum SehHandlerKind {
    /// A `__except` handler selected by a three-way filter expression.
    Except {
        /// Block evaluating the filter expression.
        filter_block: BlockId,
    },
    /// A `__finally` termination handler.
    Finally,
}

/// One normalized handler in a Windows SEH scope table.
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SehHandler {
    /// Entry block of the handler body.
    pub entry: BlockId,
    /// The handler body extent — [`HandlerBody::Unknown`] when the format
    /// encodes only the handler entry.
    pub body: HandlerBody,
    /// Native SEH handler kind.
    pub kind: SehHandlerKind,
}

/// One normalized Windows SEH guarded region.
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SehExceptionRegion {
    /// Blocks covered by the guarded `__try` body.
    pub protected_blocks: BTreeSet<BlockId>,
    /// Scope-table handlers in dispatch order.
    pub handlers: Vec<SehHandler>,
    /// Enclosing region, imported before this one, when nested.
    pub parent: Option<RegionId>,
}

/// Install a normalized frame-based SEH region into `cfg`.
///
/// `__except` becomes [`HandlerKind::Filter`] and `__finally` becomes
/// [`HandlerKind::Finally`]. Use [`ExceptionFlow`](crate::ExceptionFlow) on
/// the corresponding CFG edges to retain the filter's execute/search/continue
/// disposition and consumer-owned `EXCEPTION_RECORD`/`CONTEXT` identity.
pub fn install_seh_region<I, E>(cfg: &mut Cfg<I, E>, region: SehExceptionRegion) -> RegionId {
    let handlers = region
        .handlers
        .into_iter()
        .map(|handler| Handler {
            entry: handler.entry,
            body: handler.body,
            kind: match handler.kind {
                SehHandlerKind::Except { filter_block } => HandlerKind::Filter { filter_block },
                SehHandlerKind::Finally => HandlerKind::Finally,
            },
        })
        .collect();

    cfg.add_region(Region {
        id: RegionId::from_raw(0),
        protected_blocks: region.protected_blocks,
        handlers,
        parent: region.parent,
    })
}

/// One active x86 frame-based SEH registration.
///
/// `F` is a consumer-owned frame/registration identity and `H` is the native
/// language-handler identity. One registration can reference several lexical
/// [`Region`]s through a compiler scope table.
///
/// [`Region`]: crate::Region
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SehRegistration<F, H> {
    /// Stack-frame or registration-record identity.
    pub frame: F,
    /// Native language-handler function identity.
    pub handler: H,
    /// CFG regions described by this registration's scope table.
    pub regions: Vec<RegionId>,
}

/// Active x86 SEH registrations for one thread.
///
/// Registrations are pushed as frames become active. [`Self::dispatch_order`]
/// walks the newest registration first, matching stack search order.
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SehRegistrationChain<F, H> {
    registrations: Vec<SehRegistration<F, H>>,
}

impl<F, H> SehRegistrationChain<F, H> {
    /// Construct an empty registration chain.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            registrations: Vec::new(),
        }
    }

    /// Register a newly active frame at the head of dispatch order.
    pub fn push(&mut self, registration: SehRegistration<F, H>) {
        self.registrations.push(registration);
    }

    /// Unregister and return the most recently activated frame.
    pub fn pop(&mut self) -> Option<SehRegistration<F, H>> {
        self.registrations.pop()
    }

    /// Iterate from the innermost/newest registration outward.
    pub fn dispatch_order(&self) -> impl DoubleEndedIterator<Item = &SehRegistration<F, H>> {
        self.registrations.iter().rev()
    }

    /// Number of active registrations.
    #[must_use]
    pub fn len(&self) -> usize {
        self.registrations.len()
    }

    /// Whether the chain has no active registrations.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.registrations.is_empty()
    }
}

impl<F, H> Default for SehRegistrationChain<F, H> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn seh_scope_and_registration_order_remain_distinct() {
        let mut cfg = Cfg::<()>::new();
        let filter = cfg.new_block();
        let handler = cfg.new_block();
        let entry = cfg.entry();
        let region = install_seh_region(
            &mut cfg,
            SehExceptionRegion {
                protected_blocks: [entry].into_iter().collect(),
                handlers: alloc::vec![SehHandler {
                    entry: handler,
                    body: HandlerBody::known([handler]),
                    kind: SehHandlerKind::Except {
                        filter_block: filter,
                    },
                }],
                parent: None,
            },
        );

        let mut chain = SehRegistrationChain::new();
        chain.push(SehRegistration {
            frame: 0x1000_u32,
            handler: 0x0040_1000_u32,
            regions: alloc::vec![region],
        });
        chain.push(SehRegistration {
            frame: 0x0F00_u32,
            handler: 0x0040_2000_u32,
            regions: Vec::new(),
        });

        assert_eq!(
            chain
                .dispatch_order()
                .map(|entry| entry.frame)
                .collect::<Vec<_>>(),
            alloc::vec![0x0F00, 0x1000]
        );
        assert_eq!(
            cfg.regions()[0].handlers[0].kind,
            HandlerKind::Filter {
                filter_block: filter
            }
        );
    }
}
