//! The serialized form of a [`Cfg`] and the references a document must keep
//! true before it becomes one.
//!
//! A document is untrusted input: every later use of the CFG assumes the
//! entry is a live block and that the exception metadata names blocks that
//! exist ([`Cfg::set_entry`] asserts the first, [`Cfg::compact`] requires the
//! second), so a document that breaks either is an error at decode rather than
//! a panic at first use.

extern crate alloc;
use alloc::vec::Vec;

use crate::block::BlockId;
use crate::region::{Cleanup, HandlerRef, Region, RegionId};

use super::{Cfg, CfgStore};

/// The fields a [`Cfg`] serializes, before they are checked.
#[derive(serde::Deserialize)]
#[serde(bound(deserialize = "I: serde::Deserialize<'de>, E: serde::Deserialize<'de>"))]
pub(super) struct CfgDocument<I, E> {
    graph: CfgStore<I, E>,
    entry: BlockId,
    regions: Vec<Region>,
    /// Absent from documents written before cleanup records existed.
    #[serde(default = "Vec::new")]
    cleanups: Vec<Cleanup>,
}

/// Why a serialized CFG was rejected.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum DocumentError {
    /// The entry names no live block of the graph.
    EntryNotLive(BlockId),
    /// A handler entry, filter block, cleanup resume point, or continuation
    /// target names no live block of the graph.
    AnchorNotLive(BlockId),
    /// The region at `index` of the region list carries a different identity.
    /// Regions are looked up by position, so the two must agree.
    RegionIdMismatch {
        /// Position in the region list.
        index: usize,
        /// The identity the document stored there.
        id: RegionId,
    },
    /// A cleanup record names a handler that no region holds.
    CleanupHandlerMissing(HandlerRef),
    /// A region names a parent that is not a region of the graph.
    ParentMissing(RegionId),
}

impl core::fmt::Display for DocumentError {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::EntryNotLive(block) => {
                write!(
                    formatter,
                    "the entry {block} is not a live block of the graph"
                )
            }
            Self::AnchorNotLive(block) => write!(
                formatter,
                "exception metadata names {block}, which is not a live block of the graph"
            ),
            Self::RegionIdMismatch { index, id } => write!(
                formatter,
                "the region at position {index} carries identity {id}"
            ),
            Self::CleanupHandlerMissing(handler) => write!(
                formatter,
                "a cleanup record names handler {handler}, which no region holds"
            ),
            Self::ParentMissing(parent) => {
                write!(
                    formatter,
                    "a region names parent {parent}, which does not exist"
                )
            }
        }
    }
}

impl<I, E> TryFrom<CfgDocument<I, E>> for Cfg<I, E> {
    type Error = DocumentError;

    fn try_from(document: CfgDocument<I, E>) -> Result<Self, DocumentError> {
        let cfg = Self {
            graph: document.graph,
            entry: document.entry,
            regions: document.regions,
            cleanups: document.cleanups,
        };
        if !cfg.contains_block(cfg.entry) {
            return Err(DocumentError::EntryNotLive(cfg.entry));
        }
        if let Some(anchor) = cfg.anchors().find(|&anchor| !cfg.contains_block(anchor)) {
            return Err(DocumentError::AnchorNotLive(anchor));
        }
        cfg.check_region_identities()?;
        Ok(cfg)
    }
}

impl<I, E> Cfg<I, E> {
    /// Regions are addressed by position, cleanups by handler, and nesting by
    /// parent identity: each of those references must resolve.
    fn check_region_identities(&self) -> Result<(), DocumentError> {
        for (index, region) in self.regions.iter().enumerate() {
            if region.id.index() != index {
                return Err(DocumentError::RegionIdMismatch {
                    index,
                    id: region.id,
                });
            }
            if let Some(parent) = region.parent
                && self.region(parent).is_none()
            {
                return Err(DocumentError::ParentMissing(parent));
            }
        }
        match self
            .cleanups
            .iter()
            .find(|cleanup| self.handler(cleanup.handler).is_none())
        {
            Some(cleanup) => Err(DocumentError::CleanupHandlerMissing(cleanup.handler)),
            None => Ok(()),
        }
    }
}

#[cfg(test)]
mod tests {
    extern crate std;

    use alloc::string::ToString;
    use alloc::vec;

    use serde_json::{Value, json};

    use crate::edge::EdgeKind;
    use crate::region::{Handler, HandlerBody, HandlerKind, Region, RegionId};

    use super::*;

    fn protected_cfg() -> Cfg<u32> {
        let mut cfg = Cfg::<u32>::new();
        let body = cfg.entry();
        let handler = cfg.new_block();
        cfg.add_edge(body, handler, EdgeKind::ExceptionHandler);
        cfg.add_region(Region {
            id: RegionId::from_raw(0),
            protected_blocks: [body].into_iter().collect(),
            handlers: vec![Handler {
                entry: handler,
                body: HandlerBody::known([handler]),
                kind: HandlerKind::CatchAll,
            }],
            parent: None,
        });
        cfg
    }

    fn decode(document: &Value) -> Result<Cfg<u32>, serde_json::Error> {
        serde_json::from_value(document.clone())
    }

    #[test]
    fn a_written_cfg_reads_back_equal() {
        let cfg = protected_cfg();
        let document = serde_json::to_value(&cfg).unwrap();
        assert_eq!(decode(&document).unwrap(), cfg);
    }

    #[test]
    fn a_document_without_cleanup_records_still_loads() {
        let cfg = protected_cfg();
        let mut document = serde_json::to_value(&cfg).unwrap();
        document.as_object_mut().unwrap().remove("cleanups");
        assert_eq!(decode(&document).unwrap(), cfg);
    }

    #[test]
    fn an_entry_that_is_not_a_live_block_is_rejected() {
        let mut cfg = protected_cfg();
        let removed = cfg.new_block();
        cfg.remove_block(removed);
        let mut document = serde_json::to_value(&cfg).unwrap();
        for entry in [json!(removed.index()), json!(10_000), json!(u32::MAX)] {
            document["entry"] = entry;
            let error = decode(&document).unwrap_err();
            assert!(error.to_string().contains("entry"), "{error}");
        }
    }

    #[test]
    fn a_region_stored_under_another_identity_is_rejected() {
        let cfg = protected_cfg();
        let mut document = serde_json::to_value(&cfg).unwrap();
        document["regions"][0]["id"] = json!(5);
        let error = decode(&document).unwrap_err();
        assert!(error.to_string().contains("identity"), "{error}");
    }

    #[test]
    fn a_region_with_a_missing_parent_is_rejected() {
        let cfg = protected_cfg();
        let mut document = serde_json::to_value(&cfg).unwrap();
        document["regions"][0]["parent"] = json!(7);
        let error = decode(&document).unwrap_err();
        assert!(error.to_string().contains("parent"), "{error}");
    }

    #[test]
    fn a_cleanup_naming_a_missing_handler_is_rejected() {
        let mut cfg = protected_cfg();
        let handler = HandlerRef::new(RegionId::from_raw(0), 0);
        cfg.set_cleanup_resume(handler, cfg.regions()[0].handlers[0].entry);
        let mut document = serde_json::to_value(&cfg).unwrap();
        assert_eq!(
            decode(&document).unwrap(),
            cfg,
            "the record is valid as written"
        );
        document["cleanups"][0]["handler"]["handler"] = json!(9);
        let error = decode(&document).unwrap_err();
        assert!(error.to_string().contains("cleanup"), "{error}");
    }

    #[test]
    fn a_handler_naming_a_missing_block_is_rejected() {
        let cfg = protected_cfg();
        let mut document = serde_json::to_value(&cfg).unwrap();
        document["regions"][0]["handlers"][0]["entry"] = json!(10_000);
        let error = decode(&document).unwrap_err();
        assert!(error.to_string().contains("exception metadata"), "{error}");
    }
}
