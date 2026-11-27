//! The serialized form of a [`Cfg`] and the references a document must keep
//! true before it becomes one.
//!
//! A document is untrusted input: every later use of the CFG assumes the
//! entry is a live block and that the exception metadata keeps the
//! invariants [`Cfg::check_regions`] states ([`Cfg::set_entry`] asserts the
//! first, [`Cfg::compact`] requires the second), so a document that breaks
//! either is an error at decode rather than a panic at first use.

extern crate alloc;
use alloc::vec::Vec;

use crate::block::BlockId;
use crate::region::{Cleanup, HandlerMetadata, Region, RegionError};

use super::{Cfg, CfgStore};

/// The fields a [`Cfg`] serializes, before they are checked.
#[derive(serde::Deserialize)]
#[serde(bound(deserialize = "I: serde::Deserialize<'de>, E: serde::Deserialize<'de>"))]
pub(super) struct CfgDocument<I, E> {
    graph: CfgStore<I, E>,
    entry: BlockId,
    regions: Vec<Region>,
    /// Absent from documents written before cleanup records existed.
    #[serde(default)]
    cleanups: HandlerMetadata<Cleanup>,
}

/// Why a serialized CFG was rejected.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum DocumentError {
    /// The entry names no live block of the graph.
    EntryNotLive(BlockId),
    /// The exception metadata breaks an invariant of [`Cfg::check_regions`].
    Regions(RegionError),
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
            Self::Regions(error) => error.fmt(formatter),
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
        cfg.check_regions().map_err(DocumentError::Regions)?;
        Ok(cfg)
    }
}

#[cfg(test)]
mod tests {
    extern crate std;

    use alloc::string::ToString;
    use alloc::vec;

    use serde_json::{Value, json};

    use crate::edge::EdgeKind;
    use crate::region::{
        CompletionReason, Continuation, Handler, HandlerBody, HandlerKind, HandlerRef, Region,
        RegionId,
    };

    use super::*;

    fn protected_cfg() -> Cfg<u32> {
        let mut cfg = Cfg::<u32>::new();
        let body = cfg.entry();
        let handler = cfg.new_block();
        cfg.add_edge(body, handler, EdgeKind::ExceptionHandler);
        cfg.add_region(Region {
            protected_blocks: [body].into_iter().collect(),
            handlers: vec![Handler {
                entry: handler,
                body: HandlerBody::known([handler]),
                kind: HandlerKind::CatchAll,
            }],
            parent: None,
        })
        .unwrap();
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
        let entry = cfg.handler(handler).entry;
        cfg.set_cleanup_exit(handler, entry).unwrap();
        let mut document = serde_json::to_value(&cfg).unwrap();
        assert_eq!(
            decode(&document).unwrap(),
            cfg,
            "the record is valid as written"
        );
        document["cleanups"]["entries"][0][0]["handler"] = json!(9);
        let error = decode(&document).unwrap_err();
        assert!(error.to_string().contains("cleanup"), "{error}");
    }

    #[test]
    fn a_cleanup_listing_one_route_twice_is_rejected() {
        let mut cfg = protected_cfg();
        let handler = HandlerRef::new(RegionId::from_raw(0), 0);
        let entry = cfg.entry();
        for reason in [CompletionReason::Normal, CompletionReason::Return] {
            cfg.add_continuation(
                handler,
                Continuation {
                    reason,
                    target: entry,
                },
            )
            .unwrap();
        }
        let mut document = serde_json::to_value(&cfg).unwrap();
        assert_eq!(
            decode(&document).unwrap(),
            cfg,
            "two distinct routes are valid as written"
        );
        let routes = document["cleanups"]["entries"][0][1]["continuations"]
            .as_array_mut()
            .unwrap();
        let first = routes[0].clone();
        routes.push(first);
        let error = decode(&document).unwrap_err();
        assert!(error.to_string().contains("twice"), "{error}");
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
