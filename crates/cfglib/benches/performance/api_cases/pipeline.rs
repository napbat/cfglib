use std::convert::Infallible;

use cfglib::{
    Cfg, PassChange, PassId, PassPipeline, merge_blocks, pass_fn, remove_empty_blocks, verify,
};

use super::BenchmarkSuite;
use crate::fixtures::linear_cfg;
use crate::harness::benchmark_case;

const BLOCK_COUNT: usize = 256;

pub(super) fn register(suite: &mut BenchmarkSuite<'_>) {
    let linear = linear_cfg(BLOCK_COUNT);
    benchmark_case!(
        suite,
        "api_pass_fn",
        covers[pass_fn],
        || {
            let mut candidate = linear.clone();
            let mut pipeline = PassPipeline::<Cfg<u32>, Infallible>::new();
            pipeline.push(pass_fn(PassId::new("bench.merge"), |cfg: &mut Cfg<u32>| {
                Ok(PassChange::from_changed(merge_blocks(cfg) > 0))
            }));
            pipeline.push(pass_fn(
                PassId::new("bench.remove-empty"),
                |cfg: &mut Cfg<u32>| Ok(PassChange::from_changed(remove_empty_blocks(cfg) > 0)),
            ));
            let report = pipeline.run_infallible(&mut candidate);
            (candidate, report)
        },
        |(candidate, report)| {
            assert_eq!(report.len(), 2);
            assert_eq!(report.changed_count(), 1);
            assert!(verify(candidate).is_ok());
        }
    );
}
