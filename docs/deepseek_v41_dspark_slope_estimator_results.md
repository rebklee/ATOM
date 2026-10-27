# Online DSpark SPS: window-fit slope, TP2 three-arm comparison

One concurrent observation comparing the static cost curve, the previous
online estimator (slope only from same-request-set cohorts) and the current
one (DeepInfer-style window fit with a separate intercept). Not a
repeatability estimate.

## Configuration

- DeepSeek-V4.1-Flash, MI355X, one TP2 instance per arm on disjoint GPU pairs
  of one host, run simultaneously. `TR_HYBRID` level 0, query buckets 1/3/6,
  confidence scheduling, ragged verify, `bucket_global=true`,
  `ATOM_PREFILL_DECODE_INTERVAL=10`, memory utilization 0.7, 128 sequences.
- AgentX (`semianalysis_cc_traces_weka_062126`), concurrency 32, 900 seconds,
  real sampling (forced acceptance cannot be combined with the scheduler).
- Arms: static (`online_sps=false`), previous estimator (`online_sps=true`
  at 523e3e9e6), current estimator (`online_sps=true`). One shared TP2 profile,
  measured on an earlier image and relabelled for `TR_HYBRID` level 0, so the
  absolute fixed offsets below are larger than the earlier 5 ms; all three
  arms use the same table.

## Results

| Metric | Static | Previous online | Current online |
| --- | ---: | ---: | ---: |
| Completed requests | 1293 | 1309 | 1311 |
| Client output tokens/s | 1136.1 | 1158.9 (+2.0%) | 1169.5 (+2.9%) |
| Server generation tokens/s (mean of 10 s samples) | 972.9 | 993.4 (+2.1%) | 993.1 (+2.1%) |
| Mean ITL, ms | 15.2 | 14.0 | 13.8 |
| Mean TTFT, ms | 1329 | 1279 | 1256 |
| P50 TTFT, ms | 611 | 590 | 594 |

Online correction improves on the static curve by about 2-3% output and
8-9% ITL. The two online estimators are within noise of each other.

## What the estimators learned

- Previous: final global slope 0.007 ms/row (batch 17: 0.014), fixed 15.6 ms,
  post-target 3.9 ms.
- Current: 26 publications, 7 with a nonzero global slope (max 0.028 ms/row),
  final slope 0 over 621 supporting samples; fixed 17.3 ms, post-target 4.0 ms.
- Graph sizes concentrate around batches 12-18. Per-batch fixed offsets fall
  as the batch grows (about 17 ms at batch 4, 11 ms at batches 12-14), so the
  per-size residual medians slope down and the fit clamps to zero.

The gain over the static curve comes from the fixed and post-target terms.
On this workload the marginal per-row cost beyond the startup curve is not
measurable; the window fit can now learn one where the graph-size spread and
the cost difference exist, which the cohort rule could not.
