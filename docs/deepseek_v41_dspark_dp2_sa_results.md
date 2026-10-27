# DSpark DP2: concurrent 128-lane SA A/B

**This configuration did not show a throughput gain from online SPS.** Server
output throughput per GPU decreased 6.45% in the 900-second window. AIPerf's
completed-request output throughput decreased 2.03%, and mean request TPOT
increased 13.50%. Mean TTFT improved, but the latency distribution and cache
trajectories differ; this is not an overall performance improvement.

## Configuration and execution

- Measured production commit: `eb96d53beafd100458a258760b90e6a298053218`.
- DeepSeek-V4.1-Flash, two MI355X GPUs per arm, TP1 x DP2 with DP attention,
  no explicit EP, `TR_HYBRID` level 0, memory utilization 0.7, maximum 128
  requests per rank, five speculative tokens, query buckets 1/3/6. The DP
  token-graph cap is 768 so the configured 128 x 6 shape is executable.
- Same SA AgentX scenario, 393 traces, seed 42, concurrency 128. Both arms
  use the same joint DP planner (`bucket_global=true`); only `online_sps`
  changes from false to true. Sampling/update/window settings are 8/64/5000.
- Both arms run simultaneously on disjoint GPU pairs on one host. CPU, host
  memory and storage are shared. A resource snapshot showed all four GPUs at
  99-100% utilization and approximately 218 GB VRAM each.
- Both complete all 398 SA warmup requests, with zero cancellations/errors.
  These warmups take 989.86/993.59 seconds, excluded from measurement.
- On 2026-10-26 UTC, static profiling starts at 14:12:00.880 and online at
  14:11:59.896. Each sends for 900 seconds, then uses the scenario's 30-second
  grace plus 60-second cancellation drain. Formal windows overlap 899.016
  seconds. Both clients exit zero and report `submission_valid=true`.

The initial SPS curve contains 63 measured shapes from this exact DP2
deployment, up to 768 rows. STS temperatures are reused unchanged from the
earlier TP2 SA fit; they were not refitted on DP2. The disposable measurement
writes all points and its completion marker, but torchrun teardown stalls.
The complete measurements are validated and fitted before stopping that
container. The fitted table exactly matches `fit_sps`. Serving subsequently
validates the profile against the actual deployment.

## Results

Changes below are online relative to static. A decrease in throughput is worse;
a decrease in latency is better.

| Metric | Static SPS | Online SPS | Change |
| --- | ---: | ---: | ---: |
| Server output tokens/s/GPU, 900s window | 223.77 | 209.35 | -6.45% |
| Server output tokens/s/GPU, including drain | 206.19 | 196.22 | -4.83% |
| Completed-client output tokens/s/GPU | 141.96 | 139.08 | -2.03% |
| AIPerf total input + output tokens/s/GPU | 26,345.02 | 24,978.17 | -5.19% |
| Mean TTFT, seconds | 43.736 | 35.975 | -17.74% |
| P95 TTFT, seconds | 267.228 | 163.164 | -38.94% |
| Median TTFT, seconds | 3.752 | 5.441 | +45.02% |
| Mean request TPOT, ms/token | 225.745 | 256.215 | +13.50% |
| P95 request TPOT, ms/token | 623.695 | 761.924 | +22.16% |
| Median request TPOT, ms/token | 81.371 | 82.246 | +1.07% |
| Completed / cancelled / errors | 548 / 186 / 0 | 533 / 178 / 0 | |

Server counters sum real output token lists from one TP leader per DP rank,
then divide by two GPUs. They include partial output from unfinished/cancelled
requests and count before final scheduler/client stop filtering. Dummy outputs
are excluded; a deferred real output flushed during a dummy step is counted.
Counters are sampled every two seconds. Actual 900-second measurement brackets
are approximately 902.1 seconds, with each boundary within two seconds of the
official phase boundary; the rates use actual bracket durations. Drain brackets
are approximately 992.1 seconds. Every requested boundary is covered.

AIPerf client rates count completed requests and use the 990-second phase
including drain. Total input + output throughput includes cached prompt tokens
and is not a measure of newly computed tokens. Time-limit cancellations are
reported separately from errors. Mean/P95 latency statistics describe completed
requests, not those censored at the time limit.

## Execution evidence and interpretation

- The online model reaches version 42 with 335 accepted timing samples and no
  rejected samples. Final cost state agrees across both DP ranks. The feature
  is active; this is not a disabled-path comparison.
- All observed pure-decode steps use token graphs. Relative to each step's
  common full-width graph, static removes 26.95% of verification rows and
  online removes 22.78% in the formal window. Static records 1111 contractions
  and online 993. These are reductions relative to each arm's own batch mix,
  not a controlled same-batch comparison or whole-step FLOP reduction.
- Mean host planning including DP waiting is 0.581 ms static and 0.576 ms
  online. Online publication costs approximately 0.78-0.90 ms per publication;
  it occurs 34 times per rank within the window. This measurement does not
  indicate a large new CPU planning bottleneck.
- A mid-window snapshot shows KV utilization 98.96%/99.49%, no preemptions,
  and approximately 7.79/8.50 million queued prefill tokens. This workload has
  substantial prefill/cache pressure. The snapshot alone does not establish
  the fraction of total GPU time spent on prefill.
- For completed profiling requests, cached/input token ratios are 48.31% and
  44.79%. Cache contents, arrival times and completed request mix differ as
  the two closed-loop services progress.
- There are 461 unique request sources completed in both arms. Their mean
  TPOT is 220.08 -> 257.65 ms/token (+17.07%). Only 43 of those pairs also
  match input and cached token counts; their mean TPOT is 191.89 -> 218.43
  ms/token (+13.83%). Arrival times and competing batches remain uncontrolled.

The observed online policy gives less relative graph-row reduction and no
end-to-end throughput benefit at DP2/128 lanes. The data do not isolate whether
the cost fit, changed batch/cache trajectories, or both explain the regression.
One simultaneous A/B pair provides no repeatability or significance estimate.
The earlier TP2/16-lane gain must not be extended to this DP configuration.

This performance run does not resolve the prior output-equivalence issue:
fresh unchanged full-width references also differed. The PR remains draft;
actual EP/MoRI and multi-host model execution remain unvalidated. See the
[DP validation record](deepseek_v41_dspark_dp_validation.md).

GitHub Black, Ruff and non-GPU unit checks pass on the measured production
commit. Production content hashes match all 27 submitted files; the isolated
serving copy adds only an experiment-only host counter hook, identical in both
arms. No AITER or production implementation changes were made for this run.
