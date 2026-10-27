# DSpark DP2: interval 10, concurrent 128-lane SA A/B

**This A/B observes a throughput gain with online SPS:** server output per GPU
increases 12.37%, completed-client output increases 6.68%, and total input plus
output increases 4.29%. Mean request TPOT changes by +0.54%; mean TTFT decreases
9.39%. This is one concurrent, closed-loop observation, not a repeatability
estimate or proof that verification kernels alone are 12.37% faster.

## Configuration and validity

- Measured production commit: `eb96d53beafd100458a258760b90e6a298053218`.
- DeepSeek-V4.1-Flash, MI355X, TP1 x DP2 per arm, DP attention, no explicit EP,
  memory utilization 0.7, maximum 128 requests per rank. `TR_HYBRID` level 0,
  five speculative tokens, query buckets 1/3/6, DP token-graph cap 768.
- Same SA AgentX 393 traces, seed 42, 128 lanes per arm, 900 seconds of sending
  followed by 30 seconds grace and 60 seconds cancellation drain. Both arms use
  the joint DP planner (`bucket_global=true`); only `online_sps=false/true`
  differs. Sampling/publication/window remain 8/64/5000.
- Both arms set `ATOM_PREFILL_DECODE_INTERVAL=10`, verified in all four worker
  startup logs. Each uses a fresh model process/cache and a disjoint GPU pair
  on one shared host. All three experiment containers exit zero; the allocation
  is retained.
- Model source (27 submitted files), profile and counter probe match the
  previous interval-0 A/B. The 63-shape measured DP2 SPS profile is reused;
  STS still uses the earlier TP2 fit and has not been refitted for DP2.
- An initial interval-10 attempt fails static cache warmup at its 300-second
  drain limit; the entire pair is stopped and excluded. The valid retry uses
  a 600-second warmup drain limit in both arms. Both complete all 398 warmup
  requests, zero cancellations/errors, in 1004.22/1082.05 seconds. Warmup is
  excluded from the measurements below.
- On 2026-10-26 UTC, static profiling starts at 15:49:31.261 and online at
  15:50:53.885. Each runs the full 900 seconds; starts differ by 82.624 seconds
  and windows overlap 817.376 seconds. Both clients exit zero and report
  `submission_valid=true`, 100% TTFT/inter-token metric coverage and zero errors.

## Results

Changes are online relative to static, with interval 10 in **both** arms.

| Metric | Static SPS | Online SPS | Change |
| --- | ---: | ---: | ---: |
| Server output tokens/s/GPU, 900s | 286.32 | 321.74 | +12.37% |
| Completed-client output tokens/s/GPU | 223.91 | 238.87 | +6.68% |
| Total input + output tokens/s/GPU | 27,722.95 | 28,911.57 | +4.29% |
| Mean TTFT, seconds | 131.568 | 119.220 | -9.39% |
| P95 TTFT, seconds | 486.471 | 451.827 | -7.12% |
| Mean request TPOT, ms/token | 27.216 | 27.364 | +0.54% |
| Median request TPOT, ms/token | 27.758 | 27.809 | +0.19% |
| P95 request TPOT, ms/token | 33.967 | 33.339 | -1.85% |
| Completed / cancelled / errors | 508 / 180 / 0 | 534 / 176 / 0 | |

Server output including the 90-second drain is 268.43 -> 301.10 tokens/s/GPU
(+12.17%). These full-window results retain all completed requests and include
partial server output; no slow-request exclusion or post hoc window selection
is used for the primary comparison.

The probe counts real request token lists returned by ModelRunner postprocess,
before final scheduler/client filtering; it includes unfinished/cancelled output
and excludes dummy output. It is not the exact number delivered to clients.
Each DP rank's TP leader is counted once; summed rates are divided by two GPUs.
Two-second counters bracket the formal window by 902.09-902.14 seconds, and
rates use each actual duration. All 900/990-second boundaries are covered,
within two seconds of the requested boundary.

Client rates count completed requests over the 990-second phase including drain.
Total input plus output includes cached input tokens; it is not newly computed
token throughput. Time-limit cancellations are reported separately from errors,
and latency statistics describe completed requests, not censored requests.

## Cross-checks

- Over the genuinely simultaneous 817.376-second wall-clock interval, the same
  server probe gives 297.06 -> 328.14 tokens/s/GPU (+10.46%). The full 900-second
  comparison remains primary; this check addresses the offset phase starts.
- The independent `atom:generation_tokens_total` metric gives approximately
  267.64-267.84 -> 300.35-300.55 finalized output tokens/s/GPU including drain,
  an increase of 12.14-12.30%. Ranges include/exclude one-second boundary bins;
  they do not bound metric-refresh delay. This metric increments when requests
  finish/cancel, not on every generation step, so it does not replace the live
  900-second counter. Warmup metrics can be flushed in the first boundary bin.
  Client cancellations can precede server admission, so completed + cancelled
  client requests must not be treated as an expected server-finished count.
- In both formal windows, every approximately two-second sample advances the
  decode counters. Completed requests contain zero output gaps longer than
  10 seconds. The idle counter intervals after second 931 occur after client
  cancellation, with zero logged prefill batches; they are not decode starvation.
- The 434 unique request sources completed in both arms have mean TPOT
  27.149 -> 27.291 ms/token (+0.143 ms). The 54 pairs also matching input/cache
  token counts have 26.745 -> 26.657 ms/token. These are diagnostic subsets only;
  the primary latency statistics retain all 508/534 completed requests.

## What the execution evidence establishes

Online state reaches version 201 with 1607 lifetime accepted samples, zero
rejections and identical final state on both DP ranks. Within the formal window,
each rank publishes 182 times, averaging 1.41-1.47 ms per publication. Host
planning including DP waiting averages approximately 0.573 ms static and
0.593-0.600 ms online. All observed pure-decode forwards use token graphs.

Against each step's own common full-width graph, static removes 2.60% of rows
(376 contractions) and online removes 2.78% (435 contractions). Formal pure-decode
step counts are about 11.74K static and 11.67-11.68K online; small rank differences
come from independent counter boundaries. Online per-rank output is more balanced:
346.73/225.91 -> 329.10/314.38 tokens/s. These counters show more group output
without more decode steps, but they do not isolate same-batch kernel speedup.
The graph denominators and batch/request/cache trajectories differ.

The final fitted marginal-slope support is empty. The effective online correction
in this run is the fixed/post component (final global diagnostics approximately
7.32/3.38 ms), while the baseline graph-size curve remains measured offline.
Sampling every eight decode steps with short interval-10 decode bursts and
request-set changes can leave too few same-context graph-size contrasts for the
current slope-fit requirements. Thus this result establishes neither successful
online marginal-slope identification nor an isolated benefit from learning it.
It also does not justify attributing the throughput increase solely to the small
difference in graph-row reduction.

Completed-request cached/input ratios are 31.75% static and 34.80% online.
Closed-loop progress changes arrivals, DP load, cache and completion boundaries;
there is no same-arrival or repeated A/B control here. Keep the positive measured
end-to-end result and these attribution limits together.

## Relation to the previous interval-0 result

The previous [interval-0 A/B](deepseek_v41_dspark_dp2_sa_results.md) remains a
negative result (-6.45% server output, +13.50% mean TPOT). Its long prefill-only
stalls disappear from this interval-10 run. Enabling decode protection also
changes the scheduling tradeoff: absolute mean TTFT increases across those
separate runs (43.74 -> 131.57 seconds static, 35.98 -> 119.22 seconds online),
while TPOT drops sharply. Therefore interval 10 is not a claim of improvement
on every latency objective relative to interval 0. Within the requested
interval-10 comparison, online improves TTFT and throughput, with nearly flat TPOT.

No production source or default changes were made for this rerun. Prior GitHub
Black/Ruff/non-GPU unit checks pass on the measured production commit. This
report is documentation only and does not rerun those tests. Output equivalence
remains unresolved because previous unchanged full-width runs also differed;
performance is not an equivalence pass. The PR remains draft. Actual EP/MoRI and
multi-host model execution are still unvalidated; see the
[DP validation record](deepseek_v41_dspark_dp_validation.md).
