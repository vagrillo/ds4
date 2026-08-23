# Dynamic MoE inference: scaling routed experts up and down

DwarfStar can change, at inference time, how many routed experts each token
activates in the MoE layers. Two implementations exist. They share the same
threshold selection rule but answer different constraints, and their
mechanisms differ in one fundamental way: where the per-token expert count
lives.

| | qwen35moe | DeepSeek V4 Flash |
|---|---|---|
| Switches | `--q35-experts N`, `--q35-expert-threshold P` | `--dsv4-experts N`, `--dsv4-expert-threshold P` |
| Direction | down **and up** (expansion) | down only, 1-6 |
| Expansion influence | linear 99%..5% decay (`--q35-no-expert-decay` to disable) | n/a |
| Default N | model value (8) | model value (6) |
| N range | 1..`expert_count` | 1-6 |
| Requires `--ssd-streaming` | no | yes |
| Scope | decode path (fused kernels) | decode only, prefill always uses 6 |
| Count location | entirely on the GPU | computed on the GPU, read back per layer on the host |
| Kept weights | renormalized over the kept set | unchanged router weights, dropped mass discarded |
| Stats env | `DS4_Q35_EXPERT_STATS_EVERY` | `DS4_DSV4_EXPERT_STATS_EVERY` |

## Why change the expert count

Routed layers activate a fixed top-K per token (top-8 for Qwen3.6-35B-A3B,
top-6 for DeepSeek V4 Flash), but router scores decay quickly after the top
few ranks. Selecting fewer experts removes the matvecs — and, in SSD
streaming, the SSD fetches and expert-cache residency — of exactly the
experts that contribute least. A score threshold makes the count adaptive:
easy tokens keep few experts, hard tokens keep all of them.

In the other direction, qwen35moe also supports **expert expansion**:
selecting more experts than the model's default routing (up to
`expert_count`) computes a denser approximation of the full MoE output at a
proportionally higher compute cost. This can recover a little quality on
tasks where the router's top-8 is a tight bottleneck. By default the added
experts enter with a linearly decayed influence (99% down to 5%), since
the model never trained them to be full-strength at those ranks.

## The shared threshold rule

With `N` = the `-experts` maximum and `P` = the `-expert-threshold`
fraction:

* Without `P`: fixed top-N selection.
* With `P`: experts are kept only while their router score is at least
  `P x score(rank N/2)`. The top `N/4` ranks are always kept (dsv4 clamps
  the floor to at least 1), so every token retains a core set, and the
  count is capped at N. The result adapts per token and per layer.

Example: `--dsv4-experts 4 --dsv4-expert-threshold 0.95` uses rank 2 as the
reference, always keeps at least 1 expert, and drops any expert whose score
falls below 95% of that reference, up to a maximum of 4.

## Implementation 1: qwen35moe, all on device

Qwen3.6-35B-A3B runs fully resident on Metal through fused kernels
(`metal/qwen35.metal`), so the expert count never leaves the GPU:

* The router kernel (`kernel_qwen35_ffn_enter`) already performs the
  weighted RMS norm, the router projection, and the shared-expert gate. Its
  top-N loop applies the threshold rule directly: it selects the first
  `N/2` ranks to establish the reference probability, keeps extending while
  `p(k) >= P x p(rank N/2)` (down to the `N/4` floor, up to the N cap), and
  writes the kept count to a device buffer `n_sel[0]`.
* The selected experts' weights are **renormalized over the kept set**, so
  the routed output keeps its scale regardless of how many experts
  survived the cut. The sigmoid-gated shared expert is unaffected.
* When N exceeds the model's native top-N, the extra ranks do not get
  full influence: the kernel multiplies their raw probability by a linear
  factor, 0.99 for the first extra rank down to 0.05 for the last one
  (midpoint when there is exactly one extra), before the
  renormalization. The decay applies in prefill and decode alike, and
  `--q35-no-expert-decay` turns it off, giving every selected expert
  full influence.
* The MoE kernels (`moe_gate_up`, `moe_down`, `moe_combine`) read
  `n_sel[0]` on device and early-return for slots at or above the count:
  dropped experts are never touched. Because nothing is read back, the
  command batch is never interrupted; expansion to N larger than the
  default works the same way, just with more active slots.
* Per-layer statistics accumulate on the GPU into an atomic counter
  (`n_sel_acc`); the host reads it once per reporting interval (default
  every 128 tokens) to print the average experts/token per layer.

## Implementation 2: DeepSeek V4 Flash, host-side cut for SSD streaming

The Flash streaming path must know, per layer, which experts to `pread`
from the SSD and which expert-cache slots to bind, so the count computed on
the GPU is read back by the host:

* The router finalize kernels (`metal/dsv4_misc.metal`) run the same
  threshold rule over the top-6 scores (reference rank `N/2`, floor
  `max(N/4, 1)`, cap N) and write the kept count to a small `n_sel` graph
  tensor.
* `routed_moe_one` reads it back per layer, riding the selected-ids sync
  that the streaming decode path already performs. Cut experts are never
  fetched from SSD and never occupy expert-cache residency; only the kept
  experts are loaded and bound.
* The decode kernels always address exactly 6 slots. Tail slots above the
  kept count mirror slot 0's expert (valid quantized data) with **weight
  0**, so buffers are always valid and no NaN can leak. Unlike qwen35moe,
  the kept experts keep their original router weights: the dropped weight
  mass is discarded rather than renormalized.
* The first 3 hash-routed layers have no router scores. There the maximum
  simply truncates the fixed per-token expert table to its first N entries
  (the threshold does not apply), and the tails mirror slot 0 with weight
  0 as above.
* The streaming split path (resident experts submitted while missing
  experts load) masks its down pass to the slots it actually produced;
  cut tails are skipped there too, which is numerically identical since
  their weight is 0.
* While a cut is active, the M5 fused router-select kernel and the
  `DS4_METAL_DISABLE_ROUTER_SELECT_FUSION` fallback are excluded, because
  they do not maintain `n_sel`. Prefill always uses the full 6 experts.

The difference in count placement is the core design decision: qwen35moe is
resident, so the count stays on device and decode is never interrupted;
the Flash streaming path needs the count on the host to drive SSD I/O, so
it pays one extra 4-byte read per layer inside a sync it already needed.

## Measuring and tuning

* Start from `--q35-experts 4` / `--dsv4-experts 4` with a threshold around
  `0.95`, then watch the per-layer average printed by the stats report.
  Layers that consistently cut deep are the safest to keep cutting.
* Decode speed scales roughly with the expert count. Reference numbers for
  qwen35moe on an M4 Pro 48 GB (Q6_K_XL, `--temp 0`):

| --q35-experts | tok/s decode |
|--------------|--------------|
| 1            | 33.2         |
| 4            | 29.4         |
| 8 (default)  | 25.9         |
| 16           | 20.6         |

* For DeepSeek V4 Flash the saving is twofold: fewer expert matvecs and
  fewer SSD fetches. The effect is largest when the expert cache is cold
  or thrashing; with a hot cache the fetch saving shrinks and the speedup
  approaches the pure compute ratio.
* Accuracy is workload-dependent. Score your configuration with
  `ds4-eval` before trusting a deep cut for critical work, and remember
  that qwen35moe renormalizes while dsv4 discards the dropped weight mass,
  so the same threshold value can behave differently across the two
  models.



## Pelican riding a Bicycle with Qwen3.6-35B-A3B-UD-6b_XL

   Short test using a **6bit quantized** Qwen 35B model.



   
```Generate an SVG of a California brown pelican riding a bicycle. The bicycle must have spokes and a correctly shaped bicycle frame. The pelican must have its characteristic large pouch, and there should be a clear indication of feathers. The pelican must be clearly pedaling the bicycle. The image should show the full breeding plumage of the California brown pelican.```

**ORIGINAL**
 --nothink  (default settings)
 
 <img width="829" height="578" alt="image" src="https://github.com/user-attachments/assets/e46b818e-2e73-4b88-a50a-2b3c77e9c0b4" />



**Dynamic Expert enhance - 14 EXPERTS AVERAGE**
 --nothink --q35-experts 16  --q35-expert-threshold 95
 
<img width="604" height="408" alt="image" src="https://github.com/user-attachments/assets/46089da5-2a5f-4dce-8ca5-d446a725c159" />










 

