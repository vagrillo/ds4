# Qwen3.6-35B-A3B on ds4-server: dynamic-expert configuration comparison

Evalscope 1.10.0 against the local ds4-server OpenAI API (`http://127.0.0.1:8000/v1`),
Qwen3.6-35B-A3B-UD-Q6_K_XL.gguf, Apple M4 Pro 48 GB, thinking mode on,
`temperature 0`, `max_tokens 4096`, 0-shot. Identical questions and order for all runs;
only the server's `--q35*` flags change.

## Configurations

| Run | Server flags | Routed experts/token | Decode speed |
|---|---|---|---|
| **A** | `--q35-experts 20 --q35-expert-threshold 0.8` | ~15.5 avg (cap 20, threshold cut) | 18.5 t/s |
| **B** | *(none)* | 8 (model native) | 22.9 t/s |
| **C** | `--q35-experts 20 --q35-expert-threshold 0.8 --q35-no-expert-decay` | ~15.5 avg, extras at full influence | 18.5 t/s |

Run A uses the expansion influence decay (99% -> 50% across ranks 9..20), active since
commit ffa14fb. C isolates the decay's effect; B is the native-routing baseline.

## Results

| Benchmark | A (20 + decay) | B (default 8) | C (20, no decay) |
|---|---|---|---|
| MMLU, 8 subjects x 4 (n=32) | **96.9%** (31/32) | 93.8% (30/32) | 90.6% (29/32) |
| HumanEval pass@1, first 30 (n=30) | 96.7% (29/30) | 96.7% (29/30) | 96.7% (29/30) |
| MMLU-Pro, 8 subjects x 2 (n=16) | **81.3%** (13/16) | 75.0% (12/16) | 75.0% (12/16) |

## Where the differences come from

- **MMLU**: A solves `abstract_algebra` 4/4 where B and C get 3/4; C additionally
  drops one `machine_learning` question. Everything else is identical.
- **MMLU-Pro**: the only mover is `psychology` — A 2/2, B and C 1/2. All other
  subjects are equal (math, law, philosophy at 1/2 for every config — the hard ones).
- **HumanEval**: saturated at 29/30 for all three; the single miss is the same problem.

## Reading

1. **The decay direction looks right.** A (decay on) is never below B or C on any
   suite and takes 3 extra correct answers overall (78 samples). Without decay (C),
   expansion is slightly below native (B) on MMLU — consistent with the intuition
   that full-strength experts beyond rank 8 add noise the router never trained for,
   while decaying them keeps the benefit without the damage.
2. **None of these gaps is statistically significant.** 1-2 questions per suite at
   n=32/30/16; the honest summary is "no measurable quality cost, weak positive
   signal for the decay". A formal ranking needs hundreds of samples per suite.
3. **Cost of the configuration is real**: ~15.5 experts/token runs at 18.5 t/s vs
   22.9 t/s for native 8 (-19%). Expansion buys (at best) a marginal quality edge
   at a fifth of speed; the interesting regime for it is quality-critical single
   shots, not throughput.

## Wall-clock (per phase, single-slot server)

| Phase | A | B | C |
|---|---|---|---|
| MMLU 32 q | 57 min | 56 min | 56 min |
| HumanEval 30 q | 59 min | 53 min | 58 min |
| MMLU-Pro 16 q | 36 min | 35 min | 37 min |

## Artifacts

- Per-sample predictions: `run*/predictions/<model-id>/*.jsonl`
- Per-run reports (JSON + HTML): `run*/reports/`
- Server logs per phase: `run*/server*-phase.log`
- Chain scripts: `bench.sh` (MMLU), `bench_humaneval.sh`, `bench_mmlupro.sh`
- HumanEval dataset: `openai/openai_humaneval` (the adapter's default
  `opencompass/humaneval` no longer exists on the Hub)

Generated 2026-08-24 by evalscope 1.10.0, thinking mode, 0-shot, temperature 0.
