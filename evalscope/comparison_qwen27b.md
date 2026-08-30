# MMLU-Pro: qwen3.8-27b-nvfp4 (omlx, 4-way) vs default8 vs gate39 vs full-exp20 (210/cfg, thinking, cap 30K)

| config | n | MMLU-Pro | s/sample |
|---|---|---|---|
| qwen38-27b-nvfp4 | 0 | FAILED | 1941 |
| q35-default8 | 213 | 79.8% | 2446 |
| q35-exp20-thr08-gate27-39 | 210 | 81.4% | 110 |
| q35-experts20-thr08-decay | 213 | 79.8% | 1275 |

| subset | qwen38-27b-nvfp4 | q35-default8 | q35-exp20-thr08-gate27-39 | q35-experts20-thr08-decay |
|---|---|---|---|---|
| computer science | - | 87% | 87% | 87% |
| math | - | 93% | 93% | 93% |
| chemistry | - | 80% | 80% | 80% |
| engineering | - | 72% | 73% | 56% |
| law | - | 47% | 67% | 60% |
| biology | - | 87% | 93% | 93% |
| health | - | 87% | 87% | 87% |
| physics | - | 100% | 100% | 100% |
| business | - | 80% | 80% | 80% |
| philosophy | - | 80% | 80% | 80% |
| economics | - | 93% | 93% | 93% |
| other | - | 67% | 60% | 67% |
| psychology | - | 80% | 87% | 87% |
| history | - | 67% | 60% | 60% |

## Paired qwen27b vs default8 (completati da entrambi)

comuni: 53 | q27b 47 (88.7%) | default8 47 (88.7%) | vince q27b 0, vince default8 0 (netto +0)

## Paired qwen27b vs gate39 (completati da entrambi)

comuni: 53 | q27b 47 (88.7%) | gate39 47 (88.7%) | vince q27b 0, vince gate39 0 (netto +0)
