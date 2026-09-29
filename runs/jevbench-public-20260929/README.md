# JevBench public items: 28 small open Jev-like models

Per-item results of the **231 public JevBench items** (easy 48, original 72, hard 111) for 28 small open models
(0.15B to 0.85B parameters) that answer typed decisions with probabilities, run locally on 2026-09-29.

- Harness: the official JevBench harness ([fstandhartinger/jevbench](https://github.com/fstandhartinger/jevbench)
  @ `9ec6f15`, v1.4.2.2+3), `typesafe` adapter, one request at a time, no retries, scored by its own scorer.
- Each model ran behind its authors' own server or inference code (wrapped as `POST /v1/systemone` where the
  authors ship none), with the authors' dtype, calibration and settings, on one NVIDIA GB10 (DGX Spark).
- Where an authors' server would silently cut an over-long input, the wrapper answers HTTP 422 instead; such items
  count as wrong ("Failed" column).
- Size: parameters held by the serving process after the authors' own load code, counted at load time
  (every tensor once; unused vision towers that a loader keeps are included).

## Files

`<model>/results.jsonl` one line per item: `task_id`, `family`, `split`, `predicted`, `probs` (label -> probability),
`correct`, `valid`, `status`, `latency_s`, ... The items themselves (state, question, expected answer) are in
[`data/jevbench-public/`](../../data/jevbench-public/README.md) (MIT, copied from JevBench); `task_id` is their `id`.
`<model>/summary.json` the harness summary (accuracy, Brier, ECE bins, per-family, latency).
`<model>/manifest.json` run metadata (adapter, dataset hash, times).

## Results

Accuracy on all 231 items (%, 95% interval: bootstrap over scenarios, paraphrases of one scenario resampled together).
Rank: among the 77 systems with all 231 public items (the official JevBench v1.2 runs plus these 28).
For reference, Jev 1.13.0 (TypeSafe AI) (official run): 86.6.

| Model | Weights | Size | Acc | 95% CI | Easy | Original | Hard | Brier | ECE | Failed | p50 ms | Rank | Note |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| openjev 0.8B (NLI) | [AlexWortega/openjev](https://huggingface.co/AlexWortega/openjev) | 852.99M | **93.5** | 89.9–96.6 | 100.0 | 97.2 | 88.3 | 0.138 | 0.086 | 0 | 62 | 3 | likely trained on these public items (see below) |
| JPT-0.8B | [kirp/jpt-0.8b](https://huggingface.co/kirp/jpt-0.8b) | 852.99M | **73.6** | 67.5–79.2 | 97.9 | 83.3 | 56.8 | 0.343 | 0.071 | 0 | 43 | 25 |  |
| Intern-Decision-0.8B | [internlm/Intern-Decision-0.8B](https://huggingface.co/internlm/Intern-Decision-0.8B) | 852.99M | **70.1** | 63.7–76.2 | 100.0 | 79.2 | 51.4 | 0.376 | 0.101 | 0 | 43 | 30 |  |
| decider-0.8b | [Mapika/decider-0.8b](https://huggingface.co/Mapika/decider-0.8b) | 752.39M | **66.2** | 59.9–72.1 | 100.0 | 83.3 | 40.5 | 0.484 | 0.163 | 0 | 48 | 35 |  |
| Jev-Style-0.8B-Decision-v3 | [chaoliangUNSW/Jev-Style-0.8B-Decision-v3](https://huggingface.co/chaoliangUNSW/Jev-Style-0.8B-Decision-v3) | 752.39M | **64.5** | 58.1–70.8 | 100.0 | 81.9 | 37.8 | 0.415 | 0.083 | 0 | 144 | 37 |  |
| Kev-0.8B | [jaredpalmer/kev-0.8b](https://huggingface.co/jaredpalmer/kev-0.8b) | 752.92M | **63.6** | 56.7–69.8 | 100.0 | 81.9 | 36.0 | 0.452 | 0.043 | 0 | 71 | 39 |  |
| RSI-Jev v1.0 0.8B | [shgao/rsi-jev-v1.0-qwen3.5-0.8b](https://huggingface.co/shgao/rsi-jev-v1.0-qwen3.5-0.8b) | 759.74M | **63.6** | 56.7–70.1 | 97.9 | 69.4 | 45.0 | 0.494 | 0.123 | 0 | 36 | 39 | 35 long items silently truncated by the authors' server |
| OpenThai-SystemOne | [iapp/OpenThai-SystemOne](https://huggingface.co/iapp/OpenThai-SystemOne) | 752.67M | **62.3** | 55.5–68.9 | 100.0 | 81.9 | 33.3 | 0.558 | 0.230 | 0 | 98 | 42 |  |
| Decision 1.0 Eos | [llm-semantic-router/Decision-1.0-Eos-0.8B](https://huggingface.co/llm-semantic-router/Decision-1.0-Eos-0.8B) | 753.45M | **61.9** | 55.1–68.1 | 100.0 | 76.4 | 36.0 | 0.466 | 0.129 | 0 | 51 | 43 |  |
| Tev1-0.8B-experimental | [togethercomputer/Tev1-0.8B-experimental](https://huggingface.co/togethercomputer/Tev1-0.8B-experimental) | 852.99M | **61.9** | 55.4–68.3 | 100.0 | 76.4 | 36.0 | 0.514 | 0.151 | 0 | 41 | 43 |  |
| Von | [wfzyx/von](https://huggingface.co/wfzyx/von) | 395.31M | **59.3** | 52.6–66.0 | 100.0 | 63.9 | 38.7 | 0.487 | 0.039 | 0 | 22 | 46 | Brier/ECE in-sample (calibrated on these items) |
| lev-350m | [franckverrot/lev-350m](https://huggingface.co/franckverrot/lev-350m) | 361.01M | **58.4** | 51.7–65.0 | 97.9 | 70.8 | 33.3 | 0.503 | 0.115 | 0 | 24 | 47 |  |
| Qwen3.5-0.8B Decision Model | [mghafiri/qwen3.5-0.8B-decision-model](https://huggingface.co/mghafiri/qwen3.5-0.8B-decision-model) | 752.39M | **57.6** | 50.2–64.7 | 97.9 | 47.2 | 46.8 | 0.510 | 0.088 | 8 | 40 | 50 | 8 items over its 3072-token limit (422) |
| Tiny-Jev | [lostargon/Tiny-Jev](https://huggingface.co/lostargon/Tiny-Jev) | 595.78M | **57.6** | 50.6–64.1 | 100.0 | 77.8 | 26.1 | 0.647 | 0.308 | 0 | 35 | 50 |  |
| Bosun v3.1 0.6B | [Hanno-Labs/bosun-v3.1-0.6b](https://huggingface.co/Hanno-Labs/bosun-v3.1-0.6b) | 606.13M | **56.7** | 49.8–63.4 | 97.9 | 65.3 | 33.3 | 0.580 | 0.205 | 0 | 84 | 53 |  |
| GLiNER2.5-Decide | [fastino/GLiNER2.5-Decide](https://huggingface.co/fastino/GLiNER2.5-Decide) | 486.44M | **55.4** | 48.0–62.2 | 95.8 | 62.5 | 33.3 | 0.578 | 0.164 | 0 | 44 | 55 |  |
| SimpleJev (stock Qwen3.5-0.8B) | [Qwen/Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B) | 852.99M | **54.5** | 47.4–61.4 | 87.5 | 54.2 | 40.5 | 0.593 | 0.173 | 3 | 128 | 57 | untrained baseline (stock Qwen3.5-0.8B); 3 items over its limit (422) |
| JevEmbed-Qwen3-Embedding-0.6B | [HIT-TMG/JevEmbed-Qwen3-Embedding-0.6B](https://huggingface.co/HIT-TMG/JevEmbed-Qwen3-Embedding-0.6B) | 595.78M | **52.8** | 46.3–59.7 | 93.8 | 68.1 | 25.2 | 0.504 | 0.160 | 41 | 16 | 59 | 41 items over its 1024-token limit (422) |
| Lavoir | [moganai/lavoir](https://huggingface.co/moganai/lavoir) | 421.56M | **52.4** | 45.6–58.8 | 100.0 | 68.1 | 21.6 | 0.452 | 0.088 | 37 | 63 | 60 | 37 items over its 1024-token limit (422) |
| systemone-lite-0.5b | [dwidlee/systemone-lite-0.5b](https://huggingface.co/dwidlee/systemone-lite-0.5b) | 494.03M | **50.6** | 43.8–57.5 | 89.6 | 48.6 | 35.1 | 0.733 | 0.252 | 0 | 14 | 62 |  |
| tasksource-jev-nano-v0 | [tasksource/tasksource-jev-nano-v0](https://huggingface.co/tasksource/tasksource-jev-nano-v0) | 153.83M | **50.6** | 43.8–57.9 | 87.5 | 55.6 | 31.5 | 0.573 | 0.070 | 0 | 134 | 62 |  |
| Decision 1.0 Kai | [llm-semantic-router/Decision-1.0-Kai](https://huggingface.co/llm-semantic-router/Decision-1.0-Kai) | 571.91M | **49.4** | 42.6–56.4 | 93.8 | 54.2 | 27.0 | 0.529 | 0.136 | 44 | 23 | 64 | 44 items over its 1024-token limit (422) |
| system-one-gemma | [akash-kamat/system-one-gemma](https://github.com/akash-kamat/system-one-gemma) (GitHub; base google/gemma-3-270m) | 271.90M | **49.4** | 42.2–56.1 | 77.1 | 45.8 | 39.6 | 0.634 | 0.177 | 0 | 48 | 64 |  |
| Decision 1.0 Lex | [llm-semantic-router/decision-1.0-lex](https://huggingface.co/llm-semantic-router/decision-1.0-lex) | 571.91M | **48.9** | 41.4–55.6 | 87.5 | 58.3 | 26.1 | 0.512 | 0.111 | 44 | 25 | 67 | 44 items over its 1024-token limit (422) |
| Julia 1 | [SupersonicLabs/Julia-1](https://huggingface.co/SupersonicLabs/Julia-1) | 144.29M | **47.6** | 41.0–54.9 | 75.0 | 45.8 | 36.9 | 0.893 | 0.395 | 0 | 35 | 69 |  |
| Laya Multilingual | [convaiinnovations/laya-multilingual](https://huggingface.co/convaiinnovations/laya-multilingual) | 321.91M | **47.6** | 41.4–54.1 | 89.6 | 43.1 | 32.4 | 0.744 | 0.279 | 0 | 32 | 69 |  |
| Lumma-Fev-0.6B | [FrontiersMind/Lumma-fev-0.6b](https://huggingface.co/FrontiersMind/Lumma-fev-0.6b) | 649.28M | **45.9** | 39.1–52.9 | 83.3 | 43.1 | 31.5 | 0.795 | 0.311 | 0 | 19 | 71 |  |
| Lumma-Fev-0.1B | [FrontiersMind/Lumma-fev-0.1b](https://huggingface.co/FrontiersMind/Lumma-fev-0.1b) | 154.10M | **39.0** | 32.0–45.9 | 70.8 | 40.3 | 24.3 | 0.730 | 0.234 | 37 | 14 | 74 | 37 items over its 1408-token state limit (422) |

Notes:
- **openjev-nli-0.8b**: likely contaminated. It was fine-tuned from a checkpoint named `...-nli-v2s-jev`, and the
  repository's data mix trains on the JevBench public items (8 repeats) unless `--no-jevbench` is set, which the
  authors say they used only for later builds. Its score is far above the authors' own 4B model and Jev.
- **von**: its temperature map was fitted on these 231 items, so Brier and ECE are in-sample; accuracy is not affected.
- **simplejev-0.8b** is an untrained prompt baseline on stock Qwen3.5-0.8B (the command of the JevBench v1.4 entry).
- Latency is on a GB10 with one request at a time; it is not comparable to the official runs.
