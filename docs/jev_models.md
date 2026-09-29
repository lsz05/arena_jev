# Public Jev / Jev-like models

Collected on 2026-09-28. Sources: the Hugging Face API (related repos created after 9/14), model cards and GitHub
READMEs, the awesome-jev list, TypeSafe's official documentation.

> Note: the ecosystem is two weeks old and repos change daily. Almost all accuracies below are **self-reported by the
> authors** and were not reproduced independently. Pin a revision before comparing (for example `systemone-lite-0.5b`
> overwrites its weights in place).

The full raw list (546 repos, with category, downloads and license) is in [`../data/hf_jev_repos.csv`](../data/hf_jev_repos.csv).

## 0. Overview

| | Count |
|---|---|
| HF repos created after 9/14 whose name or tags relate to Jev/RLCD/Laya/Kev/System One/decider | 546 |
| of which quantizations or format conversions (GGUF/ONNX/MLX/CoreML/FP8…) | 193 |
| non-conversion repos | 339 |
| of which truly **general, arbitrary-option** independent model families | about 20 |

Most non-conversion repos are task fine-tunes of Laya (customer service, legal, various languages…) or personal experiments.

**Requirements for a player.** The arena asks a model to pick one of the 20–220 legal moves the caller lists, from a
text position. So a model needs:

1. open options (options are given by the caller at inference time, not fixed labels);
2. room for enough options;
3. a context that holds the position and the move list;
4. local weights or an API.

## 1. Official Jev

| Model | Notes |
|---|---|
| `jev-1.13.0` | The only official version, API only (TypeSafe, OpenRouter, Vercel AI Gateway, Cloudflare). 64k per request, of which the state plus the longest question may take 32k. Input $0.042/M tokens. Limit 1,200 req/min. |
| `jev-latest`, `jev-preview` | **Both are aliases of `jev-1.13.0`**, so `jev-latest` against `jev-preview` is self-play. |

TypeSafe has **not** released any weights. `TypeSafeAI/Step-5-Preview-*` on HF holds StepFun's 600B generative MoE and
has nothing to do with Jev; it appears to squat TypeSafe's name. Do not use it.

## 2. Tier 1: usable as players directly

Criteria: general, open Choice, many options, and mostly a bundled `POST /v1/systemone`-compatible server. A compatible
server means the arena needs **one HTTP client**: changing `base_url` changes the player.

| Model family | Sizes | Base | Architecture¹ | Options per call | Context | `/v1/systemone` | License | Against Jev (self-reported) |
|---|---|---|---|---|---|---|---|---|
| **Kev** `jaredpalmer/kev-{0.8b,4b,9b,27b}` (GitHub 7.6k★) | 0.8B / 4B / 9B / 27B | Qwen3.5-Base / Qwen3.8-27B | A5 pointer head (LoRA+head) | 255 | server 64k; state ≤384 tokens in training | ✅ the official SDK connects directly | Apache-2.0 | new-source accuracy: 27B 0.848, Jev 0.857; MMLU-Pro: 9B 0.52, Jev 0.84 |
| **decider** `Mapika/decider-{0.8b,2b,4b,35b-a3b}` | 0.8B / 2B / 4B / 35B-A3B | Qwen3.5-Base | A4 letter logits | 255 (≤10 in training) | 32k | ✅ also vLLM, GGUF | Apache-2.0 | JevBench: 35B 68.9, 2B 64.6, Jev 75.4. The 2B has 230k downloads in 30 days, the most of all |
| **openjev** `openjev/openjev` | ~27B | undisclosed | A4 letter logits | 52 per round, multi-round knockout beyond that | 16k | ✅ (vLLM shim) | **CC-BY-NC** | 10k text questions: 84.0%, Jev 85.4%; 2.3% of answers change when options are shuffled |
| **JevK5** `alibiserikbay/JevK5{,-2B,-9B}` | 2B / 4B / 9B | Qwen3.5 | A4 letter logits | 16 per round, built-in knockout (tested up to 151) | 16k | ✅ `jevk5-serve`, also GGUF | Apache-2.0² | JevBench v1.4: 62.04, Jev 63.29 |
| **Jev-Style** `chaoliangUNSW/Jev-Style-{0.8B,2B}-Decision-v3` | 0.8B / 2B | Qwen3.5 | A4 one verdict slot per option | unlimited (chunked, one softmax) | 25.6k | ✅ `jev-style serve`, also GGUF, MLX | Apache-2.0² | tweet_topic: 75.5%, Jev 79.3%; 0.5% of answers change when options are shuffled |
| **OpenThai-SystemOne** `iapp/OpenThai-SystemOne` | 0.8B | Qwen3.5-0.8B | A5 256-slot head | 255 | 64k | ✅ | Apache-2.0 | 13 subsets: 74.3, Jev 76.0. Order-sensitive with many options; needs its "order-invariant mode". Already plays ViZDoom from text states |
| **openjev (NLI)** `AlexWortega/openjev` | 0.8B / 2B / 4B / 35B-A3B | Qwen3.5 | NLI cross-encoder, one forward pass per option | unlimited (cost grows linearly) | — | ❌ has a JevBench adapter | MIT | JevBench: 4B 0.814, Jev 0.866; **fully independent of option order** (0/231). Already plays Doom, Flappy, Minecraft. Its training set includes the test sets of some benchmarks |

¹ Architecture codes follow section 4 of Jev-Survey: A2 label-conditioned encoder, A4 LLM scoring of candidate/label
tokens, A5 LLM plus a learned head or pointer.
² The training data includes GPT/Claude-generated content; check the terms before commercial use.

## 3. Tier 2: usable, with clear limits

| Model | Size | Main limits | Why try it |
|---|---|---|---|
| `autotrust/JEV-9B` (also 27B) | 9B | **at most 16 options per call**; cuts the state to 1024 tokens by default | **distilled directly from Jev 1.13 outputs**: Choice top-1 agrees with Jev 90.2%. A "Jev clone" to test whether distillation copies a playing style. Training data was obtained through OpenRouter, a ToS risk |
| `apus-ailab/APUS-OpenJev-v1-{4B,9B,35B-A3B}` | 4–35B | at most 16 candidates; uncalibrated probabilities; the evaluation set was also used for model selection | has a `/v1/systemone` server, with low/high effort levels |
| `wayfind/metask-jev-4b-policy-mix` | 4B | at most 26 options (A–Z) | has a `/v1/systemone` server, 16 languages |
| `TokenRhythm/NeoHorse-Jev-4B` | 4B | option limit not stated; 22.0% on the OpenJev chess multiple-choice set (Open-Jev-9B: 52.4) | has a `/v1/systemone` server; demos of Tetris, mahjong and other games |
| `alanhuangya/wev-{1.7b,4b,8b}` | 1.7–8B | generalizes worse than Kev-4B; training data includes Jev-distilled text | Kev-style pointer head, tested with 8–40 candidates in browser tasks; `wev serve` |
| `ZefanCai/Open-Jev-{2B,9B,27B-v1.1}` | 2–27B | one pass per candidate (220 moves means 220 passes); evaluated on synthetic data only | has a `/v1/systemone` server, no option limit |
| `shgao/rsi-jev-v2.1-qwen3.5-2b` | 2B | the training sets of 11 of its 12 benchmarks were used in training; option limit not stated | has a `/v1/systemone` server, calibrated |
| `lostargon/Tiny-Jev{,-1.7B}` | 0.6B / 1.7B | no server; 53% on unseen rule systems | tested with 151 options; loads directly with transformers |
| `vagmi/jev-lite` | Gemma-4-E4B | the alphabet limits the number of options; effectively non-commercial | has a `/v1/systemone` server |
| `kushalpatil/jevify-gemma4-26b-a4b` | 26B-A4B | thin evaluation; option limit not stated | trained with states up to 24k tokens, suits full game histories |
| `akhilaaa3/Jev-Omni` | 12B | about 50 GB in FP32; one question per call; quality above 20 options not verified | a 256-option head |
| `tianxinwei/JevAny-27B-SFT` | 27B | little interface documentation; the author does not recommend the RLCR version | MMLU-Pro 73.5%, possibly strong reasoning |
| `dwidlee/systemone-lite-0.5b` | 0.5B | weak (JevBench 50.7%); weights are overwritten in place | **trained on board games**: chess, Connect Four, Sokoban, 2048. Chess is split into "pick a piece → pick a square", ≤8 options per step |
| `flock-io/this-that-model-1.0` | 2B | narrow training domain (grid worlds); 0.48–0.65 on whole-board views | adapted from decider-2b, 255 options, trained for spatial reasoning |

## 4. Zero-training harnesses: any open LLM as a Jev-like player

These projects train nothing and read a frozen LLM's logits on the option labels. In the arena they are good
**controls**: how does the same base play without decision training?

| Project | Notes | Results (self-reported or leaderboard) |
|---|---|---|
| **SemIf** (formerly OpenJev, `TheoLeeCJ/openjev`, 4.5k★) | frozen Qwen3.5-4B / MiniCPM5-2B / Qwen3.8-27B; CUDA, MLX, GGUF, WebGPU; no `/v1/systemone` server | JevBench **#2, 74.7** (Jev 75.4). An untrained 4B ranks second |
| **jevfire** (`kikoncuo/jevfire`) | a sidecar next to vLLM that reads the LM head of an unmodified 27B; interface `/v1/decisions`; >128 options needs a vLLM patch | Decision Index **#2, 55.7** (Jev 59.5) |
| `Meanblock/JEV-CPU` | SemIf with Qwen3-0.6B on the CPU | very weak (0.44) |
| `harshatheg/Qwen-2.5-1B-RLCD`, `notnotsamuel/LFM2.5-350M-RLCD`, `monotykamary/LFM2.5-2.6B-RLCD` | RLCD in the name but **never trained**: constrained-decoding inference code plus the original weights | baselines only |

**A natural controlled experiment**: Kev-4B, decider-4b, JevK5, APUS-4B and metask-4b are all based on
**Qwen3.5-4B**, and SemIf can run an untrained Qwen3.5-4B. Comparing them on the same base isolates the effect of the
training recipe.

## 5. Not suitable as players

| Category | Models | Reason |
|---|---|---|
| context or option budget too small | the Laya family (`convaiinnovations/laya`, `laya-multilingual`, `laya-typed-decisions` and dozens of fine-tunes) | 512-token context for the English version; all options share a 192–256-token budget, the authors advise ≤20 options; near random zero-shot (typed-decisions 0.362) |
| | `com-kotobalabs/open-jev-deberta-v3-large`, `mobarmg/jev-schema-scorer-deberta-v3-large` | 512-token window |
| | `heman10x/rlcd-modernbert-151m` | hard limit of 24 options |
| | `anthonym21/qwen3-0.6b-rlcd-decision` (eve-rlcd) | 26 options, trained at 512 tokens, a research toy |
| | `pngwn/system-one-qwen3.5-4b-scorer` | 384 tokens; 0.234 with 52 options |
| Noul only | jevos (`feder-cr/jev`, 1.1k★) | answers yes/no only |
| bi-encoder / retrieval | Verdict, `HIT-TMG/JevEmbed-*`, `tasksource/tasksource-jev-nano-v0` | state and options are encoded separately, so relations between moves cannot be reasoned about |
| visual input | `OmniJev/PlayJev-0.8B`, `thaitea/laya-vision`, `guanxuyu/visual-jev-4b`, `Mapika/decider-2b-vision` | pixel input, action sets of only 2–7 |
| task-specific | `adambloebaum/laya-blackjack` (blackjack), `aimeigaoshou/agent-jev` (judges whether a coding task is done), `xuhaodev/Qwen3-1.7B-Jev` (Chinese football data), `samatv256/mini-Jev` (tool routing), `SUPER321/jevflash-doom-basic-0.6b` (4 fixed Doom actions), `argos1111/modernbert-ja-310m-jev` (Japanese), `moganai/lavoir` (customer-service clarification, NC), `flock-io/this-that-model-1.2` (policy rules) | special-purpose |
| suspicious or duplicate | `TypeSafeAI/Step-5-Preview-*` | not a TypeSafe model |
| | `vdaular/decider-35b-a3b` | a re-upload of Mapika's; use the original |
| | unusual like/download ratios: `convaiinnovations/laya` 4219/0, `AlexWortega/openjev` 615/0, `harshatheg/Qwen-2.5-1B-RLCD` 574/0 | popularity is not a quality signal |

## 6. Existing work on Jev playing games (to borrow from or reuse)

| Project | Content | Lessons for the arena |
|---|---|---|
| [jev-playground](https://github.com/hegargarcia/jev-playground) | tic-tac-toe and Connect Four. Jev against GPT-5.6 Luna, Claude Haiku 4.5, Gemini 3.5 Flash Lite, GPT-6 Astra, on the same positions with the same legal options | interactive only; **automatic tournaments are on its to-do list**, which this arena fills. Options use descriptive names such as `center`, `top_left`, with the board after the move attached |
| [jev-gomoku](https://github.com/mizchi/jev-gomoku) | Jev against Jev at gomoku (MoonBit) | one 15×15 game takes about 58k input tokens (≈$0.0024), about 0.5 s per move. Candidates are only cells near existing stones, with `WINS`/`BLOCK` hints attached. **Such hints change what is being measured** |
| Kev playground chess demo | the board as the state, legal moves as a Choice, plus a Score question evaluating the position | no playing-strength numbers |
| decider game harness | 10 text games plus Mario; zero-shot on 234 positions, greedy win rate 35B 37.2%, 2B 26.5% | Tetris first narrows to 8 candidates with a heuristic: **candidate pre-filtering** is common practice |
| systemone-lite board-game gym | chess split into two steps (pick a piece → pick a square, ≤8 options each) | hierarchical splitting is one way to handle >255 or >16 moves |
| jevlike | a small chess model against Stockfish level 0: 0 wins, 2 draws, 48 losses | a reference for the lower bound of playing strength |
| JEV-9B model card | a poker hand where Jev shoves the nuts with probability 0.70 while the solver checks; the distilled model inherits the mistake | poker can be scored move by move with a solver |
| laya-blackjack | distilled from an exact solver, reports an EV regret of 0.001 per decision | an example of a per-decision regret metric |

## 7. Direct consequences for the arena design

1. **Option limits vary widely**: 16 (JEV-9B, APUS), 26 (metask, eve), 52 per round (openjev), 255 (Kev, decider,
   OpenThai), unlimited (Jev-Style, Open-Jev, NLI models).
   - Either every player gets the same **knockout or hierarchical selection** wrapper (the wrapper itself affects the
     results, so it must be identical for everyone);
   - or the first phase uses only games with ≤16 moves: Connect Four 7, UNO, Leduc poker 3, tic-tac-toe 9.
2. **Training and game distributions differ**: decider was trained with ≤10 options and Kev with states ≤384 tokens,
   so long move lists and long positions are out of distribution for them.
3. **Most players are `/v1/systemone`-compatible**, so the arena's model interface can be a uniform
   `{name, base_url, model}`.
4. **Sensitivity to option order varies widely**: AlexWortega/openjev 0%, Jev-Style 0.5%, openjev 2.3%, Kev about 8%,
   JEV-9B 11.5%, OpenThai up to 72% with many options. So every call shuffles the options and records the order.

## 8. Suggested first lineup

| Role | Players | Question answered |
|---|---|---|
| reference | `jev-1.13.0` (API) | the baseline |
| scaling within a family | Kev 0.8B / 4B / 9B / 27B; decider 0.8B / 2B / 4B / 35B-A3B | effect of parameter count on playing strength |
| same base, different recipes (Qwen3.5-4B) | Kev-4B, decider-4b, JevK5, SemIf (untrained Qwen3.5-4B) | whether decision training helps, and which recipe is better |
| different architectures | openjev (27B, A4), AlexWortega/openjev 4B (NLI, order-invariant), Jev-Style-0.8B (verdict slots) | effect of architecture |
| Jev clone | JEV-9B (distilled from Jev) | whether distillation copies Jev's play |
| anchors | random, rule bots, Stockfish levels | to calibrate Elo |

## 9. Small models (≤1B, added 2026-09-29)

Sources:
- Decision Index 0.2.1 (71 entries): scores and ranks recomputed with the official formula; the recomputed Jev score
  matches the official one;
- the JevBench leaderboard v1.4.2.2 (91 ranked entries);
- the 231 JevBench public items (this table was written before the local runs; the local results for all 28 models
  are in [`runs/jevbench-public-20260929/`](../runs/jevbench-public-20260929/README.md)).

"—" means no data.

### 9.1 Small decoder-based decision models (0.5–0.9B, mostly Qwen-based)

| Model | Params | Base / method | Decision Index (/71) | JevBench board (/91) | 231-item accuracy | Arena access |
|---|---|---|---|---|---|---|
| JPT-0.8B (kirp) | 0.87B | Qwen3.5-0.8B-Base, LoRA | **19.2 (#47)** | — | — | unconfirmed |
| Decision 1.0 Eos (llm-semantic-router) | 0.87B | Qwen3.5-0.8B-Base, head | 18.4 (#48) | — | — | unconfirmed |
| Kev-0.8B | 0.87B | Qwen3.5-0.8B-Base, LoRA + pointer head | 14.7 (#49) | — | 63.6% | ✅ `/v1/systemone` |
| Bosun v3.1 0.6B (Hanno-Labs) | 0.60B | Qwen3-0.6B-Base, LoRA + head | 14.3 (#50) | — | — | unconfirmed |
| Tev1-0.8B-experimental (togethercomputer) | 0.87B | full fine-tune | 13.0 (#51), **ChessBench 19.5%** (Jev 17.2%) | — | — | unconfirmed |
| Intern-Decision-0.8B (internlm) | 0.87B | full fine-tune | 12.0 (#52) | — | — | unconfirmed |
| MoJev (MoLeMo-Lab) | 0.87B | head | 11.7 (#53) | — | — | unconfirmed |
| decider-0.8b (Mapika) | 0.8B | full fine-tune, letter logits | — | — | **66.2%** | ✅ |
| Jev-Style-0.8B-Decision-v3 | 0.75B | full fine-tune, a yes/no verdict per option | — | — | 64.5% | ✅ |
| OpenThai-SystemOne | 0.75B | 256-slot head, Thai + English | — | — | 62.3% | ✅ |
| kev 0.6B (research preview) | 0.6B | Qwen3-0.6B-Base | — | 24.8 (#55) | **66.7%** | ✅ (kev server) |
| kev 0.5B | 0.5B | Qwen2.5-0.5B | — | 18.9 (#61) | 49.4% | ✅ (kev server) |
| Qwen3.5-0.8B Decision Model (mghafiri) | 0.8B | Qwen3.5-0.8B | — | 14.5 (#71) | — | unconfirmed |
| RSI-Jev v1.0 0.8B (shgao) | 0.8B | Qwen3.5-0.8B-Base + scoring head | — | — | — | ✅ |
| openjev 0.8B (AlexWortega, NLI) | 0.8B | one NLI pass per option | — | — | — | needs an adapter |
| Tiny-Jev (lostargon) | 0.6B | Qwen3-0.6B + option head | — | — | — | needs an adapter |
| systemone-lite (dwidlee) | 0.5B | Qwen2.5-0.5B, **trained on board games** | — | — | self-reported 50.7% | ✅ |
| untrained baselines: SimpleJev / Raw Qwen3 0.6B | 0.6–0.8B | logits of the original model | — | 7.5 (#81) / 7.1 (#83) | — | — |

### 9.2 Encoder-based and tiny models (<0.6B, mostly BERT-family)

| Model | Params | Type | Decision Index | JevBench board | 231-item accuracy |
|---|---|---|---|---|---|
| jeff (GLiFormer) | 0.4–0.58B | reasoning tricks | 8.0 (#56) | 30.6 (#42) | 62.8% |
| Laya | 0.42B | ModernBERT-large | 6.1 (#61) | 30.3 (#43) | 58.4% |
| lev-350m | 0.35B | LFM2.5-350M | — | 28.5 (#46) | — |
| Von | 0.40B | Option-Marker | — | 27.5 (#48) | — |
| OpenDecision | 0.40B | ModernBERT-large, zero-shot | — | 21.6 (#58) | 53.2% |
| openJev Verdict / Verdict 1.4 | 0.15B | GLiClass ModernBERT-base | 2.8 (#69) | 18.1 (#64) / 19.0 (#60) | 55.4% / 57.6% |
| GLiNER2.5-Decide | 0.49B | GLiNER2 large | 11.3 (#54) | — | — |
| Lavoir | 0.40B | ModernBERT-large | 8.7 (#55) | — | — |
| Decision 1.0 Kai / Lex | 0.31B | mmBERT-base | 6.5 (#60) / 4.6 (#64) | — | — |
| system-one-gemma | 0.27B | Gemma-3-270m, LoRA + head | 5.1 (#63) | — | — |
| Julia 1 | 0.14B | mmBERT-small | 5.6 (#62) | — | — |
| open-jev-deberta-v3-large | 0.44B | DeBERTa | — | 12.6 (#73) | 52.4% |
| GLiNER 2.5 small / base / multi | 74M / 194M / 287M | GLiNER | #66 / #59 / #65 | #82 / #75 / #79 | 45.9% / 58.0% / 48.9% |
| verdict-small | 0.12B | e5-small bi-encoder | — | 5.7 (#84) | — |
| Lumma-Fev 0.1B / 0.6B | 0.15B / 0.65B | full fine-tune / LoRA | 1.8 (#70) / 3.0 (#68) | — | — |
| tasksource-jev-nano, laya-multilingual, JevEmbed-0.6B | 0.15B / 0.32B / 0.6B | bi-encoder / encoder | — | — | — (self-reported only) |

### 9.3 Small but not suitable as players

PlayJev-0.8B and laya-vision (image input), jevflash-doom-0.6B (4 fixed actions), agent-jev-0.6B (judges whether a
coding task is done), mini-Jev-0.6B (tool routing), modernbert-ja-310m (Japanese), jevos (about 1B, yes/no only),
eve-rlcd-0.6B (at most 26 options, 512-token context), LFM2.5-350M-RLCD (never trained, DI #71).

### 9.4 Language support (added 2026-09-29)

- "Card": the `language` field of the Hugging Face model card or the README, i.e. the languages the authors trained
  or evaluated.
- "Base": what the base model supports; fine-tuning may not keep it.
- 🔒 marks a gated repository: it needs an HF token and the authors' approval.

| # | Model | Params | Card | Base |
|---|---|---|---|---|
| 1 | Kev-0.8B | 0.8B | English | Qwen3.5 (201 languages and dialects) |
| 2 | decider-0.8b | 0.8B | English | Qwen3.5 |
| 3 | Jev-Style-0.8B | 0.75B | **multilingual**: 19 trained, 51 evaluated, incl. Chinese | Qwen3.5 |
| 4 | OpenThai-SystemOne | 0.75B | **Thai + English** | Qwen3.5 |
| 5 | JPT-0.8B | 0.85B | not stated | Qwen3.5 |
| 6 | Decision 1.0 Eos | 0.87B | **multilingual** (evaluated incl. the Laya multilingual suite) | Qwen3.5 |
| 7 | Tev1-0.8B | 0.87B | not stated (the card says multilingual performance is not fully evaluated) | Qwen3.5 |
| 8 | Intern-Decision-0.8B | 0.85B | not stated | Qwen3.5 |
| 9 | Bosun v3.1 0.6B | 0.6B | English | Qwen3 (100+ languages) |
| 10 | MoJev 🔒 | 0.85B | unknown (gated, the card cannot be read) | Qwen3.5 |
| 11 | Qwen3.5-0.8B Decision Model | 0.75B | **English only** (stated on the card) | Qwen3.5 |
| 12 | RSI-Jev 0.8B | 0.8B | not stated (all evaluations in English) | Qwen3.5 |
| 13 | systemone-lite-0.5B | 0.5B | not stated | Qwen2.5-0.5B-Instruct (marked English) |
| 14 | Tiny-Jev | 0.6B | English (a little Russian in training) | Qwen3 |
| 15 | openjev 0.8B (NLI) | 0.8B | English | Qwen3.5 |
| 16 | SimpleJev 0.8B | 0.87B | untrained, the original model | Qwen3.5 (201) |
| 17 | Decision 1.0 Kai | about 0.6B | **multilingual** | Vela encoder (en, zh, de, fr, ja, ar) |
| 18 | Decision 1.0 Lex | 0.57B | English specialist (use Kai for other languages) | Kai |
| 19 | Julia 1 | 0.14B | **multilingual** | mmBERT-small (1800+) |
| 20 | Lumma-Fev-0.1B | 0.15B | English | Nandi-Mini (English + 10 Indian languages) |
| 21 | Lumma-Fev-0.6B | 0.65B | **English + 10 Indian languages** | Lumma-0.6B-Base |
| 22 | system-one-gemma 🔒 (base) | 0.27B | not stated | Gemma 3 270M (multilingual) |
| 23 | GLiNER2.5-Decide | 0.34B | English (use GLiNER2.5-multi-Decide for other languages) | GLiNER2 large (en, fr, es) |
| 24 | Lavoir | 0.42B | English | ModernBERT-large (English) |
| 25 | lev-350m | 0.35B | not stated | LFM2.5-350M (en, ar, zh, fr, de, ja, ko, pt, es: 9) |
| 26 | Von | 0.4B | not stated | unknown |
| 27 | tasksource-jev-nano | 0.15B | English | LateOn (English) |
| 28 | laya-multilingual | 0.32B | **multilingual** (the card lists about 50, incl. Chinese) | mmBERT-base |
| 29 | JevEmbed-0.6B | 0.6B | not stated | Qwen3-Embedding (multilingual) |
| 30 | kev 0.6B | 0.6B | English | Qwen3 |
| 31 | kev 0.5B | 0.5B | English | Qwen2.5-0.5B |
| 32 | jeff | 0.4–0.58B | English (results cover English tasks only) | GLiFormer |
| 33 | Laya (English) | 0.42B | English (use laya-multilingual for other languages) | ModernBERT-large |
| 34 | GLiNER2.5 base | 0.19B | English | DeBERTa-v3-base |
| 35 | GLiNER2 large | 0.49B | English, French, Spanish | — |
| 36 | GLiNER2.5 multi | 0.29B | **multilingual** | mDeBERTa-v3-base |
| 37 | GLiNER2.5 small | 74M | English | — |
| 38 | openJev Verdict | 0.15B | not stated | GLiClass ModernBERT-base (English) |
| 39 | Verdict 1.4 | — | not stated | GLiClass (English) |
| 40 | OpenDecision | 0.4B | not stated | ModernBERT-large (English) |
| 41 | open-jev-deberta-v3-large | 0.43B | English only | DeBERTa-v3-large (English) |
| 42 | smalljev | — | not stated | unknown |
| 43 | verdict-small | 0.12B | **100+ languages** | multilingual-e5-small |
| 44 | Certo | — | cannot yet handle real natural-language text (per its card) | — |
