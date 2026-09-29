# Server environments

Everything under `servers/venvs/` (97 GB) and `servers/src/` (11 GB) is rebuilt, not committed. This folder records
what they contain, as used for the results on 2026-09-29 (NVIDIA GB10, aarch64, CUDA 13, Python 3.12.3,
torch 2.13.0+cu130).

- `<venv>.txt`: `pip freeze --all` of `servers/venvs/<venv>`. Editable installs are pinned to their git commit.
  Absolute paths are rewritten to `servers/...` and `~/...`.
- The torch, triton and CUDA wheels are aarch64 builds; on x86_64 install the same versions from the PyTorch index
  for your CUDA (e.g. `--index-url https://download.pytorch.org/whl/cu130`).
- Model weights come from the Hugging Face cache (`$HF_HUB_CACHE`, default `~/.cache/huggingface/hub`) at the
  revisions below or in each model file's notes. `google/gemma-3-270m` (system-one-gemma) is gated: the account
  needs access. MoJev is not included (gated, not approved).
- `servers/setup.sh` builds the first four venvs (kev, decider, jevstyle, openthai). The others were set up by hand;
  `bench/reports/*.json` has each model's exact code path, revision and notes.

## Model servers

| Model | Port | Venv | cwd | Code and weights |
|---|---|---|---|---|
| bosun-0.6b | 8125 | `b-bosun` |  | authors' remote code modeling_bosun.BosunForDecision from HF Hanno-Labs/bosun-v3.1-0.6b@1d8b6f9611f9b64b514ce8b57cd86398fbc31a3b on Qwen/Qwen3-0.6B@c1899de; adapter servers/adapters/bosun.py |
| decider-0.8b | 8102 | `decider` | `src/decider` |  |
| decision-eos-0.8b | 8122 | `b-llm2jev` |  | authors' native package decision/ (DecisionModel.from_pretrained(...).decide) from HF llm-semantic-router/Decision-1.0-Eos-0.8B@3c2d632609ceb66f3a13bbc5f77f3ab8cdeebcdd; adapter servers/adapters/decision_eos.py |
| decision-kai | 8131 | `c-decision` |  | HF llm-semantic-router/Decision-1.0-Kai@7185f514f54b (last revision bundling code; weights identical to main 9d6872cd); decision_runtime.load_native + decision_inference.SystemOne; adapter servers/adapters/decision_kai.py + decision_native.py; venv c-decision |
| decision-lex | 8132 | `c-decision` |  | HF llm-semantic-router/decision-1.0-lex@ee8e74d912fc (last revision bundling code; weights identical to main 6c5e3d48); adapter servers/adapters/decision_lex.py + decision_native.py; venv c-decision |
| gliner2.5-decide | 8134 | `c-gliner` |  | gliner2==2.0.0[local] (venv c-gliner), AutoExtractor on HF fastino/GLiNER2.5-Decide@5a7adf72a23b, FP32; adapter servers/adapters/gliner2_5_decide.py |
| intern-decision-0.8b | 8124 | `b-intern` |  | model card's inference.py (DecisionEngine, backend hf) in HF internlm/Intern-Decision-0.8B@85a0cc5a99d67ea8d56dfe98115689212867171d; adapter servers/adapters/intern_decision.py |
| jev-style-0.8b | 8103 | `jevstyle` |  |  |
| jevembed-0.6b | 8126 | `b-jevembed` |  | authors' JevEmbed server (github HITsz-TMG/JevEmbed@25cb86f495f3caaa05a610cb50d27cf924e2987b, python -m jevembed --serve) with HF HIT-TMG/JevEmbed-Qwen3-Embedding-0.6B@e4a71aacd9b47ae6106fe57b7bf1c39992dc7ab1 jevembed.yaml |
| jpt-0.8b | 8121 | `b-llm2jev` |  | llm2jev 0.6.1 (github tic-top/llm2jev@2b252d504972764211ef172c1155ac0fedc9c3de) CLI `llm2jev --backend hf --temperature 1.140`, authors' /v1/systemone server unchanged; weights kirp/jpt-0.8b@1431c0509bbc10772cd59964cc7af5835c8720d6 |
| julia-1 | 8133 | `c-julia` |  | HF SupersonicLabs/Julia-1@a85b127321d5 julia package (pip -e, venv c-julia): load_model(strict_encoding=True, max_length=8192, head_length=512) + engine.predict; adapter servers/adapters/julia_1.py |
| kev-0.8b | 8101 | `kev` | `src/kev` |  |
| lavoir | 8135 | `c-laya` |  | lavoir package (github moganai/lavoir@af79b0a7a7f9, pip -e) + laya 0.3.21 (venv c-laya); weights HF moganai/lavoir@4c5eaeb99b23 (CC-BY-NC-4.0); Lavoir.predict; adapter servers/adapters/lavoir_model.py + laya_guard.py |
| laya-multilingual | 8136 | `c-laya` |  | laya==0.3.21 (venv c-laya) on HF convaiinnovations/laya-multilingual@e4e9ddf21a7b; laya.load + Agent.predict(max_len=8192); adapter servers/adapters/laya_multilingual.py + laya_guard.py |
| lev-350m | 8144 | `d-lev` | `src/lev` | github franckverrot/lev@c48a945 python -m lev.serve via servers/adapters/lev_strict.py; weights HF franckverrot/lev-350m@ab08ad8, base LiquidAI/LFM2.5-350M@9e6c6cc |
| lumma-fev-0.1b | 8141 | `d-lumma` |  | pip lumma-fev==0.1.1 (lumma_fev.server:main) + HF FrontiersMind/Lumma-fev-0.1b@80940010c333 (trust_remote_code), via servers/adapters/lumma_fev_guarded.py |
| lumma-fev-0.6b | 8142 | `d-lumma` |  | pip lumma-fev==0.1.1 + HF FrontiersMind/Lumma-fev-0.6b@d7a23900f2c5 (trust_remote_code), via servers/adapters/lumma_fev_guarded.py |
| openjev-nli-0.8b | 8115 | `a-rsi` |  | HF AlexWortega/openjev@26de23c44b67586b4bea31c0ef2e016e3068ae66, subfolder qwen3.5-0.8b-nli-v2s-long + code/; OpenJev.decide on OpenJevCrossEncoder, wrapped by servers/adapters/openjev_nli.py; venv a-rsi |
| openthai-0.8b | 8104 | `openthai` |  |  |
| qwen3.5-0.8b-decision | 8113 | `a-rsi` |  | HF mghafiri/qwen3.5-0.8B-decision-model@4a9939034a27006b4b62ad5aed65a90b82d0d350, bundled jevlite.model.SystemOne, wrapped by servers/adapters/mghafiri_decision.py; venv a-rsi |
| rsi-jev-0.8b | 8111 | `a-rsi` | `src/RSI-Jev` | GitHub Shanghua-Gao/RSI-Jev@8f34a4f2afadba288e77905eeeec1f22ac3a0981 scripts/serve.py --ckpt <HF shgao/rsi-jev-v1.0-qwen3.5-0.8b@f9248caceb89caf2e6c968ea33bf0d6eb7f957b0> (load_release; base Qwen/Qwen3.5-0.8B-Base@dc7cdfe); venv servers/venvs/a-rsi |
| simplejev-0.8b | 8116 | `a-simplejev` | `src/simple-jev` | GitHub featherless-ai/simple-jev@c5363e0a2a9d22cb503e8e226daf6d2d5794e415, entry point simple-jev, stock Qwen/Qwen3.5-0.8B@2fc06364715b967f1860aea9cf38778875588b17; venv servers/venvs/a-simplejev |
| system-one-gemma | 8143 | `d-peft` |  | github akash-kamat/system-one-gemma@cc75aa8: infer.load_trained_model + infer.score via servers/adapters/system_one_gemma.py |
| systemone-lite-0.5b | 8112 | `a-s1lite` | `src/a-s1lite-models` | GitHub fritzprix/systemone-lite@fc6fbe3e9a3976d5c2976589f0f7176cfdcfb110 (systemone_lite.api:main); weights HF dwidlee/systemone-lite-0.5b@4214f92acfe5e23d3a3ad55429e117b85db9d584; venv servers/venvs/a-s1lite |
| tasksource-jev-nano | 8146 | `d-pylate` |  | model-card PyLate snippet (pylate 1.6.0) on HF tasksource/tasksource-jev-nano-v0@5ee8d27 via servers/adapters/tasksource_jev_nano.py |
| tev1-0.8b | 8123 | `b-llm2jev` |  | github togethercomputer/tev1@1dde7782382c9f49d627153759b8d1deab426ce0 examples/decide.py payload() + weights togethercomputer/Tev1-0.8B-experimental@6bb2dff14b38fea90ddb14d870166ccaf77374e9; adapter servers/adapters/tev1.py |
| tiny-jev-0.6b | 8114 | `a-rsi` |  | HF lostargon/Tiny-Jev@62449fb3272d68fb8c8fe8a33082566643a76d0b, AutoModel trust_remote_code (TinyJevModel), wrapped by servers/adapters/tiny_jev.py; venv a-rsi |
| von | 8145 | `d-von` | `src/von` | pip von-sdk==1.2.3 (= github wfzyx/von@fb6e7a9) `von serve` via servers/adapters/von_guarded.py; weights HF wfzyx/von@5df8185 |

## `servers/src/`

| Directory | Kind | Source | Revision |
|---|---|---|---|
| `Intern-Decision` | git | https://github.com/internlm/Intern-Decision | `2f815802058b2464144218859ea9221c1bc0d2a8` |
| `JevEmbed` | git | https://github.com/HITsz-TMG/JevEmbed | `25cb86f495f3caaa05a610cb50d27cf924e2987b` |
| `RSI-Jev` | git | https://github.com/Shanghua-Gao/RSI-Jev.git | `8f34a4f2afadba288e77905eeeec1f22ac3a0981` |
| `decider` | git | https://github.com/Mapika/decider | `23579f7a7e8f10e1045be492af3c1c05a005d67c` |
| `kev` | git | https://github.com/jaredpalmer/kev | `3e1cd3bb588a388a06827443380befece23e68c7` |
| `lavoir` | git | https://github.com/moganai/lavoir.git | `af79b0a7a7f9864343223d3f7bd40a02747d0c1f` |
| `lev` | git | https://github.com/franckverrot/lev.git | `c48a945dbf629998d7458dcc5c16f58df964db94` |
| `llm2jev` | git | https://github.com/tic-top/llm2jev | `2b252d504972764211ef172c1155ac0fedc9c3de` |
| `openthai` | git | https://github.com/iapp-technology/openthai-systemone | `5d04bcca0c58bd10e7dac2d3d369d8f760bea6cf` |
| `semantic-router` | git | https://github.com/vllm-project/semantic-router | `74ac8a0173e8b49f6e4c21288abb8cd1c5a2b1a8` |
| `simple-jev` | git | https://github.com/featherless-ai/simple-jev.git | `c5363e0a2a9d22cb503e8e226daf6d2d5794e415` |
| `system-one-gemma` | git | https://github.com/akash-kamat/system-one-gemma.git | `cc75aa8042dec965003210fdba033fee8759735e` |
| `systemone-lite` | git | https://github.com/fritzprix/systemone-lite.git | `fc6fbe3e9a3976d5c2976589f0f7176cfdcfb110` |
| `tev1` | git | https://github.com/togethercomputer/tev1 | `1dde7782382c9f49d627153759b8d1deab426ce0` |
| `von` | git | https://github.com/wfzyx/von.git | `fb6e7a937e4fc6b6e72b2ce5035edd56bc370e54` |
| `decision-kai` | HF snapshot | https://huggingface.co/llm-semantic-router/Decision-1.0-Kai | `7185f514f54b` last revision that bundles the code; weights identical to main |
| `decision-lex` | HF snapshot | https://huggingface.co/llm-semantic-router/decision-1.0-lex | `ee8e74d912fc` last revision that bundles the code; weights identical to main |
| `gliner2.5-decide` | HF snapshot | https://huggingface.co/fastino/GLiNER2.5-Decide | `5a7adf72a23b`  |
| `julia-1` | HF snapshot | https://huggingface.co/SupersonicLabs/Julia-1 | `a85b127321d5` also installed with pip -e into c-julia |
| `lavoir-weights` | HF snapshot | https://huggingface.co/moganai/lavoir | `4c5eaeb99b23` CC-BY-NC-4.0 |
| `laya-multilingual` | HF snapshot | https://huggingface.co/convaiinnovations/laya-multilingual | `e4e9ddf21a7b`  |
| `a-s1lite-models` | HF snapshot | https://huggingface.co/dwidlee/systemone-lite-0.5b | `4214f92acfe5e23d3a3ad55429e117b85db9d584` not a copy: symlinks `systemone-lite-0.5b` and `local` to the HF cache snapshot (the server treats the request's model name as a path) |

None of the git checkouts is modified; every change to an author's behaviour lives in `servers/adapters/` (for
example the length guards that answer HTTP 422 instead of truncating, and the CUDA device guard for Decision Kai/Lex).
Also needed: the JevBench harness in `third_party/jevbench` (https://github.com/fstandhartinger/jevbench @ `9ec6f15`).

## Rebuilding on another machine

1. Clone the git sources above into `servers/src/<directory>` and check out the listed commit; download the HF
   snapshots at the listed revisions into `servers/src/<directory>` (or, for `a-s1lite-models`, create the two
   symlinks to the cached snapshot).
2. For each venv: `python3.12 -m venv servers/venvs/<venv>` and `servers/venvs/<venv>/bin/pip install -r
   servers/envs/<venv>.txt` (for a local `-e servers/src/...` line, install that directory in editable mode).
3. Download the weights named in the model files (`huggingface-cli download <repo> --revision <rev>`).
4. `python3 servers/launch.py start && python3 servers/launch.py wait`; with several GPUs set
   `ARENA_GPUS=0,1,...` first (see `servers/launch.py`).
