# JevBench public items

The 231 public items of [JevBench](https://github.com/fstandhartinger/jevbench), copied unchanged from
`datasets/public/` at commit `9ec6f15a8773aaffbbb41e7d7e65a20599ab0cc3` (v1.4.2.2+3). These are the items the results in
[`runs/jevbench-public-20260929/`](../../runs/jevbench-public-20260929/README.md) were measured on.

| File | Items | Tier | sha256 |
|---|---|---|---|
| `easy.jsonl` | 48 | easy (JevBench v1.1) | `231df3c2c8e88a1a8c137ebe85de96ba70fabd330849098ac7b3c52c70b7172b` |
| `original.jsonl` | 72 | original | `5c2414edb3006b8bfcb70fda433f0f9ca015759433849f8d3104328a1f7c4180` |
| `hard.jsonl` | 111 | hard (JevBench v1.2) | `89e9e6becb33ed88c1de7d42dcc87531b2fb64cfaef4e1986faf7c37b3f80ebb` |

One item per line: `id`, `split`, `family`, `group` (paraphrases of one scenario share a group), `state`, `question`
(the typed question sent to the model), `labels`, `expected` (the correct answer) and `provenance`. The `id` is the
`task_id` in each model's `results.jsonl`.

## License

MIT, see [`LICENSE`](LICENSE) (Copyright (c) 2026 Florian Standhartinger and contributors). Every item's
`provenance.license` is `MIT` and its source an original JevBench-authored item. JevBench's imported decisions (whose
upstream text JevBench itself does not redistribute) and its held-out items are not among the public items and are not
included here.
