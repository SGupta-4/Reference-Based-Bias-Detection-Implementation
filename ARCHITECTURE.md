# ARCHITECTURE.md

**Purpose.** This is a high-level map of the system: its modules, each module's single responsibility, and how data moves between them. It holds no implementation detail. Update it **only when the shape changes**: a module is added, removed or re-scoped, the artifact layout changes, or the cache-key scheme changes. Such an update ships in the same commit as a D-### entry.

**Entry format.** This is a living document rather than a log. Each change adds a line to the Changelog at the bottom: `YYYY-MM-DD — <what changed> — D-###`.

Status: **planned** (nothing implemented yet).

---

## 1. Data flow

```
 resources/sentences/*.txt ──► data.sentences ──► SentenceUnion (texts + hashes, sorted by length)
 (T, P, N, anchors, variants)                            │
                                                         ▼
 HF / Kaggle Models ──► models.loading ──► base model (fp16 sharded | fp32 Gemma)
 adapters (store) ────► models.adapters ──► α-adapter (rank-2r concat)   ┐
 Tier-2 endpoints ────► models.interpolate ─► W(α) in place              ├─ models.spectrum: ref + 7 α
                                                         │                ┘
                                                         ▼
                                   embed.extract  (one pass, guards, pooling)
                                                         │
                                                         ▼
                        embeddings cache  (fp16 safetensors, content-keyed)
                                                         │
                                  ┌──────────────────────┼──────────────────────┐
                                  ▼                      ▼                      ▼
                           metrics.rr            metrics.seat /          metrics.cka
                        (RR matrices, B_rel)     procrustes (B)          (drift)
                                  └──────────────┬───────┴──────────────────────┘
                                                 ▼
                                   metrics.delta_b ──► ΔB table (long CSV)
                                                                   │
 WGM test / DT / ToxiGen prompts ──► bench.generate (vLLM) ──► generations (JSONL shards)
                                                 │                                  │
                               bench.wildguard / decodingtrust / toxigen (scorers)  │
                                                 ▼                                  │
                                   benchmark scores (per ckpt × group|topic)        │
                                                 └────────────► analysis.stats ◄────┘
                                                                  (join, Pearson, ROC, MAE, CIs)
                                                                        │
                                                                        ▼
                                                     analysis.tables / plots ──► results/
```

Training sits upstream of the spectrum:
`data.ft_data (WGM splits) ──► finetune.sft ──► endpoint adapters / checkpoints ──► artifact store`

## 2. Modules

| Module | Single responsibility |
|---|---|
| `rbbd.cli` | Parse the command line and dispatch to `runner`, `utils.env` (probe), `utils.store` (sync) or `embed.extract` (compare-embeddings). |
| `rbbd.config` | Load and merge YAML configs; compute the canonical config hash. |
| `rbbd.runner` | Own the stage graph; decide skip/resume from manifests; time stages. |
| `rbbd.utils.env` | Detect Kaggle; probe hardware; read secrets without exposing them. |
| `rbbd.utils.cache` | Build content keys; read/write safetensors + sidecars; log `cache hit`. |
| `rbbd.utils.manifest` | Write and validate `manifest.json`. |
| `rbbd.utils.seeds` | Seed Python, NumPy and torch in one place. |
| `rbbd.utils.guards` | Enforce the NaN/Inf invariant. |
| `rbbd.utils.logging` | Structured logs with secret redaction. |
| `rbbd.utils.store` | The private HF artifact store: privacy check, `upload_folder`, downloads (D-041). |
| `rbbd.data.groups` | The 24 DT groups, 9 topics (T2) and ToxiGen map. |
| `rbbd.data.sentences` | Load, validate and hash sentence sets; build the deduplicated union. |
| `rbbd.data.hashing` | Text normalisation and content hashes. |
| `rbbd.data.anchors` | Build word/Alpaca/Tulu anchor pools and nested size subsets. |
| `rbbd.data.ft_data` | Select the WGM train splits (D-010, D-011) and render SFT examples. |
| `rbbd.models.loading` | Load tokenizer and model at a given precision, placement and pad policy. |
| `rbbd.models.adapters` | Build exact α-combined LoRA adapters (D-007) and materialise merged α-weights for generation (D-050). |
| `rbbd.models.interpolate` | In-memory full-weight interpolation for Tier 2 (D-008). |
| `rbbd.models.spectrum` | Define the α list and checkpoint slugs; find finished u/h pairs via `plan_jobs`; switch one loaded base between checkpoints (`Activator`). |
| `rbbd.embed.pooling` | Mask-aware mean/max/last pooling. |
| `rbbd.embed.extract` | Batched final-layer extraction into the cache. |
| `rbbd.metrics.rr` | Relative representations and B_rel (Eq. 4–6). |
| `rbbd.metrics.seat` | SEAT B (Eq. 2–3). |
| `rbbd.metrics.procrustes` | Orthogonal alignment + Procrustes-SEAT. |
| `rbbd.metrics.cka` | Linear CKA and drift. |
| `rbbd.metrics.delta_b` | ΔB per (checkpoint, group, method, ablation cell) (Eq. 7). |
| `rbbd.finetune.sft` | TRL SFT for LoRA, QLoRA and full FT, resumable. |
| `rbbd.bench.generate` | `generate` stage: prompt sets once per run, one `generate-one` process per checkpoint (merge to ephemeral disk, one vLLM engine for all benchmarks), paired per-item seeds, resumable shards (D-050, D-086). |
| `rbbd.bench.score` | `score` stage: per-sample labels, per-unit scores, Δ vs ref with paired-bootstrap CIs (D-026, D-086). |
| `rbbd.bench.wildguard` | WGM subcategory prompts; WildGuard input format, card check, parse; per-topic rates. |
| `rbbd.bench.topic_map` | One-off, frozen WGM prompt → topic mapping and its blind human check (D-043). |
| `rbbd.bench.decodingtrust` | DT stereotype prompts and agreement scoring (D-022). |
| `rbbd.bench.toxigen` | ToxiGen prompts, first-statement extraction, RoBERTa scoring (D-025). |
| `rbbd.analysis.diagnose` | B-030 diagnostics from cached embeddings: method agreement, α trend, h − u contrast CI, S⁺/S⁻ decomposition, group-free control, geometry. |
| `rbbd.analysis.stats` | Join ΔB with benchmark deltas; Pearson, ROC AUC, bootstrap MAE and CIs. |
| `rbbd.analysis.tables` | Build the R1–R7 tables. |
| `rbbd.analysis.plots` | Build the F-R1–F-R5 figures. |

Dependency rule: `metrics` and `analysis` never import torch model code, so they run on CPU from caches. `bench` never imports `metrics`. Only `runner` wires stages together.

## 3. Artifact store layout

Root = `$RBBD_ARTIFACTS`. It defaults to `/kaggle/working/artifacts` on Kaggle and `./artifacts` locally (git-ignored). It is mirrored path-for-path into the private HF repo `SarthakGupta414/rbbd-artifacts` with `python -m rbbd.cli sync --path <rel>` (D-041, D-046).

```
artifacts/
├── env/<session_id>.json                       # probe output
├── manifests/<stage>/<run_key>.json
├── ftdata/<split>/<data_key>/{ids.json, stats.json}          # row IDs only, no text
├── train/<model_slug>/<regime>/<split>/seed<k>/<train_key>/  # {checkpoints/, final/ (adapter | fp16 full model), data.json, train_log.json, done.json, train.log}; aggregates only
├── adapters_combined/<model_slug>/<regime>/a<α×100>/<key>/   # interior α only (D-066); PEFT dir + rbbd_meta.json; regenerable
├── sentences/<run_key>/union.json              # authored sets + anchor pools (Alpaca/Tulu text: private store only)
├── embed/<run_key>/{index.json, timing.json}   # index: one row per (model, regime, seed, ckpt) entry
├── embeddings/<model_slug>/base/ref/<embed_key>.{safetensors,json}
├── embeddings/<model_slug>/<regime>-s<seed>/<ckpt_slug>/<embed_key>.{safetensors,json}
├── deltab/<run_key>/{delta_b.csv.gz, sanity.json, timing.json}
├── generate/<run_key>/{index.json, tags.json, timing.json, logs/}   # no text
├── generations/<bench>/<model_slug>/<base|regime-s<seed>>/<ckpt_slug>/<gen_key>/{part-XXXX.jsonl, done.json, timing.jsonl}   # private
├── scores/<bench>/<model_slug>/<base|regime-s<seed>>/<ckpt_slug>/<score_key>.jsonl   # per-sample labels, no text; private
├── score/<run_key>/{bench_scores.csv, skipped.json, wgm_jobs.json}
├── topic_map/{topic_map.csv, labelling_sheet.csv (prompt text), map_summary.json, human_labels.csv, agreement.json}   # private
└── analysis/<run_key>/{joined.csv, stats.csv, tables/, figures/}
```

- Same private repo, same paths as local: `train/<model_slug>/full/<split>/seed<k>/<train_key>/final/` holds Tier 2 full-FT endpoints (fp16 safetensors + config/tokenizer); uploaded per finished u‖h pair and restored with `cli restore` (D-072, replacing D-041's `ckpt_private/…`).
- Ephemeral: full-FT trainer checkpoints `<ephemeral>/rbbd_ckpt/<slug>/<train_key>/` (D-072).
- Repo `results/<run_name>/`: public-safe aggregates only (DC-18): `delta_b.csv` (primary cell), `delta_b_cells.csv.gz` (all cells), `sanity.json`. Mirrored to `artifacts/results/<run_name>/` for the store.
- Ephemeral (`utils.env.ephemeral_dir()`, D-045): HF cache, materialised α-weights for generation (`rbbd_merged/`, deleted after each checkpoint, D-050) and benchmark prompt sets (`rbbd_items/<run_key>/`, D-086).
- Repo `src/rbbd/resources/topic_map.csv`: the frozen WGM map (prompt sha256, topic), pinned by `bench.wgm.topic_map_sha256` (D-086). `results/<run>/bench_scores.csv`: benchmark aggregates.

Slugs:
- `model_slug` = e.g. `llama3.1-8b-it`.
- `regime` ∈ {`lora`, `qlora`, `full`}.
- `ckpt_slug` ∈ {`ref`, `a100`, `a090`, `a070`, `a050`, `a030`, `a010`, `a000`}, where `aXXX` means α = XXX/100 = weight on the unharmful endpoint.

## 4. Cache-key scheme

`key = sha256(canonical_json(fields))[:16]`, where canonical JSON has sorted keys, no whitespace and floats rounded to 6 decimals.

| Key | Fields |
|---|---|
| `config_hash` | Entire merged config minus presentation-only fields (paths, log level) |
| `data_key` | Dataset id + revision, filters, sample size, seed |
| `train_key` | model id + revision, regime, hyperparameters, data_key, seed, precision/quant, schema_version |
| `embed_key` | model {id, revision, precision, placement, quant}, layer, extract settings {poolings, layer sites, max_length, token budget, tokenizer policy}, sentence-union hash, schema_version, and `checkpoint` = {slug: ref} or {slug, α, regime, seed, merge, endpoints {train_key, sha256 of final/}}; ref and audited keys differ only in `checkpoint` (asserted, D-004, D-081) |
| `gen_key` | model {id, revision, precision, placement}, checkpoint fields (as `embed_key`), bench, item-subset hash, bench source pin (repo commit or dataset sha), sampling {n, T, top_p, max_new_tokens}, engine {vllm version, dtype, tp, max_model_len, enforce_eager, seed}, `schema_version.gen`; per-item seeds derive from the config seed and the item id (D-086) |
| `score_key` | gen_key, scorer {id + commit, threshold or decoding}, `schema_version.score` |
| `run_key` | config_hash + upstream keys |

Invalidation never deletes data. Bump the relevant `schema_version` in config, or change an input, and new keys appear. Old entries are pruned only by an explicit, logged cleanup (ROLLBACK.md).

## Changelog
- 2026-09-30 — Initial planned architecture — D-019, D-029, D-030, D-032.
- 2026-09-30 — Store moved to a private HF repo; `utils/store.py` added; CLI gains `sync` — D-041, D-046.
- 2026-10-01 — Generation serves merged α-weights from ephemeral disk instead of vLLM LoRA — D-050.
- 2026-10-01 — `embed` depends on `train` only for α-checkpoints; CLI gains `compare-embeddings` — D-057, D-058.
- 2026-10-01 — `train` stage implemented: `finetune.sft` jobs run as `cli train-one` processes (u‖h on two GPUs); α endpoints are the trained adapters — D-065, D-066, D-067.
- 2026-10-01 — Tier 2 training: batch-layout fallback, ephemeral full-FT checkpoints, per-pair store sync, CLI `restore`; full endpoints stored at their `train/` path — D-072.
- 2026-10-02 — Config overrides (`--set`, hashed) forwarded to training subprocesses; `rbbd.finetune.session` holds the pre-registered M3c probe rules; activation monitor for fp16 probes — D-076.
- 2026-10-08 — Training can pause at a session deadline (`$RBBD_TRAIN_DEADLINE_UNIX`, `train-one` exit 76): the paused job and its checkpoint sync to the store and the next session resumes it — D-079.
- 2026-10-09 — M4: anchor pools join the sentence union; `embed` sweeps ref + 7 α per spectrum on one base (`models.spectrum.Activator`) and writes `index.json`; `deltab` stage writes the long ΔB table and `results/<run>/`; `--only-ckpt` leaves sweep stages incomplete — D-081, D-082.
- 2026-10-10 — `analysis.diagnose` + CLI `diagnose-deltab` (B-030, CPU, read-only over caches) — D-084.
- 2026-10-10 — M5: `generate` (one `generate-one` process per checkpoint) and `score` (`score-wildguard` process for WGM) stages; CLI `topic-map`, `topic-agreement`; `generate/`, `score/`, `topic_map/` artifact dirs; frozen topic map pinned in the config — D-086.
