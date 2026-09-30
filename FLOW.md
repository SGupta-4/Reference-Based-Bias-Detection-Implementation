# FLOW.md

**Purpose.** This file documents how execution travels: CLI entrypoint → stage runner → module functions, in call order, with the data and cache artifact passed at each boundary. Keep one section per pipeline stage.

While a change is in progress, the `## Currently modifying` section names the exact functions and files on the path being changed. It is cleared (reset to "nothing") in the commit that finishes the change.

**Entry format** (per stage)
```
## Stage: <name>
Entry: <CLI invocation>
1. <module.function>(inputs) → outputs            [artifact read / written]
2. ...
Skip condition: ...
Resume behaviour: ...
Failure modes: ... (link B-###)
```
Status: **planned**. Function names are the intended public API and are fixed when implemented.

---

## Currently modifying
nothing

---

## Entrypoint

```
python -m rbbd.cli run --config configs/<x>.yaml [--stages a,b] [--force STAGE] [--dry-run]
 └─ cli.main(argv)
     └─ config.load(path) → Config (base.yaml ← tier file; config_hash)
     └─ utils.env.detect() → Env (kaggle?, gpus, paths); secrets read into env vars, never logged
     └─ utils.seeds.seed_all(cfg.seed)
     └─ runner.run(cfg, env, stages)
          for stage in topo_order(requested ∩ graph):
              if manifest.is_valid(stage, run_key) and not forced: log "cache hit: <stage>"; continue
              outputs = STAGE_FNS[stage](cfg, env, upstream_outputs)
              manifest.write(stage, run_key, inputs, outputs, timings, complete=True)
```
`python -m rbbd.cli probe` → `utils.env.probe()` → `artifacts/env/<session>.json` (nvidia-smi, df -h, free -g, torch/CUDA, bf16 support, gated-access checks).
`python -m rbbd.cli status --config C` → prints manifest validity per stage.

## Stage: sentences
Entry: `run --stages sentences`
1. `data.sentences.load_sets(cfg.sentences)` → {targets[variant][group], pos[variant], neg[variant], anchors[source]} [reads `resources/sentences/**`]
2. `data.sentences.validate(sets)` → raises on count/length/overlap violations (App. F stats; PLAN §2)
3. `data.anchors.subsets(anchors, sizes, seeds)` → nested index lists
4. `data.sentences.build_union(sets)` → SentenceUnion(texts, hashes, index maps) [writes `manifests/sentences/<key>.json` with per-set hashes]
Skip condition: all set hashes unchanged.
Failure modes: validation error → fix the text files, not the checks.

## Stage: ftdata
1. `data.ft_data.load_wgm_train(revision)` → rows (HF datasets, gated; token from env)
2. `data.ft_data.select_split(rows, "unharmful"|"harmful", n=8000, seed)` → row IDs (D-010, D-011) [writes `ftdata/<split>/<data_key>/ids.json, stats.json`]
3. `data.ft_data.render(rows, tokenizer)` → prompt–completion records (in-memory only; no text persisted outside the HF cache)

## Stage: train
1. `finetune.sft.train(cfg.model, cfg.regime, split, data_key, seed, device)` → endpoint dir
   - `models.loading.load_for_training(...)` (QLoRA NF4 | fp16 LoRA | fp32-master full FT)
   - TRL `SFTTrainer.train(resume_from_checkpoint=latest)`; guard on loss and grad-norm (D-006)
   [writes `train/<model>/<regime>/<split>/seed<k>/<train_key>/`; Tier 2 full endpoints are pushed to `rbbd-ckpt-private`]
2. Runner launches u and h as two processes, on `cuda:0` and `cuda:1` (D-003).
Resume behaviour: `checkpoint-*` every ~20 min of wall-clock. `complete=false` manifests trigger resume.

## Stage: embed
1. `models.loading.load_for_inference(model_cfg)` → base (fp16 sharded | fp32 Gemma), tokenizer (right pad)
2. `models.spectrum.iter_checkpoints(base, cfg)` yields (ckpt_slug, context manager):
   - `ref`: adapters disabled / base weights
   - LoRA `aXXX`: `models.adapters.build_combined(u, h, α)` → set_adapter
   - full `aXXX`: `models.interpolate.apply(base, W_h, W_u, α)` (in place)
3. For each ckpt: `utils.cache.lookup(embed_key)` → hit: log `cache hit`, no forward pass
   miss: `embed.extract.encode(model, tok, union, poolings, layer_site)`:
   - batches sorted by length (token budget)
   - forward on the base transformer only → last_hidden_state [B, T, d] (+ pre-norm hook if enabled)
   - `utils.guards.assert_finite(hidden)`
   - `embed.pooling.pool(hidden, attention_mask, kind)` → [B, d] fp32 → fp16
   - `utils.cache.write(embed_key, {mean,max,last}: [N, d], sidecar)` [writes `embeddings/.../<embed_key>.safetensors`]
Invariant: ref and every aXXX key differ only in checkpoint spec (asserted).

## Stage: deltab
1. `utils.cache.read(embed_key_ref)`, `read(embed_key_aud)` → E_ref, E_aud (CPU fp32)
2. For each ablation cell (anchor source/size/seed, attr variant, target variant, pooling, layer site):
   - `metrics.rr.relative(E_x, E_anchor)` → R [n, m]
   - `metrics.rr.bias_rel(R_T_by_group, R_P, R_N)` → B_rel per group
   - `metrics.seat.bias(...)`, `metrics.procrustes.bias_aligned(...)`, `metrics.cka.drift(...)`
3. `metrics.delta_b.table(...)` → long DataFrame [writes `deltab/<run_key>/delta_b.csv`; a copy goes to `results/<run>/delta_b.csv`]

## Stage: generate
1. `bench.<b>.build_prompts(cfg)` → prompt list + subset hash (WGM subcategory; DT 1,152 × 3 sys; ToxiGen 9 × 100)
2. `bench.generate.engine(model_cfg, lora=True)` → vLLM LLM (TP 1|2), or the HF fallback
3. For each ckpt (LoRA request per α, or the materialised Tier 2 weights): `bench.generate.run(engine, prompts, sampling, shard_size)` → JSONL shards [writes `generations/.../<gen_key>/part-XXXX.jsonl`, `done.json`]
Resume behaviour: skip shards listed in `done.json`.

## Stage: score
1. `bench.wildguard.classify(gen_dir)` (vLLM WildGuard, batched per model) → per-response harmful flag → `bench.wildguard.rate_by_topic(topic_map)`
2. `bench.decodingtrust.score(gen_dir)` → agreementIndex per group × sys prompt → BiasScore(group)
3. `bench.toxigen.score(gen_dir)` → first statement → RoBERTa → toxic fraction per group
[writes `scores/<bench>/.../<score_key>.csv`]

## Stage: analyze
1. `analysis.stats.join(delta_b.csv, scores)` → observations (ckpt ≠ ref) with Δscore = score_aud − score_ref
2. `analysis.stats.per_setting(obs)` → r, p, AUC, MAE (boot), CIs; baselines; paired RR−SEAT; threshold sweep
[writes `analysis/<run_key>/stats.csv`; mirrored to `results/`]

## Stage: report
1. `analysis.tables.build(...)` → R1–R7 Markdown/CSV
2. `analysis.plots.build(...)` → F-R1–F-R5 PNG
[writes `results/<run>/REPORT.md`, `results/<run>/figures/*.png`]
