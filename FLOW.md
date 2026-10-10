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
Status: Entrypoint, probe and sync are **implemented (M0)**; `sentences`/`ftdata` (M1), `embed` (ref M2, α sweep M4), `train` (M3) and `deltab` (M4) are implemented; other stage bodies are stubs that raise `NotImplementedError` naming their milestone. Under `--only-ckpt`, the sweep stages (`runner.SWEEP_STAGES`) leave their manifest incomplete (outcome `partial`, D-081). Function names below are the intended public API and are fixed when implemented.

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
Runner detail (implemented, M0): `runner.stage_run_keys(cfg.hash)` gives each stage `make_key({config_hash, stage, upstream run keys})`. Before a stage body runs, the runner writes `manifests/<stage>/<run_key>.json` with `complete=false`; the body may call `ctx.checkpoint(progress)`; on return the runner hashes `StageResult.outputs` (paths relative to the artifact root) and rewrites the manifest with `complete=true`. Validity = complete + same config hash + same upstream output digests + outputs unchanged. A missing upstream manifest raises `RunnerError`.

## Probe (M0)
```
python -m rbbd.cli probe [--config configs/base.yaml] [--no-require-gpu] [--skip-access] [--vllm] [--check-store]
 └─ cli._cmd_probe
     ├─ config.load(base.yaml); utils.env.detect(); utils.env.session_id()        ($RBBD_SESSION_ID)
     ├─ utils.env.probe(out_dir=<artifacts>/env/<session>, access_repos=cfg.probe.access_repos)
     │    ├─ probe_hardware(): nvidia-smi CSV, df -h, free -g, torch info → dict
     │    ├─ check_access(): HfApi.auth_check per repo (token from $HF_TOKEN) → {id: ok|error}
     │    ├─ write_json(env.json)   [redacted]
     │    └─ raise HardwareError if < 2 GPUs / sm < 7.5 / < 14 GiB   (exit code 2)
     ├─ --vllm: utils.env.probe_vllm(cfg.probe.vllm_cases, out_dir)
     │    └─ per case: subprocess `python -m rbbd.cli vllm-case --json <case> --out vllm_<name>.json`
     │         └─ utils.env.vllm_case(): mode plain | merged | lora (canary)
     │              merged: make_random_lora() → materialize_merged() → <ephemeral>/rbbd_probe/<name>
     │                      → vllm.LLM(model=<that dir>) → greedy generate → temp dir removed (D-050)
     │            on failure records failed_stage + traceback tail (D-048)
     │       → vllm_<name>.log, vllm_probe.json
     ├─ --check-store: utils.store.open_store(repo_id, "auto") → roundtrip(): write roundtrip.bin,
     │    upload_dir(env/<session>) [private check first] → forced download into empty cache → sha256 compare
     └─ write_json(env.json) with store/vllm results; exit 1 if access or store failed
```
The GPU test `tests/gpu/test_generate_gpu.py::test_vllm_hello_tp1_tp2_lora` calls `probe_vllm` directly.

## Store sync (M0)
`python -m rbbd.cli sync --path <rel>` → `utils.store.open_store(cfg.store.repo_id, cfg.store.repo_type)` (resolve type, assert private) → `HFStore.upload_dir(<artifacts>/<rel>, <rel>)` (re-asserts private, then `HfApi.upload_folder`) → prints the commit id. Next session: `HFStore.download_file` / `hf_hub_download` for the paths it needs.

`python -m rbbd.cli status --config C` → `runner.stage_status()` prints `valid | incomplete | missing | stale (reason)` per stage.

## Stage: sentences (implemented, M1)
Entry: `run --stages sentences` → `runner.STAGE_IMPLS["sentences"]` → `data.sentences.stage(ctx)`
1. `data.sentences.load_sets()` → `SentenceSets` [reads `resources/sentences/{targets.tsv, anchors.tsv, positive.src, negative.src}`]
   - `render_slots(src, k)` (k = 0 base, 1–3 Synonyms v1–v3), `render_template(tpl, group)`, `subject_variant(s, "subj_v1"|"subj_v2")`, `anchor_assignment(42, 24, 1000)`
2. `data.sentences.validate(sets)` → stats dict, or `SentenceValidationError` listing every failed check (PLAN §2; D-052)
3. `data.sentences.build_union(sets)` → `SentenceUnion(texts, hashes, index[(set, variant[, group])] → row ids)`
4. Writes `sentences/<run_key>/union.json` {union_hash, set_hashes, stats, texts, hashes, index}
3b. With `sentences.anchor_pools` (base.yaml, M4): `data.sentences.add_anchor_pools(sets, cfg)` → `data.anchors.build_pools` → `fetch_rows("alpaca"|"tulu")` [network: `tatsu-lab/alpaca` whole; `allenai/tulu-3-sft-mixture` streamed, first rows of every shard; revisions resolved and recorded] → `alpaca_candidates` / `tulu_candidates` (excluded sources) → `sample_pool` (seed 0, 1,000) and `word_pool` → `anchors/{alpaca,tulu,word}` added after validation (D-081). A smoke `subset` slices every anchor source to `n_anchors`.
Anchor-size subsets are computed at ΔB time: `data.anchors.subset_rows(n_pool, size, seed)` (nested per seed).
Skip condition: valid manifest (any edit to a source file changes the union hash and the downstream keys).
Failure modes: validation error → fix the source files, not the checks.

## Stage: ftdata (implemented, M1; runs on a Kaggle CPU session)
Entry: `run --stages ftdata` → `data.ft_data.stage(ctx)`
1. `data.ft_data.load_wgm_train(dataset, config, revision)` → (HF dataset, commit sha) [gated; token from `$HF_TOKEN`]
2. `data.ft_data.check_schema(columns)` (B-006)
3. `data.ft_data.select_split(rows, "unharmful"|"harmful", n_per_split, seed)` → indices + aggregate stats (D-010, D-011)
4. For each `ftdata.census_tokenizers` model: `token_lengths(rows, tokenizer)` via the chat template → `census(lengths)` (D-042, D-054)
5. Writes `ftdata/<run_key>/{unharmful_ids.json, harmful_ids.json, stats.json}` (indices and aggregates only; no text, D-037)
M3 renders training records with `to_messages(row)` (TRL prompt–completion, D-040).

## Stage: train (implemented, M3a green — D-069; Tier 1/2 runs in M3b/M3c)
Entry: `run --stages ftdata,train` → `runner.STAGE_IMPLS["train"]` → `finetune.sft.stage(ctx)`
1. `sft.plan_jobs(cfg, root, ftdata outputs)` → one `Job` per (model, regime, seed, split), u/h adjacent
   - reads `ftdata/<run_key>/{stats.json (dataset revision), <split>_ids.json}`
   - `spec_from_config` → `TrainSpec` (App. C hyperparameters, `data_key` = revision + split + ids) → `train_key`
   - job dir `train/<slug>/<regime>/<split>/seed<k>/<train_key>/`
2. `train.parallel: true`: each u/h pair runs as two processes `python -m rbbd.cli train-one --config C --model S --regime R --split X --seed K` with `CUDA_VISIBLE_DEVICES` 0 / 1 (D-003); logs → `<job>/train.log`. Jobs with `done.json` are skipped. (`parallel: false` runs `run_job` in-process: CPU tests.)
3. `cli._cmd_train_one` → re-plans from the config + ftdata manifest → `sft.run_job(cfg, job)`:
   - `load_rows` (WildGuardMix at the recorded revision, selected ids; text stays in the HF cache) → `models.loading.load_tokenizer` → `load_model(spec)` (lora fp16 | qlora NF4 | full fp32 masters; `device_map {"": 0}`)
   - `train_one(spec, rows, tok, model, job_dir)` (refuses > 1 visible GPU; `set_seed(spec.seed)` before the trainer, D-068):
     - `build_examples` → [{input_ids, completion_mask}] ≤ max_length; drops prompt-fills-window rows (D-055) → `data.json`
     - TRL `SFTTrainer(sft_config(spec), peft_config if LoRA/QLoRA, callbacks=[GuardCallback (D-065), WallClockSaveCallback (20 min)])`
     - `trainer.train(resume_from_checkpoint=latest_checkpoint(job/checkpoints))`
     - writes `final/` (PEFT adapter, or fp16 full model), `train_log.json`, `done.json` {global_step, losses, resumed_from, tokens_per_second, scaler_skipped_steps}
4. Stage outputs: every job's `final/` and `done.json` (directory hashes in the manifest).
Tier 2 (M3b, D-072): `plan_jobs` is seed-major and builds one candidate (spec, dir) per `train.batch_layouts` entry; `train-one` tries the active layout once and on a CUDA OOM writes `oom.json` and exits 75; `_run_parallel` relaunches a fresh process for the next layout (D-074) and logs every attempt to `Job.log_path`; `checkpoint_dir` sends full-FT checkpoints to `<ephemeral>/rbbd_ckpt/…`; with `train.sync_each_pair` `_run_parallel` calls `sync_jobs` after each pair. A new session first runs `cli restore --path train/<slug>` (→ `HFStore.download_dir`) so finished jobs are skipped.
Tier 1 (M3c, D-076): runtime decisions arrive as `--set` overrides (`config.load(path, overrides)`, hashed) and `_launch` forwards them to each `train-one`; the notebook takes them from `rbbd.finetune.session` (`probe_results` → `epoch_decision` / `gemma_fp16_ok` → `session_ok`). With `train.activation_monitor`, `train_one` attaches `ActivationMonitor` to `base_decoder(model).layers[-1]` before the trainer is built.
Deadline pause (D-079): with `$RBBD_TRAIN_DEADLINE_UNIX` set (the M3c notebook sets session start + 11 h), `run_job` → `train_one(deadline_unix=…)` adds `DeadlineCallback`; at the deadline it saves `<job>/checkpoints/checkpoint-N` and stops, `train_one` raises `SessionPaused` (no `done.json`), `train-one` exits 76, and `_run_parallel` records the job as paused, calls `sync_jobs(..., include_unfinished=True)` and raises `TrainingPaused` (train manifest stays incomplete). Next session: `cli restore --path train/<slug>` → `run` → the done job is skipped and `train_one` resumes the paused one from its checkpoint.
Throughput probe (M3c-T8, D-070): the same path with `train.max_steps`/`train.max_minutes` set; `ThroughputCallback` (steady tok/s from TRL's `num_tokens`) and `TimeLimitCallback` feed `done.json` {tokens_per_second_steady, peak_mem_gib, projected_hours}; `notebooks/m3c_probe.ipynb` tries batch layouts b4 → b2 → b1 on OOM.
Skip condition: valid manifest; per job, `done.json`.
Resume behaviour: a killed job restarts from its newest `checkpoints/checkpoint-N` (DC-13).
Failure modes: `NonFiniteError` from the guard (B-001, D-065); `DataBuildError` if a chat template breaks the prompt prefix; a failed subprocess raises with its log tail.

## α-merges (implemented, M3a; used from M4)
- LoRA: `models.adapters.read_adapter(final)` ×2 → `adapter_for_alpha(u, h, α)`: u at α=1, h at α=0, else `combine` = rank-2r concatenation, scaling 1 (D-007, D-066) → `write_adapter` (PEFT dir + `rbbd_meta.json`).
- Full FT: `models.interpolate.load_endpoint` ×2 (fp32 CPU) → `apply_alpha(model, w_h, w_u, α)` in place, or `materialize(...)` to ephemeral disk only (D-008, D-050).

## Stage: embed (ref: M2 green — D-064; α sweep: M4 — D-081)
M4 sweep (`embed.checkpoints: [ref, a100 … a000]`, depends on `sentences` + `train`):
1. `embed.extract.ftdata_outputs(cfg)` → `models.spectrum.spectra(cfg, root, outputs)` → `sft.plan_jobs` → one `Spectrum` per (model, regime, seed) with both `done.json` [`train/.../final/`]; LoRA-type first, full FT last
2. Per model: `Activator(load)` (lazy: `download` + `load_for_inference` at the reference's `LoadSpec`, layer check). Plan = `ref` then, per spectrum, every requested slug; endpoint hashes = `manifest.hash_path(final/)`
3. Per (spectrum, slug): key fields = shared fields + `spectrum.checkpoint_fields` → `assert_same_but_checkpoint(ref_fields, fields)` (D-004) → `extract_cached(cache, "embeddings/<slug>/<regime>-s<seed>/<ckpt>", …, load=lambda: activator.get(sp, slug))`:
   - `ref` → `Activator.plain()` (unloads PEFT if present; raises after full-FT weights were applied)
   - LoRA/QLoRA → `adapters.adapter_for_alpha(u, h, α)` → `write_adapter(<ephemeral tmp>)` → `PeftModel.from_pretrained` / `load_adapter` + `set_adapter` + `delete_adapter(previous)` → tmp removed
   - full → `interpolate.load_endpoint` ×2 (fp32 CPU, once per spectrum) → `apply_alpha(model, w_h, w_u, α)`
   - then `encode` as below; `ctx.checkpoint({"done": [...]})` after every entry
4. Writes `embed/<run_key>/index.json` [{model, regime, seed, ckpt, alpha, key, path}] (hashed output) and `timing.json` / `timing_only_<slug>.json` (log, not hashed)
Reference-only extraction (M2 path, `embed.checkpoints == [ref]`):
Entry: `run --stages sentences,embed` → `runner.STAGE_IMPLS["embed"]` → `embed.extract.stage(ctx)`
Depends on `sentences` only while `embed.checkpoints == [ref]` (`runner.stage_deps`, D-057).
1. Read `sentences/<run_key>/union.json` from the upstream manifest → texts, union_hash, index
2. `embed.extract.ExtractSettings.from_config(cfg.embed)` (poolings, layer sites, max length, token budget)
3. Per model in `cfg.models`: `LoadSpec(id, resolve_revision(id), precision, placement)`; key fields = {model spec, checkpoint ref, layer, settings + tokenizer policy, union_hash, schema_version.embed} → `make_key`
4. `extract_cached(cache, "embeddings/<slug>/base/ref", key_fields, texts, settings, load)`:
   - hit → `TensorCache.read` (logs `cache hit`; `load` is never called, so there is no download and no forward)
   - miss → `load()` = `models.loading.download(spec)` [timed] + `load_for_inference(spec)` [timed] + layer-count check (D-017) → `encode(model, tok, texts, settings)`:
     - `token_lengths` → `make_batches` (longest first, batch × longest ≤ token_budget)
     - per batch: tokenizer (right pad, BOS, truncation) → `base_decoder(model)(input_ids, attention_mask)` → `last_hidden_state` [B, T, d] (+ pre-norm via a forward hook on the last layer when requested)
     - `assert_finite(hidden)` → `pool_all(hidden, mask, poolings)` [B, d] fp32 → fp16 → `assert_finite(fp16)` → scatter to input order
   - `TensorCache.write(...)` {mean, max, last[, pre_norm/*]}: [N, d] fp16 + sidecar {fields, text_hashes, stats}
5. Writes `embed/<run_key>/timing.json` {download, load, extraction seconds, padding waste, peak GiB per GPU, cache hit}
`python -m rbbd.cli compare-embeddings --a <cfg> --b <cfg>` → `cached_ref_entry` ×2 → `compare_entries` (cosine per pooling; per-group B under RR and SEAT via `metrics.delta_b.from_union`) → `env/<session>/compare_<run_name>.json` (D-058).
Invariant (M4): ref and every aXXX key differ only in the checkpoint field.
GPU checks (`tests/gpu/test_extract_gpu.py`): DC-06 per D-061 calls `load_for_inference` (fp16 on cuda:0, fp32 on cuda:1) → `encode(keep_fp32=True)` on the short sentence alone vs in a padded batch (fp32, before the fp16 cast; D-063), and on the 85-text smoke union alone vs batched (fp16) → `m2_extract_gpu.json`. DC-09 hooks the last decoder layer and calls `extract_cached`.

## Stage: deltab (implemented, M4 — D-081)
Entry: `run --stages embed,deltab` → `runner.STAGE_IMPLS["deltab"]` → `metrics.delta_b.stage(ctx)` (CPU)
1. Read `embed/<run_key>/index.json` (upstream manifest) and `sentences/<run_key>/union.json` (config's sentences manifest)
2. Per spectrum (model, regime, seed): `TensorCache.read` of the model's `ref` entry + each checkpoint entry
3. `spectrum_rows(ref, entries, union, spectrum)`: per checkpoint (ref included) × `cells(union, tensors)` (primary + one-factor variations):
   - `_embedding_set` → `from_union(emb[tensor], union, target/attr variant, anchor source, anchor_subset=subset_rows(...))`
   - anchor-pool cells: RR only → `group_bias(aud, "rr") − group_bias(ref, "rr")`
   - other cells: `delta_b_all(ref_set, aud_set, rotation=R)`; R = `procrustes.fit_orthogonal(aud anchors, ref anchors)` once per (checkpoint, tensor), identity for `ref` (D-082)
4. `sanity(rows)` (M4-T5: mean RR ΔB at a100 vs a000, primary cell)
5. Writes `deltab/<run_key>/{delta_b.csv.gz, sanity.json}` (hashed) + `timing.json` (log); copies to `results/<run>/{delta_b.csv (primary cell), delta_b_cells.csv.gz, sanity.json}` (`results_dir`: `paths.results`, else `<repo>/results`)

## Diagnostics (B-030, D-084; CPU)
`python -m rbbd.cli diagnose-deltab --config C` → `analysis.diagnose.diagnose_config(cfg, root)`:
reads the config's `embed/<run_key>/index.json` + `sentences/<run_key>/union.json` (manifests by run key) → per spectrum `diagnose_spectrum(ref tensors, entries, union)` on the "mean" tensor: per-target rows (`metrics.rr.row_bias_rel`, `metrics.seat.row_bias_seat`, Procrustes via `fit_orthogonal`) → method means, α trend, h − u contrast with `template_bootstrap`, S⁺/S⁻ decomposition, Alpaca-pool control, geometry, ΔB per group → `results/<run>/b030_diagnostics.json`. Kaggle: `notebooks/b030_diagnose.ipynb` (restore → diagnose → sync `results/<run>`).

## Stage: generate (implemented, M5 — D-050, D-086)
Entry: `run --stages train,generate` → `runner.STAGE_IMPLS["generate"]` → `bench.generate.stage(ctx)` (depends on `train`)
1. Once per run key: `bench.generate.prepare_items(cfg, <ephemeral>/rbbd_items/<run_key>/, benches)` → `items_<bench>.json` (prompt text, ephemeral only) + `metas.json` {bench: {source pin, item-subset hash, n_items, tags}, revisions}
   - DT: `decodingtrust.fetch("data"|"system")` (pinned commit, sha256 check) → `build_prompts` → `subset(groups, topics)`
   - ToxiGen: `toxigen.fetch(group)` ×9 → `build_prompts(files, n, seed)`
   - WGM: `wildguard.load_prompts(revision)` (gated; ids = prompt sha256) → `[:max_prompts]`
2. `load_context` → `models.spectrum.spectra(...)` (as embed) → `plan_jobs(cfg, root, metas, spectra, revisions)` → one `GenJob` per (model, `base`|`<regime>-s<seed>` for `bench.seeds`, ckpt, bench); `gen_key` from `gen_key_fields`; out dir `generations/<bench>/<slug>/<spectrum>/<ckpt>/<gen_key>/`
3. Per checkpoint with unfinished jobs (`_checkpoint_groups`): `_launch` → `python -m rbbd.cli generate-one --model --spectrum --ckpt --items-dir` with `CUDA_VISIBLE_DEVICES` = "0" / "1" (TP 1, two in parallel) or "0,1" (TP 2); `bench.parallel: false` runs `run_checkpoint` in-process (tests)
   - `cli._cmd_generate_one` → `load_context` → `run_checkpoint(cfg, root, jobs, items_dir, spectrum)`:
     1. `materialize(load_spec, spectrum, ckpt, <ephemeral>/rbbd_merged/<slug>-<spectrum>-<ckpt>, root)`: ref → downloaded snapshot; LoRA → CPU base + `adapters.delta_w(adapter_for_alpha(u, h, α))` in fp32 → `save_pretrained`; full → `interpolate.LazyEndpoint` ×2 (one fp32 tensor at a time, B-033) → `interpolate.materialize` [writes ephemeral only; asserted]
     2. `make_engine(path, engine_settings(cfg, entry))` → `vllm_generate_fn(llm)`
     3. per bench: `generate_bench(fn, tok, items, sampling_for(cfg, bench), out_dir, shard_size, base_seed, max_prompt_tokens, key_fields)` → `render` (chat template / merged-system) → per-item seed `item_seed` → `part-XXXX.jsonl` (tmp + rename) → `done.json` [writes `generations/.../<gen_key>/`]; `_append_timing` → `timing.jsonl`
     4. `finally`: delete the merged copy
   - exit 76 = paused at `$RBBD_DEADLINE_UNIX` (`GenerationPaused`); `bench.sync_each` → `_sync` uploads that checkpoint's generation dirs to the private store
4. Writes `generate/<run_key>/{index.json (one row per job: model, spectrum, regime, seed, ckpt, α, bench, gen_key, dir), tags.json (per bench: source, subset hash, item tags; no text), timing.json (DC-14, `read_timing`; a log)}`
Resume: finished checkpoints (`done.json` for every bench) are not launched; finished shards are skipped.

## Stage: score (implemented, M5 — D-086)
Entry: `run --stages generate,score` → `runner.STAGE_IMPLS["score"]` → `bench.score.stage(ctx)`
1. Reads `generate/<run_key>/{index.json, tags.json}`; `topic_map.read_map()` only if `bench.wgm.topic_map_sha256` is set and equals `topic_map.map_sha256()`
2. Labels per job → `scores/<bench>/<slug>/<spectrum>/<ckpt>/<score_key>.jsonl` (no text; skipped if present):
   - DT: `label_dt` → `decodingtrust.classify` (CPU)
   - ToxiGen: `label_toxigen` → `toxigen.first_statement` → `toxigen.classify` (RoBERTa, cuda:0)
   - WGM: `python -m rbbd.cli score-wildguard --jobs score/<run_key>/wgm_jobs.json --revision <wgm sha>` → `classify_wgm` → `wildguard.verify_template()` → vLLM WildGuard (TP 2, greedy) → `wildguard.parse`; skipped with a reason in `skipped.json` if `bench.wildguard.classify` is false or no map is pinned
3. Per (bench, model, spectrum): `item_values` → `units_of` (DT/ToxiGen groups; WGM topics) → `aggregate` (score, Δ vs ref, `paired_bootstrap` CI; WGM `group_from_topic` rows)
[writes `score/<run_key>/{bench_scores.csv, skipped.json}` and `results/<run>/bench_scores.csv`]

## Topic map (one-off, M5 — D-043, D-086)
`python -m rbbd.cli topic-map [--revision R] [--n-sheet 100]` → `wildguard.load_prompts` → `topic_map.run_mapper` (Qwen2.5-7B, vLLM TP 2, greedy; `mapper_messages` → `parse_topic`) → `write_map` [writes `src/rbbd/resources/topic_map.csv` in the checkout + `artifacts/topic_map/{topic_map.csv, labelling_sheet.csv (private, blind), map_summary.json}`].
`python -m rbbd.cli topic-agreement --labels <filled sheet> [--map artifacts/topic_map/topic_map.csv]` → `read_labels` → `agreement` (accept ≥ 0.80) [writes `artifacts/topic_map/{human_labels.csv, agreement.json}`]; exit 2 if not accepted. Accepted → commit the map, pin `bench.wgm.topic_map_sha256`.

## Stage: analyze
1. `analysis.stats.join(delta_b.csv, scores)` → observations (ckpt ≠ ref) with Δscore = score_aud − score_ref
2. `analysis.stats.per_setting(obs)` → r, p, AUC, MAE (boot), CIs; baselines; paired RR−SEAT; threshold sweep
[writes `analysis/<run_key>/stats.csv`; mirrored to `results/`]

## Stage: report
1. `analysis.tables.build(...)` → R1–R7 Markdown/CSV
2. `analysis.plots.build(...)` → F-R1–F-R5 PNG
[writes `results/<run>/REPORT.md`, `results/<run>/figures/*.png`]
