# ROLLBACK.md

**Purpose.** Makes every milestone and every risky edit reversible, both in code and in data. Code is rolled back with git tags. Data is never deleted to roll back: the affected cache namespace is invalidated by bumping a config `schema_version` or the config hash, so new keys are written beside the old ones (ARCHITECTURE §4).

## Procedure

**At the end of every milestone:**
1. All applicable DONE.md checks pass.
2. `git tag m<N>-green && git push origin m<N>-green` (sub-milestones: `m3a-green`, …).
3. Add a row to the *Milestone tags* table: tag, commit SHA, Kaggle notebook name + version, the store commit id of `SarthakGupta414/rbbd-artifacts` after the milestone's last `sync` (D-041), date.

**Before any large or risky edit** (touching >3 modules, changing cache-key fields, pooling, metrics, merge logic, pins, or anything a DONE check depends on):
1. Add a row to *Pre-edit rollback targets*: date, planned change, rollback tag (the latest `m*-green`), files expected to change, cache namespaces that the change will invalidate.
2. Put the change in `FLOW.md › Currently modifying`.

**To roll back:**
1. `git checkout <tag> -- <files>` for a partial restore, or `git revert <commits>` on the working branch for a full one. Never force-push shared history.
2. Point notebooks at the tag: `git clone --branch <tag>`.
3. Download artifacts at the recorded store commit (`revision=<store commit>` in `hf_hub_download`/`snapshot_download`).
4. Invalidate the affected cache namespaces by bumping `schema_version` (embed / metrics / gen / score), or by reverting the config that changed. Do not delete artifact directories.
5. Re-run the DONE checks listed for that namespace (below). Record the rollback as a BUGS.md entry and, if it reverses a decision, as a superseding DECISION.md entry.

**Namespace → checks to re-run after invalidation**

| Namespace (config key) | Invalidated by | Re-run checks |
|---|---|---|
| `sentences` (set hashes) | any sentence-file edit | DC-17, DC-03–05, DC-11 |
| `embed.schema_version` | pooling, layer site, tokenizer settings, precision/placement | DC-06, DC-09, DC-10, DC-11, DC-12 |
| `train` (train_key) | hyperparameters, data filters, seeds | DC-07, DC-13, DC-11 |
| `metrics.schema_version` | RR/SEAT/Procrustes/CKA/ΔB code | DC-03, DC-04, DC-05, DC-11 |
| `gen.schema_version` | prompt building, sampling, engine | DC-11, DC-14, DC-18 |
| `score.schema_version` | scorer parsing or aggregation | DC-11 (bench parity tests in DC-02) |
| `analysis` | stats definitions | DC-08 |

**Pruning.** Old artifact keys may be deleted only after the user approves (CLAUDE.md action boundaries), and only in a commit that records what was pruned here.

## Milestone tags

| Tag | Commit | Kaggle notebook (version) | Store commit (rbbd-artifacts, D-041) | Date |
|---|---|---|---|---|
| `m0-green` | `7d956c162e9c6dfec9872104c7951a67d0f9e1d4` | `00_probe.ipynb` (Kaggle version not reported; session `20261001T094200Z`, run S03) | `a3d5ae3a9bf9c2e3a557af9a82acff77c08b13d4` | 2026-10-01 |
| `m1-green` | `8277155ce97e4ca95388fd1c1399fdc5f389769a` | `run_stage.ipynb` (CPU; Kaggle version not reported; config hash `b4a36d44ede276ec`) | not reported (sync output not pasted) | 2026-10-01 |
| `m2-green` | `8e26c8d75dfe64dc26e3fb6b1e834f038dbb46dd` | `m2_extract.ipynb` (Kaggle version not reported; pipeline runs session `20261001T113125Z` at `64498de`, GPU tests session `20261001T122025Z` at `8e26c8d`) | embeddings `a9babfcbc23f507ab7639c68870f2f58a7c30afc`; env `1eeebb91423a60312d134486d098414cc5a25842` | 2026-10-01 |
| `m3a-green` | `7b41de1964d6509b34ccfa4e00bcf245d1c77552` | `m3a_train.ipynb` (Kaggle version not reported; session `20261001T154608Z`) | train `797549be8780c968e459ce8eca9adb32c8c5f250`; env `ed4766fbd225648eeb1385862a1970ff0dd0dd44` | 2026-10-01 |
| `m3b-green` | `4a43b157c7d0c99c6cdd8ce9bc12a6d7e38627a4` | `m3b_train.ipynb` (Kaggle versions not reported; Qwen session `20261001T181946Z` at `5d43021`, Llama-1B session `20261002T063404Z` at `4a43b15`) | Qwen `train/qwen2.5-0.5b-it` `2aab01a0cf9430c2bd12ffe6e17d5e502a893316`; Llama-1B per-pair `e20cb635`, `e568ce32`, `1a7bf584`, `a8f167db` | 2026-10-02 |
| `m3c-green` | `0dd00fba87a1bdfbd32698951c723ec6e35801d7` | `m3c_train.ipynb` (Kaggle versions not reported; Llama `20261007T133950Z` and Mistral `20261008T020912Z` at `8ab97fb`, Gemma `20261008T114622Z` + `20261009T072830Z` at `0dd00fb`) | Llama train `b891b5fb`; Mistral train `abca04f5`; Gemma train `d3f33fe2` (harmful `ecbe6852`, unharmful `76fe511e`) | 2026-10-09 |
| `m4-green` | `55a2f28dde918ea1b54e25506b38f6dfc703e18a` | `m4_deltab.ipynb` (Kaggle versions not reported; sessions `20261009T101248Z`, `20261010T010156Z`, `20261010T031931Z`) | embeddings: Qwen `5311992a`, Llama-1B `8929b7df`, Llama-8B `8bac969d`, Mistral `32c2917c`, Gemma `673904d6`; results: smoke `9716aa8f`, Qwen `375350f5`, Llama-1B `d9d7d282`, Llama-8B `99442887`, Mistral `6c62c74d`, Gemma `1468305f` | 2026-10-10 |

Note: `m0-green` and `m1-green` were created as annotated tags in the agent session, but pushing tags is blocked by that session's git policy (HTTP 403). Until the user pushes it (`git tag -a m0-green 7d956c162e9c6dfec9872104c7951a67d0f9e1d4 -m "M0 green" && git push origin m0-green`, or a GitHub release on that commit), roll back to the commit SHAs above. To push both: `git tag -a m1-green 8277155ce97e4ca95388fd1c1399fdc5f389769a -m "M1 green" && git push origin m1-green` (and the `m0-green` command above). Same for `m2-green`: `git tag -a m2-green 8e26c8d75dfe64dc26e3fb6b1e834f038dbb46dd -m "M2 green" && git push origin m2-green`, and `m3a-green`: `git tag -a m3a-green 7b41de1964d6509b34ccfa4e00bcf245d1c77552 -m "M3a green" && git push origin m3a-green`, and `m3b-green`: `git tag -a m3b-green 4a43b157c7d0c99c6cdd8ce9bc12a6d7e38627a4 -m "M3b green" && git push origin m3b-green`, and `m3c-green`: `git tag -a m3c-green 0dd00fba87a1bdfbd32698951c723ec6e35801d7 -m "M3c green" && git push origin m3c-green`, and `m4-green`: `git tag -a m4-green 55a2f28dde918ea1b54e25506b38f6dfc703e18a -m "M4 green" && git push origin m4-green`.

## Pre-edit rollback targets

| Date | Planned change | Rollback tag | Files expected to change | Namespaces invalidated | Outcome |
|---|---|---|---|---|---|
| 2026-09-30 | M0: first implementation of cli/config/runner/utils + notebooks (touches > 3 modules) | none yet — commit `2df82d9` (docs-only state before M0 code) | `src/rbbd/{cli,config,runner}.py`, `src/rbbd/utils/*`, `configs/{base,smoke}.yaml`, `notebooks/*`, `tests/*`, `.gitignore`, `pyproject.toml` untouched | none (no artifacts exist yet) | Done: M0 green, tagged `m0-green` |
| 2026-10-01 | B-012: pin `torchvision==0.22.1` in `train`/`bench`; `torch_dtype=` → `dtype=` (D-060) | `m1-green` (or commit `92e836e`, the M2 state before this fix) | `pyproject.toml`, `src/rbbd/models/loading.py`, `src/rbbd/utils/env.py`, `notebooks/m2_extract.ipynb`, `tests/test_env.py` | none (no embedding was ever cached: every M2 load failed) | Done: verified on Kaggle, session `20261001T113125Z` |
| 2026-10-01 | M3a: `finetune/sft.py`, `models/{adapters,interpolate}.py`, train stage + CLI `train-one`, smoke train config, GPU tests, `notebooks/m3a_train.ipynb` (touches > 3 modules) | `m2-green` (`8e26c8d`) | `src/rbbd/{finetune/sft.py, models/adapters.py, models/interpolate.py, runner.py, cli.py}`, `configs/{base,smoke}.yaml`, `tests/{test_merge.py, test_sft.py, gpu/test_train_gpu.py}`, `notebooks/m3a_train.ipynb` | none (no `train/` artifacts exist yet) | Done: M3a green, tagged `m3a-green` |
| 2026-10-01 | M3b: Tier 2 training design (layout fallback, ephemeral full-FT checkpoints, per-pair sync, `cli restore`, tied-weight interpolation; > 3 modules) | `m3a-green` (`7b41de1`) | `src/rbbd/{finetune/sft.py, models/interpolate.py, utils/store.py, cli.py}`, `configs/{tier2_*,m3b_probe_*}.yaml`, `tests/{test_sft,test_merge,test_store}.py`, `tests/gpu/test_train_gpu.py`, `notebooks/m3b_train.ipynb` | none (`train` schema unchanged; Tier 2 has no artifacts yet) | Done: M3b green, tagged `m3b-green` |
| 2026-10-02 | M3c preparation (config overrides, activation monitor, LoRA regex, session rules, Tier 1 configs + notebook; > 3 modules) | `m3b-green` (`4a43b15`) | `src/rbbd/{config.py, cli.py, finetune/sft.py, finetune/session.py}`, `configs/{tier1_*,m3c_probe_*}.yaml`, `tests/{test_sft,test_config,test_session}.py`, `tests/gpu/test_train_gpu.py`, `notebooks/m3c_train.ipynb` | none (train keys pinned unchanged; Tier 1 has no endpoints yet) | Done: M3c green, tagged `m3c-green` |
| 2026-10-08 | D-079: deadline pause + resume for Gemma's two-session training (`DeadlineCallback`, exit 76, `TrainingPaused`, paused-job sync); Gemma config `epochs: 1` | `m3b-green` (`4a43b15`), or commit `a6eac37` (M3c state before this change; Llama/Mistral endpoints already trained there) | `src/rbbd/{finetune/sft.py, cli.py}`, `configs/tier1_gemma3-4b.yaml`, `tests/test_sft.py`, `notebooks/m3c_train.ipynb` | none (train keys unchanged; the deadline is not hashed; Gemma's key changes, but it has no endpoints yet) | Done: verified on Kaggle (D-080), M3c green |
| 2026-10-09 | M4: anchor pools in the union, `models.spectrum` + embed α sweep, `deltab` stage, runner `partial`, Procrustes completion (D-081, D-082; > 3 modules, embedding cache keys and a metric change) | `m3c-green` (`0dd00fb`) | `src/rbbd/{data/anchors.py, data/sentences.py, models/spectrum.py, embed/extract.py, metrics/delta_b.py, metrics/procrustes.py, runner.py}`, `configs/{base,tier1_mistral-7b,tier1_gemma3-4b,m2_gemma3-4b_fp16_probe}.yaml`, `tests/{conftest,test_anchors,test_spectrum,test_delta_b_table,test_merge}.py`, `notebooks/m4_deltab.ipynb` | `sentences`/`embed` (union hash changes with the pools: M2 reference caches are superseded, not deleted); no train key changes | Done: M4 green, tagged `m4-green` |
| 2026-10-10 | M5: `bench.*` (DT, ToxiGen, WildGuard, topic map, generate, score), CLI `generate-one`/`score-wildguard`/`topic-map`/`topic-agreement`, `generate`+`score` stages, `bench:` config, notebooks (D-086; > 3 modules) | `m4-green` (`55a2f28`) | `src/rbbd/{bench/*, cli.py, runner.py}`, `configs/{base,smoke}.yaml`, `tests/{test_bench_scoring,test_generate,test_leakage,test_manifest}.py`, `tests/fixtures/*`, `tests/gpu/test_generate_gpu.py`, `notebooks/m5_{topic_map,bench}.ipynb` | config hash changes (new `bench:` keys), so every stage's run key changes; train/embed/deltab outputs are cache hits by their own keys (no recompute); `generate`/`score` have no artifacts yet | Open: awaiting Kaggle |
