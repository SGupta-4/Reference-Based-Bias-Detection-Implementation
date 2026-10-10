# DONE.md

**Purpose.** This is the definition of done. A change counts as done **only** when every applicable check below passes with its stated expected output. Nothing is marked done on judgment alone. Paste the observed output, or a one-line summary with the command, into the commit message or the BUGS.md entry.

**Entry format.** Each check has an ID (`DC-##`), a command, an expected output, and a scope (when it applies). New checks are appended with the next ID and never renumbered. A check can only be retired through a DECISION.md entry.

**Placeholder rule.** Test skeletons created at planning time contain a docstring only. Any docstring containing `PLACEHOLDER(M<N>)` is **not** a passing test. DC-19 checks that no placeholder remains for the milestone being closed.

---

## Checks

| ID | Check | Command | Expected output | Applies when |
|---|---|---|---|---|
| DC-01 | Lint | `ruff check src tests` | exit 0, `All checks passed!` | every change |
| DC-02 | CPU test suite | `pytest -q -m "not gpu"` in an environment from `pip install -e ".[train,dev]"` (D-059) | all pass; wall-clock < 2 min on CPU (pytest summary time < 120 s) | every change |
| DC-03 | Identity | `pytest -q tests/test_metrics_rr.py::test_identity_delta_b_zero` | pass: ΔB(M, M) == 0 for every group, atol 1e-6, for RR and SEAT | metrics, embed, sentences |
| DC-04 | Rotation/scale invariance | `pytest -q tests/test_metrics_rr.py::test_rotation_scale_invariance` | pass: random orthogonal Q and scale s > 0 applied to one model's embeddings leave r(x) and ΔB unchanged, atol 1e-5 | metrics |
| DC-05 | Anchor permutation | `pytest -q tests/test_metrics_rr.py::test_anchor_permutation_invariance` | pass: permuting anchors leaves ΔB unchanged (atol 1e-6) | metrics, anchors |
| DC-06 | Padding invariance | CPU: `pytest -q tests/test_pooling.py`; GPU: `pytest -q -m gpu tests/gpu/test_extract_gpu.py -k padding` | pass (D-061, supersedes D-058's rule): (1) fp32, a sentence pooled alone == inside a padded batch, `assert_close(atol=1e-3, rtol=1e-4)` on the pooled vectors before the fp16 cache cast (D-063); (2) fp16, all 85 smoke sentences alone vs in their production batch, per-sentence cosine ≥ 0.9999; both for mean, max and last | pooling, extraction, loading |
| DC-07 | Merge endpoints | `pytest -q tests/test_merge.py` (+ `-m gpu tests/gpu/test_train_gpu.py::test_real_adapter_endpoints` after M3a; `RBBD_M3B_CONFIG=<tier2 cfg> pytest -m gpu tests/gpu/test_train_gpu.py -k "real_full or tier2_lora"` after each M3b session; `RBBD_M3C_CONFIG=<tier1 cfg> RBBD_M3C_SET=<overrides> pytest -m gpu tests/gpu/test_train_gpu.py -k tier1` after each M3c session) | pass: α=1 reproduces unharmful weights/ΔW exactly and α=0 the harmful ones exactly (`torch.equal`; LoRA endpoints are the trained adapters, D-066); combined ΔW == αΔW_u + (1−α)ΔW_h (fp32 tensors evaluated in float64, rtol 1e-6, atol 1e-6·max\|ΔW\|) | adapters, interpolate |
| DC-08 | Stats sanity | `pytest -q tests/test_stats.py` | pass: ROC AUC == 1.0 on perfectly separated data; \|AUC − 0.5\| < 0.05 on 10k random points (seed 0); Pearson matches `scipy.stats.pearsonr` | analysis |
| DC-09 | NaN/Inf guard | `pytest -q tests/test_guards.py::test_guard_triggers_on_injected_inf`; training (M3, D-065): `pytest -q tests/test_sft.py -k guard` | pass: `NonFiniteError` raised naming the batch; no cache file written | guards, extraction, training |
| DC-10 | Cache hit | `pytest -q tests/test_cache.py::test_second_run_logs_cache_hit_and_no_forward`; on Kaggle, run the same config twice | second run logs `cache hit` for every checkpoint and performs 0 model forward passes (forward counter == 0) | cache, embed, runner |
| DC-11 | Smoke run | `python -m rbbd.cli run --config configs/smoke.yaml` on Kaggle 2×T4 (M4: `--stages sentences,ftdata,train,embed,deltab`, D-081) | completes in < 15 min wall-clock (runner summary); writes `results/smoke/delta_b.csv` with exactly one row per (checkpoint, group) for the primary cell (8 ckpts incl. ref × 3 groups = 24 rows); from M5 on, the benchmark stages also finish | any change on the pipeline path, from M4 on (M2–M3: subset of stages) |
| DC-12 | Performance budget | M2: `python -m rbbd.cli run --config configs/tier1_llama3.1-8b.yaml --stages sentences,embed` (checkpoint `ref`); M4: `--stages embed --only-ckpt a050` on a session where a050 is not cached yet (no `--force`, which would hit the cache; D-057, D-081) | `embed/<run_key>/timing.json`: load + extraction seconds (download excluded) for all sentence sets ≤ 15 min on 2×T4; ΔB computation added from M4 | extraction, pooling, loading, metrics |
| DC-13 | Resume | `pytest -q tests/test_sft.py::test_resume_after_kill_matches_uninterrupted` (CPU, bit-identical) and `pytest -q -m gpu tests/gpu/test_train_gpu.py::test_resume_after_kill` and `tests/gpu/test_generate_gpu.py::test_resume_generation_shards` | pass: the resumed run matches the uninterrupted one (same final step / same shard set); no duplicate shards | training, generation, runner |
| DC-14 | Bench timing logged | inspect `manifests/generate/*.json` | `seconds` present per (ckpt, bench); totals within 1.2× of PLAN §8 or a BUGS entry filed | benchmarks |
| DC-15 | Docs updated | `git show --stat HEAD` | DECISION.md, FLOW.md and BUGS.md (and ARCHITECTURE.md when the shape changed) touched in the same commit as the change; `FLOW.md › Currently modifying` reads `nothing` when the change is done | every change |
| DC-16 | Environment probe | `python -m rbbd.cli probe` on Kaggle | `env.json` shows 2 GPUs with compute cap 7.5, ≥ 14 GiB each, `bf16_supported: false` (native, D-045), disk and RAM figures, `requirements.ok: true`, `access_all_ok: true` (`ok` for every repo in PLAN §5); `store.ok: true` after `probe --check-store`; no token in any output (`grep -rE "hf_[A-Za-z0-9]{30,}" artifacts/env` → no match) | M0; on any image change |
| DC-17 | Sentence-set validators | `pytest -q tests/test_sentences.py` | pass: counts (50/group, 100 P, 100 N, 1,000 anchors), mean length 7 ± 1 words, no overlaps, template disjointness, variant alignment | sentence files |
| DC-18 | No leakage | `pytest -q tests/test_leakage.py` | pass: no generated text, prompt text, secrets or WGM rows in git-tracked files or `results/`; no public dataset flags in configs | benchmarks, results, configs |
| DC-19 | No placeholders left | `grep -rn "PLACEHOLDER(M<N>)" tests` for the milestone being closed | no output | milestone close |

## Applicability by milestone

| Milestone | Required checks |
|---|---|
| M0 | DC-01, DC-02, DC-10 (unit), DC-15, DC-16, DC-19 |
| M1 | DC-01–05, DC-08, DC-09, DC-15, DC-17, DC-19 (DC-07 moved to M3 by D-056) |
| M2 | DC-01, DC-02, DC-06, DC-09, DC-10, DC-11 (sentences+embed stages), DC-12, DC-15, DC-19 |
| M3 | DC-01, DC-02, DC-07, DC-09, DC-11 (through train), DC-13, DC-15, DC-19 |
| M4 | DC-01–05, DC-10, DC-11, DC-12, DC-15, DC-19 |
| M5 | DC-01, DC-02, DC-11 (full), DC-13, DC-14, DC-15, DC-18, DC-19 |
| M6 | DC-01, DC-02, DC-08, DC-10, DC-15, DC-19 |
| M7 | DC-01, DC-02, DC-15, DC-18 |
| M8 | DC-01–DC-12 re-run on the diffed code paths, DC-15 |

## Milestone sign-off template
```
Milestone: M<N>   Tag: m<N>-green   Date:
Checks: DC-.. PASS (output: ...), ...
GPU session-hours used / budgeted: x / y
Kaggle notebook version: ...   rbbd-artifacts version: ...
ROLLBACK.md row added: yes
```
