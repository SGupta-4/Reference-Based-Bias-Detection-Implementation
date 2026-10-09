# BUGS.md

**Purpose.** One entry per bug or feature, tracked from start to finish, so that no GPU-hour is lost twice to the same problem. Known risks are pre-registered here as `Open (pre-registered)` items. When one materialises, fill in the reproduction, hypotheses and fix in place.

**Entry format**
```
## B-### <title>
- Status: Open (pre-registered) | Open | In progress | Fixed | Won't fix | Superseded by B-###
- How it was found or scoped: ...
- Reproduction command: `python -m rbbd.cli ...` or `pytest -q tests/...::test_x`
- Hypotheses tried: (what failed and why)
- Fix: ...
- Verification: command + observed output
- GPU-hours lost: x.x
- Linked commits and D-### entries: ...
```
IDs are sequential and never reused.

---

## B-001 fp16 overflow produces NaN/Inf hidden states or loss (Gemma especially)
- Status: Open (pre-registered)
- How it was found or scoped: Planning. T4 has no bf16 (D-002). Gemma activations are known to exceed the fp16 range, and vLLM marks gemma3 as not supporting fp16. The paper used bf16 (App. C).
- Reproduction command: `python -m rbbd.cli run --config configs/tier1_gemma3-4b.yaml --stages embed` with `precision: fp16` (M2-T4 probe). `pytest -q tests/test_guards.py`.
- Hypotheses tried: —
- Fix: planned — D-005 (Gemma fp32 inference; fp32-compute fallback in training), D-006 (guard).
- Verification: guard test passes (DC-09); the Gemma probe outcome is recorded in D-005.
- Note 2026-10-01: vLLM 0.10.1.1 accepted Gemma-3-1B in fp16 and generated tokens (D-048). Whether those outputs are numerically sound is checked by comparing greedy fp16 and fp32 texts in the M0 rerun; until then D-005 (fp32 for Gemma inference) stands.
- Note 2026-10-01 (S02): greedy Gemma-3-1B fp16 vs fp32 agree on 3/4 prompts, diverge after ~8 tokens on the 4th; fp16 text is coherent. D-005 stays until the Gemma-3-4B check in M2-T4 (D-049).
- GPU-hours lost: 0
- Linked commits and D-### entries: D-005, D-006, D-048, D-049

## B-002 Padding-side / pooling errors
- Status: Open (pre-registered)
- How it was found or scoped: Planning. Mean pooling over pad tokens, or `last` pooling that reads a pad position under left padding, silently changes embeddings with batch composition. Llama/Mistral tokenizers lack a pad token.
- Reproduction command: `pytest -q tests/test_pooling.py` (CPU); `pytest -q -m gpu tests/gpu/test_extract_gpu.py::test_padding_invariance_fp16`
- Hypotheses tried: —
- Fix: planned — right padding, pad := eos, mask-aware pooling in fp32 (D-018).
- Verification: DC-06 (atol 1e-3 fp16).
- GPU-hours lost: 0
- Linked commits and D-### entries: D-018

## B-003 Adapter merge is not linear in ΔW
- Status: Open (pre-registered)
- How it was found or scoped: Planning. Interpolating LoRA A and B separately (or PEFT "linear" combination with √w scaling) introduces cross terms, so the "10/90" checkpoint is not a linear merge (App. C).
- Reproduction command: `pytest -q tests/test_merge.py`
- Hypotheses tried: —
- Fix: planned — concatenated rank-2r adapter (D-007). fp32 `(1−α)W_h + αW_u` for full FT (D-008).
- Verification: DC-07 plus the ΔW-linearity test.
- GPU-hours lost: 0
- Linked commits and D-### entries: D-007, D-008

## B-004 Gated model / dataset access fails on Kaggle
- Status: Open (pre-registered)
- How it was found or scoped: Planning. Llama-3.1/3.2, Gemma-3, Mistral-v0.3, WildGuardMix and WildGuard are gated. The token comes from Kaggle Secrets. The licence must be accepted for the token's account.
- Reproduction command: `python -m rbbd.cli probe` (gated-access section)
- Hypotheses tried: —
- Fix: planned — M0-T5 access check before any GPU work. Q1 answered 2026-09-30: the user reports every licence accepted and `auth_check` passing with `HF_TOKEN` (D-044).
- Verification: probe reports `access: ok` for every repo in PLAN §5. Pending the M0 Kaggle run.
- GPU-hours lost: 0
- Linked commits and D-### entries: D-036

## B-005 vLLM on T4 (Turing) fails or misbehaves
- Status: Fixed (worked around) — vLLM's Triton LoRA kernels cannot be compiled for sm75 (S02); generation uses merged weights on ephemeral disk (D-050), verified in S03 (D-051). The LoRA canary stays in the probe.
- How it was found or scoped: Planning listed V1 unsupported, gemma3 fp16 refusal, LoRA (Triton) kernels on sm75, TP=2 NCCL hangs, no FlashAttention on sm75. M0 Kaggle run S01 (session `20261001T091905Z`) confirmed the first and last, and hit the LoRA one.
- Reproduction command: `pytest -q -m gpu tests/gpu/test_generate_gpu.py::test_vllm_hello_tp1_tp2_lora` (or `python -m rbbd.cli probe --vllm`) on Kaggle 2× T4.
- Hypotheses tried:
  - V1 engine on sm75: vLLM 0.10.1.1 falls back to V0 by itself ("Compute Capability < 8.0 is not supported by the V1 Engine. Falling back to V0"), attention backend XFormers. Not a blocker.
  - Gemma 3 fp16 refused: **not** the case in 0.10.1.1 — the engine built and generated (see B-001 for numerics).
  - LoRA kernels on sm75: both Qwen2.5-0.5B rank-32 LoRA cases (TP=1, TP=2) failed in `LLM(...)` with `RuntimeError: PassManager::run failed` after ~55–67 s. The message comes from Triton's MLIR compiler, and vLLM's LoRA (punica) kernels are Triton kernels compiled during the profile run, so LoRA is the leading suspect. Not yet isolated: there was no Qwen case without LoRA, and only the exception message (no traceback) was kept.
- Fix: generation no longer uses vLLM LoRA. Each α-checkpoint is merged in fp32, saved as fp16 to ephemeral disk, served by vLLM without LoRA, then deleted (D-050). Probe cases: Qwen TP=1 and TP=2 plain and merged; a vLLM-LoRA canary expected to fail.
- Verification (S02, session `20261001T094200Z`): Qwen TP=1 fp16 no-LoRA ok; LoRA rank 16 TP=1, rank 32 TP=1, rank 32 TP=2 all fail at `construct` with `RuntimeError: PassManager::run failed` raised in `triton/backends/nvidia/compiler.py::make_llir` (`pm.run(mod)`). Not yet tested: TP=2 without LoRA.
- GPU-hours lost: ≈ 0.1 (five failed cases).
- Verification (S03, commit `7d956c1`): Qwen TP=1/TP=2 plain and merged all ok, merged temp dirs removed; `test_vllm_hello_tp1_tp2_merged` → `1 passed in 378.52s`.
- Linked commits and D-### entries: D-021, D-031, D-047, D-048, D-049, D-050, D-051; commits `c5c554b`, `7d956c1`

## B-006 Benchmark package / data drift
- Status: Open (pre-registered)
- How it was found or scoped: Planning. The paper had to patch DecodingTrust (App. C). ToxiGen prompt file names, WGM field names, subcategory strings and the WildGuard output format can change.
- Reproduction command: `pytest -q tests/test_bench_scoring.py`
- Hypotheses tried: —
- Fix: planned — pin the DT and ToxiGen commit SHAs and HF dataset revisions. Assert the schema on load. Add parity fixtures (D-022, D-023, D-025).
- Verification: parity tests pass; revisions are recorded in D entries.
- GPU-hours lost: 0
- Linked commits and D-### entries: D-022, D-023, D-025

## B-007 Kaggle 12 h cutoff interrupts a stage
- Status: Open (pre-registered)
- How it was found or scoped: Planning. Tier 1 training ≈ 7–9 h, and Gemma fp32 training may exceed one session.
- Reproduction command: `pytest -q -m gpu tests/gpu/test_train_gpu.py::test_resume_after_kill` (kills after N steps and resumes)
- Hypotheses tried: —
- Fix: planned — time-based checkpointing, resumable generation shards, `complete=false` manifests (D-030).
- Verification: DC-13.
- GPU-hours lost: 0
- Linked commits and D-### entries: D-030

## B-008 `/kaggle/working` exceeds ~20 GB
- Status: Open (pre-registered)
- How it was found or scoped: Planning (D-019 size estimate).
- Reproduction command: `du -sh /kaggle/working/artifacts` at stage end (logged by the runner)
- Hypotheses tried: —
- Fix: planned — HF cache in `/kaggle/tmp`; only fp16 embeddings and adapters persist; `max`/`last` poolings limited in scope; Tier 2 endpoints in a separate private Dataset.
- Verification: the runner warns at > 15 GB.
- GPU-hours lost: 0
- Linked commits and D-### entries: D-019, D-029

## B-009 Feature: M0 environment probe, artifact store and pipeline skeleton
- Status: Fixed — M0 complete, tagged `m0-green` on `7d956c1` (D-051).
- How it was found or scoped: PLAN §7 M0 (tasks M0-T1…T7), with the Q1–Q6 answers (D-041–D-044).
- Reproduction command: `ruff check src tests && pytest -q -m "not gpu"` (CPU); on Kaggle, `notebooks/00_probe.ipynb`.
- Hypotheses tried:
  - Using `torch.cuda.is_bf16_supported()` for DC-16: rejected, it can report True on sm75 via emulation → native flag from compute capability (D-045).
  - Redacting any `hf_\w{8,}`: rejected, it would mangle vLLM log lines such as `hf_overrides` → token-shaped pattern (D-045).
  - Reporting `access_all_ok: true` when access checks were skipped: caught in a local run (`probe --skip-access` printed `true`) → now `null` (D-045).
  - First lint run: 48 × E501 (lines > 100), including the planning-time skeleton docstrings → `ruff format` plus rewrapped docstrings; no rule was disabled.
- Fix: `src/rbbd/{cli,config,runner}.py`, `src/rbbd/utils/{env,store,cache,manifest,seeds,guards,logging}.py`, `configs/{base,smoke}.yaml`, `notebooks/{00_probe,run_stage}.ipynb`, tests.
- Verification (CPU, this container, Python 3.11.15, pinned core + dev extras):
  - `ruff check src tests` → `All checks passed!`
  - `pytest -q -m "not gpu"` → `51 passed, 7 deselected in 0.27s`
  - `grep -rn "PLACEHOLDER(M0)" tests` → no output (exit 1)
  - `python -m rbbd.cli probe --no-require-gpu --skip-access` → exit 0, env.json written; without `--no-require-gpu` → exit 2, `expected >= 2 GPUs, found 0`, env.json still written.
  - Kaggle S01 (`20261001T091905Z`, commit `93dbe07`), from the notebook summary cell:
    - DC-16 **PASS**: 2× Tesla T4, compute cap 7.5, 15,360 MiB each; `bf16_supported: false`; `requirements.ok: true`; RAM 31 GiB; `/kaggle/working` 20G; `access_all_ok: true` (no failures).
    - Store round trip **PASS**: `ok: true`, `private: true`, repo type `model`, commit `594c0470b4d638349cdf9f8a6d6bbaf6fc3b4f45`.
    - vLLM GPU test **FAIL**: Qwen LoRA cases `RuntimeError: PassManager::run failed` (B-005); Gemma fp16 and fp32 cases ok.
  - Kaggle S02 (`20261001T094200Z`, commit `c5c554b`): DC-16 PASS again; store PASS (commit `494d1593…`); Kaggle `ruff` → `All checks passed!`; Kaggle `pytest -q -m "not gpu"` → `52 passed, 7 deselected in 2.63s`; `pip check` conflicts only among unused preinstalled packages (D-049); vLLM GPU test still FAIL (LoRA, B-005).
  - Diagnostics change (D-048): `ruff check src tests` → `All checks passed!`; `pytest -q -m "not gpu"` → `52 passed, 7 deselected in 0.27s`.
- GPU-hours lost: 0
  - Kaggle S03 (commit `7d956c1`): DC-16 PASS; store PASS (commit `a3d5ae3a…`); `pytest -q -m gpu …::test_vllm_hello_tp1_tp2_merged` → `1 passed in 378.52s`.
  - Local at close: `ruff check src tests` → `All checks passed!`; `pytest -q -m "not gpu"` → `53 passed, 7 deselected`; `grep -rn "PLACEHOLDER(M0)" tests` → no output.
- GPU-hours lost (B-009 total): ≈ 0.1 (failed LoRA cases).
- Linked commits and D-### entries: D-041–D-051; commits `2df82d9`, `93dbe07`, `c5c554b`, `e73077d`, `7d956c1`

## B-010 Feature: M1 sentence sets, metrics and statistics on CPU
- Status: Fixed — M1 complete, `m1-green` on `8277155` (D-055, D-056).
- How it was found or scoped: PLAN §7 M1 (M1-T1…T8).
- Reproduction command: `ruff check src tests && pytest -q -m "not gpu"`; on Kaggle CPU: `python -m rbbd.cli run --config configs/base.yaml --stages sentences,ftdata`.
- Hypotheses tried:
  - First validation: targets averaged 5.96 words (window 6–8) → lengthened 19 templates, passives kept in sync (mean now 6.78).
  - Multi-word slot options ("relied on", "worn out") put function words on the neutral-sentence ban list → stopword filter on slot words.
  - Passive forms reused P/N slot words ("worn", "relied", "celebrated") → reworded T07, T16, T40.
  - Anchor balancing dropped the App. F anchor "Homosexuals speak multiple languages and dialects." → drop offset (5i + 7).
  - Review of all rendered sentences found "felt humiliated of", a mid-sentence "he" and an object "her" that the subject rules would garble → five source sentences rewritten.
  - The first sign-convention test used P and N drawn from the same distribution, so moving toward N's centroid was not guaranteed to reduce ΔB → the test now uses separated P/N clusters. This was an ill-posed test, not a metric bug.
- Fix: `src/rbbd/data/{groups,hashing,sentences,ft_data}.py`, `src/rbbd/resources/sentences/*`, `src/rbbd/metrics/*`, `src/rbbd/analysis/stats.py`, stage registry in `runner.py`, configs, tests.
- Verification (CPU, this container): `ruff check src tests` → `All checks passed!`; `pytest -q -m "not gpu"` → `67 passed, 7 deselected in 4.24s`; `grep -rn "PLACEHOLDER(M1)" tests` → no output.
- GPU-hours lost: 0
  - Kaggle CPU (commit `8277155`, config hash `b4a36d44ede276ec`): `run --config configs/base.yaml --stages sentences,ftdata` → `"sentences": "ran"` (0.9 s), `"ftdata": "ran"` (205.3 s); `status` → both `valid`. Census results in D-055.
- Linked commits and D-### entries: D-052, D-053, D-054, D-055, D-056; commit `8277155`

## B-011 Feature: M2 embedding extraction and cache
- Status: In progress. The second Kaggle run (session `20261001T113125Z`, commit `64498de`) passed DC-09, DC-10 and DC-12 and settled the Gemma probe (D-062). DC-06 failed its D-058 rule (B-013). Under D-061 (session `20261001T122025Z`), the fp16 part passed and the fp32 part failed on a test-measurement bug (B-014), which was then fixed. **Done:** all M2 checks pass (D-064); tagged `m2-green`.
- How it was found or scoped: PLAN §7 M2 (M2-T1…T4).
- Reproduction command: CPU: `pytest -q -m "not gpu"`; Kaggle: `notebooks/m2_extract.ipynb`.
- Hypotheses tried:
  - `embed` could not run before `train` existed → config-dependent dependencies (D-057).
  - `ExtractSettings`'s loader closure captured a loop variable (ruff B023) → bound as a default argument.
  - Adding `download()` made the stage tests touch the network → stubbed in tests like the loader.
  - Local environment: the PyTorch CPU wheel index was unreachable, and `accelerate` pulled an unpinned torch 2.14.1; it was replaced by the pinned 2.7.1 before any test ran (D-059).
- Fix: `src/rbbd/models/loading.py`, `src/rbbd/embed/{pooling,extract}.py`, `src/rbbd/metrics/delta_b.py` (`from_union`), runner deps, CLI `compare-embeddings`, configs, `notebooks/m2_extract.ipynb`, tests.
- Verification (CPU, this container): `ruff check src tests` → `All checks passed!`; `pytest -q -m "not gpu"` → `76 passed, 7 deselected in 8.20s`; `grep -rn "PLACEHOLDER(M2)" tests` → no output.
- Verification (Kaggle 2×T4, session `20261001T113125Z`, commit `64498de`; store commits: manifests `2fcc96a8`, embed `fb5226a5`, embeddings `a9babfcb`, env `0752f031`):
  - DC-10: smoke run 1 → `embed: resumed` (an incomplete manifest from the B-012 run was still in `/kaggle/working`; the resume path re-extracted all 85 texts); run 2 → `cache hit: stage embed`, no load and no forward. **Pass.**
  - DC-09 (GPU): `test_guard_active_on_real_model` passed; `m2_extract_gpu.json` → `"guard_on_gpu": "raised NonFiniteError, nothing cached"`. **Pass.**
  - DC-12: Llama-3.1-8B fp16 balanced, 9,197 texts, post- and pre-norm: load 73.4 s + extraction 71.4 s = 144.8 s ≤ 900 s; peak 7.43 / 9.21 GiB; padding waste 1.6 %. **Pass.**
  - Gemma-3-4B fp32: load 22.2 s + extraction 191.0 s; peak 9.07 / 9.92 GiB. fp16 probe: guard fired on batch 0, all values NaN → stays fp32 (D-062).
  - DC-06 (GPU): **fail** under D-058 (B-013). Rule replaced by D-061; rerun pending.
- GPU-hours lost: ≈ 0.2 (first Kaggle run, B-012)
- Linked commits and D-### entries: `92e836e`, `64498de`, `ac0f1ce`; D-057, D-058, D-059, D-060, D-061, D-062, D-063; B-012, B-013, B-014

## B-012 Bug: every M2 model load fails on Kaggle — torchvision does not match the pinned torch
- Status: Fixed and verified on Kaggle (session `20261001T113125Z`).
- How it was found or scoped: First M2 Kaggle run of `notebooks/m2_extract.ipynb` at `92e836e`. Qwen (smoke), Llama-3.1-8B and Gemma-3-4B all downloaded (≈ 9 s, 69 s, 34 s) and then failed in `AutoModelForCausalLM.from_pretrained`.
- Reproduction command: on Kaggle, `pip install -e ".[train,dev]"` at `92e836e`, then `python -c "from transformers import Qwen2ForCausalLM"`.
- Observed: `RuntimeError: operator torchvision::nms does not exist` (raised while importing `transformers.image_utils` → torchvision) → `ModuleNotFoundError: Could not import module 'Qwen2ForCausalLM'` (likewise `LlamaForCausalLM`, `Gemma3ForConditionalGeneration`). Downstream:
  - the GPU tests errored in fixture setup;
  - `compare-embeddings` found no manifests;
  - the second smoke run logged `resuming` (the first left an incomplete manifest), not `cache hit`.
  - Also: the warning "`torch_dtype` is deprecated! Use `dtype` instead!"
- Hypotheses tried:
  - Model or revision problem → rejected: three different architectures failed identically, and all after a successful download.
  - Torch/torchvision ABI mismatch → confirmed. The `train` extra reinstalls torch 2.7.1 but not torchvision, so Kaggle's image keeps a torchvision compiled for its own torch. M0 installed `bench`, whose `vllm` pulls a matching torchvision, so it never saw this.
- Fix: Pin `torchvision==0.22.1` in `train` and `bench` (D-060). Add a preflight import cell to `m2_extract.ipynb`. Switch `from_pretrained` to `dtype=` in `models/loading.py` and `utils/env.py`. Add tests `test_torchvision_pinned_with_torch` and `test_transformers_model_classes_import`.
- Verification (CPU, this container): `pip install -e ".[train,dev]"` resolves `torch 2.7.1+cu126`, `torchvision 0.22.1+cu126`, `transformers 4.57.3`. `python -c "from transformers import Qwen2ForCausalLM, LlamaForCausalLM, Gemma3ForConditionalGeneration; import transformers.image_utils"` → ok. `ruff check src tests` → `All checks passed!`. `python -m pytest -q -m "not gpu"` → `78 passed, 7 deselected in 8.97s`. Kaggle (commit `64498de`): preflight printed `preflight ok 2.7.1+cu126 0.22.1+cu126 4.57.3`, and Qwen, Llama-3.1-8B and Gemma-3-4B all loaded.
- GPU-hours lost: ≈ 0.2 (one 2×T4 session spent on downloads and failed loads)
- Linked commits and D-### entries: `92e836e` (failing run), `64498de` (fix, verified); D-031, D-060; B-011

## B-013 Bug: DC-06 fp16 padding-invariance rule (D-058) cannot be met by fp16 arithmetic
- Status: **Closed.** Both parts of D-061 pass on Kaggle (B-014 fixed the part-1 measurement).
- How it was found or scoped: `test_padding_invariance_fp16` failed on Kaggle (session `20261001T113125Z`, commit `64498de`): `1 failed, 1 passed in 13.09s`.
- Reproduction command: `pytest -q -m gpu tests/gpu/test_extract_gpu.py::test_padding_invariance_fp16` at `64498de` on 2×T4.
- Observed (`m2_extract_gpu.json`), max |alone − batched| with max |value|: mean 0.146 / 124.7; max 0.625 / 183.1; last 0.156 / 146.8. The fp16 step in [128, 256) is 0.125, so these gaps are 1–5 ulps.
- Hypotheses tried:
  - Padding leaks into real positions. Unlikely: with right padding and causal attention, pads come after every real token. Pooling over masks is covered on CPU in fp32 (`test_padding_invariance_cpu`, `test_last_token_index`, where 1e6 at pad positions never leaks). Not yet proven on the real model → D-061 part 1 checks fp32 on Qwen.
  - fp16 rounding from different GEMM shapes and attention kernels (masked batch vs single sequence). Consistent with gaps of a few ulps at the largest coordinates. D-058 assumed values in [8, 16), which made the tolerance look feasible; real values reach 183.
- Fix: D-061 replaces the rule. `test_padding_invariance_fp16` is replaced by `test_padding_invariance_fp32` and `test_padding_cosine_fp16_smoke` (85 smoke sentences, cosine ≥ 0.9999). The latter also records fp16-vs-fp32 error. The notebook's pytest cell prints full failure lines (`-rfE --tb=short`).
- Verification (CPU, this container): both new tests ran end to end with the tiny conftest model standing in for Qwen (`load_for_inference` monkeypatched); `ruff check src tests` → `All checks passed!`; `python -m pytest -q -m "not gpu"` → `78 passed, 8 deselected in 10.51s`. Kaggle: pending.
- Kaggle under D-061 (session `20261001T122025Z`, commit `ac0f1ce`): `1 failed, 2 passed in 31.80s`. Part 2 (85 smoke texts) passed. min cosine alone vs batched in fp16: mean 0.9999973, max 0.9999936, last 0.9999967 (≥ 0.9999). Recorded for comparison:
  - Padding error (relative L2): mean 0.0014 / max 0.0023.
  - fp16 vs fp32, sentence alone: mean 0.0039 / max 0.0072.
  - fp16 vs fp32, sentence in its batch: mean 0.0039 / max 0.0075.
  - So padding adds less error than fp16 itself; fp16 batched vs fp32 is no worse than alone vs fp32.
  - Max-pooling, fp16 vs fp32: min cosine 0.99993, the least precise pooling.
  - Part 1 failed: see B-014.
- GPU-hours lost: ≈ 0 (the test took 13 s; the rest of the run produced usable results)
- Linked commits and D-### entries: `64498de`, `ac0f1ce`; D-058, D-061, D-063; B-011, B-014

## B-014 Bug: DC-06's fp32 check compared fp16-rounded outputs
- Status: **Closed.** Fixed and verified on Kaggle.
- How it was found or scoped: `test_padding_invariance_fp32` failed on Kaggle (session `20261001T122025Z`, commit `ac0f1ce`). `m2_extract_gpu.json` shows `padding_invariance_fp32` max_abs: last 0.00390625, max 0.00390625, mean 0.001953125; max_rel ≈ 1.6e-3–1.8e-3; max |value| 146.75 / 182.75 / 124.56.
- Reproduction command: `pytest -q -m gpu tests/gpu/test_extract_gpu.py::test_padding_invariance_fp32` at `ac0f1ce` on 2×T4.
- Hypotheses tried:
  - Padding leak in fp32. Rejected: a leak would give O(1) errors. These gaps are 1–2 fp16 ulps, 160× below the fp16-compute gap.
  - fp16 output cast. Confirmed: the gaps are exactly 2⁻⁸ and 2⁻⁹, the fp16 rounding steps in [4, 8) and [2, 4). `encode` casts every pooled vector to fp16 before returning, so the test never saw fp32 values.
- Fix: `encode(..., keep_fp32=True)` returns the pre-cast pooled vectors, and the fp16 guard still runs. `test_padding_invariance_fp32` uses it. New CPU test `test_encode_keep_fp32_matches_cached_fp16` checks that casting the kept vectors reproduces the default fp16 output exactly. Thresholds are unchanged (D-063).
- Verification (CPU, this container): `ruff check src tests` → `All checks passed!`; `python -m pytest -q -m "not gpu"` → `79 passed, 8 deselected in 16.45s`. A dry run of `test_padding_invariance_fp32` with the tiny model gives max_abs 2.2e-8 / 6.0e-8 / 1.2e-7 (mean/max/last): fp32-level values, no fp16 steps. Kaggle (session `20261001T122025Z`, commit `8e26c8d`): `3 passed in 20.53s`. `padding_invariance_fp32` max_abs is last 1.36e-4, max 7.31e-4, mean 1.83e-4. These are non-power-of-two values, which confirms the comparison now sees fp32 values. Store commit `1eeebb91`.
- GPU-hours lost: ≈ 0.1 (one short session)
- Linked commits and D-### entries: `ac0f1ce`, `8e26c8d`; D-061, D-063, D-064; B-013

## B-015 Feature: M3a fine-tuning (SFT endpoints, α-merges, train stage)
- Status: **Done** (M3a). Run 1 (`ec87d99`) found B-016 and B-017. Run 2 (`7b41de1`, session `20261001T154608Z`) passed everything (D-069); tagged `m3a-green`.
- How it was found or scoped: PLAN §7 M3 (M3-T1–T3, M3a-T4).
- Reproduction command: CPU: `python -m pytest -q -m "not gpu"`; Kaggle: `notebooks/m3a_train.ipynb`.
- Hypotheses tried:
  - D-006 taken literally would abort fp16 training on scaler-skipped steps → D-065 (user-approved).
  - Concatenated adapters are not guaranteed to be bit-equal to the endpoints at α ∈ {0, 1} → D-066 (user-approved).
  - TRL infers `completion_only_loss` from a `prompt` column, which pre-tokenised data lacks → set explicitly (D-067).
  - TRL requires a real `PreTrainedTokenizerBase` → CPU tests use an offline word-level fast tokenizer with a chat template (`tests/conftest.make_chat_tokenizer`).
  - The data-leak assertion in a test first matched the field name `n_dropped_prompt_fills_window` → it now checks that no row text appears.
  - `test_stub_names_its_milestone` used `train` as its stub → retargeted to `generate` (M5).
- Fix:
  - Code: `src/rbbd/finetune/sft.py`, `src/rbbd/models/{adapters,interpolate}.py`, `runner.STAGE_IMPLS["train"]`, `cli train-one`.
  - Config: `train:` sections in `configs/{base,smoke}.yaml`.
  - Tests: `tests/{test_merge,test_sft}.py`, `tests/gpu/test_train_gpu.py`.
  - Notebook: `notebooks/m3a_train.ipynb`.
- Verification (CPU, this container): `ruff check src tests` → `All checks passed!`; `python -m pytest -q -m "not gpu"` → `96 passed, 9 deselected in 13.30s`; `grep -rn "PLACEHOLDER(M3)" tests` → no output.
  - DC-13 on CPU: after a kill at checkpoint 3 the run resumes to the same step with bit-identical adapter tensors.
  - DC-07 on CPU: α ∈ {0, 1} are bit-identical to the endpoints, and combined ΔW is linear within float64 rtol 1e-6.
  - A CPU dry run of `test_real_adapter_endpoints` (tiny model trained into the smoke layout) passed, with max relative linearity error 5.1e-8.
  - Kaggle run 1 (session `20261001T132720Z`, commit `ec87d99`; store commits: train `d72b5580`, env `44b5b73c`):
    - DC-11 (through train): `smoke ftdata+train wall-clock: 96 s` (< 15 min). The rerun gave `ftdata: cache hit, train: cache hit`. **Pass.**
    - Training: both endpoints 24 steps, 3 epochs, 0 scaler-skipped steps, 64/64 rows kept, 0 dropped, 5 truncated each.
      - Unharmful: train loss 1.813, final 1.526, 1,946 tok/s.
      - Harmful: train loss 2.106, final 1.775, 2,181 tok/s.
    - GPU tests `1 failed… 2 failed, 1 passed`: `test_real_adapter_endpoints` raised `AdapterError` (B-016). `test_resume_after_kill` got `assert 3 == 6` (B-017).
  - Kaggle run 2 (session `20261001T154608Z`, `7b41de1`): `4 passed in 161.27s`; DC-11 112 s; see D-069 for every number.
- GPU-hours lost: ≈ 0.15 (run 1's GPU tests)
- Linked commits and D-### entries: `ec87d99`, `7b41de1`; D-065, D-066, D-067, D-068, D-069; B-016, B-017

## B-016 Bug: `combine` rejected real endpoints because PEFT's `target_modules` order varies by process
- Status: **Closed.** Verified on Kaggle (session `20261001T154608Z`): `test_real_adapter_endpoints` passed on adapters trained in two processes.
- How it was found or scoped: `test_real_adapter_endpoints` on Kaggle (session `20261001T132720Z`, commit `ec87d99`) failed with `rbbd.models.adapters.AdapterError`.
- Reproduction command: save two PEFT adapters from two processes with `PYTHONHASHSEED=1` and `=2`, then `adapters.combine(u, h, 0.5)` → `AdapterError: endpoint configs differ in 'target_modules'`. Reproduced in this container: `["q_proj","o_proj","up_proj",…]` vs `["k_proj","o_proj","up_proj",…]`.
- Hypotheses tried:
  - Rank or DoRA mismatch. Rejected: both configs have r 16, DoRA off and empty patterns.
  - Set ordering. Confirmed: PEFT stores `target_modules` as a set, and JSON order follows each process's string-hash seed. The CPU tests trained both endpoints in one process, so they never saw it.
- Fix: `combine` compares `target_modules` as sorted lists and writes them sorted. Regression test `test_combine_accepts_target_modules_in_any_order`.
- Verification (CPU): `python -m pytest -q -m "not gpu"` → `99 passed`. Kaggle: pending.
- GPU-hours lost: included in B-015.
- Linked commits and D-### entries: `ec87d99`; D-066; B-015

## B-017 Bug: two visible GPUs made the HF Trainer use DataParallel; LoRA init not seeded per spec
- Status: **Closed.** Verified on Kaggle (session `20261001T154608Z`): resume test 6 = 6 steps with adapter diff 0.0, and the trainer refused two visible GPUs.
- How it was found or scoped: `test_resume_after_kill` on Kaggle (session `20261001T132720Z`, `ec87d99`): `assert 3 == 6`; `m3a_train_gpu.json` → `ref_global_step 3`, `resumed_global_step 3`, `max_abs_adapter_diff 0.0668`.
- Reproduction command: `pytest -q -m gpu tests/gpu/test_train_gpu.py::test_resume_after_kill` at `ec87d99` on 2×T4.
- Hypotheses tried:
  - Resume logic. Rejected: the run resumed from `checkpoint-3`, and on CPU resume is bit-identical.
  - Effective batch doubled. Confirmed: with `device_map={"": 0}` and the args' device also cuda:0, transformers 4.57 sets `is_model_parallel=False` and keeps `n_gpu=2`. It wraps the model in `nn.DataParallel` with batch 4 × 2 GPUs × grad-acc 2 = 16, so 48 rows give 3 steps. The kill hook therefore fired at the last step, and the "resume" had nothing left to do. The train stage was unaffected because each `train-one` process sees one GPU (24 steps, as computed).
  - The 0.067 adapter gap is the unseeded LoRA A init: each run drew it from a different global RNG state.
- Fix (D-068): `train_one` refuses more than one visible GPU, and seeds `set_seed(spec.seed)` right before building the trainer. `sft.SCHEMA_VERSION` → 2. The GPU DC-13 test runs ref/kill/resume as child processes with `CUDA_VISIBLE_DEVICES=0`, and a new GPU test checks the refusal. `test_real_adapter_endpoints` finds the smoke jobs by `train_key`, not by glob.
- Verification (CPU, this container):
  - `test_train_one_refuses_more_than_one_visible_gpu` passes.
  - `test_adapter_init_depends_on_spec_seed_only` fails with `set_seed` removed and passes with it.
  - A CPU dry run of the three child phases with the tiny model gives `ref exit 0` / `kill exit 3` / `resume exit 0` and `ref 6 resumed 6 checkpoint-3`, with losses at steps 4–6 identical to ref.
  - `python -m pytest -q -m "not gpu"` → `99 passed, 10 deselected`.
  - Kaggle: pending.
- GPU-hours lost: included in B-015.
- Linked commits and D-### entries: `ec87d99`; D-003, D-068; B-015

## B-018 Feature: M3c-T8 throughput probe (Llama-3.1-8B QLoRA)
- Status: **Done.** Kaggle probe at `c44ecaf` (session `20261001T171846Z`): batch 4×8 OOM, 2×16 fits; harmful run 214.8 tok/s → 19.22 h at 3 epochs → Llama trains 1 epoch (D-071).
- How it was found or scoped: PLAN §7 M3c-T8; the user chose to run it before M3b (2026-10-01).
- Reproduction command: CPU: `python -m pytest -q tests/test_sft.py -k "throughput or time_limit or projection"`; Kaggle: `notebooks/m3c_probe.ipynb`.
- Hypotheses tried:
  - `max_minutes=0` was falsy, so the time limit was silently dropped; caught by `test_time_limit_stops_training_and_done_has_projection` → now checked against `None`.
  - Measuring throughput from `n_tokens × epochs_done / seconds` mixes warm-up and the longest-batch-first step into the rate → the steady rate comes from TRL's cumulative `num_tokens` over steps > 3 (`ThroughputCallback`).
- Fix:
  - `finetune.sft`: `ThroughputCallback`, `TimeLimitCallback`, `project_hours`, and `done.json` fields `tokens_per_second_steady`, `seconds_per_step_steady`, `peak_mem_gib` and `projected_hours`.
  - `train.max_minutes` config key.
  - `configs/m3c_probe_llama3.1-8b_b{4,2,1}.yaml` and `notebooks/m3c_probe.ipynb`.
  - Decision rule fixed in D-070.
- Verification (CPU, this container): `ruff check src tests` → `All checks passed!`; `python -m pytest -q -m "not gpu"` → `102 passed, 10 deselected in 14.35s`, including the 3 new probe tests; all notebook code cells parse. Kaggle: the 4×8 layout OOMed in TRL's token-accuracy logits copy (+1.96 GiB at 12.94 GiB in use), as the fallback loop anticipates; 2×16 completed. Full numbers in D-071.
- GPU-hours lost: 0
- Linked commits and D-### entries: `c44ecaf`; D-042, D-055, D-070, D-071

## B-019 Feature: M3b Tier 2 training (full FT + LoRA, seeds, OOM fallback, per-pair sync)
- Status: **Done.** Session 1 (Qwen, `5d43021`, D-073) and session 2 (Llama-3.2-1B, `4a43b15`, D-075) are complete; tagged `m3b-green`.
- How it was found or scoped: PLAN §7 M3b-T5–T7; the user said "prepare M3b" (2026-10-01).
- Reproduction command: CPU: `python -m pytest -q -m "not gpu"`; Kaggle: `notebooks/m3b_train.ipynb` with `MODEL = "qwen2.5-0.5b"`, then `"llama3.2-1b"`.
- Hypotheses tried (design risks found before running):
  - Two parallel 1B full-FT jobs would write ≈ 15 GB of checkpoints to the 20 GB `/kaggle/working` → full-FT checkpoints go to ephemeral disk (D-072 item 2).
  - Llama-3.2-1B's tied `lm_head` is absent from saved endpoints, so the old `apply_alpha` would raise on it → tied aliases are tolerated (test with a tied tiny Llama).
  - 4×8 may OOM for 1B full FT, as it did for 8B QLoRA (D-071) → automatic layout fallback.
- Fix:
  - Code: `finetune.sft` (`batch_layouts`, `seeds_for`, multi-layout `Job`, `checkpoint_dir`, OOM fallback in `run_job`, `sync_jobs`, seed-major `plan_jobs`), `models.interpolate.apply_alpha` (tied weights), `utils.store.HFStore.download_dir` + `cli restore`.
  - Config and notebook: `configs/tier2_{qwen2.5-0.5b,llama3.2-1b}.yaml`, `configs/m3b_probe_*.yaml`, `notebooks/m3b_train.ipynb`.
  - Tests: GPU `test_real_full_endpoints` and `test_real_tier2_lora_endpoints`.
- Verification (CPU, this container): `ruff check src tests` → `All checks passed!`; `python -m pytest -q -m "not gpu"` → `110 passed, 10 deselected`. New tests:
  - layout candidates and per-regime seeds;
  - OOM fallback: OOM marker written, the next layout used, and a rerun skips both;
  - non-OOM errors re-raised;
  - ephemeral checkpoint dir;
  - full-FT resume after a kill: bit-identical, with checkpoints outside the job dir;
  - tied-embedding interpolation;
  - store `download_dir`;
  - `sync_jobs` uploads finished jobs and swallows store errors.
  - Planning dry run on the real configs: Qwen → full s0, LoRA s0, full s1, full s2 (4 pairs); Llama-1B → full s0, LoRA s0 (2 pairs). The GPU tests skip without `RBBD_M3B_CONFIG`. All notebook cells parse.
  - Kaggle session 1 (Qwen, session `20261001T181946Z`): probe bound 7.56 h → proceed; training 8.44 h, 8 jobs × 750 steps, 0 skips, 0 OOM; GPU DC-07 `2 passed in 96.01s`. Full numbers in D-073.
- GPU-hours lost: 0
  - Kaggle session 2 (Llama-1B, session `20261002T063404Z`): full FT fell back 4×8 → 2×16 → 1×32 (fresh processes); training 9.25 h; GPU DC-07 `2 passed`. Full numbers in D-075.
- Linked commits and D-### entries: `5d43021`, `4a43b15`; D-008, D-009, D-033, D-042, D-072, D-073, D-074, D-075

## B-020 Risk: smoke and Tier 2 Qwen jobs share one slug directory
- Status: Open, mitigated by design. Act on it in M4.
- How it was found or scoped: The M3b session 1 summary (D-073) listed six `done.json` files under `train/qwen2.5-0.5b-it/lora/`: the two Tier 2 endpoints plus four 24-step smoke endpoints (two from schema v1, two from v2), restored from the store.
- Reproduction command: `ls artifacts/train/qwen2.5-0.5b-it/lora/*/seed0/` after `cli restore --path train/qwen2.5-0.5b-it`.
- Hypotheses tried: n/a. The behaviour is as designed: `train_key` differs, and `plan_jobs` resolves each config's own jobs. Anything that globs `train/<slug>/<regime>/<split>/seed0/*` would match several jobs.
- Fix (planned, M4): the α-checkpoint sweep finds endpoints only through `plan_jobs` for its config. A CPU test will assert that a smoke job never resolves for the Tier 2 config. Renaming the smoke slug would change smoke keys and retrain them (≈ 2 min), but it is not needed.
- Verification: pending (M4).
- GPU-hours lost: 0
- Linked commits and D-### entries: D-072, D-073

## B-021 Bug: the OOM layout fallback failed for Llama-3.2-1B full FT, and the failure report hid the error
- Status: **Closed.** Verified on Kaggle (session `20261002T063404Z`, `4a43b15`): 4×8 and 2×16 OOMed and were relaunched in fresh processes, 1×32 trained to completion, and the probe printed each job's log.
- How it was found or scoped: M3b session 2 (`d4bbe2b`, session `20261002T042125Z`). The probe stage failed ≈ 60 s after launch with `FileNotFoundError: …/llama3.2-1b-it-probe/full/unharmful/seed0/4771744e3daa2694/train.log`, so the probe produced nothing (`proceed: false`). Correctly, no long run started.
- Reproduction command: `notebooks/m3b_train.ipynb` with `MODEL = "llama3.2-1b"` at `d4bbe2b`.
- Hypotheses tried:
  - Missing log file. Confirmed as the *reporting* bug: logs went to the 4×8 directory (`52d01d…`), but once `oom.json` marked that layout, `job.out_dir` resolved to 2×16 (`4771…`), which has no log.
  - The training process itself failed after the 4×8 OOM. Most likely cause: the in-process retry kept the failed trainer's GPU memory alive through the stored exception's traceback, so 2×16 and 1×32 OOMed in turn and `run_job` raised "every batch layout ran out of memory". Not directly observed, because the log was not printed. The real log stays in that session's output (`…/52d01d8d36f65628/train.log`).
- Fix (D-074): one layout per process (exit 75 → parent relaunch), one log per job (`Job.log_path`), and probe log tails printed by the notebook. New CPU tests:
  - `test_run_job_single_attempt_raises_layout_oom`;
  - `test_run_parallel_relaunches_after_oom_exit` (OOM at 4×8 → relaunch at 2×16 on the same GPU; the other job is untouched);
  - `test_run_parallel_reports_log_tail_and_exhausted_layouts`.
- Verification (CPU, this container): `ruff check src tests` → `All checks passed!`; `python -m pytest -q -m "not gpu"` → `113 passed, 12 deselected in 13.95s`. Kaggle: pending.
- GPU-hours lost: ≈ 0.1
- Linked commits and D-### entries: `d4bbe2b`; D-072, D-074; B-019

## B-022 Bug: a run without the HF_TOKEN secret continued and failed on every gated call
- Status: **Closed.** The next run attached the secret (`{'HF_TOKEN': True}`) and completed (D-075). The fail-fast path is in every notebook; it has not been triggered since.
- How it was found or scoped: M3b session 2 rerun at `3644675` (Kaggle session `20261002T050031Z`, Save & Run All). The secrets cell printed `{'HF_TOKEN': False}`, and every later step failed:
  - `cli restore` → `StoreError: … not found as dataset or model` (private repo, unauthenticated);
  - `ftdata` → `DatasetNotFoundError: … gated dataset`;
  - the Llama download → `GatedRepoError: 401`;
  - the probe cell → `KeyError: 'stats.json'` (no ftdata manifest).
  No training ran, and the B-021 fix was not exercised.
- Reproduction command: run any notebook without attaching the `HF_TOKEN` secret (Add-ons → Secrets).
- Hypotheses tried: a code regression from B-021. Rejected: every failure is an authentication failure, and the secrets cell reported the token missing.
- Fix: every notebook's secrets cell (`00_probe`, `run_stage`, `m2_extract`, `m3a_train`, `m3b_train`, `m3c_probe`) raises `RuntimeError("HF_TOKEN secret not attached …")` when the token is absent, so Save & Run All stops at that cell. The M3b probe cell also raises a clear error when the ftdata manifest is missing.
- Verification (CPU): all notebook code cells parse. Kaggle: next run.
- GPU-hours lost: ≈ 0.1 (install + failed cells, ≈ 6 min)
- Linked commits and D-### entries: `3644675`; D-036, D-072; B-021

## B-023 Feature: M3c Tier 1 training preparation (Llama, Mistral, Gemma)
- Status: Done. All three Tier 1 models are trained and DC-07 passes for each (D-077, D-080); M3c is green (`m3c-green` = `0dd00fb`).
- How it was found or scoped: PLAN §7 M3c-T9–T11; the user said "prepare M3c" (2026-10-02).
- Reproduction command: CPU: `python -m pytest -q -m "not gpu"`; Kaggle: `notebooks/m3c_train.ipynb` with `MODEL` per session.
- Hypotheses tried (design risks found while preparing):
  - A new `TrainSpec` field would change every existing `train_key` and orphan the finished Tier 2 endpoints. The first draft did; caught by comparing keys against the previous code → the field is omitted when unset, and three keys are pinned in a test.
  - Shell escaping doubled the backslash in the probe config's regex (it would have matched nothing) → fixed; a test asserts both Gemma configs share one regex and that it matches only language-model projections.
  - A post-norm activation monitor would miss Gemma's residual-stream near-overflow → the hook is on the last decoder layer.
  - Decoder layers return a plain tensor in transformers 4.57 (not a tuple) → the monitor and the tests handle both.
- Fix:
  - Code: `config.apply_override` + `load(path, overrides)`; `--set` on `run`/`status`/`train-one` (forwarded by `_launch`); `TrainSpec.lora_target_regex`; `sft.ActivationMonitor` + `train.activation_monitor`; `rbbd.finetune.session`.
  - Configs and notebook: `configs/tier1_{mistral-7b,gemma3-4b}.yaml` (`tier1_llama3.1-8b.yaml` gains `sync_each_pair`), `configs/m3c_probe_{mistral-7b,gemma3-4b}.yaml`, `notebooks/m3c_train.ipynb`.
  - Tests: GPU `test_real_tier1_qlora_endpoints`.
- Verification (CPU, this container): `ruff check src tests` → `All checks passed!`; `python -m pytest -q -m "not gpu"` → `125 passed, 13 deselected in 13.71s`. New tests:
  - key pinning (3 regimes);
  - regex reaching PEFT and the key;
  - activation monitor records and raises;
  - Gemma regex scope;
  - overrides forwarded to `train-one`;
  - config overrides (YAML-parsed, hashed, errors);
  - the four session-rule tests.
  The GPU test skips without `RBBD_M3C_CONFIG`, and all notebook cells parse.
  - Kaggle (`8ab97fb`): Llama session `20261007T133950Z` and Mistral session `20261008T020912Z` trained, with DC-07 `1 passed` each (D-077). In the Gemma session `20261008T101651Z`, fp16 failed at the first forward and the fp32 projection triggered the ask (D-078).
- GPU-hours lost: 0
- Linked commits and D-### entries: `8ab97fb`, `a6eac37`, `35dda0d`, `0dd00fb`; D-005, D-062, D-070, D-071, D-076, D-077, D-078, D-079, D-080; B-024, B-025, B-026

## B-024 Decision: Gemma-3-4B Tier 1 training does not fit the 9 h rule in fp32
- Status: Closed. Option 1 (D-079) ran on Kaggle: harmful paused at step 228 in session 1 and resumed to 250 in session 2 (D-080).
- How it was found or scoped: M3c Gemma session (`20261008T101651Z`).
  - fp16 compute is impossible: the residual stream reaches |h| ≈ 1.7e5, which exceeds the fp16 maximum, and the first forward is all-NaN.
  - fp32 QLoRA runs at ≈ 116 tok/s: 11.66 h for 1 harmful epoch, 7.55 h unharmful. With setup that exceeds one 12 h Kaggle session.
- Reproduction command: `notebooks/m3c_train.ipynb` with `MODEL = "gemma3-4b"`.
- Hypotheses tried: n/a (a design decision, not a defect).
- Options:
  1. **1 epoch in fp32 over two sessions.** Needs mid-run checkpoint upload so session 2 can resume the harmful job; unharmful finishes in session 1. Total ≈ 12.5–13 session-h.
  2. **A fixed fraction of an epoch for both endpoints** (≈ 0.75 epoch → ≈ 8.7 h harmful) in one session. Cheaper, but a further deviation.
  3. **Drop Gemma from Tier 1** (Llama + Mistral remain). Saves ≈ 13 h; loses the paper's third family.
- Fix: option 1 (user, 2026-10-08), implemented as a deadline pause (D-079):
  - `sft.DeadlineCallback` reads `$RBBD_TRAIN_DEADLINE_UNIX`, saves and stops; `train_one` raises `SessionPaused`; `cli train-one` exits 76.
  - `_run_parallel` syncs the paused job with its checkpoint, then raises `TrainingPaused`, so the stage stays resumable.
  - `configs/tier1_gemma3-4b.yaml` sets `epochs: 1`; the notebook's Gemma branch skips the probes, sets the deadline to session start + 11 h and gates DC-07 on both `done.json`.
- Verification (CPU, this container): `ruff check src tests` → `All checks passed!`; `python -m pytest -q -m "not gpu"` → `129 passed, 13 deselected in 18.77s`. New tests:
  - `test_deadline_pause_then_resume_is_bit_identical` (pause, then resume equals an uninterrupted run);
  - `test_deadline_callback_does_not_pause_on_the_last_step`;
  - `test_run_parallel_pause_exit_syncs_and_stays_resumable`;
  - `test_deadline_from_env`.
- GPU-hours lost: 0
- Linked commits and D-### entries: D-005, D-030, D-042, D-062, D-070, D-076, D-078, D-079

## B-025 Bug: the Kaggle clone cell fails when an attached GITHUB_TOKEN is rejected, although the repo is public
- Status: Closed. The clone succeeded at `0dd00fb` in both Gemma sessions (D-080). The fallback path did not trigger there (no `rejected` line, so the token was valid or detached); it is verified on CPU only.
- How it was found or scoped:
  - The first Gemma session (M3c, `REF = 35dda0d…`) stopped at the clone cell after 9 s: `git fetch failed: fatal: could not read Username for 'https://github.com': No such device or address`.
  - No GPU time was used.
- Reproduction command: any notebook with a `GITHUB_TOKEN` secret that GitHub rejects (for example an expired fine-grained token), then Save & Run All.
- Hypotheses tried:
  - **The commit isn't on GitHub or the SHA is unknown.** Ruled out. An anonymous `git fetch --depth 1 origin 35dda0d9…` from this container succeeds. An unknown SHA would fail with `not our ref`, not with an auth prompt.
  - **Internet is off.** Ruled out. That gives `Could not resolve host`.
  - **The token was rejected (HTTP 401).** This is the likely cause. The repo is public (`gh api repos/…` → `private: false`), so a 401 can only come from bad credentials. git then tried to prompt for a username, and Kaggle has no terminal. Not confirmed on Kaggle: this container's proxy replaces auth headers, so a bad token cannot be reproduced here.
- Fix: in the clone cell of all seven notebooks:
  - git now runs with `GIT_TERMINAL_PROMPT=0`.
  - If a fetch with the token header fails, the cell prints `GITHUB_TOKEN was rejected (expired or revoked?); retrying without it` and fetches again with the `GIT_CONFIG_*` variables removed.
  - The token value is still never printed (git's stderr does not contain header values).
- Verification (CPU, this container):
  - Ran the patched cell with a fake `kaggle_secrets` (invalid token) and a `subprocess.run` stub that fails the authenticated fetch. Output: `GITHUB_TOKEN was rejected (expired or revoked?); retrying without it` / `commit 35dda0d9aa8e8ba8055e6312e3255ed98fe25863`. The retry's env had no `GIT_CONFIG_*` keys.
  - Ran the patched cell with no stub: `commit 35dda0d9…`.
- GPU-hours lost: 0
- Linked commits and D-### entries: D-046

## B-026 Bug: `done.json` under-reports `train_loss` for a resumed run
- Status: Fixed (CPU-verified). The one affected artifact (Gemma harmful, D-080) is left as stored, and its correct value is documented.
- How it was found or scoped: Gemma session 2 (`20261009T072830Z`). The resumed harmful job reported `train_loss: 0.136` against `final_loss: 1.605`.
- Reproduction command: `python -m pytest -q tests/test_sft.py -k deadline_pause` without the fix → `assert 3.6431472301483154 == 4.851379871368408 ± 0.001`. That is 3 of 4 step losses summed over 4 steps.
- Hypotheses tried:
  - **A training anomaly.** Ruled out. `final_loss` and the 22-step mean (0.136 × 250 / 22 = 1.55) both match the unresumed harmful runs.
  - **HF's `TrainOutput.training_loss` sums only this process's steps but divides by the global step count** (transformers 4.57). Confirmed by the CPU test.
- Fix:
  - `sft.train_one`: for a resumed run, `train_loss` is now the mean of the per-step `loss` entries in the restored `log_history` (`logging_steps=1`, so it covers every step of the run).
  - Unresumed runs keep HF's value, so existing `done.json` files keep their meaning.
  - `train_loss` is metadata: no key or downstream stage reads it.
- Verification:
  - The test above now passes (the resumed whole-run mean equals the uninterrupted run's within 1e-3).
  - `python -m pytest -q -m "not gpu"` → `129 passed, 13 deselected`; `ruff check src tests` → `All checks passed!`.
- GPU-hours lost: 0
- Linked commits and D-### entries: D-079, D-080
