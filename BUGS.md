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
