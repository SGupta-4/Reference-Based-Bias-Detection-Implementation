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
