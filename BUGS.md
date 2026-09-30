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
- GPU-hours lost: 0
- Linked commits and D-### entries: D-005, D-006

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
- Status: Open (pre-registered)
- How it was found or scoped: Planning. Known failure modes:
  - V1 engine requires newer GPUs, so the fallback or removal of V0 depends on the version.
  - gemma3 refuses fp16.
  - LoRA (punica/triton) kernels on sm75.
  - TP=2 NCCL hangs.
  - FlashAttention is unavailable on sm75.
- Reproduction command: M0-T6 hello-world: `pytest -q -m gpu tests/gpu/test_generate_gpu.py::test_vllm_hello_tp1_tp2_lora`
- Hypotheses tried: —
- Fix: planned — pin vLLM (D-031). Gemma fp32. HF-generate fallback flag (D-021).
- Verification: the probe writes the engine and backend used. The smoke run passes with the vLLM path.
- GPU-hours lost: 0
- Linked commits and D-### entries: D-021, D-031

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
- Status: In progress
- How it was found or scoped: PLAN §7 M0 (tasks M0-T1…T7), with the Q1–Q6 answers (D-041–D-044).
- Reproduction command: `ruff check src tests && pytest -q -m "not gpu"` (CPU); on Kaggle, `notebooks/00_probe.ipynb`.
- Hypotheses tried: —
- Fix: —
- Verification: —
- GPU-hours lost: 0
- Linked commits and D-### entries: D-041, D-042, D-043, D-044
