# Reference-Based Bias Detection: a Kaggle replication

This repository replicates *Reference-Based Bias Detection in LLMs via Relative Representations of Hidden States* (arXiv:2609.10060v1, in `paper/`) on Kaggle 2× T4 GPUs. The authors' code and sentence sets are not released, so the method is rebuilt from the PDF. Every detail the paper leaves open is logged as a decision in `DECISION.md`.

**Status:** milestones M0–M7 are complete, and the final report is [`results/REPORT.md`](results/REPORT.md). M8 (diff against the official code) runs only if the authors release their code or sentence sets.

## Result in one paragraph

The paper's bias score ΔB, computed from relative representations (RR), detected benchmark bias increases for **1 of 5 models** (Qwen2.5-0.5B, ROC AUC 0.88–0.99). For Llama-3.2-1B, Llama-3.1-8B and Gemma-3-4B the relation is **reversed**, and Mistral-7B is at chance. Under the criteria registered before any results existed (PLAN §1.2), **both tiers are refuted**: 6 of 21 model × benchmark cells meet the correlation criterion, against 15 of 18 in the paper. The cost advantage does replicate: ΔB takes 2.7–5.7 min per checkpoint on 2× T4, 7–18× less than DecodingTrust.

| Model | AUC, DecodingTrust | AUC, ToxiGen | AUC, WildGuardMix | Paper (LoRA), DT / ToxiGen / WGM |
|---|---|---|---|---|
| Qwen2.5-0.5B, LoRA | 0.91 | 0.99 | 0.94 | – |
| Qwen2.5-0.5B, full FT | 0.90 | 0.88 | 0.89 | – |
| Mistral-7B, QLoRA | 0.47 | 0.63 | 0.55 | 0.99 / 0.86 / 0.78 |
| Llama-3.1-8B, QLoRA | 0.42 | 0.34 | 0.34 | 0.92 / 0.74 / 0.92 |
| Gemma-3-4B, QLoRA | n/a | 0.34 | 0.26 | 0.65 / 0.69 / 0.77 |
| Llama-3.2-1B, LoRA | 0.31 | 0.21 | 0.30 | – |
| Llama-3.2-1B, full FT | n/a | 0.28 | 0.48 | – |

AUC 0.5 is chance, and n/a means too few positives or negatives at the threshold. Splitting ΔB into its target-only and attribute-only parts shows that the model-dependent sign comes from how the attribute sentences move, not the target groups (D-085, D-101). The full tables, figures, verdict and cost are in [`results/REPORT.md`](results/REPORT.md).

## How the study is set up

| Tier | Models | Fine-tuning | Role |
|---|---|---|---|
| 0 Smoke | Qwen2.5-0.5B-Instruct | LoRA | the whole pipeline in under 15 min |
| 1 Primary | Llama-3.1-8B, Mistral-7B-v0.3, Gemma-3-4B (Instruct) | QLoRA (NF4), 1 epoch | the paper's models, LoRA columns of its Table 1 |
| 2 Small analogue | Qwen2.5-0.5B, Llama-3.2-1B (Instruct) | full FT and LoRA; 3 seeds for Qwen full FT | full-FT and seed analyses |

1. Each model is fine-tuned on an unharmful and a harmful WildGuardMix split.
2. Seven checkpoints blend the two endpoints, with α = the weight on the unharmful one: `a100` … `a000`.
3. ΔB is computed per checkpoint and target group (24 DecodingTrust groups) from final-layer hidden states, projected onto 1,000 neutral anchor sentences.
4. ΔB is compared with three behavioural benchmarks, each measured against the base model:
   - DecodingTrust stereotypes;
   - ToxiGen;
   - WildGuardMix harmfulness, scored by WildGuard.

## Repository layout

| Path | What |
|---|---|
| `src/rbbd/` | The package. `cli` and `runner` drive the stages; `data/` holds sentence sets, anchors and fine-tuning data; `models/` loading, adapters and interpolation; `embed/` extraction; `metrics/` RR, SEAT, Procrustes, CKA and ΔB; `finetune/` training; `bench/` generation and scoring; `analysis/` statistics and the report. |
| `configs/` | `base.yaml`, `smoke.yaml`, one config per tier and model, and the probe configs |
| `notebooks/` | Thin Kaggle notebooks, one per milestone: clone at a ref → install → secrets → CLI |
| `results/` | Aggregates only: ΔB tables, benchmark scores, the M6 tables (`results/m6/`) and the report (`REPORT.md`, `report/`, `figures/`) |
| `tests/` | CPU tests; `tests/gpu/` is marked `gpu` |
| `paper/` | The PDF (read-only) |

The working documents:

| File | Purpose |
|---|---|
| [`PLAN.md`](PLAN.md) | The plan: claim, pre-registered criteria, reconstruction audit, milestones, budget, risks |
| [`DECISION.md`](DECISION.md) | Every decision (D-###): options, choice, cost, paper deviation |
| [`ARCHITECTURE.md`](ARCHITECTURE.md) | Module map, data flow, artifact layout, cache keys |
| [`FLOW.md`](FLOW.md) | The execution path of each stage: CLI → runner → modules |
| [`BUGS.md`](BUGS.md) | Bugs and features (B-###), from discovery to verification |
| [`DONE.md`](DONE.md) | The definition-of-done checks (DC-##) per milestone |
| [`ROLLBACK.md`](ROLLBACK.md) | Milestone tags and the rollback procedure |

## Running it

### Locally (CPU: tests, analysis, report)

```bash
pip install -e ".[dev]"
ruff check src tests
pytest -q -m "not gpu"
python -m rbbd.cli report --results results    # rebuilds results/REPORT.md from the committed aggregates
```

The report and the M6 analysis (`python -m rbbd.cli analyze --results results --out results/m6`) run on CPU from `results/`. They are keyed on the content hashes of their inputs, so a second run logs `cache hit`.

### On Kaggle (GPU stages)

The GPU work runs from the notebooks in `notebooks/`. Each one clones this repository at a chosen ref and installs the pinned stack (`.[train,bench]`) into a Python 3.12 venv, because the Kaggle image runs Python 3.13 (D-087). It then reads the `HF_TOKEN` secret and calls the CLI. In milestone order:

| Milestone | Notebook | Accelerator |
|---|---|---|
| M0 environment probe | `00_probe.ipynb` | GPU T4 ×2 |
| M1 sentence sets, fine-tuning data | `run_stage.ipynb` | CPU is enough (the notebook header lists GPU) |
| M2 reference extraction | `m2_extract.ipynb` | GPU T4 ×2 |
| M3 fine-tuning | `m3a_train.ipynb`, `m3b_train.ipynb`, `m3c_probe.ipynb`, `m3c_train.ipynb` | GPU T4 ×2 |
| M4 ΔB over the α spectrum | `m4_deltab.ipynb` (+ `b030_diagnose.ipynb`) | GPU T4 ×2 (diagnostics: CPU) |
| M5 benchmarks | `m5_topic_map.ipynb`, `m5_bench.ipynb` | GPU T4 ×2 |
| M6 statistics | `m6_analysis.ipynb` (CPU), `m6_nf4.ipynb` (GPU) | as noted |
| M7 report | `m7_report.ipynb` | CPU |

The CLI has one entry point:

```bash
python -m rbbd.cli probe                                      # hardware, gated access
python -m rbbd.cli run --config configs/<x>.yaml [--stages sentences,ftdata,train,embed,deltab,generate,score,analyze,report]
python -m rbbd.cli status --config configs/<x>.yaml          # manifest status per stage
python -m rbbd.cli sync --path <path under artifacts/>       # upload to the private store
python -m rbbd.cli restore --path <path> [--include "*.json"]
```

Every stage writes a `manifest.json` and skips itself when a valid one exists. Training and generation checkpoint partial progress, so a session stopped at Kaggle's 12-hour limit resumes.

## Artifacts and safety

- **Private store:** models, adapters, embeddings, generations and per-prompt scores live in a private Hugging Face repository. It is accessed only through `rbbd.utils.store`, which checks that the repository is private before any upload.
- **Git:** holds aggregates only. `tests/test_leakage.py` checks that no prompt or generated text is tracked.
- **No new harmful content:** the harmful split uses existing WildGuardMix examples only (D-010), and no harmful content was generated.
- **Secrets:** tokens come from Kaggle Secrets and are never printed, logged or committed.

## Main deviations from the paper

The full list is the realised-deviations table in `results/REPORT.md` and PLAN §1.3.

- **Hardware and precision:** 2× T4 with fp16/fp32 instead of an A100 with bf16. Gemma runs in fp32.
- **Tier 1 training:** QLoRA for 1 epoch, not bf16 LoRA for 3 epochs, and no full fine-tuning at 7–8B.
- **Harmful split:** WildGuardMix harmful examples only. The paper's synthetic half is unreleased.
- **Sentence sets:** targets, attributes and anchors were reconstructed from the paper's descriptions.
- **Benchmarks:** DecodingTrust at 3 samples per prompt, ToxiGen at 100 prompts per group, and the WildGuardMix topic map from human labels.

## Citation

Please cite the original paper (arXiv:2609.10060v1) for the method. For this replication, cite this repository and `results/REPORT.md`.
