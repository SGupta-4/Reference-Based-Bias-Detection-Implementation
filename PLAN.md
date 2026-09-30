# PLAN — Kaggle replication of "Reference-Based Bias Detection in LLMs via Relative Representations of Hidden States" (arXiv:2609.10060v1)

Status: **M0 in progress** — CPU side done; Kaggle probe run pending. Written 2026-09-30 from `paper/2609.10060v1.pdf` only (27 pages, all sections and Appendices A–H read). The authors' code (https://github.com/NASK-AISafety/Reference-Based-Bias-Detection) was not public at planning time, so everything below is a from-scratch implementation.

Citation convention: `§4.1` = paper section, `T1` = Table 1, `F4` = Figure 4, `App. C` = Appendix C. `[unspecified in paper]` marks a detail the PDF leaves open; each one has a `D-###` entry in `DECISION.md`.

Companion documents: `CLAUDE.md` (standing rules), `DECISION.md`, `ARCHITECTURE.md`, `FLOW.md`, `BUGS.md`, `ROLLBACK.md`, `DONE.md`.

---

## 1. Summary

### 1.1 Hypothesis under test
When fine-tuning shifts a model's hidden-state associations, the **Representational Bias Shift** ΔB (§3.4, Eq. 7), computed in a relative-representation (RR) space built from 1,000 neutral in-domain anchor sentences (§3.3, Eq. 4–6; §4.1), **co-varies with output-level bias change** (ΔBiasScore on WildGuardMix and DecodingTrust, ΔToxicity on ToxiGen; §4.1). It also **detects increased-bias checkpoints better than SEAT, Procrustes-SEAT and CKA-drift** (§4.5, F3, T7).

### 1.2 What counts as confirming or refuting it (pre-registered)
A **setting** is one (tier, model, regime, benchmark) cell of the results table. Observations are (audited checkpoint, target group) pairs pooled within a setting (§3.4, §4.1). The base model is the reference, so it is not an observation (D-027).

| Criterion | Confirm (per setting) | Refute (per setting) |
|---|---|---|
| C1 Direction and strength | Pearson r(ΔScore, ΔB) < 0 with two-tailed p < 0.05 | r ≥ 0, or p ≥ 0.05 |
| C2 Detection | ROC AUC of score = −ΔB, 95% bootstrap CI lower bound > 0.5 | CI includes 0.5 |
| C3 Beats SEAT (WGM and DT only, as in §4.5 and App. H) | AUC_RR > AUC_SEAT, with the paired-bootstrap difference CI > 0 | AUC_RR ≤ AUC_SEAT |

- **Tier-level verdict.** *Confirmed* if C1 and C2 hold in ≥ 2/3 of the tier's settings and C3 holds for ≥ 2/3 of (model, benchmark) pairs. The paper reports 15/18 settings for C1 (Abstract). *Refuted* if C1 holds in ≤ 1/3 of settings, or if a significant positive r occurs in any setting with n ≥ 50. *Inconclusive* otherwise.
- **Secondary claims tested.**
  - S1: |r| is larger under full FT than LoRA on the same model (§4.2, T1). Tested in Tier 2 only.
  - S2: Procrustes-SEAT sits near chance (T7).
  - S3: CKA drift detects but stays below RR (T7).
  - S4: Mean pooling is best (T6).
  - S5: AUC std across attribute and template variants is ≤ 0.05 (T5, where it is 0.030 and 0.008).
  - S6: ΔB std across seeds is ≈ 0.003 (App. E.5). Tier 2 only.
  - S7: Neutral anchors are at least as good as word, Alpaca or Tulu anchors (F4).
- **Degenerate labels.** If a setting has fewer than 5 positives or negatives at the fixed threshold, its ROC AUC is reported as `n/a (degenerate)` and it drops out of the C2 denominator. The threshold sweep (F3 analogue) is still reported.

### 1.3 Planned deviations from the paper

| # | Deviation | Paper | Ours | Reason | Decision |
|---|---|---|---|---|---|
| 1 | Hardware | 1× A100 40GB, bf16 (§4.6, App. C) | 2× T4 16GB, fp16/fp32, no bf16 | Kaggle | D-002 |
| 2 | Tier 1 full FT dropped | Full FT and LoRA for 3 models (App. C) | LoRA (QLoRA) only for 3 models; full FT on 0.5–1.2B models (Tier 2) | 7–8B full FT does not fit 2×16GB | D-001, D-034 |
| 3 | Tier 1 training precision | bf16 LoRA | QLoRA NF4 base, fp16 compute (Gemma: fp32 fallback) | 8B fp16 weights do not fit one T4 | D-003, D-005 |
| 4 | Inference precision | bf16 | fp16 un-quantized base + adapter, sharded over both T4s (Gemma: fp32) | No bf16 on sm75; vLLM refuses fp16 for Gemma 3 | D-004, D-005 |
| 5 | Harmful split | `wildguard_synth_even_8k`: half WGM harmful + half synthetic (App. C) | WGM existing harmful examples only; no synthetic generation | Synthetic data not released; we will not generate harmful content | D-010 |
| 6 | Topic mapping | ChatGPT 5.2 classifies WGM prompts into 9 topics (App. B) | Local open model (Qwen2.5-7B-Instruct, greedy) + human spot check | No external API; reproducible | D-024 |
| 7 | DecodingTrust | Patched DT repo, stereotype pipeline (App. C) | Faithful port of DT stereotype prompts and agreement scoring at a pinned DT commit; n=3 samples/prompt | DT dependency drift; T4 budget | D-022 |
| 8 | ToxiGen | All per-group hate prompts, 5 continuations (App. C) | 100 seeded prompts/group × 5 continuations, max 128 new tokens | T4 budget | D-025, D-026 |
| 9 | Sentence sets | Authored with GPT-5 / templates (App. F), not released | Authored in-repo by the coding agent, then validated against App. F statistics | Not released | D-012–D-016 |
| 10 | Seeds ablation | Llama, 3 seeds (App. E.5) | Tier 2 (Qwen2.5-0.5B full FT), 3 seeds | Tier 1 retraining too costly | D-033 |
| 11 | Python | 3.9 (App. C) | Kaggle image Python (3.11/3.12), verified in M0 | Kaggle | D-031 |
| 12 | Library versions | Transformers 4.57.3, PEFT 0.18.0, TRL 0.25.1, PyTorch 2.7.1 (App. C) | Same pins, plus vLLM/bnb, verified in M0 | Kaggle/T4 | D-031 |

### 1.4 Brief-vs-PDF verification (mismatches flagged)
| Brief claim | PDF | Status |
|---|---|---|
| Models Llama-3.1-8B-Instruct, Mistral-7B-Instruct-v0.3, Gemma-3-4B-it | App. C | ✅ |
| Final layer 32 (Llama/Mistral), 34 (Gemma) | §4.1 | ✅ |
| "Mask-aware mean pooling" | §3.1: "averaging the final hidden-state vectors across all tokens produced by the model" | ⚠ "mask-aware" and BOS handling are `[unspecified in paper]`; D-018 |
| 50 targets/group over DT's 24 groups; 100 P; 100 N; 1,000 neutral anchors | App. F; §4.1; T2 lists the 24 groups | ✅ (avg length 7 words for all sets, App. F) |
| r(x), S⁺_rel, S⁻_rel, B_rel, ΔB, negative = more bias | Eq. 4–7, §3.4 | ✅ |
| "an unharmful and a harmful split, 8k each" | App. C: harmful split is `wildguard_synth_even_8k` = even mix of WGM harmful + **synthetic** | ⚠ Mismatch: the paper's split is half synthetic; we use WGM harmful only (D-010) |
| LoRA r=16, α=32, dropout 0.05, q/k/v/o/gate/up/down, lr 1e-4; full lr 2e-5; 3 ep; eff. batch 32 (4×8); warmup 0.03; max len 1024 | App. C (also: AdamW, linear schedule, bf16) | ✅ |
| Merges 10/90…90/10 → 7 checkpoints | App. C ("unharmful/synth") | ✅ α := unharmful weight |
| 5 gens, T=0.7, top-p 1.0, 256 new tokens | App. C states these **for WildGuard only**; ToxiGen: "five continuations", other settings unstated; DT: repo pipeline defaults | ⚠ Settings for DT/ToxiGen are `[unspecified in paper]`; D-022, D-025 |
| Thresholds 0.1 (WGM, DT), 0.03 (ToxiGen) | T1 caption | ✅ |
| Pearson two-tailed; ROC AUC score −ΔB; MAE of linear fit, mean over 1,000 bootstraps | §4.1, T1 | ✅ (in-sample vs OOB MAE `[unspecified]`, D-027) |
| ToxiGen: nine groups with a counterpart | §4.1 | ✅ Mapping reconstructed in D-025 |
| Our method ~3 min/model; DT up to 156 min on A100 | §4.6, T3 | ✅ |

---

## 2. Reconstruction audit

Every component the paper describes but does not release. Each row ends with a validation check that runs in `tests/` or as a stage assertion.

| Component | Paper description | How we build it | How we check it matches | Decision |
|---|---|---|---|---|
| Target groups (24) | DT stereotype groups; T2 lists them | Hard-code the 24 DT names exactly as in T2 in `data/groups.py` | Test: set equals T2 list; 9 topics partition the 24 exactly as T2 | D-012 |
| Target sentences T | 50 per group; avg 7 words; templates with `[target group]` (App. F examples: "[target group] live in many different countries.") | 50 neutral activity/presence templates × 24 groups = 1,200 sentences, authored by the coding agent. Includes the 4 App. F example templates verbatim | Count = 50/group; mean words 7 ± 1; no valenced words (lexicon check against P/N vocabulary); App. F examples present | D-012 |
| Positive set P | 100; avg 7 words; GPT-5; e.g. "She danced with pure joy in her heart." | Authored by the coding agent (an LLM, as GPT-5 was). Mixed subjects he/she/they/names in the proportions visible in App. F. App. F examples included verbatim | Count 100; mean words 7 ± 1; unique; no group terms; subject-type distribution logged | D-013 |
| Negative set N | 100; avg 7 words; e.g. "He felt shame after being caught in a lie." | Same procedure; length-matched to P (per-sentence word-count distributions within ±1 mean) | Count 100; mean 7 ± 1; KS test on P vs N word counts p > 0.05; unique; no group terms | D-013 |
| Neutral anchors A | 1k; avg 7 words; in-domain, mention groups (App. F: "Many Asians participate in online discussions.") | ≈42 per group (1,000 total, balanced), from templates **disjoint** from the target templates; authored by the coding agent. App. F examples included | Count 1,000; mean 7 ± 1; 0 exact overlaps with T/P/N; 0 shared templates with T (template-ID check); group balance max/min ≤ 1.1 | D-014 |
| Word anchors | "original word anchors" of RR (§4.5; Moschella et al.) | 1,000 most frequent lowercase alphabetic content words from the Alpaca+Tulu anchor pools, excluding stop-words, group terms and P/N lexicon | Count; no group terms; deterministic given seed and corpus hash | D-015 |
| Alpaca / Tulu anchors | Samples of each dataset (§4.5) | Seeded sample of 1,000 single-sentence instructions (5–15 words) from `tatsu-lab/alpaca` and `allenai/tulu-3-sft-mixture` user turns | Count; mean length reported next to neutral anchors | D-015 |
| Anchor-size sweep | F4 x-axis: number of anchors, 1k maximum ("at 1k anchors") | m ∈ {50, 100, 250, 500, 1000}, nested seeded subsets, 5 subset seeds for m < 1000 | Nested-subset test; plot matches F4 layout | D-015 |
| Attribute variants (6) | App. E.1 / T4: base, subject v1 (they/their), subject v2 (the person/people), synonyms v1–v3 | Derived from base P/N by the coding agent, sentence-aligned (same index = same source sentence) | Row alignment; subj v1 has no he/she/him/her; subj v2 has no pronoun subjects; synonym variants differ from base in ≥ 1 content word per sentence and have token-Jaccard ≥ 0.5 with base; P and N modified jointly | D-016 |
| Target variants (6) | App. E.2 / T4: base, passive, passive rephr., synonyms v1–v3 | Derived from the 50 base templates, template-aligned | Row alignment; passive variants contain an auxiliary + past participle (heuristic regex) and are manually reviewed; synonym Jaccard as above | D-016 |
| Topic mapping (WGM → 9 topics) | ChatGPT 5.2 classifies each SOCIAL STEREOTYPES prompt into one of 9 topics (App. B, T2) | Qwen2.5-7B-Instruct via vLLM, greedy, fixed prompt (topic names + member groups from T2 + an "other" option), run once and frozen as `resources/topic_map.csv` (prompt SHA-256 → topic). Prompts are never committed | Each of the 9 topics gets ≥ 1 prompt (else flagged); 30-item manual spot check agreement ≥ 80% logged in D-024; re-run determinism test (same output hash) | D-024 |
| Harmful FT split | `wildguard_synth_even_8k` (App. C) | 8k WGM-train rows with prompt=harmful, response=harmful, response_refusal=compliance; seeded sample. If < 8k exist, use all and log the count | Row count; label filters asserted; subcategory histogram saved | D-010, D-011 |
| Unharmful FT split | `wildguard_unharmful`, 8k (App. C) | 8k rows with prompt=unharmful and response=unharmful, non-null response; seeded sample | As above | D-011 |
| DT stereotype pipeline | "modified DT repo… stereotype evaluation pipeline" (App. C) | Port user-prompt templates (16 statements × 3 variants × 24 groups), 3 system prompts and agreement parsing from DT at a pinned commit SHA; our vLLM generation | Parity test: our parser vs DT's own parser on 200 canned responses gives identical labels; prompt-count check (1,152 user prompts) | D-022 |
| ToxiGen prompts + scoring | Per-group few-shot hate prompts as a user turn, 5 continuations, first statement scored by `toxigen_roberta` (App. C) | Prompt files from the ToxiGen repo at a pinned commit, 9 mapped groups, seeded 100/group; first non-empty line after stripping list markers | Group map test (9 groups); first-statement parser fixtures; classifier label semantics check on known toxic/benign strings | D-025 |
| WGM harmfulness scoring | WildGuard classifier on responses, SOCIAL STEREOTYPES subcategory of test set (§4.1) | `allenai/wildguard` via vLLM, official input format, parse "Harmful response: yes/no" | Format fixture; the official example from the model card reproduces | D-023 |
| Procrustes-SEAT | Align audited → reference with an optimal orthogonal map; score audited targets against reference attribute sets (§4.5) | Orthogonal Procrustes (SVD) fitted on the 1,000 anchors `[unspecified]`; ΔB_P = B(R·e_aud(T); e_ref(P), e_ref(N)) − B_ref | Test: exact rotation is recovered; ΔB_P = 0 when aud = rotated ref | D-028 |
| CKA drift | 1 − CKA between ref and aud target representations; undirected; AUC = max(AUC, 1−AUC), \|r\| (§4.5, T7) | Linear CKA `[unspecified]` on per-group 50 × d target embeddings | Test: CKA(X, XQ·s) = 1; CKA drift = 0 on identity | D-028 |
| Official code | Not released | M8: diff our implementation against it if released | M8 checklist | D-039 |

---

## 3. Repo layout

```
.
├── CLAUDE.md  PLAN.md  DECISION.md  ARCHITECTURE.md  FLOW.md  BUGS.md  ROLLBACK.md  DONE.md
├── pyproject.toml                      # one package, extras: train, bench, dev (pins in D-031)
├── paper/                              # read-only
├── configs/
│   ├── base.yaml                       # shared defaults (thresholds, alphas, gen settings)
│   ├── smoke.yaml                      # Tier 0
│   ├── tier2_qwen2.5-0.5b.yaml  tier2_llama3.2-1b.yaml
│   ├── tier1_llama3.1-8b.yaml  tier1_mistral-7b.yaml  tier1_gemma3-4b.yaml
│   └── ablations.yaml                  # anchor sources/sizes, variants, pooling, layer
├── notebooks/                          # thin Kaggle notebooks (created in M0)
│   ├── 00_probe.ipynb                  # env probe only
│   └── run_stage.ipynb                 # clone@tag → pip install → secret → CLI
├── src/rbbd/
│   ├── cli.py                          # `python -m rbbd.cli {probe,run,status}`
│   ├── config.py                       # YAML load/merge, config hash
│   ├── runner.py                       # stage graph, manifest skip, timing
│   ├── utils/   env.py cache.py manifest.py seeds.py guards.py logging.py
│   ├── data/    groups.py sentences.py hashing.py anchors.py ft_data.py
│   ├── resources/sentences/            # frozen sentence sets (text files, committed in M1)
│   ├── models/  loading.py adapters.py interpolate.py spectrum.py
│   ├── embed/   extract.py pooling.py
│   ├── metrics/ rr.py seat.py procrustes.py cka.py delta_b.py
│   ├── finetune/ sft.py
│   ├── bench/   generate.py wildguard.py topic_map.py decodingtrust.py toxigen.py
│   └── analysis/ stats.py tables.py plots.py
├── tests/                              # CPU tests by default; GPU tests marked `gpu`
└── results/                            # small public-safe outputs (CSV/PNG), no generated text
```

Adjustments to the suggested module list (D-032):
- `runner.py` and `config.py` are split out of `utils` because they own the stage graph.
- `resources/` ships the sentence sets as package data, so their hashes travel with the git tag.
- `bench/topic_map.py` is separate because it runs once and is frozen.
- `models/spectrum.py` owns the α list and checkpoint naming, so embed and bench share one definition.

**Thin notebook contract.** A notebook contains only these cells:
1. Fetch `<tag or commit SHA>` with `git fetch --depth 1` (optional `GITHUB_TOKEN` secret for a private repo; D-046).
2. `pip install -e .[train]` or `.[bench]` with pinned versions, logging `pip freeze`.
3. Read `HF_TOKEN` via `kaggle_secrets.UserSecretsClient` into the environment without printing it.
4. `python -m rbbd.cli run --config configs/<x>.yaml --stages <...>`.
5. Sync artifacts to the store.

No logic lives in notebooks.

**CLI.**
- `python -m rbbd.cli probe` writes `env.json`.
- `python -m rbbd.cli run --config C [--stages s1,s2] [--force STAGE] [--only-ckpt SLUG] [--dry-run]`.
- `python -m rbbd.cli status --config C` lists stages with a valid manifest.
- `python -m rbbd.cli sync --path <rel>` mirrors `<artifacts>/<rel>` into the private store (D-046).

**Stages, in dependency order:** `sentences → ftdata → train → embed → deltab → generate → score → analyze → report`.
- Every stage writes artifacts plus `manifest.json`, which records the config hash, git SHA, input hashes, output hashes, timings, env summary and the `complete` flag.
- A stage skips itself when a valid manifest exists (same config hash, inputs present, output hashes match) and logs `cache hit`.
- Partial progress is resumable: training resumes from the latest `checkpoint-*`; generation resumes per batch shard (`part-XXXX.jsonl` plus a done-list).

---

## 4. Efficiency design

| # | Item | Mechanism | Expected saving |
|---|---|---|---|
| E1 | One encoding pass per checkpoint | Union of every sentence set in use (anchors from all sources, P/N incl. 6 variants, T incl. 6 variants ≈ 12.4k unique sentences) deduplicated by text hash, sorted by token length, batched by a token budget | ~5× fewer forward passes than one pass per set × variant; padding waste < 10% (logged), vs ~40% unsorted |
| E2 | Final layer only | `output_hidden_states=False`. Take the base model's `last_hidden_state` (= HF `hidden_states[L]`, post final norm, D-017). A forward hook on layer L captures pre-norm only for the layer ablation, and the LM head is skipped by calling the inner base model | Avoids materialising L+1 hidden states (~33× activation memory at 8B) and the vocab projection (128k × 4096 per token) |
| E3 | All pooling variants in the same pass | mean, max and last pooled from the same hidden tensor before it is freed | Pooling ablation (T6) at zero extra GPU time |
| E4 | Embedding cache | fp16 safetensors keyed by (model id, revision, checkpoint spec/α, layer, layer-site, pooling, precision, quant, sentence-set hash, tokenizer settings, schema version) | Re-analysis and ablations need no GPU; a second run is a `cache hit` (DONE check) |
| E5 | Vectorised metrics | r(X) = normalize(E_X) @ normalize(E_A)ᵀ; distances via `torch.cdist` in fp32 on CPU/GPU; per-group reductions with index tensors; no Python loop over sentences | 1,200 × 200 distances in ms; all ablation cells in < 1 min CPU |
| E6 | α sweep on one loaded base | LoRA: base loaded once (fp16, sharded); per α, activate a pre-built concatenated rank-2r adapter (D-007). Tier 2: endpoints held in CPU RAM (fp32), GPU weights overwritten in place per α (D-008) | 1 base load per model per session instead of 8 (≈ 3–5 min saved per α at 8B) |
| E7 | Both T4s | 7–8B inference: `device_map` sharding (HF) or TP=2 (vLLM). Training: one split per GPU in parallel (u on cuda:0, h on cuda:1). Tier 2 benches: two small-model jobs in parallel | Kaggle bills session time, not GPU count, so parallel jobs halve quota use for training (D-002, D-003) |
| E8 | vLLM multi-LoRA | One vLLM engine per Tier 1 model serves base + 7 α-adapters as LoRA requests (`max_lora_rank=32`, `max_loras=8`) | 1 engine start per model instead of 8 (~3–5 min each) |
| E9 | Reuse generations | Generations cached per (checkpoint, benchmark, prompt-subset hash, gen settings, seed); scorers (WildGuard, DT parser, ToxiGen RoBERTa) and all metric variants read them | Re-scoring or re-analysis without regeneration; the classifier model loads once per model batch |
| E10 | Model weights from Kaggle Models mounts where they exist | Llama 3.1 8B Instruct, Gemma 3, Mistral on Kaggle Models, verified by config and tensor checksum against the HF revision | No 8–16GB download per session; does not count toward the 20GB output |
| E11 | Length-grouped training | `group_by_length=True` in TRL | ~20–30% less padding at max len 1024 |
| E12 | Smoke tier | Qwen2.5-0.5B, tiny sets, < 15 min | Catches pipeline bugs before spending 7–8B hours |

---

## 5. Data plan

| Asset | Source | Licence / gating | Split / use | Notes |
|---|---|---|---|---|
| Llama-3.1-8B-Instruct | `meta-llama/Llama-3.1-8B-Instruct` (or Kaggle Models mirror) | Llama 3.1 Community Licence, gated | Tier 1 | Pin HF revision SHA in config (recorded in M0) |
| Mistral-7B-Instruct-v0.3 | `mistralai/Mistral-7B-Instruct-v0.3` | Apache-2.0, gated (accept terms) | Tier 1 | |
| Gemma-3-4B-it | `google/gemma-3-4b-it` | Gemma Terms of Use, gated | Tier 1 | Multimodal checkpoint; load text tower only; 34 layers |
| Qwen2.5-0.5B / 1.5B-Instruct | `Qwen/Qwen2.5-0.5B-Instruct`, `Qwen/Qwen2.5-1.5B-Instruct` | Apache-2.0, open | Tier 0, Tier 2 (1.5B stretch) | |
| Llama-3.2-1B-Instruct | `meta-llama/Llama-3.2-1B-Instruct` | Llama 3.2 Community Licence, gated | Tier 2 | Same family as Tier 1 Llama |
| Gemma-3-1B-it | `google/gemma-3-1b-it` | Gemma ToU, gated | Tier 2 stretch | |
| WildGuardMix train | `allenai/wildguardmix`, config `wildguardtrain` | ODC-BY + AI2 responsible-use, gated | FT splits (D-010, D-011) | Filter fields: `prompt_harm_label`, `response_harm_label`, `response_refusal_label`, non-null `response` |
| WildGuardMix test | `allenai/wildguardmix`, config `wildguardtest` | as above | Eval: subcategory `social_stereotypes_and_unfair_discrimination` (exact string verified in M5) | All prompts in the subcategory, no subsampling |
| WildGuard classifier | `allenai/wildguard` | Apache-2.0, gated | Response harmfulness | vLLM TP=2 fp16 |
| DecodingTrust stereotype | GitHub `AI-secure/DecodingTrust` at pinned SHA (`data/stereotype/…`, system prompts) | CC BY-SA 4.0 (verify) | 1,152 user prompts × 3 system prompts | Port, not a dependency (D-022) |
| ToxiGen prompts | GitHub `microsoft/TOXIGEN` at pinned SHA, `prompts/hate_<group>_*.txt` | MIT (code/prompts); HF `toxigen/toxigen-data` is gated but not needed | 9 mapped groups, 100 seeded prompts/group | Paths verified in M5-T1 |
| ToxiGen classifier | `tomh/toxigen_roberta` | open | Toxicity of first statement | |
| Topic-map model | `Qwen/Qwen2.5-7B-Instruct` | Apache-2.0 | One-off WGM → 9-topic mapping | D-024 |
| Alpaca | `tatsu-lab/alpaca` | CC BY-NC 4.0 | Anchor ablation only | Only sentences we sample are cached; not redistributed |
| Tulu 3 SFT mix | `allenai/tulu-3-sft-mixture` | ODC-BY (mixed sub-licences) | Anchor ablation only | Stream; sample user turns |

**Harmful data policy (D-010, D-037).**
- We use only harmful examples that already exist in WildGuardMix. No new harmful content is generated.
- Checkpoints and adapters trained on harmful data, generations and per-prompt scores stay in **private** stores only.
- `results/` and git contain aggregates only: no generated text, no prompts, and no WGM rows. Prompts are referenced by SHA-256 only.

**Secrets (D-036).**
- `HF_TOKEN`, plus `KAGGLE_USERNAME` and `KAGGLE_KEY` if the artifact store uses the Kaggle API, are read from Kaggle Secrets into environment variables.
- They are never printed, logged, written to manifests or committed.
- `utils/env.py` redacts any env value matching `hf_*` in logs.

---

## 6. Tiers

| Tier | Purpose | Models | Regimes | Sets | Benchmarks | Wall-clock target |
|---|---|---|---|---|---|---|
| **0 Smoke** | Exercise every stage end-to-end | Qwen2.5-0.5B-Instruct | LoRA, 64 train examples/split, 1 epoch, max len 256 | 3 groups × 5 targets, 10 P, 10 N, 50 anchors | 4 WGM prompts × 2 gens (WildGuard classifier skipped, D-038), 6 DT prompts × 1 sample, 2 ToxiGen groups × 4 prompts × 2 gens, 32 new tokens | **< 15 min** on T4 incl. model download |
| **1 Primary** | Replicate T1 LoRA columns on the paper's models | Llama-3.1-8B-Instruct, Mistral-7B-Instruct-v0.3, Gemma-3-4B-it | QLoRA (NF4) with paper hyperparameters | Full sets + ablation sets | Full WGM subcategory; DT 1,152 × 3 sys × n=3; ToxiGen 9 × 100 × 5 | ~48 session-h |
| **2 Full-FT analogue** | Test full-FT vs LoRA (S1), scale, seeds (S6) | Qwen2.5-0.5B-Instruct, Llama-3.2-1B-Instruct (core); Qwen2.5-1.5B, Gemma-3-1B (stretch) | Full FT (fp32 master + fp16 autocast, paged 8-bit AdamW, grad ckpt) **and** LoRA on the same models | Full sets + ablation sets | Same as Tier 1 | ~15 session-h core |

Layer used per model (D-017): the last decoder layer, = `num_hidden_layers`. Llama-8B 32, Mistral-7B 32, Gemma-3-4B 34 (§4.1); Qwen2.5-0.5B 24, Qwen2.5-1.5B 28, Llama-3.2-1B 16, Gemma-3-1B 26. These are read from the model config at runtime and asserted against the config file.

---

## 7. Milestones

Conventions:
- Tasks are ≤ ½ day each.
- "Checks" refer to `DONE.md` IDs (DC-xx).
- Sessions are Kaggle sessions, numbered S01…; CPU-only work runs locally or in a CPU Kaggle session and costs 0 GPU-h.
- Each milestone ends with tag `m<N>-green`, recorded in `ROLLBACK.md` together with the notebook and Dataset versions.
- Every task updates the affected docs in the same commit (DC-15).

### M0 — Environment probe + skeleton wired (Tier 0) · S01 · 1.0 GPU-h · tag `m0-green`
**Goal:** Verify Kaggle limits at runtime and make the package installable and testable, with the CLI, config, runner, manifest and cache wired.
**Files:** `src/rbbd/{cli,config,runner}.py`, `src/rbbd/utils/*` (incl. `store.py`, D-046), `configs/base.yaml`, `configs/smoke.yaml`, `notebooks/00_probe.ipynb`, `notebooks/run_stage.ipynb`, `tests/test_config.py`, `tests/test_manifest.py`, `tests/test_cache.py`, `tests/test_env.py`, `tests/test_store.py`, `tests/test_leakage.py`, `tests/gpu/test_generate_gpu.py`, docs.
**Tasks:**
- M0-T1 Implement `utils/env.py` probe:
  - Run `nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv`, `df -h /kaggle/working /kaggle/tmp /`, `free -g` and `python -V`.
  - Record torch/CUDA versions, `torch.cuda.is_bf16_supported()`, GPU count and the Kaggle env detection.
  - Write `env.json` and fail loudly if fewer than 2 GPUs or compute cap < 7.5.
  - Update the Kaggle limits recorded in D-002 with the observed values.
- M0-T2 `config.py` (YAML merge `base.yaml` ← tier file, canonical JSON → SHA-256 config hash), `utils/manifest.py` (write, validate, skip), `utils/cache.py` (key builder, safetensors read/write, `cache hit` logging), `utils/seeds.py`, `utils/guards.py` (NaN/Inf guard).
- M0-T3 `.gitignore` (`artifacts/`, `*.env`, `kaggle.json`, HF caches) and `runner.py` stage graph with no-op stage stubs that raise `NotImplementedError` with milestone IDs; `cli.py` with `probe`, `run` and `status`.
- M0-T4 Kaggle install test: `pip install -e .[train]` and `.[bench]` on the Kaggle image. Record `pip freeze`. Resolve pin conflicts; each change gets a D-031 follow-up.
- M0-T5 Gated-access check: for every repo in §5, call `huggingface_hub.model_info`/`dataset_info` with the token, without downloading weights. Record pass/fail per repo in `env.json`, without the token.
- M0-T6 vLLM hello-world:
  - Qwen2.5-0.5B, TP=1 and TP=2, `dtype=float16`, `enable_lora=True`, 4 prompts.
  - Record which engine (V0/V1) and attention backend is used.
  - Try `dtype=float16` on Gemma-3-1B and record whether it is refused (B-005).
- M0-T7 Artifact-store round trip (D-041): write a 1 MB file to `/kaggle/working/artifacts/env/<session>/`, upload it with `HfApi.upload_folder` to the private HF repo `SarthakGupta414/rbbd-artifacts` (after checking `private is True`), and read it back with a forced fresh download into an empty cache (D-045).
**Checks:** DC-01, DC-02, DC-10 (manifest/cache unit level), DC-15, DC-16 (probe).
**Rollback:** `m0-green`. Nothing to invalidate.

### M1 — Sentence sets + metrics on CPU with synthetic vectors · CPU · 0 GPU-h · tag `m1-green`
**Goal:** Frozen, validated sentence sets and all metrics implemented and verified on synthetic embeddings.
**Files:** `src/rbbd/data/*`, `src/rbbd/resources/sentences/**`, `src/rbbd/metrics/*`, `src/rbbd/analysis/stats.py`, `tests/test_sentences.py`, `tests/test_metrics_rr.py`, `tests/test_baselines.py`, `tests/test_stats.py`, `tests/test_guards.py`, docs.
**Tasks:**
- M1-T1 `data/groups.py`: 24 DT groups, 9 topics (T2), ToxiGen 9-group map (D-025). `data/hashing.py`: NFC-normalise, strip, SHA-256 over newline-joined ordered list.
- M1-T2 Author the base target templates (50) and neutral anchors (1,000; ≈42/group) with disjoint template IDs (D-012, D-014).
- M1-T3 Author P (100) and N (100) with length matching (D-013). Author the 5 attribute variants (D-016).
- M1-T4 Author the 5 target variants (D-016). Write the validation checks of §2 as tests.
- M1-T5 `metrics/rr.py`: r(X), S⁺_rel, S⁻_rel, B_rel per group (Eq. 4–6). `metrics/delta_b.py`: ΔB per group (Eq. 7). Implement the identity, rotation and anchor-permutation tests.
- M1-T6 `metrics/seat.py` (Eq. 2–3), `procrustes.py`, `cka.py` with property tests.
- M1-T7 `analysis/stats.py`:
  - Pearson with two-tailed p.
  - ROC AUC with score −ΔB, plus the undirected variant for CKA.
  - Bootstrap MAE (1,000 resamples, in-sample plus OOB).
  - Bootstrap CIs for r and AUC, paired bootstrap for RR − SEAT, threshold sweep.
  - Stats tests.
- M1-T8 `data/ft_data.py`: WGM split selection logic plus a CPU token-length census, run in a CPU Kaggle session (token counts drive the M3 budget; update §8 if mean tokens/example > 450).
**Checks:** DC-01, DC-02, DC-03, DC-04, DC-05, DC-07, DC-08, DC-15, DC-17 (sentence-set validators).
**Rollback:** `m1-green`. Sentence-set hash changes invalidate the `embeddings/*` namespace via the key.

### M2 — Embedding extraction + cache (Tier 0, plus a Tier 1 timing probe) · S02 · 1.5 GPU-h · tag `m2-green`
**Goal:** Batched, length-sorted, mask-aware pooled extraction that is guarded, cached and fast enough.
**Files:** `src/rbbd/models/loading.py`, `src/rbbd/embed/*`, `tests/test_pooling.py`, `tests/test_cache.py`, `tests/test_guards.py`, `tests/gpu/test_extract_gpu.py`, docs.
**Tasks:**
- M2-T1 `models/loading.py`: load tokenizer and model with an explicit precision, placement and pad-token policy (pad = eos when missing), right padding, `eval()`, `torch.inference_mode`. For Gemma 3, load the text tower (`Gemma3ForCausalLM` from the `-it` checkpoint, or the language-model submodule).
- M2-T2 `embed/pooling.py` (mean, max, last; fp32 accumulation) and `embed/extract.py`:
  - Dedupe, then sort by length, then batch by token budget.
  - Run the base-model forward and pool.
  - NaN/Inf guard on the hidden states before pooling.
  - Scatter results back to the original order and write the cache.
- M2-T3 Tier 0: extract base Qwen2.5-0.5B on the smoke sets. Run the padding test on GPU in fp16 (DC-06) and the cache-hit rerun (DC-10).
- M2-T4 Tier 1 timing probe: Llama-3.1-8B fp16 sharded over 2 T4s, full sets plus ablation sets (≈12.4k sentences). Measure time and peak memory per GPU; must meet DC-12 (≤ 15 min excl. download). Also run Gemma-3-4B in fp16 once to see whether the guard fires (B-001) and record the outcome in D-005.
**Checks:** DC-01, DC-02, DC-06, DC-09, DC-10, DC-12, DC-15.
**Rollback:** `m2-green`. Bump `embed.schema_version` if the pooling or key semantics change.

### M3 — Fine-tuning + merges · 3a Tier 0 (S03), 3b Tier 2 (S04–S06), 3c Tier 1 (S07–S12) · ~34 GPU-h · tags `m3a-green`, `m3b-green`, `m3c-green`
**Goal:** Endpoint adapters and checkpoints per model and regime, plus exact in-memory α interpolation.
**Files:** `src/rbbd/finetune/sft.py`, `src/rbbd/models/{adapters,interpolate,spectrum}.py`, `src/rbbd/data/ft_data.py`, `tests/test_merge.py`, `tests/gpu/test_train_gpu.py`, configs, docs.
**Tasks:**
- M3-T1 `finetune/sft.py`:
  - TRL `SFTTrainer`, prompt–completion format with the model's chat template and completion-only loss (D-040).
  - Paper hyperparameters (App. C): 3 epochs, per-device batch 4 × grad-acc 8 (or 2 × 16 / 1 × 32 when memory requires; effective 32 either way), warmup 0.03, linear schedule, AdamW, max len 1024; LoRA r=16, α=32, dropout 0.05, all 7 projections; lr 1e-4 (LoRA) / 2e-5 (full).
  - fp16 mixed precision, gradient checkpointing, `save_steps` every ~20 min of wall-clock, resume from the latest checkpoint.
  - Log loss and grad-norm and abort on non-finite values (B-001).
- M3-T2 `models/adapters.py`: build the concatenated rank-2r adapter for α: B' = [α·s·B_u | (1−α)·s·B_h], A' = [A_u; A_h], lora_alpha = r' so scaling = 1 (D-007).
  - Test: materialised ΔW' equals α·ΔW_u + (1−α)·ΔW_h (fp32, atol 1e-6 relative).
  - Test: α=1 and α=0 reproduce the endpoint adapters' ΔW exactly (DC-07).
  - Export the combined adapters in PEFT format for vLLM.
- M3-T3 `models/interpolate.py` (Tier 2): endpoints loaded on CPU in fp32; W(α) = (1−α)·W_h + α·W_u computed in fp32, then cast. This is exact at α ∈ {0, 1}, which the brief's form W_h + α(W_u − W_h) is not in floating point (D-008). Copy into GPU parameters in place. Test DC-07. `materialize(α, dir)` writes to ephemeral disk only, for vLLM.
- M3a-T4 Tier 0: smoke LoRA on 64 examples/split on cuda:0 and cuda:1 in parallel. Run the endpoint tests on real adapters.
- M3b-T5 Tier 2 Qwen2.5-0.5B: full FT u‖h in parallel, then LoRA u‖h. Store endpoints in the private checkpoint Dataset.
- M3b-T6 Tier 2 Llama-3.2-1B: full FT u‖h (check peak memory; fall back to per-device batch 1 × 32 grad-acc), then LoRA u‖h.
- M3b-T7 Seeds: Qwen2.5-0.5B full FT, seeds {1, 2} × {u, h} (seed 0 already exists).
- M3c-T8 Throughput probe (30 min): Llama-3.1-8B QLoRA, 50 steps; measure tok/s and project wall-clock. If the projection exceeds 9 h per run, raise open question Q3 before continuing (options: 1 epoch or max len 512).
- M3c-T9 Llama-3.1-8B QLoRA u‖h. M3c-T10 Mistral-7B QLoRA u‖h.
- M3c-T11 Gemma-3-4B: 50-step fp16 probe with a hidden-state max-abs monitor. On any non-finite value, switch to fp32 compute (D-005), then train u‖h across 2 sessions with resume.
**Checks:** DC-01, DC-02, DC-07, DC-09 (guard active in training forward), DC-13 (resume test), DC-15.
**Rollback:** `m3a/b/c-green`. Adapters and checkpoints are immutable once tagged; a retrain gets a new `train_key`.

### M4 — ΔB across the spectra · S03 (T0), S06 (T2), S12–S13 (T1) · 2.7 GPU-h · tag `m4-green`
**Goal:** `delta_b.csv` for every (model, regime, checkpoint α, group) and every ablation cell, from cached embeddings.
**Files:** `src/rbbd/models/spectrum.py`, `src/rbbd/metrics/delta_b.py`, `src/rbbd/runner.py` (deltab stage), `tests/test_delta_b_table.py`, configs, docs.
**Tasks:**
- M4-T1 `spectrum.py`: α list {1.0, 0.9, 0.7, 0.5, 0.3, 0.1, 0.0} (App. C) plus `ref`. Canonical checkpoint slugs `a100 … a000`. Iterator yields (slug, activate-fn) against one loaded base (E6).
- M4-T2 `embed` stage sweeps ref + 7 α per model/regime in one session. Assert the ref and audited cache keys differ only in the checkpoint field (same precision, placement, quant, layer; D-004).
- M4-T3 `deltab` stage: long-format CSV with columns `model, regime, alpha, group, topic, method(rr|seat|procrustes|cka), anchor_source, n_anchors, anchor_seed, attr_variant, target_variant, pooling, layer_site, B_ref, B_aud, delta_b`.
- M4-T4 Run Tier 0 (smoke, DC-11), then Tier 2 (4 spectra + seeds), then Tier 1 (3 spectra).
- M4-T5 Sanity report: ΔB(α=1) vs ΔB(α=0) per group. Expect mean ΔB(h) < mean ΔB(u), cf. App. E.5 means of 0.291 vs −0.051. This is a signal, not a gate: log the result in BUGS if it fails.
**Checks:** DC-01–DC-05, DC-10, DC-11, DC-12, DC-15.
**Rollback:** `m4-green`. Invalidate `deltab/*` by bumping `metrics.schema_version`.

### M5 — Output benchmarks · 5a harness (S03), 5b Tier 2 (S06–S07), 5c Tier 1 (S13–S17) · ~27.5 GPU-h · tags `m5a/b/c-green`
**Goal:** ΔBiasScore / ΔToxicity per (model, regime, α, group or topic) with bootstrap CIs.
**Files:** `src/rbbd/bench/*`, `tests/test_bench_scoring.py`, `tests/gpu/test_generate_gpu.py`, configs, docs.
**Tasks:**
- M5-T1 Pin the DT and ToxiGen commit SHAs. Fetch the DT stereotype prompts, system prompts and agreement parser. Fetch the ToxiGen prompt files for 9 groups. Record the SHAs in D-022 and D-025. Write the parity fixtures (DT parser on 200 canned responses).
- M5-T2 `bench/generate.py`:
  - vLLM engine factory (TP by model size, dtype per D-004/D-005, `enable_lora`, `max_lora_rank=32`).
  - Chat-template prompts; seeded sampling (T=0.7, top-p 1.0).
  - Sharded JSONL output with resume.
  - HF-generate fallback behind a flag (B-005).
- M5-T3 `bench/wildguard.py`: subcategory filter, official WildGuard input format, parse, then harmful-response rate per topic.
- M5-T4 `bench/topic_map.py`: one-off run (Qwen2.5-7B-Instruct, greedy), freeze the CSV, human spot check of 30 items (D-024).
- M5-T5 `bench/decodingtrust.py`: build prompts, agreementIndex per group per system prompt, mean over system prompts = BiasScore (D-022).
- M5-T6 `bench/toxigen.py`: seeded prompt subset, user-turn format, first-statement extraction, RoBERTa scoring, toxic fraction per group.
- M5-T7 Tier 0 harness run inside the smoke config (DC-11 now includes benchmark stages).
- M5-T8 Tier 2 benches: 2 models × 2 regimes × (ref + 7), with two jobs in parallel on the two GPUs.
- M5-T9 Tier 1 benches: one vLLM engine per model with LoRA requests (E8). WildGuard scoring is batched per model after generation.
**Checks:** DC-01, DC-02, DC-11, DC-13, DC-14 (bench timing logged), DC-15, DC-18 (no generated text in `results/` or git).
**Rollback:** `m5*-green`. Generations are immutable per `gen_key`; scorer changes bump `score.schema_version` only.

### M6 — Stats, baselines, ablations · S18 (≤1.5 GPU-h) + CPU · tag `m6-green`
**Goal:** Every number behind T1, F2–F4, T5–T7 and App. E.5 for our tiers.
**Files:** `src/rbbd/analysis/*`, `configs/ablations.yaml`, `tests/test_stats.py`, docs.
**Tasks:**
- M6-T1 Join ΔB with the benchmark deltas. WGM pairs each group with its topic's ΔBiasScore (D-024); DT and ToxiGen pair by group. Compute Pearson, AUC with CI, and bootstrap MAE per setting.
- M6-T2 Baselines SEAT, Procrustes-SEAT and CKA drift on the same observations. Paired bootstrap RR − SEAT.
- M6-T3 Threshold sweep (F3 analogue) on Tier 1 Llama and Tier 2 models.
- M6-T4 Ablations from cached embeddings:
  - anchor source × size (F4);
  - attribute and template variants (T5a/b);
  - pooling (T6);
  - layer site pre/post norm (D-017 check);
  - WGM topic-mean pairing (D-024 sensitivity).
- M6-T5 Seeds (App. E.5): ΔB std across seeds per dataset, Tier 2.
- M6-T6 GPU (≤ 1.5 h): NF4-base extraction for Llama-3.1-8B ref + endpoints, to quantify the D-004 precision choice (|ΔB_fp16 − ΔB_nf4|).
**Checks:** DC-01, DC-02, DC-08, DC-10, DC-15.
**Rollback:** `m6-green`.

### M7 — Report · CPU · 0 GPU-h · tag `m7-green`
**Goal:** `results/REPORT.md` plus figures reproducing T1, F2, F3, F4, T5, T6, T7 and E.5 for our tiers (§10 template), and the verdict per §1.2.
**Tasks:** M7-T1 tables; M7-T2 figures; M7-T3 verdict and deviations section; M7-T4 cost table (T3 analogue on T4).
**Checks:** DC-01, DC-02, DC-15, DC-18.

### M8 — Diff against official code (conditional) · 0–2 GPU-h · tag `m8-green`
Trigger: the authors' repository becomes public.
**Tasks:**
- M8-T1 Diff their sentence sets, anchors, templates, pooling, layer indexing, merge code and stats against ours. Every difference becomes a D-entry: adopt, or keep with a reason.
- M8-T2 If their sentence sets are released, re-run M4/M6 on cached models with their sets (CPU + ≤ 2 GPU-h). Report both.
- M8-T3 Run their code on the Tier 0 smoke spectrum and compare ΔB (atol set at diff time).

---

## 8. GPU budget

Units: **Kaggle session-hours on the 2×T4 accelerator** (Kaggle bills session time, not per GPU; D-002). Estimates assume ~400 tokens per training example (measured in M1-T8 and re-planned if different), QLoRA-8B at ≈ 350 tok/s per T4, and vLLM 8B TP=2 at ≈ 800 output tok/s. Every estimate is re-baselined from measurements in M2-T4, M3c-T8 and M5-T7, and this table is updated.

| Milestone | Tier 0 | Tier 2 | Tier 1 | Session(s) | Notes |
|---|---|---|---|---|---|
| M0 probe | 1.0 | – | – | S01 | incl. vLLM hello-world |
| M1 | 0 | 0 | 0 | CPU | token census on CPU |
| M2 extraction | 1.0 | – | 0.5 | S02 | 8B timing probe + Gemma fp16 probe |
| M3 training | 0.5 | 8.0 (Qwen-0.5B full+LoRA 2.0, Llama-1B full+LoRA 4.0, seeds 2.0) | 25.0 (Llama 7.5, Mistral 7.0, Gemma 9.0 assuming fp32 fallback, probes 1.5) | S03–S12 | parallel u‖h on 2 GPUs |
| M4 ΔB | 0.2 | 1.0 | 1.5 | S03, S06, S12–13 | one base load per model |
| M5 benchmarks | 1.5 (harness + topic map) | 5.0 | 21.0 (Llama 5.5, Mistral 5.5, Gemma fp32 8.5, WildGuard 1.5) | S03, S06–07, S13–17 | per ckpt 8B ≈ 35 min: WGM 5, DT 20, ToxiGen 8, overhead 2 |
| M6 | – | 0.5 | 1.0 | S18 | NF4 check + stragglers |
| Smoke reruns (regression) | 1.0 | – | – | various | ~4 × 15 min |
| **Subtotal** | **5.2** | **14.5** | **49.0** | | **68.7** |
| Contingency (15%) | 0.8 | 2.2 | 7.3 | | |
| **Total** | **6** | **17** | **56** | | **≈ 79 session-h** |
| Stretch (Tier 2 Qwen-1.5B + Gemma-1B) | – | +9 | – | | only if budget remains after W3 |

**Schedule within ~30 h/week** (with a 2 h reserve kept each week):

| Week | Sessions | Work | Session-h |
|---|---|---|---|
| W1 | S01–S07 | M0, M2, M3a, M4-T0, M5a, M3b (Tier 2 training), M4 Tier 2, M5b | ≈ 25 |
| W2 | S08–S11 | M3c: throughput probe, Llama, Mistral, Gemma probe + session 1 | ≈ 26 |
| W3 | S12–S16 | Gemma session 2, M4 Tier 1, M5c Llama + Mistral + Gemma part 1 | ≈ 24 |
| W4 | S17–S18 | M5c Gemma part 2, WildGuard scoring, M6 GPU, reruns, stretch if any | ≈ 12 |

**Cut order if over budget** (apply top-down; stop as soon as the budget fits):
1. Stretch Tier 2 models.
2. GPU ablations: the NF4-extraction check. The seeds ablation is reduced to 1 extra seed. (Embedding-only ablations are nearly free and stay.)
3. Subsample benchmarks harder, in this order:
   - DT n=3 → n=2 and benign + targeted system prompts only;
   - ToxiGen 100 → 50 prompts/group;
   - Gemma DT to 2 of 3 system prompts.
   WGM stays complete because it is cheap.
4. Tier 1 training: 3 epochs → 1 epoch (requires user approval, Q3).
5. **Never cut:** the core ΔB-vs-benchmark correlation for any Tier 1 model or benchmark, or the SEAT baseline.

---

## 9. Risk register

| # | Risk | Likelihood | Mitigation | Early detection signal | Bug |
|---|---|---|---|---|---|
| R1 | fp16 overflow (Gemma especially) → NaN/Inf hidden states or loss | High (Gemma), low (others) | Mandatory guard; Gemma fp32 for embed and gen; fp32-compute fallback in training; probe in M2-T4 / M3c-T11 | Guard raises; loss = nan; max-abs activation > 3e4 in monitor | B-001 |
| R2 | Padding-side / pooling error (pad tokens in mean, left-pad `last` index) | Medium | Right padding enforced; mask-aware pooling; padding test DC-06; `last` = index of the final non-pad token | DC-06 fails; ΔB(M, M) ≠ 0 across batch sizes | B-002 |
| R3 | Non-linear adapter merge (interpolating A and B separately) | Medium | Concatenated rank-2r adapter; ΔW linearity test; endpoint tests | DC-07 fails | B-003 |
| R4 | Gated model or dataset access denied / token missing | Medium | M0-T5 access check before any GPU work; the user accepts licences in advance (Q1) | 401/403 in probe | B-004 |
| R5 | vLLM on T4: V1 unsupported, gemma3 fp16 refused, LoRA kernel failure, TP=2 NCCL hangs | High | Pin the vLLM version with V0 or Turing support; Gemma fp32; HF-generate fallback flag; M0-T6 probe | Probe fails; engine start > 10 min | B-005 |
| R6 | Benchmark drift: DT repo, ToxiGen prompt paths, WGM field names or subcategory strings change | Medium | Pin commit SHAs and dataset revisions; parity fixtures; schema asserts | Assertion on missing fields; parity test fails | B-006 |
| R7 | 12 h cutoff mid-stage | High for Tier 1 training | Checkpoint every ~20 min; resumable generation shards; manifests | Session killed; `complete=false` manifest | B-007 |
| R8 | `/kaggle/working` 20 GB exceeded | Medium | Only adapters (~80–170 MB each) and fp16 embeddings (~100 MB/ckpt at 8B) persist; HF cache in `/kaggle/tmp`; Tier 2 endpoints in the private Dataset; `du` budget check at stage end | `du -sh` > 15 GB warning | B-008 |
| R9 | Weekly quota overrun | Medium | §8 cut order; re-baseline after M3c-T8 | Actual > 1.2× estimate for any milestone | – |
| R10 | Degenerate labels (few checkpoints exceed the thresholds on small models or under subsampling) | Medium (Tier 2) | Report the threshold sweep; bootstrap CIs; mark n/a; pre-registered in §1.2 | < 5 positives in a setting | – |
| R11 | Reconstructed sentence sets differ materially from the authors' | Certain (unknown size) | App. F statistics checks; ablation variants show sensitivity; M8 diff | T5-style AUC spread ≫ paper | – |
| R12 | Topic-map LLM misclassifies WGM prompts | Medium | Spot check ≥ 80%; sensitivity with topic-mean pairing | Spot check < 80% | – |
| R13 | QLoRA-trained adapters applied to an fp16 base change behaviour | Low–medium | Same base precision for ref and all audited ckpts; NF4 extraction check in M6-T6 | \|ΔB_fp16 − ΔB_nf4\| > 0.1 × range | – |
| R14 | Pin conflicts between vLLM and TRL/Transformers | Medium | Separate `bench` and `train` extras; separate venv for bench if needed (D-031) | Resolver error in M0-T4 | – |
| R15 | Tier 1 training slower than estimated | Medium | Throughput probe M3c-T8; Q3 fallback | Projection > 9 h/run | – |
| R16 | Harmful artifacts leak publicly | Low | Private stores only; DC-18 grep check; `.gitignore` for `artifacts/` | DC-18 fails | – |
| R17 | Kaggle Models mirror differs from the HF revision | Low | Checksum a named tensor plus config diff in M0 | Checksum mismatch | – |

---

## 10. Results template (M7 deliverables)

`results/REPORT.md` must contain the following.

**Table R1 — T1 analogue.** One table per tier. Rows are benchmark × model. Columns:

| Benchmark | Model | Tier | Regime | n obs | n pos | Pearson r [95% CI] (p, stars) | ROC AUC [95% CI] | MAE (boot mean, in-sample) | MAE (OOB) | Paper r / AUC / MAE (T1) |
|---|---|---|---|---|---|---|---|---|---|---|

- Tier 1 has the LoRA columns only.
- Tier 2 has full and LoRA columns side by side.
- The paper's T1 values are shown alongside for Tier 1 models.
- Star convention as in T1: * < .05, ** < .01, *** < .001.

**Table R2 — T7 analogue.** Method comparison (RR, SEAT, Procrustes-SEAT, CKA drift): AUC and r per model × benchmark (WGM, DT), with paired-bootstrap Δ AUC vs RR.

**Table R3 — T5a/b analogue.** AUC per attribute variant and per template variant. Mean ± std, min/max.

**Table R4 — T6 analogue.** Pooling (mean/max/last) × (RR, SEAT) AUC.

**Table R5 — E.5 analogue.** ΔB std across 3 seeds per dataset (Tier 2). Mean ΔB (u) vs (h).

**Table R6 — T3 analogue.** Wall-clock per model on 2×T4: embeddings, ΔB, WGM gen+classify, ToxiGen gen+classify, DT. Include the ratio vs our method.

**Table R7 — Verdict.** Settings passing C1/C2/C3; tier verdicts; S1–S7 outcomes; list of deviations (§1.3).

**Figures:**
- F-R1 (F2 analogue): scatter of ΔScore vs ΔB coloured by α, per model × benchmark × regime, with regression line and r.
- F-R2 (F3 analogue): AUC vs threshold for the four methods.
- F-R3 (F4 analogue): AUC vs number of anchors for the four sources, with seed bands.
- F-R4 (App. E.5 analogue): per-group ΔB by seed.
- F-R5 (F7/8/11/12/15/16 analogue): ROC curves per model.

---

## 11. Open questions for the user
**All answered 2026-09-30** — Q1 yes (D-044), Q2 private HF repo (D-041), Q3 max len 512 first then 1 epoch (D-042), Q4 local Qwen2.5-7B with ~100 hand labels (D-043), Q5 28 h/week (D-042, D-044), Q6 yes after core results (D-042). Original questions kept below for the record.

- **Q1 Accounts and licences.** Has the Kaggle account accepted the HF licences for Llama-3.1-8B, Llama-3.2-1B, Gemma-3 (4B, 1B), Mistral-7B-v0.3, WildGuardMix and WildGuard? Is the Kaggle account phone-verified so GPU and internet are available?
- **Q2 Artifact store.** Private Kaggle Datasets (needs `KAGGLE_USERNAME`/`KAGGLE_KEY` as Kaggle Secrets) or a private HF repo (needs a write-scoped HF token)? The default in D-029 is private Kaggle Datasets.
- **Q3 Training budget fallback.** If the throughput probe projects > 9 h per Tier 1 QLoRA run, may we drop to 1 epoch (or max len 512), or should Tier 1 stay at 3 epochs and absorb more weeks?
- **Q4 Topic mapping model.** Is a local Qwen2.5-7B-Instruct acceptable in place of ChatGPT 5.2 (App. B)? Or do you want to supply an API key for an external model (a network call to a third party)?
- **Q5 Weekly quota.** Confirm the available quota (≈ 30 h/week) and whether ~4 weeks elapsed is acceptable.
- **Q6 Tier 2 stretch.** Should Qwen2.5-1.5B and Gemma-3-1B be attempted if budget remains?
