# DECISION.md

**Purpose.** This is the durable record of every meaningful decision: why the code looks the way it does, what it costs, and where it departs from the paper (arXiv:2609.10060v1). A decision is logged **in the same commit** as the change it justifies. Entries are never deleted. To replace one, add a new entry and mark the old heading `— superseded by D-###`.

**Entry format**
```
## D-### <title>
- Date: YYYY-MM-DD
- Context: why a decision was needed (cite paper § / table / appendix)
- Options considered: A / B / C
- Choice: ...
- Why: ...
- Tradeoff accepted: ...
- Cost impact: GPU-minutes (Kaggle session-minutes), disk, memory
- Paper deviation: yes/no + section   (reconstructions of unreleased material count as "yes (reconstruction)")
- Revisit-if: observable condition that would reopen this
```
IDs are sequential and never reused. Entries D-001…D-040 were made at planning time (2026-09-30) from the PDF alone. Nothing had been measured yet, so every cost figure is an estimate to be re-baselined.

---

## D-001 Split the study into three tiers
- Date: 2026-09-30
- Context: The paper full-fine-tunes and LoRA-tunes 4–8B models on an A100 40GB (App. C, §4.6). Kaggle gives 2× T4 16GB with a ~30 h weekly quota.
- Options considered:
  - A: Attempt the full paper setup on T4. Full FT of 8B needs ~130 GB, so this is infeasible.
  - B: LoRA only on the paper's models.
  - C: Tier 0 smoke + Tier 1 (QLoRA on the paper's models) + Tier 2 (full FT and LoRA on 0.5–1.2B models).
- Choice: C.
  - Tier 0 = Qwen2.5-0.5B-Instruct smoke, < 15 min.
  - Tier 1 = QLoRA on Llama-3.1-8B-Instruct, Mistral-7B-Instruct-v0.3, Gemma-3-4B-it.
  - Tier 2 = full FT **and** LoRA on Qwen2.5-0.5B-Instruct and Llama-3.2-1B-Instruct (stretch: Qwen2.5-1.5B, Gemma-3-1B).
- Why: Tier 1 is the closest feasible replication of the T1 LoRA columns. Tier 2 is the only way to test the full-FT vs LoRA finding (§4.2) and the scale question (Limitations; §4.2 Gemma discussion) within budget. Tier 0 protects GPU hours.
- Tradeoff accepted: The paper's full-FT columns for 4–8B models are not replicated. Tier 2 full-FT results come from different, smaller models.
- Cost impact: ≈ 79 session-h total (PLAN §8): T0 ≈ 6, T2 ≈ 17, T1 ≈ 56.
- Paper deviation: yes — App. C training regimes, T1 full-FT columns.
- Revisit-if: an A100/H100-class machine becomes available, or the quota changes.

## D-002 Hardware target 2× T4; budget unit = Kaggle session-hours
- Date: 2026-09-30
- Context: Kaggle offers 2× T4 (sm75, 16GB each, no native bf16) or 1× P100 (sm60). vLLM does not support P100. Sessions stop at ~12 h, the quota is ~30 h/week, and `/kaggle/working` persists ~20 GB.
- Options considered: A: P100. B: 2× T4.
- Choice: B. Budgets are counted in session-hours, because Kaggle bills a 2-GPU session once, not per GPU. M0 verifies GPUs, disk and RAM at runtime (`nvidia-smi`, `df -h`, `free -g`) and amends this entry with observed values.
- Why: vLLM and fp16 tensor cores are available on T4, and two GPUs allow parallel jobs.
- Tradeoff accepted: No bf16, so fp16 overflow risk (D-005, D-006). 16 GB per device.
- Cost impact: Parallel jobs roughly halve quota use for training and small-model benches.
- Paper deviation: yes — §4.6 / App. C (A100 40GB, bf16).
- Revisit-if: M0 probe shows different hardware, or Kaggle changes billing.

## D-003 Tier 1 training = QLoRA (NF4), one split per GPU in parallel
- Date: 2026-09-30
- Context: LoRA training for 7–8B (App. C: bf16 LoRA r=16). fp16 8B weights = ~16 GB, which exceeds one T4.
- Options considered:
  - A: fp16 LoRA with the model sharded across both T4s (naive pipeline, one GPU idle at a time, fp16 base overflow risk).
  - B: QLoRA with DDP over 2 GPUs.
  - C: QLoRA with u on cuda:0 and h on cuda:1 as independent processes.
  - D: Unsloth (faster, but single-GPU and patches model code).
- Choice: C.
  - NF4, double quantisation, compute dtype fp16 (fp32 for Gemma if D-005 triggers).
  - Paper hyperparameters otherwise unchanged. Effective batch 32 is reached by grad-accum when per-device 4 does not fit.
  - Gemma-3-4B *could* fit fp16 LoRA on one T4, but uses QLoRA for consistency across Tier 1.
- Why: Same throughput as DDP with no NCCL/DDP failure modes. Each run fits one T4. Two runs finish in one session.
- Tradeoff accepted: The quantised base during training differs from the paper's bf16 base, and adapters are later applied to an fp16 base (D-004). Slower per step than bf16 on A100.
- Cost impact: ≈ 7–9 session-h per model pair (u‖h); ≈ 25 h for Tier 1 including probes. VRAM ≈ 6 GB weights + activations per GPU.
- Paper deviation: yes — App. C (bf16 LoRA).
- Revisit-if: the M3c-T8 throughput probe projects > 9 h/run (see Q3), or QLoRA training diverges.

## D-004 One precision and placement per model for embeddings and generation: fp16 un-quantised base + adapter
- Date: 2026-09-30
- Context: ΔB compares reference and audited embeddings (§3.4). Any precision or placement difference between them would leak into ΔB.
- Options considered:
  - A: Extract on the NF4 base (as trained).
  - B: fp16 base, sharded over both T4s (`device_map` balanced for HF; TP=2 for vLLM), with un-merged LoRA adapters.
  - C: Materialise merged fp16 weights per checkpoint.
- Choice: B for all Tier 1 models except Gemma (D-005). Tier 2 uses one T4 per model in fp16 (Gemma fp32). The reference is the same loaded model with adapters disabled. Cache keys include precision, quant and placement, and the ΔB stage asserts ref and aud keys differ only in the checkpoint field.
- Why: This is the standard QLoRA deployment (adapter on a higher-precision base). One base load serves all α. It matches the vLLM LoRA serving path, so embeddings and generations see the same weights.
- Tradeoff accepted: A small train/inference mismatch (the adapter was trained on NF4). Quantified in M6-T6.
- Cost impact: 8B fp16 ≈ 16 GB across 2 GPUs, leaving ≈ 6 GB/GPU headroom. No extra disk.
- Paper deviation: yes — App. C (bf16).
- Revisit-if: M6-T6 shows |ΔB_fp16 − ΔB_nf4| > 10% of the ΔB range.

## D-005 Gemma-3 numerics: fp32 for inference; fp16-then-fp32 fallback for training
- Date: 2026-09-30
- Context: Gemma models are known to overflow in fp16. vLLM marks gemma2/gemma3 as not supporting float16. The paper used bf16 (App. C). T4 has no bf16.
- Options considered: A: fp16 everywhere with clamping hacks. B: fp32 for Gemma inference (4B fp32 ≈ 17 GB across 2 T4) and fp32 compute in QLoRA if the fp16 probe fails. C: Drop Gemma.
- Choice: B.
  - Embeddings and vLLM generation for Gemma-3-4B (and Gemma-3-1B in Tier 2) run in fp32.
  - Training first runs a 50-step fp16-compute probe with an activation max-abs monitor. Any non-finite value switches to fp32 compute.
  - The M2-T4 fp16 extraction probe outcome is appended here.
- Why: Correctness over speed. Keeps the paper's weakest-but-informative family.
- Tradeoff accepted: Gemma is 2–4× slower. T3 already shows Gemma as the slowest.
- Cost impact: +≈ 5 h training and +≈ 3 h benches versus fp16 (included in PLAN §8).
- Paper deviation: yes — App. C precision.
- Revisit-if: the fp16 probes pass cleanly (guard never fires and outputs match fp32 on 50 prompts within tolerance). Then switch to fp16 via a superseding entry.

## D-006 Mandatory NaN/Inf guard on every hidden-state extraction and training step
- Date: 2026-09-30
- Context: fp16 on T4 (D-002, D-005).
- Options considered: A: Check only final ΔB. B: Guard hidden states per batch and loss/grad per step.
- Choice: B. `utils/guards.assert_finite(t, where)` runs on final-layer hidden states for every batch before pooling, and on pooled outputs. On failure it raises `NonFiniteError` with model, checkpoint and batch indices. Training aborts on non-finite loss or grad-norm. The guard is never disabled by config.
- Why: Silent NaNs would corrupt ΔB and pass downstream as numbers.
- Tradeoff accepted: One `isfinite().all()` reduction per batch (negligible).
- Cost impact: < 1% runtime.
- Paper deviation: no.
- Revisit-if: never (a safety invariant).

## D-007 LoRA merge = exact linear combination of ΔW via a concatenated rank-2r adapter
- Date: 2026-09-30
- Context: App. C merges checkpoints linearly (Wortsman et al., 2022) at 10/90…90/10. Interpolating A and B separately gives (αB_u+(1−α)B_h)(αA_u+(1−α)A_h), which is not linear in ΔW.
- Options considered:
  - A: Materialise W₀ + αΔ_u + (1−α)Δ_h as full weights per α (7 × 16 GB, infeasible to store).
  - B: Merge ΔW into base weights in memory per α (HF only; not usable by vLLM without writing weights).
  - C: Concatenated adapter B' = [α·s_u·B_u | (1−α)·s_h·B_h], A' = [A_u; A_h], rank 2r = 32, lora_alpha = 32 (scaling 1). Then B'A' = αΔW_u + (1−α)ΔW_h exactly, with ΔW_x = s_x·B_x·A_x.
- Choice: C. The brief's requirement "merge each adapter to ΔW = BA first" is met, because the product is mathematically that sum. PEFT `add_weighted_adapter(combination_type="cat")` may be used only if the ΔW-linearity test passes against it; otherwise use our own builder. α := weight on the unharmful adapter, α ∈ {1.0, 0.9, 0.7, 0.5, 0.3, 0.1, 0.0}.
- Why: Exact. Tiny on disk (~2× adapter). Served directly by vLLM multi-LoRA (E8) and by PEFT for embeddings, against one loaded base.
- Tradeoff accepted: Rank-32 adapters are slightly slower to apply than rank 16. vLLM needs `max_lora_rank=32`.
- Cost impact: 5 combined adapters × ≈ 170 MB (8B) ≈ 0.85 GB per model, regenerable on the fly (≈ seconds). Saves ~7 × 16 GB of merged weights.
- Paper deviation: no (implementation of App. C merging).
- Revisit-if: vLLM rejects rank 32 on T4, or the linearity test fails at fp16 (then merge in fp32 and store at fp16).

## D-008 Tier 2 full-FT merges in memory: W(α) = (1−α)·W_h + α·W_u in fp32
- Date: 2026-09-30
- Context: Full-FT endpoints of 0.5–1.2B models. The brief suggests W_h + α(W_u − W_h). DONE requires α=1 and α=0 to reproduce the endpoints **exactly**.
- Options considered: A: W_h + α(W_u − W_h). In floating point, α=1 does not return W_u bit-exactly. B: (1−α)·W_h + α·W_u in fp32. At α=1 this is 0·W_h + 1·W_u = W_u exactly.
- Choice: B. Endpoints are stored once in a private Kaggle Dataset (fp16 safetensors, ~1 GB for 0.5B and ~2.5 GB for 1.2B) and held in CPU RAM as fp32 during a session. GPU parameters are overwritten in place per α. For vLLM, each α is materialised to ephemeral disk (`/kaggle/tmp`), used, then deleted — never to `/kaggle/working`.
- Why: Exact endpoints. One process sweeps all α. Keeps persistent output ≤ 20 GB.
- Tradeoff accepted: Endpoints are stored in fp16 while training kept fp32 masters. Documented. Materialisation costs ~20–60 s per α for vLLM.
- Cost impact: CPU RAM ≈ 2 × 4.8 GB fp32 for 1.2B (fits in Kaggle's ~29 GB). Disk: private Dataset ≈ 7 GB for core Tier 2.
- Paper deviation: no.
- Revisit-if: the stretch models make RAM tight (then stream per tensor).

## D-009 Tier 2 models and full-FT recipe
- Date: 2026-09-30
- Context: The brief suggests Qwen2.5-0.5B/1.5B, Llama-3.2-1B, Gemma-3-1B, trained with 8-bit AdamW and gradient checkpointing.
- Options considered: all four; a subset.
- Choice: Core = Qwen2.5-0.5B-Instruct (cheap; also the seeds ablation) + Llama-3.2-1B-Instruct (same family as Tier 1 Llama, which gives a scale contrast). Stretch = Qwen2.5-1.5B-Instruct, Gemma-3-1B-it.
  - Full FT recipe: fp32 master weights + fp16 autocast, `paged_adamw_8bit` (bitsandbytes), gradient checkpointing, lr 2e-5, other hyperparameters per App. C.
  - LoRA uses the same recipe as Tier 1, but without quantisation (fp16 base).
- Why: Both regimes on the same model isolate the regime effect (S1). The 1.2B model peaks at ≈ 1.24B × (4+4+2) B ≈ 12.4 GB plus activations; paged optimizer states spill to CPU if needed.
- Tradeoff accepted: Smaller models than the paper, so Tier 2 results are an analogue, not a replication.
- Cost impact: ≈ 8 session-h training including seeds.
- Paper deviation: yes — App. C models.
- Revisit-if: the Llama-1B full FT OOMs even at per-device batch 1 (then use FSDP over 2 GPUs, logged as a new D).

## D-010 Harmful split = existing WildGuardMix harmful examples only
- Date: 2026-09-30
- Context: The paper's harmful split `wildguard_synth_even_8k` is "an even split of harmful WildGuard examples and synthetic examples" (App. C). The synthetic half is not released and its construction is not described.
- Options considered: A: Generate synthetic harmful data. B: Use only WGM's existing harmful examples.
- Choice: B. We never generate new harmful content. Select from `wildguardtrain` where `prompt_harm_label == "harmful"`, `response_harm_label == "harmful"`, `response_refusal_label == "compliance"`, and `response` is non-null. Take a seeded (seed 0) sample of 8,000. If fewer exist, use all and record the count here. Save the subcategory histogram.
- Why: Brief constraint, and ethics (paper Ethical Considerations).
- Tradeoff accepted: Our "harmful" endpoint may be less (or differently) harmful than the paper's, which shifts the spectrum's range.
- Cost impact: none.
- Paper deviation: yes (reconstruction) — App. C datasets.
- Revisit-if: the authors release `wildguard_synth_even_8k` (M8), or the count is < 4,000.

## D-011 Unharmful split definition `[unspecified in paper]`
- Date: 2026-09-30
- Context: App. C says `wildguard_unharmful` consists "of unharmful examples only", but does not define "unharmful".
- Options considered: A: response unharmful (includes refusals to harmful prompts). B: prompt unharmful and response unharmful.
- Choice: B, seeded sample of 8,000 with non-null response.
- Why: The cleanest "benign" endpoint. Avoids teaching refusal behaviour, which could itself move ΔB.
- Tradeoff accepted: May differ from the authors' filter.
- Cost impact: none.
- Paper deviation: yes (reconstruction) — App. C.
- Revisit-if: M8 shows a different filter.

## D-012 Target groups and target sentences (reconstruction)
- Date: 2026-09-30
- Context: App. F: 50 target sentences per group, average 7 words, from DT stereotype groups inserted into templates. T2 lists the 24 groups. Templates are not released beyond four examples.
- Options considered: A: Generate with an external LLM. B: Author 50 templates in-repo (the coding agent is itself an LLM, analogous to the paper's GPT-5 use), then validate.
- Choice: B. 50 neutral (non-valenced) templates, including the four App. F examples verbatim. Group surface forms are exactly the T2 names ("Black people", "Physically disabled people", …). Frozen at `src/rbbd/resources/sentences/targets/base.txt` with template IDs.
- Why: No network or API needed. Fully reviewable in git. Hash-stable.
- Tradeoff accepted: The templates are not the authors'.
- Cost impact: none (CPU, M1).
- Paper deviation: yes (reconstruction) — App. F.
- Revisit-if: M8 releases the originals.

## D-013 Positive and negative attribute sets (reconstruction)
- Date: 2026-09-30
- Context: App. F: 100 positive and 100 negative sentences, average 7 words, created with GPT-5. Examples use he/she/they subjects.
- Options considered: as D-012.
- Choice: Authored in-repo by the coding agent. Include the App. F examples verbatim. Mix subjects (he, she, they, the person, names), because App. E.1 subject variants presuppose them. Check: no group terms, no duplicates, mean length 7 ± 1 words, P/N word-count distributions matched (KS p > 0.05).
- Why: As D-012.
- Tradeoff accepted: Valence strength may differ from the authors'.
- Cost impact: none.
- Paper deviation: yes (reconstruction) — App. F.
- Revisit-if: M8.

## D-014 Neutral in-domain anchors (reconstruction)
- Date: 2026-09-30
- Context: §4.1 and App. F: 1,000 neutral sentences "drawn from the same social-group domain as the target sentences", average 7 words, e.g. "Many Asians participate in online discussions." §4.5 notes anchors are never scored.
- Options considered: A: Reuse target templates. B: Disjoint templates, balanced over the 24 groups.
- Choice: B. ≈42 anchors per group (1,000 total, balanced by rounding with a fixed group order). Templates are disjoint from the target templates (enforced by template ID). No exact overlap with T/P/N.
- Why: Prevents anchors from trivially aligning with targets. Matches the paper's description.
- Tradeoff accepted: The anchor distribution is unknown in the paper.
- Cost impact: none.
- Paper deviation: yes (reconstruction) — App. F.
- Revisit-if: M8.

## D-015 Anchor ablation sources and sizes (reconstruction)
- Date: 2026-09-30
- Context: §4.5 / F4 compare word anchors (original RR), Alpaca, Tulu and neutral sentences across numbers of anchors. The exact x-axis values are not stated in the text (F4 is an image); "1k anchors" is the maximum.
- Options considered: sizes {100, 500, 1000} vs {50, 100, 250, 500, 1000}. Word source = generic lexicon vs corpus-frequency words.
- Choice: sizes {50, 100, 250, 500, 1000}, nested seeded subsets, 5 subset seeds for m < 1000.
  - Word anchors = top-1,000 frequent lowercase alphabetic words from the Alpaca+Tulu samples, excluding stop-words, group terms and the P/N lexicon.
  - Alpaca/Tulu = seeded samples of single-sentence user instructions of 5–15 words.
- Why: Deterministic, no extra dependency, covers the F4 range.
- Tradeoff accepted: F4 sizes may differ.
- Cost impact: ≈ 3,000 extra short sentences per checkpoint in the one-pass extraction (E1), ≈ +20 s at 8B.
- Paper deviation: yes (reconstruction) — §4.5, F4.
- Revisit-if: M8.

## D-016 Attribute and target ablation variants (reconstruction)
- Date: 2026-09-30
- Context: App. E.1/E.2/T4 define 6 attribute variants (base, subject v1 they/their, subject v2 the person/people, synonyms v1–v3) and 6 target variants (base, passive, passive rephr., synonyms v1–v3). P and N are always modified jointly.
- Options considered: rule-based transforms vs authored variants.
- Choice: Authored by the coding agent, sentence-aligned to base, validated by the checks in PLAN §2.
- Why: Passive rephrasing and natural synonyms need judgment. Rules would produce ungrammatical text.
- Tradeoff accepted: Variant strength is unmeasured against the paper's.
- Cost impact: ≈ 7,000 extra sentences in the one-pass extraction, ≈ +1 min at 8B.
- Paper deviation: yes (reconstruction) — App. E.
- Revisit-if: M8.

## D-017 "Final layer" = HF `hidden_states[L]` (post final norm) `[unspecified in paper]`
- Date: 2026-09-30
- Context: §4.1: "final transformer layer, 32 for Llama and Mistral and 34 for Gemma". Whether this is before or after the final RMSNorm is not stated. In HF, `hidden_states[L]` of an L-layer model is returned after the final norm.
- Options considered: A: post-norm (`last_hidden_state`). B: pre-norm (output of block L).
- Choice: A as default, obtained from the base model's `last_hidden_state` (E2). B is captured by a hook in the same pass as an ablation (M6-T4). The layer index is asserted == `config.num_hidden_layers` (Gemma-3-4B text config: 34).
- Why: Most likely what an `output_hidden_states=True, [32]` implementation returns. Cheap to ablate.
- Tradeoff accepted: May differ from the authors'.
- Cost impact: negligible.
- Paper deviation: no (interpretation), flagged `[unspecified]`.
- Revisit-if: the pre/post-norm ablation changes AUC by > 0.05, or M8.

## D-018 Pooling and tokenisation for embeddings `[unspecified in paper]`
- Date: 2026-09-30
- Context: §3.1: "averaging the final hidden-state vectors across all tokens produced by the model". Masking, special tokens and chat templating are not stated.
- Options considered: include or exclude BOS; raw text or chat template; left or right padding.
- Choice:
  - Raw sentence text, tokenizer default `add_special_tokens=True` (BOS included), **no chat template**.
  - Right padding. Mask-aware mean over all non-pad tokens (BOS included), accumulated in fp32.
  - `max` pooling masks pads to −inf. `last` = the last non-pad token.
  - The pad token is set to EOS if missing (Llama/Mistral).
- Why: The most literal reading of §3.1. Right padding avoids position-id subtleties.
- Tradeoff accepted: BOS attention-sink effects are included. An exclude-BOS ablation is cheap and is added in M6 if the budget allows.
- Cost impact: none.
- Paper deviation: no (interpretation), flagged `[unspecified]`.
- Revisit-if: M8, or the exclude-BOS ablation changes AUC by > 0.05.

## D-019 Embedding cache: fp16 safetensors keyed by content
- Date: 2026-09-30
- Context: The brief's efficiency requirement. Many ablations reuse the same embeddings.
- Options considered: npy per set; a single HDF5; safetensors per (checkpoint, sentence-union).
- Choice: One safetensors file per (model, checkpoint, extraction config, sentence-union hash) holding tensors `mean`, `max`, `last` (and `mean_prenorm` when enabled), each [n_sentences, d] in fp16, plus a sidecar JSON with `sentence_hashes` (row order), key fields and the manifest.
  - Key = SHA-256 of canonical JSON {model_id, revision, checkpoint_spec (regime, adapter hashes, α), layer, layer_site, precision, quant, placement, tokenizer settings, sentence-union hash, schema_version}, truncated to 16 hex.
  - The metric layer selects subsets by sentence hash.
- Why: Content-addressed means stale data is never reused, and invalidation is a version bump rather than a delete.
- Tradeoff accepted: fp16 storage rounding (≤ 1e-3 relative). Metrics compute in fp32.
- Cost impact: 8B: 12.4k × 4096 × 2 B × 3 poolings ≈ 305 MB/ckpt → 24 ckpts ≈ 7.3 GB. Too much for 20 GB alongside everything else, so `max`/`last` are stored only for the Tier 1 Llama and Tier 2 runs (pooling ablation scope), which brings Tier 1 to ≈ 3.5 GB.
- Paper deviation: no.
- Revisit-if: disk pressure (B-008).

## D-020 One extraction pass per checkpoint over the union of all sentence sets
- Date: 2026-09-30
- Context: E1–E3.
- Options considered: one pass per set; one pass over the union.
- Choice: Union, deduplicated, sorted by token length, batched by a token budget (default 16k tokens per batch at 8B fp16 on 2 T4s; tuned in M2). Pooling variants are computed in the same pass.
- Why: Minimal padding and forward passes.
- Tradeoff accepted: One larger cache file per checkpoint.
- Cost impact: Target ≤ 15 min per 8B checkpoint including ablation sets (DC-12). The paper's comparable number is ~3 min on A100 (T3).
- Paper deviation: no.
- Revisit-if: DC-12 fails.

## D-021 Generation engine: vLLM on 2× T4, HF fallback
- Date: 2026-09-30
- Context: Benchmarks dominate cost (T3: up to 156 min per checkpoint on A100).
- Options considered: HF `generate` (simple, slow); vLLM (fast, T4 support caveats); TGI.
- Choice: vLLM pinned (D-031) with fp16 (Gemma fp32), TP=2 for ≥ 4B and TP=1 per GPU for Tier 2 (two jobs in parallel), LoRA enabled (`max_lora_rank=32`). Behind a config flag, an HF-generate fallback using the same sampling parameters.
- Why: 5–20× throughput over HF on batched sampling.
- Tradeoff accepted: Turing support is fragile (B-005). Sampling RNG differs from HF, so generation is seeded but not bit-reproducible across engines.
- Cost impact: See PLAN §8 (≈ 26 h benches total).
- Paper deviation: no (the paper does not name its engine).
- Revisit-if: the M0-T6 probe fails.

## D-022 DecodingTrust stereotype: port the pipeline; n=3 samples per prompt; all 3 system prompts
- Date: 2026-09-30
- Context: App. C: "modified the DecodingTrust repository… updating packages… used the repository's stereotype evaluation pipeline". Settings are `[unspecified in paper]`. DT's stereotype data = 16 stereotype topics × 3 variants × 24 groups = 1,152 user prompts. DT uses 3 system prompts (benign, untargeted, targeted) and by default 25 samples per prompt. The agreementIndex per group is the fraction of "I agree" responses.
- Options considered: A: Vendor and patch the DT repo (heavy, OpenAI-style client). B: Port the prompt templates, system prompts and agreement parser at a pinned commit, and generate with our vLLM path.
- Choice: B.
  - n=3 samples per prompt, T=0.7, top-p 1.0 (same as the WGM settings), max 150 new tokens.
  - BiasScore(group) = mean agreementIndex over the 3 system prompts.
  - Parity test: our parser vs DT's original on 200 canned responses gives identical labels.
  - The pinned DT commit SHA is recorded here in M5-T1.
- Why: Avoids DT's dependency tree. Keeps generation on one engine. n=3 gives 432 samples per group (48 prompts × 3 sys × 3), SE ≈ 0.024 on a proportion.
- Tradeoff accepted: Noisier than n=25; bootstrap CIs make this visible. The system-prompt aggregation may differ from the authors'.
- Cost impact: ≈ 10.4k generations per checkpoint → ≈ 20 min per 8B checkpoint vs ≈ 170 min at n=25.
- Paper deviation: yes (reconstruction + subsampling) — App. C.
- Revisit-if: CIs on ΔBiasScore are wider than 0.1 (then n=5 for Tier 1 Llama at least), or M8.

## D-023 WildGuardMix output benchmark
- Date: 2026-09-30
- Context: §4.1: the SOCIAL STEREOTYPES AND UNFAIR DISCRIMINATION subcategory of the test set, scored by `allenai/wildguard`. App. C: 5 responses/prompt, T=0.7, top-p 1.0, 256 new tokens.
- Options considered: subsample vs full.
- Choice: Full subcategory (no subsampling; it is small). Generate with the chat template, per App. C settings. WildGuard runs via vLLM (TP=2 fp16, greedy) with the official prompt format; parse "Harmful response: yes/no" (unparseable = excluded and counted). BiasScore(topic) = harmful-response rate.
- Why: Cheap enough to keep exact.
- Tradeoff accepted: none.
- Cost impact: ≈ 5 min gen + ≈ 1 min classify per 8B checkpoint.
- Paper deviation: no.
- Revisit-if: the subcategory has fewer than 20 prompts in some topic (reported).

## D-024 WGM topic mapping via a local open model; group-level pairing `[partly unspecified]` — superseded by D-043
- Date: 2026-09-30
- Context: App. B: ChatGPT 5.2 maps each WGM SOCIAL STEREOTYPES prompt to one of 9 topics (T2). How topic-level scores pair with the group-level ΔB is `[unspecified in paper]`. §3.4 says "each pairing of a checkpoint with a target group is one observation".
- Options considered:
  - Mapper: external API (network, key, cost) vs local open LLM vs keyword rules.
  - Pairing: each group inherits its topic's ΔBiasScore (24 obs per checkpoint) vs topic-mean ΔB (9 obs).
- Choice:
  - Mapper: `Qwen/Qwen2.5-7B-Instruct` via vLLM, greedy, fixed prompt listing the 9 topics with their member groups plus "none". Run once; freeze `resources/topic_map.csv` (prompt SHA-256 → topic). "none" prompts are excluded and counted. A 30-item human spot check records agreement here (target ≥ 80%).
  - Pairing: group inherits its topic's score (primary). Topic-mean ΔB is a sensitivity analysis.
- Why: No external calls. Deterministic. The primary pairing follows §3.4 literally.
- Tradeoff accepted: The mapper differs from the paper's. Group-level pairing repeats topic scores across the groups in a topic, so observations are not independent. Bootstrap CIs resample checkpoints × topics to respect this.
- Cost impact: ≈ 0.3 session-h once.
- Paper deviation: yes (reconstruction) — App. B.
- Revisit-if: spot-check agreement < 80%, or the user supplies an API (Q4), or M8.

## D-025 ToxiGen: 9-group map, prompt subsample, first-statement rule `[partly unspecified]`
- Date: 2026-09-30
- Context: §4.1: "nine groups that have a counterpart in our target sets". App. C: per-group few-shot hate prompts as a user turn in the chat template, five continuations, first generated statement scored by `toxigen_roberta`, fraction toxic per group. The number of prompts and the max tokens are `[unspecified]`.
- Options considered: all prompts vs a seeded subsample; 256 vs 128 max tokens.
- Choice:
  - Group map (ToxiGen → DT): asian→Asians, black→Black people, jewish→Jews, latino→Latin Americans, lgbtq→Homosexuals, mexican→Mexicans, muslim→Muslims, physical_dis→Physically disabled people, women→Women. Excluded: chinese, mental_dis, middle_east, native_american. Exact file names are verified at the pinned ToxiGen commit in M5-T1.
  - 100 seeded prompts per group × 5 continuations, T=0.7, top-p 1.0, max 128 new tokens.
  - First statement = first non-empty line after stripping list markers and quotes.
  - Toxic iff the classifier's toxic-class probability > 0.5.
- Why: Budget. The first statement rarely needs > 128 tokens.
- Tradeoff accepted: Sampling noise (bootstrap CI). Preambles such as "Sure, here are…" can be the first line; counted as the model's statement, which the paper's rule also implies.
- Cost impact: 4.5k generations per checkpoint ≈ 8 min at 8B (vs T3's 33–75 min on A100 for the full set).
- Paper deviation: yes (subsampling + reconstruction) — App. C.
- Revisit-if: the per-group CI half-width > 0.03 (the threshold), or M8.

## D-026 Subsampling policy and uncertainty reporting
- Date: 2026-09-30
- Context: The brief requires subsampling where full sets do not fit, with bootstrap CIs.
- Options considered: ad-hoc vs policy.
- Choice:
  - All subsamples are seeded (seed 0) and stored as ID lists with a hash in the manifest. The same subset is used for every checkpoint of a model, so deltas are paired.
  - Report 95% percentile-bootstrap CIs (1,000 resamples) over prompts for BiasScore deltas, and over observations for r and AUC.
  - The report lists subsample sizes next to each number.
- Why: Makes the subsampling visible and keeps deltas paired.
- Tradeoff accepted: Wider CIs than the paper.
- Cost impact: CPU only.
- Paper deviation: yes (subsampling) — App. C.
- Revisit-if: budget frees up.

## D-027 Statistics definitions `[partly unspecified]`
- Date: 2026-09-30
- Context: §4.1 / T1: pooled Pearson with two-tailed p, ROC AUC of a threshold classifier on ΔB, MAE of a linear fit as the mean over 1,000 bootstrap resamples. Thresholds: 0.1 (WGM, DT), 0.03 (ToxiGen).
- Options considered: include the base as an observation or not; in-sample vs OOB MAE.
- Choice:
  - Observations = 7 audited checkpoints × groups (24 for WGM/DT, 9 for ToxiGen). The base is excluded (it is the reference, ΔB ≡ 0).
  - Label = ΔScore > threshold. Score = −ΔB. AUC via `sklearn.metrics.roc_auc_score`.
  - MAE = mean over 1,000 bootstrap resamples of the in-sample MAE of an OLS fit ΔScore ~ ΔB (primary); OOB MAE is reported alongside.
  - CKA: AUC = max(AUC, 1−AUC), r reported as |r| (T7).
  - Seeds: bootstrap seed 0.
- Why: The most literal reading.
- Tradeoff accepted: Ambiguity recorded.
- Cost impact: CPU only.
- Paper deviation: no (interpretation), flagged.
- Revisit-if: M8.

## D-028 Baseline definitions `[partly unspecified]`
- Date: 2026-09-30
- Context: §4.5: SEAT (Eq. 2–3) in each model's own space. Procrustes-SEAT aligns audited → reference with an optimal orthogonal map and scores audited targets against the reference attribute sets. CKA drift = 1 − CKA between reference and audited target representations. The fit set and CKA variant are unspecified.
- Options considered: Procrustes fit on anchors vs on all sentences. Linear vs RBF CKA.
- Choice:
  - Procrustes is fitted on the 1,000 neutral anchors (paired across models; never scored), with centring off (to preserve cosine semantics).
  - CKA is linear, computed per group on the 50 × d target matrices, giving one drift value per (checkpoint, group) observation.
  - SEAT ΔB = B_aud − B_ref with cosine on absolute embeddings.
- Why: Anchors are the shared, non-scored sentences, which mirrors the RR construction. Linear CKA is standard and cheap.
- Tradeoff accepted: Choices may differ from the authors'.
- Cost impact: CPU only.
- Paper deviation: no (interpretation), flagged.
- Revisit-if: M8.

## D-029 Artifact store and persistence — superseded by D-041
- Date: 2026-09-30
- Context: Only `/kaggle/working` (~20 GB) persists after a session. Harmful-trained artifacts must stay private.
- Options considered: A: private Kaggle Datasets. B: private HF repos. C: notebook outputs chained as inputs.
- Choice: A (pending Q2).
  - `rbbd-artifacts` (private, versioned): adapters, embeddings, generations, scores, manifests.
  - `rbbd-ckpt-private` (private): Tier 2 full-FT endpoints.
  - Staging in `/kaggle/working/artifacts`. HF cache in `/kaggle/tmp`.
  - At session end: `kaggle datasets version` with a message containing the git SHA and stage list. The next session mounts the latest version read-only and copies what it needs.
  - Base weights come from Kaggle Models mounts when a checksum-verified mirror exists (E10).
- Why: Versioned (needed by ROLLBACK), private, and not counted against the 20 GB output.
- Tradeoff accepted: Needs Kaggle API credentials as Kaggle Secrets. Dataset versioning takes minutes.
- Cost impact: ≈ 2–5 min per session for versioning.
- Paper deviation: no.
- Revisit-if: Q2 answer is HF, or dataset versioning is too slow.

## D-030 Stage runner with manifests and resumable partial progress
- Date: 2026-09-30
- Context: The 12 h cutoff. The brief requires skip-on-valid-manifest and partial checkpointing.
- Options considered: Make-like DAG tool (Snakemake/DVC) vs a small in-package runner.
- Choice: In-package runner (`runner.py`). Each stage declares inputs and outputs. `manifest.json` = {stage, config_hash, git_sha, input_hashes, output_hashes, started, finished, seconds, env_summary, complete}. Training resumes from `checkpoint-*`. Generation writes shards plus a done-list.
- Why: No extra tooling (brief: no extra abstractions). Kaggle-friendly.
- Tradeoff accepted: Less general than DVC.
- Cost impact: negligible.
- Paper deviation: no.
- Revisit-if: the stage graph grows beyond ~12 stages.

## D-031 Version pins (starting point, verified in M0)
- Date: 2026-09-30
- Context: Paper stack: Python 3.9, PyTorch 2.7.1, Transformers 4.57.3, PEFT 0.18.0, TRL 0.25.1 (App. C). Kaggle's image ships its own torch/Python. T4 = sm75, which PyTorch 2.7 CUDA 12.6 wheels support.
- Options considered: Use Kaggle's preinstalled torch vs pin 2.7.1. One environment vs separate train/bench extras.
- Choice: `pyproject.toml` pins:
  - Core: torch 2.7.1, transformers 4.57.3, peft 0.18.0, trl 0.25.1, accelerate 1.10.1, datasets 4.1.1, safetensors 0.6.2, huggingface_hub 0.35.3, bitsandbytes 0.47.0, numpy 2.2.6, scipy 1.15.3, scikit-learn 1.7.2, pandas 2.3.3, matplotlib 3.10.6, pyyaml 6.0.2.
  - Bench: vllm 0.10.1.1 (built against torch 2.7.1).
  - Dev: pytest 8.4.2, ruff 0.13.3.
  - Python ≥ 3.10.
  - **These pins were not resolved during planning** (no installs or network allowed). M0-T4 resolves them on the Kaggle image. Any change gets a superseding D entry. If vLLM conflicts with the training stack, the `bench` stage runs in a separate venv.
- Why: Stay on the paper's versions where possible. vLLM 0.10.1.x is the last line built on torch 2.7.1 that still falls back to the V0 engine on Turing (to be verified).
- Tradeoff accepted: Reinstalling torch on Kaggle costs ~3–5 min per session.
- Cost impact: +≈ 5 min per session setup.
- Paper deviation: yes — Python version (App. C). Library adds.
- Revisit-if: M0-T4/T6 fail.

## D-032 Package layout
- Date: 2026-09-30
- Context: The brief suggests modules data/models/embed/metrics/finetune/bench/analysis/utils.
- Options considered: as suggested vs with small adjustments.
- Choice: As suggested, plus `config.py` and `runner.py` at the package root, `resources/` for frozen sentence sets and the topic map, `models/spectrum.py` for the α list, and `bench/topic_map.py`. One CLI (`python -m rbbd.cli`).
- Why: Single responsibility per module. Sentence-set hashes travel with git tags.
- Tradeoff accepted: none.
- Cost impact: none.
- Paper deviation: no.
- Revisit-if: never, unless the shape changes (update ARCHITECTURE.md).

## D-033 Seeds ablation on Tier 2 only
- Date: 2026-09-30
- Context: App. E.5 repeats Llama fine-tuning with 3 seeds (std 0.003).
- Options considered: Tier 1 Llama (+4 runs ≈ 15 h) vs Tier 2 Qwen2.5-0.5B full FT (+4 runs ≈ 2 h).
- Choice: Tier 2, seeds {0, 1, 2} × {u, h}.
- Why: Budget.
- Tradeoff accepted: A different model and regime from App. E.5.
- Cost impact: ≈ 2 session-h.
- Paper deviation: yes — App. E.5.
- Revisit-if: budget remains after W4.

## D-034 Tier 1 regime = LoRA only
- Date: 2026-09-30
- Context: D-001.
- Options considered: n/a.
- Choice: The T1 "Full fine-tuning" columns are populated only from Tier 2 models and labelled as such.
- Why: Infeasible otherwise.
- Tradeoff accepted: No direct comparison with the paper's full-FT numbers.
- Cost impact: saves an infeasible amount.
- Paper deviation: yes — T1.
- Revisit-if: bigger hardware.

## D-035 Budget cut order — superseded by D-042
- Date: 2026-09-30
- Context: The brief requires a cut order.
- Options considered: n/a.
- Choice: As PLAN §8:
  1. Stretch models.
  2. GPU ablations.
  3. Harder benchmark subsampling (DT first, then ToxiGen).
  4. Tier 1 epochs (needs user approval).
  5. Never the core ΔB-vs-benchmark correlation or SEAT.
- Why: Protects the primary hypothesis test.
- Tradeoff accepted: Ablation coverage.
- Cost impact: up to ≈ 15 h recoverable.
- Paper deviation: no.
- Revisit-if: quota changes.

## D-036 Secrets handling
- Date: 2026-09-30
- Context: Gated models and datasets need `HF_TOKEN`. Dataset versioning needs Kaggle API credentials.
- Options considered: n/a.
- Choice: Read from Kaggle Secrets into environment variables inside the notebook. Never echo them. Redact `hf_*` patterns in the log formatter. Manifests record only `token_present: true/false`. `.gitignore` covers `*.env`, `kaggle.json`.
- Why: Brief constraint.
- Tradeoff accepted: none.
- Cost impact: none.
- Paper deviation: no.
- Revisit-if: never.

## D-037 Privacy of harmful artifacts and outputs
- Date: 2026-09-30
- Context: Ethical Considerations: the authors release code and sentence sets, not checkpoints. The brief says no public Datasets or HF repos for harmful-trained checkpoints.
- Options considered: n/a.
- Choice: Adapters, checkpoints, generations and per-prompt scores go to private stores only. Git and `results/` hold aggregates. DC-18 checks that no generated text or prompt text is committed.
- Why: Safety.
- Tradeoff accepted: Others cannot re-score our generations.
- Cost impact: none.
- Paper deviation: no.
- Revisit-if: never.

## D-038 Tier 0 smoke scope; WildGuard classifier skipped in smoke
- Date: 2026-09-30
- Context: The smoke run must finish in < 15 min on T4 (DONE). WildGuard is a 7B classifier (≈ 14 GB download + load ≈ 5+ min).
- Options considered: include WildGuard; a stub scorer; skip WildGuard scoring in smoke.
- Choice: Smoke runs every stage except WildGuard classification (`bench.wildguard.classify: false` in `smoke.yaml`). The M5-T3 acceptance runs WildGuard on 20 fixed Tier 0 generations as a separate GPU test (`tests/gpu/test_generate_gpu.py`).
- Why: Keeps the smoke run fast while still covering the classifier path elsewhere.
- Tradeoff accepted: Smoke does not exercise WildGuard.
- Cost impact: saves ≈ 6 min per smoke run.
- Paper deviation: no.
- Revisit-if: WildGuard is available from a Kaggle Models mount with fast load.

## D-039 Diff against official code when released
- Date: 2026-09-30
- Context: The paper links a code repository that was not public at planning time.
- Options considered: n/a.
- Choice: Conditional milestone M8. Every difference becomes a D entry (adopt or keep with a reason). Re-run M4/M6 with released sentence sets if any.
- Why: Our reconstructions (D-010–D-018, D-022–D-028) are the largest validity risk (R11).
- Tradeoff accepted: none.
- Cost impact: 0–2 session-h.
- Paper deviation: no.
- Revisit-if: n/a.

## D-040 SFT data format and loss `[unspecified in paper]`
- Date: 2026-09-30
- Context: App. C gives hyperparameters but not the formatting, loss masking or seed.
- Options considered: full-sequence loss vs completion-only; raw vs chat template.
- Choice:
  - Prompt–completion pairs rendered with each model's chat template, loss on the completion (assistant) tokens only (TRL prompt–completion default).
  - Truncation at 1,024 tokens, keeping the prompt head.
  - Training seed 0 (seeds 1, 2 for D-033). `group_by_length=True`.
  - Gemma's chat template has no system role, so no system prompt is used for any model.
- Why: The standard instruct-SFT recipe. Consistent across models.
- Tradeoff accepted: May differ from the authors'.
- Cost impact: none.
- Paper deviation: no (interpretation), flagged.
- Revisit-if: M8.

## D-041 Artifact store = private HF repo `SarthakGupta414/rbbd-artifacts` (supersedes D-029)
- Date: 2026-09-30
- Context: User answer to Q2 (PLAN §11). The repo already exists and is private. A write token scoped to that repo is stored as the Kaggle Secret `HF_TOKEN`; the same secret also passes `auth_check` for every gated model and dataset (Q1).
- Options considered: A: private Kaggle Datasets (D-029). B: the private HF repo above.
- Choice: B.
  - Code uploads only with `HfApi.upload_folder`, into per-stage / per-checkpoint subfolders that mirror the local artifact layout (ARCHITECTURE §3), e.g. `env/<session_id>/`, `embeddings/<model>/<regime>/<ckpt>/`.
  - Code **never** calls `create_repo`, `update_repo_settings`/`update_repo_visibility`, or `push_to_hub`, and never prints the token. A test (`tests/test_store.py`) scans the store module for those calls.
  - Before every upload, `utils/store.py` reads `repo_info` and refuses to upload unless `private is True`.
  - The repo type (model or dataset) is not stated; the store resolves it at runtime by probing `repo_info` for `dataset` then `model`, and records the result in `env.json`.
  - Tier 2 full-FT endpoints go to the same repo under `ckpt_private/…` instead of a separate `rbbd-ckpt-private` Dataset.
  - Local staging stays in `/kaggle/working/artifacts`; the next session downloads what it needs with `hf_hub_download` / `snapshot_download(allow_patterns=…)`.
- Why: The user's choice. One secret covers read (gated models) and write (store). Per-file uploads avoid re-versioning a whole dataset each session.
- Tradeoff accepted: ROLLBACK records the HF repo commit SHA instead of a Kaggle Dataset version. HF storage for private repos counts against the user's HF quota.
- Cost impact: upload time ≈ minutes per session (adapters ≤ 170 MB, embeddings ≤ 300 MB per checkpoint). No Kaggle API credentials needed.
- Paper deviation: no.
- Revisit-if: uploads fail from Kaggle, or the HF storage quota is hit.

## D-042 Budget fallbacks and cut order (supersedes D-035)
- Date: 2026-09-30
- Context: User answers to Q3, Q5, Q6. Weekly quota is 28 session-h (not 30); the current window resets 2026-10-02 (~54 h after the answer).
- Options considered: n/a (user decision).
- Choice:
  - **Tier 1 training fallback (Q3),** applied only if the M3c-T8 throughput probe projects > 9 h per run: first reduce max length 1024 → 512, but **only if ≤ 5% of training examples would be truncated** at 512 (measured by the M1-T8 token census). Only if that is not enough (or not allowed by the 5% rule), drop 3 epochs → 1. Each step is logged as its own paper deviation (App. C) in a new D entry when taken.
  - **Cut order** (apply top-down; stop as soon as the budget fits):
    1. Stretch Tier 2 models.
    2. GPU ablations (NF4-extraction check; seeds reduced to 1 extra seed). Embedding-only ablations stay.
    3. Harder benchmark subsampling: DT n=3 → 2 and benign + targeted system prompts; ToxiGen 100 → 50 prompts/group; Gemma DT to 2 of 3 system prompts.
    4. Tier 1 training fallbacks as above (max len 512, then 1 epoch).
    5. **Never cut** the core ΔB-vs-benchmark correlation for any Tier 0–2 model or benchmark, or the SEAT baseline.
  - **Stretch models (Q6)** start only after the core Tier 0, Tier 1 and Tier 2 results (M4–M6 on the core models) are complete and budget remains.
  - Weekly scheduling uses 28 h with a 2 h reserve (≈ 26 h usable). PLAN §8's W1–W4 plan (≈ 25/26/24/12 h) still fits; W2 is at the limit, so any overrun moves work into W4.
- Why: User instructions.
- Tradeoff accepted: The 512 cut changes the training distribution for long WildGuardMix examples.
- Cost impact: 512 max length roughly halves Tier 1 training time if most examples are short; 1 epoch cuts it by ~3×.
- Paper deviation: yes, when a fallback is taken (App. C max length / epochs).
- Revisit-if: quota changes.

## D-043 Topic mapping validated on ~100 hand-labelled prompts (supersedes D-024)
- Date: 2026-09-30
- Context: User answer to Q4. Otherwise identical to D-024.
- Options considered: 30-item spot check (D-024) vs ~100 hand-labelled prompts.
- Choice: Mapper = local `Qwen/Qwen2.5-7B-Instruct` via vLLM, greedy, fixed prompt listing the 9 topics with their member groups (T2) plus "none"; run once and freeze `resources/topic_map.csv` (prompt SHA-256 → topic). **Validation:** ~100 WGM SOCIAL STEREOTYPES prompts, stratified by predicted topic, are hand-labelled by the user; the mapping is accepted only if agreement ≥ 80%. The hand labels are stored privately (prompt hashes + labels in the HF store), never in git. Pairing (unchanged from D-024): each group inherits its topic's ΔBiasScore (primary); topic-mean ΔB is a sensitivity analysis. Logged as a reconstruction of App. B.
- Why: User instruction; a larger validation set gives a tighter agreement estimate (±8% at 95% vs ±15% at n=30).
- Tradeoff accepted: ~1–2 h of manual labelling by the user in M5.
- Cost impact: ≈ 0.3 session-h once (unchanged).
- Paper deviation: yes (reconstruction) — App. B.
- Revisit-if: agreement < 80% (then revise the prompt or mapper model and re-validate).

## D-044 Quota facts and access confirmation recorded (amends D-002)
- Date: 2026-09-30
- Context: User answers to Q1 and Q5.
- Options considered: n/a.
- Choice: Record as facts: the Kaggle account is phone-verified; all gated licences are accepted and `auth_check` passes with `HF_TOKEN` for Llama-3.1-8B-Instruct, Llama-3.2-1B-Instruct, gemma-3-4b-it, gemma-3-1b-it, Mistral-7B-Instruct-v0.3, allenai/wildguard and allenai/wildguardmix. Weekly GPU quota = 28 session-h. The M0 probe re-verifies access for **every** repo in PLAN §5 (including open ones) and records the result in `env.json`; D-002's hardware values are still to be filled from the probe.
- Why: Keeps the budget and access facts in one place.
- Tradeoff accepted: none.
- Cost impact: none.
- Paper deviation: no.
- Revisit-if: the probe disagrees.
