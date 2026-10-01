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
- M2-T4 extraction probe outcome (2026-10-01): fp16 fails. The guard fired on batch 0 with every value NaN, so Gemma-3-4B embeddings stay fp32 (D-062).

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

## D-021 Generation engine: vLLM on 2× T4, HF fallback — superseded by D-050
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

## D-045 M0 probe semantics: native bf16, token-shaped redaction, null for skipped checks, one-session store round trip
- Date: 2026-09-30
- Context: Implementing M0-T1/T5/T7. Four details the plan left implicit turned out to matter.
- Options considered / Choice:
  1. **bf16.** `torch.cuda.is_bf16_supported()` defaults to `including_emulation=True` in recent torch and can report True on sm75, which would make DC-16's "`bf16_supported: false`" fail on a T4. Choice: `env.json.bf16_supported` = native support = min compute capability ≥ 8.0; torch's emulation-inclusive answer is kept as `torch.bf16_supported_including_emulation` for information.
  2. **Redaction pattern.** A broad `hf_[A-Za-z0-9]{8,}` rule would mangle vLLM log lines (`hf_overrides`) and a plain `grep hf_` would flag them. Choice: redact the real token shape `hf_` + ≥ 30 alphanumerics (HF tokens are `hf_` + 34), GitHub `gh?_`/`github_pat_` shapes, and the literal values of `HF_TOKEN`/`GITHUB_TOKEN`/`KAGGLE_KEY`. DC-16's leak grep is refined to the same token shape (a more precise check, not a weaker one).
  3. **Skipped checks.** `requirements.ok` and `access_all_ok` are `null` (not `true`) when their check is skipped (`--no-require-gpu`, `--skip-access`), so a skipped check can never read as a pass.
  4. **Store round trip.** PLAN M0-T7 said "read it back in a second short session". Choice: read back in the same session with `force_download=True` into a fresh empty cache directory, which proves the bytes came from the remote repo. Saves a Kaggle session.
  5. **Ephemeral dir.** Kaggle's scratch path is not documented consistently; `ephemeral_dir()` takes `$RBBD_EPHEMERAL`, else the first existing of `/kaggle/tmp`, `/kaggle/temp`, else the system temp dir. The probe records which one was used.
- Why: Each avoids a false result in DC-16 or an unnecessary session.
- Tradeoff accepted: none material.
- Cost impact: saves ≈ 0.2 session-h (no second round-trip session).
- Paper deviation: no.
- Revisit-if: the Kaggle probe shows `torch` reporting bf16 natively on T4 (it should not), or a token format change.

## D-046 CLI `sync` and `vllm-case`; `utils/store.py`; notebook clone and install
- Date: 2026-09-30
- Context: The thin-notebook contract (PLAN §3) ends with "sync artifacts to the store", which needs a CLI entry; D-041 needs a store module; the repo's visibility is unknown.
- Options considered: store code inside `utils/env.py` vs its own module; sync inside `run` vs a separate command.
- Choice:
  - New module `rbbd.utils.store` (single responsibility: the private HF store). ARCHITECTURE updated.
  - `python -m rbbd.cli sync --path <rel>` uploads `<artifacts>/<rel>` to the same path in the store. `vllm-case` is an internal subcommand used by `probe --vllm` to run each case in a fresh process.
  - Notebooks fetch `REF` (branch, tag or commit SHA) with `git fetch --depth 1`, so M0 can pin a commit before any `m0-green` tag exists. If the optional `GITHUB_TOKEN` secret is attached, it is passed as an HTTP header through `GIT_CONFIG_*` environment variables, never in the URL, argv or output.
  - `00_probe.ipynb` installs `.[train,bench,dev]` in one resolve to test that the two stacks co-install (M0-T4, R14), recording `pip freeze` before and after and `pip check`.
- Why: Keeps notebooks free of logic; works for a public or private repo.
- Tradeoff accepted: the notebook clone cell carries ~15 lines of git glue.
- Cost impact: none.
- Paper deviation: no.
- Revisit-if: `pip check` shows a train/bench conflict (then separate venv per D-031).

## D-047 vLLM probe design (M0-T6)
- Date: 2026-09-30
- Context: B-005 lists the ways vLLM can fail on Turing. The probe must find out which apply before M5 depends on vLLM.
- Options considered: probe inside one Python process vs one process per case.
- Choice: `configs/base.yaml › probe.vllm_cases`, each run by `python -m rbbd.cli vllm-case` in its own process (vLLM does not reliably free GPU memory in-process, and a refused dtype must not abort later cases):
  1. Qwen2.5-0.5B-Instruct, TP=1, fp16, with a random rank-32 LoRA (rank 2r = 32 is what D-007's α-adapters need; `max_lora_rank=32`).
  2. Same with TP=2 (tests NCCL across both T4s).
  3. Gemma-3-1B-it, TP=1, fp16 (expected to be refused; outcome recorded either way).
  4. Gemma-3-1B-it, TP=1, fp32 (the D-005 fallback).
  Settings: `max_model_len=512`, `enforce_eager=True` (skips CUDA-graph capture to shorten start-up), `gpu_memory_utilization=0.85`, 4 benign prompts, 16 new tokens, seed 0. The random LoRA's B matrices are scaled by 1e-2 so fp16 outputs stay finite. Engine class and log lines naming the engine or attention backend are stored in `vllm_probe.json`.
- Why: Covers every B-005 failure mode relevant to M5 in ≈ 15 min.
- Tradeoff accepted: `enforce_eager` differs from M5 production settings (M5 re-measures throughput).
- Cost impact: ≈ 0.25 session-h inside the M0 1.0 h budget.
- Paper deviation: no.
- Revisit-if: TP=2 hangs (then M5 uses TP=1 per GPU for all models ≤ 4B and HF fallback for 7–8B).

## D-048 Observed Kaggle environment (S01, 2026-10-01) and a diagnostic vLLM probe rerun (amends D-002, D-005, D-047)
- Date: 2026-10-01
- Context: First M0 run of `notebooks/00_probe.ipynb` on Kaggle (session `20261001T091905Z`, commit `93dbe07`). D-002 asked for observed values; the vLLM probe produced a failure that the original probe could not localise.
- Observed (facts, not choices):
  - GPUs: 2× Tesla T4, compute capability 7.5, 15,360 MiB each. `bf16_supported` (native) = false; torch's emulation-inclusive flag = **true**, confirming D-045's choice.
  - torch 2.7.1+cu126 installed over Kaggle's image; CUDA available, 2 devices.
  - RAM 31 GiB total, 29 GiB available. `/kaggle/working` 20 GB. No `/kaggle/tmp` or `/kaggle/temp`: `ephemeral_dir()` = `/tmp` on the root overlay (1.1 TB free on the host overlay; the per-session usable limit is not known).
  - All 13 repos in PLAN §5 pass `auth_check` with `HF_TOKEN`. The store `SarthakGupta414/rbbd-artifacts` resolves as a **model** repo, is private, and the 1 MB round trip matched (store commit `594c0470b4d638349cdf9f8a6d6bbaf6fc3b4f45`).
  - vLLM 0.10.1.1 falls back to the **V0** engine ("Compute Capability < 8.0 is not supported by the V1 Engine") with the **XFormers** attention backend.
  - Qwen2.5-0.5B with LoRA (rank 32, TP=1 and TP=2) failed during engine construction with `RuntimeError: PassManager::run failed`, a Triton compiler error (B-005). No no-LoRA Qwen case existed, so the cause was not isolated.
  - Gemma-3-1B-it in **fp16 was accepted** by vLLM 0.10.1.1 (D-005 expected a refusal) and produced tokens; numerical soundness unverified because decoding was sampled.
- Options considered: (A) decide the LoRA-generation path now; (B) one ~10-minute diagnostic rerun first.
- Choice: B. The probe now records the failing stage (`lora_build`, `construct`, `generate_base`, `generate_lora`) and a redacted traceback tail; decoding is greedy and texts are kept; cases added: Qwen TP=1 fp16 **no LoRA** and **LoRA rank 16**. D-005 (Gemma fp32 for inference) stays in force until the greedy fp16 vs fp32 comparison is in.
- Why: The LoRA-generation path (D-007's vLLM multi-LoRA, E8) is an architectural choice; it should rest on a localised failure, not a guess.
- Tradeoff accepted: one more short GPU session before `m0-green`.
- Cost impact: ≈ 0.2 session-h (6 cases × ~1 min + install).
- Paper deviation: no.
- Revisit-if: the rerun shows the no-LoRA Qwen case also failing (then the problem is not LoRA-specific).

## D-049 Kaggle S02 diagnostic results: vLLM LoRA is unusable on T4; Gemma fp16 numerically plausible; pins co-install (amends D-048)
- Date: 2026-10-01
- Context: Diagnostic rerun of `notebooks/00_probe.ipynb` (session `20261001T094200Z`, commit `c5c554b`), per D-048.
- Observed (facts):
  - Qwen2.5-0.5B **without LoRA** (TP=1, fp16) builds and generates coherent greedy text ("Paris. It is the largest city…").
  - **Every LoRA case fails** (rank 16 and 32, TP=1 and TP=2) in `LLM(...)` with `RuntimeError: PassManager::run failed`, raised from `triton/backends/nvidia/compiler.py::make_llir` (`pm.run(mod)`), i.e. Triton cannot lower vLLM's LoRA kernels to LLVM IR for sm75. The failure is rank-independent. vLLM 0.10.1.1 is built on torch 2.7.1, which pins its Triton version, so this cannot be fixed within the current pins.
  - Gemma-3-1B greedy: fp16 and fp32 agree on 3 of 4 prompts and diverge after ~8 tokens on the fourth ("vast, blue ocean" vs "vast, shimmering ocean"); fp16 output is coherent, with no sign of overflow garbage.
  - `pip check`: every conflict is between packages preinstalled on the Kaggle image (ydata-profiling, gradio, google-colab, google-adk, bigframes, dopamine-rl, moviepy) and versions in our resolve (matplotlib 3.10.6, pyyaml 6.0.2, pandas 2.3.3, starlette). None involves torch, vLLM, transformers, PEFT, TRL, accelerate or bitsandbytes, and none of those packages is used here. The train and bench extras co-install, so no separate venv is needed (D-031 stands).
  - Kaggle runs Python 3.12 (`/usr/local/lib/python3.12`). On Kaggle: `ruff check` → `All checks passed!`; `pytest -q -m "not gpu"` → `52 passed, 7 deselected in 2.63s`.
  - Store round trip passed again (commit `494d159353cebd5a3fafff21583da49d7ba4d787`).
- Options considered: n/a (facts). The resulting choice of how M5 generates α-checkpoints without vLLM LoRA is a separate decision put to the user.
- Choice: Record the facts. D-005 (Gemma fp32 for inference) stays: one divergent prompt out of four on the 1B model is not the "match within tolerance" its Revisit-if requires; the 4B fp16 check in M2-T4 decides it.
- Why: Evidence for the pending generation-path decision (B-005).
- Tradeoff accepted: none.
- Cost impact: S02 ≈ 0.3 session-h.
- Paper deviation: no.
- Revisit-if: a vLLM/Triton release fixes sm75 LoRA lowering.

## D-050 Generate α-checkpoints from merged fp16 weights on ephemeral disk, not vLLM LoRA (supersedes D-021; amends D-007, PLAN E8)
- Date: 2026-10-01
- Context: vLLM's LoRA kernels cannot be compiled by Triton for sm75 (D-049, B-005), so D-021/E8's "one engine serves base + 7 α-adapters as LoRA requests" is impossible on Kaggle T4s. User decision (2026-10-01) among three options.
- Options considered:
  - A: For each checkpoint, compute merged weights, save to ephemeral disk, serve with vLLM without LoRA, delete the copy.
  - B: Older vLLM/torch/Triton pins that might compile LoRA on sm75 (moves torch off the paper's 2.7.1; uncertain).
  - C: HF `generate` with PEFT adapters (5–10× slower; Tier 1 benches would exceed 100 session-h).
- Choice: A (user's choice).
  - **LoRA regimes (Tier 0, Tier 1, Tier 2 LoRA):** for α ∈ {1.0, 0.9, 0.7, 0.5, 0.3, 0.1, 0.0}, compute W = W₀ + α·ΔW_u + (1−α)·ΔW_h per target module in **fp32** (ΔW_x = s_x·B_x·A_x, D-007), cast to fp16, write a plain checkpoint to `ephemeral_dir()` (`/tmp` on Kaggle, D-048), start vLLM on it without LoRA, generate every benchmark for that checkpoint, shut the engine down, and **delete the temporary checkpoint** (the user approved this deletion as part of the path). The reference is the downloaded base model. At most one merged copy exists at a time.
  - **Full-FT regime (Tier 2):** unchanged from D-008 (materialise (1−α)W_h + αW_u to ephemeral disk), now the same code path.
  - Gemma: merged copies are written in the dtype D-005 prescribes for inference (fp32 until revisited).
  - **Embeddings are unchanged:** HF + PEFT with the exact concatenated rank-2r adapter on the fp16 base (D-004, D-007). Generation and embeddings therefore apply the same ΔW; they differ only by where fp16 rounding happens (merged weight vs adapter output).
  - Generation order per session: all benchmarks for one checkpoint before moving to the next, so each merge and engine start is paid once per checkpoint.
  - The vLLM-LoRA case stays in the M0 probe as a canary (`expect_fail: true`) so a future vLLM/Triton that fixes sm75 is noticed.
- Why: Exact maths, no pin changes, and the cost is bounded (one engine start per checkpoint).
- Tradeoff accepted: 7 engine starts + merges per model and regime instead of 1. Merged checkpoints never persist, so generation cannot be re-run without re-merging.
- Cost impact: per 8B checkpoint ≈ 1–2 min merge + write (16 GB to `/tmp`) and ≈ 2–3 min engine start, ≈ +3–5 min each → ≈ +1.5–2 session-h across Tier 1 and ≈ +0.5 h across Tier 2. PLAN §8 contingency (≈ 7 h for Tier 1) absorbs it. Ephemeral disk peak ≈ 16 GB (Llama/Mistral fp16) or ≈ 17 GB (Gemma-3-4B fp32).
- Paper deviation: no (the paper does not name its generation engine).
- Revisit-if: the LoRA canary starts passing, or the per-session ephemeral disk limit turns out to be below ~20 GB.

## D-051 M0 closed: S03 verifies the merged-weights path (D-050) and all M0 checks
- Date: 2026-10-01
- Context: Kaggle run S03 at commit `7d956c1` (`notebooks/00_probe.ipynb`). The kernel was reused from S02, so `RBBD_SESSION_ID` stayed `20261001T094200Z`; vLLM log timestamps (10:09–10:15) and the new store commit show it is a fresh run.
- Observed (facts):
  - Qwen2.5-0.5B fp16 plain on TP=1 (41.5 s) and **TP=2** (59.2 s) build and generate; greedy texts identical across TP.
  - Merged path (D-050): random rank-32 LoRA folded into fp16 weights in 4.6–5.0 s, written to `/tmp/rbbd_probe/<case>`, served by vLLM on TP=1 (47.8 s) and TP=2 (66.7 s); `merged_dir_removed: true` for both.
  - The vLLM-LoRA canary still fails at `construct` with Triton `PassManager::run failed` (as expected).
  - Gemma-3-1B fp16 and fp32 reproduce S02's greedy outputs exactly (3/4 prompts identical across dtypes).
  - `pytest -q -m gpu tests/gpu/test_generate_gpu.py::test_vllm_hello_tp1_tp2_merged` → `1 passed in 378.52s`.
  - DC-16 passed for the third time; store round trip ok (store commit `a3d5ae3a9bf9c2e3a557af9a82acff77c08b13d4`).
- Options considered: n/a (facts).
- Choice: Tag `m0-green` on `7d956c1`, the exact commit Kaggle validated (DONE M0 row: DC-01, DC-02, DC-10 unit, DC-15, DC-16, DC-19 all pass). Notebooks should restart the kernel between runs so each run gets its own session id.
- Why: The milestone tag must point at validated code.
- Tradeoff accepted: none.
- Cost impact: M0 used ≈ 1.0 session-h across S01–S03, within the 1.0 h budget.
- Paper deviation: no.
- Revisit-if: n/a.

## D-052 Sentence-set construction details (amends D-012–D-016)
- Date: 2026-10-01
- Context: M1-T2..T4 authoring. D-012–D-016 fixed what to build; these are the choices made while building it.
- Options considered: one file per variant (≈ 1,300 hand-written lines, alignment by convention) vs one annotated source per set with variants derived.
- Choice:
  - **Sources, not rendered files.** `resources/sentences/positive.src`, `negative.src` and `targets.tsv` carry synonym slots `[base|v1|v2|v3]` (1–2 per row, 4 distinct options each); `anchors.tsv` holds 42 templates. All variants are rendered at load time, so every variant is row-aligned with base by construction. Set hashes are computed on the rendered text (D-019).
  - **Subject variants by rule** (`subject_variant`): v1 maps he/she → they (and "was" → "were" right after it) and his/her/him/himself/herself → their/them/themselves; v2 maps a sentence-initial He/She → "The person", They → "People". The sources are written to keep these rules grammatical: past tense, no mid-sentence subject pronouns, no object "her".
  - **No personal names in P/N** (amends D-013, which allowed names): names carry gender and ethnic connotations, which would tie valence to groups. Subjects are He 51 / She 52 / They 47 / inanimate-or-possessive 50 across P∪N.
  - **Passive variants:** 47 of 50 target templates are transitive and get an authored passive. The 3 intransitive or copular App. F templates (T01, T03, T04) keep the base wording in `passive` (App. E.2: "grammatical inversion, without additional rewording") and are rewritten as passives in `passive_rephr`.
  - **Neutrality check:** targets and anchors may not contain a fixed list of valenced words or any content word from a P/N synonym slot (function words in multi-word options such as "relied on" are ignored).
  - **Anchors:** 42 templates × 24 groups, with groups 0–7 each dropping template (5i + 7) mod 42, giving 1,000 anchors with group sizes 41–42 and keeping all three App. F anchors.
  - Two anchor templates were replaced during review because they read wrongly for age groups ("grow up in large cities", "learn to drive as teenagers").
  - Validation results: mean words targets 6.78, P 7.19, N 7.21, anchors 7.06; P/N length KS p = 1.0.
- Why: Alignment and reviewability; one source line per sentence.
- Tradeoff accepted: Rule-derived subject variants are more uniform than the paper's (likely LLM-written) ones.
- Cost impact: none (CPU).
- Paper deviation: yes (reconstruction) — App. E, App. F.
- Revisit-if: M8 releases the authors' sets.

## D-053 Metric and statistics implementation choices
- Date: 2026-10-01
- Context: M1-T5..T7.
- Options considered: torch on GPU (PLAN E5 mentions `torch.cdist`) vs NumPy/SciPy on CPU.
- Choice:
  - Metrics run in float64 NumPy (`scipy.spatial.distance.cdist`, matrix products, `np.bincount` group means), with no Python loop over sentences. At the paper's sizes (1,200 targets × 200 attributes × 1,000 anchors) this takes well under a second on CPU, so the M4/M6 analysis needs no GPU (E5's intent).
  - Procrustes is fitted by SVD of X^T Y (d × d). CKA uses the n × n Gram form, which is cheaper than d × d when n = 50 ≪ d.
  - Statistics: `sklearn.metrics.roc_auc_score`; AUC = nan when a class has fewer than 5 rows (PLAN §1.2); OLS by `np.polyfit`; bootstrap draws that give nan are dropped and counted (`n_valid`); cluster resampling (D-024) is a `clusters=` argument.
- Why: CPU-only analysis from cached embeddings; exactness checked against direct loops over Eq. 2–6.
- Tradeoff accepted: none.
- Cost impact: none.
- Paper deviation: no.
- Revisit-if: the analysis becomes slow (> 1 min per setting).

## D-054 `ftdata` stage and token census
- Date: 2026-10-01
- Context: M1-T8. D-042's fallback depends on the share of examples over 512 tokens, and the M3 budget depends on mean length.
- Options considered: census on a sample vs all selected rows; one tokenizer vs each model's own chat template.
- Choice:
  - The `ftdata` stage loads `allenai/wildguardmix` / `wildguardtrain` at a commit SHA resolved at run time (recorded in `stats.json`), checks the schema (B-006), selects both splits (D-010, D-011, seed = config seed), and writes **row indices only** plus aggregate statistics. These are the subcategory histogram and adversarial share per split.
  - The census tokenises every selected example with each Tier 1/Tier 2 model's chat template (user + assistant turns) and reports mean, median, p95, max, and the share over 512 and over 1,024 tokens. It also reports the tokens trained over 3 epochs at max length 1,024 vs 512.
  - It runs on a Kaggle **CPU** session, because the dataset is gated and needs no GPU.
  - `smoke.yaml` overrides the split size to 64 and the census to Qwen only.
  - Stage bodies are registered in `runner.STAGE_IMPLS` and imported lazily, so CPU stages never import the GPU stack.
- Why: Measured inputs for D-042 and PLAN §8 before any GPU time is spent.
- Tradeoff accepted: The ftdata run key covers the whole config, so each tier config re-runs the (cheap) selection.
- Cost impact: ≈ 10–20 min of a Kaggle CPU session (no GPU quota).
- Paper deviation: no.
- Revisit-if: WildGuardMix has fewer than 4,000 harmful rows (D-010 Revisit-if).

## D-055 WildGuardMix census results (Kaggle CPU, 2026-10-01): splits complete, 512 fallback excluded, training budget re-baselined (amends D-010, D-042, PLAN §8)
- Date: 2026-10-01
- Context: M1-T8 `ftdata` stage on a Kaggle CPU session, commit `8277155`, config `configs/base.yaml` (hash `b4a36d44ede276ec`, ftdata run key `68e584d2b9e71a09`). Dataset `allenai/wildguardmix` / `wildguardtrain` at revision `d29c47f41c8b51348b5c8e8c81c039b3132b66d1` (86,759 rows).
- Observed (facts):
  - **Unharmful:** 16,621 eligible, 8,000 selected (all subcategory `benign`; adversarial share 0.535).
  - **Harmful:** 8,341 eligible, 8,000 selected, no shortfall (adversarial share 0.376). Largest subcategory: `social_stereotypes_and_unfair_discrimination` with 1,374 examples.
  - Chat-template lengths (tokens). Llama-3.2-1B has the same tokenizer as Llama-3.1-8B, so its numbers are identical:

    | Model | Split | Mean | Median | p95 | > 512 | > 1024 | Tokens × 3 epochs @1024 |
    |---|---|---|---|---|---|---|---|
    | Llama-3.1-8B | unharmful | 434 | 371 | 970 | 30.2% | 4.0% | 10.13 M |
    | Llama-3.1-8B | harmful | 662 | 610 | 1360 | 63.5% | 12.1% | 14.87 M |
    | Mistral-7B | unharmful | 458 | 382 | 1089 | 33.2% | 6.5% | 10.49 M |
    | Mistral-7B | harmful | 731 | 666 | 1559 | 68.1% | 17.9% | 15.83 M |
    | Gemma-3-4B | unharmful | 415 | 352 | 955 | 28.6% | 3.7% | 9.68 M |
    | Gemma-3-4B | harmful | 649 | 596 | 1357 | 61.4% | 11.5% | 14.55 M |
    | Qwen2.5-0.5B | unharmful | 430 | 367 | 970 | 29.9% | 3.9% | 10.04 M |
    | Qwen2.5-0.5B | harmful | 660 | 607 | 1358 | 63.1% | 12.0% | 14.80 M |
- Consequences (choices):
  1. **D-010 holds** with 8,000 harmful rows; nothing is upsampled.
  2. **The max-length-512 fallback of D-042 is excluded.** It is allowed only if ≤ 5% of examples are truncated, and the measured share is 29–68%. If the M3c-T8 throughput probe projects > 9 h per Tier 1 run, the remaining fallback is 3 epochs → 1 epoch (user pre-approved, Q3).
  3. **Budget re-baseline (PLAN §8).** The plan assumed 400 tokens/example (9.6 M tokens per run); harmful runs are 14.5–15.8 M (×1.5–1.65). u‖h run in parallel, so session time is set by the harmful run. At the assumed 350 tok/s per T4 for 7–8B QLoRA:
     - Llama ≈ 11.8 h, Mistral ≈ 12.6 h, Gemma ≈ 5.8 h (fp16) to ≈ 13.6 h (fp32 fallback).
     - Tier 1 training ≈ 32–40 session-h at 3 epochs (planned 25), or ≈ 11.5–14 h at 1 epoch.
     - Tier 2 training scales by ≈ 1.54: ≈ 12.3 h (planned 8.0).
     - Projected core total ≈ 96 session-h with 15% contingency at 3 epochs (planned 79), or ≈ 70 h if Tier 1 drops to 1 epoch.
     - Both depend on throughput measured in M3c-T8, which replaces the 350 tok/s assumption. Every Tier 1 harmful run already projects above D-042's 9 h trigger at the assumed throughput.
  4. **New M3 requirement:** 4–18% of examples exceed 1,024 tokens (the paper's limit). With prompt-head truncation (D-040), an example whose prompt alone fills the window keeps no completion tokens and so contributes no loss. M3-T1 must count these examples per model and drop them from the training set (logging the count) rather than train on empty targets.
- Why: Measured inputs replace the planning assumption, as D-054 intended.
- Tradeoff accepted: none yet; the epoch decision waits for measured throughput.
- Cost impact: census took 205 s on a CPU session (no GPU quota).
- Paper deviation: no (the 512 option is now off the table).
- Revisit-if: the M3c-T8 throughput differs from 350 tok/s by more than ±30%.

## D-056 DONE matrix correction: DC-07 belongs to M3, not M1
- Date: 2026-10-01
- Context: DONE.md listed "DC-07 (synthetic)" under M1 and PLAN §7 M1 listed DC-07, but the merge code (`models/adapters.py`, `models/interpolate.py`) is M3-T2/T3 and `tests/test_merge.py` was created as `PLACEHOLDER(M3)` at planning time. It was a planning inconsistency.
- Options considered: implement the merge maths early just to tick the box, or correct the matrix.
- Choice: Correct the matrix. DC-07 applies from M3 (where it was already listed). M1 is closed on DC-01–05, DC-08, DC-09, DC-15, DC-17, DC-19.
- Why: A check is only meaningful against the code it tests; nothing about merging exists in M1.
- Tradeoff accepted: none (DC-07 is still required at M3).
- Cost impact: none.
- Paper deviation: no.
- Revisit-if: n/a.

## D-057 Reference-only `embed` stage for M2; config-dependent stage dependencies
- Date: 2026-10-01
- Context: M2 extracts reference embeddings before any fine-tuning exists, but the stage graph had `embed` depend on `train` (FLOW.md). M2-T4's DC-12 command referred to checkpoint `a050`, which does not exist until M3/M4.
- Options considered: a separate probe command outside the runner; or making `embed`'s dependencies follow the configured checkpoints.
- Choice:
  - **Dependencies.** `runner.stage_deps(stage, cfg)`: `embed` depends on `sentences` only when `embed.checkpoints == [ref]` (the M2 default in `base.yaml`), and on `sentences` + `train` otherwise. Run keys use these dependencies.
  - **Stage scope.** The M2 `embed` stage extracts `ref` only and raises `NotImplementedError` (M4) for α-checkpoints. It writes `embed/<run_key>/timing.json` and records the cache files as stage outputs.
  - **Timing.** The stage times `download` (`snapshot_download` of weights, config and tokenizer files into `$HF_HOME`) separately from `load` and extraction. DC-12 = load + extraction seconds; the cost of one α-checkpoint equals the reference's, so DC-12 is measured on `ref` in M2 and re-measured on `a050` in M4.
  - **Loading.** Models load with `AutoModelForCausalLM`; `base_decoder` finds the inner text transformer (Llama/Mistral/Qwen `model.model`; Gemma 3's text `language_model`). The stage asserts that the configured layer index equals the decoder depth (D-017).
  - **Smoke subset.** `sentences.subset` (`smoke.yaml`: 3 groups × 5 targets, 10 P, 10 N, 50 anchors, base variants) is cut from the fully validated sets.
  - **Tests.** The CPU tests run the whole stage on a 2-layer random Llama built from a config, with a word-level fake tokenizer (no downloads).
- Why: Keeps the runner as the only way stages execute, and keeps cache keys honest about what each extraction depended on.
- Tradeoff accepted: The `embed` run key changes when α-checkpoints are added (as it should).
- Cost impact: none.
- Paper deviation: no.
- Revisit-if: M4 needs reference and α-checkpoints in one stage run (it will reuse the cached `ref` entry either way).

## D-058 Pre-registered M2 acceptance rules: DC-06 tolerance and the Gemma fp16 criterion (amends DONE DC-06, D-005) — DC-06 rule superseded by D-061 (the Gemma fp16 criterion stays in force)
- Date: 2026-10-01 (written before any M2 GPU result)
- Context: DC-06 said "atol 1e-3 in fp16". Embeddings are cached in fp16, whose rounding step is ≈ 0.008 for values in [8, 16) (Qwen and Llama post-norm hidden states reach such values), so a pure absolute tolerance of 1e-3 cannot be met even by identical computations that round differently. D-005 needed an explicit rule for when Gemma may run in fp16.
- Options considered: atol only; atol + rtol; cosine-only.
- Choice:
  - **DC-06:** pass if `torch.testing.assert_close(..., atol=1e-3, rtol=1e-3)` holds for mean, max and last pooling (rtol 1e-3 is torch's default for fp16). The GPU test also records max absolute and relative differences in `m2_extract_gpu.json`.
  - **Gemma fp16 criterion:** Gemma-3-4B may move from fp32 to fp16 (superseding D-005) only if all three hold for the full sentence union:
    1. the fp16 extraction completes with the NaN/Inf guard never firing;
    2. per-sentence cosine(fp16, fp32) has mean ≥ 0.999 and minimum ≥ 0.99 for every pooling;
    3. the largest per-group |B_fp16 − B_fp32| under RR and SEAT is ≤ 5% of the spread of B across groups in fp32.
    The numbers come from `python -m rbbd.cli compare-embeddings`. Otherwise Gemma stays fp32.
- Why: Rules fixed before the data cannot be tuned to the data.
- Tradeoff accepted: The 5% threshold is a judgment call; it is about one tenth of the group-to-group variation ΔB has to detect.
- Cost impact: Gemma in fp16 would halve its inference memory and roughly double throughput.
- Paper deviation: no.
- Revisit-if: n/a.

## D-059 Local CPU test environment needs the `train` extra
- Date: 2026-10-01
- Context: M2's CPU tests run a real (tiny) transformers model, so `pytest -q -m "not gpu"` now needs torch and transformers. Installing them here, the PyTorch CPU wheel index was unreachable, and `pip install accelerate` pulled an **unpinned** torch 2.14.1 from PyPI as a dependency. It was replaced at once by the pinned `torch==2.7.1`; no test ever ran on the unpinned version.
- Options considered: separate CPU-only pins; skip torch tests without torch; require `.[train,dev]` for the CPU suite.
- Choice: DC-02's environment is `pip install -e ".[train,dev]"`. Tests are never skipped for a missing dependency. When adding packages, install the pinned extra as one command so pip resolves torch to the pin.
- Why: One environment definition; no silently skipped tests.
- Tradeoff accepted: The CPU environment is a few GB larger.
- Cost impact: none on Kaggle (it installs `train` anyway).
- Paper deviation: no.
- Revisit-if: CI is added (out of scope per the brief).

## D-060 Pin torchvision 0.22.1 next to torch 2.7.1 (follow-up to D-031)
- Date: 2026-10-01
- Context: The first M2 Kaggle run (`notebooks/m2_extract.ipynb` at `92e836e`) failed on every model load (B-012). `pip install -e ".[train,dev]"` replaced Kaggle's torch with 2.7.1 but left Kaggle's preinstalled torchvision, built for a different torch. `transformers` imports torchvision through `image_utils` while resolving any model class. The torchvision C++ ops then fail to register (`RuntimeError: operator torchvision::nms does not exist`), which surfaces as `ModuleNotFoundError: Could not import module 'Qwen2ForCausalLM'` (also Llama, Gemma3). M0 never hit this because `vllm` (in `bench`) pulls a matching torchvision.
- Options considered: (a) pin `torchvision==0.22.1` (the release built against torch 2.7.1) in every extra that pins torch; (b) `pip uninstall -y torchvision` in each notebook; (c) leave Kaggle's torch alone. The user chose (a) on 2026-10-01.
- Choice: `train` and `bench` both pin `torchvision==0.22.1`. `m2_extract.ipynb` runs a preflight import of the three model classes right after install, so a mismatch fails in the install cell, not inside a stage. A CPU test (`tests/test_env.py`) checks that the pins stay paired and that the installed pair imports the model classes. Also in this commit: `from_pretrained(torch_dtype=…)` → `dtype=…` (deprecated in transformers 4.57). This changes no cache key, because the key records `precision`.
- Why: The environment is the same in every session and identical to what `pip` resolves locally. Removing packages from the image is less reproducible.
- Tradeoff accepted: The install downloads one more wheel (≈ 7 MB for cu126).
- Cost impact: ≈ +10 s setup per session. The failed run cost one T4×2 session of ≈ 0.2 h (downloads only).
- Paper deviation: no. The paper stack (App. C) does not list torchvision; it is a transitive dependency.
- Revisit-if: The torch pin changes (re-pair torchvision), or `vllm` requires a different torchvision.

## D-061 DC-06 rule replaced: fp32 exact padding check plus fp16 per-sentence cosine (supersedes the DC-06 part of D-058)
- Date: 2026-10-01 (written after the first DC-06 result, before any data under the new rule)
- Context: The first Kaggle run of `test_padding_invariance_fp16` (session `20261001T113125Z`, commit `64498de`) failed the D-058 rule `assert_close(atol=1e-3, rtol=1e-3)`. Max |alone − batched|: mean 0.146, max 0.625, last 0.156, on values up to 124.7 / 183.1 / 146.8. D-058's premise was wrong: post-norm Qwen values reach ~183, not [8, 16). In [128, 256), fp16's rounding step is 0.125, so rtol 1e-3 is about one ulp. The rule therefore demanded bit-identical results from different GEMM shapes after 24 fp16 layers. With right padding and causal attention, pad positions come after every real token and cannot reach it, so a mathematical leak is impossible in the forward pass. A leak can only come from pooling (covered on CPU by `tests/test_pooling.py`) or from mask/position handling, which an fp32 run exposes.
- Options considered: (a) fp32 exact check + fp16 per-sentence cosine; (b) loosen the fp16 rtol to 1e-2; (c) keep D-058, leaving M2 red (fp32 for Llama-8B does not fit 2×T4). The user chose (a) on 2026-10-01.
- Choice: DC-06 on GPU passes only if both tests in `tests/gpu/test_extract_gpu.py` pass:
  1. `test_padding_invariance_fp32`: Qwen2.5-0.5B in fp32. The short sentence pooled alone vs inside a padded batch of three passes `assert_close(atol=1e-3, rtol=1e-4)` for mean, max and last.
  2. `test_padding_cosine_fp16_smoke`: Qwen2.5-0.5B in fp16. For each of the 85 smoke-union sentences, compare the sentence pooled alone with the same sentence inside its production batch (the smoke `embed` settings). Per-sentence cosine ≥ 0.9999 for mean, max and last.
  The test also records, without asserting, the cosine and relative L2 error of fp16 (alone and batched) against fp32. Those numbers set padding noise against fp16's own error.
- Why: (1) proves the padding logic is exact; any leak shows up as an O(1) error, far above fp32 rounding. (2) bounds fp16 padding noise at the level that matters downstream: RR uses cosines, and SEAT uses cosines too. 0.9999 means the angle stays within 0.81°.
- Tradeoff accepted: The fp16 bound is on direction, not on single coordinates. Ref and audited checkpoints share the same tokenizer, texts and `make_batches` plan, so they see the same batch composition, and the residual noise does not differ systematically between them.
- Cost impact: GPU test time grows from ~13 s to ~1–2 min: 85 + 85 single-sentence forwards on 0.5B plus one fp32 load (~2 GB on cuda:1).
- Paper deviation: no. The paper does not describe its batching [unspecified in paper].
- Revisit-if: Part 1 fails (then it is a real bug: open a B entry), or Part 2's recorded fp16-vs-fp32 error turns out larger than the padding error by orders of magnitude (then precision, not padding, is the risk to watch in M4).

## D-062 Gemma-3-4B fp16 extraction probe failed: embeddings stay fp32 (outcome of D-005 / D-058 criterion)
- Date: 2026-10-01
- Context: M2-T4 ran `configs/m2_gemma3-4b_fp16_probe.yaml` on Kaggle 2×T4 (session `20261001T113125Z`, commit `64498de`). The NaN/Inf guard fired on the first batch: `non-finite values at embed.post_norm: nan=10485760 inf=0 shape=(256, 16, 2560)`. Every value was NaN. Under D-058 criterion 1 (guard never fires) fp16 fails, and criteria 2–3 cannot be evaluated. `compare-embeddings` had no fp16 manifest to read.
- Options considered: none open under the pre-registered rule.
- Choice: Gemma-3-4B (and Gemma-3-1B in Tier 2) embeddings stay fp32, as D-005 states. The fp32 reference run passed: `configs/tier1_gemma3-4b.yaml`, all 9,197 union texts, load 22.2 s, extraction 191.0 s, peak 9.07 / 9.92 GiB on cuda:0 / cuda:1, no guard event. `m2_extract.ipynb` now runs the comparison only when the fp16 run completes.
- Why: Pre-registered rule; the guard did its job.
- Tradeoff accepted: Gemma extraction takes ~3.2 min per checkpoint against Llama-8B fp16's 1.2 min. Over 8 checkpoints that is ~26 min of extraction.
- Cost impact: within PLAN §8 (D-005 already budgeted Gemma in fp32).
- Paper deviation: yes, App. C precision, as already logged in D-005.
- Revisit-if: never for M2. Training precision is decided separately by D-005's 50-step probe in M3.

## D-063 DC-06 part 1 measures the pooled vectors before the fp16 cache cast (implements D-061; thresholds unchanged)
- Date: 2026-10-01 (after the first run under D-061)
- Context: The first run under D-061 (session `20261001T122025Z`, commit `ac0f1ce`) passed part 2 and failed part 1. Part 1's measured gaps were max_abs 0.00390625 (last, max) and 0.001953125 (mean): exactly 2⁻⁸ and 2⁻⁹, which are fp16 rounding steps. `encode` always cast its output to fp16, the cache dtype. The test therefore compared fp16-rounded copies of fp32 results, and a sub-ulp fp32 gap that crossed a rounding boundary became a whole fp16 step (max_rel 1.6e-3 ≈ 2 fp16 ulps). D-061 states part 1 as a check of fp32 arithmetic, so the test did not implement the decision (B-014).
- Options considered: (a) compare the pooled vectors before the cast, thresholds unchanged; (b) loosen part 1's tolerance to cover fp16 output rounding. Option (b) would change a pre-registered threshold after seeing data.
- Choice: (a). `encode(..., keep_fp32=True)` returns the pooled vectors before the cast; the fp16 guard still runs. Only `test_padding_invariance_fp32` uses it, and the cache stays fp16. Tolerance stays `atol=1e-3, rtol=1e-4`.
- Why: It measures exactly what D-061 specifies. This fix is chosen after seeing a failure, so it is logged with the evidence that justifies it: the observed gaps are exact powers of two, matching fp16 rounding steps rather than fp32 compute error. Even before the fix, the fp32-compute gap (≤ 0.0039) was 160× smaller than the fp16-compute gap (0.625). A padding leak would give O(1) errors.
- Tradeoff accepted: One test-only keyword on `encode`.
- Cost impact: none (same GPU test).
- Paper deviation: no.
- Revisit-if: Part 1 fails with pre-cast vectors. That would be a real padding bug.

## D-064 M2 closed: every M2 DONE check passes; `m2-green` = `8e26c8d`
- Date: 2026-10-01
- Context: DONE requires DC-01, DC-02, DC-06, DC-09, DC-10, DC-11 (sentences+embed), DC-12, DC-15 and DC-19 for M2. Evidence comes from Kaggle 2×T4 sessions `20261001T113125Z` (commit `64498de`; pipeline runs) and `20261001T122025Z` (commits `ac0f1ce` → `8e26c8d`; GPU tests). The code paths behind the pipeline runs did not change after `64498de`. `8e26c8d` added only `keep_fp32` (default off) and test code.
- Results:
  - DC-01: `ruff check src tests` → `All checks passed!`.
  - DC-02: `python -m pytest -q -m "not gpu"` → `79 passed, 8 deselected in 11.18s`.
  - DC-06 (D-061/D-063): `3 passed, 14 warnings in 20.53s`.
    - fp32 pre-cast max_abs: last 1.36e-4, max 7.31e-4, mean 1.83e-4, all within `atol=1e-3, rtol=1e-4`. The tightest is max pooling, at 7.31e-4 against an allowance of ≥ 1e-3.
    - fp16 min cosine: mean 0.9999973, max 0.9999936, last 0.9999967 (threshold 0.9999).
  - DC-09: `test_guard_active_on_real_model` passed in both sessions.
  - DC-10: second smoke run → `cache hit: stage embed`, with no load and no forward.
  - DC-11 (sentences+embed): smoke embed stage done in 15.2 s.
  - DC-12: Llama-3.1-8B load 73.4 s + extraction 71.4 s = 144.8 s ≤ 900 s.
  - DC-15: docs in the same commits. DC-19: `grep -rn "PLACEHOLDER(M2)" tests` → no output.
- Choice: M2 is green. Tag `m2-green` on `8e26c8d75dfe64dc26e3fb6b1e834f038dbb46dd` (the exact commit run on Kaggle).
- Why: Every check passes with its stated output.
- Tradeoff accepted: DC-06's fp32 margin for max pooling is modest (7.3e-4 vs 1e-3). It is recorded so that a future drift is noticed.
- Cost impact: M2 used ≈ 0.6 session-h on Kaggle in total (three sessions).
- Paper deviation: no (precision deviations are logged in D-005/D-062).
- Revisit-if: an `embed.schema_version` bump (ROLLBACK lists DC-06, 09, 10, 11, 12 to re-run).

## D-065 Training guard under fp16 mixed precision: scaler-skipped steps are tolerated within limits (refines D-006)
- Date: 2026-10-01
- Context: D-006 says training aborts on a non-finite loss or grad-norm. Every T4 run uses fp16 autocast with a dynamic gradient scaler. The scaler starts at 2¹⁶ and deliberately overflows on some steps, mostly early. On those steps the unscaled grad-norm is inf/NaN and the optimizer step is skipped, so nothing reaches the weights. Applied literally, D-006 would abort every fp16 run. The user chose this refinement on 2026-10-01.
- Options considered: (a) tolerate scaler-skipped steps within limits; (b) strict D-006, which would force fp32 compute everywhere (QLoRA 7–8B ≈ 2–3× slower).
- Choice (a), implemented in `finetune.sft.GuardCallback`, with `logging_steps=1` so every step is checked:
  - A non-finite **loss** always aborts (`NonFiniteError`), on any step.
  - A non-finite **grad-norm** is tolerated only on a step the scaler skipped (`accelerator.optimizer_step_was_skipped`); on an applied step it aborts.
  - More than **25 consecutive** skipped steps aborts: the scale has collapsed by a factor of 2²⁵.
  - More than **5 %** skipped steps aborts once the run has at least **100** steps. A short run is exempt because its early skips while the scaler calibrates would dominate.
  - In fp32 compute the scaler is off and nothing is ever skipped, so any non-finite value aborts.
  - `done.json` records `scaler_skipped_steps`. Limits live in `train.guard` (config) and are part of nothing's cache key.
- Why: Keeps D-006's intent (no silent NaN can reach weights or results) while allowing the standard fp16 recipe.
- Tradeoff accepted: A run can still finish after a few skipped steps; the count is recorded.
- Cost impact: none (one check per logged step).
- Paper deviation: no.
- Revisit-if: a run aborts on the skip limits; then switch that model to fp32 compute per D-005 and log it.

## D-066 Spectrum endpoints are the trained adapters themselves; interior α uses the rank-2r concatenation (refines D-007)
- Date: 2026-10-01
- Context: DC-07 requires α=1 and α=0 to reproduce the endpoints exactly (`torch.equal`). D-007's concatenated adapter is exact in arithmetic, but its ΔW is a matmul over 2r ranks (half of them zero blocks). That may round differently from the endpoint's rank-r matmul, so bit equality of ΔW is not guaranteed. The user chose this option on 2026-10-01.
- Options considered: (a) a100 = u and a000 = h as trained, with concatenation only for α ∈ {0.9, 0.7, 0.5, 0.3, 0.1}; (b) concatenation for every α.
- Choice: (a). `models.adapters.adapter_for_alpha` returns the endpoint adapter objects at α ∈ {1, 0} and `combine(u, h, α)` otherwise. `combine` is still tested at every α including 0 and 1. ΔW linearity is checked with float64 evaluation of the stored fp32 tensors, rtol 1e-6 with atol 1e-6·max|ΔW|; elements near zero have no meaningful relative error. A combined adapter keeps PEFT's config format (r = lora_alpha = 2r, scaling 1). Its provenance (α, s_u, s_h) goes to a sidecar `rbbd_meta.json`, because PEFT warns on unknown config keys.
- Why: The endpoints are exact by construction, and interior points are exact up to fp32 rounding (measured ≤ 5e-8 relative on CPU).
- Tradeoff accepted: Endpoints are served at rank r and interior points at rank 2r. The two paths differ only by fp32 rounding.
- Cost impact: none.
- Paper deviation: no (App. C merging, implemented exactly).
- Revisit-if: vLLM ever serves adapters directly again (D-050); then check that rank-r and rank-2r adapters coexist in one engine.

## D-067 M3a implementation choices for `finetune.sft` and the train stage (implements D-003, D-009, D-040, D-055)
- Date: 2026-10-01
- Context: M3-T1–T3 and M3a-T4. Details the plan left open.
- Choices:
  - **Tokenisation** reproduces TRL 0.25.1's prompt–completion path:
    - The prompt is rendered with `add_generation_prompt=True`, then prompt + completion.
    - `completion_mask` covers the assistant tokens.
    - The pair is truncated from the right at `max_length`.
    - Doing this in `build_examples` lets us **drop and count examples whose prompt fills the window** (D-055 item 4; `data.json` → `n_dropped_prompt_fills_window`).
    - It fails loudly (`DataBuildError`) if a chat template breaks the prompt-prefix property. TRL only warns in that case.
    - The pre-tokenised dataset is passed to `SFTTrainer` with `completion_only_loss=True` set explicitly; TRL cannot infer it without a `prompt` column.
  - **Hyperparameters** (App. C):
    - 3 epochs, per-device batch 4 × grad-acc 8, warmup 0.03, linear schedule, max length 1,024.
    - LoRA r 16, α 32, dropout 0.05 on all 7 projections; lr 1e-4 (LoRA/QLoRA) and 2e-5 (full).
    - `group_by_length`; seed = data seed = the job seed.
    - [unspecified in paper], TrainingArguments defaults: `max_grad_norm` 1.0, `weight_decay` 0.
    - Optimizer: `adamw_torch` for LoRA/QLoRA (App. C AdamW); `paged_adamw_8bit` for full FT (D-009).
  - **Precision:**
    - LoRA: fp16 base with fp32 LoRA weights (PEFT autocast).
    - QLoRA: NF4 + double quantisation, compute fp16 (D-003).
    - Full: fp32 masters with fp16 autocast. The endpoint is saved as fp16 safetensors (D-008).
    - A `models:` entry may set `train_compute: fp32` (D-005).
  - **Checkpoints:**
    - A trainer checkpoint is written every 20 min of wall-clock time (`WallClockSaveCallback`), and a job resumes from the newest one.
    - `save_total_limit=1`: the trainer itself replaces the previous checkpoint. This is the library's normal rotation of transient training state inside the job directory, not an ad hoc deletion of artifacts.
    - A job with `done.json` is never retrained.
  - **Jobs:**
    - One per (model, regime, seed, split), in `train/<slug>/<regime>/<split>/seed<k>/<train_key>/`.
    - Each job directory holds `checkpoints/`, `final/`, `data.json`, `train_log.json`, `done.json` and `train.log`. Everything is aggregate-only: no row text is written (D-037).
    - u‖h run as two `cli train-one` processes with `CUDA_VISIBLE_DEVICES` 0/1 (D-003).
    - `done.json` records `tokens_per_second` for the M3c-T8 throughput probe.
  - **Smoke** (Tier 0 only): effective batch 8 (4 × 2) instead of 32, so 64 rows × 3 epochs give 24 optimizer steps rather than 6, and the adapters move measurably for DC-07.
  - **Config hash:** adding the `train:` section to `base.yaml` changes every config's hash, so the M2 configs' `sentences`/`embed` manifests re-run once. The embedding tensor cache is keyed by content, not by config hash, so nothing is re-extracted.
  - **Full-FT merges** (`models.interpolate`): `materialize` refuses any path under the artifact root or `/kaggle/working`. It uses `utils.env.ephemeral_dir()` (D-050), not D-008's `/kaggle/tmp`, which does not exist on current Kaggle images (M0 probe).
- Why: Each item follows the paper where it speaks and records the choice where it does not.
- Tradeoff accepted: The smoke run's batch differs from the paper's; smoke results are pipeline checks only.
- Cost impact: Smoke training ≈ 2–5 min on 2×T4 (estimate; measured in M3a).
- Paper deviation: smoke batch only (Tier 0). Otherwise no.
- Revisit-if: TRL's tokenisation changes (pinned 0.25.1), or M3c-T8 shows checkpointing overhead > 5 %.

## D-068 One visible GPU per training process, per-spec LoRA init seed, train schema v2 (refines D-003, D-067)
- Date: 2026-10-01
- Context: The first M3a Kaggle run (session `20261001T132720Z`, commit `ec87d99`) found two training-path defects (B-017).
  - Under pytest, both T4s were visible. The HF Trainer then wrapped a `device_map={"": 0}` model in DataParallel and doubled the effective batch: 3 steps instead of 6.
  - The LoRA A init came from whatever global RNG state preceded `SFTTrainer`. The test's two "identical" runs therefore differed by up to 0.067 in adapter weights.
  - The train stage itself was unaffected by DataParallel: each `train-one` process sees one GPU. Both smoke endpoints ran 24 steps, as planned.
- Options considered: force `args._n_gpu = 1` (a private field) vs. refuse multi-GPU visibility; seed in `cli train-one` only vs. inside `train_one`.
- Choice:
  - `train_one` raises unless at most one CUDA device is visible. The App. C effective batch can then never change silently.
  - The GPU DC-13 test runs each phase in a child process with `CUDA_VISIBLE_DEVICES=0`, so the "kill" is a real process exit.
  - `train_one` calls `transformers.set_seed(spec.seed)` immediately before building the trainer, so the initial adapter depends only on the spec.
  - `sft.SCHEMA_VERSION` 1 → 2: the same spec now yields different (reproducible) weights, so every `train_key` changes. The smoke endpoints retrain once (≈ 90 s); no Tier 1/2 training had run.
- Why: Hyperparameters and initialisation must be functions of the spec alone.
- Tradeoff accepted: A single process can no longer train on two GPUs. Nothing planned needs that (D-003).
- Cost impact: ≈ 2 min to retrain the smoke endpoints.
- Paper deviation: no.
- Revisit-if: FSDP/DDP training becomes necessary (D-009 Revisit-if), which would launch through `accelerate`/`torchrun` instead.

## D-069 M3a closed: every M3a DONE check passes; `m3a-green` = `7b41de1`
- Date: 2026-10-01
- Context: DONE lists DC-01, DC-02, DC-07, DC-09, DC-11 (through train), DC-13, DC-15 and DC-19 for M3. M3a covers them for Tier 0. Evidence is Kaggle 2×T4 session `20261001T154608Z` at commit `7b41de1`; store commits: train `797549be`, env `ed4766fb`.
- Results:
  - **DC-01:** `ruff check src tests` → `All checks passed!`.
  - **DC-02:** `python -m pytest -q -m "not gpu"` → `99 passed, 10 deselected in 80.20s`.
  - **DC-07:**
    - CPU (`tests/test_merge.py`): α ∈ {0, 1} are `torch.equal` to the endpoints.
    - GPU (`test_real_adapter_endpoints`, real smoke adapters, 168 modules, r 16): both endpoints trained (max|ΔW| 5.5e-4 for u and 5.4e-4 for h). Linearity max relative error is 4.9e-8 at α 0.1, 5.0e-16 at 0.5 and 4.8e-8 at 0.9 (bound 1e-6). The a050 adapter gives a finite forward on the fp16 base.
  - **DC-09 (training):** the guard tests in `tests/test_sft.py` pass. Both smoke runs had 0 scaler-skipped steps.
  - **DC-11 (through train):** `smoke ftdata+train wall-clock: 112 s` (< 15 min); the identical rerun → `ftdata: cache hit, train: cache hit`.
  - **DC-13:**
    - CPU: a run resumed after a kill at checkpoint-3 is bit-identical to an uninterrupted one.
    - GPU (child processes, one visible GPU): uninterrupted run 6 steps; killed run exit 3 after checkpoint-3; resumed run 6 steps from `checkpoint-3`. Adapter difference 0.0 (max |value| 0.034).
    - `test_train_one_refuses_two_visible_gpus` passed.
  - **DC-15:** docs updated in the same commits. **DC-19:** `grep -rn "PLACEHOLDER(M3)" tests` → no output.
  - **Training record (aggregates):**
    - Unharmful: train loss 1.813, final 1.526, 1,911 tok/s.
    - Harmful: train loss 2.106, final 1.775, 2,149 tok/s.
    - 24 steps each; 64/64 rows kept, 0 dropped, 5 truncated.
    - These match run 1 (`ec87d99`) to 4 decimals. The only change between the runs was the init seeding.
- Choice: M3a is green. Tag `m3a-green` on `7b41de1964d6509b34ccfa4e00bcf245d1c77552`, the exact commit run on Kaggle.
- Why: Every check passes with its stated output.
- Tradeoff accepted: Tier 1/2 training (M3b, M3c) is still ahead; `m3-green` as a whole needs M3b and M3c.
- Cost impact: M3a used ≈ 0.6 session-h (two runs).
- Paper deviation: smoke batch only (D-067).
- Revisit-if: `train` schema bump; then re-run DC-07, DC-11 and DC-13 (ROLLBACK).

## D-070 M3c-T8 throughput probe for Llama-3.1-8B QLoRA and its pre-registered decision rule (implements D-042, D-055)
- Date: 2026-10-01 (written before any probe data)
- Context: D-055 projected Tier 1 training at an assumed 350 tok/s per T4. At that rate every harmful run exceeds D-042's 9 h trigger. The user chose to run this probe before M3b (2026-10-01).
- Options considered: probe one model now and the others in their own sessions; or probe all three Tier 1 models now (≈ 2 h of downloads plus probes).
- Choice:
  - **Probe** (`notebooks/m3c_probe.ipynb`, `configs/m3c_probe_llama3.1-8b_b{4,2,1}.yaml`):
    - QLoRA on the real WildGuardMix splits, the same 8,000 ids per split as the full run.
    - Unharmful on cuda:0 and harmful on cuda:1, in parallel as in production (D-003). Paper hyperparameters except `max_steps: 25`, `max_minutes: 20` and no checkpoint inside the window.
    - Batch layouts are tried in order 4×8, 2×16, 1×32 (effective batch 32 in all), moving on only after a CUDA out-of-memory failure.
    - `done.json` records:
      - `tokens_per_second_steady`: Δ`num_tokens`/Δtime over logged steps > 3. These are non-pad tokens, the same count as `data.json`'s `n_tokens`.
      - `seconds_per_step_steady` and `peak_mem_gib`.
      - `projected_hours` = n_tokens × epochs / tokens_per_second_steady for 3 epochs and for 1.
    - Only aggregate JSON is synced; probe adapters are not uploaded.
  - **Decision rule** (per model, applied to Llama now):
    1. **Batch layout:** the first of 4×8, 2×16, 1×32 that completes becomes that model's Tier 1 layout. The effective batch stays 32, so this is not a paper deviation.
    2. **Epochs:**
       - If the **harmful** run's `projected_hours["3_epochs"]` ≤ 9 h, the model trains for 3 epochs (App. C). The harmful run is the longer one and sets the session time, since u‖h run in parallel.
       - Otherwise it trains for 1 epoch. That is D-042's pre-approved Q3 fallback; the 512-token option is excluded by D-055. A new D entry logs it as a paper deviation (App. C epochs).
       - If even 1 epoch projects > 9 h, stop and ask the user.
    3. If the measured rate differs from 350 tok/s by more than ±30 % (D-055 Revisit-if), PLAN §8's budget is re-baselined in the same D entry.
  - Mistral-7B and Gemma-3-4B are decided by the same rule from their own probe at the start of their M3c session. For Gemma this is D-005's 50-step fp16 probe.
- Why: The epoch choice and the week's budget split depend on a measured rate. Fixing the rule first keeps the choice independent of the outcome.
- Tradeoff accepted:
  - 25 steps (≈ 800 of 8,000 examples) under `group_by_length`'s random megabatches is a sample. Its error is assumed small next to the 9 h margin.
  - Checkpoint saves (every 20 min in real runs) are excluded from the measurement; they are expected to cost < 1 %.
  - Per-model epoch counts could differ across Tier 1 models; any difference is recorded.
- Cost impact: ≈ 0.6–0.8 session-h (download ≈ 2 min, data selection ≈ 20 s, ≤ 20 min training per tried layout).
- Paper deviation: no (the probe itself). The epoch fallback, if taken, gets its own D entry.
- Revisit-if: the probe's two splits disagree by > 20 % in tok/s at similar lengths (suggests contention), or real runs drift > 15 % from the probe.

## D-071 Llama-3.1-8B Tier 1 trains for 1 epoch at batch 2×16: D-070 rule applied to the measured probe (amends D-003, D-042, D-055; PLAN §8)
- Date: 2026-10-01
- Context: Probe run of `notebooks/m3c_probe.ipynb` on Kaggle 2×T4 (session `20261001T171846Z`, commit `c44ecaf`). Store commits: manifests `cf24d67a`, ftdata `4b9d42b4`, env `1500ad17`. Splits are the same 8,000 ids per split as the full run.
- Observed (facts):
  - **Batch 4×8:** `torch.OutOfMemoryError` after 131 s. It happened in TRL's token-accuracy metric (`outputs.logits[..., :-1, :].contiguous()`, +1.96 GiB) with 12.94 GiB already in use on a 14.56 GiB T4.
  - **Batch 2×16:** completed. The 20-minute cap stopped it at 9 steps (unharmful) and 8 steps (harmful); no scaler-skipped steps.

    | Split | n_tokens (1 epoch) | Truncated | Dropped | Steady tok/s | s/step | Peak GiB | Projected 3 ep | Projected 1 ep |
    |---|---|---|---|---|---|---|---|---|
    | unharmful | 3,375,072 | 319 | 0 | 204.7 | 116.7 | 9.91 | 13.74 h | 4.58 h |
    | harmful | 4,955,020 | 965 | 0 | 214.8 | 149.6 | 9.91 | 19.22 h | 6.41 h |

  - The two splits agree within 5 % in tok/s: no sign of contention (D-070 Revisit-if not met).
  - No example's prompt filled the 1,024-token window, so D-055 item 4's drop path removes nothing for Llama.
- Choice (mechanical application of D-070):
  1. **Batch layout** 2 × grad-acc 16 (effective 32, App. C), written into `configs/tier1_llama3.1-8b.yaml`.
  2. **Epochs: 1** for both Llama endpoints. The harmful run projects 19.22 h at 3 epochs (> 9 h) and 6.41 h at 1 epoch (≤ 9 h). This is D-042's pre-approved Q3 fallback; the 512-token option is excluded by D-055. The linear schedule and 3 % warm-up now span one epoch.
  3. **Budget re-baseline:** the measured rate (≈ 210 tok/s) is −39 % against the assumed 350 tok/s (D-055 Revisit-if: ±30 %).
     - Llama ≈ 6.6 session-h: 6.41 h training plus ≈ 10 min of load, data and checkpoint overhead.
     - Mistral-7B (1 epoch, ≈ 5.28 M harmful tokens) ≈ 6.8 h if it runs at Llama's rate, probably less (smaller model and 32k vocabulary).
     - Gemma-3-4B: unknown until its D-005 probe. If fp16 compute holds, roughly 3–4 h. If it needs fp32 compute (likely, given D-062), it may exceed 9 h even at 1 epoch, and D-070 then requires asking the user.
     - Tier 1 training ≈ 17–21 h if Gemma stays ≤ 9 h; D-055 had 11.5–14 h at 1 epoch. The core total moves from ≈ 70 to ≈ 77–81 session-h (15 % contingency included). At 26 usable h/week that is about 3 weeks; D-042's cut order applies if a week overruns.
- Why: Pre-registered rule (D-070) and pre-approved fallback (D-042/Q3).
- Tradeoff accepted:
  - Llama's endpoints see 1/3 of the paper's training passes, so the u–h gap may be smaller than in the paper. ΔB and the benchmark deltas are computed on the same endpoints, so the correlation test stays internally consistent.
  - Tier 2 keeps 3 epochs (small models, D-009), so tiers differ in epochs.
- Cost impact: Llama training 6.6 h instead of 19.4 h at 3 epochs; the probe cost ≈ 0.6 session-h.
- Paper deviation: **yes**, App. C epochs (3 → 1) for Llama-3.1-8B, plus the batch layout (no deviation in effective batch).
- Revisit-if: real Llama training drifts > 15 % from the projection (D-070), or the weekly quota grows enough to afford 3 epochs (≈ +12.8 h).

## D-072 M3b Tier 2 training design and the pre-registered session rule (implements D-008, D-009, D-033; amends D-041 store path, D-067)
- Date: 2026-10-01 (written before any Tier 2 GPU data)
- Context: M3b trains Qwen2.5-0.5B (full s0, LoRA s0, full s1, full s2) and Llama-3.2-1B (full s0, LoRA s0), each u‖h on the two T4s, at 3 epochs (Tier 2 keeps App. C epochs, D-071).
  - Full FT of a 1.2B model holds fp32 masters, gradients and 8-bit optimizer state (≈ 12.4 GB, D-009). Its trainer checkpoints are ≈ 7.5 GB per job.
  - Tier 2 throughput is unmeasured.
  - Llama-3.2-1B ties `lm_head` to `embed_tokens`.
- Choices:
  1. **Batch-layout fallback:** `train.batch_layouts` (per regime) lists [b, g] pairs with b × g = 32; Tier 2 uses 4×8 → 2×16 → 1×32.
     - Each layout has its own `train_key` and job directory.
     - A CUDA OOM writes `oom.json` there and `run_job` retries the next layout with a freshly loaded model; any other error propagates.
     - The job resolves to the first layout with `done.json`. The effective batch is unchanged (App. C).
     - `group_by_length` puts the longest batch first, so an OOM appears in the first steps.
  2. **Checkpoints:** `train.checkpoint_root: {full: ephemeral}` writes full-FT trainer checkpoints to `<ephemeral>/rbbd_ckpt/<slug>/<train_key>`. A crash can then resume within the session but not across sessions.
     - The finals stay in `/kaggle/working`.
     - LoRA checkpoints stay in the job directory.
     - Train subprocesses set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (fragmentation only; results unchanged).
  3. **Seeds:** `train.seeds_by_regime` gives the seeds ablation (D-033) to full FT only. Jobs run **seed-major** (every regime at seed 0 first), so a short session loses extra seeds before core endpoints (D-042 cut order).
  4. **Store:**
     - With `train.sync_each_pair: true`, each finished u‖h pair is uploaded right away (`sft.sync_jobs`). Failures are logged, not raised.
     - A new session runs `cli restore --path train/<slug>` (`HFStore.download_dir`, private check first) to bring back finished jobs, which the stage then skips.
     - Full-FT endpoints are stored at their job path `train/<slug>/full/<split>/seed<k>/<train_key>/final/` in the private repo. This replaces D-041's `ckpt_private/tier2/…` layout: one layout for every regime.
  5. **Tied weights:** `interpolate.apply_alpha` accepts endpoints without a tied alias (`lm_head.weight`), writes the stored tensor, and the alias follows. Any other missing key still raises.
  6. **Session probe and rule (pre-registered):**
     - Each M3b session first runs `configs/m3b_probe_<model>.yaml`: full FT, u‖h, 30 steps or 8 min, separate `*-probe` slug, never synced.
     - **Bound** = the harmful probe's `projected_hours["3_epochs"]` × pending u‖h pairs. LoRA and the unharmful split are bounded by the harmful full-FT rate.
     - The long run starts only if the bound is ≤ 11 h (12 h session minus setup, tests and sync).
     - Otherwise the notebook stops and I ask the user. Options would include D-042 cut item 2 (one extra seed instead of two) or a cross-session plan.
  7. **DC-07 on real Tier 2 endpoints:** `test_real_full_endpoints` (exact α ∈ {0, 1} on the GPU model, finite α = 0.5 forward) and `test_real_tier2_lora_endpoints`, selected by `RBBD_M3B_CONFIG`.
  8. **Execution:** `notebooks/m3b_train.ipynb` with `MODEL` = `qwen2.5-0.5b` (session 1) or `llama3.2-1b` (session 2), run as *Save & Run All* so it needs no open browser.
- Why: The plan fits inside one session per model, survives a dead session at the cost of at most one pair, and never fills `/kaggle/working`.
- Tradeoff accepted: A crash mid-pair in full FT restarts that pair from step 0 in the next session (its checkpoints were ephemeral). The probe's 8 minutes are a small sample.
- Cost impact: ≈ 0.2 h per session for probe and setup. Training per D-055 ≈ 12.3 session-h for core Tier 2; the probes replace that estimate.
- Paper deviation: no (Tier 2 is already an analogue, D-009).
- Revisit-if: a session bound exceeds 11 h, or a full-FT layout fails even at 1×32 (then D-009's FSDP fallback).
