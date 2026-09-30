# CLAUDE.md — standing rules for the coding agent

Project: Kaggle replication of *Reference-Based Bias Detection in LLMs via Relative Representations of Hidden States* (arXiv:2609.10060v1, `paper/2609.10060v1.pdf`). Read `PLAN.md` first. The authors' code is not released, so everything is built from the PDF. Every unreleased detail is a logged reconstruction.

## The seven working rules (apply to every change)

1. **DECISION.md.** Log every meaningful decision in the **same commit** as the change it justifies.
   - Format: `## D-### <title>` with Date, Context, Options considered, Choice, Why, Tradeoff accepted, Cost impact (GPU-minutes, disk, memory), Paper deviation (yes/no + section), Revisit-if.
   - To replace an entry, add a new one and mark the old heading `— superseded by D-###`. Never delete entries.
2. **Comment the flow, not the syntax.**
   - Comments on non-obvious logic say what the block is for, what calls it, what it assumes already exists (shapes, dtypes, device, padding side, cache keys), and which paper equation or section it implements.
   - Every public function has a docstring with tensor shapes.
   - Never write comments that restate the code.
3. **FLOW.md.** Keep the CLI → runner → module call path current, one section per stage, with the artifact passed at each boundary. While a change is in progress, `## Currently modifying` names the exact functions and files on the changed path. Reset it to `nothing` when done.
4. **BUGS.md.** One entry per bug or feature, start to finish: `## B-### <title>` with Status, How found/scoped, Reproduction command, Hypotheses tried (what failed and why), Fix, Verification (command + observed output), GPU-hours lost, Linked commits and D-### entries.
5. **ARCHITECTURE.md.** High-level map only: modules, single responsibility, data movement, artifact layout, cache keys. Update it only when the shape changes.
6. **DONE.md.** Done means every applicable `DC-##` check passes with its stated expected output. Nothing is done on judgment. Placeholder tests (`PLACEHOLDER(M<N>)`) never count as passing.
7. **ROLLBACK.md.**
   - End every milestone with tag `m<N>-green`, recorded with the Kaggle notebook version and Dataset version.
   - Before any large or risky edit, record the rollback target.
   - Roll back via the tag, restore files, invalidate caches by bumping `schema_version` or the config hash (never by deleting data), then re-run the listed DONE checks.

## Tiers
| Tier | What | Models | Budget |
|---|---|---|---|
| 0 Smoke | Whole pipeline on tiny sets, **< 15 min** on T4 | Qwen2.5-0.5B-Instruct | ≈ 6 session-h total |
| 1 Primary | QLoRA replication of T1 LoRA columns | Llama-3.1-8B-Instruct, Mistral-7B-Instruct-v0.3, Gemma-3-4B-it | ≈ 56 session-h |
| 2 Full-FT analogue | Full FT **and** LoRA on small models; seeds | Qwen2.5-0.5B-Instruct, Llama-3.2-1B-Instruct (stretch: Qwen2.5-1.5B, Gemma-3-1B) | ≈ 17 session-h |

## Kaggle constraints (verify at runtime; never assume)
- 2× T4 16 GB, sm75, **no bf16**. P100 is not used (vLLM unsupported). Verify with `python -m rbbd.cli probe` (`nvidia-smi`, `df -h`, `free -g`).
- Sessions stop at ~12 h. The quota is 28 session-h/week (D-042) and a 2-GPU session is billed once. `/kaggle/working` persists ~20 GB; everything else is lost.
- Every stage writes a `manifest.json` and skips itself on a valid manifest. Training and generation checkpoint partial progress.
- **fp16 overflow:** every hidden-state extraction runs the NaN/Inf guard and fails loudly. Gemma runs in fp32 for inference (D-005).
- **Never store merged full-weight checkpoints persistently.**
  - LoRA: store the two adapters. α-checkpoints are exact concatenated rank-2r adapters built in memory (D-007).
  - Full FT: store the two endpoints privately and interpolate in memory as (1−α)W_h + αW_u (D-008).
- Reference and audited checkpoints are always extracted with **identical precision, quantisation and placement** (D-004).
- Secrets: `HF_TOKEN` (read for gated repos + write to the private store) and optional `GITHUB_TOKEN` (clone only) come from Kaggle Secrets. Never print, log or commit them.
- Artifact store: private HF repo `SarthakGupta414/rbbd-artifacts` (D-041). Upload only via `HfApi.upload_folder` after checking `private is True`. Never call `create_repo`, change repo visibility, or `push_to_hub`.
- Harmful-trained adapters and checkpoints, generations and per-prompt scores go to **private** stores only. Git and `results/` hold aggregates only. Never generate new harmful content (D-010, D-037).

## Action boundaries
- Proceed without asking: reading, editing code and docs, CPU tests.
- **Ask first:** installing packages outside the pinned `pyproject.toml`, running on GPU, downloading weights, any network call not in the plan, deleting any file or artifact, making anything public.

## File map
| Path | What |
|---|---|
| `PLAN.md` | Plan: summary, reconstruction audit, milestones M0–M8, budget, risks, results template |
| `DECISION.md` | Decision log (D-###) |
| `ARCHITECTURE.md` | Module map, data flow, artifact layout, cache keys |
| `FLOW.md` | Execution path per stage; `Currently modifying` |
| `BUGS.md` | Bugs/features/pre-registered risks (B-###) |
| `ROLLBACK.md` | Rollback procedure, milestone tags, pre-edit targets |
| `DONE.md` | Definition-of-done checks (DC-##) |
| `pyproject.toml` | Package + pinned extras `train`, `bench`, `dev` |
| `configs/` | `base.yaml`, `smoke.yaml`, tier configs, `ablations.yaml` |
| `src/rbbd/` | Package: `cli`, `config`, `runner`, `utils/`, `data/`, `resources/`, `models/`, `embed/`, `metrics/`, `finetune/`, `bench/`, `analysis/` |
| `tests/` | CPU tests; `tests/gpu/` marked `gpu` |
| `notebooks/` | Thin Kaggle notebooks: clone@tag → install → secret → CLI |
| `paper/` | The PDF. **Read-only; never modify.** |

## Commands
- Lint: `ruff check src tests`
- CPU tests: `pytest -q -m "not gpu"`
- Probe: `python -m rbbd.cli probe`
- Run: `python -m rbbd.cli run --config configs/<x>.yaml [--stages ...]`
- Status: `python -m rbbd.cli status --config configs/<x>.yaml`

## Conventions
- α = weight on the **unharmful** endpoint. Checkpoints: `ref`, `a100` (unharmful), `a090`, `a070`, `a050`, `a030`, `a010`, `a000` (harmful).
- Negative ΔB = the group moved toward negative attributes = increased bias (§3.4).
- Cite the paper as `§x.y`, `T#`, `F#`, `App. X` in docstrings and D entries. Mark gaps `[unspecified in paper]`.
