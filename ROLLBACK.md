# ROLLBACK.md

**Purpose.** Makes every milestone and every risky edit reversible, both in code and in data. Code is rolled back with git tags. Data is never deleted to roll back: the affected cache namespace is invalidated by bumping a config `schema_version` or the config hash, so new keys are written beside the old ones (ARCHITECTURE §4).

## Procedure

**At the end of every milestone:**
1. All applicable DONE.md checks pass.
2. `git tag m<N>-green && git push origin m<N>-green` (sub-milestones: `m3a-green`, …).
3. Add a row to the *Milestone tags* table: tag, commit SHA, Kaggle notebook name + version, the store commit id of `SarthakGupta414/rbbd-artifacts` after the milestone's last `sync` (D-041), date.

**Before any large or risky edit** (touching >3 modules, changing cache-key fields, pooling, metrics, merge logic, pins, or anything a DONE check depends on):
1. Add a row to *Pre-edit rollback targets*: date, planned change, rollback tag (the latest `m*-green`), files expected to change, cache namespaces that the change will invalidate.
2. Put the change in `FLOW.md › Currently modifying`.

**To roll back:**
1. `git checkout <tag> -- <files>` for a partial restore, or `git revert <commits>` on the working branch for a full one. Never force-push shared history.
2. Point notebooks at the tag: `git clone --branch <tag>`.
3. Download artifacts at the recorded store commit (`revision=<store commit>` in `hf_hub_download`/`snapshot_download`).
4. Invalidate the affected cache namespaces by bumping `schema_version` (embed / metrics / gen / score), or by reverting the config that changed. Do not delete artifact directories.
5. Re-run the DONE checks listed for that namespace (below). Record the rollback as a BUGS.md entry and, if it reverses a decision, as a superseding DECISION.md entry.

**Namespace → checks to re-run after invalidation**

| Namespace (config key) | Invalidated by | Re-run checks |
|---|---|---|
| `sentences` (set hashes) | any sentence-file edit | DC-17, DC-03–05, DC-11 |
| `embed.schema_version` | pooling, layer site, tokenizer settings, precision/placement | DC-06, DC-09, DC-10, DC-11, DC-12 |
| `train` (train_key) | hyperparameters, data filters, seeds | DC-07, DC-13, DC-11 |
| `metrics.schema_version` | RR/SEAT/Procrustes/CKA/ΔB code | DC-03, DC-04, DC-05, DC-11 |
| `gen.schema_version` | prompt building, sampling, engine | DC-11, DC-14, DC-18 |
| `score.schema_version` | scorer parsing or aggregation | DC-11 (bench parity tests in DC-02) |
| `analysis` | stats definitions | DC-08 |

**Pruning.** Old artifact keys may be deleted only after the user approves (CLAUDE.md action boundaries), and only in a commit that records what was pruned here.

## Milestone tags

| Tag | Commit | Kaggle notebook (version) | Store commit (rbbd-artifacts, D-041) | Date |
|---|---|---|---|---|
| *(none yet — `m0-green` waits for the Kaggle probe run)* | | | | |

## Pre-edit rollback targets

| Date | Planned change | Rollback tag | Files expected to change | Namespaces invalidated | Outcome |
|---|---|---|---|---|---|
| 2026-09-30 | M0: first implementation of cli/config/runner/utils + notebooks (touches > 3 modules) | none yet — commit `2df82d9` (docs-only state before M0 code) | `src/rbbd/{cli,config,runner}.py`, `src/rbbd/utils/*`, `configs/{base,smoke}.yaml`, `notebooks/*`, `tests/*`, `.gitignore`, `pyproject.toml` untouched | none (no artifacts exist yet) | CPU checks green; Kaggle pending |
