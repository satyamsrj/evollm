# EvoLLM Implementation Plan: Evidence-Gated Weight-Only Continual Learning Across 5 Architectures

> **Execution Environment**: Windows 11 · CPU (`torch.float32` uniform compute + `float32` `PrecisionMaster` scoped to adapted `target_modules`) · 16.79 GB Host RAM  
> **Scope**: 5 Distinct Open-Weight LLM Architectures × 5 Methods (`B0`–`B4`) × 100 Continual Learning Events across 10 Verified HuggingFace Corpora + 6 Held-Out Evaluation Probes  
> **Integrity Mandate**: Strictly enforces `RULE.md` — zero synthetic/mock/fallback metrics, SHA-256 manifest verification (`evollm/data/slicer.py`), fail-closed provenance (`SOURCE_UNAVAILABLE`), and direct empirical telemetry grounding (`metrics.jsonl` / `checkpoint_state.json` / `eval_summary.json`).

---

## 1. System Architecture & Core Components

1. **Evidence Qualification Engine (`evollm/core/evidence.py`)**:
   - Classifies each incoming dialogue turn into five categories (`CORR`, `KNOW`, `PREF`, `HYPO`, `QURY`, plus `SSL` when verified source provenance is present).
   - Enforces a fail-closed provenance gate: assertions lacking `source_reference` fail closed as `SOURCE_UNAVAILABLE` ($\nabla_\theta = 0$), and unverified conversational dialogue (`wildchat`) is rejected in `B4` while absorbed by ungated baselines (`B1`, `B2`, `B3`).
2. **Per-Class Multi-Objective Routing (`evollm/core/loss.py`)**:
   - Routes `EXPLICIT_CORRECTION` ($\mathcal{C}_{corr}$) to length-normalized SimPO preference margin loss ($\mathcal{L}_{CORR}$).
   - Routes `VERIFIED_FACT` ($\mathcal{C}_{know}$) to target-masked cross-entropy ($\mathcal{L}_{KNOW}$).
   - Routes verified educational/general text ($\mathcal{C}_{ssl}$) to self-supervised loss ($\mathcal{L}_{SSL}$).
3. **Associative Union SCGP Spectral Filter (`evollm/core/scgp_filter.py`)**:
   - Pre-computes rank-$K$ singular subspaces ($U_K, V_K$) once at $t=0$ via `torch.linalg.svd` (with randomized fallback for $\min(d_{\text{out}}, d_{\text{in}}) > 2048$).
   - Safely caps $K = \min(\text{top\_k}, \max(1, \lfloor 0.25 \cdot \min(d_{\text{out}}, d_{\text{in}}) \rfloor))$ to protect narrow GQA `k_proj`/`v_proj` layers.
   - Evaluates the union subspace projection associatively in $\mathcal{O}(d \cdot K \cdot r)$ time:
     $$\mathcal{P}_{\parallel}(\Delta W) = \frac{\gamma}{r}\Big[U_K(U_K^\top B)A + B(A V_K)V_K^\top - U_K(U_K^\top B)(A V_K)V_K^\top\Big]$$
     $$\Delta W_{\text{clean}} = \text{clip}_{\text{Fro}}\big(\Delta W - \beta \cdot \mathcal{P}_{\parallel}(\Delta W), \rho\big)$$
   - Logs directional alignment $\cos\phi$ and `core_energy_rejected_pct` on every update.
4. **Target-Scoped FP32 PrecisionMaster & Stateless NSGD (`evollm/core/precision_master.py`, `evollm/core/optimizer.py`, `evollm/core/engine.py`)**:
   - Maintains FP32 master weights exclusively for `target_modules` (reducing host RAM overhead by >90% on MoE and 350M–1.3B models), records `fp32_underflow_ratio` (share of non-zero delta elements with $|x| < 10^{-4}$), and syncs updates back to the serving model with zero persistent optimizer-state memory ($\Omega_{\mathrm{opt}} = 0$) for `B1`/`B3`/`B4`.
   - Enforces exact leaf-name LoRA wrapping (`evollm/core/lora.py`) so `"dense"` does not match `dense_h_to_4h` and MoE routers remain frozen.

---

## 2. Evaluated Model Architectures (`experiments/scale_5arch_15gb/models.yaml`)

| Model Key | HuggingFace ID | Params (Active / Total) | Family | Adapted Target Modules (Leaf Exact Match) | LoRA Rank / Alpha |
| :--- | :--- | :---: | :--- | :--- | :---: |
| `pythia-160m` | `EleutherAI/pythia-160m` | 160M | GPT-NeoX | `query_key_value`, `dense` | `8 / 16.0` |
| `opt-350m` | `facebook/opt-350m` | 350M | OPT | `q_proj`, `k_proj`, `v_proj`, `out_proj` | `8 / 16.0` |
| `smollm2-360m` | `HuggingFaceTB/SmolLM2-360M-Instruct` | 360M | Llama (GQA) | `q_proj`, `k_proj`, `v_proj`, `o_proj` | `8 / 16.0` |
| `qwen2.5-0.5b` | `Qwen/Qwen2.5-0.5B-Instruct` | 494M | Qwen2 (GQA) | `q_proj`, `k_proj`, `v_proj`, `o_proj` | `8 / 16.0` |
| `granite-moe-1b` | `ibm-granite/granite-3.1-1b-a400m-instruct` | 400M act / 1.3B tot | GraniteMoE | `q_proj`, `k_proj`, `v_proj`, `o_proj` | `8 / 16.0` |

---

## 3. Experimental Methods & Calibrated Hyperparameters (`experiments/scale_5arch_15gb/config.yaml`)

- **Shared Settings**: `device: "auto"` (CPU fallback), `compute_dtype: "float32"`, `master_dtype: "float32"`, `steps_per_conv: 4`, `max_seq_len: 128`, `eval_probe_sample_limit: 20`, `eval_checkpoints: [0, 50, 100]`, `max_events: 100`.
- **Method Taxonomy (`build_method_config` in `run_stream.py`)**:
  - **B0 (`Frozen Base`)**: No updates (`θ_t = θ_0`).
  - **B1 (`Naive LoRA SGD`)**: Stateless NSGD (`base_lr_ssl=1.0e-2`, `base_lr_correction=2.0e-2`), `subspace_suppression=0.0`, `frobenius_clip_rho=1e6`, `max_step_drift=1e6`, no EQE gating.
  - **B2 (`Standard LoRA AdamW`)**: Stateful `AdamW` (`base_lr_ssl=5.0e-5`, `base_lr_correction=1.0e-4`, `lambda_global_reg=1e-3`), `subspace_suppression=0.0`, `frobenius_clip_rho=1e6`, `max_step_drift=1e6`, no EQE gating.
  - **B3 (`EvoLLM w/o Gating`)**: Stateless NSGD (`base_lr_ssl=1.0e-2`, `base_lr_correction=2.0e-2`), SCGP active (`scgp_top_k=128`, `subspace_suppression=0.90`, `frobenius_clip_rho=0.5`, `max_step_drift=0.25`), all turns routed as `L_SSL` without EQE gating.
  - **B4 (`EvoLLM Full / CPL-v2`)**: Stateless NSGD (`base_lr_ssl=1.0e-2`, `base_lr_correction=2.0e-2`), SCGP active (`scgp_top_k=128`, `subspace_suppression=0.90`, `frobenius_clip_rho=0.5`, `max_step_drift=0.25`), 5-class EQE active with fail-closed provenance gating (`SOURCE_UNAVAILABLE` on unverified `wildchat`).
- **Contamination & Resume Guarantees (`run_stream.py`)**:
  - Asserts `sha256(theta_start) == sha256(theta_0)` across all methods per model.
  - Reuses the verified `t=0` baseline evaluation across `B0`–`B4` of the same model.
  - Restarts incomplete runs (`0 < last_event_idx < max_events`) cleanly from event 0 since intermediate model weights are not serialized.

---

## 4. Datasets & Evaluation Probes (`experiments/scale_5arch_15gb/data_slices/`)

- **10 Training Stream Corpora (100 events balanced across slices)**: `wildchat_train.jsonl`, `wikipedia_train.jsonl`, `fineweb_edu_train.jsonl`, `natural_questions_train.jsonl`, `ms_marco_train.jsonl`, `hotpot_qa_train.jsonl`, `scientific_papers_train.jsonl`, `stackexchange_train.jsonl`, `oasst2_train.jsonl`, `cc_news_train.jsonl` (each paired with a SHA-256 manifest in `data_manifests/`).
- **6 Held-Out Probes (20 samples each per checkpoint)**:
  1. `wikitext103_val.jsonl` → WikiText-103 Perplexity ($\Delta\text{PPL}$)
  2. `arc_challenge_test.jsonl` → ARC-Challenge Accuracy (batched candidate log-likelihood)
  3. `truthful_qa_val.jsonl` → TruthfulQA MC1 Accuracy (`mc1_targets.choices` / `labels`)
  4. `squad_v2_val.jsonl` → SQuAD v2 Token F1 (greedy generation with `max_new_tokens=16`, `stop_strings=["\n"]`)
  5. `mmlu_test.jsonl` → MMLU Accuracy (batched candidate log-likelihood)
  6. `gsm8k_test.jsonl` → GSM8K Exact Match (`max_new_tokens=32`, numeric extraction)
- **Continual Learning Stream Probes**:
  - `KAA (%)`: Uniformly strided generation recall over verified non-`wildchat` events seen up to `step_idx`.
  - `FAR (%)`: Identical evaluation across `B0`–`B4` on unverified/distractor turns (`wildchat` / missing `source_reference` / hypothetical).

---

## 5. Figures, Manuscript Generation & Verification Pipeline

1. **Empirical Plotting (`evollm/paper/figures.py`)**:
   - `generate_all_publication_figures()` reads strictly from `eval_summary.json`, `checkpoint_state.json`, and `metrics.jsonl` (failing closed with `RuntimeError("SOURCE_UNAVAILABLE: ...")` if absent) and exports:
     - `fig1_retention_curves.png` (Relative $\Delta\text{PPL}/\text{PPL}_0$ over $t$)
     - `fig2_pareto_frontier.png` (KAA vs. Foundational Perplexity Retention across all architectures × methods)
     - `fig3_transfer_matrix.png` (Cross-architecture downstream benchmark shift & safety matrix for `B4`)
     - `fig4_evidence_scgp_dynamics.png` (EQE stream routing shares + `core_energy_rejected_pct` & $\cos\phi$)
     - `fig5_cross_arch_radar.png` (6-axis `B4` radar vs. empirical `B1` mean)
2. **Manuscript Compilation (`evollm/paper/generator.py`)**:
   - Populates `research_paper/main.tex`, `research_paper/paper_draft.md`, and `research_paper/references.bib` strictly from completed `eval_summary.json` records (no hardcoded fallback metrics; dynamic Table 4 cross-architecture aggregation via `_compute_cl_summary()`).

---

## 6. Execution Phases & Current Status

- [x] **Phase 1 — Codebase Audit & Core Fixes**: Removed synthetic data/scripts, upgraded SCGP to $\mathcal{O}(d \cdot K \cdot r)$ union projection (`K=128`, `β=0.90`), scoped `PrecisionMaster` to `target_modules`, fixed `B1`/`B2` drift caps & AdamW LR, fixed evaluators (`TruthfulQA`, `GSM8K`, batched MCQ), removed synthetic figure generators, verified `38/38` unit tests.
- [x] **Phase 2 — 20-Event Plasticity & Protection Gate (`Pythia-160M`)**: Verified `B3` (`ΔPPL=+1.13`) and `B4` (`ΔPPL=+0.45`) outperform `B1` (`ΔPPL=+1.36`) with `17.4%` SCGP core rejection.
- [x] **Phase 3a — `pythia-160m` (`B0`–`B4`, 100 events)**: Completed.
- [x] **Phase 3b — `opt-350m` (`B0`–`B4`, 100 events)**: Completed.
- [ ] **Phase 3c — `smollm2-360m` (`B0`–`B4`, 100 events)**: Currently executing (`task-2560`).
- [ ] **Phase 3d — `qwen2.5-0.5b` (`B0`–`B4`, 100 events)**: Next in queue (`--model qwen2.5-0.5b --method all --max_events 100 --force`).
- [ ] **Phase 3e — `granite-moe-1b` (`B0`–`B4`, 100 events)**: Final architecture run (`--model granite-moe-1b --method all --max_events 100 --force`).
- [ ] **Phase 4 — Figure Export, Manuscript Compilation & Final Verification**:
  - Run `generate_all_publication_figures("experiments/scale_5arch_15gb", "experiments/scale_5arch_15gb/reports/figures")` and sync figures to `research_paper/figures/`.
  - Run `PaperGenerator(output_dir="research_paper").compile_all(results=eval_summary)` and verify zero `SOURCE_UNAVAILABLE` entries across all 25 runs.
  - Run `python -m unittest discover -s tests -p "test_*.py"`.
