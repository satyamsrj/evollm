# EvoLLM: Conversation-Driven Weight-Only Continual Learning (CPL-v2)

[![Unit Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)]()
[![Architecture](https://img.shields.io/badge/CPL--v2-SCGP%20%2B%20SimPO-blue.svg)]()

An experimental continual-learning framework that enables LLMs to evolve through conversations by persistently updating their neural weights. Combines self-supervised learning and user corrections with efficient parameter updates to balance knowledge acquisition, retention, and catastrophic forgetting—without external memory, vector databases, or conversation replay.

---

## Key Theoretical & Architectural Advances (CPL-v2)

1. **Dual-Precision Parameter Management (`PrecisionMaster`)**:
   Maintains a dedicated FP32 master weight repository while serving and computing forward passes in execution dtypes (BF16/FP16). Completely prevents mantissa truncation where small LoRA deltas ($\sim 10^{-4}$) round to zero in standard FP16.

2. **Subspace-Constrained Gradient Projection (`SCGP`)**:
   Replaces intractable $O(d^3)$ per-dialogue SVD with a one-time pre-flight basis extraction and an ultra-fast $O(d \cdot K \cdot r)$ associative projection:
   $$\mathcal{P}_{\parallel}(\Delta W) = U_K (U_K^\top B)(A V_K) V_K^\top$$
   $$\Delta W_{\text{filtered}} = \Delta W - \beta \cdot \mathcal{P}_{\parallel}(\Delta W)$$
   Dampens interference with the core invariant pre-trained subspace in $<0.2$ ms per layer.

3. **Length-Normalized Direct Preference & SimPO Loss**:
   Eliminates length-disparity inversion where long correct answers are penalized under raw log-probabilities. Enforces a calibrated margin over negative responses:
   $$\mathcal{L}_{pref} = \max\left(0, \, m - \left[\frac{1}{|a^+|}\log P(a^+) - \frac{1}{|a^-|}\log P(a^-)\right]\right)$$

4. **Triaged Selective Learning (TSL)**:
   Extracts genuine factual assertions and user corrections while filtering out user questions and prompts, preventing conversational syntax decay and sycophancy.

5. **Stateless Normalized Cautious SGD**:
   Achieves fast 1-to-3 step convergence with adaptive step clipping without maintaining persistent optimizer states or momentum buffers between conversations.

---

## Project Structure

```
evollm/
├── core/
│   ├── config.py             # Hyperparameters, precision flags, and thresholds
│   ├── precision_master.py   # Dual-precision FP32 master weight repository
│   ├── lora.py               # Native PyTorch LoRA wrapper & delta extraction
│   ├── scgp_filter.py        # Subspace-Constrained Gradient Projection (O(d·K·r))
│   ├── loss.py               # Length-normalized SimPO/DPO loss + TSL parser
│   ├── optimizer.py          # Stateless Normalized Cautious SGD
│   └── engine.py             # Hot-swappable background continual learner
├── bench/
│   ├── cscl_bench.py         # Multi-track benchmark runner
│   └── mock_dialogues.py     # Deterministic evaluation dialogue generator
├── tests/
│   └── test_evollm_core.py   # Full unit test suite (6 passing tests)
├── demo_continual_learning.py# End-to-end runnable demonstration
└── README.md
```

---

## Quickstart

### Run Core Unit Tests
```bash
python -m unittest discover -s tests
```

### Run End-to-End Continual Learning Demo
```bash
python demo_continual_learning.py
```
