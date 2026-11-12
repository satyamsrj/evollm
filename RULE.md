# Repository Rules & Engineering Guidelines (RULE.md)

This document defines the non-negotiable operational, architectural, and version-control rules for the `evollm` repository. All contributors and automated agents must adhere strictly to these mandates.

---

## 1. Zero Unnecessary Comments & Clean Codebase Mandate

- **No Redundant or Verbose Comments**: Do not write comments that restate what the code clearly expresses (e.g., `# set x to 1`, `# return result`). Comments are permitted only when explaining non-obvious mathematical derivations, hardware constraints, or algorithm invariants.
- **Self-Documenting Code**: Write expressive, idiomatic variable and function names. Rely on Python type annotations (`typing`) rather than prose descriptions of types.
- **No Dead or Commented-Out Code**: Never commit commented-out code blocks, abandoned experiments, or `TODO` placeholders. Unused code must be removed immediately.

---

## 2. Professional Directory Structure & Clean Repository Organization

- **Strict Modularity**: Maintain a professional, clean folder hierarchy:
  - `evollm/core/`: Production-grade mathematical engines, optimizers, adapters, and filters.
  - `evollm/bench/`: Authoritative benchmark evaluation suites and metrics engines.
  - `tests/`: Automated unit and integration test suites.
- **Zero Orphan or Temp Files**: No scratch scripts, temporary artifacts, local caches, debug logs, or ad-hoc dumps are allowed in the repository root or source packages.
- **Ignored Artifacts**: Ensure `.gitignore` actively excludes:
  - Python bytecode (`__pycache__/`, `*.py[cod]`)
  - Virtual environments (`venv/`, `.venv/`, `env/`)
  - Checkpoint dumps, logs, and OS-generated files (`.DS_Store`, `Thumbs.db`).

---

## 3. Atomic GitHub Push & Individual File Commit Protocol

- **Push Only on Explicit User Command**: Do NOT automatically commit or push to GitHub after creating or modifying files. Commits and pushes are executed only when the user explicitly instructs it (e.g., "push to GitHub", "commit and push"). Writing a file to disk does not trigger a push.
- **Push Files Individually**: When instructed to push, do not bundle disparate changes across multiple files into a single bulk commit. Stage, commit, and push each file separately in sequence.
- **No Blanket `git add .`**: Never run indiscriminate bulk staging (`git add .` or `git add -A`). Stage files individually:
  ```bash
  git add <specific-file>
  git commit -m "<type>(<scope>): <precise description>"
  git push origin <branch>
  ```
- **Semantic, Atomic Commit Messages**: Each commit must document the exact modification made to that specific file (e.g., `feat(core): implement dual-precision master weight manager in precision_master.py`).
- **Clean Git History**: Keep commit trees linear, well-annotated, and auditable.

---

## 4. Zero Fake, Mock, or Synthetic Data & Zero Unverified Assumptions

- **Strict Prohibition Against Synthetic / Mock Data**: Never create, seed, hardcode, or fall back to dummy, mock, placeholder, or synthetic data in production pipelines, weight updates, databases, or API contracts.
- **Authoritative Provenance Required**: All data processed by continual learning engines must originate from verified user dialogues, authoritative real-world corpora, or user-supplied ground-truth corrections.
- **Zero Unverified Assumptions**: Never make inferences or synthetic guesses regarding missing attributes, parameters, or domain facts. 
- **Mandatory Fail-Closed Policy**: When authoritative data cannot be validated or parsed:
  - **Do NOT** substitute approximate or synthetic values.
  - **Halt and Return**: Emit explicit `SOURCE_UNAVAILABLE` or `DATA_UNVERIFIED` state and terminate dependent execution.

---

## 5. Professional Production Standards

- **Deterministic Execution**: All stochastic operations must allow deterministic seeding for reproducibility.
- **Strict Typing**: All core functions and methods must have full type annotations.
- **Comprehensive Verification**: Any change to mathematical operators, projection matrices, or loss functions must be backed by a corresponding unit test in `tests/` before pushing.

