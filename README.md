# Protein Embedding Classifier (PEC)

A research framework for protein function prediction using precomputed embeddings.

**Important:** This repository does **not** generate embeddings. It consumes precomputed embeddings as feature vectors for supervised classification.

---

## Repository Branch Strategy

This repository uses a **documentation-first** branch model to maintain stability and clarity.

### Branch Overview

| Branch | Purpose | Status |
|--------|---------|--------|
| `master` | **Stable, minimal branch** - Contains only high-level vision documents, contracts, and basic repository metadata. This is the canonical reference for project architecture and specifications. | Protected |
| `pre-embedding-dev` | **Active development** - Implementation of the pre-embedding dataset layer according to [PEC Pre-Embedding Dataset Contract v0.1](docs/PEC-Pre-Embedding-Dataset-Contract-v0.1.md). All current implementation code lives here. | Active |

### Future Branch Placeholders

The following branches are planned but not yet created:
- `embeddings-dev` - Embedding loading and processing layer
- `classifier-dev` - Classifier training and evaluation layer
- `ensemble-dev` - Ensemble methods and voting strategies
- `benchmark-dev` - Benchmark orchestration and reporting

### How to Contribute

1. **For new feature development:**
   - Branch from `pre-embedding-dev` (or the appropriate `-dev` branch for your component)
   - Create a feature branch: `feature/<component>-<description>`
   - Submit PR to the relevant `-dev` branch

2. **For contract/vision updates:**
   - Propose changes via PR to `master`
   - Ensure all `-dev` branches are updated to reflect contract changes

3. **For bug fixes:**
   - Identify the affected component branch
   - Submit PR to the appropriate `-dev` branch

### Current Implementation Status

- **Pre-embedding dataset layer:** Development in progress on `pre-embedding-dev` branch
- **Embedding layer:** Not yet started
- **Classifier layer:** Not yet started
- **Ensemble layer:** Not yet started

---

## Project Documentation

### Vision and Architecture

- [PEC Final-System Functional Vision](docs/PEC-Final-System-Functional-Vision.md) - High-level system architecture and responsibilities
- [PEC Pre-Embedding Dataset Contract v0.1](docs/PEC-Pre-Embedding-Dataset-Contract-v0.1.md) - Contract for the pre-embedding dataset layer

### Key Principles

1. **Configuration-first execution** - All behavior is driven by explicit configuration
2. **Dataset plurality** - Support for multiple dataset variants and organizations
3. **Artifact traceability** - Every major artifact is self-descriptive with mandatory lineage
4. **Reproducibility** - Fixed seeds, deterministic operations, and manifest-based orchestration

---

## Quick Start

### For Developers

1. Clone the repository:
   ```bash
   git clone https://github.com/alexdorocode/protein-embedding-classifier.git
   cd protein-embedding-classifier
   ```

2. Switch to the development branch:
   ```bash
   git checkout pre-embedding-dev
   ```

3. Install dependencies (see `pyproject.toml` on the `pre-embedding-dev` branch)

4. Begin development according to the contract specifications

### For Reviewers

All vision documents and contracts are available on the `master` branch in the `docs/` directory. The `master` branch contains no implementation code - only the architectural specifications that define what the system should do.

---

## Repository Structure (master branch)

```
protein-embedding-classifier/
├── README.md                    # This file - branch strategy and overview
├── .gitignore                  # Git ignore patterns
├── docs/
│   ├── PEC-Final-System-Functional-Vision.md    # System architecture vision
│   └── PEC-Pre-Embedding-Dataset-Contract-v0.1.md  # Dataset layer contract
└── artifacts/                  # Generated artifacts (ignored by git)
```

### Implementation Branches

The `pre-embedding-dev` branch contains:
- Full Python package: `protein_embedding_classifier/`
- Configuration files: `config/`
- Test suite: `tests/`
- Dependency files: `pyproject.toml`, `poetry.lock`, `pytest.ini`
- All implementation code for the pre-embedding dataset layer

---

## License

This project is licensed under the terms specified in the LICENSE file (to be added).

## Contact

For questions about the project architecture, refer to the vision and contract documents in `docs/`.

For implementation questions, refer to the `pre-embedding-dev` branch or create an issue.
