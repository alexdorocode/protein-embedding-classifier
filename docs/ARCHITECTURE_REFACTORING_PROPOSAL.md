# Architecture Refactoring Proposal - Protein Embedding Classifier

## 📋 Executive Summary

**Objective:** Transform the current Protein Embedding Classifier (PEC) repository into a **modular, maintainable, and production-ready** research platform that can support the entire PhD thesis lifecycle and beyond.

**Current State:** 108 Python files, 134 total code files, ~78MB, with some modular structure but inconsistent organization and naming.

**Target State:** Clean, modular architecture with clear separation of concerns, ready for open-source publication.

**Estimated Effort for First Iteration:** **~25-30 hours** (achievable)

---

## 🔍 1. Current Repository Radiography

### 1.1 Directory Structure Analysis

```
protein-embedding-classifier/ (78MB total)
├── protein_embedding_classifier/ (572KB) - Main PEC module
│   ├── classifiers/ - Model implementations
│   │   ├── base.py, linear.py, mlp_protein_classifier.py
│   │   ├── random_forest.py, registry.py
│   │   └── __init__.py
│   ├── core/ - Core functionality
│   │   ├── embeddings.py
│   │   ├── decision/ - Decision policies
│   │   │   └── decision_policy.py
│   │   └── statistics/ - Statistical tests
│   │       ├── friedman_test.py, nemenyi_test.py
│   │       └── ranking_utils.py
│   ├── data/ - Data handling
│   │   ├── splits/ - Data splitting strategies
│   │   │   ├── base.py, cross_validation.py, independent.py
│   │   │   ├── zero_shot_csv.py, zero_shot_organism.py
│   │   │   └── zero_shot_random.py
│   │   ├── protein_loader.py
│   │   ├── dataset_builder.py
│   │   └── label_loader.py
│   ├── config/ - Configuration
│   │   └── __init__.py
│   └── main.py - Entry point
│
├── pec/ (192KB) - PEC Dataset Module (confusing name)
│   ├── dataset/ - Dataset generation
│   │   ├── generator/ - Dataset generation
│   │   │   ├── generator.py, models.py
│   │   │   └── __init__.py
│   │   ├── export/ - Dataset export
│   │   │   ├── exporter.py, models.py
│   │   │   └── __init__.py
│   │   ├── input/ - Data input
│   │   │   ├── reader.py, normalizer.py, models.py
│   │   │   └── __init__.py
│   │   ├── lineage/ - Dataset lineage
│   │   │   ├── builder.py, models.py
│   │   │   └── __init__.py
│   │   ├── splits/ - Split strategies
│   │   │   ├── strategies.py, models.py
│   │   │   └── __init__.py
│   │   ├── policies/ - Dataset policies
│   │   │   ├── validator.py, models.py
│   │   │   └── __init__.py
│   │   ├── contracts.py - Dataset contracts
│   │   └── __init__.py
│   └── README.md
│
├── dataset_designer_runs/ (61MB) - Dataset runs from mfp-dataset-designer
│   ├── 20260803_0258_7672b947/ - Humans run
│   ├── 20260803_0304_a68aa0bb/ - Model organisms run
│   ├── README.md - Run documentation
│   ├── run_loader.py - Run loading module
│   └── runs_catalog.json - Run catalog
│
├── config/ (72KB) - Configuration files
│   ├── db.yaml, embeddings.yaml, pipeline.yaml
│   ├── problems.yaml, training/
│   ├── model_sweep/ - Sweep configurations
│   │   ├── sweep_config_*.yaml (6 files)
│   └── runs_config.yaml - Runs configuration
│
├── docs/ (68KB) - Documentation
│   ├── PEC-Final-System-Functional-Vision.md
│   ├── PEC-Pre-Embedding-Dataset-Contract-v0.1.md
│   ├── TERMINOLOGY_CLARIFICATION.md
│   └── (new) ARCHITECTURE_REFACTORING_PROPOSAL.md
│
├── tests/ (328KB) - Test suite
│   └── (test files)
│
├── artifacts/ - Artifacts directory
├── .gitignore, pyproject.toml, pytest.ini
├── FUNCTIONAL_CONTRACT.md
├── PEC-IMPLEMENTATION-NOTES.md
├── pec_pre_embedding_dataset_contract_v0_1.md
├── README.md
└── (log files, etc.)
```

### 1.2 Current Module Analysis

#### ✅ **Well-Structured Modules**

| Module | Location | Responsibility | Status |
|--------|----------|----------------|--------|
| Classifiers | `protein_embedding_classifier/classifiers/` | Model implementations | ✅ Good |
| Data Splits | `protein_embedding_classifier/data/splits/` | Splitting strategies | ✅ Good |
| Core Statistics | `protein_embedding_classifier/core/statistics/` | Statistical tests | ✅ Good |
| Dataset Designer Runs | `dataset_designer_runs/` | Dataset runs storage | ✅ Good (new) |

#### ⚠️ **Problematic Modules**

| Module | Issue | Severity |
|--------|-------|----------|
| `pec/` | Confusing name (PEC vs pec), overlaps with `protein_embedding_classifier/` | ⚠️ High |
| `protein_embedding_classifier/core/` | Mixed responsibilities (embeddings + decision + statistics) | ⚠️ Medium |
| `protein_embedding_classifier/data/` | Good structure but overlaps with `pec/dataset/` | ⚠️ Medium |
| Config files | Scattered across multiple directories | ⚠️ Medium |
| Main entry point | `main.py` exists but usage unclear | ⚠️ Medium |

#### 📊 **Code Distribution**

- **Total Python files:** 108
- **Total code files:** 134 (including configs, docs, etc.)
- **Repository size:** 78MB (61MB in dataset_designer_runs)
- **Main modules:** 2 primary (`protein_embedding_classifier/`, `pec/`)

---

## 🎯 2. Target Architecture: Modular Design

### 2.1 Proposed Module Structure

```
protein-embedding-classifier/ (Repository Root)
├── src/ - Main source code (NEW)
│   ├── input/ - Input Module
│   │   ├── csv_loader.py
│   │   ├── db_loader.py
│   │   ├── api_loader.py (for bioinformatics APIs)
│   │   ├── validators.py
│   │   └── __init__.py
│   │
│   ├── dataset_builder/ - Dataset Building Module
│   │   ├── transformers/ - Data transformations
│   │   │   ├── normalizer.py
│   │   │   ├── filter.py
│   │   │   └── __init__.py
│   │   ├── embedding_integration/ - Embedding handling
│   │   │   ├── pooling.py
│   │   │   ├── concatenator.py
│   │   │   └── __init__.py
│   │   ├── generators/ - Dataset generators
│   │   │   ├── target_non_target.py
│   │   │   └── __init__.py
│   │   ├── metadata/ - Run metadata
│   │   │   ├── run_metadata.py
│   │   │   └── __init__.py
│   │   └── __init__.py
│   │
│   ├── training/ - Training Module
│   │   ├── models/ - Model architectures
│   │   │   ├── base.py
│   │   │   ├── mlp.py
│   │   │   ├── random_forest.py
│   │   │   ├── linear.py
│   │   │   ├── xgboost.py
│   │   │   └── __init__.py
│   │   ├── losses.py
│   │   ├── metrics.py
│   │   ├── optimizer.py
│   │   ├── train_loop.py
│   │   ├── wandb_integration.py
│   │   └── __init__.py
│   │
│   ├── prediction/ - Prediction Module
│   │   ├── predictor.py
│   │   ├── batch_predictor.py
│   │   └── __init__.py
│   │
│   ├── explainability/ - Explainability Module
│   │   ├── feature_importance.py
│   │   ├── embedding_saliency.py
│   │   ├── shap_analysis.py
│   │   ├── visualization/
│   │   │   ├── plotter.py
│   │   │   └── __init__.py
│   │   └── __init__.py
│   │
│   ├── output/ - Output Module
│   │   ├── writers/ - Output writers
│   │   │   ├── csv_writer.py
│   │   │   ├── json_writer.py
│   │   │   ├── report_writer.py
│   │   │   └── __init__.py
│   │   ├── results_manager.py
│   │   └── __init__.py
│   │
│   └── orchestration/ - Orchestration Module
│       ├── experiment_definitions.py
│       ├── runner.py
│       ├── sweep_manager.py
│       ├── benchmark.py
│       └── __init__.py
│
├── configs/ - Configuration files (REORGANIZED)
│   ├── datasets/ - Dataset configurations
│   │   ├── humans.yaml
│   │   ├── model_organisms.yaml
│   │   └── template.yaml
│   ├── models/ - Model configurations
│   │   ├── mlp.yaml
│   │   ├── random_forest.yaml
│   │   └── template.yaml
│   ├── experiments/ - Experiment configurations
│   │   ├── benchmark.yaml
│   │   ├── cross_species.yaml
│   │   └── template.yaml
│   ├── embeddings/ - Embedding configurations
│   │   └── go_embeddings.yaml
│   └── sweeps/ - Hyperparameter sweep configurations
│       ├── mlp_sweep.yaml
│       ├── rf_sweep.yaml
│       └── xgb_sweep.yaml
│
├── data/ - Data directory (SYMBOLIC LINKS)
│   ├── datasets/ -> /path/to/actual/datasets
│   ├── embeddings/ -> /path/to/actual/embeddings
│   └── runs/ -> ../dataset_designer_runs/
│
├── datasets/ - Dataset runs (RENAMED from dataset_designer_runs)
│   ├── 20260803_0258_7672b947/ - Humans run
│   ├── 20260803_0304_a68aa0bb/ - Model organisms run
│   ├── README.md
│   ├── run_loader.py
│   └── runs_catalog.json
│
├── notebooks/ - Jupyter notebooks (NEW)
│   ├── exploration/
│   ├── analysis/
│   └── examples/
│
├── tests/ - Test suite (REORGANIZED)
│   ├── unit/
│   │   ├── test_input.py
│   │   ├── test_dataset_builder.py
│   │   ├── test_training.py
│   │   ├── test_prediction.py
│   │   ├── test_explainability.py
│   │   └── test_output.py
│   ├── integration/
│   │   ├── test_pipeline.py
│   │   └── test_orchestration.py
│   └── fixtures/
│       └── test_data.py
│
├── docs/ - Documentation (ENHANCED)
│   ├── ARCHITECTURE.md
│   ├── USAGE.md
│   ├── API.md
│   ├── CONTRIBUTING.md
│   ├── TERMINOLOGY.md
│   └── examples/
│
├── scripts/ - Utility scripts (NEW)
│   ├── train.py
│   ├── predict.py
│   ├── benchmark.py
│   ├── export_dataset.py
│   └── analyze_results.py
│
├── .github/ - GitHub configuration (NEW)
│   ├── workflows/
│   │   ├── ci.yml
│   │   ├── test.yml
│   │   └── docs.yml
│   └── ISSUE_TEMPLATE.md
│
├── pyproject.toml - Project configuration
├── README.md - Main documentation
├── poetry.lock
└── .gitignore
```

### 2.2 Module Responsibilities

| Module | Responsibility | Key Functions | Dependencies |
|--------|---------------|---------------|--------------|
| **Input** | Raw data loading | read_csv, read_db, validate | None |
| **Dataset Builder** | Dataset construction | transform, filter, integrate embeddings | Input |
| **Training** | Model training | train_model, validate, save_checkpoint | Dataset Builder |
| **Prediction** | Model inference | predict, batch_predict | Training |
| **Explainability** | Model interpretation | explain, visualize | Training, Prediction |
| **Output** | Results management | save_results, generate_reports | All |
| **Orchestration** | Experiment management | run_experiment, benchmark | All |

---

## ⚡ 3. Change Classification by Impact/Effort

### 3.1 Essential Changes (≤ 30 hours total)

#### 🟢 **Category A: Directory Restructuring (8-10 hours)**

| Task | Effort | Priority | Description |
|------|--------|----------|-------------|
| Create `src/` directory | 1h | High | New root for all source code |
| Move `protein_embedding_classifier/` → `src/training/` | 2h | High | Rename and reorganize |
| Move `pec/` → `src/dataset_builder/` | 2h | High | Rename and reorganize |
| Create `src/input/` | 1h | High | Consolidate data loading |
| Create `src/prediction/` | 1h | High | Extract prediction logic |
| Create `src/explainability/` | 1h | Medium | Move explainability code |
| Create `src/output/` | 1h | Medium | Consolidate output handling |
| Create `src/orchestration/` | 1h | Medium | Move experiment orchestration |

#### 🟢 **Category B: Code Refactoring (10-12 hours)**

| Task | Effort | Priority | Description |
|------|--------|----------|-------------|
| Refactor `main.py` → CLI entry points | 2h | High | Create clear CLI commands |
| Consolidate config files | 2h | High | Move all configs to `configs/` |
| Create module `__init__.py` files | 1h | High | Proper module exports |
| Update imports across modules | 3h | High | Fix all import paths |
| Create `scripts/` directory | 1h | Medium | Utility scripts |
| Update `pyproject.toml` | 1h | Medium | Add CLI entry points |

#### 🟢 **Category C: Documentation (5-8 hours)**

| Task | Effort | Priority | Description |
|------|--------|----------|-------------|
| Create `docs/ARCHITECTURE.md` | 2h | High | Architecture documentation |
| Create `docs/USAGE.md` | 2h | High | Usage guide |
| Update `README.md` | 2h | High | Main documentation |
| Create module docstrings | 2h | Medium | Code documentation |

**Total Essential: ~25-30 hours** ✅

### 3.2 Intermediate Changes (Future Iterations)

| Task | Effort | Priority | Description |
|------|--------|----------|-------------|
| Add type hints throughout | 10-15h | Medium | Improve code maintainability |
| Expand test coverage | 10-15h | Medium | Add unit and integration tests |
| Optimize embedding loading | 5-8h | Low | Performance improvements |
| Add logging standardization | 3-5h | Medium | Consistent logging |
| Create API documentation | 5-8h | Low | Auto-generated docs |

### 3.3 Advanced Changes (Long-term)

| Task | Effort | Priority | Description |
|------|--------|----------|-------------|
| Web interface/UI | 20-40h | Low | Optional web frontend |
| Docker containerization | 5-10h | Low | Container support |
| CI/CD pipeline | 5-10h | Low | Automated testing/deployment |
| Database integration | 10-20h | Low | Persistent storage |
| Advanced explainability | 10-15h | Low | More techniques |

---

## 📅 4. First Iteration Plan (30 Hours)

### 4.1 Phase 1: Preparation (2 hours)

**Tasks:**
1. ✅ Create backup of current repository
2. ✅ Create new branch: `refactor/modular-architecture`
3. ✅ Document current state (this document)
4. ✅ Define target structure

**Deliverables:**
- Backup created
- Branch created
- This proposal document

### 4.2 Phase 2: Directory Restructuring (8 hours)

**Tasks:**
1. **Create `src/` directory** (0.5h)
   - Create `src/` directory
   - Update `pyproject.toml` to include `src/`

2. **Move and rename `protein_embedding_classifier/` → `src/training/`** (2h)
   - Move directory
   - Update all imports
   - Update `__init__.py` files
   - Rename references in configs

3. **Move and rename `pec/` → `src/dataset_builder/`** (2h)
   - Move directory
   - Update all imports
   - Update `__init__.py` files
   - Rename references in configs

4. **Create new modules** (3.5h)
   - Create `src/input/` with consolidated loading
   - Create `src/prediction/` with prediction logic
   - Create `src/explainability/` with explainability code
   - Create `src/output/` with output handling
   - Create `src/orchestration/` with experiment management

**Deliverables:**
- Clean `src/` directory structure
- All code moved to new locations
- Basic imports working

### 4.3 Phase 3: Code Refactoring (10 hours)

**Tasks:**
1. **Refactor entry points** (2h)
   - Create `scripts/train.py`
   - Create `scripts/predict.py`
   - Create `scripts/benchmark.py`
   - Update `pyproject.toml` with CLI commands

2. **Consolidate configurations** (2h)
   - Move all config files to `configs/`
   - Organize by category (datasets, models, experiments)
   - Update all config references

3. **Update imports** (3h)
   - Fix all import paths across modules
   - Ensure circular dependencies are resolved
   - Test basic imports

4. **Create module interfaces** (3h)
   - Define clear public APIs for each module
   - Create proper `__init__.py` exports
   - Document module interfaces

**Deliverables:**
- Working CLI commands
- Consolidated configuration
- Clean import structure

### 4.4 Phase 4: Documentation (8 hours)

**Tasks:**
1. **Create architecture documentation** (2h)
   - `docs/ARCHITECTURE.md`
   - Module descriptions
   - Dependency diagram
   - Data flow diagram

2. **Create usage documentation** (2h)
   - `docs/USAGE.md`
   - How to train models
   - How to make predictions
   - How to run experiments

3. **Update README** (2h)
   - Project overview
   - Installation instructions
   - Quick start guide
   - Module descriptions

4. **Add code documentation** (2h)
   - Module docstrings
   - Function docstrings
   - Type hints where missing

**Deliverables:**
- Complete documentation set
- Clear usage instructions
- Architecture overview

### 4.5 Phase 5: Testing and Validation (2 hours)

**Tasks:**
1. Run existing tests
2. Fix any broken imports
3. Validate basic functionality
4. Create simple integration test

**Deliverables:**
- All tests passing
- Basic functionality verified
- Integration test created

---

## 🎯 5. Detailed Module Mapping

### 5.1 Current → Target Mapping

#### Input Module

| Current Location | Target Location | Action |
|-----------------|----------------|--------|
| `protein_embedding_classifier/data/protein_loader.py` | `src/input/protein_loader.py` | Move |
| `pec/dataset/input/reader.py` | `src/input/reader.py` | Move |
| `pec/dataset/input/normalizer.py` | `src/input/normalizer.py` | Move |
| `pec/dataset/input/models.py` | `src/input/models.py` | Move |

**New files to create:**
- `src/input/__init__.py`
- `src/input/validators.py`
- `src/input/api_loader.py` (for bioinformatics APIs)

#### Dataset Builder Module

| Current Location | Target Location | Action |
|-----------------|----------------|--------|
| `pec/dataset/` (all files) | `src/dataset_builder/` | Move |
| `protein_embedding_classifier/data/dataset_builder.py` | `src/dataset_builder/builders/` | Move |
| `protein_embedding_classifier/data/label_loader.py` | `src/dataset_builder/label_loader.py` | Move |
| `dataset_designer_runs/run_loader.py` | `src/dataset_builder/run_loader.py` | Move |

**New files to create:**
- `src/dataset_builder/__init__.py`
- `src/dataset_builder/transformers/`
- `src/dataset_builder/embedding_integration/`

#### Training Module

| Current Location | Target Location | Action |
|-----------------|----------------|--------|
| `protein_embedding_classifier/classifiers/` | `src/training/models/` | Move |
| `protein_embedding_classifier/core/embeddings.py` | `src/training/embedding_handler.py` | Move |
| `protein_embedding_classifier/main.py` | `scripts/train.py` | Move & Refactor |

**New files to create:**
- `src/training/__init__.py`
- `src/training/train_loop.py`
- `src/training/wandb_integration.py`
- `src/training/losses.py`
- `src/training/metrics.py`

#### Prediction Module

| Current Location | Target Location | Action |
|-----------------|----------------|--------|
| (Currently in classifiers) | `src/prediction/predictor.py` | Extract |

**New files to create:**
- `src/prediction/__init__.py`
- `src/prediction/batch_predictor.py`

#### Explainability Module

| Current Location | Target Location | Action |
|-----------------|----------------|--------|
| (Currently scattered) | `src/explainability/` | Consolidate |

**New files to create:**
- `src/explainability/__init__.py`
- `src/explainability/feature_importance.py`
- `src/explainability/embedding_saliency.py`
- `src/explainability/visualization/`

#### Output Module

| Current Location | Target Location | Action |
|-----------------|----------------|--------|
| (Currently scattered) | `src/output/` | Consolidate |

**New files to create:**
- `src/output/__init__.py`
- `src/output/writers/`
- `src/output/results_manager.py`

#### Orchestration Module

| Current Location | Target Location | Action |
|-----------------|----------------|--------|
| (Currently in various places) | `src/orchestration/` | Consolidate |

**New files to create:**
- `src/orchestration/__init__.py`
- `src/orchestration/experiment_definitions.py`
- `src/orchestration/runner.py`
- `src/orchestration/benchmark.py`

---

## 📊 6. Effort Estimation by Module

### 6.1 Input Module
- **Current state:** Partially implemented, scattered
- **Target state:** Consolidated, clear API
- **Effort:** 2-3 hours
- **Complexity:** Low
- **Dependencies:** None

### 6.2 Dataset Builder Module
- **Current state:** Well-structured in `pec/`, but confusing name
- **Target state:** Renamed, consolidated with other dataset code
- **Effort:** 3-4 hours
- **Complexity:** Medium (import updates)
- **Dependencies:** Input Module

### 6.3 Training Module
- **Current state:** Well-structured in `protein_embedding_classifier/classifiers/`
- **Target state:** Renamed, expanded with training utilities
- **Effort:** 4-5 hours
- **Complexity:** Medium (refactor main.py)
- **Dependencies:** Dataset Builder, Input

### 6.4 Prediction Module
- **Current state:** Scattered in classifiers
- **Target state:** Consolidated, clear API
- **Effort:** 2-3 hours
- **Complexity:** Low
- **Dependencies:** Training

### 6.5 Explainability Module
- **Current state:** Minimal/nonexistent
- **Target state:** New module
- **Effort:** 3-4 hours
- **Complexity:** Medium
- **Dependencies:** Training, Prediction

### 6.6 Output Module
- **Current state:** Scattered
- **Target state:** Consolidated
- **Effort:** 2-3 hours
- **Complexity:** Low
- **Dependencies:** All modules

### 6.7 Orchestration Module
- **Current state:** Partial, scattered
- **Target state:** Consolidated
- **Effort:** 3-4 hours
- **Complexity:** Medium
- **Dependencies:** All modules

**Total Estimated Effort: ~22-29 hours** ✅ (Within 30-hour target)

---

## ✅ 7. Quality and Maintainability Criteria

### 7.1 Evaluation Checklist

After refactoring, the repository should pass these criteria:

#### ✅ Entenidor (Understandable)
- [ ] New contributor understands each module's purpose in <1 hour
- [ ] Clear README with project overview
- [ ] Module documentation available
- [ ] Examples provided for common tasks

#### ✅ Modular (Modular)
- [ ] Each module has single responsibility
- [ ] Clear dependencies between modules
- [ ] No circular dependencies
- [ ] Easy to add new functionality to existing modules

#### ✅ Escalable (Scalable)
- [ ] Easy to add new datasets
- [ ] Easy to add new models
- [ ] Easy to add new experiments
- [ ] Easy to add new explainability techniques

#### ✅ Mantenible (Maintainable)
- [ ] Comprehensive documentation
- [ ] Clear code structure
- [ ] Type hints where appropriate
- [ ] Consistent coding style
- [ ] Proper error handling

### 7.2 Module Interface Contracts

Each module should expose a clear, minimal public API:

#### Input Module
```python
# Public API
from src.input import (
    CSVLoader,
    DatabaseLoader,
    APILoader,
    DataValidator,
    load_data,
    validate_data
)
```

#### Dataset Builder Module
```python
# Public API
from src.dataset_builder import (
    DatasetBuilder,
    EmbeddingIntegrator,
    DatasetGenerator,
    build_dataset,
    integrate_embeddings
)
```

#### Training Module
```python
# Public API
from src.training import (
    ModelRegistry,
    Trainer,
    TrainingConfig,
    train_model,
    load_model
)
```

#### Prediction Module
```python
# Public API
from src.prediction import (
    Predictor,
    BatchPredictor,
    predict,
    batch_predict
)
```

#### Explainability Module
```python
# Public API
from src.explainability import (
    Explainer,
    FeatureImportance,
    EmbeddingSaliency,
    explain,
    visualize
)
```

#### Output Module
```python
# Public API
from src.output import (
    ResultsWriter,
    ReportGenerator,
    save_results,
    generate_report
)
```

#### Orchestration Module
```python
# Public API
from src.orchestration import (
    ExperimentRunner,
    BenchmarkManager,
    run_experiment,
    run_benchmark
)
```

---

## 🚀 8. Implementation Roadmap

### 8.1 Week 1: Core Refactoring (25-30 hours)
- [ ] Create `src/` directory structure
- [ ] Move and rename modules
- [ ] Update imports
- [ ] Create CLI entry points
- [ ] Consolidate configurations
- [ ] Create basic documentation
- [ ] Test and validate

### 8.2 Week 2: Enhancements (Optional)
- [ ] Add type hints
- [ ] Expand test coverage
- [ ] Add more documentation
- [ ] Optimize performance

### 8.3 Week 3: Advanced Features (Optional)
- [ ] Add explainability techniques
- [ ] Create web interface
- [ ] Add Docker support
- [ ] Set up CI/CD

---

## 📝 9. Risk Assessment

### 9.1 Low Risks
- **Directory restructuring:** Straightforward, mostly file moves
- **Module renaming:** Simple, just update imports
- **Documentation:** Always beneficial

### 9.2 Medium Risks
- **Import updates:** May break existing code if not done carefully
- **Configuration consolidation:** Need to ensure all configs are found
- **CLI refactoring:** Need to test all entry points

### 9.3 Mitigation Strategies
- Create comprehensive backup before starting
- Work in feature branch
- Test frequently during refactoring
- Use git bisect if issues arise
- Document all changes

---

## 🎯 10. Success Metrics

### 10.1 Quantitative Metrics
- **Module count:** 7 clear modules (Input, Dataset Builder, Training, Prediction, Explainability, Output, Orchestration)
- **File count:** ~100-110 Python files (similar to current)
- **Test coverage:** Maintain or improve current coverage
- **Documentation:** 100% of modules documented

### 10.2 Qualitative Metrics
- **New contributor time:** <1 hour to understand the project
- **Task completion time:** <30 minutes to find where to implement a new feature
- **Bug fix time:** <1 hour to locate and fix a bug
- **Onboarding time:** <1 day for new lab member to contribute

---

## 📚 11. Appendix: Current vs Target Comparison

### 11.1 Current Structure Issues

```
❌ PROBLEMS:
├── Confusing naming: pec/ vs protein_embedding_classifier/
├── Scattered configurations: config/ + embedded configs
├── Mixed responsibilities: core/ contains embeddings + decision + statistics
├── No clear entry points: main.py exists but usage unclear
├── Limited documentation: Some modules undocumented
├── No CLI: No clear command-line interface
└── Overlapping functionality: dataset handling in multiple places
```

### 11.2 Target Structure Benefits

```
✅ IMPROVEMENTS:
├── Clear naming: src/input/, src/dataset_builder/, etc.
├── Consolidated configurations: configs/ directory
├── Single responsibilities: Each module does one thing
├── Clear entry points: scripts/train.py, scripts/predict.py
├── Comprehensive documentation: All modules documented
├── CLI interface: Easy command-line usage
└── No overlaps: Clear separation of concerns
```

---

## 🏁 Conclusion

**The proposed refactoring is FEASIBLE within 30 hours** and will transform PEC into a professional, maintainable research platform.

### Key Benefits:
1. ✅ **Clear architecture** that new lab members can understand
2. ✅ **Modular design** that supports long-term maintenance
3. ✅ **Scalable structure** for adding new features
4. ✅ **Production-ready** for open-source publication
5. ✅ **Maintainable** beyond the original author's PhD

### Next Steps:
1. **Approve this proposal** and create the refactoring branch
2. **Allocate 30 hours** for the first iteration
3. **Implement in phases** as outlined above
4. **Test thoroughly** after each phase
5. **Document all changes** for future reference

---

**Proposal Status:** ✅ Ready for Implementation  
**Estimated Effort:** 25-30 hours  
**Confidence Level:** High  
**Risk Level:** Low-Medium  

*Document created: 2026-08-03*  
*Author: PEC Architecture Team*  
*Version: 1.0*
