# 30-Hour Refactoring Plan - Protein Embedding Classifier

## 🎯 Objective

**Transform the PEC repository into a modular, maintainable architecture in 30 hours or less.**

This document provides a **detailed, hour-by-hour plan** for the first iteration of refactoring, focusing on the essential changes that provide the most value for long-term maintainability.

---

## 📅 Overview

| Phase | Duration | Focus | Deliverables |
|-------|----------|-------|--------------|
| **Phase 0** | 1 hour | Preparation | Backup, branch, documentation |
| **Phase 1** | 8 hours | Directory Restructuring | New `src/` structure, module moves |
| **Phase 2** | 10 hours | Code Refactoring | Imports, CLI, configs |
| **Phase 3** | 8 hours | Documentation | Architecture, usage, README |
| **Phase 4** | 3 hours | Testing & Validation | Tests passing, basic validation |

**Total: 30 hours**

---

## ⏰ Detailed Hour-by-Hour Plan

---

## Phase 0: Preparation (1 hour)

### Hour 0: Setup and Backup
**Tasks:**
- [ ] Create full backup of current repository
- [ ] Verify backup integrity
- [ ] Create new branch: `refactor/modular-architecture`
- [ ] Push branch to remote (backup)

**Commands:**
```bash
# Create backup
cd /path/to/pec
cp -r protein-embedding-classifier protein-embedding-classifier_backup_20260803

# Create and checkout new branch
git checkout -b refactor/modular-architecture
git push origin refactor/modular-architecture
```

**Deliverables:**
- ✅ Backup created: `protein-embedding-classifier_backup_20260803/`
- ✅ Branch created: `refactor/modular-architecture`
- ✅ Branch pushed to GitHub

---

## Phase 1: Directory Restructuring (8 hours)

### Hour 1: Create src/ Directory Structure
**Tasks:**
- [ ] Create `src/` directory
- [ ] Create module directories under `src/`:
  - `src/input/`
  - `src/dataset_builder/`
  - `src/training/`
  - `src/prediction/`
  - `src/explainability/`
  - `src/output/`
  - `src/orchestration/`
- [ ] Create empty `__init__.py` files in each directory
- [ ] Update `pyproject.toml` to include `src/` in path

**Commands:**
```bash
# Create directory structure
mkdir -p src/{input,dataset_builder,training,prediction,explainability,output,orchestration}

# Create __init__.py files
for dir in src/*/; do touch "$dir/__init__.py"; done

# Update pyproject.toml
# Add: packages = [{include = "src"}]
```

**Deliverables:**
- ✅ `src/` directory with all subdirectories
- ✅ `__init__.py` in each module directory
- ✅ Updated `pyproject.toml`

### Hour 2: Move Input Module
**Tasks:**
- [ ] Move `protein_embedding_classifier/data/protein_loader.py` → `src/input/protein_loader.py`
- [ ] Move `pec/dataset/input/reader.py` → `src/input/reader.py`
- [ ] Move `pec/dataset/input/normalizer.py` → `src/input/normalizer.py`
- [ ] Move `pec/dataset/input/models.py` → `src/input/models.py`
- [ ] Create `src/input/validators.py` (new)
- [ ] Update `src/input/__init__.py` with exports

**Deliverables:**
- ✅ All input-related files in `src/input/`
- ✅ Basic input module structure

### Hour 3: Move Dataset Builder Module (Part 1)
**Tasks:**
- [ ] Move `pec/dataset/generator/` → `src/dataset_builder/generators/`
- [ ] Move `pec/dataset/export/` → `src/dataset_builder/export/`
- [ ] Move `pec/dataset/lineage/` → `src/dataset_builder/lineage/`
- [ ] Move `pec/dataset/policies/` → `src/dataset_builder/policies/`
- [ ] Move `pec/dataset/contracts.py` → `src/dataset_builder/contracts.py`
- [ ] Move `pec/dataset/splits/` → `src/dataset_builder/splits/`

**Deliverables:**
- ✅ PEC dataset files moved to `src/dataset_builder/`

### Hour 4: Move Dataset Builder Module (Part 2)
**Tasks:**
- [ ] Move `protein_embedding_classifier/data/dataset_builder.py` → `src/dataset_builder/builders/dataset_builder.py`
- [ ] Move `protein_embedding_classifier/data/label_loader.py` → `src/dataset_builder/label_loader.py`
- [ ] Move `dataset_designer_runs/run_loader.py` → `src/dataset_builder/run_loader.py`
- [ ] Create `src/dataset_builder/transformers/` directory
- [ ] Create `src/dataset_builder/embedding_integration/` directory
- [ ] Update `src/dataset_builder/__init__.py`

**Deliverables:**
- ✅ All dataset builder files consolidated
- ✅ Dataset builder module complete

### Hour 5: Move Training Module
**Tasks:**
- [ ] Move `protein_embedding_classifier/classifiers/` → `src/training/models/`
- [ ] Move `protein_embedding_classifier/core/embeddings.py` → `src/training/embedding_handler.py`
- [ ] Move `protein_embedding_classifier/core/decision/` → `src/training/decision/`
- [ ] Move `protein_embedding_classifier/core/statistics/` → `src/training/statistics/`
- [ ] Create `src/training/train_loop.py` (new)
- [ ] Create `src/training/wandb_integration.py` (new)
- [ ] Update `src/training/__init__.py`

**Deliverables:**
- ✅ Training module structure complete

### Hour 6: Create Prediction and Explainability Modules
**Tasks:**
- [ ] Create `src/prediction/predictor.py` (extract from classifiers)
- [ ] Create `src/prediction/batch_predictor.py` (new)
- [ ] Update `src/prediction/__init__.py`
- [ ] Create `src/explainability/feature_importance.py` (new)
- [ ] Create `src/explainability/embedding_saliency.py` (new)
- [ ] Create `src/explainability/visualization/` directory
- [ ] Update `src/explainability/__init__.py`

**Deliverables:**
- ✅ Prediction module created
- ✅ Explainability module created

### Hour 7: Create Output and Orchestration Modules
**Tasks:**
- [ ] Create `src/output/writers/` directory
- [ ] Create `src/output/results_manager.py` (new)
- [ ] Update `src/output/__init__.py`
- [ ] Create `src/orchestration/experiment_definitions.py` (new)
- [ ] Create `src/orchestration/runner.py` (new)
- [ ] Create `src/orchestration/benchmark.py` (new)
- [ ] Update `src/orchestration/__init__.py`

**Deliverables:**
- ✅ Output module created
- ✅ Orchestration module created

### Hour 8: Cleanup and Verify Structure
**Tasks:**
- [ ] Remove old `protein_embedding_classifier/` directory
- [ ] Remove old `pec/` directory
- [ ] Verify all files are in correct locations
- [ ] Check for any missed files
- [ ] Update `.gitignore` if needed

**Commands:**
```bash
# Verify structure
find src/ -type f -name "*.py" | sort

# Check for orphaned files
find . -name "*.py" -path "*/protein_embedding_classifier/*" -o -name "*.py" -path "*/pec/*"
```

**Deliverables:**
- ✅ Clean directory structure
- ✅ No orphaned files

---

## Phase 2: Code Refactoring (10 hours)

### Hour 9: Consolidate Configurations
**Tasks:**
- [ ] Create `configs/datasets/` directory
- [ ] Create `configs/models/` directory
- [ ] Create `configs/experiments/` directory
- [ ] Move `config/db.yaml` → `configs/datasets/db.yaml`
- [ ] Move `config/embeddings.yaml` → `configs/embeddings.yaml`
- [ ] Move `config/pipeline.yaml` → `configs/pipeline.yaml`
- [ ] Move `config/problems.yaml` → `configs/datasets/problems.yaml`
- [ ] Move `config/training/` → `configs/models/`
- [ ] Move `config/model_sweep/` → `configs/sweeps/`
- [ ] Update all config references in code

**Deliverables:**
- ✅ All configs in `configs/` directory
- ✅ Configs organized by category

### Hour 10: Update Imports (Part 1)
**Tasks:**
- [ ] Update imports in `src/input/` modules
- [ ] Update imports in `src/dataset_builder/` modules
- [ ] Fix any circular dependencies
- [ ] Test basic imports

**Focus:** Input and Dataset Builder modules

**Deliverables:**
- ✅ Input module imports working
- ✅ Dataset builder imports working

### Hour 11: Update Imports (Part 2)
**Tasks:**
- [ ] Update imports in `src/training/` modules
- [ ] Update imports in `src/prediction/` modules
- [ ] Update imports in `src/explainability/` modules
- [ ] Fix any circular dependencies

**Focus:** Training, Prediction, Explainability modules

**Deliverables:**
- ✅ Training module imports working
- ✅ Prediction and Explainability imports working

### Hour 12: Update Imports (Part 3)
**Tasks:**
- [ ] Update imports in `src/output/` modules
- [ ] Update imports in `src/orchestration/` modules
- [ ] Update any remaining imports
- [ ] Test all module imports

**Focus:** Output and Orchestration modules

**Deliverables:**
- ✅ All module imports working

### Hour 13: Create CLI Entry Points
**Tasks:**
- [ ] Create `scripts/` directory
- [ ] Create `scripts/train.py` (from `protein_embedding_classifier/main.py`)
- [ ] Create `scripts/predict.py` (new)
- [ ] Create `scripts/benchmark.py` (new)
- [ ] Create `scripts/export_dataset.py` (new)
- [ ] Update `pyproject.toml` with CLI entry points

**Example `pyproject.toml` update:**
```toml
[tool.poetry.scripts]
train = "scripts.train:main"
predict = "scripts.predict:main"
benchmark = "scripts.benchmark:main"
export-dataset = "scripts.export_dataset:main"
```

**Deliverables:**
- ✅ `scripts/` directory with CLI scripts
- ✅ Updated `pyproject.toml`

### Hour 14: Refactor Main Entry Points
**Tasks:**
- [ ] Refactor `scripts/train.py` to use new module structure
- [ ] Ensure it imports from `src/` modules
- [ ] Test basic CLI functionality
- [ ] Add argument parsing (argparse/click)

**Deliverables:**
- ✅ Working `train` CLI command

### Hour 15: Create Additional CLI Commands
**Tasks:**
- [ ] Implement `scripts/predict.py`
- [ ] Implement `scripts/benchmark.py`
- [ ] Implement `scripts/export_dataset.py`
- [ ] Test all CLI commands

**Deliverables:**
- ✅ All CLI commands working

### Hour 16: Update Module __init__.py Files
**Tasks:**
- [ ] Update `src/input/__init__.py` with public API
- [ ] Update `src/dataset_builder/__init__.py` with public API
- [ ] Update `src/training/__init__.py` with public API
- [ ] Update `src/prediction/__init__.py` with public API
- [ ] Update `src/explainability/__init__.py` with public API
- [ ] Update `src/output/__init__.py` with public API
- [ ] Update `src/orchestration/__init__.py` with public API
- [ ] Update `src/__init__.py` with all module exports

**Deliverables:**
- ✅ All `__init__.py` files with proper exports

### Hour 17: Create Module Interfaces
**Tasks:**
- [ ] Define clear public APIs for each module
- [ ] Document module interfaces in docstrings
- [ ] Create type hints for public functions
- [ ] Ensure consistent naming conventions

**Deliverables:**
- ✅ Clear module interfaces
- ✅ Type hints added

### Hour 18: Fix Any Remaining Issues
**Tasks:**
- [ ] Run basic tests to find import errors
- [ ] Fix any broken imports
- [ ] Fix any missing dependencies
- [ ] Verify basic functionality

**Commands:**
```bash
# Test imports
python -c "from src.input import *; print('Input OK')"
python -c "from src.dataset_builder import *; print('Dataset Builder OK')"
python -c "from src.training import *; print('Training OK')"
```

**Deliverables:**
- ✅ All basic imports working
- ✅ No critical errors

---

## Phase 3: Documentation (8 hours)

### Hour 19: Create Architecture Documentation
**Tasks:**
- [ ] Create `docs/ARCHITECTURE.md`
- [ ] Document module structure
- [ ] Create dependency diagram (ASCII or Mermaid)
- [ ] Document data flow
- [ ] Document module responsibilities

**Deliverables:**
- ✅ `docs/ARCHITECTURE.md`

### Hour 20: Create Usage Documentation
**Tasks:**
- [ ] Create `docs/USAGE.md`
- [ ] Document installation
- [ ] Document basic usage
- [ ] Document CLI commands
- [ ] Document common workflows

**Deliverables:**
- ✅ `docs/USAGE.md`

### Hour 21: Update README
**Tasks:**
- [ ] Update project description
- [ ] Add installation instructions
- [ ] Add quick start guide
- [ ] Add module overview
- [ ] Add contribution guidelines
- [ ] Add license information

**Deliverables:**
- ✅ Updated `README.md`

### Hour 22: Add Module Documentation
**Tasks:**
- [ ] Add docstrings to all public functions
- [ ] Add type hints where missing
- [ ] Document module purposes
- [ ] Add examples in docstrings

**Focus:** Input, Dataset Builder, Training modules

**Deliverables:**
- ✅ Input module documented
- ✅ Dataset Builder module documented
- ✅ Training module documented

### Hour 23: Add More Module Documentation
**Tasks:**
- [ ] Add docstrings to Prediction module
- [ ] Add docstrings to Explainability module
- [ ] Add docstrings to Output module
- [ ] Add docstrings to Orchestration module

**Focus:** Prediction, Explainability, Output, Orchestration modules

**Deliverables:**
- ✅ All modules documented

### Hour 24: Create Examples
**Tasks:**
- [ ] Create `docs/examples/` directory
- [ ] Add basic training example
- [ ] Add prediction example
- [ ] Add benchmark example
- [ ] Add dataset export example

**Deliverables:**
- ✅ `docs/examples/` with usage examples

### Hour 25: Create API Documentation
**Tasks:**
- [ ] Create `docs/API.md`
- [ ] Document all public APIs
- [ ] Document module interfaces
- [ ] Add code examples

**Deliverables:**
- ✅ `docs/API.md`

### Hour 26: Final Documentation Review
**Tasks:**
- [ ] Review all documentation for consistency
- [ ] Fix any errors or omissions
- [ ] Ensure all modules are covered
- [ ] Verify examples work

**Deliverables:**
- ✅ All documentation complete and consistent

---

## Phase 4: Testing and Validation (3 hours)

### Hour 27: Run Existing Tests
**Tasks:**
- [ ] Run existing test suite
- [ ] Identify broken tests
- [ ] Fix import-related test failures
- [ ] Fix path-related test failures

**Commands:**
```bash
# Run tests
pytest tests/ -v

# Run specific test files
pytest tests/unit/ -v
pytest tests/integration/ -v
```

**Deliverables:**
- ✅ All existing tests passing (or documented failures)

### Hour 28: Create Integration Test
**Tasks:**
- [ ] Create basic integration test
- [ ] Test CLI commands
- [ ] Test module imports
- [ ] Test basic functionality

**Example test:**
```python
# tests/integration/test_refactoring.py
def test_module_imports():
    """Test that all modules can be imported"""
    from src.input import CSVLoader
    from src.dataset_builder import DatasetBuilder
    from src.training import Trainer
    from src.prediction import Predictor
    from src.explainability import Explainer
    from src.output import ResultsWriter
    from src.orchestration import ExperimentRunner
    
    assert CSVLoader is not None
    assert DatasetBuilder is not None
    # ... etc

def test_cli_commands():
    """Test that CLI commands work"""
    import subprocess
    
    # Test train command
    result = subprocess.run(["python", "-m", "scripts.train", "--help"], 
                          capture_output=True, text=True)
    assert result.returncode == 0
    assert "usage:" in result.stdout
```

**Deliverables:**
- ✅ Integration test created

### Hour 29: Final Validation
**Tasks:**
- [ ] Run all tests
- [ ] Verify CLI commands work
- [ ] Verify module imports work
- [ ] Verify basic functionality
- [ ] Check for any remaining issues

**Commands:**
```bash
# Final validation
python -c "from src import *; print('All imports OK')"
pytest tests/ -v --tb=short
python scripts/train.py --help
python scripts/predict.py --help
```

**Deliverables:**
- ✅ All validation tests passing
- ✅ Basic functionality verified

---

## 📋 Checkpoints and Milestones

### Checkpoint 1: After Phase 0 (1 hour)
- ✅ Backup created
- ✅ Branch created
- ✅ Ready to start refactoring

### Checkpoint 2: After Phase 1 (9 hours total)
- ✅ `src/` directory structure created
- ✅ All modules moved to new locations
- ✅ Old directories removed
- ✅ Basic structure in place

### Checkpoint 3: After Phase 2 (19 hours total)
- ✅ All imports updated
- ✅ CLI commands working
- ✅ Configurations consolidated
- ✅ Module interfaces defined

### Checkpoint 4: After Phase 3 (27 hours total)
- ✅ All documentation created
- ✅ README updated
- ✅ Examples provided
- ✅ API documented

### Checkpoint 5: After Phase 4 (30 hours total)
- ✅ All tests passing
- ✅ Basic functionality verified
- ✅ Ready for review

---

## 🎯 Success Criteria

### Minimum Viable Refactoring (Must Have)
- [ ] Clean `src/` directory structure with 7 modules
- [ ] All code moved to new locations
- [ ] Basic imports working
- [ ] CLI entry points functional
- [ ] Basic documentation in place
- [ ] All tests passing (or documented failures)

### Nice to Have (If Time Permits)
- [ ] Type hints added
- [ ] Comprehensive docstrings
- [ ] Examples created
- [ ] API documentation
- [ ] Integration tests

### Out of Scope (Future Work)
- [ ] Advanced explainability techniques
- [ ] Web interface
- [ ] Docker containerization
- [ ] CI/CD pipeline
- [ ] Performance optimization

---

## 🚨 Contingency Plans

### If Running Behind Schedule

**Option 1: Reduce Scope (After 20 hours)**
- Focus on core modules only (Input, Dataset Builder, Training)
- Skip Explainability and Output modules
- Create basic documentation only

**Option 2: Split into Phases (After 15 hours)**
- Complete Phase 1 and Phase 2
- Create PR for directory restructuring
- Continue with documentation in next iteration

### If Encountering Major Issues

**Issue: Circular Dependencies**
- Solution: Create interface modules or use lazy imports
- Example: `src/training/` imports from `src/dataset_builder/` via interface

**Issue: Broken Imports**
- Solution: Use `sed` to replace old paths with new paths
- Command: `sed -i 's/from protein_embedding_classifier/from src.training/g' **/*.py`

**Issue: Missing Files**
- Solution: Check git history or backup
- Command: `git checkout HEAD -- path/to/missing/file`

---

## 📊 Progress Tracking Template

```markdown
## Refactoring Progress

### Phase 0: Preparation
- [x] Backup created
- [x] Branch created
- [ ] Documentation reviewed

### Phase 1: Directory Restructuring (0/8 hours)
- [ ] src/ directory created
- [ ] Input module moved
- [ ] Dataset Builder module moved
- [ ] Training module moved
- [ ] Prediction module created
- [ ] Explainability module created
- [ ] Output module created
- [ ] Orchestration module created

### Phase 2: Code Refactoring (0/10 hours)
- [ ] Configurations consolidated
- [ ] Imports updated (Input/Dataset Builder)
- [ ] Imports updated (Training/Prediction)
- [ ] Imports updated (Explainability/Output/Orchestration)
- [ ] CLI entry points created
- [ ] CLI commands implemented
- [ ] Module interfaces defined
- [ ] Issues fixed

### Phase 3: Documentation (0/8 hours)
- [ ] Architecture documentation
- [ ] Usage documentation
- [ ] README updated
- [ ] Module documentation (Core)
- [ ] Module documentation (Remaining)
- [ ] Examples created
- [ ] API documentation
- [ ] Documentation review

### Phase 4: Testing (0/3 hours)
- [ ] Existing tests run
- [ ] Integration test created
- [ ] Final validation

**Total: 0/30 hours**
```

---

## 🏁 Final Deliverables

After 30 hours, you should have:

### Code Structure
```
protein-embedding-classifier/
├── src/
│   ├── input/
│   ├── dataset_builder/
│   ├── training/
│   ├── prediction/
│   ├── explainability/
│   ├── output/
│   └── orchestration/
├── configs/
│   ├── datasets/
│   ├── models/
│   ├── experiments/
│   └── sweeps/
├── datasets/ (renamed from dataset_designer_runs)
├── scripts/
│   ├── train.py
│   ├── predict.py
│   ├── benchmark.py
│   └── export_dataset.py
├── docs/
│   ├── ARCHITECTURE.md
│   ├── USAGE.md
│   ├── API.md
│   └── examples/
├── tests/
├── pyproject.toml
└── README.md
```

### Documentation
- ✅ `docs/ARCHITECTURE.md` - Module structure and responsibilities
- ✅ `docs/USAGE.md` - How to use the system
- ✅ `docs/API.md` - Public API documentation
- ✅ `README.md` - Project overview and quick start
- ✅ `docs/examples/` - Usage examples

### Functionality
- ✅ CLI commands: `train`, `predict`, `benchmark`, `export-dataset`
- ✅ All modules importable
- ✅ Basic functionality working
- ✅ Tests passing

---

## 🎓 Next Steps After 30 Hours

1. **Review the refactoring** with the team
2. **Create a Pull Request** to master
3. **Address any feedback** from reviewers
4. **Plan next iteration** for:
   - Type hints
   - Expanded test coverage
   - Performance optimization
   - Advanced features

---

**Document Status:** ✅ Ready for Implementation  
**Estimated Duration:** 30 hours  
**Confidence Level:** High  
**Risk Level:** Low  

*Created: 2026-08-03*  
*Version: 1.0*
