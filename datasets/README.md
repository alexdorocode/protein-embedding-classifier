# Dataset Designer Runs - Protein Embedding Classifier

## 🚨 IMPORTANT TERMINOLOGY CLARIFICATION

**THIS PROJECT DOES NOT STUDY MULTIFUNCTIONAL PROTEINS (MFP).**

We study **Target Proteins (TP)** and **Non-Target Proteins (NTP)** for classification tasks.

### ❌ Deprecated Terms (DO NOT USE)
- `MFP` - Multifunctional Protein
- `MF` - Multifunctional
- `NMF` - Non-Multifunctional
- `mf_*` - Any prefix containing "mf"
- `multifunctional` - In any context

### ✅ Correct Terms (USE THESE)
- `TP` - Target Protein
- `NTP` - Non-Target Protein
- `target_protein` - Full form of TP
- `non_target_protein` - Full form of NTP
- `tp_id` - Target Protein identifier
- `ntp_id` - Non-Target Protein identifier

### 📝 Historical Context
Some filenames and variable names in the codebase still contain `mf_` or `MF` prefixes due to historical naming. **These are legacy names only.** The actual content of these files refers to TP/NTP concepts:

- `mf_assignments.csv` → Contains **TP to NTP assignments** (not multifunctional assignments)
- `mf_metricas.csv` → Contains **TP GO similarity metrics** (not multifunctional metrics)
- `mf_nmf_pairs.csv` → Contains **TP-NTP pairs** (not multifunctional vs non-multifunctional)
- `mf_not_possible.csv` → Contains **TP without NTP assignments** (not multifunctional not possible)

**Always interpret `mf_*` as referring to TP (Target Protein) and `nmf_*` or candidates as referring to NTP (Non-Target Protein).**

---

## 📁 Overview

This directory contains the output runs from the `mfp-dataset-designer` tool, integrated into the Protein Embedding Classifier (PEC) project. Each run represents a complete execution of the target/non-target dataset design pipeline.

### Current Runs

| Run ID | Short ID | Species Category | Species | Date | Status |
|--------|----------|------------------|---------|------|--------|
| `20260803_0258_7672b947` | 0258 | humans | Homo sapiens | 2026-08-03 | ✅ Completed |
| `20260803_0304_a68aa0bb` | 0304 | model_organisms | Multi-species | 2026-08-03 | ✅ Completed |

### Species Categories

#### 🧬 Humans (0258)
- **Scientific Name**: Homo sapiens
- **Organism Filter**: `["Homo sapiens (Human)."]`
- **Characteristics**: Human proteins only
- **Use Case**: Human-specific model training and evaluation

#### 🌱 Model Organisms (0304)
- **Scientific Names**:
  - Arabidopsis thaliana (Mouse-ear cress)
  - Escherichia coli (strain K12)
  - Mus musculus (Mouse)
  - Saccharomyces cerevisiae (Baker's yeast)
- **Organism Filter**: All four species above
- **Characteristics**: Multi-species dataset
- **Use Case**: Cross-species model training and transfer learning evaluation

---

## 🏗️ Run Structure (File Contract)

Each run directory contains a standardized set of files that form the **Run File Contract**. This contract ensures that any pipeline consuming these runs knows exactly what each file contains and how to use it.

### 📊 File Inventory

#### Primary Files (For Model Training)

| Filename | Role | Stage | Description | Key Columns |
|----------|------|-------|-------------|-------------|
| `final_dataset.csv` | **PRIMARY** | Final | Main dataset for training and evaluation | `tp_id`, `ntp_id` |
| `target_non_target_dataset.csv` | Alternative | Final | Same as final_dataset but with different column names | `target_id`, `non_target_id` |
| `target_non_target_pairs.csv` | Alternative | Final | Redundant with target_non_target_dataset.csv | `target_id`, `non_target_id` |

#### Assignment Files

| Filename | Role | Stage | Description | Key Columns | Terminology Note |
|----------|------|-------|-------------|-------------|------------------|
| `target_assignments.csv` | Assignment | Filter 2 | **CORRECT TERMINOLOGY**: TP to NTP assignments | `target_id`, `num_candidates`, `candidates` | ✅ Uses correct TP/NTP terms |
| `mf_assignments.csv` | Assignment | Filter 2 | **LEGACY NAME**: Same content as target_assignments.csv | `mf_id` (actually TP), `num_candidates`, `candidates` | ⚠️ `mf_id` = TP, despite filename |

#### Metrics Files

| Filename | Role | Stage | Description | Terminology Note |
|----------|------|-------|-------------|------------------|
| `mf_metricas.csv` | Metrics | Filter 2 | **LEGACY NAME**: GO similarity metrics for TP proteins | ⚠️ Despite filename, contains TP metrics only |
| `candidatas_metricas.csv` | Metrics | Filter 1 | GO similarity metrics for NTP candidates | ✅ Correct |

#### Filter Output Files

| Filename | Role | Stage | Description |
|----------|------|-------|-------------|
| `filter1_output.csv` | Filter Output | Filter 1 | Results after sequence length and identity filtering |
| `filter2_output.csv` | Filter Output | Filter 2 | Results after Poisson-weighted assignment |
| `filter3_output.csv` | Filter Output | Filter 3 | Results after outlier removal |

#### Filter Tracking Files

| Filename | Role | Stage | Description |
|----------|------|-------|-------------|
| `matches_primer_filtro.csv` | Tracking | Filter 1 | TP proteins that found matches in first filter |
| `no_matches_primer_filtro.csv` | Tracking | Filter 1 | TP proteins without matches in first filter |

#### Legacy Pair Files

| Filename | Role | Stage | Description | Terminology Note |
|----------|------|-------|-------------|------------------|
| `mf_nmf_pairs.csv` | Pairs | Filter 2 | **LEGACY NAME**: TP-NTP pairs from filter 2 | ⚠️ Despite filename, contains TP-NTP pairs |
| `mf_not_possible.csv` | Excluded | Filter 2 | **LEGACY NAME**: TP without NTP assignments | ⚠️ Despite filename, contains TP identifiers |
| `target_not_possible.csv` | Excluded | Final | TP without final assignments | ✅ Correct |

#### Metadata and Logging

| Filename | Role | Description | Key Information |
|----------|------|-------------|-----------------|
| `run_metadata.json` | Metadata | Run configuration and statistics | `run_id`, `species`, `species_category`, `statistics`, `terminology` |
| `pipeline.log` | Logging | Complete execution log | Timestamps, parameters, warnings, execution times |

---

## 🔍 File Content Details

### final_dataset.csv (PRIMARY TRAINING FILE)

**Structure:**
```csv
tp_id,ntp_id
ABCA3_HUMAN,XRN1_HUMAN
PLCB1_HUMAN,A0A2R8YDE6_HUMAN
...
```

**Columns:**
- `tp_id` (string): Target Protein identifier (e.g., `ABCA3_HUMAN`)
- `ntp_id` (string): Non-Target Protein identifier (e.g., `XRN1_HUMAN`)

**Usage:**
- Primary file for loading TP/NTP pairs
- Each row represents one TP-NTP pair for classification
- Use this for model training and evaluation

**Example (Run 0258 - Humans):**
```
ABCA3_HUMAN,XRN1_HUMAN
PLCB1_HUMAN,A0A2R8YDE6_HUMAN
SYLC_HUMAN,LATS1_HUMAN
```

**Example (Run 0304 - Model Organisms):**
```
RYR2_MOUSE,SSPO_MOUSE
RYR2_MOUSE,RN213_MOUSE
PUTA_ECOLI,ABCB5_MOUSE
```

### target_assignments.csv (CORRECT TERMINOLOGY)

**Structure:**
```csv
target_id,num_candidates,candidates
ABCA3_HUMAN,1,XRN1_HUMAN
NOTC1_HUMAN,2,"TASO2_HUMAN,ABCA2_HUMAN"
...
```

**Columns:**
- `target_id` (string): Target Protein identifier (TP)
- `num_candidates` (integer): Number of NTP candidates assigned to this TP
- `candidates` (string): Comma-separated list of NTP identifiers

**Usage:**
- Understand how many NTP candidates each TP has
- Reconstruct the assignment process
- Validate data consistency

### mf_assignments.csv (LEGACY NAME - SAME CONTENT)

**⚠️ IMPORTANT:** Despite the filename containing "mf_", this file has the **exact same content** as `target_assignments.csv`.

**Structure:**
```csv
mf_id,num_candidates,candidates
ABCA3_HUMAN,1,XRN1_HUMAN
NOTC1_HUMAN,2,"TASO2_HUMAN,ABCA2_HUMAN"
...
```

**Columns:**
- `mf_id` (string): **ACTUALLY Target Protein (TP) identifier** - despite the name
- `num_candidates` (integer): Number of NTP candidates
- `candidates` (string): Comma-separated list of NTP identifiers

**Terminology Mapping:**
- `mf_id` → `target_id` → **TP (Target Protein)**
- `candidates` → **NTP (Non-Target Protein)**

---

## 🎯 Pipeline Stages

Each run goes through a 5-step pipeline:

### Stage 1: Data Loading and Preparation
- Load Target Protein (TP) data
- Load Non-Target Protein (NTP) candidate data
- Apply organism filter
- Apply blacklist filter

### Stage 2: First Filter (Sequence Filter)
- Filter by sequence length variance (< 0.05)
- Filter by sequence identity (< 0.95)
- Output: `filter1_output.csv`, `matches_primer_filtro.csv`, `no_matches_primer_filtro.csv`

### Stage 3: GO Similarity Metrics Calculation
- Calculate GO similarity metrics (Resnik, Lin, Schlicker, Wang)
- Calculate MBL (Molecular Function Breadth Level) metrics
- Output: `mf_metricas.csv` (TP metrics), `candidatas_metricas.csv` (NTP metrics)

### Stage 4: Second Filter (Assignment Filter)
- Rank candidates by |sim_wang_avg - mbl_avg|
- Sample assignments with Poisson-like weighted distribution
- Output: `filter2_output.csv`, `mf_assignments.csv`, `target_assignments.csv`, `mf_nmf_pairs.csv`, `mf_not_possible.csv`

### Stage 5: Third Filter (Outlier Filter)
- Keep candidates with low |sim_wang_max - mbl_min| scores
- Output: `filter3_output.csv`, `final_dataset.csv`, `target_non_target_dataset.csv`, `target_non_target_pairs.csv`, `target_not_possible.csv`

---

## 🔧 Using Runs in PEC Pipelines

### Loading a Run (Recommended Approach)

```python
import pandas as pd
import json
from pathlib import Path

def load_run(run_id, base_path='dataset_designer_runs'):
    """
    Load a complete run dataset.
    
    Args:
        run_id: Full run ID (e.g., '20260803_0258_7672b947')
        base_path: Base directory for runs
        
    Returns:
        dict: {
            'metadata': dict,           # Run metadata
            'tp_ntp_pairs': DataFrame,  # Main dataset (tp_id, ntp_id)
            'assignments': DataFrame,   # TP to NTP assignments
            'tp_metrics': DataFrame,    # TP GO metrics
            'ntp_metrics': DataFrame,   # NTP GO metrics
            'log': str                 # Pipeline log
        }
    """
    run_path = Path(base_path) / run_id
    
    # Load metadata
    with open(run_path / 'run_metadata.json', 'r') as f:
        metadata = json.load(f)
    
    # Load main dataset (use final_dataset.csv as primary)
    tp_ntp_pairs = pd.read_csv(run_path / 'final_dataset.csv')
    
    # Load assignments (prefer target_assignments.csv for correct terminology)
    assignments_path = run_path / 'target_assignments.csv'
    if assignments_path.exists():
        assignments = pd.read_csv(assignments_path)
    else:
        # Fallback to legacy file
        assignments = pd.read_csv(run_path / 'mf_assignments.csv')
        assignments = assignments.rename(columns={'mf_id': 'target_id'})
    
    # Load metrics
    tp_metrics = pd.read_csv(run_path / 'mf_metricas.csv')
    ntp_metrics = pd.read_csv(run_path / 'candidatas_metricas.csv')
    
    # Load log
    with open(run_path / 'pipeline.log', 'r') as f:
        log = f.read()
    
    return {
        'metadata': metadata,
        'tp_ntp_pairs': tp_ntp_pairs,
        'assignments': assignments,
        'tp_metrics': tp_metrics,
        'ntp_metrics': ntp_metrics,
        'log': log
    }

# Example usage
run_data = load_run('20260803_0258_7672b947')
print(f"Run: {run_data['metadata']['run_id']}")
print(f"Species: {run_data['metadata']['species']}")
print(f"TP-NTP pairs: {len(run_data['tp_ntp_pairs'])}")
```

### Accessing Specific Files

```python
# For humans run (0258)
humans_run_id = '20260803_0258_7672b947'
humans_tp_ntp = pd.read_csv(f'dataset_designer_runs/{humans_run_id}/final_dataset.csv')

# For model organisms run (0304)
model_run_id = '20260803_0304_a68aa0bb'
model_tp_ntp = pd.read_csv(f'dataset_designer_runs/{model_run_id}/final_dataset.csv')
```

### Comparing Species

```python
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Load both runs
humans_data = load_run('20260803_0258_7672b947')
model_data = load_run('20260803_0304_a68aa0bb')

# Prepare data for classification
# Note: You need to load embeddings separately and join with tp_ntp_pairs
# This is a conceptual example

# Train on humans, test on model organisms (cross-species evaluation)
# Or train separate models for each species
# Or combine and use species as a feature
```

---

## 📊 Run Statistics

### Run 0258 (Humans)

| Metric | Value |
|--------|-------|
| Run ID | 20260803_0258_7672b947 |
| Species | Homo sapiens |
| Config Hash | 7672b947 |
| Filter 1 Rows | 1,162,306 |
| Filter 2 Rows | 22,856 |
| Filter 3 Rows | 2,351 |
| Final Rows | 2,351 |
| Duration | 1,415.6 seconds |
| Random Seed | 42 |

### Run 0304 (Model Organisms)

| Metric | Value |
|--------|-------|
| Run ID | 20260803_0304_a68aa0bb |
| Species | Multi-species |
| Config Hash | a68aa0bb |
| Filter 1 Rows | 241,917 |
| Filter 2 Rows | 19,532 |
| Filter 3 Rows | 1,188 |
| Final Rows | 1,188 |
| Duration | 422.1 seconds |
| Random Seed | 42 |

---

## 🎯 Experiment Design Patterns

### Pattern 1: Single Species Training

```python
# Train on humans only
run_id = '20260803_0258_7672b947'
data = load_run(run_id)
# Use data['tp_ntp_pairs'] for training
```

### Pattern 2: Cross-Species Comparison

```python
# Compare performance between species
humans_run = '20260803_0258_7672b947'
model_run = '20260803_0304_a68aa0bb'

# Load both datasets
humans_data = load_run(humans_run)
model_data = load_run(model_run)

# Train same model architecture on both
# Compare metrics
```

### Pattern 3: Combined Multi-Species Training

```python
# Combine both runs for multi-species training
combined_pairs = pd.concat([
    load_run('20260803_0258_7672b947')['tp_ntp_pairs'],
    load_run('20260803_0304_a68aa0bb')['tp_ntp_pairs']
], ignore_index=True)

# Add species indicator
combined_pairs['species'] = ['humans'] * len(humans_data['tp_ntp_pairs']) + \
                            ['model_organisms'] * len(model_data['tp_ntp_pairs'])
```

### Pattern 4: Transfer Learning

```python
# Train on model organisms, test on humans (or vice versa)
train_run = '20260803_0304_a68aa0bb'  # Model organisms
test_run = '20260803_0258_7672b947'   # Humans

# Load training data
train_data = load_run(train_run)
# Load test data
test_data = load_run(test_run)

# Train on model organisms, evaluate on humans
# This tests cross-species generalization
```

---

## ✅ Validation and Quality Checks

### File Existence Check

```python
def validate_run(run_id, base_path='dataset_designer_runs'):
    """Validate that a run has all required files."""
    run_path = Path(base_path) / run_id
    
    required_files = [
        'final_dataset.csv',
        'run_metadata.json',
        'pipeline.log',
        'target_assignments.csv',
        'mf_assignments.csv',
        'mf_metricas.csv',
        'candidatas_metricas.csv',
        'filter1_output.csv',
        'filter2_output.csv',
        'filter3_output.csv'
    ]
    
    missing = []
    for f in required_files:
        if not (run_path / f).exists():
            missing.append(f)
    
    if missing:
        print(f"❌ Run {run_id} is missing: {missing}")
        return False
    else:
        print(f"✅ Run {run_id} has all required files")
        return True
```

### Data Consistency Check

```python
def check_data_consistency(run_id, base_path='dataset_designer_runs'):
    """Check that TP identifiers are consistent across files."""
    run_path = Path(base_path) / run_id
    
    # Load files
    final_dataset = pd.read_csv(run_path / 'final_dataset.csv')
    assignments = pd.read_csv(run_path / 'target_assignments.csv')
    
    # Get unique TP from final dataset
    tp_in_final = set(final_dataset['tp_id'].unique())
    
    # Get unique TP from assignments
    tp_in_assignments = set(assignments['target_id'].unique())
    
    # Check consistency
    if tp_in_final.issubset(tp_in_assignments):
        print(f"✅ All TP in final_dataset exist in assignments")
        return True
    else:
        missing_tp = tp_in_final - tp_in_assignments
        print(f"❌ TP in final_dataset but not in assignments: {missing_tp}")
        return False
```

---

## 📚 Additional Resources

- **Central Catalog**: `runs_catalog.json` - Machine-readable catalog of all runs
- **Pipeline Design**: See `PEC-IMPLEMENTATION-NOTES.md` in project root
- **Functional Contract**: See `FUNCTIONAL_CONTRACT.md` in project root
- **Source Tool**: [mfp-dataset-designer](https://github.com/alexdorocode/mfp-dataset-designer)

---

## 🔄 Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2026-08-03 | Initial documentation with terminology clarification |

---

## 📝 Contributing

When adding new runs:

1. **Follow the naming convention**: `YYYYMMDD_HHMM_hash`
2. **Include all required files** from the Run File Contract
3. **Update runs_catalog.json** with the new run metadata
4. **Update this README** if new file types are introduced
5. **Ensure terminology consistency**: Use TP/NTP, not MF/NMF

---

## 🎓 Key Takeaways

1. **Terminology**: Always use **TP** (Target Protein) and **NTP** (Non-Target Protein)
2. **Primary File**: Use `final_dataset.csv` for model training
3. **Species**: 0258 = Humans, 0304 = Model Organisms
4. **Legacy Files**: Files with `mf_` prefixes contain TP/NTP data, not multifunctional data
5. **Validation**: Always check file existence and data consistency before using a run

---

*Last updated: 2026-08-03*
*Maintainer: Protein Embedding Classifier Team*
