# Terminology Clarification - Protein Embedding Classifier

## 🚨 CRITICAL: Project Terminology Standard

**Effective Date: 2026-08-03**  
**Version: 1.0**  
**Status: MANDATORY**

---

## 📢 EXECUTIVE SUMMARY

**THIS PROJECT DOES NOT STUDY MULTIFUNCTIONAL PROTEINS (MFP).**

The Protein Embedding Classifier (PEC) project studies **Target Proteins (TP)** and **Non-Target Proteins (NTP)** for classification tasks using protein embeddings.

### ❌ PROHIBITED TERMS

The following terms **MUST NOT** be used in any new code, documentation, or communication:

- `MFP` - Multifunctional Protein
- `MF` - When referring to proteins (Multifunctional)
- `NMF` - Non-Multifunctional
- `mf_*` - Any prefix containing "mf" when referring to proteins
- `multifunctional` - When describing protein classification
- `multifunctionality` - When describing the classification task

### ✅ APPROVED TERMS

Use these terms instead:

| Old Term | New Term | Context |
|----------|----------|---------|
| MFP | TP | Target Protein |
| MF | TP | Target Protein |
| NMF | NTP | Non-Target Protein |
| multifunctional protein | Target Protein (TP) | Protein classification |
| non-multifunctional protein | Non-Target Protein (NTP) | Protein classification |
| mf_id | tp_id | Identifier column |
| nmf_id | ntp_id | Identifier column |
| mf_assignments | target_assignments | File naming |

---

## 🔍 EXCEPTION: Gene Ontology MF

**IMPORTANT EXCEPTION:** The term "MF" **IS ALLOWED** when referring to **Molecular Function**, one of the three Gene Ontology (GO) categories:

- **BP** - Biological Process ✅
- **MF** - Molecular Function ✅ (ALLOWED in GO context only)
- **CC** - Cellular Component ✅

### Examples of Correct Usage:

```python
# ✅ CORRECT - GO ontology context
ontology = "MF"  # Molecular Function from Gene Ontology
file_name = "mf_protein_embeddings.csv"  # Contains Molecular Function embeddings

# ❌ INCORRECT - Protein classification context
label = "MF"  # Should be "TP" for Target Protein
category = "multifunctional"  # Should be "target_protein"
```

### How to Distinguish:

| Context | "MF" Meaning | Allowed? |
|---------|--------------|----------|
| Gene Ontology | Molecular Function | ✅ YES |
| Protein Classification | Multifunctional Protein | ❌ NO |
| File naming (GO) | mf_embeddings.csv | ✅ YES |
| File naming (protein) | mf_assignments.csv | ⚠️ LEGACY (see below) |

---

## 📁 LEGACY FILES AND HISTORICAL CONTEXT

### The Problem

The `mfp-dataset-designer` tool (source of our dataset runs) was originally developed with "multifunctional protein" terminology. When it was integrated into PEC, the terminology was changed to "Target Protein / Non-Target Protein" but some filenames retained the old naming.

### Legacy Files in dataset_designer_runs/

The following files contain **TP/NTP data** despite having "mf_" or "MF" in their names:

| Legacy Filename | Actual Content | Correct Interpretation |
|----------------|----------------|------------------------|
| `mf_assignments.csv` | TP to NTP assignments | `mf_id` = TP, `candidates` = NTP |
| `mf_metricas.csv` | TP GO metrics | Metrics for Target Proteins only |
| `mf_nmf_pairs.csv` | TP-NTP pairs | Pairs of Target and Non-Target Proteins |
| `mf_not_possible.csv` | TP without assignments | Target Proteins without NTP candidates |

### How to Handle Legacy Files

1. **When reading these files:**
   - `mf_id` column → Treat as `tp_id` (Target Protein)
   - `candidates` column → Treat as `ntp_id` (Non-Target Protein)
   - File content is TP/NTP, not multifunctional

2. **When creating new files:**
   - Use correct terminology: `target_assignments.csv`, `tp_metrics.csv`, etc.
   - Do NOT create new files with `mf_` prefixes

3. **When documenting:**
   - Clearly state that legacy files use old naming but contain TP/NTP data
   - Add terminology notes to all legacy file documentation

---

## 🏗️ TERMINOLOGY IN DIFFERENT CONTEXTS

### 1. Protein Classification (PEC Core)

| Term | Meaning | Usage |
|------|---------|-------|
| TP | Target Protein | ✅ Primary classification label |
| NTP | Non-Target Protein | ✅ Primary classification label |
| Target Protein | Full form of TP | ✅ Use in documentation |
| Non-Target Protein | Full form of NTP | ✅ Use in documentation |

### 2. Gene Ontology (GO)

| Term | Meaning | Usage |
|------|---------|-------|
| BP | Biological Process | ✅ GO ontology |
| MF | Molecular Function | ✅ GO ontology (exception) |
| CC | Cellular Component | ✅ GO ontology |

### 3. Dataset Files

| File Type | Old Naming | New Naming | Status |
|-----------|------------|------------|--------|
| Main dataset | - | `final_dataset.csv` | ✅ Current |
| Main dataset | - | `target_non_target_dataset.csv` | ✅ Current |
| Assignments | `mf_assignments.csv` | `target_assignments.csv` | ⚠️ Both exist |
| Metrics (TP) | `mf_metricas.csv` | `tp_metrics.csv` | ⚠️ Legacy exists |
| Metrics (NTP) | `candidatas_metricas.csv` | `ntp_metrics.csv` | ✅ Current |

### 4. Code Variables

| Old Variable | New Variable | Context |
|--------------|--------------|---------|
| `mf_id` | `tp_id` | Protein identifier |
| `nmf_id` | `ntp_id` | Protein identifier |
| `mf_proteins` | `target_proteins` | Protein list |
| `nmf_proteins` | `non_target_proteins` | Protein list |
| `is_mf` | `is_target` | Boolean label |
| `mf_label` | `target_label` | Classification label |

---

## 📋 TERMINOLOGY COMPLIANCE CHECKLIST

### For New Code

- [ ] Use `TP` and `NTP` for protein classification
- [ ] Use `tp_id` and `ntp_id` for column names
- [ ] Use `target_` prefix for TP-related files/variables
- [ ] Use `non_target_` prefix for NTP-related files/variables
- [ ] Do NOT use `mf_`, `MF`, `multifunctional` for protein classification
- [ ] Use `MF` only for Gene Ontology Molecular Function

### For Documentation

- [ ] Clearly state that PEC studies TP/NTP, not MFP
- [ ] Add terminology clarification to all relevant docs
- [ ] Document legacy file naming with actual content
- [ ] Use consistent terminology throughout

### For Data Files

- [ ] Prefer `final_dataset.csv` for main training data
- [ ] Use `target_assignments.csv` over `mf_assignments.csv`
- [ ] Document all legacy files with terminology notes
- [ ] Validate that legacy files are interpreted correctly

---

## 🔧 IMPLEMENTATION GUIDELINES

### Loading Legacy Files

```python
# ✅ CORRECT - Handling legacy file with proper interpretation
import pandas as pd

# Load legacy file but interpret correctly
df = pd.read_csv('mf_assignments.csv')
# Rename columns to reflect actual content
df = df.rename(columns={'mf_id': 'tp_id'})
# Now df['tp_id'] contains Target Protein identifiers

# Or use the run_loader module which handles this automatically
from dataset_designer_runs.run_loader import load_run
run_data = load_run('20260803_0258_7672b947')
# run_data.assignments already has correct column names
```

### Creating New Files

```python
# ✅ CORRECT - Using new terminology
df.to_csv('target_assignments.csv')  # Not mf_assignments.csv

# ✅ CORRECT - Column names
df = pd.DataFrame({
    'tp_id': target_proteins,
    'ntp_id': non_target_proteins,
    'label': ['TP'] * len(target_proteins) + ['NTP'] * len(non_target_proteins)
})

# ❌ INCORRECT - Using old terminology
df.to_csv('mf_assignments.csv')  # Don't create new legacy files
```

### Documentation Examples

```markdown
# ✅ CORRECT Documentation

## Dataset Structure

The dataset contains:
- **Target Proteins (TP)**: Proteins of interest for classification
- **Non-Target Proteins (NTP)**: Counter-examples for classification

### Legacy Files

**Note:** Some files have legacy names containing "mf_" but contain TP/NTP data:
- `mf_assignments.csv`: Contains TP to NTP assignments (not multifunctional)
- `mf_metricas.csv`: Contains TP metrics (not multifunctional metrics)

# ❌ INCORRECT Documentation

## Dataset Structure

The dataset contains:
- **MF Proteins**: Multifunctional proteins  # ❌ Wrong
- **NMF Proteins**: Non-multifunctional proteins  # ❌ Wrong
```

---

## 📊 TERMINOLOGY MAPPING TABLE

| Legacy Term | Current Term | Context | Status |
|-------------|--------------|---------|--------|
| MFP | TP | Protein classification | ❌ Deprecated |
| MF (protein) | TP | Protein classification | ❌ Deprecated |
| NMF | NTP | Protein classification | ❌ Deprecated |
| mf_id | tp_id | Column name | ⚠️ Legacy |
| nmf_id | ntp_id | Column name | ⚠️ Legacy |
| mf_assignments | target_assignments | Filename | ⚠️ Legacy |
| mf_metricas | tp_metrics | Filename | ⚠️ Legacy |
| MF (GO) | MF | Gene Ontology | ✅ Allowed |
| BP (GO) | BP | Gene Ontology | ✅ Allowed |
| CC (GO) | CC | Gene Ontology | ✅ Allowed |

---

## 🎯 QUICK REFERENCE

### Do Use:
- ✅ TP / Target Protein
- ✅ NTP / Non-Target Protein
- ✅ tp_id / ntp_id
- ✅ target_assignments.csv
- ✅ MF (only for Gene Ontology Molecular Function)

### Don't Use:
- ❌ MFP / Multifunctional Protein
- ❌ MF (for protein classification)
- ❌ NMF / Non-Multifunctional
- ❌ mf_id / nmf_id (in new code)
- ❌ multifunctional / multifunctionality (for classification)

---

## 📞 SUPPORT AND CLARIFICATION

If you are unsure about terminology usage:

1. **Check this document first**
2. **Look at the context**: Is it about protein classification or Gene Ontology?
3. **When in doubt**: Use TP/NTP for protein classification
4. **For legacy code**: Add comments explaining the terminology mapping

### Example Comment for Legacy Code:

```python
# LEGACY TERMINOLOGY: This file uses 'mf_id' but it actually refers to
# Target Protein (TP), not Multifunctional Protein. The 'candidates'
# column contains Non-Target Protein (NTP) identifiers.
# See TERMINOLOGY_CLARIFICATION.md for details.
df = pd.read_csv('mf_assignments.csv')
```

---

## 📝 Version History

| Version | Date | Changes | Author |
|---------|------|---------|--------|
| 1.0 | 2026-08-03 | Initial terminology standard | PEC Team |

---

## 🔒 Approval

This terminology standard is **MANDATORY** for all contributors to the Protein Embedding Classifier project. All new code, documentation, and communications must comply with these guidelines.

**Approved by:** Protein Embedding Classifier Team  
**Effective:** 2026-08-03  
**Review date:** 2026-12-03 (4 months)

---

*For questions or clarifications, contact the PEC team or refer to the project documentation.*
