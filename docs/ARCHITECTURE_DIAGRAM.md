# Architecture Diagrams - Protein Embedding Classifier

## 📊 Visual Architecture Overview

This document contains visual diagrams of the proposed architecture for the Protein Embedding Classifier project.

---

## 🏗️ 1. Target Module Architecture (Mermaid)

```mermaid
flowchart TB
    subgraph "Protein Embedding Classifier"
        direction TB
        
        subgraph "src/"
            direction TB
            
            subgraph "Input Module"
                I1["input/__init__.py"]
                I2["csv_loader.py"]
                I3["db_loader.py"]
                I4["api_loader.py"]
                I5["validators.py"]
                I6["models.py"]
            end
            
            subgraph "Dataset Builder Module"
                DB1["dataset_builder/__init__.py"]
                DB2["transformers/"]
                DB3["embedding_integration/"]
                DB4["generators/"]
                DB5["lineage/"]
                DB6["policies/"]
                DB7["splits/"]
                DB8["run_loader.py"]
                DB9["contracts.py"]
            end
            
            subgraph "Training Module"
                T1["training/__init__.py"]
                T2["models/"]
                T3["train_loop.py"]
                T4["wandb_integration.py"]
                T5["losses.py"]
                T6["metrics.py"]
                T7["embedding_handler.py"]
                T8["decision/"]
                T9["statistics/"]
            end
            
            subgraph "Prediction Module"
                P1["prediction/__init__.py"]
                P2["predictor.py"]
                P3["batch_predictor.py"]
            end
            
            subgraph "Explainability Module"
                E1["explainability/__init__.py"]
                E2["feature_importance.py"]
                E3["embedding_saliency.py"]
                E4["shap_analysis.py"]
                E5["visualization/"]
            end
            
            subgraph "Output Module"
                O1["output/__init__.py"]
                O2["writers/"]
                O3["results_manager.py"]
            end
            
            subgraph "Orchestration Module"
                OR1["orchestration/__init__.py"]
                OR2["experiment_definitions.py"]
                OR3["runner.py"]
                OR4["sweep_manager.py"]
                OR5["benchmark.py"]
            end
        end
        
        subgraph "configs/"
            C1["datasets/"]
            C2["models/"]
            C3["experiments/"]
            C4["sweeps/"]
            C5["pipeline.yaml"]
            C6["embeddings.yaml"]
        end
        
        subgraph "scripts/"
            S1["train.py"]
            S2["predict.py"]
            S3["benchmark.py"]
            S4["export_dataset.py"]
        end
        
        subgraph "datasets/"
            D1["20260803_0258_7672b947/"]
            D2["20260803_0304_a68aa0bb/"]
            D3["runs_catalog.json"]
            D4["run_loader.py"]
            D5["README.md"]
        end
    end
    
    subgraph "External"
        EXT1["Raw Data\n(CSV, DB, APIs)"]
        EXT2["Embeddings\n(GO, 3Di, etc.)"]
        EXT3["Weights & Biases"]
        EXT4["GitHub"]
    end
    
    %% Data Flow
    EXT1 -->|"Load"| I2
    EXT1 -->|"Load"| I3
    EXT1 -->|"Load"| I4
    I2 -->|"Validate"| I5
    I3 -->|"Validate"| I5
    I4 -->|"Validate"| I5
    
    I2 -->|"Data"| DB2
    I3 -->|"Data"| DB2
    I4 -->|"Data"| DB2
    EXT2 -->|"Embeddings"| DB3
    DB2 -->|"Transform"| DB3
    DB3 -->|"Integrate"| DB4
    DB4 -->|"Generate"| DB8
    DB8 -->|"Dataset"| T2
    
    T2 -->|"Train"| T3
    T3 -->|"Metrics"| T6
    T3 -->|"Loss"| T5
    T3 -->|"W&B"| T4
    T3 -->|"Model"| P2
    
    P2 -->|"Predict"| P3
    P2 -->|"Explain"| E2
    P2 -->|"Explain"| E3
    P2 -->|"Explain"| E4
    
    E2 -->|"Visualize"| E5
    E3 -->|"Visualize"| E5
    E4 -->|"Visualize"| E5
    
    P3 -->|"Results"| O2
    E5 -->|"Reports"| O2
    T6 -->|"Metrics"| O2
    
    O2 -->|"Save"| EXT4
    T4 -->|"Log"| EXT3
    
    %% Module Dependencies
    DB2 -.->|"Uses"| I2
    DB2 -.->|"Uses"| I3
    DB2 -.->|"Uses"| I4
    DB3 -.->|"Uses"| EXT2
    T2 -.->|"Uses"| DB8
    T3 -.->|"Uses"| T2
    P2 -.->|"Uses"| T2
    E2 -.->|"Uses"| T2
    E3 -.->|"Uses"| T2
    OR2 -.->|"Uses"| DB8
    OR2 -.->|"Uses"| T2
    OR3 -.->|"Uses"| OR2
    OR5 -.->|"Uses"| OR3
    
    %% CLI Commands
    S1 -->|"Uses"| DB8
    S1 -->|"Uses"| T3
    S2 -->|"Uses"| P2
    S3 -->|"Uses"| OR5
    S4 -->|"Uses"| DB8
```

---

## 🔄 2. Data Flow Diagram

```mermaid
flowchart LR
    subgraph "Input Phase"
        A["Raw Data\n(Proteins, GO Terms)"]
        B["Embeddings\n(Pre-computed)"]
        C[Input Module]
    end
    
    subgraph "Dataset Building Phase"
        D[Dataset Builder Module]
        E["TP/NTP Pairs"]
        F["Metadata"]
    end
    
    subgraph "Training Phase"
        G[Training Module]
        H["Trained Model"]
        I["Metrics"]
    end
    
    subgraph "Inference Phase"
        J[Prediction Module]
        K["Predictions"]
        L[Explainability Module]
        M["Explanations"]
    end
    
    subgraph "Output Phase"
        N[Output Module]
        O["Results\n(CSV, JSON, Reports)"]
    end
    
    subgraph "Orchestration"
        P[Orchestration Module]
        Q["Experiments\n(Benchmarking, Sweeps)"]
    end
    
    A -->|"Load"| C
    B -->|"Load"| C
    C -->|"Process"| D
    D -->|"Generate"| E
    D -->|"Store"| F
    
    E -->|"Train"| G
    G -->|"Produce"| H
    G -->|"Calculate"| I
    
    H -->|"Predict"| J
    J -->|"Output"| K
    H -->|"Explain"| L
    L -->|"Output"| M
    
    K -->|"Save"| N
    M -->|"Save"| N
    I -->|"Save"| N
    N -->|"Generate"| O
    
    E -->|"Use"| P
    H -->|"Use"| P
    P -->|"Run"| Q
    Q -->|"Use"| G
    Q -->|"Use"| J
    Q -->|"Use"| L
```

---

## 📦 3. Module Dependencies Diagram

```mermaid
flowchart TB
    subgraph "Modules"
        direction TB
        
        Input[("Input Module")]
        DatasetBuilder[("Dataset Builder Module")]
        Training[("Training Module")]
        Prediction[("Prediction Module")]
        Explainability[("Explainability Module")]
        Output[("Output Module")]
        Orchestration[("Orchestration Module")]
    end
    
    subgraph "External"
        Data[("Raw Data")]
        Embeddings[("Embeddings")]
        W&B[("Weights & Biases")]
        Storage[("Storage\n(GitHub, Local)")]
    end
    
    %% Dependencies
    Input -->|"Provides data to"| DatasetBuilder
    DatasetBuilder -->|"Provides datasets to"| Training
    DatasetBuilder -->|"Provides datasets to"| Prediction
    DatasetBuilder -->|"Provides datasets to"| Orchestration
    
    Training -->|"Uses"| DatasetBuilder
    Training -->|"Saves to"| W&B
    Training -->|"Produces models for"| Prediction
    Training -->|"Produces models for"| Explainability
    
    Prediction -->|"Uses models from"| Training
    Explainability -->|"Uses models from"| Training
    
    Output -->|"Receives from"| Training
    Output -->|"Receives from"| Prediction
    Output -->|"Receives from"| Explainability
    Output -->|"Saves to"| Storage
    
    Orchestration -->|"Uses"| DatasetBuilder
    Orchestration -->|"Uses"| Training
    Orchestration -->|"Uses"| Prediction
    Orchestration -->|"Uses"| Explainability
    Orchestration -->|"Uses"| Output
    
    %% External dependencies
    Input -->|"Loads"| Data
    Input -->|"Loads"| Embeddings
    DatasetBuilder -->|"Uses"| Embeddings
    
    %% Style
    classDef module fill:#f9f,stroke:#333
    classDef external fill:#bbf,stroke:#333
    
    class Input,DatasetBuilder,Training,Prediction,Explainability,Output,Orchestration module
    class Data,Embeddings,W&B,Storage external
```

---

## 🗺️ 4. Current vs Target Architecture Comparison

### 4.1 Current Architecture

```mermaid
flowchart TB
    subgraph "Current Structure"
        PEC["protein_embedding_classifier/"]
        pec["pec/"]
        dd["dataset_designer_runs/"]
        config["config/"]
        tests["tests/"]
        docs["docs/"]
    end
    
    PEC -->|"Contains"| classifiers
    PEC -->|"Contains"| core
    PEC -->|"Contains"| data
    
    pec -->|"Contains"| dataset
    pec -->|"Contains"| input
    pec -->|"Contains"| generator
    pec -->|"Contains"| export
    pec -->|"Contains"| lineage
    pec -->|"Contains"| policies
    pec -->|"Contains"| splits
    
    core -->|"Contains"| embeddings
    core -->|"Contains"| decision
    core -->|"Contains"| statistics
    
    data -->|"Contains"| splits
    data -->|"Contains"| protein_loader
    data -->|"Contains"| dataset_builder
    data -->|"Contains"| label_loader
    
    %% Issues
    PEC -.->|"Overlap?"| pec
    PEC -.->|"Confusing"| pec
    core -.->|"Mixed responsibilities"| embeddings
    data -.->|"Overlap?"| pec
```

### 4.2 Target Architecture

```mermaid
flowchart TB
    subgraph "Target Structure"
        src["src/"]
        configs["configs/"]
        datasets["datasets/"]
        scripts["scripts/"]
        tests["tests/"]
        docs["docs/"]
    end
    
    src -->|"Contains"| input
    src -->|"Contains"| dataset_builder
    src -->|"Contains"| training
    src -->|"Contains"| prediction
    src -->|"Contains"| explainability
    src -->|"Contains"| output
    src -->|"Contains"| orchestration
    
    configs -->|"Contains"| datasets
    configs -->|"Contains"| models
    configs -->|"Contains"| experiments
    configs -->|"Contains"| sweeps
    
    scripts -->|"Contains"| train
    scripts -->|"Contains"| predict
    scripts -->|"Contains"| benchmark
    scripts -->|"Contains"| export_dataset
    
    %% Clean structure
    input -->|"Clear dependency"| dataset_builder
    dataset_builder -->|"Clear dependency"| training
    training -->|"Clear dependency"| prediction
    training -->|"Clear dependency"| explainability
    prediction -->|"Clear dependency"| output
    explainability -->|"Clear dependency"| output
    orchestration -->|"Uses all"| input
    orchestration -->|"Uses all"| dataset_builder
    orchestration -->|"Uses all"| training
```

---

## 🏗️ 5. Class Diagram (Key Classes)

```mermaid
classDiagram
    %% Input Module
    class CSVLoader {
        +load(file_path: str) DataFrame
        +validate(data: DataFrame) bool
    }
    
    class DatabaseLoader {
        +connect(db_url: str)
        +query(sql: str) DataFrame
    }
    
    class DataValidator {
        +validate_schema(data: DataFrame) bool
        +check_nulls(data: DataFrame) bool
    }
    
    %% Dataset Builder Module
    class DatasetBuilder {
        +build_dataset(config: dict) Dataset
        +apply_transformations(data: DataFrame) DataFrame
    }
    
    class EmbeddingIntegrator {
        +integrate_embeddings(dataset: Dataset, embeddings: dict) Dataset
        +pool_embeddings(embeddings: list) array
    }
    
    class DatasetGenerator {
        +generate_pairs(tp_proteins: list, ntp_proteins: list) Dataset
        +create_splits(dataset: Dataset, strategy: str) dict
    }
    
    %% Training Module
    class ModelRegistry {
        +register_model(name: str, model_class: type)
        +get_model(name: str) BaseModel
        +list_models() list
    }
    
    class BaseModel {
        <<abstract>>
        +train(X: array, y: array) None
        +predict(X: array) array
        +save(path: str) None
        +load(path: str) BaseModel
    }
    
    class MLPModel {
        +train(X: array, y: array) None
        +predict(X: array) array
    }
    
    class Trainer {
        +train_model(model: BaseModel, data: Dataset, config: dict) TrainedModel
        +validate(model: BaseModel, data: Dataset) dict
    }
    
    %% Prediction Module
    class Predictor {
        +predict(model: BaseModel, data: DataFrame) DataFrame
        +predict_proba(model: BaseModel, data: DataFrame) DataFrame
    }
    
    class BatchPredictor {
        +predict_batch(model: BaseModel, data: list) list
    }
    
    %% Explainability Module
    class Explainer {
        <<abstract>>
        +explain(model: BaseModel, data: DataFrame) dict
    }
    
    class FeatureImportance {
        +explain(model: BaseModel, data: DataFrame) dict
    }
    
    class EmbeddingSaliency {
        +explain(model: BaseModel, embeddings: array) dict
    }
    
    %% Output Module
    class ResultsWriter {
        +write_csv(data: DataFrame, path: str) None
        +write_json(data: dict, path: str) None
    }
    
    class ReportGenerator {
        +generate_report(results: dict, template: str) str
    }
    
    %% Orchestration Module
    class ExperimentRunner {
        +run_experiment(config: dict) dict
        +run_benchmark(config: dict) dict
    }
    
    class SweepManager {
        +run_sweep(config: dict) list
    }
    
    %% Relationships
    CSVLoader <|-- DatabaseLoader
    DatasetBuilder o-- EmbeddingIntegrator
    DatasetBuilder o-- DatasetGenerator
    BaseModel <|-- MLPModel
    ModelRegistry o-- BaseModel
    Trainer o-- BaseModel
    Predictor o-- BaseModel
    BatchPredictor o-- Predictor
    Explainer <|-- FeatureImportance
    Explainer <|-- EmbeddingSaliency
    ExperimentRunner o-- Trainer
    ExperimentRunner o-- Predictor
    ExperimentRunner o-- Explainer
```

---

## 📁 6. File Structure Tree (Text Representation)

```
protein-embedding-classifier/
├── src/
│   ├── __init__.py
│   │
│   ├── input/
│   │   ├── __init__.py
│   │   ├── csv_loader.py
│   │   ├── db_loader.py
│   │   ├── api_loader.py
│   │   ├── validators.py
│   │   └── models.py
│   │
│   ├── dataset_builder/
│   │   ├── __init__.py
│   │   ├── transformers/
│   │   │   ├── __init__.py
│   │   │   ├── normalizer.py
│   │   │   └── filter.py
│   │   ├── embedding_integration/
│   │   │   ├── __init__.py
│   │   │   ├── pooling.py
│   │   │   └── concatenator.py
│   │   ├── generators/
│   │   │   ├── __init__.py
│   │   │   └── target_non_target.py
│   │   ├── lineage/
│   │   │   ├── __init__.py
│   │   │   ├── builder.py
│   │   │   └── models.py
│   │   ├── policies/
│   │   │   ├── __init__.py
│   │   │   ├── validator.py
│   │   │   └── models.py
│   │   ├── splits/
│   │   │   ├── __init__.py
│   │   │   ├── strategies.py
│   │   │   └── models.py
│   │   ├── run_loader.py
│   │   ├── contracts.py
│   │   └── label_loader.py
│   │
│   ├── training/
│   │   ├── __init__.py
│   │   ├── models/
│   │   │   ├── __init__.py
│   │   │   ├── base.py
│   │   │   ├── mlp.py
│   │   │   ├── random_forest.py
│   │   │   ├── linear.py
│   │   │   └── xgboost.py
│   │   ├── train_loop.py
│   │   ├── wandb_integration.py
│   │   ├── losses.py
│   │   ├── metrics.py
│   │   ├── embedding_handler.py
│   │   ├── decision/
│   │   │   ├── __init__.py
│   │   │   └── decision_policy.py
│   │   └── statistics/
│   │       ├── __init__.py
│   │       ├── friedman_test.py
│   │       ├── nemenyi_test.py
│   │       └── ranking_utils.py
│   │
│   ├── prediction/
│   │   ├── __init__.py
│   │   ├── predictor.py
│   │   └── batch_predictor.py
│   │
│   ├── explainability/
│   │   ├── __init__.py
│   │   ├── feature_importance.py
│   │   ├── embedding_saliency.py
│   │   ├── shap_analysis.py
│   │   └── visualization/
│   │       ├── __init__.py
│   │       └── plotter.py
│   │
│   ├── output/
│   │   ├── __init__.py
│   │   ├── writers/
│   │   │   ├── __init__.py
│   │   │   ├── csv_writer.py
│   │   │   ├── json_writer.py
│   │   │   └── report_writer.py
│   │   └── results_manager.py
│   │
│   └── orchestration/
│       ├── __init__.py
│       ├── experiment_definitions.py
│       ├── runner.py
│       ├── sweep_manager.py
│       └── benchmark.py
│
├── configs/
│   ├── datasets/
│   │   ├── humans.yaml
│   │   ├── model_organisms.yaml
│   │   └── template.yaml
│   ├── models/
│   │   ├── mlp.yaml
│   │   ├── random_forest.yaml
│   │   └── template.yaml
│   ├── experiments/
│   │   ├── benchmark.yaml
│   │   ├── cross_species.yaml
│   │   └── template.yaml
│   ├── sweeps/
│   │   ├── mlp_sweep.yaml
│   │   ├── rf_sweep.yaml
│   │   └── xgb_sweep.yaml
│   ├── pipeline.yaml
│   └── embeddings.yaml
│
├── datasets/
│   ├── 20260803_0258_7672b947/
│   │   ├── final_dataset.csv
│   │   ├── run_metadata.json
│   │   ├── pipeline.log
│   │   └── ...
│   ├── 20260803_0304_a68aa0bb/
│   │   ├── final_dataset.csv
│   │   ├── run_metadata.json
│   │   └── ...
│   ├── README.md
│   ├── run_loader.py
│   └── runs_catalog.json
│
├── scripts/
│   ├── __init__.py
│   ├── train.py
│   ├── predict.py
│   ├── benchmark.py
│   └── export_dataset.py
│
├── tests/
│   ├── __init__.py
│   ├── unit/
│   │   ├── __init__.py
│   │   ├── test_input.py
│   │   ├── test_dataset_builder.py
│   │   ├── test_training.py
│   │   ├── test_prediction.py
│   │   ├── test_explainability.py
│   │   └── test_output.py
│   ├── integration/
│   │   ├── __init__.py
│   │   ├── test_pipeline.py
│   │   └── test_orchestration.py
│   └── fixtures/
│       └── test_data.py
│
├── docs/
│   ├── ARCHITECTURE.md
│   ├── USAGE.md
│   ├── API.md
│   ├── CONTRIBUTING.md
│   ├── TERMINOLOGY.md
│   ├── ARCHITECTURE_REFACTORING_PROPOSAL.md
│   ├── REFACTORING_30H_PLAN.md
│   └── examples/
│       ├── train_model.py
│       ├── make_prediction.py
│       └── run_benchmark.py
│
├── notebooks/
│   ├── exploration/
│   ├── analysis/
│   └── examples/
│
├── .github/
│   └── workflows/
│       ├── ci.yml
│       ├── test.yml
│       └── docs.yml
│
├── pyproject.toml
├── poetry.lock
├── README.md
└── .gitignore
```

---

## 🎯 7. Data Flow Sankey Diagram

```mermaid
flowchart LR
    subgraph " "
        A["Raw Data\n100%"]
    end
    
    subgraph " "
        B["Input Module\n95%"]
        C["Rejected\n5%"]
    end
    
    subgraph " "
        D["Dataset Builder\n90%"]
        E["Invalid\n10%"]
    end
    
    subgraph " "
        F["Training\n80%"]
        G["Validation\n20%"]
    end
    
    subgraph " "
        H["Prediction\n70%"]
        I["Explainability\n30%"]
    end
    
    subgraph " "
        J["Output\n100%"]
    end
    
    A -->|"100%"| B
    B -->|"95%"| D
    B -->|"5%"| C
    D -->|"80%"| F
    D -->|"20%"| G
    F -->|"70%"| H
    F -->|"30%"| I
    H -->|"100%"| J
    I -->|"100%"| J
```

---

## 📝 How to Use These Diagrams

### For Documentation
1. Copy the Mermaid code into any Markdown file
2. GitHub will render it automatically
3. For local viewing, use a Mermaid-compatible editor (VS Code with Mermaid plugin, Obsidian, etc.)

### For Presentations
1. Use Mermaid Live Editor: [https://mermaid.live](https://mermaid.live)
2. Copy the diagram code
3. Customize as needed
4. Export as SVG/PNG

### For Development
1. Use these diagrams as reference when implementing
2. Update diagrams as the architecture evolves
3. Keep diagrams in sync with actual code structure

---

## 🔗 Related Documents

- [Architecture Refactoring Proposal](ARCHITECTURE_REFACTORING_PROPOSAL.md)
- [30-Hour Refactoring Plan](REFACTORING_30H_PLAN.md)
- [Terminology Clarification](../TERMINOLOGY_CLARIFICATION.md)

---

**Document Status:** ✅ Complete  
**Last Updated:** 2026-08-03  
**Version:** 1.0  
**Author:** PEC Architecture Team
