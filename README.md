# Protein Embedding Classifier (PEC)

## 🎯 Overview

**Protein Embedding Classifier (PEC)** is a modular, scalable research platform for studying protein classification using embeddings. The system supports the entire experimental lifecycle from data loading to model training, prediction, and explainability.

### 🚨 Important Terminology Note

**This project uses TP (Target Protein) and NTP (Non-Target Protein), NOT MFP (Multifunctional Protein).**

Please refer to [docs/TERMINOLOGY_CLARIFICATION.md](docs/TERMINOLOGY_CLARIFICATION.md) for complete terminology guidelines.

---

## 🏗️ Architecture

PEC follows a **modular architecture** with 7 main modules:

```
protein-embedding-classifier/
├── src/
│   ├── input/                    # Data loading (CSV, DB, API, Proteins)
│   ├── dataset_builder/         # Dataset construction and management
│   ├── training/                # Model training and validation
│   ├── prediction/              # Model inference
│   ├── explainability/          # Model explainability (feature importance, saliency)
│   ├── output/                  # Results management and export
│   └── orchestration/           # Experiment management and benchmarking
│
├── configs/                     # Configuration files
│   ├── datasets/                # Dataset configurations
│   ├── models/                  # Model configurations
│   ├── experiments/             # Experiment configurations
│   └── sweeps/                  # Hyperparameter sweep configurations
│
├── datasets/                    # Dataset runs (formerly dataset_designer_runs)
│   ├── 20260803_0258_7672b947/ # Humans dataset
│   ├── 20260803_0304_a68aa0bb/ # Model organisms dataset
│   ├── runs_catalog.json        # Run metadata catalog
│   ├── run_loader.py            # Run loading utilities
│   └── README.md                # Dataset documentation
│
├── scripts/                     # CLI entry points
│   ├── train.py                 # Train models
│   ├── predict.py               # Make predictions
│   ├── benchmark.py             # Run benchmarks
│   └── export_dataset.py        # Export datasets
│
├── docs/                        # Documentation
│   ├── ARCHITECTURE.md          # Architecture overview
│   ├── USAGE.md                 # Usage guide
│   ├── API.md                   # API documentation
│   ├── TERMINOLOGY_CLARIFICATION.md # Terminology standards
│   └── examples/                # Example scripts
│
├── tests/                       # Test suite
├── pyproject.toml               # Project configuration
└── README.md                    # This file
```

### Module Responsibilities

| Module | Responsibility | Key Classes/Functions |
|--------|---------------|----------------------|
| **Input** | Raw data loading | `CSVLoader`, `DatabaseLoader`, `APILoader`, `ProteinLoader` |
| **Dataset Builder** | Dataset construction | `DatasetBuilder`, `RunLoader`, `LabelLoader` |
| **Training** | Model training | `get_classifier()`, `BaseClassifier`, `EmbeddingStore` |
| **Prediction** | Model inference | `Predictor`, `BatchPredictor`, `predict()`, `batch_predict()` |
| **Explainability** | Model interpretation | `FeatureImportance`, `EmbeddingSaliency`, `explain()`, `visualize()` |
| **Output** | Results management | `ResultsManager`, `CSVWriter`, `JSONWriter`, `ReportWriter` |
| **Orchestration** | Experiment management | `ExperimentRunner`, `SweepManager`, `BenchmarkManager` |

---

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/alexdorocode/protein-embedding-classifier.git
cd protein-embedding-classifier

# Install dependencies (using Poetry)
poetry install

# Or using pip
pip install -e .
```

### Basic Usage

#### 1. Load a Dataset

```python
from src.dataset_builder import load_run

# Load humans dataset
run_data = load_run('20260803_0258_7672b947', base_path='datasets')
print(f"Loaded {len(run_data.tp_ntp_pairs)} TP-NTP pairs")
print(f"Species: {run_data.metadata['species']}")
```

#### 2. Train a Model

```python
from src.training import get_classifier

# Get a classifier
model = get_classifier('mlp')  # or 'lr', 'rf'

# Train on your data
# model.fit(X_train, y_train)
```

#### 3. Make Predictions

```python
from src.prediction import predict

# Make predictions
predictions = predict(model, X_test)
```

#### 4. Generate Explanations

```python
from src.explainability import explain, visualize

# Explain model predictions
explanation = explain(model, X_test, method='feature_importance')

# Visualize explanations
fig = visualize(explanation, plot_type='bar')
fig.savefig('feature_importance.png')
```

#### 5. Run an Experiment

```python
from src.orchestration import run_experiment

# Run a predefined experiment
results = run_experiment('humans_mlp')
print(results)
```

---

## 📦 CLI Commands

PEC provides several command-line interfaces for common tasks:

### Train a Model

```bash
# Basic training
python -m scripts.train --dataset 20260803_0258_7672b947 --model mlp

# With configuration
python -m scripts.train --dataset 20260803_0258_7672b947 --model mlp --config configs/experiments/training_config.yaml

# With W&B logging
python -m scripts.train --dataset 20260803_0258_7672b947 --model mlp --wandb
```

**Options:**
- `--dataset`: Run ID to use for training (required)
- `--model`: Model name (mlp, lr, rf) (required)
- `--config`: Path to training configuration YAML
- `--output`: Output directory for results (default: results)
- `--base-path`: Base path for dataset runs (default: datasets)
- `--wandb`: Enable Weights & Biases logging
- `--project`: W&B project name (default: protein-embedding-classifier)
- `--experiment`: W&B experiment name
- `--verbose`: Enable verbose logging

### Make Predictions

```bash
# Basic prediction
python -m scripts.predict --model path/to/model.pkl --data path/to/data.csv

# With batch processing
python -m scripts.predict --model path/to/model.pkl --data path/to/data.csv --batch-size 64

# Get probabilities
python -m scripts.predict --model path/to/model.pkl --data path/to/data.csv --probabilities
```

**Options:**
- `--model`: Path to trained model file (required)
- `--data`: Path to input data file (required)
- `--output`: Path to save predictions (default: predictions.csv)
- `--batch-size`: Batch size for prediction (default: 32)
- `--probabilities`: Output probability predictions instead of class predictions
- `--verbose`: Enable verbose logging

### Run Benchmark

```bash
# Run benchmark with configuration
python -m scripts.benchmark --config configs/experiments/benchmark.yaml

# With W&B logging
python -m scripts.benchmark --config configs/experiments/benchmark.yaml --wandb
```

**Options:**
- `--config`: Path to benchmark configuration YAML (required)
- `--output`: Path to save benchmark results (default: benchmark_results.json)
- `--wandb`: Enable Weights & Biases logging
- `--project`: W&B project name
- `--verbose`: Enable verbose logging

### Export Dataset

```bash
# Export dataset to CSV
python -m scripts.export_dataset --run 20260803_0258_7672b947 --format csv

# Export with metadata
python -m scripts.export_dataset --run 20260803_0258_7672b947 --format json --include-metadata
```

**Options:**
- `--run`: Run ID to export (required)
- `--output`: Output file path (default: <run_id>_dataset.<format>)
- `--format`: Output format (csv, json, pickle) (default: csv)
- `--base-path`: Base path for dataset runs (default: datasets)
- `--include-metadata`: Include run metadata in export
- `--verbose`: Enable verbose logging

---

## 📚 Available Datasets

PEC currently includes two main dataset runs:

### Humans Dataset (Run 0258)
- **Run ID:** `20260803_0258_7672b947`
- **Species:** Homo sapiens (Human)
- **Description:** Human proteins for TP/NTP classification
- **Size:** 2,351 TP-NTP pairs
- **Location:** `datasets/20260803_0258_7672b947/`

### Model Organisms Dataset (Run 0304)
- **Run ID:** `20260803_0304_a68aa0bb`
- **Species:** Multi-species (Arabidopsis thaliana, Escherichia coli, Mus musculus, Saccharomyces cerevisiae)
- **Description:** Model organism proteins for TP/NTP classification
- **Size:** 1,188 TP-NTP pairs
- **Location:** `datasets/20260803_0304_a68aa0bb/`

### Loading Datasets

```python
from src.dataset_builder import load_run, list_runs

# List all available runs
runs = list_runs()
print(f"Available runs: {runs}")

# Load a specific run
run_data = load_run('20260803_0258_7672b947')

# Access dataset
print(run_data.tp_ntp_pairs.head())  # TP-NTP pairs
print(run_data.metadata)            # Run metadata
print(run_data.assignments.head())   # TP to NTP assignments
```

---

## 🎯 Available Models

PEC supports multiple classifier types:

| Model | Description | Configuration |
|-------|-------------|---------------|
| `mlp` | Multi-Layer Perceptron | Hidden layers, dropout, learning rate |
| `lr` | Logistic Regression | Regularization, max iterations |
| `rf` | Random Forest | Number of trees, max depth |

### Model Configuration

```python
from src.training import get_classifier

# Get a model with default configuration
model = get_classifier('mlp')

# Get a model with custom configuration
model = get_classifier('mlp', hidden_size=[128, 64], dropout=0.2, learning_rate=0.001)
```

---

## 🔬 Available Experiments

PEC includes predefined experiments for common workflows:

| Experiment | Description | Dataset | Model |
|------------|-------------|---------|-------|
| `humans_mlp` | Train MLP on humans | 0258 | mlp |
| `model_organisms_mlp` | Train MLP on model organisms | 0304 | mlp |
| `humans_rf` | Train Random Forest on humans | 0258 | rf |

### Running Experiments

```python
from src.orchestration import run_experiment, run_sweep, run_benchmark

# Run a single experiment
results = run_experiment('humans_mlp')

# Run a hyperparameter sweep
sweep_config = {
    'experiment': 'humans_mlp',
    'parameters': {
        'learning_rate': [0.001, 0.01, 0.1],
        'hidden_size': [[64, 32], [128, 64]],
    },
    'runs': 3
}
sweep_results = run_sweep(sweep_config)

# Run a benchmark
benchmark_config = {
    'name': 'model_comparison',
    'models': ['mlp', 'lr', 'rf'],
    'datasets': ['20260803_0258_7672b947', '20260803_0304_a68aa0bb'],
    'metrics': ['accuracy', 'f1', 'precision', 'recall']
}
benchmark_results = run_benchmark(benchmark_config)
```

---

## 📊 Configuration

PEC uses YAML files for configuration. See `configs/` directory for examples.

### Configuration Structure

```yaml
# configs/experiments/training_config.yaml
training:
  epochs: 100
  batch_size: 32
  learning_rate: 0.001
  hidden_size: [128, 64]
  dropout: 0.2
  random_state: 42

evaluation:
  test_size: 0.2
  random_state: 42
  metrics: ['accuracy', 'f1', 'precision', 'recall']

wandb:
  project: protein-embedding-classifier
  entity: your-team
  log_metrics: true
```

---

## 🔧 Development

### Project Structure

```
src/
├── input/
│   ├── __init__.py
│   ├── csv_loader.py
│   ├── db_loader.py
│   ├── api_loader.py
│   ├── protein_loader.py
│   ├── reader.py
│   ├── normalizer.py
│   ├── validators.py
│   └── models.py
│
├── dataset_builder/
│   ├── __init__.py
│   ├── builders/
│   │   └── dataset_builder.py
│   ├── generators/
│   │   └── generator.py
│   ├── export/
│   │   └── exporter.py
│   ├── lineage/
│   │   └── builder.py
│   ├── policies/
│   │   └── validator.py
│   ├── splits/
│   │   ├── base.py
│   │   ├── strategies.py
│   │   └── ...
│   ├── run_loader.py
│   ├── label_loader.py
│   ├── contracts.py
│   └── models.py
│
├── training/
│   ├── __init__.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── base_classifier.py
│   │   ├── mlp_protein_classifier.py
│   │   ├── linear.py
│   │   ├── linear_classifier.py
│   │   ├── random_forest.py
│   │   └── random_forest_classifier.py
│   ├── losses.py
│   ├── metrics.py
│   ├── train_loop.py
│   ├── wandb_integration.py
│   ├── embedding_handler.py
│   ├── embedding_loading.py
│   ├── embeddings.py
│   ├── decision/
│   │   └── decision_policy.py
│   ├── statistics/
│   │   ├── friedman_test.py
│   │   ├── nemenyi_test.py
│   │   └── ranking_utils.py
│   ├── ensemble/
│   │   └── soft_voting_service.py
│   ├── layer_aggregation/
│   │   ├── attention.py
│   │   ├── base.py
│   │   └── mean.py
│   ├── pipeline.py
│   ├── experiment.py
│   ├── tracking.py
│   └── logging_config.py
│
├── prediction/
│   ├── __init__.py
│   └── predictor.py
│
├── explainability/
│   ├── __init__.py
│   ├── feature_importance.py
│   ├── embedding_saliency.py
│   └── visualization/
│       ├── __init__.py
│       └── plotter.py
│
├── output/
│   ├── __init__.py
│   ├── results_manager.py
│   └── writers/
│       ├── __init__.py
│       ├── csv_writer.py
│       ├── json_writer.py
│       └── report_writer.py
│
└── orchestration/
    ├── __init__.py
    ├── experiment_runner.py
    ├── experiment_definitions.py
    ├── sweep_manager.py
    └── benchmark_manager.py
```

### Adding New Features

When adding new functionality, follow these guidelines:

1. **Place code in the appropriate module**
2. **Follow the existing code style**
3. **Add comprehensive docstrings**
4. **Add type hints**
5. **Update the module's `__init__.py`** to export new classes/functions
6. **Add tests** in the `tests/` directory
7. **Update documentation**

### Code Style

- Use **snake_case** for function and variable names
- Use **PascalCase** for class names
- Use **UPPER_CASE** for constants
- Add **type hints** to all public functions
- Add **docstrings** following Google style
- Use **logging** instead of print statements

---

## 🧪 Testing

Run the test suite:

```bash
# Run all tests
pytest tests/

# Run specific test module
pytest tests/unit/

# Run with verbose output
pytest -v tests/

# Run with coverage
pytest --cov=src tests/
```

---

## 📖 Documentation

- [Architecture Overview](docs/ARCHITECTURE.md) - Detailed architecture documentation
- [Usage Guide](docs/USAGE.md) - Comprehensive usage examples
- [API Documentation](docs/API.md) - Module APIs and functions
- [Terminology Clarification](docs/TERMINOLOGY_CLARIFICATION.md) - **IMPORTANT: TP/NTP vs MFP**

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -m 'Add some feature'`)
4. Push to the branch (`git push origin feature/your-feature`)
5. Open a Pull Request

### Pull Request Guidelines

- Follow the code style guidelines
- Add tests for new functionality
- Update documentation
- Keep commits atomic and well-described
- Reference any related issues

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 📞 Support

For questions or issues, please open a GitHub issue or contact the maintainers.

---

**Maintainers:** Protein Embedding Classifier Team  
**Version:** 1.0  
**Last Updated:** 2026-08-03
