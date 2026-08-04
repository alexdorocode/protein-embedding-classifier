# Usage Guide - Protein Embedding Classifier

## 📖 Table of Contents

1. [Quick Start](#-quick-start)
2. [Module Usage](#-module-usage)
   - [Input Module](#input-module)
   - [Dataset Builder Module](#dataset-builder-module)
   - [Training Module](#training-module)
   - [Prediction Module](#prediction-module)
   - [Explainability Module](#explainability-module)
   - [Output Module](#output-module)
   - [Orchestration Module](#orchestration-module)
3. [CLI Usage](#-cli-usage)
4. [Common Workflows](#-common-workflows)
5. [Configuration](#-configuration)
6. [Examples](#-examples)

---

## 🚀 Quick Start

### Import the Modules

```python
# Import all modules
from src.input import CSVLoader, DataValidator
from src.dataset_builder import DatasetBuilder, load_run, list_runs
from src.training import get_classifier, BaseClassifier
from src.prediction import predict, batch_predict, Predictor
from src.explainability import explain, visualize, FeatureImportance
from src.output import save_results, ResultsManager
from src.orchestration import run_experiment, run_sweep, run_benchmark
```

### Basic Workflow

```python
# 1. Load data
from src.dataset_builder import load_run
run_data = load_run('20260803_0258_7672b947')
dataset = run_data.tp_ntp_pairs

# 2. Train model
from src.training import get_classifier
model = get_classifier('mlp')
# model.fit(X_train, y_train)  # Train with your data

# 3. Make predictions
from src.prediction import predict
predictions = predict(model, X_test)

# 4. Explain predictions
from src.explainability import explain
explanation = explain(model, X_test, method='feature_importance')

# 5. Save results
from src.output import save_results
save_results({'predictions': predictions, 'model': model}, 'my_experiment')
```

---

## 📦 Module Usage

### Input Module

The Input Module handles loading data from various sources.

#### Load CSV Data

```python
from src.input import CSVLoader

loader = CSVLoader()
data = loader.load('path/to/data.csv')
```

#### Load from Database

```python
from src.input import DatabaseLoader

loader = DatabaseLoader()
loader.connect('postgresql://user:password@localhost/db')
data = loader.load('SELECT * FROM proteins')
```

#### Load Protein Data

```python
from src.input import ProteinLoader

loader = ProteinLoader()
proteins = loader.load('path/to/proteins.fasta')
```

#### Validate Data

```python
from src.input import DataValidator, CSVLoader

loader = CSVLoader()
data = loader.load('data.csv')

validator = DataValidator()
schema = {
    'required_columns': ['protein_id', 'embedding'],
    'null_threshold': 0.0,
    'types': {'protein_id': str, 'embedding': list}
}

is_valid = validator.validate_dataframe(data, schema)
print(f"Data is valid: {is_valid}")
```

---

### Dataset Builder Module

The Dataset Builder Module handles dataset construction and management.

#### Load a Run

```python
from src.dataset_builder import load_run

# Load humans dataset
run_data = load_run('20260803_0258_7672b947')

# Access components
print(run_data.run_id)           # Run identifier
print(run_data.metadata)         # Run metadata
print(run_data.tp_ntp_pairs)     # TP-NTP pairs DataFrame
print(run_data.assignments)      # TP to NTP assignments
```

#### List Available Runs

```python
from src.dataset_builder import list_runs

runs = list_runs()
print(f"Available runs: {runs}")
# Output: ['20260803_0258_7672b947', '20260803_0304_a68aa0bb']
```

#### Build a Dataset

```python
from src.dataset_builder import DatasetBuilder

config = {
    'dataset_type': 'tp_ntp',
    'ratio': '1:1',
    'random_seed': 42
}

builder = DatasetBuilder(config)
dataset = builder.build()
```

#### Access Run Metadata

```python
from src.dataset_builder import load_run

run_data = load_run('20260803_0258_7672b947')
metadata = run_data.metadata

print(f"Run ID: {metadata['run_id']}")
print(f"Species: {metadata['species']}")
print(f"Species Category: {metadata['species_category']}")
print(f"Timestamp: {metadata['timestamp']}")
print(f"Statistics: {metadata['statistics']}")
```

---

### Training Module

The Training Module handles model training and validation.

#### Get a Classifier

```python
from src.training import get_classifier

# Available models: 'mlp', 'lr' (logistic regression), 'rf' (random forest)
model = get_classifier('mlp')
```

#### Get a Classifier with Custom Configuration

```python
from src.training import get_classifier

model = get_classifier(
    'mlp',
    hidden_size=[128, 64, 32],
    dropout=0.2,
    learning_rate=0.001,
    batch_size=32
)
```

#### Train a Model

```python
from src.training import get_classifier
import numpy as np

# Generate sample data
X_train = np.random.randn(100, 10)  # 100 samples, 10 features
y_train = np.random.randint(0, 2, 100)  # Binary labels

# Get and train model
model = get_classifier('mlp')
model.fit(X_train, y_train)
```

#### Use BaseClassifier Interface

```python
from src.training import BaseClassifier

class MyCustomClassifier(BaseClassifier):
    def fit(self, X, y):
        # Implement training
        pass
    
    def predict(self, X):
        # Implement prediction
        pass
```

---

### Prediction Module

The Prediction Module handles model inference.

#### Make a Single Prediction

```python
from src.prediction import predict
import numpy as np

# Sample data
X_test = np.random.randn(1, 10)  # Single sample

# Make prediction
prediction = predict(model, X_test)
print(f"Prediction: {prediction}")
```

#### Make Batch Predictions

```python
from src.prediction import batch_predict
import numpy as np

# Multiple samples
X_test = np.random.randn(10, 10)  # 10 samples

# Batch prediction with batch size of 5
predictions = batch_predict(model, X_test.tolist(), batch_size=5)
print(f"Predictions: {predictions}")
```

#### Use Predictor Class

```python
from src.prediction import Predictor

predictor = Predictor(model)

# Single prediction
prediction = predictor.predict(X_test[0])

# Probability prediction
probabilities = predictor.predict_proba(X_test[0])

# Batch prediction
batch_predictions = predictor.predict_batch(X_test.tolist())
```

---

### Explainability Module

The Explainability Module provides tools for interpreting model predictions.

#### Feature Importance

```python
from src.explainability import explain, FeatureImportance
import numpy as np

# Sample data
X_test = np.random.randn(10, 10)

# Explain using feature importance
explanation = explain(model, X_test, method='feature_importance')

print(f"Method: {explanation['method']}")
print(f"Feature importance: {explanation['importance']}")
print(f"Top features: {explanation['feature_names'][:5]}")
```

#### Embedding Saliency

```python
from src.explainability import explain
import numpy as np

# Sample embeddings
embeddings = np.random.randn(5, 300)  # 5 samples, 300-dimensional embeddings

# Explain using saliency
explanation = explain(model, embeddings, method='gradient')

print(f"Method: {explanation['method']}")
print(f"Saliency maps: {len(explanation['saliency_maps'])}")
```

#### Visualize Explanations

```python
from src.explainability import explain, visualize

# Get explanation
explanation = explain(model, X_test, method='feature_importance')

# Create bar plot
fig = visualize(explanation, plot_type='bar')
fig.savefig('feature_importance.png')

# Create heatmap
fig = visualize(explanation, plot_type='heatmap')
fig.savefig('saliency_heatmap.png')
```

#### Get Top Features

```python
from src.explainability import FeatureImportance

fi = FeatureImportance(model)
explanation = fi.explain(X_test, method='coef')
top_features = fi.get_top_features(explanation, n=5)

for feature, importance in top_features:
    print(f"{feature}: {importance:.4f}")
```

---

### Output Module

The Output Module handles saving and managing results.

#### Save Results

```python
from src.output import save_results
import pandas as pd

# Sample results
results = {
    'predictions': pd.DataFrame({'pred': [0, 1, 0, 1]}),
    'metrics': {'accuracy': 0.95, 'f1': 0.93},
    'model_type': 'mlp'
}

# Save as JSON
save_results(results, 'experiment_results', format='json')

# Save as CSV (for DataFrame)
save_results(results['predictions'], 'predictions', format='csv')

# Save as pickle
save_results(results, 'experiment_results', format='pkl')
```

#### Use ResultsManager

```python
from src.output import ResultsManager

manager = ResultsManager(output_dir='my_results')

# Save results
manager.save_results({'accuracy': 0.95}, 'metrics', format='json')

# Save experiment
manager.save_experiment('my_experiment', {
    'model': model,
    'predictions': predictions,
    'metrics': {'accuracy': 0.95}
})
```

#### Generate Reports

```python
from src.output import generate_report

results = {
    'experiment_name': 'humans_mlp',
    'config': {'model': 'mlp', 'dataset': '20260803_0258_7672b947'},
    'metrics': {'accuracy': 0.95, 'f1': 0.93}
}

# Generate markdown report
report = generate_report(results, template='default')
print(report)

# Save as HTML
generate_report(results, template='default', output_path='report.html')
```

---

### Orchestration Module

The Orchestration Module handles experiment management and benchmarking.

#### Run a Single Experiment

```python
from src.orchestration import run_experiment

# Run a predefined experiment
results = run_experiment('humans_mlp')

print(f"Experiment: {results['experiment_name']}")
print(f"Timestamp: {results['timestamp']}")
print(f"Config: {results['config']}")
```

#### Run a Hyperparameter Sweep

```python
from src.orchestration import run_sweep

sweep_config = {
    'experiment': 'humans_mlp',
    'parameters': {
        'learning_rate': [0.001, 0.01, 0.1],
        'hidden_size': [[64, 32], [128, 64]],
        'dropout': [0.1, 0.2, 0.3]
    },
    'runs': 2,  # Run each combination twice
    'metric': 'accuracy'  # Metric to optimize
}

sweep_results = run_sweep(sweep_config)
print(f"Ran {len(sweep_results)} sweep combinations")
```

#### Run a Benchmark

```python
from src.orchestration import run_benchmark

benchmark_config = {
    'name': 'model_comparison',
    'models': ['mlp', 'lr', 'rf'],
    'datasets': ['20260803_0258_7672b947', '20260803_0304_a68aa0bb'],
    'metrics': ['accuracy', 'f1', 'precision', 'recall'],
    'training_config': {
        'random_state': 42
    }
}

benchmark_results = run_benchmark(benchmark_config)
print(f"Benchmark: {benchmark_results['benchmark_name']}")
print(f"Comparison table:\n{benchmark_results['comparison']}")
```

#### Cross-Species Comparison

```python
from src.orchestration import BenchmarkManager

manager = BenchmarkManager()
results = manager.compare_species(
    species_list=['20260803_0258_7672b947', '20260803_0304_a68aa0bb'],
    model_name='mlp',
    config={'random_state': 42}
)

print(f"Cross-species comparison results: {results}")
```

---

## 💻 CLI Usage

### Train a Model

```bash
# Basic training
python -m scripts.train --dataset 20260803_0258_7672b947 --model mlp

# With custom output directory
python -m scripts.train --dataset 20260803_0258_7672b947 --model mlp --output my_results

# With W&B logging
python -m scripts.train --dataset 20260803_0258_7672b947 --model mlp --wandb --project my_project

# With configuration file
python -m scripts.train --dataset 20260803_0258_7672b947 --model mlp --config configs/my_config.yaml
```

### Make Predictions

```bash
# Basic prediction
python -m scripts.predict --model path/to/model.pkl --data path/to/test_data.csv

# With batch processing
python -m scripts.predict --model path/to/model.pkl --data path/to/test_data.csv --batch-size 64

# Get probability predictions
python -m scripts.predict --model path/to/model.pkl --data path/to/test_data.csv --probabilities

# Save to custom output
python -m scripts.predict --model path/to/model.pkl --data path/to/test_data.csv --output my_predictions.csv
```

### Run Benchmark

```bash
# Run with configuration file
python -m scripts.benchmark --config configs/benchmark.yaml

# With W&B logging
python -m scripts.benchmark --config configs/benchmark.yaml --wandb

# Save to custom output
python -m scripts.benchmark --config configs/benchmark.yaml --output my_benchmark.json
```

### Export Dataset

```bash
# Export to CSV
python -m scripts.export_dataset --run 20260803_0258_7672b947 --format csv

# Export to JSON with metadata
python -m scripts.export_dataset --run 20260803_0258_7672b947 --format json --include-metadata

# Export to custom path
python -m scripts.export_dataset --run 20260803_0258_7672b947 --output my_dataset.csv
```

---

## 🔄 Common Workflows

### Workflow 1: Train and Evaluate on Humans Dataset

```python
from src.dataset_builder import load_run
from src.training import get_classifier
from src.prediction import predict
from src.training.metrics import MetricFactory

# 1. Load dataset
run_data = load_run('20260803_0258_7672b947')
dataset = run_data.tp_ntp_pairs

# 2. Prepare data (simplified example)
# In practice, you would split into train/test and prepare features
X_train = dataset.iloc[:, :-1].values  # Features
y_train = dataset.iloc[:, -1].values   # Labels (if available)

# 3. Train model
model = get_classifier('mlp')
model.fit(X_train, y_train)

# 4. Evaluate
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2)
model.fit(X_train, y_train)

# Calculate metrics
from sklearn.metrics import accuracy_score, f1_score
metrics = {
    'accuracy': accuracy_score(y_test, model.predict(X_test)),
    'f1': f1_score(y_test, model.predict(X_test), average='weighted')
}
print(f"Metrics: {metrics}")
```

### Workflow 2: Cross-Species Transfer Learning

```python
from src.dataset_builder import load_run
from src.training import get_classifier
from src.orchestration import BenchmarkManager

# Train on model organisms, test on humans
manager = BenchmarkManager()

results = manager.compare_species(
    species_list=['20260803_0304_a68aa0bb', '20260803_0258_7672b947'],
    model_name='mlp',
    config={'random_state': 42}
)

print("Cross-species transfer learning results:")
print(results)
```

### Workflow 3: Hyperparameter Optimization

```python
from src.orchestration import run_sweep

sweep_config = {
    'experiment': 'humans_mlp',
    'parameters': {
        'learning_rate': [0.0001, 0.001, 0.01],
        'hidden_size': [[64], [128], [256, 128]],
        'dropout': [0.1, 0.2, 0.3],
        'batch_size': [16, 32, 64]
    },
    'runs': 2,
    'metric': 'accuracy'
}

sweep_results = run_sweep(sweep_config)

# Find best configuration
best_result = None
best_accuracy = 0
for result in sweep_results:
    if result['metrics']['accuracy'] > best_accuracy:
        best_accuracy = result['metrics']['accuracy']
        best_result = result

print(f"Best accuracy: {best_accuracy}")
print(f"Best config: {best_result['config']}")
```

### Workflow 4: Model Explainability

```python
from src.dataset_builder import load_run
from src.training import get_classifier
from src.explainability import explain, visualize, FeatureImportance
import numpy as np

# Load data
run_data = load_run('20260803_0258_7672b947')
dataset = run_data.tp_ntp_pairs

# Train model
model = get_classifier('mlp')
# model.fit(X_train, y_train)  # Train first

# Select a sample for explanation
sample = dataset.iloc[:10]  # First 10 samples
X_sample = sample.iloc[:, :-1].values if len(sample.columns) > 1 else sample.values

# Generate explanations
explanation = explain(model, X_sample, method='feature_importance')

# Visualize
fig = visualize(explanation, plot_type='bar', figsize=(12, 6))
fig.savefig('feature_importance.png')

# Get top features
fi = FeatureImportance(model)
top_features = fi.get_top_features(explanation, n=10)
print("Top 10 features:")
for feature, importance in top_features:
    print(f"  {feature}: {importance:.4f}")
```

### Workflow 5: Complete Experiment Pipeline

```python
from src.dataset_builder import load_run
from src.training import get_classifier
from src.prediction import predict
from src.explainability import explain
from src.output import save_results, generate_report
from datetime import datetime

# 1. Load dataset
run_id = '20260803_0258_7672b947'
run_data = load_run(run_id)
dataset = run_data.tp_ntp_pairs

# 2. Prepare data
# Split into features and labels
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values if len(dataset.columns) > 1 else None

# 3. Train model
model = get_classifier('mlp')
model.fit(X, y)

# 4. Make predictions
predictions = predict(model, X[:10])  # Predict on first 10 samples

# 5. Generate explanations
explanation = explain(model, X[:10], method='feature_importance')

# 6. Save results
results = {
    'experiment_name': f'full_pipeline_{datetime.now().strftime("%Y%m%d_%H%M")}',
    'run_id': run_id,
    'species': run_data.metadata['species'],
    'model_type': 'mlp',
    'num_samples': len(dataset),
    'predictions': predictions.tolist() if hasattr(predictions, 'tolist') else predictions,
    'explanation': explanation,
    'timestamp': datetime.now().isoformat()
}

save_results(results, f'full_pipeline_{run_id}', format='json')

# 7. Generate report
report = generate_report(results, template='detailed')
with open(f'full_pipeline_{run_id}_report.md', 'w') as f:
    f.write(report)

print("✅ Complete experiment pipeline finished!")
```

---

## ⚙️ Configuration

PEC uses YAML files for configuration. All configuration files are located in the `configs/` directory.

### Configuration Structure

```yaml
# configs/experiments/training_config.yaml

# Training configuration
training:
  model_type: mlp
  epochs: 100
  batch_size: 32
  learning_rate: 0.001
  hidden_size: [128, 64]
  dropout: 0.2
  random_state: 42
  early_stopping: true
  patience: 10

# Evaluation configuration
evaluation:
  test_size: 0.2
  random_state: 42
  metrics:
    - accuracy
    - f1
    - precision
    - recall
    - roc_auc

# Weights & Biases configuration
wandb:
  enabled: true
  project: protein-embedding-classifier
  entity: your-team-name
  run_name: experiment_${timestamp}
  log_metrics: true
  log_model: true
  save_code: true
```

### Available Configuration Files

```
configs/
├── datasets/
│   ├── humans.yaml              # Humans dataset configuration
│   ├── model_organisms.yaml     # Model organisms dataset configuration
│   └── template.yaml            # Dataset configuration template
│
├── models/
│   ├── mlp.yaml                 # MLP model configuration
│   ├── random_forest.yaml       # Random Forest configuration
│   └── template.yaml            # Model configuration template
│
├── experiments/
│   ├── benchmark.yaml           # Benchmark configuration
│   ├── cross_species.yaml       # Cross-species configuration
│   ├── runs_config.yaml         # Runs configuration
│   └── template.yaml            # Experiment configuration template
│
└── sweeps/
    ├── mlp_sweep.yaml           # MLP hyperparameter sweep
    ├── rf_sweep.yaml            # Random Forest sweep
    └── template.yaml            # Sweep configuration template
```

### Using Configuration Files

```python
import yaml

# Load configuration
with open('configs/experiments/training_config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Access configuration
print(config['training']['learning_rate'])
print(config['evaluation']['test_size'])
```

---

## 📚 Examples

See the `docs/examples/` directory for complete example scripts:

- `train_model.py` - Complete training example
- `make_prediction.py` - Prediction example
- `run_benchmark.py` - Benchmarking example
- `explain_model.py` - Explainability example
- `export_dataset.py` - Dataset export example

---

## 🆘 Troubleshooting

### Common Issues

#### ImportError: Module not found

**Problem:** `ImportError: No module named 'src'`

**Solution:** 
1. Make sure you're running from the project root directory
2. Install the package in development mode: `pip install -e .`
3. Or add the project root to PYTHONPATH: `export PYTHONPATH=$(pwd):$PYTHONPATH`

#### ModuleNotFoundError: No module named 'pandas'

**Problem:** Missing dependencies

**Solution:** Install dependencies with Poetry or pip:
```bash
poetry install
# or
pip install pandas numpy scikit-learn pyyaml wandb
```

#### FileNotFoundError: Dataset not found

**Problem:** `FileNotFoundError: [Errno 2] No such file or directory: 'datasets/20260803_0258_7672b947/final_dataset.csv'`

**Solution:** 
1. Check that the dataset exists in the `datasets/` directory
2. Verify the run ID is correct
3. Use the correct base path: `load_run(run_id, base_path='dataset_designer_runs')`

#### ValueError: Unknown classifier

**Problem:** `ValueError: Unknown classifier: xyz`

**Solution:** Check available classifiers:
```python
from src.orchestration import ExperimentDefinitions
defs = ExperimentDefinitions()
print(defs.list_experiments())
```

---

## 📞 Support

For additional help:
- Check the [Architecture Documentation](ARCHITECTURE.md)
- Check the [API Documentation](API.md)
- Open a GitHub issue
- Contact the maintainers

---

**Last Updated:** 2026-08-03  
**Version:** 1.0  
**Maintainers:** Protein Embedding Classifier Team
