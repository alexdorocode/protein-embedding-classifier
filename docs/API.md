# API Documentation - Protein Embedding Classifier

## 📖 Table of Contents

1. [Input Module API](#-input-module-api)
2. [Dataset Builder Module API](#-dataset-builder-module-api)
3. [Training Module API](#-training-module-api)
4. [Prediction Module API](#-prediction-module-api)
5. [Explainability Module API](#-explainability-module-api)
6. [Output Module API](#-output-module-api)
7. [Orchestration Module API](#-orchestration-module-api)

---

## 📥 Input Module API

### `src.input`

The Input Module provides functionality for loading and validating data from various sources.

#### Classes

##### `CSVLoader`

Load data from CSV files.

**Methods:**

- `__init__()`: Initialize the CSV loader.
- `load(file_path: str, **kwargs) -> pd.DataFrame`: Load a CSV file.
  - `file_path`: Path to the CSV file
  - `**kwargs`: Additional arguments for `pd.read_csv()`
  - Returns: DataFrame with loaded data
  - Raises: `FileNotFoundError`, `pd.errors.EmptyDataError`

**Example:**
```python
from src.input import CSVLoader

loader = CSVLoader()
data = loader.load('data.csv')
```

##### `DatabaseLoader`

Load data from SQL databases.

**Methods:**

- `__init__()`: Initialize the database loader.
- `connect(db_url: str) -> Engine`: Connect to a database.
  - `db_url`: Database connection URL
  - Returns: SQLAlchemy engine
- `load(query: str, db_url: str = None, **kwargs) -> pd.DataFrame`: Load data from a query.
  - `query`: SQL query to execute
  - `db_url`: Database connection URL (optional if already connected)
  - `**kwargs`: Additional arguments for `pd.read_sql()`
  - Returns: DataFrame with query results

**Example:**
```python
from src.input import DatabaseLoader

loader = DatabaseLoader()
loader.connect('postgresql://user:password@localhost/db')
data = loader.load('SELECT * FROM proteins')
```

##### `APILoader`

Load data from bioinformatics APIs.

**Methods:**

- `__init__(base_url: str = None, api_key: str = None)`: Initialize the API loader.
  - `base_url`: Base URL for the API
  - `api_key`: API key for authentication
- `load(endpoint: str, params: dict = None, **kwargs) -> any`: Load data from an API endpoint.
  - `endpoint`: API endpoint (relative or absolute URL)
  - `params`: Query parameters
  - `**kwargs`: Additional arguments
  - Returns: API response data (dict, list, etc.)
- `load_to_dataframe(endpoint: str, params: dict = None, records_path: str = None, **kwargs) -> pd.DataFrame`: Load data and convert to DataFrame.
  - `records_path`: Path to records in JSON response (e.g., 'results.data')
  - Returns: DataFrame with API data
- `load_with_retry(endpoint: str, params: dict = None, max_retries: int = 3, delay: float = 1.0, **kwargs) -> any`: Load with retry logic.
  - `max_retries`: Maximum number of retry attempts
  - `delay`: Delay between retries in seconds
  - Returns: API response data

**Example:**
```python
from src.input import APILoader

loader = APILoader(base_url='https://api.example.com', api_key='your_key')
data = loader.load('/proteins')
df = loader.load_to_dataframe('/proteins', records_path='results')
```

##### `ProteinLoader`

Load protein-specific data.

**Methods:**

- `load(file_path: str, **kwargs) -> any`: Load protein data from a file.

**Example:**
```python
from src.input import ProteinLoader

loader = ProteinLoader()
proteins = loader.load('proteins.fasta')
```

##### `UniverseReader`

Read TP-NTP universe input files.

**Methods:**

- `__init__()`: Initialize the universe reader.
- `read(file_path: str) -> List[UniverseRecord]`: Read universe file.
- `read_csv(file_path: str) -> pd.DataFrame`: Read universe CSV file.

**Example:**
```python
from src.input import UniverseReader

reader = UniverseReader()
records = reader.read('universe.csv')
```

##### `UniverseNormalizer`

Normalize universe records.

**Methods:**

- `__init__()`: Initialize the normalizer.
- `normalize(records: List[UniverseRecord]) -> List[UniverseRecord]`: Normalize records.
- `validate(records: List[UniverseRecord]) -> bool`: Validate records.

**Example:**
```python
from src.input import UniverseNormalizer

normalizer = UniverseNormalizer()
normalized = normalizer.normalize(records)
```

##### `DataValidator`

Validate data structure and content.

**Methods:**

- `validate_schema(data: pd.DataFrame, required_columns: List[str]) -> bool`: Validate required columns.
- `check_nulls(data: pd.DataFrame, threshold: float = 0.0) -> bool`: Check for null values.
- `validate_types(data: pd.DataFrame, expected_types: Dict[str, type]) -> bool`: Validate column types.
- `validate_range(data: pd.DataFrame, column: str, min_val: float = None, max_val: float = None) -> bool`: Validate value ranges.
- `validate_unique(data: pd.DataFrame, column: str) -> bool`: Validate unique values.
- `validate_dataframe(data: pd.DataFrame, schema: Dict[str, Any]) -> bool`: Comprehensive validation.

**Example:**
```python
from src.input import DataValidator

validator = DataValidator()
schema = {
    'required_columns': ['protein_id', 'embedding'],
    'null_threshold': 0.0,
    'types': {'protein_id': str, 'embedding': list}
}
is_valid = validator.validate_dataframe(data, schema)
```

#### Functions

##### `load_data(source: str, source_type: str = 'csv', **kwargs) -> any`

Load data from a specified source.

**Parameters:**
- `source`: Path or identifier for the data source
- `source_type`: Type of source ('csv', 'db', 'api', 'protein')
- `**kwargs`: Additional arguments for the specific loader

**Returns:** Loaded data

**Raises:** `ValueError` if source_type is not supported

**Example:**
```python
from src.input import load_data

data = load_data('data.csv', source_type='csv')
data = load_data('SELECT * FROM table', source_type='db', db_url='...')
```

##### `validate_data(data: any, schema: dict = None) -> bool`

Validate loaded data against a schema.

**Parameters:**
- `data`: Data to validate
- `schema`: Validation schema (optional)

**Returns:** `True` if data is valid

**Example:**
```python
from src.input import validate_data

is_valid = validate_data(data)
is_valid = validate_data(data, schema={'required_columns': ['col1', 'col2']})
```

---

## 🗃️ Dataset Builder Module API

### `src.dataset_builder`

The Dataset Builder Module provides functionality for constructing and managing datasets.

#### Classes

##### `DatasetBuilder`

Build datasets from configurations.

**Methods:**

- `__init__(config: dict)`: Initialize with configuration.
- `build(data: any = None) -> any`: Build a dataset.
- `apply_transformations(data: pd.DataFrame) -> pd.DataFrame`: Apply transformations.
- `integrate_embeddings(data: pd.DataFrame, embeddings: dict) -> pd.DataFrame`: Integrate embeddings.

**Example:**
```python
from src.dataset_builder import DatasetBuilder

config = {'dataset_type': 'tp_ntp', 'ratio': '1:1'}
builder = DatasetBuilder(config)
dataset = builder.build()
```

##### `RunLoader`

Load dataset designer runs.

**Methods:**

- `__init__(base_path: str = 'datasets')`: Initialize with base path.
- `list_runs() -> List[str]`: List all available runs.
- `load_run(run_id: str) -> RunData`: Load a specific run.

**Example:**
```python
from src.dataset_builder import RunLoader, load_run

# Using load_run function
run_data = load_run('20260803_0258_7672b947')

# Using RunLoader class
loader = RunLoader()
run_data = loader.load_run('20260803_0258_7672b947')
```

##### `LabelLoader`

Load labels for datasets.

**Methods:**

- `__init__()`: Initialize the label loader.
- `load(file_path: str) -> pd.DataFrame`: Load labels from file.
- `load_from_run(run_id: str) -> pd.DataFrame`: Load labels from a run.

**Example:**
```python
from src.dataset_builder import LabelLoader

loader = LabelLoader()
labels = loader.load('labels.csv')
```

##### `DatasetVariantGenerator`

Generate dataset variants.

**Methods:**

- `__init__()`: Initialize the generator.
- `generate(universe: any, policy: any, seed: int = 42) -> DatasetVariant`: Generate a variant.

**Example:**
```python
from src.dataset_builder.generators import DatasetVariantGenerator

generator = DatasetVariantGenerator()
variant = generator.generate(universe, policy)
```

##### `BundleExporter`

Export dataset bundles.

**Methods:**

- `__init__(config: dict)`: Initialize with configuration.
- `export(dataset: any, output_path: str) -> None`: Export a dataset bundle.

**Example:**
```python
from src.dataset_builder.export import BundleExporter

exporter = BundleExporter(config)
exporter.export(dataset, 'output_path')
```

##### `LineageBuilder`

Build dataset lineage.

**Methods:**

- `__init__()`: Initialize the lineage builder.
- `build(dataset: any, sources: List[any]) -> LineageManifest`: Build lineage manifest.

**Example:**
```python
from src.dataset_builder.lineage import LineageBuilder

builder = LineageBuilder()
lineage = builder.build(dataset, sources)
```

##### `PolicyValidator`

Validate dataset policies.

**Methods:**

- `__init__()`: Initialize the validator.
- `validate(policy: any, data: any) -> bool`: Validate a policy against data.

**Example:**
```python
from src.dataset_builder.policies import PolicyValidator

validator = PolicyValidator()
is_valid = validator.validate(policy, data)
```

#### Data Models

##### `UniverseRecord`

Represents a row from a target-candidate universe.

**Attributes:**
- `target_id: str` - Unique identifier for the target protein
- `target_label: str` - Label for the target
- `candidate_ids: List[str]` - List of candidate protein IDs
- `candidate_count: int` - Number of candidates
- `source_file: str` - Path to the source file
- `source_row_id: Any` - Row identifier in the source file
- `organism: str` - Organism identifier (optional)
- `taxonomy_id: str` - Taxonomy identifier (optional)
- `pool_metadata: PoolMetadata` - Metadata about candidate pool
- `raw_payload: Any` - Raw data from source (optional)

##### `PoolConstraints`

Constraints applied to the candidate pool.

**Attributes:**
- `len_variance: Optional[float]` - Variance in candidate sequence lengths
- `max_sequence_identity: Optional[float]` - Maximum sequence identity threshold
- `min_candidates: Optional[int]` - Minimum number of candidates required

##### `PoolMetadata`

Metadata about the candidate pool generation.

**Attributes:**
- `generation_source: str` - Source of the candidate pool
- `constraints_snapshot: PoolConstraints` - Constraints applied during generation

##### `UniverseManifest`

Manifest for a normalized target-candidate universe.

**Attributes:**
- `universe_id: str` - Unique identifier
- `source_file: str` - Path to the source input file
- `record_count: int` - Number of UniverseRecord instances
- `target_count: int` - Number of unique targets
- `total_candidates: int` - Total number of candidate entries
- `generated_at: str` - Timestamp of universe creation
- `schema_version: str` - Version of the universe schema

#### Functions

##### `build_dataset(config: dict, data: any = None) -> any`

Build a dataset from configuration.

**Parameters:**
- `config`: Dataset configuration
- `data`: Optional input data

**Returns:** Built dataset

**Example:**
```python
from src.dataset_builder import build_dataset

dataset = build_dataset({'dataset_type': 'tp_ntp'})
```

##### `export_dataset(dataset: any, config: dict) -> None`

Export a dataset.

**Parameters:**
- `dataset`: Dataset to export
- `config`: Export configuration

**Example:**
```python
from src.dataset_builder import export_dataset

export_dataset(dataset, {'format': 'csv', 'path': 'output.csv'})
```

##### `load_run(run_id: str, base_path: str = 'datasets') -> RunData`

Load a dataset designer run.

**Parameters:**
- `run_id`: Run identifier
- `base_path`: Base path for runs

**Returns:** `RunData` object with metadata, pairs, and assignments

**Example:**
```python
from src.dataset_builder import load_run

run_data = load_run('20260803_0258_7672b947')
```

##### `list_runs(base_path: str = 'datasets') -> List[str]`

List all available runs.

**Parameters:**
- `base_path`: Base path for runs

**Returns:** List of run IDs

**Example:**
```python
from src.dataset_builder import list_runs

runs = list_runs()
```

---

## 🎓 Training Module API

### `src.training`

The Training Module provides functionality for training models.

#### Classes

##### `BaseClassifier` (Abstract Base Class)

Base interface for all classifiers.

**Methods:**

- `fit(X: array, y: array) -> None`: Fit the model to training data.
- `predict(X: array) -> array`: Make predictions on new data.
- `predict_proba(X: array) -> array`: Make probability predictions (optional).
- `save(path: str) -> None`: Save the model to file.
- `load(path: str) -> BaseClassifier`: Load the model from file.

**Example:**
```python
from src.training.models.base import BaseClassifier

class MyClassifier(BaseClassifier):
    def fit(self, X, y):
        # Implement training
        pass
    
    def predict(self, X):
        # Implement prediction
        pass
```

##### `MLPProteinClassifier`

Multi-Layer Perceptron classifier for protein embeddings.

**Methods:**

- `__init__(hidden_size: List[int] = [128, 64], dropout: float = 0.2, learning_rate: float = 0.001, **kwargs)`: Initialize the MLP.
- `fit(X: array, y: array) -> None`: Train the MLP.
- `predict(X: array) -> array`: Make predictions.
- `predict_proba(X: array) -> array`: Make probability predictions.

**Example:**
```python
from src.training.models.mlp_protein_classifier import MLPProteinClassifier

model = MLPProteinClassifier(hidden_size=[128, 64], dropout=0.2)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

##### `LinearClassifier`

Logistic Regression classifier.

**Methods:**

- `__init__(C: float = 1.0, max_iter: int = 1000, class_weight: str = 'balanced', random_state: int = 42)`: Initialize the classifier.
- `fit(X: array, y: array) -> None`: Train the classifier.
- `predict(X: array) -> array`: Make predictions.
- `predict_proba(X: array) -> array`: Make probability predictions.

**Example:**
```python
from src.training.models.linear import LogisticRegressionClassifier

model = LogisticRegressionClassifier(C=1.0, max_iter=1000)
model.fit(X_train, y_train)
```

##### `RandomForestClassifierWrapper`

Random Forest classifier wrapper.

**Methods:**

- `__init__(n_estimators: int = 200, max_depth: int = None, min_samples_split: int = 2, random_state: int = 42, n_jobs: int = -1)`: Initialize the classifier.
- `fit(X: array, y: array) -> None`: Train the classifier.
- `predict(X: array) -> array`: Make predictions.
- `predict_proba(X: array) -> array`: Make probability predictions.

**Example:**
```python
from src.training.models.random_forest import RandomForestClassifierWrapper

model = RandomForestClassifierWrapper(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
```

##### `Trainer`

Handles model training with validation and logging.

**Methods:**

- `__init__(config: dict = None)`: Initialize the trainer.
- `train(model: BaseClassifier, data: any, validation_data: any = None) -> BaseClassifier`: Train a model.
- `train_with_split(model: BaseClassifier, data: any, test_size: float = 0.2, random_state: int = 42) -> Tuple[BaseClassifier, dict]`: Train with automatic train/test split.

**Example:**
```python
from src.training import Trainer
from src.training.models.mlp_protein_classifier import MLPProteinClassifier

trainer = Trainer()
model = MLPProteinClassifier()
trained_model, metrics = trainer.train_with_split(model, dataset)
```

##### `WandbIntegration`

Handles Weights & Biases integration.

**Methods:**

- `__init__(project: str = 'protein-embedding-classifier', entity: str = None, config: dict = None)`: Initialize W&B integration.
- `init_run(run_name: str = None, run_config: dict = None) -> None`: Initialize a W&B run.
- `log_metrics(metrics: dict, step: int = None) -> None`: Log metrics to W&B.
- `log_artifact(path: str, name: str, artifact_type: str = 'dataset') -> None`: Log an artifact to W&B.
- `log_model(model: any, model_name: str, model_path: str = None) -> None`: Log a model to W&B.
- `finish_run() -> None`: Finish the current W&B run.

**Example:**
```python
from src.training import WandbIntegration

wandb = WandbIntegration(project='my_project')
wandb.init_run('my_experiment')
wandb.log_metrics({'accuracy': 0.95})
wandb.finish_run()
```

##### `LossFactory`

Factory for creating loss functions.

**Methods:**

- `get_loss(loss_name: str) -> Callable`: Get a loss function by name.
- `binary_crossentropy(y_true: array, y_pred: array) -> float`: Binary cross-entropy loss.
- `categorical_crossentropy(y_true: array, y_pred: array) -> float`: Categorical cross-entropy loss.
- `mse(y_true: array, y_pred: array) -> float`: Mean squared error loss.
- `mae(y_true: array, y_pred: array) -> float`: Mean absolute error loss.

**Example:**
```python
from src.training import LossFactory

loss_fn = LossFactory.get_loss('binary_crossentropy')
loss = loss_fn(y_true, y_pred)
```

##### `MetricFactory`

Factory for creating metric functions.

**Methods:**

- `get_metric(metric_name: str) -> Callable`: Get a metric function by name.
- `accuracy(y_true: array, y_pred: array) -> float`: Calculate accuracy.
- `f1(y_true: array, y_pred: array, average: str = 'weighted') -> float`: Calculate F1 score.
- `precision(y_true: array, y_pred: array, average: str = 'weighted') -> float`: Calculate precision.
- `recall(y_true: array, y_pred: array, average: str = 'weighted') -> float`: Calculate recall.
- `roc_auc(y_true: array, y_pred_proba: array) -> float`: Calculate ROC AUC score.
- `pr_auc(y_true: array, y_pred_proba: array) -> float`: Calculate Precision-Recall AUC score.
- `calculate_all_metrics(y_true: array, y_pred: array, y_pred_proba: array = None) -> dict`: Calculate all available metrics.

**Example:**
```python
from src.training import MetricFactory

metric_fn = MetricFactory.get_metric('f1')
f1 = metric_fn(y_true, y_pred)

# Or calculate all metrics
metrics = MetricFactory.calculate_all_metrics(y_true, y_pred, y_pred_proba)
```

#### Functions

##### `get_classifier(name: str, **kwargs) -> BaseClassifier`

Get a classifier instance by name.

**Parameters:**
- `name`: Name of the classifier ('mlp', 'lr', 'rf')
- `**kwargs`: Arguments to pass to the classifier constructor

**Returns:** Classifier instance

**Raises:** `ValueError` if classifier not found

**Example:**
```python
from src.training import get_classifier

model = get_classifier('mlp')
model = get_classifier('mlp', hidden_size=[128, 64], dropout=0.2)
```

---

## 🔮 Prediction Module API

### `src.prediction`

The Prediction Module provides functionality for making predictions with trained models.

#### Classes

##### `Predictor`

Handles predictions using a trained model.

**Methods:**

- `__init__(model: any)`: Initialize with a trained model.
- `predict(data: Union[array, DataFrame, dict, list], **kwargs) -> any`: Make a prediction.
- `predict_proba(data: Union[array, DataFrame, dict, list], **kwargs) -> array`: Make probability predictions.
- `_prepare_data(data: Union[array, DataFrame, dict, list]) -> array`: Prepare data for prediction.

**Example:**
```python
from src.prediction import Predictor

predictor = Predictor(model)
prediction = predictor.predict(X_test)
probabilities = predictor.predict_proba(X_test)
```

##### `BatchPredictor`

Handles batch predictions.

**Methods:**

- `__init__(model: any, batch_size: int = 32)`: Initialize with a model and batch size.
- `predict_batch(data_list: list, **kwargs) -> list`: Make predictions on a batch of data.
- `predict_batch_proba(data_list: list, **kwargs) -> list`: Make probability predictions on a batch of data.

**Example:**
```python
from src.prediction import BatchPredictor

batch_predictor = BatchPredictor(model, batch_size=64)
predictions = batch_predictor.predict_batch(data_list)
```

#### Functions

##### `predict(model: any, data: any, **kwargs) -> any`

Make a prediction using a trained model.

**Parameters:**
- `model`: Trained model
- `data`: Input data for prediction
- `**kwargs`: Additional prediction arguments

**Returns:** Prediction result

**Example:**
```python
from src.prediction import predict

prediction = predict(model, X_test)
```

##### `batch_predict(model: any, data_list: list, **kwargs) -> list`

Make batch predictions.

**Parameters:**
- `model`: Trained model
- `data_list`: List of input data for prediction
- `**kwargs`: Additional prediction arguments

**Returns:** List of prediction results

**Example:**
```python
from src.prediction import batch_predict

predictions = batch_predict(model, data_list, batch_size=32)
```

---

## 🔍 Explainability Module API

### `src.explainability`

The Explainability Module provides tools for interpreting model predictions.

#### Classes

##### `FeatureImportance`

Calculates feature importance for trained models.

**Methods:**

- `__init__(model: any)`: Initialize with a trained model.
- `explain(data: Union[array, DataFrame], method: str = 'auto', **kwargs) -> dict`: Calculate feature importance.
- `_detect_method() -> str`: Detect the appropriate method for the model.
- `_coef_importance(data: Union[array, DataFrame], **kwargs) -> dict`: Calculate importance from model coefficients.
- `_feature_importances_importance(data: Union[array, DataFrame], **kwargs) -> dict`: Calculate importance from feature_importances_ attribute.
- `_permutation_importance(data: Union[array, DataFrame], n_repeats: int = 10, **kwargs) -> dict`: Calculate permutation feature importance.
- `get_top_features(importance: dict, n: int = 10) -> List[Tuple[str, float]]`: Get top N most important features.

**Example:**
```python
from src.explainability import FeatureImportance

fi = FeatureImportance(model)
explanation = fi.explain(X_test, method='feature_importance')
top_features = fi.get_top_features(explanation, n=10)
```

##### `EmbeddingSaliency`

Generates saliency maps for embedding-based models.

**Methods:**

- `__init__(model: any)`: Initialize with a trained model.
- `explain(embeddings: array, method: str = 'gradient', **kwargs) -> dict`: Generate saliency maps.
- `_gradient_saliency(embeddings: array, target_class: int = None, **kwargs) -> dict`: Calculate saliency using gradients.
- `_integrated_gradients(embeddings: array, steps: int = 50, **kwargs) -> dict`: Calculate integrated gradients saliency.
- `get_salient_features(saliency: dict, n: int = 10) -> List[dict]`: Get most salient features/embedding dimensions.

**Example:**
```python
from src.explainability import EmbeddingSaliency

sal = EmbeddingSaliency(model)
explanation = sal.explain(embeddings, method='gradient')
salient_features = sal.get_salient_features(explanation, n=10)
```

##### `Plotter`

Creates visualizations for explainability results.

**Methods:**

- `__init__()`: Initialize the plotter.
- `plot(data: dict, plot_type: str = 'bar', **kwargs) -> Figure`: Create a plot from explanation data.
- `_plot_bar(data: dict, **kwargs) -> Figure`: Create a bar plot.
- `_plot_heatmap(data: dict, **kwargs) -> Figure`: Create a heatmap plot.
- `_plot_line(data: dict, **kwargs) -> Figure`: Create a line plot.
- `_plot_scatter(data: dict, **kwargs) -> Figure`: Create a scatter plot.
- `plot_feature_importance(importance: dict, n: int = 10, **kwargs) -> Figure`: Plot feature importance.
- `plot_saliency_map(saliency: array, **kwargs) -> Figure`: Plot a saliency map.
- `save_figure(fig: Figure, path: str, **kwargs) -> None`: Save a figure to file.

**Example:**
```python
from src.explainability import Plotter

plotter = Plotter()
fig = plotter.plot(explanation, plot_type='bar')
plotter.save_figure(fig, 'feature_importance.png')
```

#### Functions

##### `explain(model: any, data: any, method: str = 'feature_importance', **kwargs) -> any`

Generate explanations for model predictions.

**Parameters:**
- `model`: Trained model
- `data`: Input data
- `method`: Explainability method to use ('feature_importance', 'embedding_saliency')
- `**kwargs`: Additional arguments for the method

**Returns:** Explanation results

**Raises:** `ValueError` if method not found

**Example:**
```python
from src.explainability import explain

explanation = explain(model, X_test, method='feature_importance')
explanation = explain(model, embeddings, method='gradient')
```

##### `visualize(explanation: any, visualization_type: str = 'bar', **kwargs) -> any`

Visualize explanation results.

**Parameters:**
- `explanation`: Explanation results to visualize
- `visualization_type`: Type of visualization ('bar', 'heatmap', 'line', 'scatter')
- `**kwargs`: Additional visualization arguments

**Returns:** Matplotlib figure

**Raises:** `ValueError` if visualization type not found

**Example:**
```python
from src.explainability import visualize

fig = visualize(explanation, plot_type='bar')
fig = visualize(explanation, plot_type='heatmap', figsize=(10, 8))
```

---

## 💾 Output Module API

### `src.output`

The Output Module provides functionality for saving and managing results.

#### Classes

##### `ResultsManager`

Manages saving and organizing of results.

**Methods:**

- `__init__(output_dir: str = 'results')`: Initialize with output directory.
- `save_results(results: any, name: str, format: str = 'auto', **kwargs) -> Path`: Save results.
- `_detect_format(results: any) -> str`: Detect appropriate format for results.
- `_save_json(results: any, path: Path) -> None`: Save results as JSON.
- `_save_pickle(results: any, path: Path) -> None`: Save results as pickle.
- `_save_csv(results: any, path: Path) -> None`: Save results as CSV.
- `save_experiment(experiment_name: str, results: dict) -> Path`: Save a complete experiment.

**Example:**
```python
from src.output import ResultsManager

manager = ResultsManager('my_results')
manager.save_results({'accuracy': 0.95}, 'metrics', format='json')
manager.save_experiment('experiment_1', {'model': model, 'metrics': metrics})
```

##### `CSVWriter`

Writes data to CSV format.

**Methods:**

- `write(data: any, output_path: str, **kwargs) -> None`: Write data to CSV file.
- `_to_dataframe(data: any) -> pd.DataFrame`: Convert data to DataFrame.

**Example:**
```python
from src.output import CSVWriter

writer = CSVWriter()
writer.write(dataframe, 'output.csv')
writer.write({'col1': [1, 2], 'col2': [3, 4]}, 'output.csv')
```

##### `JSONWriter`

Writes data to JSON format.

**Methods:**

- `write(data: any, output_path: str, **kwargs) -> None`: Write data to JSON file.

**Example:**
```python
from src.output import JSONWriter

writer = JSONWriter()
writer.write({'accuracy': 0.95, 'f1': 0.93}, 'metrics.json')
```

##### `ReportWriter`

Generates reports from results.

**Methods:**

- `__init__()`: Initialize the report writer.
- `generate(results: dict, template: str = 'default', output_path: str = None) -> str`: Generate a report.
- `_generate_default_report(results: dict) -> str`: Generate a default report.
- `_generate_detailed_report(results: dict) -> str`: Generate a detailed report.
- `save_as_html(results: dict, output_path: str, template: str = 'default') -> str`: Save report as HTML.

**Example:**
```python
from src.output import ReportWriter

writer = ReportWriter()
report = writer.generate({'accuracy': 0.95}, template='default')
writer.generate(results, template='detailed', output_path='report.md')
```

#### Functions

##### `save_results(results: any, output_path: str, format: str = 'csv', **kwargs) -> None`

Save results in the specified format.

**Parameters:**
- `results`: Results to save
- `output_path`: Path to save results
- `format`: Output format ('csv', 'json', 'report')
- `**kwargs`: Additional arguments for the writer

**Raises:** `ValueError` if format not found

**Example:**
```python
from src.output import save_results

save_results({'accuracy': 0.95}, 'metrics.json', format='json')
save_results(dataframe, 'data.csv', format='csv')
```

##### `generate_report(results: any, template: str = 'default', **kwargs) -> str`

Generate a report from results.

**Parameters:**
- `results`: Results to include in report
- `template`: Report template to use ('default', 'detailed')
- `**kwargs`: Additional report arguments

**Returns:** Path to generated report or report string

**Example:**
```python
from src.output import generate_report

report = generate_report({'accuracy': 0.95}, template='default')
generate_report(results, template='detailed', output_path='report.html')
```

---

## 🎯 Orchestration Module API

### `src.orchestration`

The Orchestration Module provides functionality for managing experiments, sweeps, and benchmarks.

#### Classes

##### `ExperimentRunner`

Runs individual experiments.

**Methods:**

- `__init__()`: Initialize the experiment runner.
- `run(experiment_name: str, config: dict = None) -> dict`: Run an experiment.

**Example:**
```python
from src.orchestration import ExperimentRunner

runner = ExperimentRunner()
results = runner.run('humans_mlp')
```

##### `ExperimentDefinitions`

Defines available experiments.

**Methods:**

- `__init__()`: Initialize with default experiments.
- `get_experiment(experiment_name: str) -> ExperimentConfig`: Get an experiment configuration.
- `list_experiments() -> List[str]`: List all available experiments.
- `add_experiment(config: ExperimentConfig) -> None`: Add a new experiment configuration.
- `remove_experiment(experiment_name: str) -> None`: Remove an experiment configuration.

**Example:**
```python
from src.orchestration import ExperimentDefinitions

defs = ExperimentDefinitions()
experiments = defs.list_experiments()
exp_config = defs.get_experiment('humans_mlp')
```

##### `SweepManager`

Manages hyperparameter sweeps.

**Methods:**

- `__init__()`: Initialize the sweep manager.
- `run_sweep(sweep_config: dict) -> List[dict]`: Run a hyperparameter sweep.
- `run_grid_search(sweep_config: dict) -> dict`: Run a grid search and return best configuration.

**Example:**
```python
from src.orchestration import SweepManager

manager = SweepManager()
sweep_results = manager.run_sweep({
    'experiment': 'humans_mlp',
    'parameters': {'learning_rate': [0.001, 0.01]},
    'runs': 2
})
```

##### `BenchmarkManager`

Manages model benchmarking and comparison.

**Methods:**

- `__init__()`: Initialize the benchmark manager.
- `run(benchmark_config: dict) -> dict`: Run a benchmark comparison.
- `_generate_comparison_table(results: dict, metrics: List[str]) -> pd.DataFrame`: Generate a comparison table.
- `compare_species(species_list: List[str], model_name: str = 'mlp', config: dict = None) -> dict`: Compare model performance across species.

**Example:**
```python
from src.orchestration import BenchmarkManager

manager = BenchmarkManager()
results = manager.run({
    'name': 'model_comparison',
    'models': ['mlp', 'lr', 'rf'],
    'datasets': ['20260803_0258_7672b947', '20260803_0304_a68aa0bb']
})
```

#### Functions

##### `run_experiment(experiment_name: str, config: dict = None) -> dict`

Run a single experiment.

**Parameters:**
- `experiment_name`: Name of the experiment
- `config`: Experiment configuration (optional)

**Returns:** Experiment results

**Example:**
```python
from src.orchestration import run_experiment

results = run_experiment('humans_mlp')
results = run_experiment('humans_mlp', {'learning_rate': 0.01})
```

##### `run_sweep(sweep_config: dict) -> list`

Run a hyperparameter sweep.

**Parameters:**
- `sweep_config`: Sweep configuration

**Returns:** List of sweep results

**Example:**
```python
from src.orchestration import run_sweep

sweep_results = run_sweep({
    'experiment': 'humans_mlp',
    'parameters': {'learning_rate': [0.001, 0.01]},
    'runs': 2
})
```

##### `run_benchmark(benchmark_config: dict) -> dict`

Run a benchmark comparison.

**Parameters:**
- `benchmark_config`: Benchmark configuration

**Returns:** Benchmark results

**Example:**
```python
from src.orchestration import run_benchmark

benchmark_results = run_benchmark({
    'name': 'model_comparison',
    'models': ['mlp', 'lr', 'rf'],
    'datasets': ['20260803_0258_7672b947']
})
```

---

## 📚 Data Models

### Input Models (`src.input.models`)

- `SourceType` (Enum): CSV, DATABASE, API, PROTEIN, JSON, EXCEL
- `DataFormat` (Enum): DATAFRAME, DICT, LIST, JSON
- `InputConfig` (dataclass): Configuration for data loading

### Dataset Builder Models (`src.dataset_builder.models`)

- `PoolConstraints` (dataclass): Constraints for candidate pool
- `PoolMetadata` (dataclass): Metadata about candidate pool
- `UniverseRecord` (dataclass): Single row from target-candidate universe
- `UniverseManifest` (dataclass): Manifest for normalized universe

---

## 🎓 Best Practices

### Using the API

1. **Import from module, not directly from files:**
   ```python
   # Good
   from src.training import get_classifier
   
   # Bad
   from src.training.models.registry import get_classifier
   ```

2. **Use type hints:**
   ```python
   # Good
   def process_data(data: pd.DataFrame) -> pd.DataFrame:
       pass
   
   # Bad
   def process_data(data):
       pass
   ```

3. **Handle exceptions:**
   ```python
   try:
       model = get_classifier('mlp')
   except ValueError as e:
       print(f"Error: {e}")
   ```

4. **Use logging:**
   ```python
   import logging
   logger = logging.getLogger(__name__)
   logger.info("Processing data...")
   ```

---

## 📞 Support

For additional help or questions about the API:
- Check the [Architecture Documentation](ARCHITECTURE.md)
- Check the [Usage Guide](USAGE.md)
- Open a GitHub issue
- Contact the maintainers

---

**Last Updated:** 2026-08-03  
**Version:** 1.0  
**Maintainers:** Protein Embedding Classifier Team
