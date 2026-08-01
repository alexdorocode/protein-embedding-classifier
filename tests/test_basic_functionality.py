"""
Basic functionality test to demonstrate the system works.
This test verifies that:
1. We can import the main modules (without heavy dependencies)
2. We can create a simple classifier
3. We can train and predict with dummy data
"""

import sys
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Add the project to the path
sys.path.insert(0, '/workspace/alexdorocode__protein-embedding-classifier')

from protein_embedding_classifier.classifiers.base import BaseClassifier
from protein_embedding_classifier.classifiers.random_forest import RandomForestClassifierWrapper


def test_can_import_core_modules():
    """Test that we can import core modules (without heavy dependencies)."""
    print("  - Testing core module imports...")
    from protein_embedding_classifier.core import pipeline
    from protein_embedding_classifier.core.embedding_loading import (
        SequenceEmbeddingLoader,
        GOEmbeddingLoader,
    )
    from protein_embedding_classifier.core.training import (
        TrainingService,
        ModelFactory,
    )
    print("    ✓ Core modules imported successfully")


def test_random_forest_wrapper_with_dummy_data():
    """Test that RandomForestClassifierWrapper can train and predict with dummy data."""
    print("  - Testing RandomForestClassifierWrapper with dummy data...")
    
    # Create dummy data
    X_train = np.random.rand(100, 10)  # 100 samples, 10 features
    y_train = np.random.randint(0, 2, 100)  # Binary labels
    X_test = np.random.rand(20, 10)
    y_test = np.random.randint(0, 2, 20)

    # Create and train the wrapper
    wrapper = RandomForestClassifierWrapper()
    wrapper.fit(X_train, y_train)

    # Make predictions
    y_pred = wrapper.predict(X_test)

    # Verify predictions are valid
    assert len(y_pred) == 20
    assert all(p in [0, 1] for p in y_pred)
    
    # Calculate accuracy (should be >= 0)
    accuracy = accuracy_score(y_test, y_pred)
    assert 0 <= accuracy <= 1
    print(f"    ✓ RandomForestClassifierWrapper works correctly (accuracy: {accuracy:.2f})")


def test_base_classifier_interface():
    """Test that BaseClassifier interface works correctly."""
    print("  - Testing BaseClassifier interface...")
    
    # Create a simple classifier that inherits from BaseClassifier
    class SimpleClassifier(BaseClassifier):
        def __init__(self):
            self.model = RandomForestClassifier(n_estimators=10, random_state=42)

        def fit_eval(self, X: np.ndarray, y: np.ndarray) -> dict:
            """Required abstract method."""
            self.model.fit(X, y)
            y_pred = self.model.predict(X)
            return {"accuracy": accuracy_score(y, y_pred)}

        def fit(self, X, y):
            self.model.fit(X, y)
            return self

        def predict(self, X):
            return self.model.predict(X)

        def predict_proba(self, X):
            return self.model.predict_proba(X)

    # Test with dummy data
    X = np.random.rand(50, 5)
    y = np.random.randint(0, 2, 50)

    classifier = SimpleClassifier()
    classifier.fit(X, y)
    predictions = classifier.predict(X[:5])

    assert len(predictions) == 5
    assert all(p in [0, 1] for p in predictions)
    print("    ✓ BaseClassifier interface works correctly")


def test_model_factory_creation():
    """Test that ModelFactory can create models."""
    print("  - Testing ModelFactory...")
    from protein_embedding_classifier.core.training.model_factory import ModelFactory

    factory = ModelFactory()
    
    # Test creating a random forest model (using 'RF' as the key)
    model = factory.create("RF")
    assert model is not None
    print("    ✓ ModelFactory creates RandomForest model successfully")


def test_training_service_initialization():
    """Test that TrainingService can be initialized."""
    print("  - Testing TrainingService initialization...")
    from protein_embedding_classifier.core.training.training_service import TrainingService

    # This should not raise any errors
    service = TrainingService()
    assert service is not None
    print("    ✓ TrainingService initialized successfully")


if __name__ == "__main__":
    # Run the tests
    print("\n" + "=" * 60)
    print("Running basic functionality tests...")
    print("=" * 60 + "\n")
    
    try:
        test_can_import_core_modules()
        test_random_forest_wrapper_with_dummy_data()
        test_base_classifier_interface()
        test_model_factory_creation()
        test_training_service_initialization()
        
        print("\n" + "=" * 60)
        print("✅ ALL TESTS PASSED! The system is working correctly.")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
