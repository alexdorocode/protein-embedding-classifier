import numpy as np
import pytest

from src.training.embedding_loading import (
    EmbeddingBundle,
    GOEmbeddingLoader,
    LayerAggregationStrategy,
)
from src.dataset_builder.dataset_builder import DatasetBundle


def _sample_layered_embeddings():
    return {
        "ESM3c": {
            "P1": {
                0: np.array([1.0, 2.0], dtype=np.float32),
                1: np.array([3.0, 4.0], dtype=np.float32),
            },
            "P2": {
                0: np.array([5.0, 6.0], dtype=np.float32),
                1: np.array([7.0, 8.0], dtype=np.float32),
            },
        }
    }


def test_layer_aggregation_mean_max_concat_modes():
    layered = _sample_layered_embeddings()

    mean_result = LayerAggregationStrategy("mean").aggregate(layered)
    assert np.allclose(mean_result["ESM3c"]["P1"], np.array([2.0, 3.0], dtype=np.float32))

    max_result = LayerAggregationStrategy("max").aggregate(layered)
    assert np.allclose(max_result["ESM3c"]["P1"], np.array([3.0, 4.0], dtype=np.float32))

    mean_max_result = LayerAggregationStrategy("mean_max").aggregate(layered)
    assert np.allclose(mean_max_result["ESM3c"]["P1"], np.array([2.0, 3.0, 3.0, 4.0], dtype=np.float32))

    concat_result = LayerAggregationStrategy("concat").aggregate(layered)
    assert np.allclose(concat_result["ESM3c"]["P1"], np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32))


def test_layer_aggregation_none_splits_multilayer_model_views():
    layered = _sample_layered_embeddings()
    result = LayerAggregationStrategy("none").aggregate(layered)

    assert "ESM3c__layer_0" in result
    assert "ESM3c__layer_1" in result
    assert np.allclose(result["ESM3c__layer_0"]["P1"], np.array([1.0, 2.0], dtype=np.float32))
    assert np.allclose(result["ESM3c__layer_1"]["P1"], np.array([3.0, 4.0], dtype=np.float32))


def test_embedding_bundle_builds_ordered_matrices():
    dataset_bundle = DatasetBundle(
        train_ids=["P2", "P1"],
        val_ids=["P3"],
        test_ids=["P4"],
        y_train=np.array(["a", "b"], dtype=object),
        y_val=np.array(["c"], dtype=object),
        y_test=np.array(["d"], dtype=object),
    )

    raw_embeddings = {
        "ESM3c": {
            "P1": np.array([1.0, 1.0], dtype=np.float32),
            "P2": np.array([2.0, 2.0], dtype=np.float32),
            "P3": np.array([3.0, 3.0], dtype=np.float32),
            "P4": np.array([4.0, 4.0], dtype=np.float32),
        }
    }

    bundle = EmbeddingBundle.from_dataset(dataset_bundle, raw_embeddings)

    assert bundle.X_train["ESM3c"].shape == (2, 2)
    assert np.allclose(bundle.X_train["ESM3c"], np.array([[2.0, 2.0], [1.0, 1.0]], dtype=np.float32))
    assert np.allclose(bundle.X_val["ESM3c"], np.array([[3.0, 3.0]], dtype=np.float32))
    assert np.allclose(bundle.X_test["ESM3c"], np.array([[4.0, 4.0]], dtype=np.float32))


def test_embedding_bundle_raises_on_missing_embeddings():
    dataset_bundle = DatasetBundle(
        train_ids=["P1"],
        val_ids=["P2"],
        test_ids=["P3"],
        y_train=np.array(["a"], dtype=object),
        y_val=np.array(["b"], dtype=object),
        y_test=np.array(["c"], dtype=object),
    )

    raw_embeddings = {
        "GeOKG": {
            "P1": np.array([1.0], dtype=np.float32),
            "P2": np.array([2.0], dtype=np.float32),
        }
    }

    with pytest.raises(ValueError, match="Missing embeddings"):
        EmbeddingBundle.from_dataset(dataset_bundle, raw_embeddings)


def test_go_embedding_loader_returns_geokg_view(tmp_path):
    go_dir = tmp_path / "go"
    go_dir.mkdir(parents=True, exist_ok=True)

    csv_bp = go_dir / "bp.csv"
    csv_mf = go_dir / "mf.csv"
    csv_cc = go_dir / "cc.csv"

    csv_bp.write_text('UniProt_ID,gope\nP1,"[0.1, 0.2]"\n', encoding="utf-8")
    csv_mf.write_text('UniProt_ID,gope\nP1,"[0.3, 0.4]"\n', encoding="utf-8")
    csv_cc.write_text('UniProt_ID,gope\nP1,"[0.5, 0.6]"\n', encoding="utf-8")

    loader = GOEmbeddingLoader()
    result = loader.load(
        {
            "GOPE": {
                "enabled": True,
                "file_info": {
                    "folder": str(go_dir),
                    "file_name_BP": "bp.csv",
                    "file_name_MF": "mf.csv",
                    "file_name_CC": "cc.csv",
                    "accession_column": "UniProt_ID",
                    "embedding_column": "gope",
                },
                "models": {"GeOKG": {"enabled": True}},
            }
        },
        accessions=["P1"],
    )

    assert "GeOKG" in result
    assert "P1" in result["GeOKG"]
    assert result["GeOKG"]["P1"].shape == (6,)