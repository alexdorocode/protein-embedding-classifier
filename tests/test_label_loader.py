import pandas as pd

from protein_embedding_classifier.data.label_loader import LabelLoader


def test_label_loader_collapses_singleton_duplicate_labels(tmp_path):
    csv_path = tmp_path / "labels.csv"
    df = pd.DataFrame(
        {
            "accession": ["P1", "P1", "P2"],
            "label": [True, True, False],
        }
    )
    df.to_csv(csv_path, index=False)

    loader = LabelLoader(
        source="file",
        file_path=str(csv_path),
        accession_column="accession",
        label_column="label",
        artifacts_dir=str(tmp_path / "artifacts"),
    )

    labels = loader.load(db_accessions=["P1", "P2"])

    assert labels["P1"] is True
    assert labels["P2"] is False


def test_label_loader_keeps_true_multilabel_values(tmp_path):
    csv_path = tmp_path / "labels.csv"
    df = pd.DataFrame(
        {
            "accession": ["P1", "P1", "P2"],
            "label": ["GO:1", "GO:2", "GO:1"],
        }
    )
    df.to_csv(csv_path, index=False)

    loader = LabelLoader(
        source="file",
        file_path=str(csv_path),
        accession_column="accession",
        label_column="label",
        artifacts_dir=str(tmp_path / "artifacts"),
    )

    labels = loader.load(db_accessions=["P1", "P2"])

    assert labels["P1"] == ["GO:1", "GO:2"]
    assert labels["P2"] == "GO:1"
