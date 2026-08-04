from __future__ import annotations

import csv

import pytest

from src.dataset_builder.splits.independent import IndependentValidationTrainTestSplit


def _write_split_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["accession", "split"])
        writer.writeheader()
        writer.writerows(rows)


def _metadata_for(accessions):
    return {acc: {"organism": "Homo sapiens"} for acc in accessions}


def _labels_for(accessions):
    return {acc: int(idx % 2) for idx, acc in enumerate(accessions)}


def test_csv_validation_split_partitions_correctly(tmp_path):
    accessions = [f"A{i}" for i in range(1, 9)]
    labels = _labels_for(accessions)
    metadata = _metadata_for(accessions)

    split_csv = tmp_path / "split.csv"
    _write_split_csv(
        split_csv,
        [
            {"accession": "A1", "split": "validation"},
            {"accession": "A2", "split": "validation"},
        ],
    )

    conf = {
        "validation": {
            "strategy": "csv",
            "csv": {
                "csv_path": str(split_csv),
                "accession_column": "accession",
                "split_column": "split",
                "validation_values": ["validation"],
            },
        },
        "train_test": {
            "strategy": "random",
            "random": {"test_size": 0.25, "random_state": 7},
        },
    }

    manager = IndependentValidationTrainTestSplit(conf)
    train_ids, val_ids, test_ids = manager.split(accessions, labels, metadata)

    assert set(val_ids) == {"A1", "A2"}
    assert not set(train_ids).intersection(val_ids)
    assert not set(test_ids).intersection(val_ids)
    assert set(train_ids).union(set(val_ids)).union(set(test_ids)) == set(accessions)


def test_zero_shot_overlap_raises_error(tmp_path):
    accessions = [f"A{i}" for i in range(1, 9)]
    labels = _labels_for(accessions)
    metadata = _metadata_for(accessions)

    split_csv = tmp_path / "split.csv"
    _write_split_csv(
        split_csv,
        [
            {"accession": "A1", "split": "validation"},
            {"accession": "A1", "split": "zs"},
        ],
    )

    conf = {
        "validation": {
            "strategy": "csv",
            "csv": {
                "csv_path": str(split_csv),
                "accession_column": "accession",
                "split_column": "split",
                "validation_values": ["validation"],
            },
        },
        "train_test": {
            "strategy": "random",
            "random": {"test_size": 0.25, "random_state": 7},
        },
        "zero_shot": {
            "strategy": "csv",
            "csv": {
                "csv_path": str(split_csv),
                "accession_column": "accession",
                "split_column": "split",
                "validation_values": ["validation", "zs"],
            },
        },
    }

    manager = IndependentValidationTrainTestSplit(conf)
    with pytest.raises(ValueError, match="Duplicate accession assignments"):
        manager.split(accessions, labels, metadata)


def test_cross_validation_excludes_zero_shot_ids(tmp_path):
    accessions = [f"A{i}" for i in range(1, 11)]
    labels = _labels_for(accessions)
    metadata = _metadata_for(accessions)

    split_csv = tmp_path / "split.csv"
    _write_split_csv(
        split_csv,
        [
            {"accession": "A1", "split": "validation"},
            {"accession": "A9", "split": "zero_shot"},
            {"accession": "A10", "split": "zero_shot"},
        ],
    )

    conf = {
        "validation": {
            "strategy": "csv",
            "csv": {
                "csv_path": str(split_csv),
                "accession_column": "accession",
                "split_column": "split",
                "validation_values": ["validation"],
            },
        },
        "train_test": {
            "strategy": "cross_validation",
            "cross_validation": {"n_splits": 3, "fold_index": 0, "random_state": 13},
        },
        "zero_shot": {
            "strategy": "csv",
            "csv": {
                "csv_path": str(split_csv),
                "accession_column": "accession",
                "split_column": "split",
                "validation_values": ["zero_shot"],
            },
        },
    }

    manager = IndependentValidationTrainTestSplit(conf)
    train_ids, val_ids, test_ids = manager.split(accessions, labels, metadata)
    zero_ids = set(manager.get_zero_shot())

    assert zero_ids == {"A9", "A10"}
    assert zero_ids.isdisjoint(set(train_ids))
    assert zero_ids.isdisjoint(set(val_ids))
    assert zero_ids.isdisjoint(set(test_ids))
