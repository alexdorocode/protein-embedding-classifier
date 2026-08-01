import numpy as np

from protein_embedding_classifier.core.training.problem_specification import ProblemSpecification


def test_problem_specification_binary_config():
    labels = np.array([0, 1, 0, 1], dtype=object)
    spec = ProblemSpecification.from_labels(labels)

    assert spec.problem_type == "binary"
    assert spec.output_size == 2
    assert spec.loss_name == "BCEWithLogitsLoss"


def test_problem_specification_multilabel_config():
    labels = np.array([
        ["GO:1", "GO:2"],
        ["GO:2"],
        ["GO:3", "GO:1"],
    ], dtype=object)
    spec = ProblemSpecification.from_labels(labels)

    assert spec.problem_type == "multilabel"
    assert spec.output_size == 3
    assert spec.loss_name == "BCEWithLogitsLoss"


def test_problem_specification_multiclass_config():
    labels = np.array(["A", "B", "C", "A"], dtype=object)
    spec = ProblemSpecification.from_labels(labels)

    assert spec.problem_type == "multiclass"
    assert spec.output_size == 3
    assert spec.loss_name == "CrossEntropyLoss"


def test_problem_specification_loss_selection_and_output_size_behavior():
    binary = ProblemSpecification.from_labels(np.array(["yes", "no"], dtype=object))
    multiclass = ProblemSpecification.from_labels(np.array([0, 1, 2, 3], dtype=object))

    assert binary.loss_name == "BCEWithLogitsLoss"
    assert binary.output_size == 2
    assert multiclass.loss_name == "CrossEntropyLoss"
    assert multiclass.output_size == 4


def test_problem_specification_singleton_list_labels_are_not_multilabel():
    labels = np.empty(4, dtype=object)
    labels[0] = [True]
    labels[1] = [False]
    labels[2] = [True]
    labels[3] = [False]

    spec = ProblemSpecification.from_labels(labels)

    assert spec.problem_type == "binary"
    assert spec.output_size == 2
    assert spec.loss_name == "BCEWithLogitsLoss"
