# src/core/experiment.py


""""

Experiment specification and tag generation

Defines ExperimentSpec dataclass to encapsulate experiment parameters and make_tags function to generate descriptive tags
based on the specification.

Responsabilities:
- ExperimentSpec: Holds parameters like task, model, embedding type, layers, aggregation method, and aggregation parameters.
- make_tags: Generates a list of tags that describe the experiment configuration for logging and tracking purposes.

What Not to Include:
- Experiment execution logic
- Database interactions
- Classifier implementations

"""

from dataclasses import dataclass
from typing import List, Optional


@dataclass(frozen=True)
class ExperimentSpec:
    dataset: str
    task: str
    model: str               # ESM, Prot-T5, Ankh3
    embedding_type: str      # esm2_t33_650M, prot_t5_xl, ...
    layers: Optional[List[int]] = None
    aggregation: Optional[str] = None  # mean, soft_attention, hard_attention_top_k
    aggregation_param: Optional[str] = None  # top_5, threshold_050, ...

def make_tags(spec: ExperimentSpec) -> list[str]:
    tags = []

    # model & task
    tags.append(spec.dataset)
    tags.append(spec.model)
    tags.append(spec.task)

    # aggregation
    if spec.aggregation:
        tags.append(spec.aggregation)

    if spec.aggregation_param:
        tags.append(spec.aggregation_param)

    # layers
    if spec.layers:
        sorted_layers = sorted(spec.layers)

        # concatenated layer code: 021424
        layer_code = "".join(f"{l:02d}" for l in sorted_layers)
        tags.append(layer_code)

        for l in sorted_layers:
            tags.append(f"has_layer_{l}")

    return tags
