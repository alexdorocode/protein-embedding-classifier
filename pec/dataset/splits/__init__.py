"""
Splits Module

Responsible for creating split artifacts with leakage guards.

Key Classes:
- SplitStrategy: Abstract base class for split strategies
- GroupByTargetSplitStrategy: Groups by target_id to prevent leakage
- SplitManifest: Canonical split manifest
"""

from src.dataset_builder.splits.models import SplitManifest
from src.dataset_builder.splits.strategies import SplitStrategy, GroupByTargetSplitStrategy

__all__ = ["SplitStrategy", "GroupByTargetSplitStrategy", "SplitManifest"]
