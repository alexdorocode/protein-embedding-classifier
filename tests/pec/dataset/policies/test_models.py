"""
Tests for policy models.
"""

import pytest
from pec.dataset.policies.models import (
    DatasetPolicy,
    SelectionStrategy,
    RatioPolicy,
    CandidatePoolPolicy,
    RandomizationConfig,
    OrganismPolicy,
    DuplicatePolicy,
)


class TestSelectionStrategy:
    """Tests for SelectionStrategy."""
    
    def test_default_values(self):
        """Test default values."""
        strategy = SelectionStrategy()
        assert strategy.mode == "sample_without_replacement"
        assert strategy.candidate_scope == "per_target"
        assert strategy.assignment_strategy == "global_unique_candidates"
    
    def test_custom_values(self):
        """Test custom values."""
        strategy = SelectionStrategy(
            mode="sample_with_replacement",
            candidate_scope="global",
            assignment_strategy="per_target_unique",
        )
        assert strategy.mode == "sample_with_replacement"
        assert strategy.candidate_scope == "global"
        assert strategy.assignment_strategy == "per_target_unique"
    
    def test_to_dict(self):
        """Test to_dict."""
        strategy = SelectionStrategy(
            mode="use_all",
            candidate_scope="per_target",
            assignment_strategy="global_unique_candidates",
        )
        d = strategy.to_dict()
        assert d["mode"] == "use_all"
        assert d["candidate_scope"] == "per_target"
    
    def test_from_dict(self):
        """Test from_dict."""
        data = {
            "mode": "sample_without_replacement",
            "candidate_scope": "global",
            "assignment_strategy": "per_target_unique",
        }
        strategy = SelectionStrategy.from_dict(data)
        assert strategy.mode == "sample_without_replacement"
        assert strategy.candidate_scope == "global"


class TestRatioPolicy:
    """Tests for RatioPolicy."""
    
    def test_default_values(self):
        """Test default values."""
        policy = RatioPolicy()
        assert policy.positive_unit == "target"
        assert policy.negative_unit == "candidate_assignment"
        assert policy.target_to_negative_ratio == "1:1"
    
    def test_custom_ratio(self):
        """Test custom ratio."""
        policy = RatioPolicy(target_to_negative_ratio="1:3")
        assert policy.target_to_negative_ratio == "1:3"
    
    def test_get_ratio_tuple(self):
        """Test get_ratio_tuple."""
        policy = RatioPolicy(target_to_negative_ratio="1:5")
        pos, neg = policy.get_ratio_tuple()
        assert pos == 1
        assert neg == 5
    
    def test_get_ratio_tuple_invalid(self):
        """Test get_ratio_tuple with invalid format."""
        policy = RatioPolicy(target_to_negative_ratio="invalid")
        with pytest.raises(ValueError, match="Invalid ratio format"):
            policy.get_ratio_tuple()


class TestCandidatePoolPolicy:
    """Tests for CandidatePoolPolicy."""
    
    def test_default_values(self):
        """Test default values."""
        policy = CandidatePoolPolicy()
        assert policy.min_pool_size == 5
        assert policy.max_pool_size is None
        assert policy.scarcity_mode == "drop_target"
    
    def test_custom_values(self):
        """Test custom values."""
        policy = CandidatePoolPolicy(
            min_pool_size=10,
            max_pool_size=100,
            scarcity_mode="relax_ratio",
        )
        assert policy.min_pool_size == 10
        assert policy.max_pool_size == 100
        assert policy.scarcity_mode == "relax_ratio"
    
    def test_to_dict_excludes_none(self):
        """Test that to_dict excludes None values."""
        policy = CandidatePoolPolicy(
            min_pool_size=5,
            max_pool_size=None,
            scarcity_mode="drop_target",
        )
        d = policy.to_dict()
        assert "min_pool_size" in d
        assert "max_pool_size" not in d
        assert "scarcity_mode" in d


class TestRandomizationConfig:
    """Tests for RandomizationConfig."""
    
    def test_default_values(self):
        """Test default values."""
        config = RandomizationConfig()
        assert config.enabled is True
        assert config.seed_scope == "global"
    
    def test_custom_values(self):
        """Test custom values."""
        config = RandomizationConfig(
            enabled=False,
            seed_scope="per_target",
        )
        assert config.enabled is False
        assert config.seed_scope == "per_target"


class TestOrganismPolicy:
    """Tests for OrganismPolicy."""
    
    def test_default_values(self):
        """Test default values."""
        policy = OrganismPolicy()
        assert policy.mode == "preserve_source"


class TestDuplicatePolicy:
    """Tests for DuplicatePolicy."""
    
    def test_default_values(self):
        """Test default values."""
        policy = DuplicatePolicy()
        assert policy.allow_same_candidate_across_targets is False
        assert policy.allow_same_target_across_variants is True


class TestDatasetPolicy:
    """Tests for DatasetPolicy."""
    
    def test_basic_creation(self):
        """Test basic DatasetPolicy creation."""
        policy = DatasetPolicy(
            policy_id="test_policy",
            source_universe_id="test_universe",
        )
        assert policy.policy_id == "test_policy"
        assert policy.source_universe_id == "test_universe"
    
    def test_to_dict(self):
        """Test to_dict."""
        policy = DatasetPolicy(
            policy_id="test_policy",
            source_universe_id="test_universe",
        )
        d = policy.to_dict()
        assert d["policy_id"] == "test_policy"
        assert d["source_universe_id"] == "test_universe"
        assert "selection_strategy" in d
        assert "ratio_policy" in d
    
    def test_from_dict(self):
        """Test from_dict."""
        data = {
            "policy_id": "test_policy",
            "source_universe_id": "test_universe",
            "selection_strategy": {
                "mode": "sample_without_replacement",
            },
            "ratio_policy": {
                "target_to_negative_ratio": "1:3",
            },
        }
        policy = DatasetPolicy.from_dict(data)
        assert policy.policy_id == "test_policy"
        assert policy.ratio_policy.target_to_negative_ratio == "1:3"
    
    def test_to_json(self):
        """Test JSON serialization."""
        policy = DatasetPolicy(
            policy_id="test_policy",
            source_universe_id="test_universe",
        )
        json_str = policy.to_json()
        assert "test_policy" in json_str
        assert "test_universe" in json_str
    
    def test_from_json(self):
        """Test JSON deserialization."""
        json_str = '{"policy_id": "test_policy", "source_universe_id": "test_universe"}'
        policy = DatasetPolicy.from_json(json_str)
        assert policy.policy_id == "test_policy"
    
    def test_get_variant_count(self):
        """Test get_variant_count."""
        policy = DatasetPolicy(
            policy_id="test_policy",
            source_universe_id="test_universe",
        )
        assert policy.get_variant_count() == 25
    
    def test_get_ratio_families(self):
        """Test get_ratio_families."""
        policy = DatasetPolicy(
            policy_id="test_policy",
            source_universe_id="test_universe",
        )
        families = policy.get_ratio_families()
        assert "1:1" in families
        assert "1:3" in families
        assert "1:5" in families
