"""
Tests for UniverseRecord and related models.
"""

import pytest
from pec.dataset.input.models import UniverseRecord, PoolConstraints, PoolMetadata, UniverseManifest


class TestPoolConstraints:
    """Tests for PoolConstraints."""
    
    def test_default_values(self):
        """Test default values."""
        constraints = PoolConstraints()
        assert constraints.len_variance is None
        assert constraints.max_sequence_identity is None
        assert constraints.min_candidates is None
    
    def test_with_values(self):
        """Test with explicit values."""
        constraints = PoolConstraints(
            len_variance=0.5,
            max_sequence_identity=0.9,
            min_candidates=5,
        )
        assert constraints.len_variance == 0.5
        assert constraints.max_sequence_identity == 0.9
        assert constraints.min_candidates == 5
    
    def test_to_dict_excludes_none(self):
        """Test that to_dict excludes None values."""
        constraints = PoolConstraints(
            len_variance=0.5,
            max_sequence_identity=None,
            min_candidates=5,
        )
        d = constraints.to_dict()
        assert "len_variance" in d
        assert "max_sequence_identity" not in d
        assert "min_candidates" in d
    
    def test_from_dict(self):
        """Test from_dict."""
        data = {
            "len_variance": 0.5,
            "max_sequence_identity": 0.9,
            "min_candidates": 5,
        }
        constraints = PoolConstraints.from_dict(data)
        assert constraints.len_variance == 0.5
        assert constraints.max_sequence_identity == 0.9
        assert constraints.min_candidates == 5


class TestPoolMetadata:
    """Tests for PoolMetadata."""
    
    def test_default_source(self):
        """Test default generation source."""
        metadata = PoolMetadata(generation_source="test")
        assert metadata.generation_source == "test"
        assert isinstance(metadata.constraints_snapshot, PoolConstraints)
    
    def test_to_dict(self):
        """Test to_dict."""
        constraints = PoolConstraints(len_variance=0.5)
        metadata = PoolMetadata(
            generation_source="test",
            constraints_snapshot=constraints,
        )
        d = metadata.to_dict()
        assert d["generation_source"] == "test"
        assert "constraints_snapshot" in d


class TestUniverseRecord:
    """Tests for UniverseRecord."""
    
    def test_basic_creation(self):
        """Test basic UniverseRecord creation."""
        record = UniverseRecord(
            target_id="target1",
            target_label="positive",
            candidate_ids=["cand1", "cand2", "cand3"],
            source_file="test.csv",
            source_row_id=1,
        )
        assert record.target_id == "target1"
        assert record.target_label == "positive"
        assert record.candidate_ids == ["cand1", "cand2", "cand3"]
        assert record.candidate_count == 3
        assert record.source_file == "test.csv"
        assert record.source_row_id == 1
    
    def test_candidate_count_auto(self):
        """Test that candidate_count is automatically calculated."""
        record = UniverseRecord(
            target_id="target1",
            target_label="positive",
            candidate_ids=["cand1", "cand2", "cand3", "cand4"],
            source_file="test.csv",
            source_row_id=1,
        )
        assert record.candidate_count == 4
    
    def test_with_organism(self):
        """Test with organism metadata."""
        record = UniverseRecord(
            target_id="target1",
            target_label="positive",
            candidate_ids=["cand1"],
            source_file="test.csv",
            source_row_id=1,
            organism="Homo sapiens",
            taxonomy_id="9606",
        )
        assert record.organism == "Homo sapiens"
        assert record.taxonomy_id == "9606"
    
    def test_empty_target_id_raises(self):
        """Test that empty target_id raises ValueError."""
        with pytest.raises(ValueError, match="target_id cannot be empty"):
            UniverseRecord(
                target_id="",
                target_label="positive",
                candidate_ids=["cand1"],
                source_file="test.csv",
                source_row_id=1,
            )
    
    def test_empty_candidate_ids_raises(self):
        """Test that empty candidate_ids raises ValueError."""
        with pytest.raises(ValueError, match="target_id target1 has no candidates"):
            UniverseRecord(
                target_id="target1",
                target_label="positive",
                candidate_ids=[],
                source_file="test.csv",
                source_row_id=1,
            )
    
    def test_non_list_candidate_ids_raises(self):
        """Test that non-list candidate_ids raises ValueError."""
        with pytest.raises(ValueError, match="candidate_ids must be a list"):
            UniverseRecord(
                target_id="target1",
                target_label="positive",
                candidate_ids="cand1,cand2",  # String, not list
                source_file="test.csv",
                source_row_id=1,
            )
    
    def test_to_dict(self):
        """Test to_dict."""
        record = UniverseRecord(
            target_id="target1",
            target_label="positive",
            candidate_ids=["cand1", "cand2"],
            source_file="test.csv",
            source_row_id=1,
            organism="Homo sapiens",
        )
        d = record.to_dict()
        assert d["target_id"] == "target1"
        assert d["target_label"] == "positive"
        assert d["candidate_ids"] == ["cand1", "cand2"]
        assert d["candidate_count"] == 2
        assert d["organism"] == "Homo sapiens"
    
    def test_from_dict(self):
        """Test from_dict."""
        data = {
            "target_id": "target1",
            "target_label": "positive",
            "candidate_ids": ["cand1", "cand2"],
            "source_file": "test.csv",
            "source_row_id": "1",
        }
        record = UniverseRecord.from_dict(data)
        assert record.target_id == "target1"
        assert record.candidate_count == 2
    
    def test_to_json(self):
        """Test JSON serialization."""
        record = UniverseRecord(
            target_id="target1",
            target_label="positive",
            candidate_ids=["cand1"],
            source_file="test.csv",
            source_row_id=1,
        )
        json_str = record.to_json()
        assert "target1" in json_str
        assert "cand1" in json_str
    
    def test_from_json(self):
        """Test JSON deserialization."""
        json_str = '{"target_id": "target1", "target_label": "positive", "candidate_ids": ["cand1"], "source_file": "test.csv", "source_row_id": 1}'
        record = UniverseRecord.from_json(json_str)
        assert record.target_id == "target1"


class TestUniverseManifest:
    """Tests for UniverseManifest."""
    
    def test_basic_creation(self):
        """Test basic UniverseManifest creation."""
        manifest = UniverseManifest(
            universe_id="universe1",
            source_file="test.csv",
            record_count=10,
            target_count=10,
            total_candidates=100,
            generated_at="2026-01-01T00:00:00Z",
        )
        assert manifest.universe_id == "universe1"
        assert manifest.record_count == 10
        assert manifest.target_count == 10
        assert manifest.total_candidates == 100
    
    def test_to_dict(self):
        """Test to_dict."""
        manifest = UniverseManifest(
            universe_id="universe1",
            source_file="test.csv",
            record_count=10,
            target_count=10,
            total_candidates=100,
            generated_at="2026-01-01T00:00:00Z",
        )
        d = manifest.to_dict()
        assert d["universe_id"] == "universe1"
        assert d["record_count"] == 10
    
    def test_from_dict(self):
        """Test from_dict."""
        data = {
            "universe_id": "universe1",
            "source_file": "test.csv",
            "record_count": 10,
            "target_count": 10,
            "total_candidates": 100,
            "generated_at": "2026-01-01T00:00:00Z",
        }
        manifest = UniverseManifest.from_dict(data)
        assert manifest.universe_id == "universe1"
    
    def test_to_json(self):
        """Test JSON serialization."""
        manifest = UniverseManifest(
            universe_id="universe1",
            source_file="test.csv",
            record_count=10,
            target_count=10,
            total_candidates=100,
            generated_at="2026-01-01T00:00:00Z",
        )
        json_str = manifest.to_json()
        assert "universe1" in json_str
