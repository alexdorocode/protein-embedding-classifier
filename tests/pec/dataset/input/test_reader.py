"""
Tests for UniverseReader.
"""

import pytest
import tempfile
import os
from pathlib import Path
from src.dataset_builder.input.reader import UniverseReader
from src.dataset_builder.input.models import UniverseRecord


class TestUniverseReader:
    """Tests for UniverseReader."""
    
    def test_read_csv_basic(self):
        """Test reading a basic CSV file."""
        reader = UniverseReader()
        
        # Create a temporary CSV file
        csv_content = """target_id,candidate_ids
target1,cand1;cand2;cand3
target2,cand4;cand5"""
        
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write(csv_content)
            temp_path = Path(f.name)
        
        try:
            records = reader.read_csv(temp_path)
            assert len(records) == 2
            assert records[0].target_id == "target1"
            assert records[0].candidate_ids == ["cand1", "cand2", "cand3"]
            assert records[1].target_id == "target2"
            assert records[1].candidate_ids == ["cand4", "cand5"]
        finally:
            os.unlink(temp_path)
    
    def test_read_csv_with_organism(self):
        """Test reading CSV with organism column."""
        reader = UniverseReader()
        
        csv_content = """target_id,candidate_ids,organism,taxonomy_id
target1,cand1;cand2,Homo sapiens,9606
target2,cand3;cand4,Mus musculus,10090"""
        
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write(csv_content)
            temp_path = Path(f.name)
        
        try:
            records = reader.read_csv(temp_path)
            assert len(records) == 2
            assert records[0].organism == "Homo sapiens"
            assert records[0].taxonomy_id == "9606"
            assert records[1].organism == "Mus musculus"
            assert records[1].taxonomy_id == "10090"
        finally:
            os.unlink(temp_path)
    
    def test_read_csv_with_target_label(self):
        """Test reading CSV with target_label column."""
        reader = UniverseReader()
        
        csv_content = """target_id,candidate_ids,target_label
target1,cand1;cand2,positive
target2,cand3;cand4,negative"""
        
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write(csv_content)
            temp_path = Path(f.name)
        
        try:
            records = reader.read_csv(temp_path)
            assert records[0].target_label == "positive"
            assert records[1].target_label == "negative"
        finally:
            os.unlink(temp_path)
    
    def test_read_csv_custom_separator(self):
        """Test reading CSV with custom candidate separator."""
        reader = UniverseReader()
        
        # Use pipe separator in candidate_ids
        csv_content = """target_id,candidate_ids
target1,cand1|cand2|cand3"""
        
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write(csv_content)
            temp_path = Path(f.name)
        
        try:
            records = reader.read_csv(temp_path, candidate_separator="|")
            assert len(records) == 1
            assert records[0].candidate_ids == ["cand1", "cand2", "cand3"]
        finally:
            os.unlink(temp_path)
    
    def test_read_csv_missing_columns(self):
        """Test that missing columns raise ValueError."""
        reader = UniverseReader()
        
        csv_content = """target_id
target1"""
        
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write(csv_content)
            temp_path = Path(f.name)
        
        try:
            with pytest.raises(ValueError, match="Missing required columns"):
                reader.read_csv(temp_path)
        finally:
            os.unlink(temp_path)
    
    def test_read_csv_file_not_found(self):
        """Test that non-existent file raises FileNotFoundError."""
        reader = UniverseReader()
        with pytest.raises(FileNotFoundError):
            reader.read_csv(Path("/nonexistent/file.csv"))
    
    def test_read_csv_skips_malformed_rows(self):
        """Test that malformed rows are skipped with warning."""
        reader = UniverseReader()
        
        csv_content = """target_id,candidate_ids
target1,cand1;cand2
,invalid
"""
        
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write(csv_content)
            temp_path = Path(f.name)
        
        try:
            records = reader.read_csv(temp_path)
            # Should have 1 valid record (the malformed row is skipped)
            assert len(records) == 1
            assert records[0].target_id == "target1"
        finally:
            os.unlink(temp_path)
    
    def test_read_csv_preserves_raw_payload(self):
        """Test that extra columns are preserved in raw_payload."""
        reader = UniverseReader()
        
        csv_content = """target_id,candidate_ids,extra_col1,extra_col2
target1,cand1;cand2,value1,value2"""
        
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write(csv_content)
            temp_path = Path(f.name)
        
        try:
            records = reader.read_csv(temp_path)
            assert len(records) == 1
            assert records[0].raw_payload is not None
            assert records[0].raw_payload["extra_col1"] == "value1"
            assert records[0].raw_payload["extra_col2"] == "value2"
        finally:
            os.unlink(temp_path)
    
    def test_read_jsonl(self):
        """Test reading JSONL file."""
        reader = UniverseReader()
        
        jsonl_content = '{"target_id": "target1", "target_label": "positive", "candidate_ids": ["cand1", "cand2"], "source_file": "test.jsonl", "source_row_id": 1}\n'
        jsonl_content += '{"target_id": "target2", "target_label": "positive", "candidate_ids": ["cand3"], "source_file": "test.jsonl", "source_row_id": 2}\n'
        
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write(jsonl_content)
            temp_path = Path(f.name)
        
        try:
            records = reader.read_jsonl(temp_path)
            assert len(records) == 2
            assert records[0].target_id == "target1"
            assert records[1].target_id == "target2"
        finally:
            os.unlink(temp_path)
