#!/usr/bin/env python3
"""
Script to update imports from old paths to new modular paths.

This script helps migrate imports from:
- protein_embedding_classifier/* -> src/training/*
- pec/* -> src/dataset_builder/*

Usage:
    python scripts/update_imports.py --dry-run
    python scripts/update_imports.py --apply
"""

import argparse
import re
from pathlib import Path
from typing import Dict, List, Tuple
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


# Mapping of old import paths to new paths
IMPORT_MAPPING = {
    # protein_embedding_classifier -> src/training
    r'from protein_embedding_classifier\.classifiers': 'from src.training.models',
    r'from protein_embedding_classifier\.classifiers\.': 'from src.training.models.',
    r'from protein_embedding_classifier\.core': 'from src.training',
    r'from protein_embedding_classifier\.core\.': 'from src.training.',
    r'from protein_embedding_classifier\.data': 'from src.dataset_builder',
    r'from protein_embedding_classifier\.data\.': 'from src.dataset_builder.',
    
    # pec -> src/dataset_builder
    r'from pec\.dataset': 'from src.dataset_builder',
    r'from pec\.dataset\.': 'from src.dataset_builder.',
    
    # Specific file mappings
    r'from protein_embedding_classifier\.data\.protein_loader': 'from src.input.protein_loader',
    r'from protein_embedding_classifier\.data\.label_loader': 'from src.dataset_builder.label_loader',
    r'from protein_embedding_classifier\.data\.dataset_builder': 'from src.dataset_builder.builders.dataset_builder',
    r'from protein_embedding_classifier\.core\.embeddings': 'from src.training.embedding_handler',
    r'from protein_embedding_classifier\.core\.decision': 'from src.training.decision',
    r'from protein_embedding_classifier\.core\.statistics': 'from src.training.statistics',
}

# Files to skip (already updated or special cases)
SKIP_FILES = [
    'scripts/update_imports.py',
    'pyproject.toml',
]

# Directories to process
PROCESS_DIRS = ['src/', 'scripts/', 'tests/']


class ImportUpdater:
    """Handles updating imports in Python files."""
    
    def __init__(self, dry_run: bool = True):
        self.dry_run = dry_run
        self.files_updated = 0
        self.imports_updated = 0
        self.errors = []
    
    def update_file(self, file_path: Path) -> bool:
        """Update imports in a single file."""
        if file_path.name in SKIP_FILES:
            return False
        
        try:
            content = file_path.read_text()
            original_content = content
            
            # Apply all import mappings
            for old_pattern, new_pattern in IMPORT_MAPPING.items():
                content, num_replacements = self._replace_pattern(content, old_pattern, new_pattern)
                self.imports_updated += num_replacements
            
            # Only write if content changed
            if content != original_content:
                if not self.dry_run:
                    file_path.write_text(content)
                self.files_updated += 1
                return True
            
            return False
            
        except Exception as e:
            self.errors.append((file_path, str(e)))
            return False
    
    def _replace_pattern(self, content: str, old_pattern: str, new_pattern: str) -> Tuple[str, int]:
        """Replace a pattern in content."""
        # Use regex to match the pattern
        pattern = re.compile(old_pattern)
        new_content, num_replacements = pattern.subn(new_pattern, content)
        return new_content, num_replacements
    
    def update_all(self, base_dir: Path = Path('.')) -> None:
        """Update imports in all Python files."""
        logger.info("Starting import update...")
        logger.info(f"Dry run: {self.dry_run}")
        
        for process_dir in PROCESS_DIRS:
            dir_path = base_dir / process_dir
            if dir_path.exists():
                self._process_directory(dir_path)
        
        # Also process root directory
        self._process_directory(base_dir)
        
        logger.info(f"\nUpdate complete!")
        logger.info(f"Files updated: {self.files_updated}")
        logger.info(f"Imports updated: {self.imports_updated}")
        
        if self.errors:
            logger.warning(f"\nErrors encountered: {len(self.errors)}")
            for file_path, error in self.errors[:5]:  # Show first 5 errors
                logger.warning(f"  {file_path}: {error}")
    
    def _process_directory(self, dir_path: Path) -> None:
        """Process all Python files in a directory."""
        for py_file in dir_path.rglob('*.py'):
            if self.update_file(py_file):
                logger.info(f"Updated: {py_file.relative_to(dir_path)}")


def main():
    parser = argparse.ArgumentParser(
        description='Update imports from old paths to new modular paths'
    )
    parser.add_argument(
        '--apply',
        action='store_true',
        help='Actually apply the changes (default is dry run)'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Show detailed output'
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    dry_run = not args.apply
    updater = ImportUpdater(dry_run=dry_run)
    updater.update_all()
    
    if dry_run:
        logger.info("\nRun with --apply to actually make these changes")
    
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
