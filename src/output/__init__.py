"""
Output Module

This module handles all output operations for the Protein Embedding Classifier.

Responsibilities:
- Saving results in various formats
- Generating reports
- Managing output directories
- Exporting datasets

Author: Protein Embedding Classifier Team
Version: 1.0
Date: 2026-08-03
"""

from .results_manager import ResultsManager
from .writers.csv_writer import CSVWriter
from .writers.json_writer import JSONWriter
from .writers.report_writer import ReportWriter

# Public API
__all__ = [
    'ResultsManager',
    'CSVWriter',
    'JSONWriter',
    'ReportWriter',
    'save_results',
    'generate_report',
]


def save_results(results: any, output_path: str, format: str = 'csv', **kwargs) -> None:
    """
    Save results in the specified format.
    
    Args:
        results: Results to save
        output_path: Path to save results
        format: Output format ('csv', 'json', 'report')
        **kwargs: Additional arguments for the writer
    """
    writers = {
        'csv': CSVWriter,
        'json': JSONWriter,
        'report': ReportWriter,
    }
    
    if format not in writers:
        raise ValueError(f"Unknown output format: {format}. "
                        f"Available: {list(writers.keys())}")
    
    writer_class = writers[format]
    writer = writer_class()
    writer.write(results, output_path, **kwargs)


def generate_report(results: any, template: str = 'default', **kwargs) -> str:
    """
    Generate a report from results.
    
    Args:
        results: Results to include in report
        template: Report template to use
        **kwargs: Additional report arguments
        
    Returns:
        Path to generated report
    """
    report_writer = ReportWriter()
    return report_writer.generate(results, template, **kwargs)
