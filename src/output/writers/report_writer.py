"""
Report Writer Module

Handles generating reports from results.

Author: Protein Embedding Classifier Team
Version: 1.0
Date: 2026-08-03
"""

from pathlib import Path
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class ReportWriter:
    """
    Generates reports from experiment results.
    """
    
    def __init__(self):
        """Initialize the report writer."""
        pass
    
    def generate(self, results: Dict[str, Any], template: str = 'default', 
                 output_path: Optional[str] = None) -> str:
        """
        Generate a report from results.
        
        Args:
            results: Results to include in report
            template: Report template to use
            output_path: Path to save report (optional)
            
        Returns:
            Path to generated report or report string
        """
        if template == 'default':
            report = self._generate_default_report(results)
        elif template == 'detailed':
            report = self._generate_detailed_report(results)
        else:
            raise ValueError(f"Unknown report template: {template}")
        
        if output_path:
            output_path = Path(output_path)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report)
            logger.info(f"Report written to: {output_path}")
            return str(output_path)
        
        return report
    
    def _generate_default_report(self, results: Dict[str, Any]) -> str:
        """Generate a default report."""
        report = []
        report.append("# Protein Embedding Classifier - Experiment Report")
        report.append("")
        report.append("## Summary")
        report.append("")
        
        # Add basic info
        if 'config' in results:
            report.append("### Configuration")
            report.append("")
            for key, value in results['config'].items():
                report.append(f"- **{key}**: {value}")
            report.append("")
        
        # Add metrics
        if 'metrics' in results:
            report.append("### Metrics")
            report.append("")
            for metric, value in results['metrics'].items():
                report.append(f"- **{metric}**: {value:.4f}")
            report.append("")
        
        # Add model info
        if 'model' in results and hasattr(results['model'], '__class__'):
            report.append("### Model")
            report.append("")
            report.append(f"- **Type**: {results['model'].__class__.__name__}")
            report.append("")
        
        return "\n".join(report)
    
    def _generate_detailed_report(self, results: Dict[str, Any]) -> str:
        """Generate a detailed report."""
        report = self._generate_default_report(results)
        
        # Add additional sections
        if 'predictions' in results:
            report += "\n\n### Predictions Sample\n\n"
            # Show first few predictions
            predictions = results['predictions']
            if hasattr(predictions, 'head'):
                report += predictions.head(10).to_markdown()
            else:
                report += str(predictions[:10])
        
        if 'explanations' in results:
            report += "\n\n### Explanations\n\n"
            report += "Feature importance and saliency maps available."
        
        return report
    
    def save_as_html(self, results: Dict[str, Any], output_path: str, 
                     template: str = 'default') -> str:
        """
        Save report as HTML.
        
        Args:
            results: Results to include
            output_path: Path to save HTML report
            template: Report template
            
        Returns:
            Path to saved HTML report
        """
        try:
            import markdown
            md_report = self.generate(results, template)
            html_report = markdown.markdown(md_report)
            
            output_path = Path(output_path)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(f"""
                <!DOCTYPE html>
                <html>
                <head>
                    <title>PEC Experiment Report</title>
                    <style>
                        body {{ font-family: Arial, sans-serif; margin: 40px; }}
                        h1, h2, h3 {{ color: #333; }}
                        table {{ border-collapse: collapse; width: 100%; }}
                        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                        th {{ background-color: #f2f2f2; }}
                    </style>
                </head>
                <body>
                    {html_report}
                </body>
                </html>
                """)
            
            logger.info(f"HTML report written to: {output_path}")
            return str(output_path)
        except ImportError:
            logger.warning("markdown module not available, saving as text")
            return self.generate(results, template, output_path)
