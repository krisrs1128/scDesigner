#!/usr/bin/env python3
"""
Notebook Validation Script
==========================

This script validates all Jupyter notebooks in the examples directory,
checking if they can be executed without errors. It provides detailed
error reporting and execution statistics.

Usage:
    python validate_notebooks.py [options]

Options:
    --timeout SECONDS    Set execution timeout per cell (default: 300)
    --verbose           Enable verbose output
    --no-color          Disable colored output
    --continue-on-error Continue execution even if notebooks fail
    --log-file FILE     Output log file path
    --log-format FORMAT Log format: text or json
"""

import os
import sys
import time
import json
import argparse
import traceback
import re
from pathlib import Path
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
from nbconvert.preprocessors.execute import CellExecutionError

# Color codes for terminal output
class Colors:
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'

@dataclass
class ExecutionResult:
    """Store the result of notebook execution"""
    notebook_path: str
    success: bool
    execution_time: float
    error_cell: int = -1
    error_message: str = ""
    error_traceback: str = ""
    error_type: str = ""  # Add error type field
    total_cells: int = 0

@dataclass
class ValidationReport:
    """Store overall validation results"""
    results: List[ExecutionResult] = field(default_factory=list)
    total_notebooks: int = 0
    successful_notebooks: int = 0
    failed_notebooks: int = 0
    total_execution_time: float = 0.0

class LogManager:
    """Manage logging output to files"""
    
    def __init__(self, log_file=None, log_format='text'):
        self.log_file = log_file
        self.log_format = log_format
        self.enabled = log_file is not None
        
        if self.enabled:
            # Create log directory if it doesn't exist
            Path(self.log_file).parent.mkdir(parents=True, exist_ok=True)
    
    def write_log(self, report: ValidationReport, script_info: dict):
        """Write the validation report to log file"""
        if not self.enabled:
            return
        
        try:
            if self.log_format == 'json':
                self._write_json_log(report, script_info)
            else:
                self._write_text_log(report, script_info)
            
            print(f"\n📝 Detailed log saved to: {self.log_file}")
            
        except Exception as e:
            print(f"⚠️  Warning: Could not write log file: {e}")
    
    def _write_json_log(self, report: ValidationReport, script_info: dict):
        """Write JSON format log"""
        log_data = {
            'timestamp': datetime.now().isoformat(),
            'script': 'validate_notebooks.py',
            'script_info': script_info,
            'summary': {
                'total_notebooks': report.total_notebooks,
                'successful_notebooks': report.successful_notebooks,
                'failed_notebooks': report.failed_notebooks,
                'success_rate': (report.successful_notebooks / report.total_notebooks * 100) if report.total_notebooks > 0 else 0,
                'total_execution_time': report.total_execution_time
            },
            'results': []
        }
        
        for result in report.results:
            result_dict = asdict(result)
            result_dict['notebook_name'] = Path(result.notebook_path).name
            log_data['results'].append(result_dict)
        
        with open(self.log_file, 'w', encoding='utf-8') as f:
            json.dump(log_data, f, indent=2, ensure_ascii=False)
    
    def _write_text_log(self, report: ValidationReport, script_info: dict):
        """Write human-readable text log"""
        with open(self.log_file, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("NOTEBOOK VALIDATION REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            # Header information
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Script: validate_notebooks.py\n")
            f.write(f"Timeout: {script_info.get('timeout', 'N/A')} seconds\n")
            f.write(f"Verbose mode: {script_info.get('verbose', False)}\n")
            f.write(f"Directory: {script_info.get('directory', 'examples')}\n")
            f.write("\n")
            
            # Summary
            success_rate = (report.successful_notebooks / report.total_notebooks * 100) if report.total_notebooks > 0 else 0
            f.write("SUMMARY:\n")
            f.write("-" * 40 + "\n")
            f.write(f"Total notebooks: {report.total_notebooks}\n")
            f.write(f"Successful: {report.successful_notebooks} ({success_rate:.1f}%)\n")
            f.write(f"Failed: {report.failed_notebooks}\n")
            f.write(f"Total execution time: {report.total_execution_time:.2f}s\n")
            f.write("\n")
            
            # Detailed results
            f.write("DETAILED RESULTS:\n")
            f.write("-" * 50 + "\n\n")
            
            for result in report.results:
                notebook_name = Path(result.notebook_path).name
                status = "✅ PASS" if result.success else "❌ FAIL"
                
                f.write(f"{status} {notebook_name}\n")
                f.write(f"  Cells: {result.total_cells}\n")
                f.write(f"  Execution time: {result.execution_time:.2f}s\n")
                
                if not result.success:
                    if result.error_cell >= 0:
                        f.write(f"  Error at cell: {result.error_cell + 1}\n")
                    
                    # Write specific error information
                    if result.error_type and result.error_message:
                        f.write(f"  Error type: {result.error_type}\n")
                        f.write(f"  Error message: {result.error_message}\n")
                    elif result.error_message:
                        error_lines = result.error_message.split('\n')
                        if len(error_lines) > 10:
                            f.write(f"  Error message (truncated):")
                            for line in error_lines[:10]:
                                f.write(f"    {line}\n")
                            f.write(f"    ... (truncated)\n")
                        else:
                            f.write(f"  Error message:")
                            for line in error_lines:
                                f.write(f"    {line}\n")
                        
                    # If not verbose, only show the first 10 lines of the error traceback
                    if not script_info.get('verbose', False) and result.error_traceback:
                        f.write(f"  Error traceback:\n")
                        traceback_lines = result.error_traceback.split('\n')
                        # Ensure we do not exceed the available lines
                        for line in traceback_lines[:min(10, len(traceback_lines))]:  # Only show the first 10 lines
                            if line.strip():
                                f.write(f"    {line}\n")
                        if len(traceback_lines) > 10:
                            f.write(f"    ... (truncated)\n")
                    
                    # If verbose, show the full error traceback
                    if script_info.get('verbose', False) and result.error_traceback:
                        f.write(f"  Full Error traceback:\n")
                        traceback_lines = result.error_traceback.split('\n')
                        for line in traceback_lines:
                            if line.strip():
                                f.write(f"    {line}\n")
                
                f.write("\n")
            
            # Failed notebooks summary
            if report.failed_notebooks > 0:
                f.write("FAILED NOTEBOOKS SUMMARY:\n")
                f.write("-" * 40 + "\n")
                for result in report.results:
                    if not result.success:
                        notebook_name = Path(result.notebook_path).name
                        f.write(f"• {notebook_name}")
                        if result.error_cell >= 0:
                            f.write(f" (cell {result.error_cell + 1})")
                        if result.error_type:
                            f.write(f" - {result.error_type}")
                        f.write("\n")

def generate_log_filename():
    """Generate a default log filename with timestamp"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"logs/validation_report_{timestamp}.log"

class NotebookValidator:
    """Main class for validating Jupyter notebooks"""
    
    def __init__(self, timeout: int = 300, verbose: bool = False, 
                 use_color: bool = True, continue_on_error: bool = True,
                 log_manager: LogManager = None):
        self.timeout = timeout
        self.verbose = verbose
        self.use_color = use_color
        self.continue_on_error = continue_on_error
        self.log_manager = log_manager
        
        # Disable colors if requested or if not in terminal
        if not use_color or not sys.stdout.isatty():
            for attr in dir(Colors):
                if not attr.startswith('_'):
                    setattr(Colors, attr, '')
    
    def colorize(self, text: str, color: str) -> str:
        """Apply color to text if colors are enabled"""
        return f"{color}{text}{Colors.END}"
    
    def print_header(self):
        """Print script header"""
        header = """
╔══════════════════════════════════════════════════════════════╗
║                    Notebook Validation Tool                  ║
║              Checking examples/ directory notebooks          ║
╚══════════════════════════════════════════════════════════════╝
        """
        print(self.colorize(header, Colors.CYAN + Colors.BOLD))
    
    def find_notebooks(self, directory: str = "examples") -> List[str]:
        """Find all .ipynb files in the specified directory (non-recursive)"""
        examples_dir = Path(directory)
        
        if not examples_dir.exists():
            raise FileNotFoundError(f"Directory '{directory}' not found")
        
        # Find all .ipynb files in the directory (non-recursive)
        notebooks = list(examples_dir.glob("*.ipynb"))
        
        # Filter out checkpoint files
        notebooks = [nb for nb in notebooks if ".ipynb_checkpoints" not in str(nb)]
        
        return [str(nb) for nb in sorted(notebooks)]
    
    def extract_error_info(self, error: Exception, traceback_str: str = None) -> Tuple[str, str, str]:
        """Extract detailed error information from exception
        
        Returns:
            Tuple of (error_type, error_message, formatted_traceback)
        """
        error_type = type(error).__name__
        error_message = str(error)
        
        # If it's a CellExecutionError, try to extract the real underlying error
        if isinstance(error, CellExecutionError):
            if hasattr(error, 'traceback') and error.traceback:
                traceback_str = error.traceback
            
            # Try to extract the real exception from the traceback
            if traceback_str:
                # Look for the last exception in the traceback
                lines = traceback_str.split('\n')
                
                # Find the actual error line (usually the last non-empty line)
                for line in reversed(lines):
                    line = line.strip()
                    if line and ':' in line:
                        # Common pattern: "ErrorType: Error message"
                        if any(error_name in line for error_name in [
                            'Error:', 'Exception:', 'ModuleNotFoundError:', 'ImportError:', 
                            'NameError:', 'AttributeError:', 'KeyError:', 'ValueError:', 
                            'TypeError:', 'FileNotFoundError:', 'IndexError:'
                        ]):
                            # Split on first colon to separate error type and message
                            if ':' in line:
                                potential_error_type = line.split(':')[0].strip()
                                potential_message = ':'.join(line.split(':')[1:]).strip()
                                
                                # If it looks like an exception name, use it
                                if potential_error_type.endswith('Error') or potential_error_type.endswith('Exception'):
                                    error_type = potential_error_type
                                    if potential_message:
                                        error_message = potential_message
                            break
                
                # Also try to find import-related errors specifically
                for line in lines:
                    if 'ModuleNotFoundError' in line or 'ImportError' in line:
                        if 'No module named' in line:
                            # Extract module name
                            match = re.search(r"No module named '([^']+)'", line)
                            if match:
                                module_name = match.group(1)
                                error_type = "ModuleNotFoundError"
                                error_message = f"No module named '{module_name}'"
                                break
        
        # Clean up error message - remove common prefixes
        if error_message.startswith("An error occurred while executing the following cell:"):
            # Extract just the essential error info
            lines = error_message.split('\n')
            for line in lines:
                if any(err in line for err in ['Error:', 'Exception:', 'ModuleNotFoundError:', 'ImportError:']):
                    error_message = line.strip()
                    break
        
        return error_type, error_message, traceback_str or ""
    
    def execute_notebook(self, notebook_path: str) -> ExecutionResult:
        """Execute a single notebook and return the result"""
        start_time = time.time()
        result = ExecutionResult(
            notebook_path=notebook_path,
            success=False,
            execution_time=0.0
        )
        
        try:
            # Read the notebook
            with open(notebook_path, 'r', encoding='utf-8') as f:
                nb = nbformat.read(f, as_version=4)
            
            result.total_cells = len(nb.cells)
            
            # Create executor
            executor = ExecutePreprocessor(
                timeout=self.timeout,
                kernel_name='python3',
                allow_errors=False  # We want to catch errors ourselves
            )
            
            if self.verbose:
                print(f"  Executing {result.total_cells} cells...")
            
            # Execute the notebook
            executor.preprocess(nb, {'metadata': {'path': os.path.dirname(notebook_path)}})
            
            result.success = True
            
        except CellExecutionError as e:
            result.error_cell = e.cell_index if hasattr(e, 'cell_index') else -1
            
            # Extract detailed error information
            error_type, error_message, error_traceback = self.extract_error_info(e)
            result.error_type = error_type
            result.error_message = error_message
            result.error_traceback = error_traceback
            
        except Exception as e:
            error_type, error_message, error_traceback = self.extract_error_info(e, traceback.format_exc())
            result.error_type = error_type
            result.error_message = f"Unexpected error: {error_message}"
            result.error_traceback = error_traceback
        
        finally:
            result.execution_time = time.time() - start_time
        
        return result
    
    def print_progress(self, current: int, total: int, notebook_name: str):
        """Print execution progress"""
        percentage = (current / total) * 100
        progress_bar = "█" * int(percentage // 5) + "░" * (20 - int(percentage // 5))
        
        status_line = f"[{current:2d}/{total:2d}] {progress_bar} {percentage:5.1f}% | {notebook_name}"
        print(f"\r{status_line}", end="", flush=True)
        if current == total:
            print()  # New line after completion
    
    def print_result(self, result: ExecutionResult, index: int, total: int):
        """Print the result of a single notebook execution"""
        notebook_name = Path(result.notebook_path).name
        
        if result.success:
            status = self.colorize("✓ PASS", Colors.GREEN + Colors.BOLD)
            time_str = self.colorize(f"{result.execution_time:.2f}s", Colors.BLUE)
            print(f"{status} {notebook_name} ({result.total_cells} cells, {time_str})")
        else:
            status = self.colorize("✗ FAIL", Colors.RED + Colors.BOLD)
            time_str = self.colorize(f"{result.execution_time:.2f}s", Colors.BLUE)
            error_info = ""
            if result.error_cell >= 0:
                error_info = f" at cell {result.error_cell + 1}"
            
            print(f"{status} {notebook_name} ({result.total_cells} cells, {time_str}){error_info}")
            
            # Show specific error type and message
            
            if result.error_type and result.error_message:
                error_type_colored = self.colorize(result.error_type, Colors.RED + Colors.BOLD)
                print(f"      {error_type_colored}: {result.error_message}")
            elif result.error_message:
                # Fallback to just the error message
                error_lines = result.error_message.split('\n')
                if len(error_lines) > 10:
                    print("  Error message (truncated):")
                    for line in error_lines[:10]:
                        print(f"    {line}")
                    print("    ... (truncated)")
                else:
                    print("  Error message:")
                    for line in error_lines:
                        print(f"    {line}")
    
    def print_summary(self, report: ValidationReport):
        """Print final validation summary"""
        print("\n" + "="*70)
        print(self.colorize("VALIDATION SUMMARY", Colors.BOLD + Colors.UNDERLINE))
        print("="*70)
        
        # Overall statistics
        success_rate = (report.successful_notebooks / report.total_notebooks * 100) if report.total_notebooks > 0 else 0
        
        print(f"Total notebooks: {report.total_notebooks}")
        print(f"Successful: {self.colorize(str(report.successful_notebooks), Colors.GREEN)} "
              f"({success_rate:.1f}%)")
        print(f"Failed: {self.colorize(str(report.failed_notebooks), Colors.RED)}")
        print(f"Total execution time: {self.colorize(f'{report.total_execution_time:.2f}s', Colors.BLUE)}")
        
        if report.failed_notebooks > 0:
            print(f"\n{self.colorize('FAILED NOTEBOOKS:', Colors.RED + Colors.BOLD)}")
            for result in report.results:
                if not result.success:
                    notebook_name = Path(result.notebook_path).name
                    print(f"  • {notebook_name}")
                    if result.error_cell >= 0:
                        error_info = f"Cell {result.error_cell + 1}: "
                        if result.error_type:
                            error_info += f"{result.error_type} - {result.error_message[:60]}..."
                        else:
                            error_info += f"{result.error_message[:80]}..."
                        print(f"    {error_info}")
        
        print("\n" + "="*70)
        
        # Exit status
        if report.failed_notebooks == 0:
            print(self.colorize("🎉 All notebooks executed successfully!", Colors.GREEN + Colors.BOLD))
            return 0
        else:
            print(self.colorize(f"❌ {report.failed_notebooks} notebook(s) failed validation", Colors.RED + Colors.BOLD))
            return 1
    
    def validate_all(self, directory: str = "examples") -> Tuple[int, ValidationReport]:
        """Main validation function"""
        self.print_header()
        
        try:
            # Find notebooks
            notebooks = self.find_notebooks(directory)
            
            if not notebooks:
                print(self.colorize(f"No notebooks found in '{directory}' directory", Colors.YELLOW))
                return 0, ValidationReport()
            
            print(f"Found {len(notebooks)} notebook(s) to validate:")
            if self.log_manager and self.log_manager.enabled:
                print(f"📝 Logging to: {self.log_manager.log_file}")
            print()
            
            # Initialize report
            report = ValidationReport()
            report.total_notebooks = len(notebooks)
            
            # Execute each notebook
            for i, notebook_path in enumerate(notebooks, 1):
                notebook_name = Path(notebook_path).name
                
                print(f"{self.colorize('Running:', Colors.YELLOW)} {notebook_name}")
                
                # Execute notebook
                result = self.execute_notebook(notebook_path)
                report.results.append(result)
                report.total_execution_time += result.execution_time
                
                if result.success:
                    report.successful_notebooks += 1
                else:
                    report.failed_notebooks += 1
                
                # Print result
                self.print_result(result, i, len(notebooks))
                
                # Check if we should continue
                if not result.success and not self.continue_on_error:
                    print(f"\n{self.colorize('Stopping execution due to failure', Colors.RED)}")
                    break
                
                print()  # Add spacing between notebooks
            
            # Print summary
            exit_code = self.print_summary(report)
            return exit_code, report
            
        except Exception as e:
            print(f"{self.colorize('Fatal error:', Colors.RED)} {str(e)}")
            if self.verbose:
                traceback.print_exc()
            return 1, ValidationReport()

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Validate Jupyter notebooks in the examples directory",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python validate_notebooks.py                    # Basic validation
  python validate_notebooks.py --verbose          # Verbose output
  python validate_notebooks.py --timeout 600      # 10-minute timeout
  python validate_notebooks.py --no-color         # Disable colors
  python validate_notebooks.py --log-file report.log  # Save log to a designated file
        """
    )
    
    parser.add_argument(
        '--timeout', 
        type=int, 
        default=300,
        help='Execution timeout per cell in seconds (default: 300)'
    )
    
    parser.add_argument(
        '--verbose', 
        action='store_true',
        help='Enable verbose output'
    )
    
    parser.add_argument(
        '--no-color', 
        action='store_true',
        help='Disable colored output'
    )
    
    parser.add_argument(
        '--stop-on-error', 
        action='store_true',
        help='Stop execution when first error is encountered'
    )
    
    parser.add_argument(
        '--directory', 
        default='..',
        help='Directory to search for notebooks (default: examples)'
    )
    
    parser.add_argument(
        '--log-file', '-l',
        type=str,
        help='Output log file path (default: auto-generated in logs/)'
    )
    
    parser.add_argument(
        '--log-format',
        choices=['text', 'json'],
        default='text',
        help='Log file format (default: text)'
    )
    
    parser.add_argument(
        '--no-log',
        action='store_true',
        help='Disable log file output'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    log_manager = None
    if not args.no_log:
        log_file = args.log_file if args.log_file else generate_log_filename()
        log_manager = LogManager(log_file, args.log_format)
    
    # Create validator
    validator = NotebookValidator(
        timeout=args.timeout,
        verbose=args.verbose,
        use_color=not args.no_color,
        continue_on_error=not args.stop_on_error,
        log_manager=log_manager
    )
    
    # Run validation
    try:
        exit_code, report = validator.validate_all(args.directory)
        
        # Write log if enabled
        if log_manager:
            script_info = {
                'timeout': args.timeout,
                'verbose': args.verbose,
                'directory': args.directory,
                'stop_on_error': args.stop_on_error
            }
            log_manager.write_log(report, script_info)
        
        sys.exit(exit_code)
        
    except KeyboardInterrupt:
        print(f"\n{validator.colorize('Validation interrupted by user', Colors.YELLOW)}")
        sys.exit(130)
    except Exception as e:
        print(f"{validator.colorize('Unexpected error:', Colors.RED)} {str(e)}")
        if args.verbose:
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main() 