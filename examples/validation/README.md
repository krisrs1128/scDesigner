# Notebook Validation Script

This script is used to validate Jupyter notebooks located in the `examples` directory. It executes each notebook and reports any errors encountered during execution, providing truncated error messages, execution statistics, and the location of the error.

## Quick Start

To use the script, navigate to the `validation` directory and run the script using Python:

```bash
cd examples/validation && python validate_notebooks.py
```

It will automatically validate all notebooks in the `examples` directory, print the error messages and save the log to the `logs` directory under the `validation` directory.

## Options
- `--directory DIR`: Specify the directory to search for notebooks (default: `examples`).
- `--timeout SECONDS`: Set the execution timeout per cell (default: 300 seconds).
- `--verbose`: Enable verbose output for more detailed execution logs.
- `--no-color`: Disable colored output in the terminal.
- `--stop-on-error`: Stop execution when the first error is encountered.
- `--log-file FILE`: Specify the output log file path (default: auto-generated in `logs/`).
- `--log-format FORMAT`: Set the log file format, either `text` or `json` (default: `text`).
- `--no-log`: Disable log file output.

## Examples

- Basic validation:
  ```bash
  python validate_notebooks.py
  ```

- Verbose output:
  ```bash
  python validate_notebooks.py --verbose
  ```

- Set a 10-minute timeout:
  ```bash
  python validate_notebooks.py --timeout 600
  ```

- Disable colors in output:
  ```bash
  python validate_notebooks.py --no-color
  ```

- Save log to a designated file:
  ```bash
  python validate_notebooks.py --log-file report.log
  ```

## Logging

The script can generate logs in either text or JSON format, providing a detailed report of the validation process, including the number of notebooks validated, success rate, and any errors encountered.

## Exit Codes

- `0`: All notebooks executed successfully.
- `1`: One or more notebooks failed validation.
- `130`: Validation was interrupted by the user.

## Additional Information

The script uses the `nbconvert` library to execute notebooks and handle errors. It provides a summary of the validation process, including the total execution time and a list of any failed notebooks. 