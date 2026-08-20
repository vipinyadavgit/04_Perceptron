# Perceptron From Scratch

> A small, beginner-friendly implementation of a single-layer perceptron in Python and NumPy.

This project teaches the basic mechanics of supervised learning by training a perceptron to recognize the **AND logical gate**. It does not use Scikit-learn, PyTorch, or another machine-learning framework.

## What This Project Demonstrates

- How a perceptron calculates a weighted sum
- How a step activation function produces a prediction
- How prediction errors update weights and bias
- How training progresses over multiple epochs
- How to run a Python project with [UV](https://docs.astral.sh/uv/)

## Learning Target: AND Gate

The model learns that the output is `1` only when both inputs are `1`:

| Input 1 | Input 2 | Expected output |
|:-------:|:-------:|:---------------:|
| `0` | `0` | `0` |
| `0` | `1` | `0` |
| `1` | `0` | `0` |
| `1` | `1` | `1` |

## How the Perceptron Works

For each input, the model calculates:

```text
weighted_sum = (input_1 * weight_1) + (input_2 * weight_2) + bias
```

It then applies a step function:

```text
prediction = 1, when weighted_sum >= 0
prediction = 0, otherwise
```

When the prediction is incorrect, the model updates its parameters using:

```text
error = expected_output - prediction
weights = weights + learning_rate * error * inputs
bias = bias + learning_rate * error
```

Training stops early when an entire epoch completes without an incorrect prediction.

## Prerequisites

- Windows 10 or later
- Python 3.14 or later
- UV installed and available in your PATH
- A Command Prompt, PowerShell terminal, or VS Code terminal

## Setup with UV in Command Prompt

Open **Command Prompt** and move into the project directory:

```cmd
cd /d "D:\Projects\AI Projects\Perceptron"
```

Check that UV is available:

```cmd
uv --version
```

If `uv` is not recognized, install it from Command Prompt with:

```cmd
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Close and reopen Command Prompt after installation, then verify again:

```cmd
uv --version
```

Create or update the project virtual environment and install the locked dependencies:

```cmd
uv sync
```

## Run the Perceptron

Run the program through UV:

```cmd
uv run main.py
```

UV automatically uses the project's `.venv` environment for this command. You can also run the file directly after syncing:

```cmd
.venv\Scripts\python.exe main.py
```

In VS Code, open `main.py` and select **Run Python File**. Make sure the selected interpreter is:

```text
.venv\Scripts\python.exe
```

## Expected Result

The program prints the training details for each epoch, followed by the final model. A successful run ends with predictions like:

```text
Training is successful

Input: [0 0] Expected: 0 Predicted: 0
Input: [0 1] Expected: 0 Predicted: 0
Input: [1 0] Expected: 0 Predicted: 0
Input: [1 1] Expected: 1 Predicted: 1
```

## Project Structure

```text
.
|-- main.py            # Perceptron implementation and training script
|-- pyproject.toml     # UV and Python project configuration
|-- uv.lock            # Locked project dependency information
|-- .python-version    # Project Python version
|-- requirements.txt   # Optional requirements file
|-- README.md          # Project documentation
`-- .venv\             # Local virtual environment created by UV
```

## Configuration

The main training settings are defined near the top of `main.py`:

```python
learning_rate = 0.1
epochs = 10
```

Try changing these values and observe how the number of epochs and parameter updates change.

## Troubleshooting

### `uv` is not recognized

UV is either not installed or its installation directory is missing from PATH. Install UV using the command above, reopen Command Prompt, and run `uv --version`.

### `ModuleNotFoundError: No module named 'numpy'`

Sync the project dependencies and run the script through UV:

```cmd
uv sync
uv run main.py
```

### VS Code uses the wrong Python interpreter

Use `Ctrl+Shift+P`, select **Python: Select Interpreter**, and choose the interpreter inside the project `.venv` folder.

## Next Steps

- Add OR and NOT gate examples
- Refactor the training loop into reusable functions
- Add automated tests for predictions
- Experiment with different learning rates and initial weights