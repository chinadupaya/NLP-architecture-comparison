
## NLP Architecture Comparison (GRU, LSTM, Transformer)

This project compares GRU, LSTM, and a decoder-only Transformer on the same tokenized Sherlock Holmes dataset, tracking training metrics, resource usage, and grammar quality of generated text.

The primary notebooks are:
- `TokenGRU.ipynb`
- `TokenLSTM.ipynb`
- `TokenTransformer_Step.ipynb`

Each notebook follows the same flow: load data → tokenize → build model → train with checkpoints (and optional resource monitoring/W&B) → generate text → evaluate grammar.

## 1) Prerequisites

- Python 3.10–3.13
- Jupyter
- TensorFlow (CPU is fine)
- Java 17+ (required for local LanguageTool in grammar evaluation)
- Optional: Weights & Biases (W&B) account and API key
- macOS users: Homebrew for installing Java and Graphviz if you want model plots

## 2) Environment setup

Create an isolated virtual environment and install dependencies:
```
python -m venv ~/nlp-tf-env
source ~/nlp-tf-env/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Register a dedicated Jupyter kernel (so nbconvert/notebooks use this env):
```
python -m ipykernel install --user --name nlp-tf220 --display-name "nlp-tf220"
```

Verify the kernel is available in Jupyter and in `nbconvert` runs.

## 3) Data and tokenizer

The repository includes:
- `sherlockholmes.txt` (training text)
- `tokenizer/merges.txt` and `tokenizer/vocab.json` (BPE tokenizer files)

No extra steps are needed; notebooks read these directly from the repo root.

## 4) W&B (optional but recommended)

Set your W&B API key for logging metrics:
```
export WANDB_API_KEY=YOUR_WANDB_KEY
wandb login --relogin $WANDB_API_KEY
```
Optionally set default metadata:
```
export WANDB_ENTITY=your_entity
export WANDB_PROJECT=my-first-project
export WANDB_RUN_GROUP=gru-experiment
```
You can also run offline:
```
export WANDB_MODE=offline
```

## 5) Grammar evaluation (LanguageTool)

LanguageTool requires Java 17+ when using the local server via `language_tool_python`.

macOS (Homebrew):
```
brew install openjdk@17
export JAVA_HOME=$(/usr/libexec/java_home -v 17)
java -version  # should show 17.x
```

The notebooks call `language_tool_python.LanguageTool('en-GB')`. If Java < 17 is active, you will see an error. Ensure `JAVA_HOME` points to JDK 17 before running grammar evaluation.

## 6) Running the experiments

You can run notebooks interactively in Jupyter Lab/Notebook using the `nlp-tf220` kernel, or non-interactively via `nbconvert` to ensure identical runs.

Non-interactive run (recommended for reproducibility):
```
export JAVA_HOME=$(/usr/libexec/java_home -v 17)  # for grammar evaluation
python -m jupyter nbconvert \
  --to notebook --inplace \
  --ExecutePreprocessor.kernel_name=nlp-tf220 \
  --ExecutePreprocessor.timeout=7200 \
  --execute /absolute/path/to/NLP-architecture-comparison/TokenGRU.ipynb

python -m jupyter nbconvert \
  --to notebook --inplace \
  --ExecutePreprocessor.kernel_name=nlp-tf220 \
  --ExecutePreprocessor.timeout=7200 \
  --execute /absolute/path/to/NLP-architecture-comparison/TokenLSTM.ipynb

python -m jupyter nbconvert \
  --to notebook --inplace \
  --ExecutePreprocessor.kernel_name=nlp-tf220 \
  --ExecutePreprocessor.timeout=7200 \
  --execute /absolute/path/to/NLP-architecture-comparison/TokenTransformer_Step.ipynb
```

Notes:
- Checkpoints are saved under `tmp-*/checkpoints/` as `model_epoch_XX.weights.h5`.
- Training epochs (`EPOCHS`) and generation settings are defined in each notebook.

## 7) Resource monitoring

`resource_monitor.py` provides a Keras callback `ResourceMonitorCB` to collect CPU, memory, and (if present) GPU stats. Some notebooks have it enabled/disabled for parity. If plotting raises shape-mismatch errors on macOS, align series lengths in the notebook’s plotting cell (do not modify `resource_monitor.py`).

## 8) Model plots (optional)

Install dependencies:
```
pip install pydot graphviz
brew install graphviz
```
Then in a notebook after the model is built:
```
from tensorflow.keras.utils import plot_model
plot_model(model, to_file='model.png', show_shapes=True, expand_nested=True, dpi=200)
```

## 9) Reproducibility tips

- Keep the same tokenizer files, dataset, and hyperparameters across all runs.
- Use the same virtual environment and Jupyter kernel for all notebooks.
- Pin package versions in `requirements.txt`.
- Avoid modifying GRU/LSTM notebooks if strict comparability is required.

## 10) Troubleshooting

- LanguageTool Java error: Ensure `JAVA_HOME` points to Java 17 (`/usr/libexec/java_home -v 17`).
- Kernel died or wrong environment: Recreate the env and kernel; pick `nlp-tf220` when running.
- Timeout in nbconvert: Increase `--ExecutePreprocessor.timeout` (e.g., 7200 seconds).
- W&B login issues: Re-run `wandb login` or set `WANDB_API_KEY`; use `WANDB_MODE=offline` to avoid network issues.
- Missing checkpoint file in GRU: Some notebooks may reference a specific epoch; ensure the file exists or train to that epoch.

## Installing (legacy quick start)
```
pip install -r requirements.txt
```

## Adding any libraries
When adding new libraries, update the lockfile:
```
pip freeze > requirements.txt
```
