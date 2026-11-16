# Repository Guidelines

## Project Structure & Module Organization

Mingat is a Mongolian-history language model implemented from scratch in PyTorch. Source modules live at the repository root: `model.py` defines the transformer, `tokenizer.py` implements byte-level BPE, and `train.py` handles training. `prepare.py` cleans books; `generate.py` provides text completion; `memory.py`, `persona.py`, and `program_qa.py` implement retrieval, character training math, and weight-space Q&A experiments. `sweep.py` and `train_colab.ipynb` support GPU experiments. Checks live in `tests.py`; public sample data lives in `examples/sample_qa.csv`. Private corpora belong in `data/`, and generated checkpoints belong in `out/`.

## Build, Test, and Development Commands

There is no separate build step. Run scripts from the repository root.

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install torch numpy openpyxl
python tests.py
python prepare.py --zip data/mingat_books.zip
python train.py --data data/clean --test data/test --preset micro --dropout 0.1 --epochs 6 --threads 8
python generate.py --model out/model.pt
```

These commands create an environment, install dependencies, run all checks, prepare local books, train a model, and start interactive completion. Preparation requires books under `data/novels/`; generation requires a trained checkpoint. Training can take hours.

## Coding Style & Naming Conventions

Use four-space indentation, `snake_case` for functions and variables, `PascalCase` for classes, and `UPPER_SNAKE_CASE` for constants. Follow existing type hints, module docstrings, and explanatory comments for numerical logic. Use `pathlib.Path` and explicit UTF-8 encoding for text files. No formatter or linter configuration is checked in; match surrounding code.

## Testing Guidelines

`python tests.py` runs Python assertion checks and module self-tests without pytest or unittest. Add `test_<behavior>()` functions to `tests.py` and invoke them in its main block. Use fixed seeds, tiny models, and temporary directories. Cover changed numerical behavior and relevant checkpoint/resume paths. No coverage percentage is configured. Keep `prepare.py`'s `TEST_BOOKS` excluded from training.

## Commit & Pull Request Guidelines

History uses descriptive, sentence-case subjects such as `Add generate.py: write with a trained model`; follow that style. PRs should explain the change, link relevant issues, and report validation commands and results. For training changes, record settings and comparable held-out scores. Update `README.md` when commands or behavior change. Keep private books, checkpoints, and local environments out of commits; their directories are gitignored.
