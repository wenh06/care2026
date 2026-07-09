# CLAUDE.md — CARE2026 mandatory coding rules

Violation of these rules is considered a **bug**. Do not finish the task
until all rules are satisfied.

## Rules

### 1. All imports at file top

- MUST place every `import` / `from ... import` at the top of the file.
- MUST NOT put imports inside functions or conditional blocks.

```python
# BAD
def foo():
    import numpy as np

# GOOD
import numpy as np

def foo():
    ...
```

### 2. Collect results before printing

- MUST collect experiment metrics into a `dict` first.
- MUST print a unified summary table AFTER all experiments complete.
- tqdm progress bars for real-time feedback are fine and encouraged.

```python
# BAD
for exp in experiments:
    result = run(exp)
    print(result)          # gets lost in warnings

# GOOD
results = {}
for exp in tqdm(experiments):
    results[label] = run(exp)
_print_summary(results)    # clean table at end
```

### 3. Every eval/analysis script MUST support `--output`

- MUST add `parser.add_argument("--output", ...)` to save results to file.

### 4. Test before declaring done

- MUST run at minimum `python3 -c "from module import Thing"` after writing code.
- MUST verify new CLI arguments parse correctly before committing.

## Pre-commit checklist

Before presenting code as complete, verify:

- [ ] No inline imports — all imports at file top.
- [ ] Results collected into dict, printed at end.
- [ ] `--output` argument present in new eval/analysis scripts.
- [ ] Smoke test: `python3 -c "import <new_module>"` passes.
- [ ] CLI arg test: `python3 -c "... parse_args(...)"` passes for new args.

## Repository Layout

- `models/` — VNet backbone + nnUNet wrappers + custom loss/trainer
- `scripts/` — data prep (`prep_nnunet_*.py`), evaluation, sweep analysis
- `cfg.py` — all training & inference configs (torch_ecg CFG)
- `predict.py` — low-level volume inference; `pipeline.py` — high-level CLI
- `tmp/nnUNet_results/` — trained nnUNet model directories (5-fold CV)
- `checkpoints/` — VNet .safetensors files + nnUNet symlinks
- `log/` — training logs (VNet CSV, nnUNet TXT)
