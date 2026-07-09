# CLAUDE.md — CARE2026 project instructions

## Coding Rules

1. **All imports at top** of every file. No inline imports inside functions.
2. **Collect results into a dict** first, then print/write a unified table after
   all experiments complete. tqdm progress bars for real-time feedback are fine.
3. **Support `--output`** in analysis/eval scripts to save results to file.
4. **Test before committing**: at minimum `python3 -c "from module import Thing"`,
   and ideally a functional smoke test for new CLI arguments or logic.

## Repository Layout

- `models/` — VNet backbone + nnUNet wrappers + custom loss/trainer
- `scripts/` — data prep (`prep_nnunet_*.py`), evaluation, sweep analysis
- `cfg.py` — all training & inference configs (torch_ecg CFG)
- `predict.py` — low-level volume inference; `pipeline.py` — high-level CLI
- `tmp/nnUNet_results/` — trained nnUNet model directories (5-fold CV)
- `checkpoints/` — VNet .safetensors files + nnUNet symlinks
- `log/` — training logs (VNet CSV, nnUNet TXT)
