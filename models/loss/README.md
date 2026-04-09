# Loss Functions

Custom loss functions for the CARE 2026 Left Atrium challenge segmentation tasks.

| File | Description |
|------|-------------|
| [`region_loss.py`](region_loss.py) | Region-based losses: Dice, Tversky, Focal Dice |
| [`boundary_loss.py`](boundary_loss.py) | Boundary-aware losses |
| [`distribution_loss.py`](distribution_loss.py) | Distribution-based losses: KL divergence, Wasserstein |
| [`compound_loss.py`](compound_loss.py) | Compound losses combining multiple objectives |
