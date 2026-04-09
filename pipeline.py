"""
Inference pipeline for the CARE 2026 Left Atrium challenge.

Supports all three tasks:
- Task 1: LA scar quantification from LGE-MRI
- Task 2: LA cavity segmentation from LGE-MRI
- Task 3: LA multi-structure segmentation from CT

The pipeline follows a coarse-to-fine strategy for MRI tasks and a
direct segmentation approach for CT.
"""
