# Models

This folder contains model definitions for the CARE 2026 Left Atrium challenge.

## Architectures

- [`vnet.py`](vnet.py): 3D VNet for volumetric segmentation.
- [`nested_vnet.py`](nested_vnet.py): Nested (UNet++-style) 3D VNet.
- [`layers.py`](layers.py): Shared building blocks (convolution blocks, attention modules, etc.).

## Loss Functions

See [`loss/`](loss/) for custom loss functions.
