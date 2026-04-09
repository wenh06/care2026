# MICCAI CARE 2026 - Left Atrial Segmentation and Analysis

[![Formatting](https://github.com/wenh06/care2026/actions/workflows/check-formatting.yml/badge.svg)](https://github.com/wenh06/care2026/actions/workflows/check-formatting.yml)
[![Docker CI](https://github.com/wenh06/care2026/actions/workflows/docker-test.yml/badge.svg)](https://github.com/wenh06/care2026/actions/workflows/docker-test.yml)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

<!-- toc -->

- [Introduction](#introduction)
- [Tasks](#tasks)
- [Data](#data)
- [Description of files/folders (modules)](#description-of-filesfoldersmodules)
- [Deep learning models for medical image studies](#deep-learning-models-for-medical-image-studies)
- [Leaderboards](#leaderboards)
- [Citations](#citations)

<!-- tocstop -->

## Introduction

Atrial fibrillation (AF), the most prevalent cardiac arrhythmia, is poised to escalate in frequency due to aging populations. Radiofrequency catheter ablation is a common AF therapy, but faces challenges due to high recurrence rates. Cardiac digital twin technology provides personalized _in-silico_ cardiac representations to infer multi-scale properties associated with cardiac mechanisms, showing great promise in personalized targeted ablation of persistent AF.

To create such a digital twin, it is important to reconstruct the left atrial (LA) geometry with the location of scars from LGE MRI. However, automatic quantification and analysis of LA scars is quite challenging due to the low image quality, thin atrial wall (~1–3 mm), the surrounding enhanced regions, and the complex and highly individual patterns of scarring. Deep learning (DL) methods have shown promise in LGE MRI analysis, yet their performance often falters in new domains due to domain shifts across imaging centers and scanners.

CARE-Left Atrium aims to address these issues, driving the advancement of DL models that precisely delineate LA cavity and scars, and ultimately revolutionize personalized AF treatment. This challenge extends beyond purely MRI-based approaches by also providing multi-center CT data for left atrial multi-structure segmentation.

The challenge page is at [https://www.zmic.org.cn/care_2026/track_leftatrium/](https://www.zmic.org.cn/care_2026/track_leftatrium/).

## Tasks

The challenge comprises three tasks:

| Task | Input | Target | Metrics |
|------|-------|--------|---------|
| **Task 1**: LA scar quantification | LGE-MRI (Center A) | LA scar segmentation (`scar_predict.nii.gz`) | G-DSC, ACC, SEN |
| **Task 2**: LA cavity segmentation | LGE-MRI (Centers A, B, C) | LA cavity mask (`LA_predict.nii.gz`) | DSC, HD |
| **Task 3**: LA multi-structure segmentation | CT (Center D) | LA + pulmonary veins + left atrial appendage (`cardiac_predict.nii.gz`) | DSC, HD |

> **Note:** Prizes are awarded only for Task 1 and Task 3.

### Task 1: LA Scar Quantification (LGE-MRI)

Segment the LA scarring region from LGE-MRI. This is the most challenging task due to the extremely thin atrial wall, the low signal-to-noise ratio of LGE-MRI, the patchy and diffuse nature of scars, and the small training set (60 samples). Evaluated by Generalized Dice (G-DSC), Accuracy (ACC), and Sensitivity (SEN).

### Task 2: LA Cavity Segmentation (LGE-MRI)

Segment the left atrial cavity from multi-center LGE-MRI. The key challenge here is cross-center generalization: training data comes from Center A only, while the test set spans three centers (A, B, C) with different scanners (Siemens vs. Philips), field strengths (1.5T vs. 3T), and spatial resolutions. Evaluated by DSC and Hausdorff Distance.

### Task 3: LA Multi-Structure Segmentation (CT)

Segment the left atrium, pulmonary veins, and left atrial appendage jointly from CT scans. The challenge here lies in the complex topology and high inter-subject shape variability of pulmonary veins and the left atrial appendage. Evaluated by DSC and Hausdorff Distance.

## Data

| Center | Modality | # Training | # Validation | # Test | Tasks |
|--------|----------|-----------|-------------|--------|-------|
| Center A (Utah NAMIC-CARMA) | LGE-MRI (0.625×0.625×2.5 mm) | 60 (T1), 130 (T2) | 10 | 24 (T1), 14 (T2) | Task 1, Task 2 |
| Center B (Beth Israel, Boston) | LGE-MRI (1.4×1.4×1.4 mm) | — | — | 20 | Task 2 |
| Center C (King's College London) | LGE-MRI (1.3×1.3×4.0 mm) | — | 10 | 10 | Task 2 |
| Center D (Fuzhou University Hospital) | CT (0.30–0.80×0.30–0.80×0.5 mm) | 150 | 20 | 130 | Task 3 |

Data are provided in NIfTI format:

- `enhanced.nii.gz` — LGE-MRI or CT image
- `atriumSegImgMO.nii.gz` — LA cavity segmentation label
- `scarSegImgM.nii.gz` — LA scar segmentation label (Task 1)
- `cardiacSegImgMO.nii.gz` — multi-structure label: LA + pulmonary veins + left atrial appendage (Task 3)

## Description of files/folders (modules)

### Files

<details>
<summary>Click to view the details</summary>

- [README.md](README.md): this file, project documentation.
- [cfg.py](cfg.py): centralized configuration for all tasks (BaseCfg, TrainCfg, ModelCfg).
- [const.py](const.py): constant definitions, including URLs for downloading trained model weights.
- [data_reader.py](data_reader.py): data reader covering LGE-MRI and CT, including file listing, data loading, and modality-specific preprocessing.
- [dataset.py](dataset.py): PyTorch `Dataset` classes for all three tasks.
- [Dockerfile](Dockerfile): Docker image definition for challenge submission (base: `pytorch/pytorch:2.9.1-cuda12.8-cudnn9-runtime`).
- [evaluate-results](evaluate-results): records of evaluation results on the validation set.
- [outputs.py](outputs.py): output dataclass containers for model predictions.
- [pipeline.py](pipeline.py): inference pipeline (coarse-to-fine for MRI tasks; direct for CT), serving as the entry point for submission.
- [post_docker_build.py](post_docker_build.py): downloads and caches our own trained model weights into the Docker image at build time.
- [predict.py](predict.py): command line interface for the submission (entry point for the Docker `ENTRYPOINT`).
- [requirements.txt](requirements.txt): full requirements for local development.
- [requirements-docker.txt](requirements-docker.txt): requirements for the Docker image (torch pre-installed in base image).
- [requirements-no-torch.txt](requirements-no-torch.txt): requirements excluding all torch-related packages.
- [trainer.py](trainer.py): trainer class supporting single-task and multi-task training.
- [train_models.ipynb](train_models.ipynb): notebook for interactive model training and experimentation.

</details>

### Folders (Modules)

- [models](models): model architecture definitions.
  - [vnet.py](models/vnet.py): 3D VNet for volumetric segmentation.
  - [nested_vnet.py](models/nested_vnet.py): Nested (UNet++-style) 3D VNet.
  - [layers.py](models/layers.py): shared building blocks (convolution blocks, attention modules, etc.).
  - [loss/](models/loss): custom loss functions — region-based (Dice, Tversky, Focal Dice), boundary-aware, distribution-based, and compound losses.
- [utils](utils): utility functions.
  - [scoring_metrics.py](utils/scoring_metrics.py): evaluation metrics for all three tasks (DSC, HD, G-DSC, ACC, SEN).
  - [mclahe.py](utils/mclahe.py): Multidimensional Contrast Limited Adaptive Histogram Equalization (MCLAHE) for LGE-MRI contrast enhancement.

## Deep learning models for medical image studies

[MONAI Model Zoo](https://monai.io/model-zoo.html) | [MONAI at GitHub](https://github.com/Project-MONAI)

[NVIDIA: Visual Foundation Models for Medical Image Analysis](https://developer.nvidia.com/blog/visual-foundation-models-for-medical-image-analysis/)

## Leaderboards

Leaderboards will be released after test results submission. See the [challenge page](https://www.zmic.org.cn/care_2026/track_leftatrium/) for updates.

## Citations

**Please cite these papers when using the challenge data:**

```bibtex
@article{li2022atrialjsqnet,
    title={AtrialJSQnet: a new framework for joint segmentation and quantification of left atrium and scars incorporating spatial and shape information},
    author={Li, Lei and Zimmer, Veronika A and Schnabel, Julia A and Zhuang, Xiahai},
    journal={Medical image analysis},
    volume={76},
    pages={102303},
    year={2022},
    publisher={Elsevier}
}

@article{GAO2023BayeSeg,
    title={BayeSeg: Bayesian modeling for medical image segmentation with interpretable generalizability},
    journal={Medical Image Analysis},
    volume={89},
    pages={102889},
    year={2023},
    author={Shangqi Gao and Hangqi Zhou and Yibo Gao and Xiahai Zhuang},
}

@article{zhuang2019multivariate,
    title={Multivariate mixture model for myocardial segmentation combining multi-source images},
    author={Zhuang, Xiahai},
    journal={IEEE transactions on pattern analysis and machine intelligence},
    volume={41},
    number={12},
    pages={2933--2946},
    year={2019},
}

@article{zhuang2016multi,
    title={Multi-scale patch and multi-modality atlases for whole heart segmentation of MRI},
    author={Zhuang, Xiahai and Shen, Juan},
    journal={Medical image analysis},
    volume={31},
    pages={77--87},
    year={2016},
    publisher={Elsevier}
}
```
