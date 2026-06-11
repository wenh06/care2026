---
layout: page
permalink: /instruction_left_atrium/
---

# CARE 2026 – Left Atrium Track Test Phase Docker Submission Instructions

Thank you for participating in the CARE 2026 Left Atrium Track!

> **NOTE:** This year, CARE-Left Atrium only award prizes for task1 (LA scar quantification) and task 3 (LA multi-structure segmentation). But participants can still submit your docker image for task 2 (LA cavity segmentation), we will conduct evaluation and post results on leaderboards.

During the **test phase**, each team must submit a **Docker image** containing their trained algorithm. The organizers will evaluate this image on a hidden test set to ensure fairness and reproducibility.

## Submission Overview

Prepare a Docker image with your prediction pipeline and submit the following via email:

- **Email address**: [care26challenge@163.com]() or [care2026challenge@outlook.com]() 
- **Email subject format**: [CARE-Left Atrium Test] `Team-Name` – Docker Submission

Include in the email body:
- A **download link** to your Docker image (in `.tar.gz` or `.tar` format), hosted on Google Drive, Mega, Baidu Netdisk, or a similar service.
- The **command** to run the Docker container (if not using an ENTRYPOINT), along with any additional instructions required to execute your pipeline.
- The specific **task** this Docker image is for (submit only for the tasks you are participating in):
  - `LA scar quantification`
  - `LA cavity segmentation`
  - `LA multi-structure segmentation`


Each team is allowed **up to two submissions per task**. Failed submissions (e.g., due to runtime errors or invalid outputs) **do not count** toward this limit. After a successful first submission, we will provide a feedback report with evaluation metrics. You may then submit an improved Docker image for the second attempt.

## Test Data Format

The directory structure is as follows:

```
LA-Test-Dataset/
├── task1
│  ├── test_01
│   │  ├──enhanced.nii.gz
│  ├── test_02
...

├── task2
│  ├── test_01
│   │  ├──enhanced.nii.gz
│  ├── test_02
...

├── task3
│  ├── test_01
│   │  ├──0001.nii.gz #Note that the input filename for Task 3 uses case IDs (e.g., 0001.nii.gz) instead of enhanced.nii.gz
│  ├── test_02
...
```

- All images are in `.nii.gz` format.

## Docker Requirements

### Input/Output Mounts

Your Docker container must:
- **Read input images** from the mounted `/input` directory (read-only).
- **Write predictions** to the mounted `/output` directory.

We will load and run your Docker image using commands similar to the following:

```bash
docker load -i your_docker_image.tar.gz

docker run \
  -v /path/to/Left Atrium-Test-Dataset:/input:ro \
  -v /path/to/output_dir:/output \
  -it your_image_name [your_command_if_needed]
```

- The container should run in a non-interactive mode if possible; avoid requiring user input.
- Ensure the container exits gracefully upon completion.

The expected output structure: 

- For LA scar quantification task: Predictions in `/output/`, structured as:
```
├── task1
│  ├── test_01
│   │  ├──01_pred.nii.gz    #Only need to predict scar
│  ├── test_02
│   │  ├──02_pred.nii.gz
```

- For LA cavity segmentation task: Predictions in `/output/`, structured as:
```
├── task2
│  ├── test_01
│   │  ├──01_pred.nii.gz
│  ├── test_02
│   │  ├──02_pred.nii.gz
```

- For LA multi-structure segmentation task: Predictions in `/output/`, structured as:
```
├── task3
│  ├── test_01
│   │  ├──01_pred.nii.gz
│  ├── test_02
│   │  ├──02_pred.nii.gz
```

### Special Note on Cascaded Pipelines
If your pipeline utilizes a cascaded approach,such as segmenting the left atrial cavity first and then quantifying the left atrial scar based on that segmentation output,please note that the organizers Will not provide any left atrial cavity labels or masks (atriumSegImgM0.nii.gz) in the official test set. The test phase evaluation will strictly follow these rules:

- **Input Limitation**: The input directory for the scar quantification task will ONLY contain the raw LGE MRI (enhanced.nii.gz). No reference anatomical segmentations will be pre-loaded.

- **Full Automation Required**: If your scar quantification module depends on cavity segmentation or cropping, your Docker container must internally perform the cavity segmentation first using your own trained models/algorithms. * End-to-End Submission: You must package this entire multi-stage pipeline into a single, self-contained Docker image that takes the raw image as input and directly outputs the final scar prediction (_pred.nii.gz) to the /output directory.

- **Evaluation Focus**: We will only evaluate the final left atrial scar quantification performance. Any intermediate cavity masks generated inside your container are solely for your pipeline's internal use and will not be evaluated or stored by the platform.

### Naming Convention

To facilitate organization, name your Docker image files as follows:

```
LA_scar_quantification-<TeamName>.tar.gz
LA_cavity_segmentation-<TeamName>.tar.gz
LA_multi-structure_segmentation-<TeamName>.tar.gz
```

### Example: Building Your Docker Image

Here is a minimal Dockerfile example for a Python-based pipeline:

```dockerfile
FROM python:3.9-slim

WORKDIR /workspace

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENTRYPOINT ["python", "main.py"]
```

To build and export the image:

```bash
docker build -t your_image_name .
docker save your_image_name | gzip > your_image_name.tar.gz
```

- Replace `main.py` with your entry script, which should handle reading from `/input`, processing all cases, and writing to `/output`.
- Test your image locally by mounting sample directories to simulate the evaluation environment.

### Additional Guidelines
- **Predictions**: Must match the input images' shapes, orientations, and data types.
- **No Training**: The container should only perform inference; no training or model updates are allowed.
- **No Internet Access**: The container must run offline—do not include dependencies requiring network access (e.g., no pip installs at runtime).
- **Efficiency**: Optimize for reasonable runtime on standard hardware (e.g., GPU if specified; indicate in your instructions if a GPU is required via `--gpus all`).
- **Error Handling**: Include robust logging and error handling to aid debugging.

## Evaluation Details

Predictions will be evaluated offline by the organizers using metrics such as Dice coefficient (DICE) and Hausdorff distance (HD). Results will be published on the test leaderboard.

## Final Notes
- Submit separate Docker images for each task if participating in multiple.
- Ensure output names and formats are consistent with validation phase submissions.
- The validation phase remains active during the test phase; you may continue submitting validation predictions as per previous instructions, and the leaderboard will update accordingly.
- For questions or issues, contact: [care26challenge@163.com]() or [care2026challenge@outlook.com]()

## Important Dates
- **Test Phase Start**: June 10, 2026
- **Docker Submission Deadline**: July 10, 2026
- **Abstract & Paper Submission Deadline**: July 20, 2026