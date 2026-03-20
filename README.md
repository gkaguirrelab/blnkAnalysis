# BLNK Analysis Pipeline

This repository contains the full analysis pipeline for the BLNK project, enabling automated extraction of eye-related features from dual-eye video recordings.

The pipeline is designed to be:
- modular
- reproducible
- interactive (via Jupyter Notebook)
- compatible with downstream MATLAB and Python workflows

It builds on the PyLids framework and extends it with custom preprocessing, video handling, and analysis logic.

---

## Overview

The BLNK pipeline processes raw dual-eye videos and produces structured feature outputs for each eye.

For each input video, the pipeline:

1. Splits the video into left-eye and right-eye streams  
2. Crops the eye region using an elliptical mask  
3. Applies preprocessing:
   - brightness thresholding
   - contrast / gamma / brightness adjustments
   - padding to a consistent size  
4. Runs eye-feature extraction via PyLids  
5. Saves results to MATLAB-compatible `.mat` files  

Optional visualization outputs can also be generated for debugging and parameter tuning.

---

## Repository Structure

```
blnkAnalysis/
│
├── blnk_analysis_pipeline.ipynb   # Main interactive pipeline (entry point)
├── blnk_analysis_pipeline.py      # Core processing + preprocessing logic
├── subject_settings/              # Per-subject parameter configurations
├── pylids/                        # Submodule dependency (feature extraction backend)
├── README.md
```

---

## Subject Settings

The `subject_settings/` directory stores predefined parameter configurations for individual subjects.

Each subject may require slightly different preprocessing due to:
- differences in lighting conditions
- camera positioning
- eye appearance variability
- recording-specific artifacts

These settings typically include:
- crop ellipse parameters
- threshold values
- brightness / contrast / gamma adjustments
- subject-specific preprocessing overrides

### Why this matters

Rather than manually tuning parameters each time, this system allows:
- consistent preprocessing across sessions for the same subject
- reproducibility of results
- easier batch processing

### Recommended usage

- When analyzing a known subject → load their saved settings  
- When analyzing a new subject → tune parameters and save a new configuration  

---

## Installation

### 1. Clone the repository (with submodules)

```bash
git clone --recurse-submodules git@github.com:zkelly1/blnkAnalysis.git
cd blnkAnalysis
```

> Note: The `--recurse-submodules` flag is required to properly clone the `pylids` dependency.

---

### 2. Set up the environment

Follow the setup instructions inside:

```
pylids/
```

This involves:
- creating a Python/Conda environment
- installing required dependencies
---

---

## Usage

The pipeline is designed to be used through the Jupyter Notebook:

```
blnk_analysis_pipeline.ipynb
```

---

## Recommended Workflow

### Step 1 — Launch notebook

### Step 1 — Launch notebook

You can run the pipeline using either Jupyter Notebook or VS Code.

#### Option A — Jupyter Notebook Interface

To use the jupyter native notebook interface, do the following

```bash
jupyter notebook
```

Then open: 
`blnk_analysis_pipeline.ipynb` 

#### Option B — VS Code Notebook Interface

You can open and run the pipeline directly in VS Code using its built-in notebook interface:

1. Open the repository folder in VS Code  
2. Navigate to:
`blnk_analysis_pipeline.ipynb` 
3. Open the notebook file  
4. In the top-right corner, select the correct Python kernel  
- This should be the environment where PyLids and all dependencies are installed (titled pylids)
5. Run cells interactively using:
- ▶ Run Cell buttons, or  
- Shift + Enter  

---

> Important: If the notebook does not run correctly, the most common issue is selecting the wrong Python kernel.

> Tip: VS Code provides a smoother development experience with variable inspection, inline outputs, and easier debugging compared to the classic Jupyter interface.
---

### Step 2 — Configure environment

- Verify kernel
- Install missing dependencies (if needed)
- Import custom modules

---

### Step 3 — Select input videos

Provide:
- a single video path, OR
- a directory of videos, OR
- a list of filepaths

---

### Step 4 — Define output directory

Choose a location where processed results will be saved.

---

### Step 5 — Load subject settings (if available)

If the subject has an existing configuration in `subject_settings/`, load it to ensure consistent preprocessing.

Otherwise, manually define parameters.

---

### Step 6 — Set preprocessing parameters

Key parameters include:

| Parameter | Description |
|----------|------------|
| crop_ellipse | Defines region of interest for eye |
| target_size | Output frame size after padding |
| whiteness_threshold | Removes bright artifacts |
| contrast, gamma, brightness | Image enhancement |

---

### Step 7 — Verify parameters

Run the verification step to visually confirm:

- correct eye isolation
- proper cropping
- no loss of important features
- reasonable brightness/contrast

---

### Step 8 — Run full analysis

Execute the pipeline to process all videos and generate outputs.

---

## Output

For each processed video, the pipeline produces:

- `.mat` files containing:
  - extracted eye features
  - metadata
  - preprocessing parameters

Optional visualization outputs may also be generated.

Example:

```
video_left_eyeFeatures.mat
video_right_eyeFeatures.mat
```

---

## GPU Acceleration

If a CUDA-enabled GPU is available, PyTorch can accelerate parts of the pipeline.

Check availability:

```python
import torch
print(torch.cuda.is_available())
```

You can also run:

```bash
nvidia-smi
```

---

## Notes and Best Practices

- Always verify preprocessing parameters before batch runs  
- Use subject-specific settings when available  

---

## Common Issues

### GPU not detected
- Ensure CUDA is installed
- Verify correct PyTorch version

### Import errors
- Check that `pylids` submodule is initialized
- Confirm correct Python environment
- Confirm installation of extra libraries this project uses ontop of PyLids

### Poor feature extraction
- Revisit preprocessing parameters
- Adjust ellipse and thresholds

---

## Development Notes

This repository combines:
- a reusable Python processing pipeline
- an interactive notebook interface
- integration with PyLids for feature extraction

---