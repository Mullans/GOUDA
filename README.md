
# GOUDA

[![Anaconda-Server Badge](https://anaconda.org/smullan/gouda/badges/version.svg)](https://anaconda.org/smullan/gouda)
[![pypi](https://img.shields.io/pypi/v/gouda)](https://pypi.org/project/GOUDA/)

**G**ood **O**ld **U**tilities for **D**ata **A**nalysis

This is my Python toolkit of reusable utilities built up over years of data analysis and computer vision research. GOUDA covers the tasks that come up repeatedly across projects: navigating file systems, loading and processing images, visualizing results, and computing statistics.

## Features

### Path Management

- `GoudaPath` — a chainable, callable `PathLike` that makes working with research directory structures concise. Resolves multiple subdirectories at once, integrates with file/image discovery, and works anywhere a path is expected.

### Image Processing

- Mask operations: semi-transparent overlays, border extraction, positive/negative colorization, image/mask/overlay lineups
- Segmentation utilities: GrabCut with probability thresholds, multi-threshold hysteresis-style binarization
- Spatial transforms: aspect-ratio-preserving resize, rotation with correct bounding-box handling, polar-to-Cartesian unrolling
- Cropping: bounding box extraction, crop-to-mask, crop-to-content
- Sharpness metrics: Laplacian variance, Sobel variance
- RGB-corrected `imread`/`imwrite` (OpenCV defaults to BGR)

### Visualization

- `print_grid` — display any collection of images as a grid. Accepts flat lists, nested lists, dicts with per-image titles/labels, or NumPy arrays of 2–5 dimensions. Layout is automatic.

### Data Utilities

- `to_uint8` — smart normalization that infers the input range (`[0,1]`, `[-1,1]`, `[0,255]`, or auto-rescale)
- `rescale`, `arr_sample`, `iter_batch`, `flip_dict`, and other NumPy/Python helpers
- `ParallelStats` — numerically stable online mean/variance (Welford algorithm) with support for aggregating across separate datasets

### More

...And plenty of other utility methods for everything from general data structures to specific hypothesis testing algorithms.

## Usage

### Navigating a project directory

```python
from gouda import GoudaPath

data = GoudaPath("data/experiments")

# Get multiple child paths at once
train, val, test = data("train", "val", "test")

# Chain into subdirectories — works anywhere a path is expected
weights_dir = data("outputs")("weights")
weights_dir.ensure_dir()  # create if it doesn't exist

# Find all images under a directory
image_paths = data("train").find_images(recursive=True)
```

### Displaying images during exploratory analysis

```python
from gouda.display import print_grid

# Flat list — grid layout is automatic
print_grid(images)

# Per-image titles and colormaps using dicts
print_grid([
    {"image": original,    "title": "Input"},
    {"image": pred_mask,   "title": "Prediction", "cmap": "hot"},
    {"image": ground_truth,"title": "Ground Truth"},
])

# Automatically arrange N images into a square grid
print_grid(images, do_squarify=True, figsize=(12, 12))
```

### Image processing

```python
from gouda.image import add_mask, padded_resize, mask_by_triplet

# Overlay a probability map on an image as a semi-transparent colored mask
overlay = add_mask(image, prob_map, color="red", opacity=0.5)

# Resize to a target shape, padding to preserve aspect ratio instead of squeezing
resized = padded_resize(image, size=(512, 512))

# Convert a continuous probability map to a binary mask using dual thresholds
#   (only keeps foreground regions where the peak signal exceeds upper_thresh)
binary = mask_by_triplet(prob_map, lower_thresh=0.3, upper_thresh=0.75)
```

## Installation

```bash
pip install gouda
```

```bash
conda install gouda -c smullan
```
