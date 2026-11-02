# Breast Cancer Screening Classifier

Deep neural network models for breast cancer classification on screening mammograms, based on the paper [Deep Neural Networks Improve Radiologists' Performance in Breast Cancer Screening](https://ieeexplore.ieee.org/document/8861376).

This repository lets you run pretrained models on screening mammography exams and get per-breast probabilities of benign and malignant findings. Two models are provided:

- **Image-only**: takes mammogram images as input.
- **Image-and-heatmaps**: takes mammogram images plus patch-classifier heatmaps as input.

## Table of Contents

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Usage](#usage)
  - [Exam-level predictions](#exam-level-predictions)
  - [Single-image predictions](#single-image-predictions)
- [Data Format](#data-format)
- [Pipeline](#pipeline)
  - [Configuration variables](#configuration-variables)
  - [Preprocessing](#preprocessing)
  - [Heatmap generation](#heatmap-generation)
  - [Running the models](#running-the-models)
- [Converting DICOM to PNG](#converting-dicom-to-png)
- [License](#license)

## Overview

**Input**
- **Images**: 2 CC-view mammograms (2677x1942) and 2 MLO-view mammograms (2974x1748). Each is a 16-bit PNG and is standardized individually before being fed to the models.
- **Heatmaps** (image-and-heatmaps model only): output of the patch classifier, resized to match its mammogram. Each mammogram has two heatmaps (benign and malignant), with pixel values between 0 and 1.

**Output**

Four probabilities per exam: `left_benign`, `right_benign`, `left_malignant`, `right_malignant`.

Both models operate on screening exams with four standard views (L-CC, R-CC, L-MLO, R-MLO). Four sample exams are included in `sample_data/images`, with the exam list in `sample_data/exam_list_before_cropping.pkl`. The heatmap generator and the classifiers are implemented in PyTorch.

## Prerequisites

| Package | Version |
| ------- | ------- |
| Python | 3.6 |
| PyTorch | 0.4.1 |
| torchvision | 0.2.0 |
| NumPy | 1.14.3 |
| SciPy | 1.0.0 |
| H5py | 2.7.1 |
| imageio | 2.4.1 |
| pandas | 0.22.0 |
| tqdm | 4.19.8 |
| opencv-python | 3.4.2 |

Converting DICOM files additionally requires `pypng` 0.0.19 and `pydicom` 1.2.2 (see [Converting DICOM to PNG](#converting-dicom-to-png)).

## Quick Start

Run the full pipeline on the sample exams:

```bash
bash run.sh
```

This crops the images, computes optimal window centers, generates heatmaps, runs both classifiers, and saves predictions as CSV files.

- A GPU is used by default. To run on CPU, set `DEVICE_TYPE` to `'cpu'` in `run.sh`.
- If you run the Python scripts individually, add the path to this repository to your `PYTHONPATH`.

## Usage

### Exam-level predictions

The default workflow uses the **view-wise** model, the best-performing model. It takes all four views of an exam as input and outputs one prediction set per exam.

Expected output for the sample exams:

**Image-only model** (`sample_output/image_predictions.csv`)

| index | left_benign | right_benign | left_malignant | right_malignant |
| ----- | ----------- | ------------ | -------------- | --------------- |
| 0     | 0.0580      | 0.0754       | 0.0091         | 0.0179          |
| 1     | 0.0646      | 0.9536       | 0.0012         | 0.7258          |
| 2     | 0.4388      | 0.3526       | 0.2325         | 0.1061          |
| 3     | 0.3765      | 0.6483       | 0.0909         | 0.2579          |

**Image-and-heatmaps model** (`sample_output/imageheatmap_predictions.csv`)

| index | left_benign | right_benign | left_malignant | right_malignant |
| ----- | ----------- | ------------ | -------------- | --------------- |
| 0     | 0.0612      | 0.0555       | 0.0099         | 0.0063          |
| 1     | 0.0507      | 0.8025       | 0.0009         | 0.9000          |
| 2     | 0.2877      | 0.2286       | 0.2524         | 0.0461          |
| 3     | 0.4181      | 0.3172       | 0.3174         | 0.0485          |

### Single-image predictions

An **image-wise** model is also provided. It performs worse than the view-wise model, and its outputs will differ from the exam-level results above. Because it predicts from each image independently, it is released mainly to support transfer learning.

```bash
bash run_single.sh "sample_data/images/0_L_CC.png" "L-CC"
```

Arguments:
1. Path to a mammogram image.
2. The view of that image (`L-CC`, `R-CC`, `L-MLO`, or `R-MLO`).

Expected output:

```
Stage 1: Crop Mammograms
Stage 2: Extract Centers
Stage 3: Generate Heatmaps
Stage 4a: Run Classifier (Image)
{"benign": 0.040191903710365295, "malignant": 0.008045293390750885}
Stage 4b: Run Classifier (Image+Heatmaps)
{"benign": 0.052365876734256744, "malignant": 0.005510155577212572}
```

A [sample notebook](sample_notebook.ipynb) demonstrates running the classifiers with and without heatmaps (preprocessing is not included).

## Data Format

Each exam requires at least four images, one per view (L-CC, L-MLO, R-CC, R-MLO).

Original 12-bit mammograms are stored as rescaled 16-bit PNGs. This preserves pixel-intensity granularity while displaying correctly in standard image viewers.

### Exam list

`sample_data/exam_list_before_cropping.pkl` holds the exam list before preprocessing. Each exam is a dictionary:

```python
{
  'horizontal_flip': 'NO',
  'L-CC': ['0_L_CC'],
  'R-CC': ['0_R_CC'],
  'L-MLO': ['0_L_MLO'],
  'R-MLO': ['0_R_MLO'],
}
```

- The view keys (`L-CC`, `R-CC`, `L-MLO`, `R-MLO`) map to lists of image filenames, **without** extension or directory.
- `L-CC` and `L-MLO` images are expected to face right; `R-CC` and `R-MLO` images are expected to face left.
- `horizontal_flip` indicates whether all images in the exam are flipped horizontally relative to this expectation.

### Metadata added by preprocessing

Extra per-image information is stored as dictionaries keyed by view, with one list entry per image. For example, `window_location` (the top, bottom, left, and right edges of the cropping window) has four keys, each holding a list of values for that view's images. Preprocessing also adds `rightmost_points`, `bottommost_points`, `distance_from_starting_side`, and `best_center`; see [Preprocessing](#preprocessing) for their meanings.

Example exam after cropping and optimal-center extraction:

```python
{
  'horizontal_flip': 'NO',
  'L-CC': ['0_L_CC'],
  'R-CC': ['0_R_CC'],
  'L-MLO': ['0_L_MLO'],
  'R-MLO': ['0_R_MLO'],
  'window_location': {
    'L-CC': [(353, 4009, 0, 2440)],
    'R-CC': [(71, 3771, 952, 3328)],
    'L-MLO': [(0, 3818, 0, 2607)],
    'R-MLO': [(0, 3724, 848, 3328)]
  },
  'rightmost_points': {
    'L-CC': [((1879, 1958), 2389)],
    'R-CC': [((2207, 2287), 2326)],
    'L-MLO': [((2493, 2548), 2556)],
    'R-MLO': [((2492, 2523), 2430)]
  },
  'bottommost_points': {
    'L-CC': [(3605, (100, 100))],
    'R-CC': [(3649, (101, 106))],
    'L-MLO': [(3767, (1456, 1524))],
    'R-MLO': [(3673, (1164, 1184))]
  },
  'distance_from_starting_side': {
    'L-CC': [0],
    'R-CC': [0],
    'L-MLO': [0],
    'R-MLO': [0]
  },
  'best_center': {
    'L-CC': [(1850, 1417)],
    'R-CC': [(2173, 1354)],
    'L-MLO': [(2279, 1681)],
    'R-MLO': [(2185, 1555)]
  }
}
```

### Sample labels

| index | left_benign | right_benign | left_malignant | right_malignant |
| ----- | ----------- | ------------ | -------------- | --------------- |
| 0     | 0           | 0            | 0              | 0               |
| 1     | 0           | 0            | 0              | 1               |
| 2     | 1           | 0            | 0              | 0               |
| 3     | 1           | 1            | 1              | 1               |

## Pipeline

The pipeline has four stages:

1. Crop mammograms
2. Calculate optimal centers
3. Generate heatmaps
4. Run classifiers

### Configuration variables

Set these in `run.sh`.

**Runtime settings**

| Variable | Description | Default |
| -------- | ----------- | ------- |
| `NUM_PROCESSES` | Processes used in preprocessing (`src/cropping/crop_mammogram.py`, `src/optimal_centers/get_optimal_centers.py`) | 10 |
| `DEVICE_TYPE` | `'cpu'` or `'gpu'`, used for heatmap generation and classifiers | `'gpu'` |
| `NUM_EPOCHS` | Number of epochs averaged in classifier output | 10 |
| `HEATMAP_BATCH_SIZE` | Batch size for heatmap generation | 100 |
| `GPU_NUMBER` | Which GPU to use when several are available | 0 |

**Input paths**

| Variable | Description |
| -------- | ----------- |
| `DATA_FOLDER` | Directory containing the mammograms |
| `INITIAL_EXAM_LIST_PATH` | Initial exam list (no metadata) |
| `PATCH_MODEL_PATH` | Saved weights of the patch classifier |
| `IMAGE_MODEL_PATH` | Saved weights of the image-only model |
| `IMAGEHEATMAPS_MODEL_PATH` | Saved weights of the image-and-heatmaps model |

**Output paths**

| Variable | Description |
| -------- | ----------- |
| `CROPPED_IMAGE_PATH` | Directory for cropped mammograms |
| `CROPPED_EXAM_LIST_PATH` | Exam list with cropping metadata |
| `EXAM_LIST_PATH` | Exam list with best-center metadata |
| `HEATMAPS_PATH` | Directory for heatmaps |
| `IMAGE_PREDICTIONS_PATH` | Image-only predictions |
| `IMAGEHEATMAPS_PREDICTIONS_PATH` | Image-and-heatmaps predictions |

### Preprocessing

#### 1. Crop mammograms

```bash
python3 src/cropping/crop_mammogram.py \
    --input-data-folder $DATA_FOLDER \
    --output-data-folder $CROPPED_IMAGE_PATH \
    --exam-list-path $INITIAL_EXAM_LIST_PATH \
    --cropped-exam-list-path $CROPPED_EXAM_LIST_PATH \
    --num-processes $NUM_PROCESSES
```

This script crops each mammogram around the breast and discards the background, which speeds up image loading and segmentation. Cropped images are saved to `$CROPPED_IMAGE_PATH`, and a new exam list with extra metadata is written to `$CROPPED_EXAM_LIST_PATH`. Images that fail to crop are dropped. Add `--verbose` for per-image output.

Added metadata:
- `window_location`: location of the cropping window relative to the original DICOM image, so a segmentation map can be cropped identically for training.
- `rightmost_points`: rightmost nonzero pixels after correct flipping.
- `bottommost_points`: bottommost nonzero pixels after correct flipping.
- `distance_from_starting_side`: flags a zero-value gap between the image edge and the breast on the side where the breast should begin (where no gap is expected). Depending on the dataset, this can reveal an incorrect `horizontal_flip` value.

#### 2. Calculate optimal centers

```bash
python3 src/optimal_centers/get_optimal_centers.py \
    --cropped-exam-list-path $CROPPED_EXAM_LIST_PATH \
    --data-prefix $CROPPED_IMAGE_PATH \
    --output-exam-list-path $EXAM_LIST_PATH \
    --num-processes $NUM_PROCESSES
```

This writes a new exam list with extra metadata to `$EXAM_LIST_PATH`.

Added metadata:
- `best_center`: optimal center point of the augmentation window for each image. A window centered exactly on `best_center` can extend past the image boundary, usually when the cropped image is smaller than the window. In that case the image is padded and the window is shifted inside the padded image during augmentation.

### Heatmap generation

```bash
python3 src/heatmaps/run_producer.py \
    --model-path $PATCH_MODEL_PATH \
    --data-path $EXAM_LIST_PATH \
    --image-path $CROPPED_IMAGE_PATH \
    --batch-size $HEATMAP_BATCH_SIZE \
    --output-heatmap-path $HEATMAPS_PATH \
    --device-type $DEVICE_TYPE \
    --gpu-number $GPU_NUMBER
```

`src/heatmaps/run_producer.py` builds heatmaps by combining patch-level predictions and saves them in HDF5 format to `$HEATMAPS_PATH`.

- `$DEVICE_TYPE` is `'gpu'` or `'cpu'`.
- Adjust `$HEATMAP_BATCH_SIZE` to fit available memory.
- `--gpu-number` (optional) selects the GPU.

### Running the models

`src/modeling/run_model.py` predicts from cropped images, with or without heatmaps.

- To use heatmaps, pass `--use-heatmaps` along with the appropriate `--model-path` and `--heatmaps-path`.
- `--batch-size` (optional) can be tuned to available memory.
- `--gpu-number` (optional) selects the GPU.

#### Image-only model

```bash
python3 src/modeling/run_model.py \
    --model-path $IMAGE_MODEL_PATH \
    --data-path $EXAM_LIST_PATH \
    --image-path $CROPPED_IMAGE_PATH \
    --output-path $IMAGE_PREDICTIONS_PATH \
    --use-augmentation \
    --num-epochs $NUM_EPOCHS \
    --device-type $DEVICE_TYPE \
    --gpu-number $GPU_NUMBER
```

Predicts from images alone over `$NUM_EPOCHS` epochs with random augmentation, and writes per-exam averaged predictions to `$IMAGE_PREDICTIONS_PATH`.

#### Image-and-heatmaps model

```bash
python3 src/modeling/run_model.py \
    --model-path $IMAGEHEATMAPS_MODEL_PATH \
    --data-path $EXAM_LIST_PATH \
    --image-path $CROPPED_IMAGE_PATH \
    --output-path $IMAGEHEATMAPS_PREDICTIONS_PATH \
    --use-heatmaps \
    --heatmaps-path $HEATMAPS_PATH \
    --use-augmentation \
    --num-epochs $NUM_EPOCHS \
    --device-type $DEVICE_TYPE \
    --gpu-number $GPU_NUMBER
```

Predicts from images and heatmaps over `$NUM_EPOCHS` epochs with random augmentation, and writes per-exam averaged predictions to `$IMAGEHEATMAPS_PREDICTIONS_PATH`.

## Converting DICOM to PNG

DICOM files can be converted to the 16-bit PNG format this code expects using the function below (requires `pypng` 0.0.19 and `pydicom` 1.2.2).

```python
import png
import pydicom

def save_dicom_image_as_png(dicom_filename, png_filename, bitdepth=12):
    """
    Save a 12-bit mammogram from DICOM as a rescaled 16-bit PNG file.
    :param dicom_filename: path to input DICOM file.
    :param png_filename: path to output PNG file.
    :param bitdepth: bit depth of the input image. Use 12 for 12-bit mammograms.
    """
    image = pydicom.read_file(dicom_filename).pixel_array
    with open(png_filename, 'wb') as f:
        writer = png.Writer(height=image.shape[0], width=image.shape[1], bitdepth=bitdepth, greyscale=True)
        writer.write(f, image.tolist())
```

## License

This repository is licensed under the GNU AGPLv3.
