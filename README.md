# surgical-tool-detection

# Surgical Tool Detection Using Deep Learning

A computer vision project for detecting and localizing surgical instruments in laparoscopic images using **YOLO-based object detection**. The system identifies surgical instruments, predicts their bounding boxes, and assigns confidence scores to detections. The project also investigates dataset balancing and augmentation to improve detection performance across different instrument classes.

## Overview

Automatic surgical instrument detection is an important task in computer-assisted surgery and surgical video analysis. Identifying instruments in surgical scenes can provide useful information for understanding surgical workflows, analyzing instrument usage, and developing intelligent surgical assistance systems.

This project implements a multi-class object detection pipeline trained on annotated surgical images. It focuses on five surgical instrument categories and evaluates model performance using standard object detection metrics.

## Objectives

- Detect multiple surgical instruments within a single image.
- Localize each detected instrument using bounding boxes.
- Classify instruments into five predefined categories.
- Investigate the impact of dataset imbalance and augmentation on detection performance.
- Evaluate detection quality using precision, recall, and mean Average Precision (mAP).
- Explore applications of deep learning in surgical scene understanding.

## Surgical Instrument Classes

The model recognizes the following five classes:

| Class ID | Instrument |
|---:|---|
| 0 | Bipolar |
| 1 | Grasper |
| 2 | Hook |
| 3 | Irrigator |
| 4 | Scissors |

The class IDs correspond to the dataset's YOLO annotation mapping.

## Technology Stack

| Technology | Purpose |
|---|---|
| Python | Implementation and experimentation |
| Ultralytics YOLO | Object detection and model training |
| PyTorch | Deep learning framework |
| OpenCV | Image and video processing |
| Roboflow | Dataset preparation and annotation management |
| Google Colab | Model training and GPU experimentation |
| Matplotlib | Training curves and result visualization |

## Dataset

The project uses a surgical instrument object detection dataset prepared and annotated using Roboflow.

The dataset contains images and corresponding bounding-box labels in YOLO format and is divided into training, validation, and test subsets.

### Dataset distribution

| Dataset split | Images | Labels |
|---|---:|---:|
| Training | 1,740 | 1,740 |
| Validation | 370 | 370 |
| Testing | 227 | 227 |
| **Total** | **2,337** | **2,337** |

Dataset source: [Roboflow](https://roboflow.com/)

The project dataset was associated with the Roboflow project `tools-snfsg`. The reported split counts describe the recorded dataset export; subsequent augmentation or balancing may change the effective training data.

### YOLO annotation format

Each image has a corresponding text annotation file. A YOLO bounding-box annotation contains:

```text
class_id x_center y_center width height
```

The coordinates are normalized relative to the image width and height.

Images containing multiple instruments can have multiple annotation lines, one for each labeled object.

## Methodology

The project follows an object detection workflow consisting of dataset preparation, model training, evaluation, and inference.

1. **Dataset preparation:** Organize the surgical images and their bounding-box annotations.
2. **Annotation verification:** Check class IDs, image-label correspondence, and bounding-box validity.
3. **Data augmentation:** Explore transformations and class balancing to address imbalanced instrument representation.
4. **Model training:** Train a YOLO object detection model using the prepared dataset.
5. **Validation:** Evaluate predictions on the validation split and monitor training performance.
6. **Model evaluation:** Measure precision, recall, mAP@50, and mAP@50–95.
7. **Inference:** Apply trained weights to unseen images or surgical videos and visualize the predicted bounding boxes.

## Dataset Balancing and Augmentation

Surgical instrument datasets can contain unequal numbers of examples across classes. Underrepresented classes may be harder to learn, potentially reducing their individual detection performance.

The project investigates balancing and augmentation strategies to improve the representation of surgical instrument categories.

Augmentation should preserve valid bounding-box annotations. For object detection, geometric transformations must update the coordinates of the affected boxes, and any augmentation strategy should be checked for duplicate samples, invalid boxes, and incorrect class mappings.

The experiments compare a baseline configuration with a rebalanced experiment to investigate the relationship between dataset composition and overall detection performance.

## Model Training

The project uses the Ultralytics YOLO framework for multi-class object detection.

Training requires a dataset configuration file, commonly named `data.yaml`, containing the paths to the dataset splits and the class mapping.

Example configuration:

```yaml
train: ../train/images
val: ../valid/images
test: ../test/images

nc: 5

names:
  0: bipolar
  1: grasper
  2: hook
  3: irrigator
  4: scissors
```

The paths must match the actual dataset directory structure.

### Training configuration

The training environment previously used for the project included Google Colab with an NVIDIA Tesla T4 GPU and the Ultralytics YOLO framework.

The exact model variant, epoch count, batch size, image size, optimizer, and learning rate should be taken from the final experiment's training configuration (`args.yaml`) rather than inferred from the dataset configuration.

## Evaluation Metrics

The model is evaluated using standard object detection metrics.

- **Precision:** Measures the proportion of predicted detections that are correct.
- **Recall:** Measures the proportion of ground-truth objects successfully detected.
- **mAP@50:** Mean Average Precision at an IoU threshold of 0.50.
- **mAP@50–95:** Mean Average Precision averaged across IoU thresholds from 0.50 to 0.95.

Intersection over Union (IoU) measures the overlap between a predicted bounding box and its ground-truth bounding box.

A higher mAP@50 indicates stronger detection performance at the 0.50 IoU threshold, while mAP@50–95 provides a more stringent assessment of detection and localization quality.

## Experimental Results

The following results were reported during earlier project experiments.

| Experiment | mAP@50 | mAP@50–95 |
|---|---:|---:|
| Baseline | 96.04% | 58.25% |
| Rebalanced dataset | 94.51% | 57.64% |

These values are from previously reported experimental runs and should not be interpreted as a verified comparison of the final ZIP's model weights until the corresponding run artifacts are checked.

### Results interpretation

The baseline achieved higher reported mAP values than the rebalanced experiment. This illustrates that balancing a dataset does not automatically improve overall detection performance.

The effect of augmentation and balancing should also be assessed using per-class Average Precision, precision, recall, the confusion matrix, and the quality of bounding-box localization. A final comparison should use consistent evaluation settings and the same held-out test set.

## Inference

After training, the saved best-performing weights can be used to detect surgical instruments in new images.

With the Ultralytics Python API:

```python
from ultralytics import YOLO

model = YOLO("path/to/best.pt")

results = model.predict(
    source="path/to/test_image.jpg",
    conf=0.25,
    save=True
)
```

For video inference:

```python
from ultralytics import YOLO

model = YOLO("path/to/best.pt")

results = model.predict(
    source="path/to/surgical_video.mp4",
    conf=0.25,
    save=True
)
```

Replace the paths with the actual locations of the trained weights and input files. The confidence threshold can be adjusted to control which detections are retained.

The output includes predicted classes, confidence scores, and bounding boxes for the detected instruments.

## Applications

Potential applications of surgical instrument detection include:

- Surgical video analysis and scene understanding.
- Instrument localization and usage analysis.
- Surgical workflow recognition.
- Computer-assisted surgical training.
- Research into intelligent and context-aware surgical systems.

## Limitations

- Detection performance depends on annotation quality, dataset diversity, and class distribution.
- Instrument occlusion, motion blur, small objects, and challenging lighting can reduce detection accuracy.
- Performance on one dataset does not guarantee generalization to different hospitals, procedures, or camera systems.
- The reported experimental metrics require validation against the corresponding final model and test configuration.
- This project is intended for educational and research purposes and is not validated for clinical decision-making.

## Future Work

- Compare different YOLO model variants under consistent training and evaluation conditions.
- Improve detection of underrepresented surgical instrument classes.
- Investigate augmentation methods that preserve annotation integrity.
- Evaluate performance on independent surgical datasets.
- Extend image-based detection to real-time surgical video analysis.
- Explore instrument tracking and surgical workflow recognition.

## References and Resources

- [Ultralytics Documentation](https://docs.ultralytics.com/)
- [Roboflow](https://roboflow.com/)
- [OpenCV Documentation](https://docs.opencv.org/)
- [SurgBench Dataset on Hugging Face](https://huggingface.co/datasets/JianhuiWei/SurgBench_NIPS25)

## License and Dataset Attribution

The dataset export was previously associated with a CC BY 4.0 license. Verify the current dataset license and attribution requirements before redistributing images, annotations, or derived datasets.

The code in this repository should have its own clearly specified license if you intend to permit others to reuse it.

## Disclaimer

This project is an academic and research-oriented implementation of deep learning-based surgical instrument detection. It is not intended to replace professional medical judgment or to be used as a clinically validated medical device.
