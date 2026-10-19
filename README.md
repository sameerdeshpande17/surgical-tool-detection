# surgical-tool-detection

## Overview

Surgical Tool Recognition Using Deep Learning is a computer vision project that detects and identifies surgical instruments in laparoscopic surgical images using YOLO-based object detection.

The objective of this project is to develop an automated system that can recognize multiple surgical instruments in a single image, supporting surgical video analysis and potential applications in computer-assisted surgery.

I developed this project using the M2CAI16 Tool Detection dataset and trained an object detection model using the Ultralytics YOLO framework. I also explored dataset balancing and augmentation to study their impact on detection performance.

## Objectives

- Detect multiple surgical instruments in a single image.
- Train a deep learning model for surgical instrument detection.
- Evaluate the model using precision, recall, mAP@50, and mAP@50–95.
- Investigate the effect of dataset balancing and augmentation on model performance.
- Explore potential applications in surgical video analysis and computer-assisted systems.

## Model Used

I used **YOLO11m**, initialized with the pretrained `yolo11m.pt` weights, for surgical instrument detection.

YOLO is an object detection framework that predicts object locations using bounding boxes and assigns class labels to detected objects. Its ability to detect multiple objects in one image makes it suitable for recognizing surgical instruments that may appear together in a surgical scene.

## Dataset

I used the **M2CAI16 Tool Detection dataset** for training and evaluating the model. The dataset was prepared in YOLO-compatible format, with images and corresponding bounding-box annotations.

### Instrument Classes

The model recognizes five surgical instrument classes:

1. **Bipolar**
2. **Grasper**
3. **Hook**
4. **Irrigator**
5. **Scissors**

### Dataset Distribution

The dataset was divided into training, validation, and test sets.

| Dataset split | Number of images |
|---|---:|
| Training | 1,740 |
| Validation | 370 |
| Testing | 227 |
| **Total** | **2,337** |

The training set was used to optimize the model, the validation set to monitor performance during development, and the held-out test set to evaluate the trained model.

## Methodology

### 1. Dataset Preparation

I prepared the dataset in YOLO format, including images, class labels, and bounding-box annotations. The dataset was divided into training, validation, and test subsets.

### 2. Model Training

I initialized YOLO11m using pretrained weights and configured the model to detect the five surgical instrument classes. The model was trained using the prepared dataset.

### 3. Dataset Balancing and Augmentation

I explored balancing and augmentation to improve the representation of underrepresented instrument classes. The augmented experiment targeted 1,055 annotated instances per class.

This experiment allowed me to compare the original training approach with a balanced and augmented approach.

### 4. Model Evaluation

I evaluated the detection models using the following metrics:

- **Precision:** The proportion of predicted detections that were correct.
- **Recall:** The proportion of ground-truth objects that were detected.
- **mAP@50:** Mean Average Precision at an Intersection over Union (IoU) threshold of 0.50.
- **mAP@50–95:** Mean Average Precision averaged across IoU thresholds from 0.50 to 0.95.

## Experimental Results

I compared the baseline model with the model trained using the balanced and augmented dataset. The recorded evaluation results are shown below.

| Metric | Baseline Model | Balanced/Augmented Model |
|---|---:|---:|
| Precision | 93.10% | 92.16% |
| Recall | 95.00% | 91.63% |
| mAP@50 | 96.04% | 94.51% |
| mAP@50–95 | 58.25% | 57.64% |

The baseline model achieved an **mAP@50 of 96.04%** and an **mAP@50–95 of 58.25%**. It also achieved higher precision and recall than the balanced and augmented experiment.

Although balancing and augmentation were intended to improve class representation, they did not improve the overall reported detection metrics in this comparison. This highlights the importance of evaluating data preparation strategies rather than assuming that a more balanced dataset will always produce better results.

These metrics describe object detection performance and should not be interpreted as conventional classification accuracy.

## Technologies Used

- Python
- PyTorch
- Ultralytics YOLO
- OpenCV
- Roboflow
- Google Colab
- NumPy
- Matplotlib

## Installation

Install the required packages:

```bash
pip install ultralytics opencv-python numpy matplotlib
```

## Training

The following is an example of how to initialize YOLO11m and train it using a YOLO-format dataset configuration:

```python
from ultralytics import YOLO

model = YOLO("yolo11m.pt")

model.train(
    data="data.yaml",
    epochs=100,
    imgsz=640,
    batch=16
)
```

The training parameters shown above are an example configuration. They should be replaced with the actual parameters used in a particular experiment when reproducing the recorded results.

The `data.yaml` file must specify the paths to the training, validation, and test datasets, along with the number of classes and their names.

## Inference

After training, a saved model can be used to detect surgical instruments in an image:

```python
from ultralytics import YOLO

model = YOLO("best.pt")

results = model.predict(
    source="test_image.jpg",
    conf=0.25,
    save=True
)
```

The model produces bounding boxes, predicted class labels, and confidence scores for detected instruments. The confidence threshold can be adjusted depending on the application.

## Challenges and Observations

During development, I explored several challenges associated with surgical instrument detection:

- **Class imbalance:** Some instrument classes have fewer annotated examples than others.
- **Small or overlapping instruments:** Instruments may be partially visible, overlap with other objects, or appear at different scales.
- **Dataset augmentation:** Increasing the number of samples does not automatically improve generalization.
- **Evaluation:** Precision, recall, and mAP provide complementary information about detection performance.

The comparison between the baseline and augmented experiments reinforced the importance of consistent evaluation and careful selection of training strategies.

## Applications

Potential applications of this project include:

- Surgical video analysis
- Instrument detection and localization
- Computer-assisted surgery research
- Surgical workflow analysis
- Development of intelligent operating-room systems
- Medical computer vision research

This project is intended for research and educational purposes. It is not a clinically validated system and should not be used to guide medical decisions.

## Future Work

Possible improvements include:

- Evaluating different YOLO model sizes and architectures.
- Improving detection of underrepresented instrument classes.
- Experimenting with class-aware augmentation strategies.
- Analyzing per-class precision, recall, and average precision.
- Evaluating performance on longer surgical videos.
- Exploring real-time inference and deployment.
- Testing generalization on additional surgical datasets.

## References

1. CAMMA — Surgical datasets and research resources: https://camma.unistra.fr/datasets/
2. Ultralytics — YOLO11 documentation: https://docs.ultralytics.com/models/yolo11/
3. Roboflow — Dataset preparation and annotation tools: https://roboflow.com/

---

**Project:** Surgical Tool Detection

**Model:** YOLO11m  
**Framework:** Ultralytics YOLO  
**Number of classes:** 5  
**Primary evaluation metric:** mAP@50

