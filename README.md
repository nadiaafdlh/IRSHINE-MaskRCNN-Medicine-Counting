# IRSHINE-Medicine-Counting-MaskR-CNN
Mask R-CNN-based instance segmentation model for automated pharmaceutical pill counting — part of the IRSHINE undergraduate thesis project.

---
# IRSHINE — Mask R-CNN Medicine Counting & Instance Segmentation

IRSHINE (Irregular Shape Instance Counting System) is an instance segmentation-based system developed to automatically detect, segment, and count medicine objects based on their dosage form.

This repository contains the **Mask R-CNN implementation and experiments** conducted for IRSHINE.

## 🎯 Project Objective

The main objective is to develop a computer vision system capable of:

* Detecting individual medicine objects
* Generating an instance-level segmentation mask for each medicine
* Distinguishing different dosage forms
* Automatically counting medicine instances in an image

The dataset contains three medicine classes:

* `tablet`
* `capsule`
* `softgel`

## 📊 Dataset

The dataset was prepared and annotated using Roboflow and exported in COCO segmentation format.

| Split      |    Images | Instances |
| ---------- | --------: | --------: |
| Train      |     1,098 |     7,110 |
| Validation |       103 |       635 |
| Test       |        53 |       347 |
| **Total**  | **1,254** | **8,092** |

### Instance Distribution

| Class   | Train | Validation | Test | Total |
| ------- | ----: | ---------: | ---: | ----: |
| Capsule | 2,289 |        183 |  123 | 2,595 |
| Softgel | 2,148 |        209 |  103 | 2,460 |
| Tablet  | 2,673 |        243 |  121 | 3,037 |

## 🧠 Why Mask R-CNN?

Mask R-CNN was selected because IRSHINE requires **instance segmentation**, rather than only object detection.

Unlike bounding-box detection, instance segmentation allows the model to identify the individual pixel-level region of each medicine object. This is important when multiple medicines appear close together or overlap.

Mask R-CNN was also used as one of the main models for comparison with YOLOv8m-seg.

---

# 🔑 Key Model Configuration

## 1. ResNet-101 + FPN Backbone

The experiment uses **ResNet-101 with Feature Pyramid Network (FPN)** as the backbone.

* ResNet-101 provides deeper feature extraction than ResNet-50
* Higher feature capacity can help distinguish visually similar medicine objects
* FPN enables feature extraction at multiple scales
* COCO pretrained weights are used to benefit from transfer learning

The deeper backbone was selected as part of the final Mask R-CNN experiment to improve feature representation for the medicine dataset.

## 2. AdamW Optimizer

AdamW was used instead of the default SGD optimizer.

The main considerations were:

* Adaptive parameter updates
* Faster adaptation of newly initialized detection/segmentation heads
* Decoupled weight decay
* Better regularization during fine-tuning

The optimizer configuration was combined with a relatively small learning rate to reduce the risk of overfitting during transfer learning.

## 3. Cosine Annealing Warm Restarts

A `CosineAnnealingWarmRestarts` learning-rate scheduler was used.

Configuration:

```text
T0 = 30 epochs
T_mult = 2
```

The learning rate gradually decreases and periodically restarts to a higher value.

This was used to provide additional opportunities for optimization during the 150-epoch training process.

## 4. Input Resolution — 1024 × 1024

The input resolution was increased to **1024 × 1024**.

This configuration was chosen because the dataset contains relatively small medicine objects that can appear close together or overlap.

Higher resolution provides more spatial information for:

* Small object detection
* Object boundary identification
* Instance mask generation

The trade-off is higher GPU memory usage and longer training time.

## 5. Mosaic Augmentation

Mosaic augmentation was incorporated into the training pipeline.

Four training images are combined into a single image, increasing the variation of object:

* Position
* Scale
* Density
* Spatial arrangement

This is particularly relevant for a medicine counting task where multiple objects can appear in different arrangements.

## 6. Training

The final experiment was trained for:

```text
Epochs      : 150
Backbone   : ResNet-101 + FPN
Optimizer  : AdamW
Scheduler  : CosineAnnealingWarmRestarts
Input Size : 1024 × 1024
Augmentation: Mosaic
```

Checkpoints, training logs, and experiment results were saved throughout the training process.

---

# 🔄 Training Pipeline

```
1. Environment Check
2. Install Dependencies
3. Load Dataset from Roboflow
4. Dataset Analysis
5. Dataset & DataLoader
6. Mosaic Augmentation
7. ResNet-101 + FPN
8. AdamW Optimizer
9. CosineAnnealingWarmRestarts
10. 150-Epoch Training
11. Model Evaluation
12. Best Model Selection
13. TTA Inference
14. Medicine Detection + Instance Segmentation
15. Object Counting
```

## 📈 Evaluation


The model is evaluated using multiple metrics to assess both segmentation and detection performance:

* **mAP** = Overall detection/segmentation performance
* **IoU** = Overlap between predicted and ground-truth masks
* **Precision** = Correctness of positive predictions
* **Recall** = Completeness of detected objects
* **F1-Score** = Balance between precision and recall
* **Accuracy** = Additional classification/detection performance measure


The evaluation pipeline also includes visualization and custom-image inference.

---

# 🔬 Experimentation

The Mask R-CNN notebook contains multiple experiments, with the final experiment using:

```text
ResNet-101 + FPN
AdamW
CosineAnnealingWarmRestarts
Mosaic Augmentation
1024 × 1024 input
150 epochs
```

The notebook also compares the training behavior and results of previous experiments to determine the final configuration.

---

# ⚖️ Comparison with YOLOv8m-seg

Mask R-CNN was compared with **YOLOv8m-seg** as two different approaches to instance segmentation.

| Aspect                | Mask R-CNN                             | YOLOv8m-seg           |
| --------------------- | -------------------------------------- | --------------------- |
| Architecture          | Two-stage                              | One-stage             |
| Instance Segmentation | ✓                                      | ✓                     |
| Backbone              | ResNet-101 + FPN                       | YOLOv8 architecture   |
| Main Strength         | Detailed feature & mask representation | Speed and efficiency  |
| Inference             | Relatively slower                      | Faster                |
| IRSHINE Role          | Accuracy-oriented comparison           | Accuracy + efficiency |

The comparison helps evaluate the trade-off between **segmentation performance and computational efficiency** for the IRSHINE use case.

For the YOLOv8m-seg implementation, see the corresponding IRSHINE YOLOv8m-seg repository.

---

# 🛠️ Technologies

* Python
* PyTorch
* Torchvision
* OpenCV
* Albumentations
* Roboflow
* pycocotools
* scikit-learn
* Matplotlib
* Seaborn
* Google Colab

## 📁 Notebook Structure

```text
CELL 1  → Environment Check
CELL 2  → Install Dependencies
CELL 3  → Google Drive Setup
CELL 4  → Dataset Download
CELL 5  → Dataset Analysis
CELL 6  → Dataset & DataLoader
CELL 7  → Model Configuration
CELL 8  → Training Utilities
CELL 9  → 150-Epoch Training
CELL 10 → Experiment Visualization
CELL 11 → Best Model + TTA Inference
CELL 12 → Final Evaluation
CELL 13 → Custom Image Inference
```

## 🚀 How to Run

1. Open the notebook in Google Colab.
2. Enable GPU runtime.
3. Install the required dependencies.
4. Configure the dataset source.
5. Run the notebook sequentially.
6. Train or load the trained Mask R-CNN model.
7. Run evaluation and custom-image inference.

> **Note:** API keys and other credentials should be stored securely and should not be committed to a public GitHub repository.
