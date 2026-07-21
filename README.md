# VisionGuard: Diabetic Retinopathy Grading via Knowledge Distillation

VisionGuard is an on-device diabetic retinopathy (DR) grading system designed for deployment on low-resource Android hardware, aimed at making DR screening more accessible in settings without specialist ophthalmology equipment or reliable connectivity.

Diabetic retinopathy is a leading cause of preventable blindness, and early grading of severity from a retinal fundus image is critical for timely treatment. Most high-accuracy DR grading models are large and require cloud inference, which is impractical in low-connectivity or low-resource clinical settings. VisionGuard addresses this by using **knowledge distillation** to compress a large, high-capacity EfficientNet-B4 teacher model into a lightweight MobileNetV3-Small student — small enough to run directly on a mid-range Android phone via ONNX Runtime, without sacrificing grading accuracy.

Beyond the core model, this repo documents a **cross-device robustness suite** (four modules addressing calibration, preprocessing, domain shift, and out-of-distribution detection) built to handle a real-world deployment problem: fundus images vary significantly across different camera hardware, and models trained on one dataset often degrade badly on images from another. The repo also includes **honest failure reporting** — several architectural and augmentation ideas were tested and rejected during development, and those negative results are documented here rather than omitted, in the interest of giving an accurate research narrative.

## Problem Statement

- **Task:** Grade diabetic retinopathy severity from a retinal fundus photograph into standard clinical severity classes (No DR, Mild, Moderate, Severe, Proliferative DR).
- **Constraint:** Inference must run on-device (Android), not in the cloud, to support low-connectivity clinical settings.
- **Challenge:** Models trained on one fundus camera/dataset often fail to generalize to images captured on different hardware (domain shift) — a major real-world deployment barrier that this project explicitly investigates and addresses.

## Key Results

| Model | QWK (Quadratic Weighted Kappa) | Notes |
|---|---|---|
| EfficientNet-B4 (Teacher) | ~0.90 | Trained on APTOS 2019 |
| MobileNetV3-Small (Student) | **0.9103 – 0.9112** | Distilled from teacher; outperforms teacher despite being ~10x smaller in parameter count |

**Why QWK?** Quadratic Weighted Kappa is the standard metric for ordinal medical grading tasks like this one (also the official metric for the original APTOS Kaggle competition). Unlike plain accuracy, QWK penalizes predictions in proportion to how far off they are from the true grade — e.g., predicting "Severe" when the true label is "No DR" is penalized far more heavily than predicting "Mild" when the true label is "No DR." This makes it a clinically meaningful metric, since the cost of a misdiagnosis scales with how wrong it is.

- **Deployment:** Exported to ONNX and run via **ONNX Runtime** on Android (tested on a Redmi Note 14 5G)
- **Dataset:** [APTOS 2019 Blindness Detection](https://www.kaggle.com/c/aptos2019-blindness-detection) (primary training/validation set); EyePACS used for cross-domain testing (see Failure Reporting below)

## Architecture

**Teacher model:** EfficientNet-B4 — a high-capacity convolutional network chosen for strong baseline accuracy on fundus images, at the cost of being too large/slow for on-device inference.

**Student model:** MobileNetV3-Small — a mobile-optimized architecture chosen specifically for its small footprint and low latency, making it suitable for real-time on-device inference.

**Knowledge distillation:** The student is trained using a combination of (a) the standard hard-label classification loss against ground truth grades, and (b) a distillation loss that encourages the student's output distribution to match the teacher's "soft" probability outputs. This transfers the teacher's learned decision boundaries into a far smaller model. Notably, the distilled student **outperformed** its own teacher on held-out validation data — a result worth highlighting, since it suggests the distillation process itself acted as a form of regularization.

**Preprocessing pipeline** (applied before both training and inference):
1. Green-channel extraction — isolates the color channel with the highest contrast for retinal vasculature and lesions
2. CLAHE (Contrast Limited Adaptive Histogram Equalization) — enhances local contrast without over-amplifying noise
3. Ben Graham crop — crops away irrelevant black borders around the circular fundus image
4. Resize to 384×384 — standardizes input size for the network

**Export pipeline:** Trained PyTorch model → ONNX format → deployed via ONNX Runtime Mobile for on-device Android inference.

### Cross-Device Robustness Suite (4 Modules)

A key real-world problem for medical imaging models is that a model trained on images from one fundus camera often performs significantly worse on images from a different camera/device, due to differences in color balance, resolution, lighting, and lens characteristics — this is "domain shift." To make VisionGuard robust to this, four modules were designed and tested:

1. **Calibration fine-tuning** — A lightweight fine-tuning step that adapts the trained model to a small sample of images from a new target device, without requiring full retraining.
2. **Preprocessing calibration** — Device-specific normalization applied during preprocessing, to reduce visual discrepancies between source and target devices before the image even reaches the model.
3. **Domain adaptation (DANN-style)** — An adversarial domain adaptation approach (inspired by Domain-Adversarial Neural Networks), where the model is trained to extract features that are useful for DR grading but not predictive of which source device/dataset the image came from — encouraging domain-invariant representations.
4. **OOD (Out-of-Distribution) detection** — Uses MC-Dropout (Monte Carlo Dropout: running multiple stochastic forward passes at inference time and measuring prediction variance) to estimate the model's uncertainty on a given input. If uncertainty is too high, the input is flagged as out-of-distribution rather than given a potentially unreliable grade. This module was validated on real APTOS data at **QWK 0.9112**, and further tested with negative test cases specifically designed to confirm the safety gate correctly identifies and rejects inputs it shouldn't confidently grade (e.g. corrupted, blurry, or clearly non-fundus images) — an important safety property for any model intended for clinical-adjacent use.

## Honest Failure Reporting

A core principle of this project is to document what *didn't* work alongside what did — negative results are scientifically useful and are reported here rather than hidden:

- **Domain shift (EyePACS mixing):** When EyePACS data was mixed into training/evaluation alongside APTOS, QWK dropped sharply from ~0.91 to **~0.58**. This was a significant finding: it confirmed that the model, despite strong in-distribution performance, does not generalize well across datasets/cameras out of the box. This result directly motivated the design of the 4-module cross-device robustness suite described above — rather than treating domain shift as a footnote, it became a primary research focus.

- **MSAG (Multi-Scale Attention Gates):** An attempt to improve the model by adding multi-scale attention gating did **not** improve grading performance over the baseline architecture. This is recorded as a negative result — a useful data point for anyone considering the same approach, and evidence that added architectural complexity isn't automatically beneficial for this task.

- **cGAN-based data augmentation:** Conditional GANs were explored as a way to synthetically augment underrepresented DR severity classes (the APTOS dataset is class-imbalanced, with far fewer severe/proliferative examples). This approach suffered from **mode collapse** — the generator began producing near-identical, low-diversity synthetic images — and was ultimately abandoned in favor of standard augmentation techniques (rotation, flipping, color jitter, etc.).

- **Preprocessing inconsistency (flagged, not yet resolved):** A discrepancy was identified between the teacher and student training notebooks: the teacher notebook applies circle-cropping *before* green-channel extraction, while the student notebook does not apply circle-cropping at all before this step. This is flagged here as a known issue for future correction, since consistent preprocessing between teacher and student is important for clean distillation.

## Repository Structure

```
visionguard-dr-grading/
├── README.md
├── requirements.txt
├── notebooks/
│   ├── 01_teacher_efficientnet_b4.ipynb
│   ├── 02_student_distillation.ipynb
│   ├── 03_module1_calibration_finetuning.ipynb
│   ├── 04_module4_ood_detection.ipynb
│   └── ...
├── src/
│   ├── preprocessing.py
│   ├── onnx_export.py
│   └── android_inference/
└── docs/
    └── architecture_diagram.png
```

## Reproducing This Work

1. Clone the repo and install dependencies:
   ```bash
   git clone https://github.com/AnanthKumar1810/visionguard-dr-grading.git
   cd visionguard-dr-grading
   pip install -r requirements.txt
   ```
2. Download the [APTOS 2019 dataset](https://www.kaggle.com/c/aptos2019-blindness-detection) from Kaggle.
3. Run notebooks in order starting with `01_teacher_efficientnet_b4.ipynb`.

Trained model weights are not stored in this repository (see `.gitignore`). [Add link here once weights are hosted — e.g. Hugging Face Hub, Kaggle Datasets, or GitHub Releases.]

## Status

Actively developed as part of ongoing DR grading research, with results being prepared for submission (targeting IEEE EMBC / MICCAI).

## Author

**S. Ananth Kumar (Siddhu)**
CSE (AI/ML), MVGR College of Engineering, Vizianagaram
GitHub: [@AnanthKumar1810](https://github.com/AnanthKumar1810)
Kaggle: [s567890](https://www.kaggle.com/s567890)
