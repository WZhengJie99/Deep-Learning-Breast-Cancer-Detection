<div align="center">

# Deep Learning Breast Cancer Detection

**Exploring Deep Learning models for enhancing Breast Cancer diagnosis using Mammography Images**

![Python](https://img.shields.io/badge/python-3.10.18-blue.svg) ![TensorFlow](https://img.shields.io/badge/TensorFlow-2.19.0-orange.svg) ![Keras](https://img.shields.io/badge/Keras-3.10.0-red.svg)  

<p style="display:flex; gap:10px; align-items:flex-start;">

  <img style="height:400px; object-fit:cover;" alt="A_0005_1 RIGHT_CC" src="https://github.com/user-attachments/assets/551ce974-b2d7-4a5c-87b7-15bcba048682" />
  <img style="height:400px; object-fit:cover;" alt="A_1276_1 RIGHT_CC" src="https://github.com/user-attachments/assets/95ed2c5e-02ab-4ef9-bc26-fe79fffd3a36" />
  <img style="height:400px; object-fit:cover;" alt="A_1399_1 RIGHT_CC" src="https://github.com/user-attachments/assets/b677b356-0c46-4eb0-b1b3-374136a9bc2f" />

</p>

</div>

## Table of Contents

* [Project Overview](#project-overview)
* [Dataset: Mini DDSM](#dataset-mini-ddsm)
* [Models Implemented](#models-implemented)
* [Results Summary](#results-summary)
* [Multi-Branch Model Architecture](#multi-branch-model-architecture)
* [Future Directions](#future-directions)
* [Installation and Running](#installation-and-running)
* [Using the UI](#using-the-ui)
* [Disclaimer](#disclaimer)
* [References](#references)

---

## Project Overview

Breast cancer ranks among the top causes of death for women worldwide, making early and precise diagnosis essential for survival.
This project investigates **deep learning methods for breast cancer detection using mammography images**, utilizing the **Mini-DDSM dataset** along with several CNN-based approaches such as transfer learning and hybrid multi-branch architectures.
Although the models are not yet suitable for clinical use, the findings shed light on promising research directions.

- **Dataset**: [Mini-DDSM](https://www.kaggle.com/datasets/cheddad/miniddsm2)
- **Objective**: Improving screening accuracy, classifying mammograms into **Benign**, **Cancer**, and **Normal** for early treatment
- **Workflow**: Based on François Chollet’s *Deep Learning Workflow*
- **UI**: Simple Flask-based app for testing predictions locally

---

## Dataset: Mini DDSM

- Kaggle Source: https://www.kaggle.com/datasets/cheddad/miniddsm2

- Reference paper/Mini-DDSM: C.D. Lekamlage, F. Afzal, E. Westerberg and A. Cheddad, “Mini-DDSM: Mammography-based Automatic Age Estimation,” in the 3rd International Conference on Digital Medicine and Image Processing (DMIP 2020), ACM, Kyoto, Japan, November 06-09, 2020, pp: 1-6. https://doi.org/10.1145/3441369.3441370

- Reference DDSM: Michael Heath, Kevin Bowyer, Daniel Kopans, Richard Moore and W. Philip Kegelmeyer, in Proceedings of the Fifth International Workshop on Digital Mammography, M.J. Yaffe, ed., 212-218, Medical Physics Publishing, 2001. ISBN 1-930524-00-5

---

## Models Implemented

- **Naive Baseline CNN** – simple reference model
- **Baseline CNN** – batch normalization, L2 regularization, Global Average Pooling
- **Overfitting Model** – tests complexity limitations
- **Regularized Model** – mititgate overfitting
- **Wider, Deeper, Narrower CNNs** – structural variations
- **Transfer Learning** - EfficientNetB0
- **Hybrid Multi-Branch Model** – Narrower CNN + EfficientNetB0 (best-performing)
- **Modified U-Net** – mask segmentation-based approach

---

## Results Summary

<div align="center"><img alt="Model Evaluation Results Comparison" src="https://github.com/user-attachments/assets/d39782ad-19ce-464f-8874-fc99ba52ca06" /></div>


**Findings:**
- Narrower architecture improved generalization compared to Wider and Deeper models.
- Transfer learning alone tended to misclassify other classes as Cancer.
- Multi-branch hybrid was the most balanced performer.
- Prediction for cancer class requires advanced segemntation for masks to identify subtle ROIs.

---

## Multi-Branch Model Architecture

The **Multi-Branch Model Architecture** was the most effective model in this project.
It combines the strengths of a **custom Narrower CNN** and a **pre-trained EfficientNetB0** transfer learning branch.

### Architecture Overview

<div align="center">
<img width="90%" alt="3-Branch Model Architecture" src="https://github.com/user-attachments/assets/dbf13c82-a8bf-4fab-922a-0681c4cc2418" />
</div>

- **Branch 1**: Narrower CNN
  - Better generalization under dataset constraints.
  - Classifies Benign and Normal class.
  - convolution, pooling, and dropout layers.
    
- **Branch 2**:
  - Leverages pre-trained knowledge from ImageNet.
  - Transfer Learning EfficientNetB0 for Cancer class.
  - Training is done in two phases: first freezing EfficientNet, then fine-tuning the top layers for improved performance.
  
- **Concatonation**:
  - Concatenated feature maps → Dense layers → Softmax activation

### Performance
- Accuracy: **65%**
- F1 (Normal): **0.82**
- F1 (Benign): **0.54**
- F1 (Cancer): **0.60**

---

## Future Directions

- Finding better models for each class (Benign, Cancer, Normal) to improve multi branch accuracy.
- Implement and explore advanced segmentation procedures on cancer class and masks to improve identification of cancer calcifications.
- Highlight breast cancer localizations if cancer is detected.
- State cancer stage present and address reasons for cancer biomarkers.

---

## Installation and Running

### Prerequisites  

Before running the UI or training models, ensure the following dependencies are installed:  

- **Python**: v3.10.18  
- **TensorFlow**: v2.19.0  
- **Keras**: v3.10.0  

### 1. Clone the Repository

```
git clone https://github.com/WZhengJie99/Deep-Learning-Breast-Cancer-Detection.git
cd Deep-Learning-Breast-Cancer-Detection
```

### 2. Dataset Setup

- Download Mini-DDSM dataset.
- Place dataset in the project directory (or update paths in the code).

### 3. Running Models

Jupyter notebooks (.ipynb) contain pre-processing steps and implementations.

### 4. Running UI

Execute `WZJ_Cancer_Predictor.bat` or run:

```
python app.py
```

Then open: http://127.0.0.1:5000/

---

## Using the UI

>[!NOTE]
>The UI uses the Narrower and Transfer Learning Multi-Branch Model Architecture for its prediction.

>[!IMPORTANT]
>This application is meant solely for educational, research, and demonstration use. It is not a medical instrument, and its forecasts should not be viewed as a clinical assessment.

<div align="center">
<img width="90%" alt="UI App localhost Implementation Secondary Image" src="https://github.com/user-attachments/assets/b0b2a3f5-b42d-40c6-9413-d758b8d1d3c0" />
</div><br>

1. Upload a Mammogram Image (PNG or JPEG) by clicking `Upload Image`.
2. Click on `Predict` for the model to process the image and perform classification.
3. View results, the UI will display the predicted class and a confidence score (probability) for each class: Normal, Benign, or Cancer.

---

## Disclaimer

- This project and application is meant solely for educational, research, and demonstration use. It is not yet suitable as a medical instrument, and its forecasts should not be viewed as a clinical assessment.

---

## References  

- Mini-DDSM Dataset and Mammogram Images used – Cheddad, A. (2021) The Complete Mini-DDSM [Data set]. Kaggle. [Online] Available at: https://www.kaggle.com/datasets/cheddad/miniddsm2 (Accessed: 2025).
- Lekamlage, C.D., Afzal, F., Westerberg, E. and Cheddad, A. (2020) ‘Mini-DDSM: Mammography-based Automatic Age Estimation’, in Proceedings of the 3rd International Conference on Digital Medicine and Image Processing (DMIP 2020), Kyoto, Japan, 6–9 November. New York: ACM. [Online] Available at: https://arxiv.org/pdf/2010.00494v1 (Accessed: 2025).
- Heath, M., Bowyer, K., Kopans, D., Moore, R. and Kegelmeyer, W.P. (2001) ‘The Digital Database for Screening Mammography’, in Yaffe, M.J. (ed.) Proceedings of the Fifth International Workshop on Digital Mammography. Madison, WI: Medical Physics Publishing, pp. 212–218. [Online] Available at: http://www.eng.usf.edu/cvprg/Mammography/software/HeathEtAlIWDM_2000.pdf (Accessed: 2025).

- François Chollet *Deep Learning with Python* - Chollet, F. (2017) Deep learning with Python. Shelter Island, NY: Manning Publications. ISBN: 9781617294433. [Online] Available at:	 https://www.manning.com/books/deep-learning-with-python (Accessed: 2025).
- zohyan (2019) François Chollet's Deep Learning with Python: The universal workflow of machine learning. GitHub. [Online] Available at: https://github.com/zohyan/Deep-Learning-with-python/blob/master/4_5_The_universal_workflow_of_machine_learning.ipynb (Accessed: 2025).

- Modified U-Net for Mammography - Hossain, M.S., (2022). Microc alcification segmentation using modified u-net segmentation network from mammogram images. Journal of King Saud University-Computer and Information Sciences, 34(2), pp.86-94. Available at: https://doi.org/10.1016/j.jksuci.2019.10.014 (Accessed: 2025).


---

