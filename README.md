# ARECANUT DETECTION THROUGH SATELLITE IMAGERY

A Machine Learning and Deep Learning project designed to detect arecanut plantations from satellite imagery using multiple computer vision techniques, traditional ML models, and YOLO-based object detection.

This project includes:
• Statistical detection methods
• Feature-based classification
• Deep learning–based object detection
• Full dataset preparation
• Complete training notebooks and results
• Full project documentation (PDF)


# 1. PROJECT OVERVIEW

This project aims to detect arecanut plantation regions from satellite images.
The system uses a multi-approach strategy involving:

1. Classical ML models (SVM, Random Forest, Logistic Regression)
2. Feature extraction (RGB, Grayscale, GLCM, LBP, Histograms)
3. Deep learning (CNN, MobileNetV2)
4. YOLOv8 object detection

The project successfully identifies plantation regions (not individual trees) due to the resolution limits of satellite imagery.


# 2. FEATURES

• Multi-approach detection system
• Grayscale, RGB, texture-based feature extraction
• Machine learning classification (SVM, RF, LR, KNN)
• Deep learning object detection using YOLOv8
• Data preprocessing: image tiling, normalization, splitting
• Model training pipelines with evaluation metrics
• Complete visualization of outputs (bounding boxes, accuracy graphs)
• 108-page project documentation included


# 3. PROJECT DIRECTORY STRUCTURE

Arecanut_Detection_Satellite_Imagery
│
├── Arecanut_Detection/
│   ├── Model_Statistical_Detection/
│   │   ├── Arecanut_Detection.ipynb
│   │   ├── reference_images/
│   │   │   ├── positive/
│   │   │   └── negative/
│   │   ├── test_images/
│   │   ├── using_logistic_regression/
│   │   ├── using_random_forest/
│   │   └── using_svm/
│   │
│   └── YOLO_Detection/
│       ├── Training_YOLO_Model.ipynb
│       ├── Testing_YOLO_Model.ipynb
│       ├── data.yaml
│       ├── images/
│       ├── labels/
│       ├── train/
│       ├── val/
│       ├── test/
│       ├── Results/
│       └── YOLO_training_results/
│
├── Classification/
│   ├── Demo_Classification/
│   │   ├── grayscale_feature/
│   │   ├── RGB_feature/
│   │   ├── Image/
│   │   └── splitted_images/
│   │
│   └── Trial_Classification/
│       ├── classification_using_models.ipynb
│       ├── grayscale_feature/
│       ├── RGB_feature/
│       ├── Image/
│       ├── manually_labelled_images/
│       └── splitted_images/
├── README.txt
├── requirement.txt
└── Arecanut_Detection_with_ML.pdf     (Full 108-page project report)


# 4. GETTING STARTED

# 4.1 REQUIREMENTS

• Python 3.7+
• Jupyter Notebook
• Google Colab (recommended for GPU training)
• Necessary Python libraries listed in requirements.txt

# 4.2 INSTALLATION

1. Clone the repository:
   git clone [https://github.com/Prajnavi/Arecanut_Detection_Satellite_Imagery.git](https://github.com/Prajnavi/Arecanut_Detection_Satellite_Imagery.git)
   cd Arecanut_Detection_Satellite_Imagery

2. Install dependencies:
   pip install -r requirements.txt

3. Launch Jupyter Notebook:
   jupyter notebook

# 5. METHODS IMPLEMENTED

# 5.1 STATISTICAL DETECTION METHODS

Support Vector Machine (SVM)
• Features used: LBP + HSV histogram
• Works with sliding window detection
• Good for texture-heavy images

Random Forest
• Uses texture + color features
• Robust in different lighting conditions
• Handles nonlinear patterns well

Logistic Regression
• Baseline classifier
• Uses grayscale statistical features
• Suitable for interpretability

# 5.2 DEEP LEARNING (YOLO)

YOLOv5 / YOLOv8
• Used for object detection
• Trained on 400+ labeled image tiles
• Outputs bounding boxes for plantation regions
• Training notebooks included
• Performance metrics saved in YOLO_training_results/


# 6. FEATURE EXTRACTION

# 6.1 RGB Feature Extraction

• Color histograms
• RGB channel statistics
• CSV output for model training

# 6.2 Grayscale Feature Extraction

• GLCM (Contrast, Energy, Correlation, Homogeneity)
• LBP
• Edge density
• Used for binary classification (arecanut vs non-arecanut)


# 7. RESULTS & PERFORMANCE

# 7.1 Statistical Models

• SVM: High accuracy with texture features
• Random Forest: Most robust classical model
• Logistic Regression: Clean baseline performance

# 7.2  YOLO Detection

• Training split: 80% train / 20% validation
• Metrics measured: mAP, Precision, Recall
• Results stored in YOLO_training_results/

# 7.3  Classification

• RGB vs Grayscale feature comparison
• Model comparison notebook included

Actual detailed results are available in the PDF report.


# 8. USAGE

# 8.1 Running Statistical Detection

1. Open the notebook:Arecanut_Detection_Satellite_Imagery/Arecanut_Detection/Model_Statistical_Detection/Arecanut_Detection.ipynb
2. Update dataset paths
3. Run complete pipeline

# 8.2 Training YOLO

1. Open Training_YOLO_Model.ipynb
2. Load your dataset
3. Run the training cells

# 8.3 Feature Extraction

Navigate to Classification folders to run grayscale/RGB feature extraction.


# 9. DATASET DETAILS

• 400+ satellite images
• YOLO-format bounding box annotations
• 1 class: Arecanut plantation
• Preprocessing includes:
	• Image cropping
 	• Image tiling
 	• Manual annotation
 	• Train/validation/test splits


# 10. YOLO CONFIGURATION (data.yaml)

train: path/to/train/images
val: path/to/val/images
test: path/to/test/images
nc: 1
names:
arecanut


# 11. MODEL COMPARISON TABLE

| Method              | Accuracy            | Precision   | Recall   | F1-Score  |
|---------------------|---------------------|-------------|----------|-----------|
| SVM                 | 65.22%              | 0.65        | 1.00     | 0.79      |
| Random Forest       | 78.26%              | 0.75        | 1.00     | 0.86      |
| Logistic Regression | 82.61%              | 0.79        | 1.00     | 0.88      |
| YOLOv8              | — (Detection Model) | 0.938       | 0.864    | 0.82      |


# 12. AUTHOR INFORMATION

Author: Prajnavi
Role: ML Model Development, Data Processing, Image Classification
Email: [prajnavimuniyal@gmail.com]


# 13. ACKNOWLEDGMENTS

• Google Earth Pro – Satellite image source
• YOLO community – Model architecture
• Open-source ML and CV libraries
• Faculty and mentors for guidance
• Other Project Contributors [ Tulasi Shetty and Monisha S]


# 14. FINAL NOTE

This project showcases the power of machine learning and deep learning for agricultural monitoring and satellite-based crop detection. It combines academic research with practical implementation, delivering a scalable approach for detecting arecanut plantations.s