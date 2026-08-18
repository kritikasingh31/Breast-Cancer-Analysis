# ML Assignment - 2: Classification Model Comparison

## a. Problem Statement
The objective is to implement and compare multiple classification models on the same public classification dataset. The models are evaluated using Accuracy, AUC, Precision, Recall, F1 Score and Matthews Correlation Coefficient (MCC), followed by development and deployment of an interactive Streamlit application.

## b. Dataset Description
**Dataset:** Breast Cancer Wisconsin (Diagnostic)  
**Original repository:** UCI Machine Learning Repository; the same dataset is distributed reproducibly by scikit-learn.

- Instances: 569
- Features: 30
- Type: Binary classification
- Target: malignant / benign
- Train/test split: 80% / 20%
- Random state: 42

This satisfies the assignment minimum of 12 features and 500 instances.

## c. GitHub Repository Link
**Replace with your actual repository URL:** `https://github.com/kritikasingh31/Breast-Cancer-Analysis.git`

## d. Models Used
The assignment requires six models but explicitly lists five named models. Therefore, Support Vector Machine (SVM) is used as the sixth model for the required six-model comparison.

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbor (kNN)
4. Gaussian Naive Bayes
5. Random Forest (Ensemble)
6. Support Vector Machine (SVM)

### Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9649 | 0.9960 | 0.9595 | 0.9861 | 0.9726 | 0.9245 |
| Decision Tree | 0.9211 | 0.9448 | 0.9091 | 0.9722 | 0.9396 | 0.8299 |
| kNN | 0.9561 | 0.9825 | 0.9467 | 0.9861 | 0.9660 | 0.9058 |
| Naive Bayes | 0.9386 | 0.9934 | 0.9114 | 1.0000 | 0.9536 | 0.8715 |
| Random Forest | 0.9737 | 0.9944 | 0.9600 | 1.0000 | 0.9796 | 0.9442 |
| SVM | 0.9825 | 0.9947 | 0.9730 | 1.0000 | 0.9863 | 0.9626 |

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Achieved 96.49% accuracy and 0.9960 AUC. Precision was 0.9595, recall 0.9861, F1 0.9726 and MCC 0.9245, indicating strong overall performance. |
| Decision Tree | Achieved 92.11% accuracy, the lowest among the six models. Although recall was high at 0.9722, precision was 0.9091 and MCC was 0.8299, indicating comparatively weaker performance. |
| kNN | Achieved 95.61% accuracy and 0.9825 AUC. Precision was 0.9467, recall 0.9861, F1 0.9660 and MCC 0.9058, showing strong but not leading performance. |
| Naive Bayes | Achieved 93.86% accuracy and 0.9934 AUC. It obtained 100% recall, but precision was 0.9114. F1 was 0.9536 and MCC 0.8715, indicating good but comparatively lower overall performance. |
| Random Forest | Achieved 97.37% accuracy, 0.9944 AUC and 100% recall. F1 of 0.9796 and MCC of 0.9442 demonstrate excellent performance and improvement over a single Decision Tree. |
| SVM | Achieved the highest accuracy (98.25%), precision (0.9730), F1 (0.9863) and MCC (0.9626), with 100% recall and 0.9947 AUC. It showed the strongest overall performance. ||
| **Overall Winner** | **SVM** Support Vector Machine (SVM) is the overall winner. It achieved the highest accuracy (98.25%), precision (97.30%), F1 score (98.63%) and MCC (0.9626), together with 100% recall and an AUC of 0.9947. Based on the combined evaluation of the six metrics on the same test set, SVM provided the strongest overall classification performance. |

## Streamlit Features
- Test CSV upload
- Model-selection dropdown
- Accuracy, AUC, Precision, Recall, F1 and MCC
- Confusion matrix
- Classification report
- Comparison of all six models
- Test-data preview

## Live Streamlit App Link
https://breast-cancer-analysis-idp9rdbcgz4x2nktemyqel.streamlit.app/

## Project Structure
```text
ml_assignment_2_solution/
├── app.py
├── train_models.py
├── requirements.txt
├── README.md
├── test_data.csv
├── model_metrics.csv
├── metadata.json
└── model/
    ├── logistic_regression.joblib
    ├── decision_tree.joblib
    ├── knn.joblib
    ├── naive_bayes.joblib
    ├── random_forest.joblib
    └── svm.joblib
```
