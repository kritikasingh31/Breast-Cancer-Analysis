# ML Assignment - 2: Classification Model Comparison

## a. Problem Statement
Implement classification models on one public classification dataset, evaluate them using Accuracy, AUC, Precision, Recall, F1 Score and MCC, compare their performance, and demonstrate the models through an interactive Streamlit application.

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
**Replace with your actual repository URL:** `https://github.com/<YOUR-USERNAME>/<YOUR-REPOSITORY>`

## d. Models Used
The assignment says "all 6 ML models" but its numbered list names only five models. This implementation uses the five named models plus **SVM** as the sixth model.

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
| Logistic Regression | Accuracy=0.9649; AUC=0.9960; F1=0.9726; MCC=0.9245. Compare with the winner using all six metrics and the same test set. |
| Decision Tree | Accuracy=0.9211; AUC=0.9448; F1=0.9396; MCC=0.8299. Compare with the winner using all six metrics and the same test set. |
| kNN | Accuracy=0.9561; AUC=0.9825; F1=0.9660; MCC=0.9058. Compare with the winner using all six metrics and the same test set. |
| Naive Bayes | Accuracy=0.9386; AUC=0.9934; F1=0.9536; MCC=0.8715. Compare with the winner using all six metrics and the same test set. |
| Random Forest | Accuracy=0.9737; AUC=0.9944; F1=0.9796; MCC=0.9442. Compare with the winner using all six metrics and the same test set. |
| SVM | Accuracy=0.9825; AUC=0.9947; F1=0.9863; MCC=0.9626. Strongest overall combined performance in this experiment. ||
| **Overall Winner** | **SVM** based on the combined F1/AUC/MCC comparison for this run. |

## Streamlit Features
- Test CSV upload
- Model-selection dropdown
- Accuracy, AUC, Precision, Recall, F1 and MCC
- Confusion matrix
- Classification report
- Comparison of all six models
- Test-data preview

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

## Run Locally
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
python train_models.py
streamlit run app.py
```

## Deployment
Push the complete folder to GitHub, then deploy `app.py` using Streamlit Community Cloud. Replace the GitHub and Streamlit URLs in your final PDF.

## BITS Virtual Lab Screenshot
Run the project in BITS Virtual Lab and include one genuine screenshot as required by the assignment.

## Academic Integrity
The assignment states that AI tools are allowed for learning support but not direct copy-paste submissions. Understand and customize this reference implementation, change/justify hyperparameters or UI where appropriate, and maintain your own GitHub commit history before submission.
