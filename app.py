import streamlit as st
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef, confusion_matrix, classification_report

st.set_page_config(page_title="ML Classification Dashboard", page_icon="🤖", layout="wide")
BASE = Path(__file__).resolve().parent
MODEL_DIR = BASE / "model"

MODEL_FILES = {
    "Logistic Regression": MODEL_DIR / "logistic_regression.joblib",
    "Decision Tree": MODEL_DIR / "decision_tree.joblib",
    "kNN": MODEL_DIR / "knn.joblib",
    "Naive Bayes": MODEL_DIR / "naive_bayes.joblib",
    "Random Forest": MODEL_DIR / "random_forest.joblib",
    "SVM": MODEL_DIR / "svm.joblib",
}

st.title("🤖 ML Classification Model Dashboard")
st.caption("ML Assignment - 2 | Breast Cancer Wisconsin (Diagnostic) Dataset")
st.info("Upload test data containing the same 30 feature columns as test_data.csv and a target column with malignant/benign labels.")

uploaded = st.sidebar.file_uploader("Upload test CSV", type=["csv"])
selected_model = st.sidebar.selectbox("Select a model", list(MODEL_FILES))

df = pd.read_csv(uploaded) if uploaded is not None else pd.read_csv(BASE / "test_data.csv")
if uploaded is None:
    st.sidebar.success("Using bundled test_data.csv")
else:
    st.sidebar.success("Uploaded test data loaded")

if "target" not in df.columns:
    st.error("CSV must contain a 'target' column.")
    st.stop()

model = joblib.load(MODEL_FILES[selected_model])
X = df.drop(columns=["target"])
y = df["target"].astype(str).str.lower().str.strip()

expected = list(getattr(model, "feature_names_in_", X.columns))
missing = sorted(set(expected) - set(X.columns))
if missing:
    st.error(f"Missing feature columns: {missing}")
    st.stop()
X = X[expected]

pred = model.predict(X)
proba = model.predict_proba(X)
classes = list(model.classes_)
score = proba[:, classes.index("benign")]

accuracy = accuracy_score(y, pred)
auc = roc_auc_score((y == "benign").astype(int), score)
precision = precision_score(y, pred, pos_label="benign", zero_division=0)
recall = recall_score(y, pred, pos_label="benign", zero_division=0)
f1 = f1_score(y, pred, pos_label="benign", zero_division=0)
mcc = matthews_corrcoef(y, pred)

st.subheader(f"Evaluation: {selected_model}")
cols = st.columns(6)
for col, label, value in zip(cols, ["Accuracy", "AUC", "Precision", "Recall", "F1 Score", "MCC"], [accuracy, auc, precision, recall, f1, mcc]):
    col.metric(label, f"{value:.4f}")

left, right = st.columns(2)
with left:
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y, pred, labels=["malignant", "benign"])
    st.dataframe(pd.DataFrame(cm, index=["Actual malignant", "Actual benign"], columns=["Predicted malignant", "Predicted benign"]), use_container_width=True)
with right:
    st.subheader("Classification Report")
    report = classification_report(y, pred, labels=["malignant", "benign"], target_names=["malignant", "benign"], output_dict=True, zero_division=0)
    st.dataframe(pd.DataFrame(report).T.round(4), use_container_width=True)

st.subheader("Test Data Preview")
st.dataframe(df.head(10), use_container_width=True)

st.subheader("All Model Results")
metrics_path = BASE / "model_metrics.csv"
if metrics_path.exists():
    result = pd.read_csv(metrics_path)
    st.dataframe(result.style.format({c: "{:.4f}" for c in result.columns if c != "ML Model Name"}), use_container_width=True)

st.divider()
st.caption("Customize the UI, wording, hyperparameters, observations and GitHub history before academic submission.")
