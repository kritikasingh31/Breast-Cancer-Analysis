from pathlib import Path
import json
import joblib
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef

BASE = Path(__file__).resolve().parent
MODEL_DIR = BASE / "model"
MODEL_DIR.mkdir(exist_ok=True)

data = load_breast_cancer(as_frame=True)
X = data.data.copy()
y = data.target.map({0: "malignant", 1: "benign"})
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

models = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=5000, random_state=42))
    ]),
    "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
    "kNN": Pipeline([
        ("scaler", StandardScaler()),
        ("model", KNeighborsClassifier(n_neighbors=7))
    ]),
    "Naive Bayes": GaussianNB(),
    "Random Forest": RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1),
    "SVM": Pipeline([
        ("scaler", StandardScaler()),
        ("model", SVC(kernel="rbf", C=2.0, probability=True, random_state=42))
    ])
}

def benign_scores(model, Xp):
    p = model.predict_proba(Xp)
    classes = list(model.classes_)
    return p[:, classes.index("benign")]

rows = []
for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    score = benign_scores(model, X_test)
    rows.append({
        "ML Model Name": name,
        "Accuracy": accuracy_score(y_test, pred),
        "AUC": roc_auc_score((y_test == "benign").astype(int), score),
        "Precision": precision_score(y_test, pred, pos_label="benign"),
        "Recall": recall_score(y_test, pred, pos_label="benign"),
        "F1": f1_score(y_test, pred, pos_label="benign"),
        "MCC": matthews_corrcoef(y_test, pred)
    })
    filename = name.lower().replace(" ", "_").replace("-", "") + ".joblib"
    joblib.dump(model, MODEL_DIR / filename)

pd.DataFrame(rows).to_csv(BASE / "model_metrics.csv", index=False)
test_df = X_test.copy()
test_df["target"] = y_test.values
test_df.to_csv(BASE / "test_data.csv", index=False)

(BASE / "metadata.json").write_text(json.dumps({
    "dataset": "Breast Cancer Wisconsin (Diagnostic)",
    "source": "UCI Machine Learning Repository (via scikit-learn)",
    "n_instances": int(X.shape[0]),
    "n_features": int(X.shape[1]),
    "target_classes": ["malignant", "benign"],
    "test_size": 0.20,
    "random_state": 42
}, indent=2))
print(pd.DataFrame(rows).round(4).to_string(index=False))
