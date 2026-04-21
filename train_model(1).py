# train_model.py  — fast and modular training script
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import classification_report, recall_score, f1_score, precision_score, accuracy_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
# Optional models (uncomment when needed)
# from sklearn.ensemble import HistGradientBoostingClassifier
# from sklearn.neural_network import MLPClassifier

from custom_transformers import TargetEncoder

# =================== CONFIG ===================
RANDOM_STATE = 42
TARGET_COL = "isDefault"

# Use a small sample to make it fast on low-end machines (set to None for full data)
SAMPLE_N = 1000

# Run which models: start simple with Logistic Regression only
MODELS_TO_RUN = ["LogReg"]  # e.g., ["LogReg", "HistGB", "MLP"]

# Hyperparameter search mode: 'none' | 'random' | 'grid'
SEARCH_MODE = "random"

# RandomizedSearch iterations and CV folds
N_ITER = 20
CV_FOLDS = 3

# Multi-metric scoring; refit by recall for imbalanced problems
MULTI_SCORING = {"f1": "f1", "recall": "recall", "precision": "precision", "roc_auc": "roc_auc"}
REFIT_METRIC = "recall"

# Cardinality threshold to decide high-cardinality vs low-cardinality
CARDINALITY_THRESHOLD = 20
# =============================================

# --------- Data loading ---------
df = pd.read_csv("data.csv")

# Optional sampling for fast iteration
if SAMPLE_N is not None and SAMPLE_N < len(df):
    df = df.sample(SAMPLE_N, random_state=RANDOM_STATE).reset_index(drop=True)

# --------- Dynamic preprocessor builder ---------
def build_preprocessor(df, target_col, cardinality_threshold=20, use_target_encoder=True):
    """Build a ColumnTransformer dynamically based on column types and cardinality.
    Only add branches that actually have columns."""
    X = df.drop(columns=[target_col])
    y = df[target_col]

    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

    low_card = [c for c in categorical_cols if df[c].nunique() <= cardinality_threshold]
    high_card = [c for c in categorical_cols if df[c].nunique() > cardinality_threshold]

    transformers = []

    if numeric_cols:
        transformers.append((
            "num",
            Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ]),
            numeric_cols
        ))

    if low_card:
        transformers.append((
            "onehot",
            Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore"))
            ]),
            low_card
        ))

    if use_target_encoder and high_card:
        transformers.append((
            "tgt",
            Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("tgt", TargetEncoder())
            ]),
            high_card
        ))

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False
    )

    info = {
        "numeric_cols": numeric_cols,
        "low_card_cols": low_card,
        "high_card_cols": high_card
    }
    return preprocessor, info

# Build preprocessor
preprocess, pre_info = build_preprocessor(
    df, TARGET_COL, cardinality_threshold=CARDINALITY_THRESHOLD, use_target_encoder=True
)

# --------- Train/test split ---------
X = df.drop(columns=[TARGET_COL])
y = df[TARGET_COL]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)

# --------- Models and search spaces ---------
def get_models():
    models = {}
    if "LogReg" in MODELS_TO_RUN:
        models["LogReg"] = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    # if "HistGB" in MODELS_TO_RUN:
    #     models["HistGB"] = HistGradientBoostingClassifier(random_state=RANDOM_STATE)
    # if "MLP" in MODELS_TO_RUN:
    #     models["MLP"] = MLPClassifier(max_iter=300, random_state=RANDOM_STATE, early_stopping=True)
    return models

def get_search_space():
    """Return grid or list-based distributions for RandomizedSearch (no SciPy dependency)."""
    space = {}

    if "LogReg" in MODELS_TO_RUN:
        if SEARCH_MODE == "grid":
            space["LogReg"] = {
                "C": [0.001, 0.01, 0.1, 1, 10],
                "class_weight": [None, "balanced"],
                "penalty": ["l2"],
            }
        elif SEARCH_MODE == "random":
            space["LogReg"] = {
                "C": list(np.logspace(-3, 1, 30)),  # 0.001 ~ 10
                "class_weight": [None, "balanced"],
                "penalty": ["l2"],
            }
        else:  # none
            space["LogReg"] = {}

    # if "HistGB" in MODELS_TO_RUN:
    #     if SEARCH_MODE == "grid":
    #         space["HistGB"] = {
    #             "learning_rate": [0.05, 0.1],
    #             "max_depth": [None, 5],
    #             "max_leaf_nodes": [31, 63],
    #         }
    #     elif SEARCH_MODE == "random":
    #         space["HistGB"] = {
    #             "learning_rate": list(np.logspace(-3, -1, 20)),  # 0.001 ~ 0.1
    #             "max_depth": [None, 3, 5],
    #             "max_leaf_nodes": list(range(16, 129)),
    #         }
    #     else:
    #         space["HistGB"] = {}

    # if "MLP" in MODELS_TO_RUN:
    #     if SEARCH_MODE == "grid":
    #         space["MLP"] = {
    #             "hidden_layer_sizes": [(64,), (128,), (128, 64)],
    #             "alpha": [1e-4, 1e-3, 1e-2],
    #             "learning_rate_init": [1e-3, 1e-2],
    #         }
    #     elif SEARCH_MODE == "random":
    #         space["MLP"] = {
    #             "hidden_layer_sizes": [(64,), (96,), (128,), (128, 64)],
    #             "alpha": list(np.logspace(-5, -1, 20)),
    #             "learning_rate_init": list(np.logspace(-4, -2, 12)),
    #         }
    #     else:
    #         space["MLP"] = {}

    return space

def prepend_prefix(d):
    return {f"model__{k}": v for k, v in d.items()}

cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

def run_one_model(name, base_model, search_space):
    pipe = Pipeline([("pre", preprocess), ("model", base_model)])

    if SEARCH_MODE == "none" or not search_space:
        print(f"[{name}] Fit without hyperparameter search ...")
        clf = pipe.fit(X_train, y_train)
        return clf, {}
    else:
        space_prefixed = prepend_prefix(search_space)
        if SEARCH_MODE == "grid":
            search = GridSearchCV(
                pipe, param_grid=space_prefixed, cv=cv,
                scoring=MULTI_SCORING, refit=REFIT_METRIC,
                n_jobs=-1, verbose=1
            )
        else:  # random
            search = RandomizedSearchCV(
                pipe, param_distributions=space_prefixed, n_iter=N_ITER, cv=cv,
                scoring=MULTI_SCORING, refit=REFIT_METRIC,
                n_jobs=-1, verbose=1, random_state=RANDOM_STATE
            )
        search.fit(X_train, y_train)
        print(f"[{name}] best params: {search.best_params_}")
        print(f"[{name}] CV best {REFIT_METRIC}: {search.best_score_:.4f}")
        return search.best_estimator_, search.best_params_

# --------- Train & evaluate ---------
models = get_models()
spaces = get_search_space()

best_pipeline = None
best_recall = -1
rows = []

for name, model in models.items():
    print(f"\n===== Run {name} ({SEARCH_MODE}) =====")
    est, best_params = run_one_model(name, model, spaces.get(name, {}))

    y_pred = est.predict(X_test)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1v = f1_score(y_test, y_pred, zero_division=0)
    acc = accuracy_score(y_test, y_pred)
    pre = precision_score(y_test, y_pred, zero_division=0)
    print("Report:\n", classification_report(y_test, y_pred, zero_division=0))

    rows.append({"Model": name, "Accuracy": acc, "Precision": pre, "Recall": rec, "F1": f1v, "Params": best_params})
    if rec > best_recall:
        best_recall = rec
        best_pipeline = est

print("\nSummary:")
print(pd.DataFrame(rows).set_index("Model"))

# --------- Save best pipeline (preprocessor + model) ---------
joblib.dump(best_pipeline, "model.pkl")
print("\nSaved best pipeline -> model.pkl")
