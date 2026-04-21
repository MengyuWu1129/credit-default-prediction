import pandas as pd
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, precision_score, recall_score, f1_score
)
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.impute import SimpleImputer
from sklearn.utils import resample

RANDOM_STATE = 42
CV_FOLDS = 5
N_JOBS = -1


numeric_features = [...]  
categorical_features = [...]  

preprocess = ColumnTransformer(
    transformers=[
        ("num", Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]), numeric_features),
        ("cat", Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]), categorical_features),
    ]
)


def get_models():
    return {
        "LogisticRegression": LogisticRegression(solver="lbfgs", max_iter=1000, random_state=RANDOM_STATE),
        "GradientBoosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
        "MLPClassifier": MLPClassifier(max_iter=300, random_state=RANDOM_STATE),
    }


def get_param_grids():
    return {
        "LogisticRegression": {
            "C": [0.01, 0.1, 1, 10],
            "class_weight": [None, "balanced"],
            "penalty": ["l2"],  
        },
        "GradientBoosting": {
            "n_estimators": [100, 150, 200],
            "learning_rate": [0.01, 0.05, 0.1],
            "max_depth": [3, 4, 5],
            "subsample": [0.8, 1.0],
        },
        "MLPClassifier": {
            "hidden_layer_sizes": [(64,), (128,), (128, 64)],
            "alpha": [1e-4, 1e-3, 1e-2],
            "learning_rate_init": [0.001, 0.01],
        },
    }


def prepend_prefix(grid):
    return {f"model__{k}": v for k, v in grid.items()}


def refine_grid(model_name, best_params):

    refined = {}
    if model_name == "LogisticRegression":
        C_best = best_params["model__C"]
        if C_best <= 0.01:
            candidates = [0.005, 0.01, 0.02]
        elif C_best >= 10:
            candidates = [5, 10, 20]
        else:
            candidates = [C_best / 2, C_best, C_best * 2]
        refined["model__C"] = sorted(set(round(c, 6) for c in candidates))
        refined["model__class_weight"] = [best_params["model__class_weight"]]
        refined["model__penalty"] = ["l2"]

    elif model_name == "GradientBoosting":
        n_best = best_params["model__n_estimators"]
        refined["model__n_estimators"] = sorted({max(50, n_best - 50), n_best, n_best + 50})
        lr_best = best_params["model__learning_rate"]
        refined["model__learning_rate"] = sorted({
            round(max(lr_best / 2, 1e-4), 5),
            lr_best,
            round(lr_best * 1.5, 5),
        })
        depth_best = best_params["model__max_depth"]
        refined["model__max_depth"] = sorted({max(1, depth_best - 1), depth_best, depth_best + 1})
        subs_best = best_params["model__subsample"]
        subs_candidates = [subs_best]
        for delta in (-0.1, 0.1):
            v = round(subs_best + delta, 2)
            if 0 < v <= 1:
                subs_candidates.append(v)
        refined["model__subsample"] = sorted(set(subs_candidates))

    elif model_name == "MLPClassifier":
        h_best = best_params["model__hidden_layer_sizes"]
        refined["model__hidden_layer_sizes"] = list({
            h_best,
            (max(32, h_best[0] // 2),),
            (h_best[0] * 2,) + tuple(h_best[1:]),
        })
        alpha_best = best_params["model__alpha"]
        refined["model__alpha"] = sorted({alpha_best / 10, alpha_best, alpha_best * 10})
        lr_init_best = best_params["model__learning_rate_init"]
        refined["model__learning_rate_init"] = sorted({lr_init_best / 2, lr_init_best, lr_init_best * 2})

    return refined


multi_scoring = {
    "f1": "f1",
    "recall": "recall",
    "precision": "precision",
    "roc_auc": "roc_auc",
}

def run_search(model_name, model, param_grid, X_train, y_train, refit_metric="recall"):
    pipe = Pipeline([("pre", preprocess), ("model", model)])
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    search = GridSearchCV(
        pipe,
        param_grid=param_grid,
        cv=cv,
        scoring=multi_scoring,
        refit=refit_metric,
        n_jobs=N_JOBS,
        verbose=1,
    )
    search.fit(X_train, y_train)
    print(f"[{model_name}] best params:", search.best_params_)
    print(f"[{model_name}] CV best {refit_metric}:", search.best_score_)
    return search


def main(X_train, X_test, y_train, y_test):
    models = get_models()
    grids = get_param_grids()

    results = []

    for name, model in models.items():
        print(f"\n===== {name}: Broad Search =====")
        broad_grid = prepend_prefix(grids[name])
        broad_search = run_search(name, model, broad_grid, X_train, y_train, refit_metric="recall")

    
        broad_pred = broad_search.best_estimator_.predict(X_test)
        print("Broad test report:\n", classification_report(y_test, broad_pred, zero_division=0))

        
        print(f"\n===== {name}: Refined Search =====")
        refined_grid = refine_grid(name, broad_search.best_params_)
        print("Refined grid:", refined_grid)
        refined_search = run_search(name, model, refined_grid, X_train, y_train, refit_metric="recall")

        refined_pred = refined_search.best_estimator_.predict(X_test)
        acc = accuracy_score(y_test, refined_pred)
        prec = precision_score(y_test, refined_pred, zero_division=0)
        rec = recall_score(y_test, refined_pred, zero_division=0)
        f1 = f1_score(y_test, refined_pred, zero_division=0)

        print("Refined test report:\n", classification_report(y_test, refined_pred, zero_division=0))
        print("Confusion Matrix:\n", confusion_matrix(y_test, refined_pred))

        results.append({
            "Model": name,
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1": f1,
            "Broad_Params": broad_search.best_params_,
            "Refined_Params": refined_search.best_params_,
        })

    summary = pd.DataFrame(results).set_index("Model")
    print("\n===== Final Summary (Test Set) =====")
    print(summary[["Accuracy", "Precision", "Recall", "F1"]].round(4))
    print("\nParameter evolution:")
    for m, row in summary.iterrows():
        print(f"- {m}: broad={row['Broad_Params']} -> refined={row['Refined_Params']}")


