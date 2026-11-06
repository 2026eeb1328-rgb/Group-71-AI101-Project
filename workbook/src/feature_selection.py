"""Leakage-safe feature-count selection for the AI101 project."""
from functools import partial
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_selection_pipeline(random_state=42):
    """Build a complete pipeline so every CV fold learns preprocessing internally."""
    mutual_info = partial(mutual_info_classif, random_state=random_state)
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("select", SelectKBest(score_func=mutual_info)),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            solver="lbfgs", max_iter=3000
        )),
    ])


def tune_feature_count(X_train, y_train, random_state=42, n_jobs=-1):
    """
    Tune feature count K and Logistic Regression C using 5-fold stratified CV.
    CV runs on the training partition only. The test set must not be passed here.
    """
    pipeline = build_selection_pipeline(random_state=random_state)
    param_grid = {
        "select__k": [5, 8, 10, 12, 15, 18, 20],
        "model__C": [0.01, 0.1, 1, 10, 100],
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="f1_macro",
        cv=cv,
        n_jobs=n_jobs,
        refit=True,
        return_train_score=True,
    )
    search.fit(X_train, y_train)
    return search
