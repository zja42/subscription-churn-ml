#!/usr/bin/env python3
"""Train, compare, and explain subscription churn models."""

from __future__ import annotations

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "telco_customer_churn.csv"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
CACHE = ROOT / ".cache"
CACHE.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(CACHE / "matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import clone
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    silhouette_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_val_predict,
    train_test_split,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC

SEED = 42
TARGET = "Churn"
ID_COLUMN = "customerID"
NUMERIC = ["tenure", "MonthlyCharges", "TotalCharges"]
CATEGORICAL = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]


def load_data() -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    data = pd.read_csv(DATA_PATH)
    data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors="coerce")
    y = data[TARGET].map({"No": 0, "Yes": 1})
    if y.isna().any():
        raise ValueError("Unexpected Churn value.")
    customer_ids = data[ID_COLUMN]
    X = data[NUMERIC + CATEGORICAL].copy()
    return X, y.astype(int), customer_ids


def preprocessor() -> ColumnTransformer:
    numeric = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        [("numeric", numeric, NUMERIC), ("categorical", categorical, CATEGORICAL)],
        verbose_feature_names_out=False,
    )


def model_registry() -> dict[str, tuple[object, dict[str, list[object]]]]:
    return {
        "Dummy baseline": (DummyClassifier(strategy="prior"), {}),
        "Ridge logistic": (
            LogisticRegression(
                max_iter=3000, class_weight="balanced", solver="lbfgs", random_state=SEED
            ),
            {"classifier__C": [0.05, 0.2, 1.0, 5.0]},
        ),
        "LASSO logistic": (
            LogisticRegression(
                max_iter=3000,
                class_weight="balanced",
                solver="liblinear",
                l1_ratio=1.0,
                random_state=SEED,
            ),
            {"classifier__C": [0.02, 0.1, 0.5, 2.0]},
        ),
        "k-nearest neighbors": (
            KNeighborsClassifier(),
            {
                "classifier__n_neighbors": [15, 35, 75],
                "classifier__weights": ["uniform", "distance"],
            },
        ),
        "RBF support vector machine": (
            SVC(probability=True, class_weight="balanced", random_state=SEED),
            {"classifier__C": [0.5, 1.0, 2.0], "classifier__gamma": ["scale", 0.02]},
        ),
        "Random forest": (
            RandomForestClassifier(
                n_estimators=350,
                class_weight="balanced_subsample",
                n_jobs=1,
                random_state=SEED,
            ),
            {
                "classifier__max_depth": [7, None],
                "classifier__min_samples_leaf": [3, 12],
            },
        ),
        "Histogram gradient boosting": (
            HistGradientBoostingClassifier(
                class_weight="balanced", max_iter=250, random_state=SEED
            ),
            {
                "classifier__learning_rate": [0.04, 0.08],
                "classifier__max_leaf_nodes": [15, 31],
            },
        ),
        "Neural network": (
            MLPClassifier(
                max_iter=600,
                early_stopping=True,
                validation_fraction=0.15,
                random_state=SEED,
            ),
            {
                "classifier__hidden_layer_sizes": [(32,), (48, 24)],
                "classifier__alpha": [0.001, 0.01],
            },
        ),
    }


def classification_metrics(y: pd.Series, probability: np.ndarray, threshold: float) -> dict[str, float | int]:
    prediction = (probability >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, prediction).ravel()
    return {
        "roc_auc": roc_auc_score(y, probability),
        "pr_auc": average_precision_score(y, probability),
        "brier_score": brier_score_loss(y, probability),
        "precision": precision_score(y, prediction, zero_division=0),
        "recall": recall_score(y, prediction, zero_division=0),
        "f1": f1_score(y, prediction, zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y, prediction),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def choose_value_threshold(y: pd.Series, probability: np.ndarray) -> tuple[float, pd.DataFrame]:
    """Choose a threshold under an explicit retention campaign scenario."""
    thresholds = np.linspace(0.05, 0.80, 151)
    rows = []
    for threshold in thresholds:
        target = probability >= threshold
        true_churn = int(((y.to_numpy() == 1) & target).sum())
        non_churn = int(((y.to_numpy() == 0) & target).sum())
        # Scenario: $30 contact/offer cost, 25% save rate, $300 saved contribution.
        expected_value = true_churn * (0.25 * 300 - 30) - non_churn * 30
        rows.append(
            {
                "threshold": threshold,
                "customers_targeted": int(target.sum()),
                "churners_reached": true_churn,
                "non_churners_targeted": non_churn,
                "expected_value": expected_value,
            }
        )
    table = pd.DataFrame(rows)
    best = table.loc[table["expected_value"].idxmax()]
    return float(best["threshold"]), table


def bootstrap_intervals(y: pd.Series, probability: np.ndarray, repeats: int = 500) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    y_array = y.to_numpy()
    records = []
    for _ in range(repeats):
        index = rng.integers(0, len(y_array), len(y_array))
        sample_y = y_array[index]
        if len(np.unique(sample_y)) < 2:
            continue
        sample_p = probability[index]
        records.append(
            {
                "roc_auc": roc_auc_score(sample_y, sample_p),
                "pr_auc": average_precision_score(sample_y, sample_p),
            }
        )
    distribution = pd.DataFrame(records)
    return pd.DataFrame(
        [
            {
                "metric": metric,
                "estimate": roc_auc_score(y, probability)
                if metric == "roc_auc"
                else average_precision_score(y, probability),
                "lower_95": distribution[metric].quantile(0.025),
                "upper_95": distribution[metric].quantile(0.975),
            }
            for metric in ["roc_auc", "pr_auc"]
        ]
    )


def customer_segments(X: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    transformed = preprocessor().fit_transform(X)
    pca = PCA(n_components=0.90, random_state=SEED)
    reduced = pca.fit_transform(transformed)
    candidates = []
    for k in range(2, 7):
        labels = KMeans(n_clusters=k, n_init=20, random_state=SEED).fit_predict(reduced)
        candidates.append((k, silhouette_score(reduced, labels)))
    best_k = max(candidates, key=lambda item: item[1])[0]
    labels = KMeans(n_clusters=best_k, n_init=30, random_state=SEED).fit_predict(reduced)
    segment_data = X.copy()
    segment_data["churn"] = y.to_numpy()
    segment_data["segment"] = labels
    profile = (
        segment_data.groupby("segment")
        .agg(
            customers=("churn", "size"),
            churn_rate=("churn", "mean"),
            median_tenure=("tenure", "median"),
            median_monthly_charge=("MonthlyCharges", "median"),
            month_to_month_share=("Contract", lambda s: (s == "Month-to-month").mean()),
            fiber_share=("InternetService", lambda s: (s == "Fiber optic").mean()),
            tech_support_share=("TechSupport", lambda s: (s == "Yes").mean()),
        )
        .reset_index()
        .sort_values("churn_rate", ascending=False)
    )
    scores = pd.DataFrame(candidates, columns=["clusters", "silhouette"])
    return profile, scores, reduced[:, :2], labels


def save_figures(
    y_test: pd.Series,
    probabilities: dict[str, np.ndarray],
    test_metrics: pd.DataFrame,
    threshold_table: pd.DataFrame,
    importance: pd.DataFrame,
    pca_coordinates: np.ndarray,
    segment_labels: np.ndarray,
    segment_profile: pd.DataFrame,
) -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    palette = {"ROC-AUC": "#0f766e", "PR-AUC": "#d97706"}

    ordered = test_metrics.sort_values("roc_auc")
    fig, ax = plt.subplots(figsize=(9.2, 6.4))
    y_pos = np.arange(len(ordered))
    ax.barh(y_pos - 0.17, ordered["roc_auc"], height=0.33, color=palette["ROC-AUC"], label="ROC-AUC")
    ax.barh(y_pos + 0.17, ordered["pr_auc"], height=0.33, color=palette["PR-AUC"], label="PR-AUC")
    ax.set_yticks(y_pos, ordered["model"])
    ax.set_xlim(0, 1)
    ax.set_xlabel("Held-out test score")
    ax.set_title("Model comparison on the untouched test set", loc="left", weight="bold")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(FIGURES / "model_comparison.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.8))
    for name, probability in probabilities.items():
        fpr, tpr, _ = roc_curve(y_test, probability)
        axes[0].plot(fpr, tpr, label=f"{name} ({roc_auc_score(y_test, probability):.3f})")
        precision, recall, _ = precision_recall_curve(y_test, probability)
        axes[1].plot(recall, precision, label=f"{name} ({average_precision_score(y_test, probability):.3f})")
    axes[0].plot([0, 1], [0, 1], "--", color="#94a3b8")
    axes[1].axhline(y_test.mean(), linestyle="--", color="#94a3b8")
    axes[0].set(title="ROC curves", xlabel="False positive rate", ylabel="True positive rate")
    axes[1].set(title="Precision-recall curves", xlabel="Recall", ylabel="Precision")
    axes[1].legend(fontsize=7, loc="lower left")
    fig.tight_layout()
    fig.savefig(FIGURES / "roc_pr_curves.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    ax.plot(threshold_table["threshold"], threshold_table["expected_value"], color="#0f766e", linewidth=2.4)
    best = threshold_table.loc[threshold_table["expected_value"].idxmax()]
    ax.scatter(best["threshold"], best["expected_value"], color="#be123c", zorder=3)
    ax.set(
        title="Illustrative retention value by decision threshold",
        xlabel="Churn probability threshold",
        ylabel="Expected value on development sample ($)",
    )
    fig.tight_layout()
    fig.savefig(FIGURES / "threshold_value.png", dpi=180)
    plt.close(fig)

    top = importance.head(12).sort_values("importance_mean")
    fig, ax = plt.subplots(figsize=(8.3, 5.6))
    ax.barh(top["feature"], top["importance_mean"], xerr=top["importance_std"], color="#0f766e", alpha=0.88)
    ax.set(title="Champion model permutation importance", xlabel="Mean decrease in test ROC-AUC")
    fig.tight_layout()
    fig.savefig(FIGURES / "feature_importance.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.8, 5.8))
    scatter = ax.scatter(pca_coordinates[:, 0], pca_coordinates[:, 1], c=segment_labels, cmap="tab10", s=12, alpha=0.55)
    ax.set(title="Customer segments in principal-component space", xlabel="Principal component 1", ylabel="Principal component 2")
    ax.legend(*scatter.legend_elements(), title="Segment", loc="best")
    fig.tight_layout()
    fig.savefig(FIGURES / "customer_segments.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    profile = segment_profile.sort_values("churn_rate")
    ax.barh(profile["segment"].astype(str), profile["churn_rate"], color="#d97706")
    ax.set(title="Observed churn rate by customer segment", xlabel="Churn rate", ylabel="Segment")
    ax.xaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    fig.tight_layout()
    fig.savefig(FIGURES / "segment_churn.png", dpi=180)
    plt.close(fig)


def write_report(
    n_customers: int,
    churn_rate: float,
    champion_name: str,
    best_params: dict[str, object],
    metrics: pd.Series,
    intervals: pd.DataFrame,
    threshold: float,
    threshold_row: pd.Series,
    importance: pd.DataFrame,
    segment_profile: pd.DataFrame,
    cv_table: pd.DataFrame,
    test_table: pd.DataFrame,
) -> None:
    cv_rows = "\n".join(
        f"| {r.model} | {r.cv_roc_auc:.3f} | {r.cv_pr_auc:.3f} | {r.cv_brier:.3f} |"
        for r in cv_table.itertuples()
    )
    test_rows = "\n".join(
        f"| {r.model} | {r.roc_auc:.3f} | {r.pr_auc:.3f} | {r.brier_score:.3f} | {r.recall:.3f} | {r.precision:.3f} |"
        for r in test_table.sort_values("roc_auc", ascending=False).itertuples()
    )
    importance_rows = "\n".join(
        f"| {r.feature} | {r.importance_mean:.4f} |"
        for r in importance.head(10).itertuples()
    )
    segment_rows = "\n".join(
        f"| {int(r.segment)} | {int(r.customers):,} | {r.churn_rate:.1%} | {r.median_tenure:.0f} | ${r.median_monthly_charge:.2f} | {r.month_to_month_share:.1%} | {r.tech_support_share:.1%} |"
        for r in segment_profile.itertuples()
    )
    roc_ci = intervals.loc[intervals.metric == "roc_auc"].iloc[0]
    pr_ci = intervals.loc[intervals.metric == "pr_auc"].iloc[0]
    top_features = ", ".join(importance.head(5)["feature"])
    report = f"""# Subscription Churn Product Report

## Executive Summary

This project analyzes **{n_customers:,} subscription customers**, of whom **{churn_rate:.1%}** churned. Eight statistical and machine-learning approaches were compared using stratified five-fold cross-validation. **{champion_name}** was selected before the test set was evaluated.

On the untouched test set, the selected model achieved **{metrics.roc_auc:.3f} ROC-AUC** (bootstrap 95% CI **{roc_ci.lower_95:.3f}-{roc_ci.upper_95:.3f}**) and **{metrics.pr_auc:.3f} PR-AUC** (95% CI **{pr_ci.lower_95:.3f}-{pr_ci.upper_95:.3f}**). The fitted settings were `{json.dumps(best_params, sort_keys=True)}`.

![Model comparison](figures/model_comparison.png)

## Learning Design

The workflow follows the model-selection ideas in UBC STAT 406: preprocessing is fit inside each fold; model choice is based on cross-validation; an untouched test set estimates generalization; and bootstrap resampling quantifies uncertainty. The comparison covers regularized linear models, kNN, an RBF kernel machine, a random forest, gradient boosting, and a neural network.

| Model | CV ROC-AUC | CV PR-AUC | CV Brier |
|---|---:|---:|---:|
{cv_rows}

## Held-Out Performance

The 0.50 threshold is shown here for a common operating-point comparison. ROC-AUC and PR-AUC evaluate ranking across thresholds; the Brier score evaluates probability error, where lower is better.

| Model | ROC-AUC | PR-AUC | Brier | Recall | Precision |
|---|---:|---:|---:|---:|---:|
{test_rows}

![ROC and precision-recall curves](figures/roc_pr_curves.png)

## Product Decision Threshold

A threshold should reflect the economics of a retention action rather than maximize accuracy. The illustrative scenario assumes a **$30** contact/offer cost, a **25%** save rate among customers who would churn, and **$300** in retained contribution per successful save. Under those assumptions, the development-set rule selected a threshold of **{threshold:.3f}**. Applied once to the test set, it targets **{int(threshold_row.customers_targeted):,}** customers and reaches **{int(threshold_row.churners_reached):,}** observed churners.

This dollar estimate is a scenario analysis, not realized revenue. Product owners should replace the assumptions with experimental offer uptake, contribution margin, and treatment-effect estimates.

![Threshold value](figures/threshold_value.png)

## Churn Drivers

Permutation importance on the test sample identifies the variables most useful to the selected model. The leading fields are **{top_features}**.

| Feature | Mean ROC-AUC decrease |
|---|---:|
{importance_rows}

![Feature importance](figures/feature_importance.png)

These are predictive associations. Contract changes, support bundles, and pricing interventions should be tested experimentally before being treated as causes of churn.

## Customer Segments

PCA retained 90% of the encoded feature variance before K-means clustering. The number of clusters was selected from 2-6 using silhouette score.

| Segment | Customers | Churn rate | Median tenure | Median monthly charge | Month-to-month | Tech support |
|---:|---:|---:|---:|---:|---:|---:|
{segment_rows}

![Customer segments](figures/customer_segments.png)

![Segment churn](figures/segment_churn.png)

Segments with high churn, short tenure, month-to-month contracts, and limited support adoption are candidates for onboarding and service interventions. Segments should be validated for stability and business usefulness before activation.

## Product Recommendations

1. Score customers regularly and send the top risk tier to a controlled retention experiment.
2. Test early-tenure onboarding, annual-contract migration, and technical-support bundles as separate treatments.
3. Track incremental retention and contribution margin, not model accuracy alone.
4. Monitor calibration, segment mix, and feature distributions after deployment.
5. Exclude protected or sensitive attributes from offer eligibility decisions unless legal and fairness review supports their use.

## Limitations

- The IBM sample represents a fictional telecommunications company and has no event timestamp, so random-split performance may overstate future-period performance.
- Churn prediction estimates who is likely to leave, not who will respond to an intervention; uplift modeling requires treatment data.
- The value threshold depends on explicit assumptions and must be recalibrated for a real product.
- Customer clusters summarize similarity but do not establish stable behavioral personas.
"""
    (ROOT / "PRODUCT_REPORT.md").write_text(report, encoding="utf-8")


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)
    X, y, customer_ids = load_data()
    indices = np.arange(len(X))
    train_idx, test_idx = train_test_split(
        indices, test_size=0.25, stratify=y, random_state=SEED
    )
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    scoring = {"roc_auc": "roc_auc", "pr_auc": "average_precision", "brier": "neg_brier_score"}
    searches: dict[str, GridSearchCV] = {}
    cv_records = []
    for name, (classifier, grid) in model_registry().items():
        print(f"Selecting {name}...")
        pipeline = Pipeline([("preprocess", preprocessor()), ("classifier", classifier)])
        search = GridSearchCV(
            pipeline,
            grid or {},
            scoring=scoring,
            refit="roc_auc",
            cv=cv,
            n_jobs=1,
            return_train_score=False,
        ).fit(X_train, y_train)
        searches[name] = search
        best = search.best_index_
        cv_records.append(
            {
                "model": name,
                "cv_roc_auc": search.cv_results_["mean_test_roc_auc"][best],
                "cv_pr_auc": search.cv_results_["mean_test_pr_auc"][best],
                "cv_brier": -search.cv_results_["mean_test_brier"][best],
                "best_params": json.dumps(search.best_params_, sort_keys=True),
            }
        )
    cv_table = pd.DataFrame(cv_records).sort_values("cv_roc_auc", ascending=False)
    champion_name = str(cv_table.iloc[0]["model"])
    champion_template = searches[champion_name].best_estimator_

    print(f"Selecting business threshold for {champion_name}...")
    oof_probability = cross_val_predict(
        champion_template,
        X_train,
        y_train,
        cv=cv,
        method="predict_proba",
        n_jobs=1,
    )[:, 1]
    threshold, development_threshold_table = choose_value_threshold(y_train, oof_probability)

    test_records = []
    test_probabilities = {}
    for name, search in searches.items():
        probability = search.best_estimator_.predict_proba(X_test)[:, 1]
        test_probabilities[name] = probability
        test_records.append({"model": name, **classification_metrics(y_test, probability, 0.50)})
    test_table = pd.DataFrame(test_records)
    champion = searches[champion_name].best_estimator_
    champion_probability = test_probabilities[champion_name]
    champion_metrics = pd.Series(classification_metrics(y_test, champion_probability, threshold))
    intervals = bootstrap_intervals(y_test, champion_probability)

    _, test_threshold_table = choose_value_threshold(y_test, champion_probability)
    chosen_test_row = test_threshold_table.iloc[(test_threshold_table["threshold"] - threshold).abs().argmin()]

    print("Calculating permutation importance...")
    permutation = permutation_importance(
        champion,
        X_test,
        y_test,
        scoring="roc_auc",
        n_repeats=20,
        random_state=SEED,
        n_jobs=1,
    )
    importance = pd.DataFrame(
        {
            "feature": X.columns,
            "importance_mean": permutation.importances_mean,
            "importance_std": permutation.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)

    print("Building PCA and K-means customer segments...")
    segment_profile, silhouette_scores, pca_coordinates, segment_labels = customer_segments(X, y)

    scored = pd.DataFrame(
        {
            "customerID": customer_ids.iloc[test_idx].to_numpy(),
            "actual_churn": y_test.to_numpy(),
            "churn_probability": champion_probability,
            "target_at_value_threshold": (champion_probability >= threshold).astype(int),
        }
    ).sort_values("churn_probability", ascending=False)

    cv_table.to_csv(RESULTS / "cross_validation_metrics.csv", index=False)
    test_table.to_csv(RESULTS / "test_metrics.csv", index=False)
    intervals.to_csv(RESULTS / "bootstrap_intervals.csv", index=False)
    importance.to_csv(RESULTS / "permutation_importance.csv", index=False)
    development_threshold_table.to_csv(RESULTS / "threshold_value_curve.csv", index=False)
    segment_profile.to_csv(RESULTS / "customer_segments.csv", index=False)
    silhouette_scores.to_csv(RESULTS / "silhouette_scores.csv", index=False)
    scored.to_csv(RESULTS / "test_customer_scores.csv", index=False)
    summary = {
        "customers": len(X),
        "churn_rate": float(y.mean()),
        "train_customers": len(train_idx),
        "test_customers": len(test_idx),
        "champion_model": champion_name,
        "champion_parameters": searches[champion_name].best_params_,
        "value_threshold": threshold,
        "test_metrics_at_value_threshold": champion_metrics.to_dict(),
    }
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    save_figures(
        y_test,
        test_probabilities,
        test_table,
        development_threshold_table,
        importance,
        pca_coordinates,
        segment_labels,
        segment_profile,
    )
    write_report(
        len(X),
        float(y.mean()),
        champion_name,
        searches[champion_name].best_params_,
        champion_metrics,
        intervals,
        threshold,
        chosen_test_row,
        importance,
        segment_profile,
        cv_table,
        test_table,
    )
    print(
        f"Done: {champion_name}; test ROC-AUC={champion_metrics.roc_auc:.3f}; "
        f"PR-AUC={champion_metrics.pr_auc:.3f}; threshold={threshold:.3f}"
    )


if __name__ == "__main__":
    main()
