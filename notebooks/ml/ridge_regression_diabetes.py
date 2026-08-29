import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")

with app.setup:
    import importlib

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # scikit-learn による回帰の基本例

    scikit-learn 同梱の diabetes データセットを使い、線形回帰と Ridge 回帰の
    基本的な流れを確認する。追加のダウンロードが要らないので、そのまま実行できる。
    """)
    return


@app.cell
def _():
    sk_datasets = importlib.import_module("sklearn.datasets")
    dataset = sk_datasets.load_diabetes(as_frame=True)
    data_frame = dataset.frame.rename(columns={"target": "target"})
    feature_columns = [column for column in data_frame.columns if column != "target"]
    data_frame.head()
    return data_frame, feature_columns


@app.cell
def _(data_frame, feature_columns):
    sk_linear_model = importlib.import_module("sklearn.linear_model")
    sk_metrics = importlib.import_module("sklearn.metrics")
    sk_model_selection = importlib.import_module("sklearn.model_selection")
    sk_pipeline = importlib.import_module("sklearn.pipeline")
    sk_preprocessing = importlib.import_module("sklearn.preprocessing")

    candidate_alphas = [0.1, 1.0, 10.0]
    cv = sk_model_selection.KFold(n_splits=4, shuffle=True, random_state=71)
    cv_rows = []
    _features = data_frame[feature_columns]
    _target = data_frame["target"]
    for alpha in candidate_alphas:
        estimator = sk_pipeline.make_pipeline(sk_preprocessing.StandardScaler(), sk_linear_model.Ridge(alpha=alpha))
        scores = sk_model_selection.cross_validate(estimator, _features, _target, cv=cv, scoring={"mae": "neg_mean_absolute_error", "r2": "r2"})
        cv_rows.append({"alpha": alpha, "mae": -scores["test_mae"].mean(), "r2": scores["test_r2"].mean()})

    cv_results = pd.DataFrame(cv_rows).sort_values("mae").reset_index(drop=True)
    best_alpha = float(cv_results.iloc[0]["alpha"])
    return best_alpha, cv_results, sk_linear_model, sk_metrics, sk_model_selection, sk_pipeline, sk_preprocessing


@app.cell
def _(cv_results):
    cv_results
    return


@app.cell
def _(best_alpha, data_frame, feature_columns, sk_linear_model, sk_metrics, sk_model_selection, sk_pipeline, sk_preprocessing):
    _features = data_frame[feature_columns]
    _target = data_frame["target"]
    train_x, test_x, train_y, test_y = sk_model_selection.train_test_split(_features, _target, test_size=0.25, random_state=71)
    ridge_pipeline = sk_pipeline.make_pipeline(sk_preprocessing.StandardScaler(), sk_linear_model.Ridge(alpha=best_alpha))
    ridge_pipeline.fit(train_x, train_y)
    predictions = ridge_pipeline.predict(test_x)
    metrics_frame = pd.DataFrame(
        {
            "metric": ["MAE", "RMSE", "R2"],
            "value": [sk_metrics.mean_absolute_error(test_y, predictions), np.sqrt(sk_metrics.mean_squared_error(test_y, predictions)), sk_metrics.r2_score(test_y, predictions)],
        }
    )
    coefficients = pd.Series(ridge_pipeline.named_steps["ridge"].coef_, index=feature_columns, name="coefficient").sort_values()
    return coefficients, metrics_frame, predictions, test_y


@app.cell
def _(metrics_frame):
    metrics_frame
    return


@app.cell
def _(coefficients):
    _fig, _ax = plt.subplots(figsize=(8, 4))
    coefficients.plot.barh(ax=_ax, color="tab:blue")
    _ax.set_title("Ridge coefficients on standardized features")
    _ax.set_xlabel("coefficient")
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(predictions, test_y):
    _fig, _ax = plt.subplots(figsize=(5, 5))
    _ax.scatter(test_y, predictions, alpha=0.7)
    diagonal_min = min(test_y.min(), predictions.min())
    diagonal_max = max(test_y.max(), predictions.max())
    _ax.plot([diagonal_min, diagonal_max], [diagonal_min, diagonal_max], linestyle="--", color="black")
    _ax.set_xlabel("actual target")
    _ax.set_ylabel("predicted target")
    _ax.set_title("Holdout predictions")
    _fig.tight_layout()
    _fig
    return


if __name__ == "__main__":
    app.run()
