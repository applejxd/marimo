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
    # LightGBM で回帰タスクを学ぶ

    元 notebook と同じく、表形式回帰・特徴量重要度・ハイパーパラメータ探索を扱う。
    Boston housing は現在の scikit-learn で削除されているため、
    ここでは組み込みの diabetes データセットへ置き換えている。
    """)
    return


@app.cell
def _():
    sk_datasets = importlib.import_module("sklearn.datasets")
    dataset = sk_datasets.load_diabetes(as_frame=True)
    frame = dataset.frame.rename(columns={"target": "target"})
    feature_columns = [column for column in frame.columns if column != "target"]
    frame.head()
    return feature_columns, frame


@app.cell
def _():
    lgb = importlib.import_module("lightgbm")
    sk_metrics = importlib.import_module("sklearn.metrics")
    sk_model_selection = importlib.import_module("sklearn.model_selection")
    return lgb, sk_metrics, sk_model_selection


@app.cell
def _(frame, lgb, sk_model_selection):
    train_frame, valid_frame = sk_model_selection.train_test_split(frame, test_size=0.25, random_state=71)

    def fit_booster(train_x, train_y, valid_x, valid_y, params=None):
        booster_params = {
            "objective": "regression",
            "metric": "l1",
            "learning_rate": 0.05,
            "num_leaves": 31,
            "feature_fraction": 0.9,
            "bagging_fraction": 0.9,
            "bagging_freq": 1,
            "seed": 71,
            "verbosity": -1,
        }
        if params is not None:
            booster_params.update(params)
        train_set = lgb.Dataset(train_x, label=train_y)
        valid_set = lgb.Dataset(valid_x, label=valid_y)
        booster = lgb.train(
            booster_params,
            train_set,
            valid_sets=[train_set, valid_set],
            valid_names=["train", "valid"],
            num_boost_round=400,
            callbacks=[lgb.early_stopping(25, verbose=False), lgb.log_evaluation(0)],
        )
        return booster

    return fit_booster, train_frame, valid_frame


@app.cell
def _(feature_columns, fit_booster, pd, train_frame, valid_frame):
    baseline_model = fit_booster(train_frame[feature_columns], train_frame["target"], valid_frame[feature_columns], valid_frame["target"])
    importance_frame = pd.DataFrame({"feature": feature_columns, "importance": baseline_model.feature_importance(importance_type="gain")}).sort_values("importance", ascending=False)
    return baseline_model, importance_frame


@app.cell
def _(importance_frame):
    importance_frame
    return


@app.cell
def _(baseline_model, feature_columns, sk_metrics, valid_frame):
    baseline_predictions = baseline_model.predict(valid_frame[feature_columns])
    baseline_metrics = pd.DataFrame(
        {
            "metric": ["MAE", "RMSE", "R2"],
            "baseline": [sk_metrics.mean_absolute_error(valid_frame["target"], baseline_predictions), np.sqrt(sk_metrics.mean_squared_error(valid_frame["target"], baseline_predictions)), sk_metrics.r2_score(valid_frame["target"], baseline_predictions)],
        }
    )
    return baseline_metrics, baseline_predictions


@app.cell
def _(baseline_metrics):
    baseline_metrics
    return


@app.cell
def _(baseline_predictions, valid_frame):
    _fig, _ax = plt.subplots(figsize=(5, 5))
    _ax.scatter(valid_frame["target"], baseline_predictions, alpha=0.7)
    diagonal_min = min(valid_frame["target"].min(), baseline_predictions.min())
    diagonal_max = max(valid_frame["target"].max(), baseline_predictions.max())
    _ax.plot([diagonal_min, diagonal_max], [diagonal_min, diagonal_max], linestyle="--", color="black")
    _ax.set_xlabel("actual target")
    _ax.set_ylabel("predicted target")
    _ax.set_title("Baseline LightGBM predictions")
    _fig.tight_layout()
    _fig
    return


@app.cell
def _():
    hyperopt = importlib.import_module("hyperopt")
    history = []
    search_space = {
        "num_leaves": hyperopt.hp.choice("num_leaves", list(range(15, 64, 8))),
        "feature_fraction": hyperopt.hp.uniform("feature_fraction", 0.7, 1.0),
        "bagging_fraction": hyperopt.hp.uniform("bagging_fraction", 0.7, 1.0),
        "min_data_in_leaf": hyperopt.hp.choice("min_data_in_leaf", [10, 20, 30, 40]),
        "lambda_l2": hyperopt.hp.loguniform("lambda_l2", -4.0, 1.0),
    }
    return history, hyperopt, search_space


@app.cell
def _(feature_columns, fit_booster, history, hyperopt, sk_metrics, sk_model_selection, train_frame):
    def objective(params):
        cv = sk_model_selection.KFold(n_splits=4, shuffle=True, random_state=71)
        scores = []
        for train_index, valid_index in cv.split(train_frame):
            cv_train = train_frame.iloc[train_index]
            cv_valid = train_frame.iloc[valid_index]
            model = fit_booster(cv_train[feature_columns], cv_train["target"], cv_valid[feature_columns], cv_valid["target"], params=params)
            predictions = model.predict(cv_valid[feature_columns])
            scores.append(sk_metrics.mean_absolute_error(cv_valid["target"], predictions))
        mean_score = float(np.mean(scores))
        history.append({"params": dict(params), "mae": mean_score})
        return {"loss": mean_score, "status": hyperopt.STATUS_OK}

    return objective


@app.cell
def _(history, hyperopt, objective, search_space):
    trials = hyperopt.Trials()
    hyperopt.fmin(fn=objective, space=search_space, algo=hyperopt.tpe.suggest, max_evals=12, trials=trials, rstate=np.random.default_rng(71))
    history_frame = pd.DataFrame(history).sort_values("mae").reset_index(drop=True)
    return history_frame


@app.cell
def _(history_frame):
    history_frame
    return


@app.cell
def _(feature_columns, fit_booster, history_frame, train_frame, valid_frame):
    best_params = dict(history_frame.iloc[0]["params"])
    tuned_model = fit_booster(train_frame[feature_columns], train_frame["target"], valid_frame[feature_columns], valid_frame["target"], params=best_params)
    tuned_predictions = tuned_model.predict(valid_frame[feature_columns])
    return best_params, tuned_predictions


@app.cell
def _(baseline_metrics, best_params, sk_metrics, tuned_predictions, valid_frame):
    comparison = baseline_metrics.copy()
    comparison["tuned"] = [sk_metrics.mean_absolute_error(valid_frame["target"], tuned_predictions), np.sqrt(sk_metrics.mean_squared_error(valid_frame["target"], tuned_predictions)), sk_metrics.r2_score(valid_frame["target"], tuned_predictions)]
    mo.vstack([comparison, pd.DataFrame({"best_params": [best_params]})])
    return


if __name__ == "__main__":
    app.run()
