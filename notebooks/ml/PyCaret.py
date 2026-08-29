import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")

with app.setup:
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt
    import pandas as pd
    import tensorflow as tf


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # PyCaret による回帰ワークフロー

    scikit-learn 同梱の diabetes データセットを題材に、PyCaret による回帰の
    ワークフロー（前処理・モデル比較・チューニング・評価）を表中心で確認する。
    実行には `uv sync --group specialized --group gpu` が必要である。
    """)
    return


@app.cell
def _():
    from sklearn.datasets import load_diabetes

    diabetes = load_diabetes(as_frame=True)
    frame = diabetes.frame.rename(columns={"target": "Progression"})
    model_frame = frame.sample(frac=0.85, random_state=786).reset_index(drop=True)
    unseen_frame = frame.drop(model_frame.index).reset_index(drop=True)
    return model_frame, unseen_frame


@app.cell
def _(model_frame):
    import os

    pycaret_log = Path(__file__).resolve().parents[2] / "build" / "pycaret" / "regression.log"
    pycaret_log.parent.mkdir(parents=True, exist_ok=True)
    os.environ["PYCARET_CUSTOM_LOGGING_PATH"] = str(pycaret_log)

    from pycaret.regression import RegressionExperiment

    experiment = RegressionExperiment()
    experiment.setup(
        data=model_frame,
        target="Progression",
        session_id=123,
        fold=3,
        n_jobs=1,
        normalize=True,
        transform_target=True,
        system_log=False,
        verbose=False,
    )
    available_models = experiment.models().head(10)
    return available_models, experiment


@app.cell
def _(available_models):
    available_models
    return


@app.cell
def _(experiment):
    experiment.compare_models(sort="MAE", n_select=3, verbose=False)
    leaderboard = experiment.pull().reset_index(drop=True)
    return leaderboard


@app.cell
def _(leaderboard):
    leaderboard
    return


@app.cell
def _(experiment):
    lightgbm_model = experiment.create_model("lightgbm", verbose=False)
    tuned_lightgbm = experiment.tune_model(lightgbm_model, n_iter=8, optimize="MAE", verbose=False)
    tuning_table = experiment.pull().reset_index(drop=True)
    experiment.predict_model(tuned_lightgbm)
    holdout_table = experiment.pull().reset_index(drop=True)
    return holdout_table, tuned_lightgbm, tuning_table


@app.cell
def _(tuning_table):
    tuning_table
    return


@app.cell
def _(holdout_table):
    holdout_table.head()
    return


@app.cell
def _(experiment, tuned_lightgbm, unseen_frame):
    from sklearn.metrics import mean_absolute_error as _mean_absolute_error
    from sklearn.metrics import r2_score as _r2_score

    final_model = experiment.finalize_model(tuned_lightgbm)
    unseen_predictions = experiment.predict_model(final_model, data=unseen_frame)
    prediction_column = next(column for column in unseen_predictions.columns if column.lower().startswith("prediction"))
    unseen_metrics = pd.DataFrame(
        {
            "metric": ["R2", "MAE"],
            "value": [
                _r2_score(unseen_predictions["Progression"], unseen_predictions[prediction_column]),
                _mean_absolute_error(unseen_predictions["Progression"], unseen_predictions[prediction_column]),
            ],
        }
    )
    artifact_dir = Path(__file__).with_name("pycaret_artifacts")
    artifact_dir.mkdir(exist_ok=True)
    model_path = artifact_dir / "diabetes_lightgbm"
    experiment.save_model(final_model, str(model_path))
    reloaded_model = experiment.load_model(str(model_path))
    reloaded_predictions = experiment.predict_model(reloaded_model, data=unseen_frame)
    saved_pickle = model_path.with_suffix(".pkl")
    if saved_pickle.exists():
        saved_pickle.unlink()
    if artifact_dir.exists() and not any(artifact_dir.iterdir()):
        artifact_dir.rmdir()
    return reloaded_predictions, unseen_metrics


@app.cell
def _(unseen_metrics):
    unseen_metrics
    return


@app.cell
def _(model_frame, unseen_frame):
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    x = model_frame.drop(columns="Progression")
    y = model_frame["Progression"]
    x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=0.2, random_state=123)
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_valid_scaled = scaler.transform(x_valid)
    x_unseen_scaled = scaler.transform(unseen_frame.drop(columns="Progression"))
    return x_train_scaled, x_unseen_scaled, x_valid_scaled, y_train, y_valid


@app.cell
def _(x_train_scaled, y_train):
    tf.keras.utils.set_random_seed(123)
    keras_model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(x_train_scaled.shape[1],)),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(1),
    ], name="regression_mlp")
    keras_model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    history = keras_model.fit(x_train_scaled, y_train, validation_split=0.2, epochs=80, batch_size=32, verbose=0)
    history_frame = pd.DataFrame(history.history)
    return history_frame, keras_model


@app.cell
def _(history_frame):
    _fig, _ax = plt.subplots(figsize=(6, 4))
    _ax.plot(history_frame["mae"], label="train")
    _ax.plot(history_frame["val_mae"], label="validation")
    _ax.set_title("Keras regression MAE")
    _ax.legend()
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(keras_model, pd, x_unseen_scaled, x_valid_scaled, y_valid):
    from sklearn.metrics import mean_absolute_error, r2_score

    valid_predictions = keras_model.predict(x_valid_scaled, verbose=0).reshape(-1)
    keras_unseen_predictions = keras_model.predict(x_unseen_scaled, verbose=0).reshape(-1)
    keras_metrics = pd.DataFrame({"metric": ["validation_mae", "validation_r2", "unseen_prediction_mean"], "value": [mean_absolute_error(y_valid, valid_predictions), r2_score(y_valid, valid_predictions), keras_unseen_predictions.mean()]})
    return keras_metrics


@app.cell
def _(keras_metrics):
    keras_metrics
    return


if __name__ == "__main__":
    app.run()
