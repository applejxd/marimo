import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")

with app.setup:
    import importlib.util

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import tensorflow as tf


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # MNIST を使った Keras 画像分類

    全結合ネットワークと CNN を同じ前処理で学習し、loss/accuracy と誤分類例を比較する。
    元 notebook にあった PyCaret 版は `pycaret` がある環境だけで追加比較できるようにした。
    """)
    return


@app.cell
def _():
    tf.keras.utils.set_random_seed(42)
    device_name = "/GPU:0" if tf.config.list_physical_devices("GPU") else "/CPU:0"
    runtime_frame = pd.DataFrame({"item": ["tensorflow", "keras", "device"], "value": [tf.__version__, tf.keras.__version__, device_name]})
    runtime_frame
    return device_name


@app.cell
def _():
    (x_train_full, y_train_full), (x_test_full, y_test_full) = tf.keras.datasets.mnist.load_data()
    x_train = x_train_full[:15000].astype("float32") / 255.0
    y_train = y_train_full[:15000]
    x_test = x_test_full[:3000].astype("float32") / 255.0
    y_test = y_test_full[:3000]
    x_train_cnn = x_train[..., None]
    x_test_cnn = x_test[..., None]
    return x_test, x_test_cnn, x_train, x_train_cnn, y_test, y_train


@app.cell
def _(x_train, y_train):
    _fig, _axes = plt.subplots(2, 5, figsize=(8, 4))
    for idx, _axis in enumerate(_axes.ravel()):
        _axis.imshow(x_train[idx], cmap="gray")
        _axis.set_title(f"label={y_train[idx]}")
        _axis.axis("off")
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(device_name, x_train, y_train):
    with tf.device(device_name):
        dense_model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(28, 28)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation="sigmoid"),
            tf.keras.layers.Dense(128, activation="sigmoid"),
            tf.keras.layers.Dense(10, activation="softmax"),
        ], name="dense_mnist")
        dense_model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        dense_history = dense_model.fit(x_train, y_train, validation_split=0.1, epochs=3, batch_size=128, verbose=0)
    dense_history_frame = pd.DataFrame(dense_history.history)
    return dense_history_frame, dense_model


@app.cell
def _(dense_model, x_test, y_test):
    dense_metrics = dense_model.evaluate(x_test, y_test, verbose=0, return_dict=True)
    return dense_metrics


@app.cell
def _(device_name, x_train_cnn, y_train):
    with tf.device(device_name):
        cnn_model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(28, 28, 1)),
            tf.keras.layers.Conv2D(32, kernel_size=(5, 5), activation="relu"),
            tf.keras.layers.MaxPool2D((2, 2)),
            tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
            tf.keras.layers.MaxPool2D((2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(10, activation="softmax"),
        ], name="cnn_mnist")
        cnn_model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        cnn_history = cnn_model.fit(x_train_cnn, y_train, validation_split=0.1, epochs=3, batch_size=128, verbose=0)
    cnn_history_frame = pd.DataFrame(cnn_history.history)
    return cnn_history_frame, cnn_model


@app.cell
def _(cnn_model, x_test_cnn, y_test):
    cnn_metrics = cnn_model.evaluate(x_test_cnn, y_test, verbose=0, return_dict=True)
    return cnn_metrics


@app.cell
def _(cnn_history_frame, dense_history_frame):
    _fig, _axes = plt.subplots(1, 2, figsize=(10, 4))
    _axes[0].plot(dense_history_frame["accuracy"], label="dense train")
    _axes[0].plot(dense_history_frame["val_accuracy"], label="dense val")
    _axes[0].plot(cnn_history_frame["accuracy"], label="cnn train")
    _axes[0].plot(cnn_history_frame["val_accuracy"], label="cnn val")
    _axes[0].set_title("Accuracy history")
    _axes[0].legend()
    _axes[1].plot(dense_history_frame["loss"], label="dense train")
    _axes[1].plot(dense_history_frame["val_loss"], label="dense val")
    _axes[1].plot(cnn_history_frame["loss"], label="cnn train")
    _axes[1].plot(cnn_history_frame["val_loss"], label="cnn val")
    _axes[1].set_title("Loss history")
    _axes[1].legend()
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(cnn_metrics, dense_metrics):
    metrics_frame = pd.DataFrame([{"model": "dense", **dense_metrics}, {"model": "cnn", **cnn_metrics}])
    metrics_frame
    return


@app.cell
def _(cnn_model, x_test_cnn, y_test):
    cnn_probs = cnn_model.predict(x_test_cnn, verbose=0)
    cnn_predictions = np.argmax(cnn_probs, axis=1)
    mistake_indices = np.where(cnn_predictions != y_test)[0][:6]
    return cnn_predictions, mistake_indices


@app.cell
def _(cnn_predictions, mistake_indices, x_test, y_test):
    _fig, _axes = plt.subplots(2, 3, figsize=(8, 5))
    for _axis, index in zip(_axes.ravel(), mistake_indices, strict=False):
        _axis.imshow(x_test[index], cmap="gray")
        _axis.set_title(f"pred={cnn_predictions[index]}, true={y_test[index]}")
        _axis.axis("off")
    _fig.tight_layout()
    _fig
    return


@app.cell
def _():
    has_pycaret = importlib.util.find_spec("pycaret") is not None
    if not has_pycaret:
        _warning = mo.callout("PyCaret section is skipped because `pycaret` is not installed.", kind="warn")
    return has_pycaret


@app.cell
def _(has_pycaret, pd, x_train, y_train):
    pycaret_summary = pd.DataFrame({"note": ["Install the specialized dependency group to run the PyCaret section."]})
    if has_pycaret:
        import os
        from pathlib import Path

        pycaret_log = Path(__file__).resolve().parents[3] / "build" / "pycaret" / "classification.log"
        pycaret_log.parent.mkdir(parents=True, exist_ok=True)
        os.environ["PYCARET_CUSTOM_LOGGING_PATH"] = str(pycaret_log)

        from pycaret.classification import ClassificationExperiment

        flat_train = x_train[:3000].reshape(3000, -1)
        pycaret_frame = pd.DataFrame(flat_train, columns=[f"px_{idx:03d}" for idx in range(flat_train.shape[1])])
        pycaret_frame["label"] = y_train[:3000]
        experiment = ClassificationExperiment()
        experiment.setup(
            data=pycaret_frame,
            target="label",
            session_id=123,
            fold=3,
            n_jobs=1,
            normalize=True,
            system_log=False,
            verbose=False,
        )
        experiment.compare_models(include=["lr", "rf"], n_select=2, verbose=False)
        pycaret_summary = experiment.pull()
    return pycaret_summary


@app.cell
def _(pycaret_summary):
    pycaret_summary
    return


if __name__ == "__main__":
    app.run()
