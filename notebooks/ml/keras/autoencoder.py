import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import tensorflow as tf
    from tensorflow import keras


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Keras オートエンコーダで異常検知

    元 notebook は Kaggle の credit-card fraud データセットを API token で取得していた。
    この marimo 版では秘密情報を使わずに再現できることを優先し、
    `sklearn.make_classification` で作る不均衡な疑似トランザクションデータへ置き換えている。
    正常データで自己再構成を学習し、再構成誤差で異常を検出する意図はそのままである。
    """)
    return


@app.cell
def _():
    tf.keras.utils.set_random_seed(42)
    device_name = "/GPU:0" if tf.config.list_physical_devices("GPU") else "/CPU:0"
    runtime_frame = pd.DataFrame({"package": ["tensorflow", "keras", "numpy", "device"], "value": [tf.__version__, tf.keras.__version__, np.__version__, device_name]})
    runtime_frame
    return device_name


@app.cell
def _():
    from sklearn.datasets import make_classification

    feature_names = [f"V{i:02d}" for i in range(1, 29)] + ["Amount", "Velocity"]
    features, labels = make_classification(n_samples=12000, n_features=30, n_informative=12, n_redundant=10, n_clusters_per_class=1, weights=[0.992, 0.008], class_sep=2.0, flip_y=0.001, random_state=42)
    transactions = pd.DataFrame(features, columns=feature_names)
    transactions["Class"] = labels
    return transactions


@app.cell
def _(transactions):
    class_balance = transactions["Class"].value_counts().rename_axis("Class").reset_index(name="count")
    class_balance["ratio"] = class_balance["count"] / class_balance["count"].sum()
    class_balance
    return


@app.cell
def _(transactions):
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    train_frame, test_frame = train_test_split(transactions, test_size=0.33, random_state=2018, stratify=transactions["Class"])
    train_normal = train_frame.loc[train_frame["Class"] == 0].drop(columns="Class")
    test_features = test_frame.drop(columns="Class")
    test_labels = test_frame["Class"].to_numpy()
    scaler = StandardScaler()
    x_train = scaler.fit_transform(train_normal)
    x_test = scaler.transform(test_features)
    return test_labels, x_test, x_train


@app.cell
def _(device_name, x_train):
    with tf.device(device_name):
        autoencoder = keras.Sequential([
            keras.layers.Input(shape=(x_train.shape[1],)),
            keras.layers.Dense(24, activation="relu"),
            keras.layers.Dense(12, activation="relu"),
            keras.layers.Dense(4, activation="relu", name="bottleneck"),
            keras.layers.Dense(12, activation="relu"),
            keras.layers.Dense(24, activation="relu"),
            keras.layers.Dense(x_train.shape[1], activation="linear"),
        ], name="transaction_autoencoder")
        autoencoder.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3), loss="mse")
        history = autoencoder.fit(x_train, x_train, validation_split=0.1, epochs=20, batch_size=256, verbose=0)
    history_frame = pd.DataFrame(history.history)
    return autoencoder, history_frame


@app.cell
def _(history_frame):
    history_frame.tail()
    return


@app.cell
def _(history_frame):
    _fig, _ax = plt.subplots(figsize=(6, 4))
    _ax.plot(history_frame.index + 1, history_frame["loss"], label="train")
    _ax.plot(history_frame.index + 1, history_frame["val_loss"], label="validation")
    _ax.set_xlabel("epoch")
    _ax.set_ylabel("mse")
    _ax.set_title("Autoencoder reconstruction loss")
    _ax.legend()
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(autoencoder, x_test, x_train):
    train_reconstruction = autoencoder.predict(x_train, verbose=0)
    test_reconstruction = autoencoder.predict(x_test, verbose=0)
    train_errors = np.mean(np.square(x_train - train_reconstruction), axis=1)
    test_errors = np.mean(np.square(x_test - test_reconstruction), axis=1)
    threshold = float(np.quantile(train_errors, 0.99))
    predicted_labels = (test_errors >= threshold).astype(int)
    return predicted_labels, test_errors, threshold, train_errors


@app.cell
def _(pd, predicted_labels, test_errors, test_labels, threshold):
    from sklearn.metrics import (
        average_precision_score,
        confusion_matrix,
        precision_recall_curve,
        precision_recall_fscore_support,
        roc_auc_score,
        roc_curve,
    )

    precision, recall, _ = precision_recall_curve(test_labels, test_errors)
    fpr, tpr, _ = roc_curve(test_labels, test_errors)
    prf = precision_recall_fscore_support(test_labels, predicted_labels, average="binary", zero_division=0)
    confusion = confusion_matrix(test_labels, predicted_labels)
    metrics_frame = pd.DataFrame({"metric": ["Average Precision", "ROC AUC", "Precision@threshold", "Recall@threshold", "F1@threshold", "threshold"], "value": [average_precision_score(test_labels, test_errors), roc_auc_score(test_labels, test_errors), prf[0], prf[1], prf[2], threshold]})
    curves = {"precision": precision, "recall": recall, "fpr": fpr, "tpr": tpr}
    return confusion, curves, metrics_frame


@app.cell
def _(metrics_frame):
    metrics_frame
    return


@app.cell
def _(curves, test_errors, test_labels, threshold, train_errors):
    _fig, _axes = plt.subplots(1, 3, figsize=(15, 4))
    _axes[0].plot(curves["recall"], curves["precision"], color="tab:blue")
    _axes[0].set_xlabel("recall")
    _axes[0].set_ylabel("precision")
    _axes[0].set_title("Precision-Recall")
    _axes[1].plot(curves["fpr"], curves["tpr"], color="tab:red")
    _axes[1].plot([0, 1], [0, 1], linestyle="--", color="black")
    _axes[1].set_xlabel("false positive rate")
    _axes[1].set_ylabel("true positive rate")
    _axes[1].set_title("ROC")
    _axes[2].hist(train_errors, bins=40, alpha=0.6, label="train normal", density=True)
    _axes[2].hist(test_errors[test_labels == 0], bins=40, alpha=0.5, label="test normal", density=True)
    _axes[2].hist(test_errors[test_labels == 1], bins=40, alpha=0.5, label="test anomaly", density=True)
    _axes[2].axvline(threshold, color="black", linestyle="--", label="threshold")
    _axes[2].set_xlabel("reconstruction error")
    _axes[2].set_title("Error distribution")
    _axes[2].legend()
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(confusion):
    _fig, _ax = plt.subplots(figsize=(4, 4))
    _image = _ax.imshow(confusion, cmap="Reds")
    for row in range(confusion.shape[0]):
        for col in range(confusion.shape[1]):
            _ax.text(col, row, confusion[row, col], ha="center", va="center")
    _ax.set_xlabel("predicted")
    _ax.set_ylabel("true")
    _ax.set_title("Confusion matrix at threshold")
    _fig.colorbar(_image, ax=_ax)
    _fig.tight_layout()
    _fig
    return


if __name__ == "__main__":
    app.run()
