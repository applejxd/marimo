import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # 手書きニューラルネットによる数字分類

    元 notebook は OpenML の MNIST と PyCaret を経由していましたが、
    ここでは自前実装の順伝播・逆伝播を学ぶ意図を保ちつつ、
    追加ダウンロード不要の `load_digits()` を使います。
    """)
    return


@app.cell
def _():
    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split

    digits = load_digits()
    features = digits.data.astype(np.float64) / 16.0
    labels = digits.target.astype(np.int64)
    x_train, x_test, y_train_labels, y_test_labels = train_test_split(features, labels, test_size=0.25, random_state=71, stratify=labels)
    y_train = np.eye(10)[y_train_labels]
    y_test = np.eye(10)[y_test_labels]
    return digits, x_test, x_train, y_test, y_test_labels, y_train


@app.cell
def _(digits):
    _fig, _axes = plt.subplots(2, 5, figsize=(8, 4))
    for idx, _axis in enumerate(_axes.ravel()):
        _axis.imshow(digits.images[idx], cmap="gray_r")
        _axis.set_title(f"label={digits.target[idx]}")
        _axis.axis("off")
    _fig.tight_layout()
    _fig
    return


@app.cell
def _():
    from dataclasses import dataclass

    class Sigmoid:
        def __init__(self):
            self.out = None

        def forward(self, x):
            self.out = 1.0 / (1.0 + np.exp(-x))
            return self.out

        def backward(self, dout):
            return dout * self.out * (1.0 - self.out)

    @dataclass
    class Affine:
        w: np.ndarray
        b: np.ndarray
        x: np.ndarray | None = None
        dw: np.ndarray | None = None
        db: np.ndarray | None = None

        def forward(self, x):
            self.x = x
            return x @ self.w + self.b

        def backward(self, dout):
            self.dw = self.x.T @ dout
            self.db = np.sum(dout, axis=0)
            return dout @ self.w.T

    def softmax(x):
        shifted = x - np.max(x, axis=1, keepdims=True)
        exp_x = np.exp(shifted)
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def cross_entropy_error(y_pred, y_true):
        return -np.mean(np.sum(y_true * np.log(y_pred + 1e-7), axis=1))

    @dataclass
    class SoftmaxWithLoss:
        loss: float | None = None
        y_pred: np.ndarray | None = None
        y_true: np.ndarray | None = None

        def forward(self, x, y_true):
            self.y_true = y_true
            self.y_pred = softmax(x)
            self.loss = cross_entropy_error(self.y_pred, y_true)
            return self.loss

        def backward(self, dout=1.0):
            return dout * (self.y_pred - self.y_true) / self.y_true.shape[0]

    return Affine, Sigmoid, SoftmaxWithLoss


@app.cell
def _(Affine, Sigmoid, SoftmaxWithLoss):
    from collections import OrderedDict

    class MyNet:
        def __init__(self, input_size, hidden_sizes, output_size, weight_scale=0.05):
            self.params = {
                "W1": weight_scale * np.random.randn(input_size, hidden_sizes[0]),
                "b1": np.zeros(hidden_sizes[0]),
                "W2": weight_scale * np.random.randn(hidden_sizes[0], hidden_sizes[1]),
                "b2": np.zeros(hidden_sizes[1]),
                "W3": weight_scale * np.random.randn(hidden_sizes[1], output_size),
                "b3": np.zeros(output_size),
            }
            self.layers = OrderedDict([
                ("Affine1", Affine(self.params["W1"], self.params["b1"])),
                ("Sigmoid1", Sigmoid()),
                ("Affine2", Affine(self.params["W2"], self.params["b2"])),
                ("Sigmoid2", Sigmoid()),
                ("Affine3", Affine(self.params["W3"], self.params["b3"])),
            ])
            self.last_layer = SoftmaxWithLoss()

        def predict(self, x):
            for layer in self.layers.values():
                x = layer.forward(x)
            return x

        def loss(self, x, y_true):
            return self.last_layer.forward(self.predict(x), y_true)

        def accuracy(self, x, y_true):
            logits = self.predict(x)
            return float(np.mean(np.argmax(logits, axis=1) == np.argmax(y_true, axis=1)))

        def gradient(self, x, y_true):
            self.loss(x, y_true)
            dout = self.last_layer.backward()
            for layer in reversed(list(self.layers.values())):
                dout = layer.backward(dout)
            return {
                "W1": self.layers["Affine1"].dw,
                "b1": self.layers["Affine1"].db,
                "W2": self.layers["Affine2"].dw,
                "b2": self.layers["Affine2"].db,
                "W3": self.layers["Affine3"].dw,
                "b3": self.layers["Affine3"].db,
            }

    return MyNet


@app.cell
def _(MyNet, x_test, x_train, y_test, y_train):
    np.random.seed(71)
    rng = np.random.default_rng(71)
    network = MyNet(input_size=x_train.shape[1], hidden_sizes=[64, 32], output_size=y_train.shape[1])
    history_rows = []
    for iteration in range(1, 601):
        batch_indices = rng.choice(len(x_train), size=64, replace=False)
        x_batch = x_train[batch_indices]
        y_batch = y_train[batch_indices]
        gradients = network.gradient(x_batch, y_batch)
        for key in gradients:
            network.params[key] -= 0.4 * gradients[key]
        if iteration == 1 or iteration % 20 == 0:
            history_rows.append({"iteration": iteration, "loss": network.loss(x_batch, y_batch), "train_accuracy": network.accuracy(x_train, y_train), "test_accuracy": network.accuracy(x_test, y_test)})

    history_frame = pd.DataFrame(history_rows)
    test_predictions = np.argmax(network.predict(x_test), axis=1)
    return history_frame, test_predictions


@app.cell
def _(history_frame):
    history_frame.tail()
    return


@app.cell
def _(history_frame):
    _fig, _axes = plt.subplots(1, 2, figsize=(10, 4))
    _axes[0].plot(history_frame["iteration"], history_frame["loss"], marker="o")
    _axes[0].set_title("mini-batch loss")
    _axes[1].plot(history_frame["iteration"], history_frame["train_accuracy"], label="train")
    _axes[1].plot(history_frame["iteration"], history_frame["test_accuracy"], label="test")
    _axes[1].legend()
    _axes[1].set_title("accuracy")
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(pd, test_predictions, y_test_labels):
    from sklearn.metrics import accuracy_score, confusion_matrix

    accuracy = accuracy_score(y_test_labels, test_predictions)
    confusion = confusion_matrix(y_test_labels, test_predictions)
    metrics_frame = pd.DataFrame({"metric": ["test_accuracy"], "value": [accuracy]})
    return confusion, metrics_frame


@app.cell
def _(metrics_frame):
    metrics_frame
    return


@app.cell
def _(confusion):
    _fig, _ax = plt.subplots(figsize=(6, 5))
    _image = _ax.imshow(confusion, cmap="Blues")
    for row in range(confusion.shape[0]):
        for col in range(confusion.shape[1]):
            _ax.text(col, row, confusion[row, col], ha="center", va="center", fontsize=8)
    _ax.set_xlabel("predicted")
    _ax.set_ylabel("true")
    _ax.set_title("Confusion matrix")
    _fig.colorbar(_image, ax=_ax)
    _fig.tight_layout()
    _fig
    return


if __name__ == "__main__":
    app.run()
