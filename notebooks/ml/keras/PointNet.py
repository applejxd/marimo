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
    # PointNet 風モデルで点群分類

    元 notebook は ModelNet10 をダウンロードして TensorBoard で学習していた。
    この版では大容量ダウンロードや UI 依存を避けるため、球・立方体・円柱・円錐の合成点群を自前生成する。
    PointNet の要点である `Conv1D + GlobalMaxPooling + T-Net` は残しつつ、短時間で export できる教材に整えている。
    """)
    return


@app.cell
def _():
    tf.keras.utils.set_random_seed(1234)
    device_name = "/GPU:0" if tf.config.list_physical_devices("GPU") else "/CPU:0"
    pd.DataFrame({"item": ["tensorflow", "device"], "value": [tf.__version__, device_name]})
    return device_name


@app.cell
def _():
    def rotation_matrix(rng):
        angle_z = rng.uniform(0.0, 2.0 * np.pi)
        angle_x = rng.uniform(-0.4, 0.4)
        cz, sz = np.cos(angle_z), np.sin(angle_z)
        cx, sx = np.cos(angle_x), np.sin(angle_x)
        rot_z = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]])
        rot_x = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]])
        return rot_z @ rot_x

    def sample_sphere(num_points, rng):
        points = rng.normal(size=(num_points, 3))
        return points / np.linalg.norm(points, axis=1, keepdims=True)

    def sample_cube(num_points, rng):
        points = rng.uniform(-1.0, 1.0, size=(num_points, 3))
        faces = rng.integers(0, 3, size=num_points)
        signs = rng.choice([-1.0, 1.0], size=num_points)
        points[np.arange(num_points), faces] = signs
        return points

    def sample_cylinder(num_points, rng):
        theta = rng.uniform(0.0, 2.0 * np.pi, size=num_points)
        z = rng.uniform(-1.0, 1.0, size=num_points)
        return np.column_stack([np.cos(theta), np.sin(theta), z])

    def sample_cone(num_points, rng):
        theta = rng.uniform(0.0, 2.0 * np.pi, size=num_points)
        height = rng.uniform(0.0, 1.0, size=num_points)
        radius = 1.0 - height
        return np.column_stack([radius * np.cos(theta), radius * np.sin(theta), 2.0 * height - 1.0])

    shape_generators = {"sphere": sample_sphere, "cube": sample_cube, "cylinder": sample_cylinder, "cone": sample_cone}
    return rotation_matrix, shape_generators


@app.cell
def _(rotation_matrix, shape_generators):
    def make_dataset(samples_per_class, num_points, seed):
        rng = np.random.default_rng(seed)
        clouds, labels = [], []
        for label, generator in enumerate(shape_generators.values()):
            for _ in range(samples_per_class):
                points = generator(num_points, rng)
                points = points @ rotation_matrix(rng).T
                points = points + rng.normal(scale=0.02, size=points.shape)
                clouds.append(points.astype(np.float32))
                labels.append(label)
        return np.stack(clouds), np.array(labels, dtype=np.int64)

    class_names = list(shape_generators)
    train_points, train_labels = make_dataset(80, 256, 1234)
    test_points, test_labels = make_dataset(20, 256, 4321)
    return class_names, test_labels, test_points, train_labels, train_points


@app.cell
def _(class_names, train_points):
    _fig = plt.figure(figsize=(10, 8))
    for idx, class_name in enumerate(class_names, start=1):
        _axis = _fig.add_subplot(2, 2, idx, projection="3d")
        _points = train_points[(idx - 1) * 80]
        _axis.scatter(_points[:, 0], _points[:, 1], _points[:, 2], s=4)
        _axis.set_title(class_name)
        _axis.set_axis_off()
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(test_labels, test_points, train_labels, train_points):
    train_dataset = tf.data.Dataset.from_tensor_slices((train_points, train_labels)).shuffle(len(train_points), seed=1234).batch(32).prefetch(tf.data.AUTOTUNE)
    test_dataset = tf.data.Dataset.from_tensor_slices((test_points, test_labels)).batch(32).prefetch(tf.data.AUTOTUNE)
    return test_dataset, train_dataset


@app.cell
def _():
    def conv_bn(x, filters):
        x = keras.layers.Conv1D(filters, kernel_size=1, padding="valid")(x)
        x = keras.layers.BatchNormalization(momentum=0.0)(x)
        return keras.layers.Activation("relu")(x)

    def dense_bn(x, units):
        x = keras.layers.Dense(units)(x)
        x = keras.layers.BatchNormalization(momentum=0.0)(x)
        return keras.layers.Activation("relu")(x)

    class OrthogonalRegularizer(keras.regularizers.Regularizer):
        def __init__(self, num_features, l2reg=0.001):
            self.num_features = num_features
            self.l2reg = l2reg
            self.eye = tf.eye(num_features)

        def __call__(self, weights):
            matrix = tf.reshape(weights, (-1, self.num_features, self.num_features))
            matrix_product = tf.matmul(matrix, matrix, transpose_b=True)
            return tf.reduce_sum(self.l2reg * tf.square(matrix_product - self.eye))

    def tnet(inputs, num_features):
        bias = keras.initializers.Constant(np.eye(num_features).flatten())
        regularizer = OrthogonalRegularizer(num_features)
        x = conv_bn(inputs, 32)
        x = conv_bn(x, 64)
        x = conv_bn(x, 128)
        x = keras.layers.GlobalMaxPooling1D()(x)
        x = dense_bn(x, 64)
        x = dense_bn(x, 32)
        transform = keras.layers.Dense(num_features * num_features, kernel_initializer="zeros", bias_initializer=bias, activity_regularizer=regularizer)(x)
        transform = keras.layers.Reshape((num_features, num_features))(transform)
        return keras.layers.Dot(axes=(2, 1))([inputs, transform])

    return conv_bn, dense_bn, tnet


@app.cell
def _(class_names, conv_bn, dense_bn, device_name, train_points, tnet):
    with tf.device(device_name):
        inputs = keras.Input(shape=(train_points.shape[1], train_points.shape[2]))
        x = tnet(inputs, 3)
        x = conv_bn(x, 32)
        x = conv_bn(x, 32)
        x = tnet(x, 32)
        x = conv_bn(x, 32)
        x = conv_bn(x, 64)
        x = conv_bn(x, 128)
        x = keras.layers.GlobalMaxPooling1D()(x)
        x = dense_bn(x, 128)
        x = keras.layers.Dropout(0.3)(x)
        x = dense_bn(x, 64)
        outputs = keras.layers.Dense(len(class_names), activation="softmax")(x)
        model = keras.Model(inputs=inputs, outputs=outputs, name="synthetic_pointnet")
    return model


@app.cell
def _(model, test_dataset, train_dataset):
    model.compile(loss="sparse_categorical_crossentropy", optimizer=keras.optimizers.Adam(learning_rate=1e-3), metrics=["sparse_categorical_accuracy"])
    history = model.fit(train_dataset, epochs=12, validation_data=test_dataset, verbose=0, callbacks=[keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)])
    history_frame = pd.DataFrame(history.history)
    test_metrics = model.evaluate(test_dataset, verbose=0, return_dict=True)
    return history_frame, test_metrics


@app.cell
def _(history_frame):
    _fig, _axes = plt.subplots(1, 2, figsize=(10, 4))
    _axes[0].plot(history_frame["loss"], label="train")
    _axes[0].plot(history_frame["val_loss"], label="validation")
    _axes[0].legend()
    _axes[0].set_title("Loss")
    _axes[1].plot(history_frame["sparse_categorical_accuracy"], label="train")
    _axes[1].plot(history_frame["val_sparse_categorical_accuracy"], label="validation")
    _axes[1].legend()
    _axes[1].set_title("Accuracy")
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(pd, test_metrics):
    pd.DataFrame([test_metrics])
    return


@app.cell
def _(model, np, test_labels, test_points):
    from sklearn.metrics import confusion_matrix

    probs = model.predict(test_points, verbose=0)
    predictions = np.argmax(probs, axis=1)
    confusion = confusion_matrix(test_labels, predictions)
    return confusion, predictions


@app.cell
def _(class_names, confusion):
    _fig, _ax = plt.subplots(figsize=(5, 4))
    _image = _ax.imshow(confusion, cmap="Purples")
    _ax.set_xticks(range(len(class_names)), class_names, rotation=30)
    _ax.set_yticks(range(len(class_names)), class_names)
    for row in range(confusion.shape[0]):
        for col in range(confusion.shape[1]):
            _ax.text(col, row, confusion[row, col], ha="center", va="center", fontsize=8)
    _ax.set_title("Confusion matrix")
    _fig.colorbar(_image, ax=_ax)
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(class_names, predictions, test_labels, test_points):
    _fig = plt.figure(figsize=(10, 8))
    for axis_index, sample_index in enumerate([0, 20, 40, 60], start=1):
        _axis = _fig.add_subplot(2, 2, axis_index, projection="3d")
        _points = test_points[sample_index]
        _axis.scatter(_points[:, 0], _points[:, 1], _points[:, 2], s=4)
        _axis.set_title(f"pred={class_names[predictions[sample_index]]}, true={class_names[test_labels[sample_index]]}")
        _axis.set_axis_off()
    _fig.tight_layout()
    _fig
    return


if __name__ == "__main__":
    app.run()
