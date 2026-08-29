import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import pandas as pd
    import tensorflow as tf


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # TensorFlow テンソル操作メモ

    変形・拡張・反復・分割・結合・縮約といった基本操作を、
    小さなテンソルで順に確認する。乱数シードを固定しているので、
    export 実行時も同じ結果を再現できる。
    """)
    return


@app.cell
def _():
    tf.keras.utils.set_random_seed(42)
    device_summary = pd.DataFrame(
        {
            "device_type": ["CPU", "GPU"],
            "visible_devices": [
                len(tf.config.list_physical_devices("CPU")),
                len(tf.config.list_physical_devices("GPU")),
            ],
        }
    )
    device_summary
    return


@app.cell
def _():
    a_tensor = tf.random.uniform(shape=[4, 2, 3], minval=-10, maxval=10, dtype=tf.float64)
    b_tensor = tf.constant([[[1, 3, 5], [7, 11, 13]]], dtype=tf.float64)
    return a_tensor, b_tensor


@app.cell
def _(a_tensor, b_tensor):
    reshaped = tf.reshape(a_tensor, shape=(-1, 3, 2))
    expanded = tf.expand_dims(a_tensor, axis=2)
    padded = tf.pad(a_tensor, paddings=[[0, 0], [0, 0], [0, 1]])
    tiled = tf.tile(a_tensor, multiples=(1, 1, 2))
    tiled_with_axis = tf.tile(tf.expand_dims(a_tensor, 0), [2, 1, 1, 1])
    split_tensors = tf.split(a_tensor, num_or_size_splits=[2, 1], axis=-1)
    concatenated = tf.concat([a_tensor, b_tensor], axis=0)
    squeezed = tf.squeeze(tf.expand_dims(a_tensor, axis=2))
    return concatenated, expanded, padded, reshaped, split_tensors, squeezed, tiled, tiled_with_axis


@app.cell
def _(a_tensor, b_tensor, concatenated, expanded, padded, pd, reshaped, split_tensors, squeezed, tiled, tiled_with_axis):
    operation_table = pd.DataFrame(
        [
            ("original a_tensor", tuple(a_tensor.shape)),
            ("original b_tensor", tuple(b_tensor.shape)),
            ("reshape(-1, 3, 2)", tuple(reshaped.shape)),
            ("expand_dims(axis=2)", tuple(expanded.shape)),
            ("pad(last axis +1)", tuple(padded.shape)),
            ("tile(multiples=(1,1,2))", tuple(tiled.shape)),
            ("tile(expand_dims, [2,1,1,1])", tuple(tiled_with_axis.shape)),
            ("split sizes [2,1]", [tuple(item.shape) for item in split_tensors]),
            ("concat(axis=0)", tuple(concatenated.shape)),
            ("squeeze(expand_dims(...))", tuple(squeezed.shape)),
        ],
        columns=["operation", "shape"],
    )
    operation_table
    return


@app.cell
def _(concatenated, expanded, padded, split_tensors):
    mo.tabs({"expanded": expanded, "padded": padded, "split[0]": split_tensors[0], "split[1]": split_tensors[1], "concatenated": concatenated})
    return


if __name__ == "__main__":
    app.run()
