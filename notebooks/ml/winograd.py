import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import numpy as np
    import pandas as pd
    from scipy.signal import correlate2d


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Winograd 畳み込みの最小例

    4×4 の入力と 3×3 のカーネルに対して、通常の相互相関と
    Winograd 最小フィルタリング `F(2×2, 3×3)` を比較します。
    深層学習ライブラリの `conv2d` は信号処理の畳み込みではなく
    相互相関として実装されることが多いため、ここでも
    `scipy.signal.correlate2d` を基準にします。
    """)
    return


@app.cell
def _():
    input_matrix = np.array(
        [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0], [13.0, 14.0, 15.0, 16.0]]
    )
    kernel = np.array([[1.0, 0.0, -1.0], [1.0, 0.0, -1.0], [1.0, 0.0, -1.0]])
    standard_output = correlate2d(input_matrix, kernel, mode="valid")
    return input_matrix, kernel, standard_output


@app.cell
def _():
    g_matrix = np.array([[1.0, 0.0, 0.0], [0.5, 0.5, 0.5], [0.5, -0.5, 0.5], [0.0, 0.0, 1.0]])
    b_matrix = np.array([[1.0, 0.0, -1.0, 0.0], [0.0, 1.0, 1.0, 0.0], [0.0, -1.0, 1.0, 0.0], [0.0, 1.0, 0.0, -1.0]])
    a_matrix = np.array([[1.0, 1.0, 1.0, 0.0], [0.0, 1.0, -1.0, -1.0]])
    return a_matrix, b_matrix, g_matrix


@app.cell
def _(a_matrix, b_matrix, g_matrix, input_matrix, kernel):
    transformed_kernel = g_matrix @ kernel @ g_matrix.T
    transformed_input = b_matrix @ input_matrix @ b_matrix.T
    hadamard_product = transformed_kernel * transformed_input
    winograd_output = a_matrix @ hadamard_product @ a_matrix.T
    return hadamard_product, transformed_input, transformed_kernel, winograd_output


@app.cell
def _(hadamard_product, standard_output, transformed_input, transformed_kernel, winograd_output):
    comparison = pd.DataFrame(
        {
            "standard": standard_output.reshape(-1),
            "winograd": winograd_output.reshape(-1),
            "abs_error": np.abs(standard_output - winograd_output).reshape(-1),
        },
        index=["(0,0)", "(0,1)", "(1,0)", "(1,1)"],
    )
    transform_shapes = pd.DataFrame(
        {
            "array": ["GgGᵀ", "BdBᵀ", "elementwise product"],
            "shape": [tuple(transformed_kernel.shape), tuple(transformed_input.shape), tuple(hadamard_product.shape)],
        }
    )
    return comparison, transform_shapes


@app.cell
def _(comparison, transform_shapes):
    mo.vstack([transform_shapes, comparison])
    return


@app.cell
def _(comparison):
    print(f"allclose={np.allclose(comparison['standard'], comparison['winograd'])}")
    print(f"max_abs_error={comparison['abs_error'].max():.6f}")
    return


if __name__ == "__main__":
    app.run()
