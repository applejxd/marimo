import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd

    return mo, np, pd


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # NumPy / pandas のデータ型メモ

    NumPy の dtype と pandas の型がどう対応し、どこで暗黙の変換が起きるかを
    小さな例で確認する。
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## データ同士の変換
    """)
    return


@app.cell
def _(np, pd):
    array_2d = np.array([[2, 3, 5], [7, 11, 13]])
    frame_from_array = pd.DataFrame(array_2d, index=range(2, 4), columns=["A", "B", "C"])
    print(array_2d)
    print(type(frame_from_array))
    frame_from_array
    return array_2d, frame_from_array


@app.cell
def _(pd):
    series_example = pd.Series([2, 3, 5, 7, 11], index=range(1, 11, 2))
    print(series_example)
    print(type(series_example.values))
    print(series_example.values)
    return (series_example,)


@app.cell
def _(pd):
    data_frame_values = pd.DataFrame([[2, 3, 5], [7, 11, 13]], index=[4, 2], columns=["A", "B", "C"])
    data_frame_values
    print(type(data_frame_values.values))
    print(data_frame_values.values)
    return (data_frame_values,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 固定長データ型の宣言
    """)
    return


@app.cell
def _(np):
    np.zeros(5)
    return


@app.cell
def _(np):
    42 * np.ones((2, 3))
    return


@app.cell
def _(pd):
    fixed_series = pd.Series(42, index=range(5), dtype=int)
    fixed_series
    return


@app.cell
def _(pd):
    fixed_frame = pd.DataFrame(42, index=range(5), columns=["A", "B"], dtype=int)
    fixed_frame
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 可変長データ型の宣言
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ndarray を 1 次元配列として扱う場合
    """)
    return


@app.cell
def _(np):
    n1array_linear = np.empty(0)
    n1array_pairs = np.empty(0)
    for _index in range(5):
        n1array_linear = np.append(n1array_linear, _index)
        n1array_pairs = np.append(n1array_pairs, [_index, 2 * _index])
    print(type(n1array_linear))
    print(n1array_linear)
    print(n1array_pairs)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ndarray を 2 次元配列として扱う場合
    """)
    return


@app.cell
def _(np):
    n2array_variable = np.empty((0, 3))
    for _index in range(4):
        n2array_variable = np.append(
            n2array_variable,
            [[_index, 2 * _index, 3 * _index]],
            axis=0,
        )
    print(n2array_variable.shape)
    print(n2array_variable)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Series（1 次元配列）を扱う場合
    """)
    return


@app.cell
def _(pd):
    series_parts = [pd.Series([_index]) for _index in range(5)]
    variable_series = pd.concat(series_parts, ignore_index=True)
    variable_series
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    DataFrame（2 次元配列）を扱う場合
    """)
    return


@app.cell
def _(pd):
    variable_frame = pd.DataFrame(index=range(5), columns=[], dtype=float)
    variable_frame["Prime"] = [2, 3, 5, 7, 11]
    variable_frame["Fibonacci"] = [1, 1, 2, 3, 5]
    variable_frame
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 要素の格納と取り出し
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ndarray はスライスによって格納指定・取り出しが可能
    """)
    return


@app.cell
def _(np):
    sliced_array = np.zeros((5, 7))
    sliced_array[2:4, 3:7] = 42
    print(sliced_array)
    print(sliced_array[1:3, 2:5])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    DataFrame のラベル名による指定
    """)
    return


@app.cell
def _(pd):
    label_frame = pd.DataFrame(0, index=range(5), columns=["A", "B", "C"], dtype=int)
    label_frame.loc[1:3, ["B"]] = 42
    label_frame
    label_frame.loc[2:4, ["B", "C"]]
    return (label_frame,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    DataFrame の番号による指定
    """)
    return


@app.cell
def _(label_frame):
    label_frame.iloc[1:3, 1:]
    return


if __name__ == "__main__":
    app.run()
