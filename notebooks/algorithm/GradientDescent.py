import marimo

__generated_with = "0.24.0"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <a href="https://colab.research.google.com/github/applejxd/colaboratory/blob/master/GradientDescent.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    原点からずれた点を中心に,、近似的に円状に分布する点を生成する関数
    """)
    return


@app.cell
def _():
    from typing import List

    import numpy as np

    # 再現性のためシードは固定
    np.random.seed(42)

    def get_point() -> List[float]:
        r = 10. + np.random.normal(0,0.5)
        center = [0.1, -0.2]

        theta = np.random.uniform(0, 2*np.pi)
        return [r*np.cos(theta)+center[0], r*np.sin(theta)+center[1]]
    get_point()
    return get_point, np


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    データ生成
    """)
    return


@app.cell
def _(get_point, np):
    data_list = []
    for k in range(50):
        data_list.append(get_point())

    data = np.array(data_list)
    data
    return (data,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    プロット
    """)
    return


@app.cell
def _(data):
    import matplotlib.pyplot as plt

    # アスペクト比 1:1
    plt.axes().set_aspect('equal', 'datalim')
    # 散布図
    plt.scatter(data[:,0],data[:,1])
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    このデータを中心・半径が未知の円として非線形フィッティング。
    MSSは
    \begin{equation}
      f(x_0,y_0,r) = \frac{1}{N}\sum_{k=1}^N
        \left[\frac{(x_i-x_0)^2+(y_i-y_0)^2}{r^2} - 1\right]^2
    \end{equation}
    となる。
    ただし計算の都合上、無次元で平方根が含まれない形になっている。
    この関数の最適化を最急降下法で行う。
    """)
    return


@app.cell
def _(data):
    def residue(x0, y0, r) -> float:
        result = 0.
        for k in range(len(data)):
            result += (((data[k,0]-x0)**2+(data[k,1]-y0)**2)/r**2-1)**2
        return result/len(data)
    residue(0.1, -0.2, 10.)
    return


@app.function
def back_tracking():
    return 0.5


if __name__ == "__main__":
    app.run()
