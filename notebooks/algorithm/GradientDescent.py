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
    # 最急降下法

    円状に分布する点群へ円を当てはめる問題を題材に、最急降下法を実装する。
    勾配は解析的に導いて数値微分で検算し、1 歩の大きさはバックトラッキング
    直線探索で決める。収束の様子と当てはめ結果を図で確認する。
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

    _fig, _ax = plt.subplots(figsize=(4.4, 4.0), dpi=100)
    _ax.scatter(data[:, 0], data[:, 1], s=12)
    _ax.set_aspect("equal")
    _ax.set_xlabel("$x$")
    _ax.set_ylabel("$y$")
    _ax.set_title("generated points")
    _fig.tight_layout()
    _fig
    return (plt,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    このデータを中心・半径が未知の円として非線形フィッティング。
    MSSは
    $$
      f(x_0,y_0,r) = \frac{1}{N}\sum_{k=1}^N
        \left[\frac{(x_i-x_0)^2+(y_i-y_0)^2}{r^2} - 1\right]^2
    $$
    となる。
    ただし計算の都合上、無次元で平方根が含まれない形になっている。
    この関数の最適化を最急降下法で行う。
    """)
    return


@app.cell
def _(data):
    def residue(x0: float, y0: float, r: float) -> float:
        gap = ((data[:, 0] - x0) ** 2 + (data[:, 1] - y0) ** 2) / r**2 - 1
        return float((gap**2).mean())

    residue(0.1, -0.2, 10.0)
    return (residue,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    最急降下法は勾配 $\nabla f$ の逆向きへ進む。上の $f$ は各項が簡単なので、
    勾配は手で書き下せる。$g_i\equiv\dfrac{(x_i-x_0)^2+(y_i-y_0)^2}{r^2}-1$ と置くと
    $$
    \frac{∂f}{∂x_0}=\frac{1}{N}\sum_i2g_i\cdot\frac{-2(x_i-x_0)}{r^2},\quad
    \frac{∂f}{∂y_0}=\frac{1}{N}\sum_i2g_i\cdot\frac{-2(y_i-y_0)}{r^2},
    $$
    $$
    \frac{∂f}{∂r}=\frac{1}{N}\sum_i2g_i\cdot\frac{-2\left[(x_i-x_0)^2+(y_i-y_0)^2\right]}{r^3}
    $$
    である。手で導いた式は写し間違えやすいので、中心差分による数値微分と
    突き合わせて確認する。
    """)
    return


@app.cell
def _(data, np, residue):
    def gradient(x0: float, y0: float, r: float) -> np.ndarray:
        dx = data[:, 0] - x0
        dy = data[:, 1] - y0
        squared = dx**2 + dy**2
        gap = squared / r**2 - 1
        return np.array(
            [
                (2 * gap * (-2 * dx / r**2)).mean(),
                (2 * gap * (-2 * dy / r**2)).mean(),
                (2 * gap * (-2 * squared / r**3)).mean(),
            ]
        )

    _params = np.array([0.5, -0.6, 9.0])
    _numeric = np.zeros(3)
    _h = 1e-6
    for _index in range(3):
        _plus, _minus = _params.copy(), _params.copy()
        _plus[_index] += _h
        _minus[_index] -= _h
        _numeric[_index] = (residue(*_plus) - residue(*_minus)) / (2 * _h)

    f"解析勾配と数値微分の最大差 = {np.abs(gradient(*_params) - _numeric).max():.2e}"
    return (gradient,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    残る問題は 1 歩の大きさである。固定の学習率だと、大きすぎれば発散し、
    小さすぎれば進まない。ここではバックトラッキング直線探索を使い、
    Armijo 条件
    $$
    f(\vec{p}-s\nabla f)\leq f(\vec{p})-\alpha s\lVert\nabla f\rVert^2
    $$
    を満たすまで step $s$ を $\beta$ 倍に縮める。十分小さい $s$ では必ず成立するので、
    この探索は有限回で終わる。
    """)
    return


@app.cell
def _(np, residue):
    def back_tracking(
        params: np.ndarray,
        direction: np.ndarray,
        alpha: float = 0.3,
        beta: float = 0.5,
        initial: float = 1.0,
    ) -> float:
        """Armijo 条件を満たす step 幅を返す。"""
        step = initial
        current = residue(*params)
        slope = float(direction @ direction)
        while residue(*(params - step * direction)) > current - alpha * step * slope:
            step *= beta
            if step < 1e-12:
                break
        return step

    return (back_tracking,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    初期値を中心 $(0.5,-0.6)$・半径 $9$ として 200 回反復する。
    """)
    return


@app.cell
def _(back_tracking, gradient, np, residue):
    _params = np.array([0.5, -0.6, 9.0])
    history = [np.append(_params, residue(*_params))]
    for _step in range(200):
        _grad = gradient(*_params)
        _params = _params - back_tracking(_params, _grad) * _grad
        history.append(np.append(_params, residue(*_params)))
    history = np.array(history)

    (
        f"推定値: 中心 ({history[-1, 0]:.3f}, {history[-1, 1]:.3f}), 半径 {history[-1, 2]:.3f}",
        "真値: 中心 (0.1, -0.2), 半径 10（半径には標準偏差 0.5 のばらつきがある）",
        f"残差 {history[0, 3]:.4f} -> {history[-1, 3]:.6f}",
    )
    return (history,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    左は残差の推移（対数軸）、右は推定した円と観測点である。
    残差は最初の数十回で 1 桁以上落ち、そのあとは平坦になる。観測自体に
    半径のばらつきがあるため、残差はゼロにはならない。
    """)
    return


@app.cell
def _(data, history, np, plt):
    _fig, _axes = plt.subplots(1, 2, figsize=(9.6, 4.0), dpi=100)

    _axes[0].semilogy(history[:, 3])
    _axes[0].set_xlabel("iteration")
    _axes[0].set_ylabel("residue")
    _axes[0].set_title("convergence")
    _axes[0].grid(alpha=0.3)

    _theta = np.linspace(0, 2 * np.pi, 200)
    _x0, _y0, _r = history[-1, :3]
    _axes[1].scatter(data[:, 0], data[:, 1], s=12, label="observed")
    _axes[1].plot(_x0 + _r * np.cos(_theta), _y0 + _r * np.sin(_theta), color="crimson", label="fitted")
    _axes[1].set_aspect("equal")
    _axes[1].legend()
    _axes[1].set_title("fitted circle")

    _fig.tight_layout()
    _fig
    return


if __name__ == "__main__":
    app.run()
