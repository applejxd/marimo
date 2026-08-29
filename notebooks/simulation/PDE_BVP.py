import marimo

__generated_with = "0.24.0"
app = marimo.App()


with app.setup:
    import marimo as mo


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # 境界値問題と固有値問題
    legacy/simulation/PDE_BVP.ipynb を marimo 向けに整理し、1 次元シュレディンガー方程式と 2 次元ポアソン方程式を差分法で扱う。
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 前準備 (可視化等)
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    タイマー関数
    """)
    return


@app.cell
def _():
    import time
    from contextlib import contextmanager

    @contextmanager
    def timer(name):
        t0 = time.time()
        yield
        print(f'[{name}] done in {time.time() - t0:.0f} s')

    return (timer,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 1次元：シュレディンガー方程式の定常解

    - 境界値問題として解く (定常状態)
    - 境界条件は固定端条件 (無限井戸型ポテンシャル)
    - 解くのは固有値問題
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    シュレディンガー方程式は
    $$
      \left(-\frac{\hbar^2}{2m}\nabla^2+U(x)\right) \psi(x) = E \psi(x)
    $$
    である。
    1次元の場合に適切に無次元化すると
    $$
      \left(-\frac{d^2}{dx^2}+U(x)\right)\psi(x)=E\psi
    $$
    となる。
    ハミルトニアンに含まれる2階微分の単純な差分表示は
    $$
    \begin{aligned}
      &\left.\frac{d^2}{dx^2}\psi(x)\right|_{x=x_i}
      \sim\left.\frac{d}{dx}\frac{\psi(x+a/2)-\psi(x-a/2)}{a}\right|_{x=x_i} \\
      &\sim\left.\frac{d}{dx}\frac{\psi(x+a/2)-\psi(x-a/2)}{a}\right|_{x=x_i} \\
      &\sim\frac{\psi(x_i+a)-2\psi(x_i)+\psi(x_i-a)}{a^2}
    \end{aligned}
    $$
    となる。
    これはいわゆるハミルトニアンの行列表示である。
    """)
    return


@app.cell
def _():
    import matplotlib.pyplot as plt
    from scipy import sparse
    _grid_step = 0.02
    N = 400
    hamiltonian = (
        -sparse.eye(N, k=1) + 2 * sparse.identity(N) - sparse.eye(N, k=-1)
    ) / _grid_step**2
    plt.spy(hamiltonian, markersize=1)
    return hamiltonian, plt, sparse


@app.cell
def _(hamiltonian, plt):
    import numpy as np
    from scipy.sparse import linalg

    sn = 40
    eigenvalues, eigenvectors = linalg.eigsh(hamiltonian, k=sn, which="SM")
    eigenvectors = np.transpose(eigenvectors)

    level = np.arange(sn)
    print()
    plt.plot(level, eigenvalues[level])
    return eigenvectors, linalg, np


@app.cell
def _(eigenvectors, plt):
    for i in range(5):
        plt.plot(eigenvectors[i], label='level=%d' %i)
    plt.legend()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 2次元：ポアソン方程式

    - 境界値問題として解く
    - 2次元空間
    - 境界条件は固定端条件 (端で0)
    - 解くのは連立1次方程式
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    状況は[こちら](https://takun-physics.net/10186/)と同じ。
    $$
    \left(\frac{\partial^2}{\partial x^2}+\frac{\partial^2}{\partial y^2}\right)
    p(x,y)=b(x,y)
    $$
    を解く。

    グリッドは$N_x×N_y$にとって
    $$
    p(x_0+i\Delta x, y_0+j\Delta y)≡p(x_i,y_j)≡p_{ij}\equiv p_{i+j×N_x}
    $$
    と差分化した未知変数をベクトル形式にする。
    シュレディンガー方程式の場合と同様に差分を実施する。
    """)
    return


@app.cell
def _(plt, sparse):
    N_1 = 60
    _grid_step_2d = 1.0 / (N_1 + 1)
    axis_operator = sparse.diags(
        diagonals=[1.0, -2.0, 1.0],
        offsets=[-1, 0, 1],
        shape=(N_1, N_1),
        format="csr",
    ) / _grid_step_2d**2
    operator = sparse.kronsum(axis_operator, axis_operator, format="csr")
    plt.spy(operator, markersize=1)
    return N_1, operator


@app.cell
def _(N_1, linalg, np, operator, timer):
    source = np.zeros(N_1 * N_1)
    source[int(N_1 / 4 + N_1 * N_1 / 4)] = 100
    source[int(3 / 4 * N_1 + N_1 * 3 / 4 * N_1)] = -100
    with timer('CG method'):
        x, _ = linalg.cg(-operator, -source, maxiter=500)
    return (x,)


@app.cell
def _(N_1, np, plt, x):
    solution_mat = np.array([[x[row_idx + col_idx * N_1] for row_idx in range(N_1)] for col_idx in range(N_1)])
    plt.imshow(solution_mat, origin="lower")
    plt.show()
    return


if __name__ == "__main__":
    app.run()
