import marimo

__generated_with = "0.24.0"
app = marimo.App()


with app.setup:
    from pathlib import Path

    import marimo as mo


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # 放物型偏微分方程式
    legacy/simulation/ParabolicPDE.ipynb を marimo 向けに整理し、拡散方程式・バーガース方程式・KdV 方程式を再現可能なアーティファクト生成付きで解きます。
    """)
    return


@app.cell
def _():
    import base64

    artifacts_dir = Path(__file__).resolve().parent / "_generated" / "ParabolicPDE"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    def gif_image(path: Path, alt: str):
        encoded = base64.b64encode(path.read_bytes()).decode()
        return mo.image(f"data:image/gif;base64,{encoded}", alt=alt)

    return artifacts_dir, gif_image


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 基本的な放物線型: 拡散方程式
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    拡散方程式は
    \begin{equation}
        \frac{∂u}{∂t}=D\nabla^2u.
    \end{equation}

    まずは1次元の場合で、適切に規格化して$D=1$として解く.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    まず数値計算の設定を行う
    """)
    return


@app.cell
def _():
    import numpy as np
    from scipy import sparse

    x_min, x_max = 0.02, 4
    x_list = np.arange(-x_max, x_max, x_min)
    return np, sparse, x_list, x_min


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    中心差分では
    \begin{equation}
        \frac{∂^2u}{∂x^2}(x_i)
        \simeq \frac{1}{h^2}(u(x_{i+1})-2u(x_i)+u(x_{i-1})).
    \end{equation}
    境界条件は十分遠方では0となる想定で課す。
    今回はディリクレ境界条件で
    \begin{equation}
        u(x_0)=U(x_N)=0
    \end{equation}
    とする。
    いつでも適切に$u$を再定義することで、境界の値をこのように設定することができる。

    これに対応して係数行列を修正する。
    """)
    return


@app.cell
def _(sparse, x_list, x_min):
    _u_1 = sparse.lil_matrix(sparse.eye(len(x_list), k=-1))
    _u_1[0, :] = 0
    _u_1 = sparse.csr_matrix(_u_1)
    _u0 = sparse.identity(len(x_list))
    _u1 = sparse.lil_matrix(sparse.eye(len(x_list), k=1))
    _u1[-1, :] = 0
    _u1 = sparse.csr_matrix(_u1)
    coeff_mat = (_u1 - 2 * _u0 + _u_1) / x_min ** 2
    return (coeff_mat,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    初期条件と積分の詳細を決める。

    [数値解の不安定性より時間方向の刻み幅 $k$ は空間方向の刻み幅 $h$ に対して十分に小さく取る必要がある](http://www.nibb.ac.jp/miyakohp/asari/htdocs/?page_id=60)。

    具体的には
    \begin{equation}
        \frac{Dk}{h^2}\leq\frac{1}{2}.
    \end{equation}
    これは係数行列の最大固有値の絶対値が1以下になる条件である.
    """)
    return


@app.cell
def _(np, x_list, x_min):
    # デルタ関数の差分化
    u_start = np.zeros(len(x_list))
    u_start[int(len(x_list)/2)] = 1/x_min

    t_min, t_max = 0.001, 0.04
    t_list = np.arange(0, t_max, t_min)
    return t_list, t_max, u_start


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    scipy で数値積分
    """)
    return


@app.cell
def _(coeff_mat, t_list, t_max, u_start):
    from scipy.integrate import solve_ivp

    def diff_operator(t, u_list):
        return coeff_mat * u_list
    _u_sol = solve_ivp(
        diff_operator,
        t_span=(0, t_max),
        y0=u_start,
        method="RK45",
        dense_output=True,
        max_step=t_list[1] - t_list[0],
        rtol=1e-08,
    )
    u_list = _u_sol.sol(t_list).T
    return (u_list,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    可視化に必要な matplotlib を読み込む
    """)
    return


@app.cell
def _():
    import matplotlib.animation as animation
    import matplotlib.pyplot as plt

    return animation, plt


@app.cell
def _(animation, artifacts_dir, plt, t_list, u_list, x_list):
    _fig, _ax = plt.subplots()
    _ax.set_xlim([-1.5, 1.5])
    _ax.set_ylim([0, 5])
    _line, = _ax.plot([], [])

    def _animate(frame):
        t = t_list[frame]
        rho = u_list[frame]
        _ax.set_title(f"t={t:.3f}")
        _line.set_data(x_list, rho)
        return (_line,)
    diffusion_1d_gif = artifacts_dir / "diffusion_1d.gif"
    _ani = animation.FuncAnimation(_fig, _animate, frames=len(t_list), interval=50)
    _ani.save(diffusion_1d_gif, writer="pillow", dpi=120)
    plt.close(_fig)
    return (diffusion_1d_gif,)


@app.cell
def _(diffusion_1d_gif, gif_image):
    gif_image(diffusion_1d_gif, alt="1次元拡散方程式の時間発展")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 非線形性のある放物線型: バーガース方程式
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    バーガース方程式は以下:
    \begin{equation}
    \frac{∂u(x,t)}{∂t}=-u(x,t)\frac{∂u(x,t)}{∂x}+D\frac{∂^2u(x,t)}{∂x^2}.
    \end{equation}

    以下の条件で解く:
    - $D=0.01$
    - $t\in[0,2], x\in[0,1]$
    - $u(0,t)=u(1,t)=0$
    - $u(x,0)=\sin(2\pi x)$
    """)
    return


@app.cell
def _(np):
    diffusion_const = 0.01
    x_diff = 0.02
    x_list_1 = np.arange(0, 1, x_diff)
    t_diff = 0.5 * x_diff ** 2 / diffusion_const
    t_list_1 = np.arange(0, 2, t_diff)
    u_start_1 = np.sin(2 * np.pi * x_list_1)
    return diffusion_const, t_list_1, u_start_1, x_diff, x_list_1


@app.cell
def _(diffusion_const, sparse, x_diff, x_list_1):
    _u_1 = sparse.lil_matrix(sparse.eye(len(x_list_1), k=-1))
    _u_1[0, :] = 0
    _u_1 = sparse.csr_matrix(_u_1)
    _u0 = sparse.identity(len(x_list_1))
    _u1 = sparse.lil_matrix(sparse.eye(len(x_list_1), k=1))
    _u1[-1, :] = 0
    _u1 = sparse.csr_matrix(_u1)
    du = (_u1 - _u_1) / (2 * x_diff)
    ddu = (_u1 - 2 * _u0 + _u_1) / x_diff ** 2

    def diff_operator_1(t, u_list):
        return -u_list * (du * u_list) + diffusion_const * (ddu * u_list)

    return (diff_operator_1,)


@app.cell
def _(diff_operator_1, solve_ivp, t_list_1, u_start_1):
    _u_sol = solve_ivp(
        diff_operator_1,
        t_span=(0, 2),
        y0=u_start_1,
        method="RK45",
        dense_output=True,
        max_step=t_list_1[1] - t_list_1[0],
        rtol=1e-08,
    )
    u_list_1 = _u_sol.sol(t_list_1).T
    return (u_list_1,)


@app.cell
def _(animation, artifacts_dir, plt, t_list_1, u_list_1, x_list_1):
    _fig, _ax = plt.subplots()
    _ax.set_xlim([x_list_1[0], x_list_1[-1]])
    _ax.set_ylim([-1, 1])
    _line, = _ax.plot([], [])

    def _animate(frame):
        t = t_list_1[frame]
        rho = u_list_1[frame]
        _ax.set_title(f"t={t:.3f}")
        _line.set_data(x_list_1, rho)
        return (_line,)
    burgers_1d_gif = artifacts_dir / "burgers_1d.gif"
    _ani = animation.FuncAnimation(_fig, _animate, frames=len(t_list_1), interval=20)
    _ani.save(burgers_1d_gif, writer="pillow", dpi=120)
    plt.close(_fig)
    return (burgers_1d_gif,)


@app.cell
def _(burgers_1d_gif, gif_image):
    gif_image(burgers_1d_gif, alt="バーガース方程式の時間発展")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### 非線形と3階微分を含む放物線型: KdV 方程式

    - 初期値問題として解く
    - 空間方向は周期境界条件
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    [python で学ぶ計算物理](http://www.physics.okayama-u.ac.jp/~otsuki/lecture/CompPhys2/pde/kdv.html)から移植。典型的なパラメータの

    \begin{equation}
    \frac{∂u(x,t)}{∂t}+6u(x,t)\frac{∂u(x,t)}{∂x}+\frac{∂^3u(x,t)}{∂x^3}=0
    \end{equation}

    を数値的に解く。
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    空間差分による係数行列を定義
    """)
    return


@app.cell
def _(np, sparse):
    def make_differential_ops(nx: int, dx: float):
        # 周期境界条件
        f0 = np.identity(nx, dtype=int)  # f_{i}
        f1 = np.roll(f0, 1, axis=1)  # f_{i+1}
        f2 = np.roll(f0, 2, axis=1)  # f_{i+2}
        f_1 = f1.transpose()  # f_{i-1}
        f_2 = f2.transpose()  # f_{i-2}

        # (f_{i+1} - f_{i-1}) / (2 dx)
        deriv1 = sparse.csr_matrix(f1 - f_1) / (2.0 * dx)

        # (f_{i+1} - 2f_{i} + f_{i-1}) / (dx^2)
        deriv2 = sparse.csr_matrix(f1 - 2.0 * f0 + f_1) / dx**2

        # (f_{i+2} - 2f_{i+1} + 2f_{i-1} - f_{i-2}) / (2 dx^3)
        deriv3 = sparse.csr_matrix(f2 - 2.0 * f1 + 2.0 * f_1 - f_2) / (2.0 * dx**3)

        return deriv1, deriv2, deriv3

    return (make_differential_ops,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    KdV 方程式の空間差分による係数行列を定義
    """)
    return


@app.function
def f_kdv(t, u, df1, df3):
    u_x = df1.dot(u)
    u_xxx = df3.dot(u)
    return -6.0 * u * u_x - u_xxx


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    時間方向へ数値積分を実施し、再利用できる `.npz` アーティファクトも保存する
    """)
    return


@app.cell
def _(artifacts_dir, make_differential_ops, np):
    from scipy.integrate import solve_ivp as _solve_ivp

    nx = 200
    kdv_x_max = 100.0
    x = np.linspace(0, kdv_x_max, nx, endpoint=False)
    dx = x[1] - x[0]
    u0_kdv = np.sin(x * (2.0 * np.pi / kdv_x_max))
    op_df1, _, op_df3 = make_differential_ops(nx, dx)
    kdv_t_max = 4.0
    nt = 61
    t = np.linspace(0, kdv_t_max, nt)
    sol = _solve_ivp(
        f_kdv,
        (0, kdv_t_max),
        u0_kdv,
        args=(op_df1, op_df3),
        dense_output=True,
        max_step=t[1] - t[0],
        rtol=1e-08,
    )
    u_tx = sol.sol(t).T
    kdv_dataset_path = artifacts_dir / "kdv_solve_ivp.npz"
    np.savez(kdv_dataset_path, x=x, t=t, u_tx=u_tx)
    return kdv_dataset_path, x, t, u_tx


@app.cell
def _(animation, plt):
    def save_animation(x, t, u_tx, ymin, ymax, filename):
        _fig, _ax = plt.subplots()
        _ax.set_xlabel("$x$")
        _ax.set_ylabel("$u(x)$")
        _ax.set_ylim((ymin, ymax))
        artists = []
        for t_idx in range(t.size):
            artist = _ax.plot(x, u_tx[t_idx, :], "-b")
            artist = artist + [_ax.text(0.05, 1.05, f"t = {t[t_idx]:.2f}", transform=_ax.transAxes)]
            artists.append(artist)
        anim = animation.ArtistAnimation(_fig, artists, interval=100, repeat=False)
        anim.save(filename, writer="pillow")
        plt.close(_fig)

    return (save_animation,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    保存した `.npz` を読み直してアニメーションを描画
    """)
    return


@app.cell
def _(artifacts_dir, kdv_dataset_path, np, save_animation):
    with np.load(kdv_dataset_path) as npz:
        x_kdv = npz["x"]
        t_kdv = npz["t"]
        u_tx_kdv = npz["u_tx"]

    kdv_gif = artifacts_dir / "kdv_solve_ivp.gif"
    save_animation(x_kdv, t_kdv, u_tx_kdv, ymin=-1.5, ymax=3.0, filename=kdv_gif)
    return (kdv_gif,)


@app.cell
def _(gif_image, kdv_gif):
    gif_image(kdv_gif, alt="KdV 方程式の時間発展")
    return


if __name__ == "__main__":
    app.run()
