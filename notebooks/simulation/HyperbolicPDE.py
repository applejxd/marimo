import marimo

__generated_with = "0.24.0"
app = marimo.App()


with app.setup:
    from pathlib import Path

    import marimo as mo


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # 双曲型偏微分方程式
    legacy/simulation/HyperbolicPDE.ipynb を marimo 向けに整理し、1 次元・2 次元の波動方程式を再現可能な GIF 出力付きで解きます。
    """)
    return


@app.cell
def _():
    import base64

    artifacts_dir = Path(__file__).resolve().parent / "_generated" / "HyperbolicPDE"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    def gif_image(path: Path, alt: str):
        encoded = base64.b64encode(path.read_bytes()).decode()
        return mo.image(f"data:image/gif;base64,{encoded}", alt=alt)

    return artifacts_dir, gif_image


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 双曲型: 波動方程式
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    波動方程式
    $$
        \frac{∂^2u}{∂t^2}=c^2\frac{∂^2u}{∂x^2}
    $$
    を解く。条件は
    - $c=1$
    - $x\in[0,1]$,
    - $t\in[0,0.4]$,
    - $u(0,t)=u(1,t)=0$ (境界条件).

    ただし数値不安定性を避けるために、
    時間方向の刻み幅$k$は空間方向の刻み幅$h$に対して
    $$
        k\leq\frac{h}{c}
    $$
    を満たすように選ぶ (CFL 条件)。
    """)
    return


@app.cell
def _():
    import numpy as np
    speed = 1.0
    x_diff = 0.02
    x_list = np.arange(0, 1, x_diff)
    N = len(x_list)
    t_diff = x_diff / speed
    t_list = np.arange(0, 0.4, t_diff)
    return N, np, t_list, x_diff, x_list


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### 初期条件
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    初期条件は
    $$
        u(x,0)=
        \begin{cases}
            &\dfrac{1}{2}\cos(8\pi(x-1/2))+\dfrac{1}{2}\quad(3/8\leq x\leq5/8), \\
            &0\quad(\text{otherwise}),
        \end{cases}
    $$
    および
    $$
        \frac{∂u}{∂t}(x,0)=0
    $$
    とする.
    """)
    return


@app.cell
def _(np, x_diff, x_list):
    import matplotlib.pyplot as plt

    u_start = 0.5 * np.cos(8 * np.pi * (x_list - 0.5)) + 0.5
    # otherwise
    u_start[0:int(3/8/x_diff)] = u_start[int(5/8/x_diff):] = 0

    plt.plot(x_list, u_start)
    plt.show()
    return plt, u_start


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### 係数行列
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    数値積分は連立1次方程式の形
    $$\begin{aligned}
        &\frac{∂u}{∂t}(x,t)≡v(x,t), \\
        &\frac{∂v}{∂t}=c^2\frac{∂^2u}{∂x^2}
    \end{aligned}$$
    で解く.係数行列は
    $$
        \begin{pmatrix}
            ∂_tu \\ ∂_tv
        \end{pmatrix}
        =
        \begin{pmatrix}
            O & I \\ c^2∂_x^2 & O
        \end{pmatrix}
        \begin{pmatrix}
            u \\ v
        \end{pmatrix}
    $$
    の差分化である。
    """)
    return


@app.cell
def _(N, np):
    import scipy.sparse as sparse

    # _1 はひとつ前の意
    u_1 = sparse.dia_array((np.ones(N), [-1]), shape=(N, N))

    # 行列の端をどれくらい表示するか
    np.set_printoptions(edgeitems=3)
    # 桁数をどうするか
    np.set_printoptions(precision=3)

    u_1.toarray()
    return sparse, u_1


@app.cell
def _(N, np, sparse):
    # 1 はひとつ後の意
    u1 = sparse.dia_array((np.ones(N), [1]), shape=(N, N))
    u1.toarray()
    return (u1,)


@app.cell
def _(N, sparse, u1, u_1, x_diff):
    id_mat = sparse.identity(N)

    diff_mat = (u1 -2 * id_mat + u_1) / (x_diff ** 2)
    diff_mat.toarray()
    return diff_mat, id_mat


@app.cell
def _(N, diff_mat, id_mat, sparse):
    _zero_mat = sparse.csr_matrix((N, N))
    coeff_mat = sparse.bmat([[_zero_mat, id_mat], [diff_mat, _zero_mat]])
    coeff_mat.toarray()
    return (coeff_mat,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### 積分
    """)
    return


@app.cell
def _(coeff_mat, np, t_list, u_start, x_list):
    from scipy.integrate import solve_ivp

    def _diff_operator(t, u_list):
        return coeff_mat @ u_list
    _u_sol = solve_ivp(
        _diff_operator,
        t_span=(0, 0.4),
        y0=np.hstack([u_start, np.zeros(len(x_list))]),
        method="RK45",
        dense_output=True,
        max_step=t_list[1] - t_list[0],
        rtol=1e-08,
    )
    u_list = _u_sol.sol(t_list).T[:, 0:len(x_list)]
    return (u_list,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### 可視化
    """)
    return


@app.cell
def _():
    import matplotlib.animation as animation

    return (animation,)


@app.cell
def _(animation, artifacts_dir, plt, t_list, u_list, x_list):
    _fig, _ax = plt.subplots()
    _ax.set_xlim([0, 1])
    _ax.set_ylim([0, 1.1])
    line, = _ax.plot([], [])

    def _animate(frame):
        t = t_list[frame]
        rho = u_list[frame]
        _ax.set_title(f"t={t:.3f}")
        line.set_data(x_list, rho)
        return (line,)
    wave_1d_gif = artifacts_dir / "wave_1d.gif"
    _ani = animation.FuncAnimation(_fig, _animate, frames=len(t_list), interval=50)
    _ani.save(wave_1d_gif, writer="pillow", dpi=120)
    plt.close(_fig)
    return (wave_1d_gif,)


@app.cell
def _(gif_image, wave_1d_gif):
    gif_image(wave_1d_gif, alt="1次元波動方程式の時間発展")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 双曲型・2次元
    """)
    return


@app.cell
def _(np):
    speed_2d = 1.0
    x_diff_1 = 0.05
    x_list_1 = np.arange(0, 1, x_diff_1)
    y_list = np.arange(0, 1, x_diff_1)
    t_diff_2d = x_diff_1 / speed_2d
    t_max = 0.8
    t_list_1 = np.arange(0, t_max, t_diff_2d)
    return t_list_1, t_max, x_diff_1, x_list_1, y_list


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    初期条件の設定（ソースを2個追加）
    """)
    return


@app.cell
def _(np, plt, x_diff_1, x_list_1):
    u_start_original = 0.5 * np.cos(8 * np.pi * (x_list_1 - 0.5)) + 0.5
    first_peak = u_start_original.copy()
    first_peak[:int(1 / 8 / x_diff_1)] = first_peak[int(3 / 8 / x_diff_1):] = 0
    first_peak = np.tensordot(first_peak, first_peak, axes=0)
    # 1次元と同様に cos 関数から削って作る
    second_peak = u_start_original.copy()
    # 1/8 < x < 3/8
    second_peak[:int(5 / 8 / x_diff_1)] = second_peak[int(7 / 8 / x_diff_1):] = 0
    second_peak = np.tensordot(second_peak, second_peak, axes=0)
    u_start_1 = first_peak + second_peak
    # 5/8 < x < 7/8
    # 初期条件
    plt.imshow(u_start_1)
    return (u_start_1,)


@app.cell
def _(sparse, x_list_1, y_list):
    x_num, y_num = (len(x_list_1), len(y_list))
    mat_num = x_num * y_num
    u_10 = sparse.lil_matrix(sparse.eye(mat_num, k=-1))
    u_10[0::y_num, :] = 0
    for _idx in range(0, x_num):
        u_10[_idx * y_num, _idx * y_num] = 1
    u_10 = sparse.csr_matrix(u_10)
    u_10.toarray()
    return mat_num, u_10, x_num, y_num


@app.cell
def _(mat_num, sparse, x_num, y_num):
    u10 = sparse.lil_matrix(sparse.eye(mat_num, k=1))
    u10[y_num - 1::y_num, :] = 0
    for _idx in range(0, x_num):
        u10[(_idx + 1) * y_num - 1, (_idx + 1) * y_num - 1] = 1
    u10 = sparse.csr_matrix(u10)
    u10.toarray()
    return (u10,)


@app.cell
def _(mat_num, sparse, y_num):
    u0_1 = sparse.lil_matrix(sparse.eye(mat_num, k=-y_num))
    u0_1[:y_num - 1, :] = 0
    for _idx in range(0, y_num):
        u0_1[_idx, _idx] = 1
    u0_1 = sparse.csr_matrix(u0_1)
    u0_1.toarray()
    return (u0_1,)


@app.cell
def _(mat_num, sparse, x_num, y_num):
    u01 = sparse.lil_matrix(sparse.eye(mat_num, k=y_num))
    u01[(x_num - 1) * y_num:, :] = 0
    for _idx in range(0, y_num):
        u01[(x_num - 1) * y_num + _idx, (x_num - 1) * y_num + _idx] = 1
    u01 = sparse.csr_matrix(u01)
    u01.toarray()
    return (u01,)


@app.cell
def _(mat_num, sparse, u01, u0_1, u10, u_10, x_diff_1):
    # 微分演算子
    id_mat_1 = sparse.identity(mat_num)
    diff_mat_1 = (-4 * id_mat_1 + u_10 + u10 + u0_1 + u01) / x_diff_1 ** 2
    diff_mat_1.toarray()
    return diff_mat_1, id_mat_1


@app.cell
def _(diff_mat_1, id_mat_1, mat_num, sparse):
    _zero_mat = sparse.csr_matrix((mat_num, mat_num))
    coeff_mat_1 = sparse.bmat([[_zero_mat, id_mat_1], [diff_mat_1, _zero_mat]])
    coeff_mat_1.toarray()
    return (coeff_mat_1,)


@app.cell
def _(coeff_mat_1, mat_num, np, solve_ivp, t_list_1, t_max, u_start_1):
    def _diff_operator(t, u_list):
        return coeff_mat_1 @ u_list
    _u_sol = solve_ivp(
        _diff_operator,
        t_span=(0, t_max),
        y0=np.hstack([u_start_1.reshape(-1), np.zeros(mat_num)]),
        method="RK45",
        dense_output=True,
        max_step=t_list_1[1] - t_list_1[0],
        rtol=1e-08,
    )
    u_list_1 = _u_sol.sol(t_list_1).T[:, :mat_num]
    return (u_list_1,)


@app.cell
def _(plt, u_list_1, x_num, y_num):
    plt.imshow(u_list_1[-1].reshape(x_num, y_num), origin="lower")
    return


@app.cell
def _(animation, artifacts_dir, plt, t_list_1, u_list_1, x_num, y_num):
    _fig, _ax = plt.subplots()
    graph = _ax.imshow(u_list_1[0].reshape(x_num, y_num), origin="lower")

    def _animate(frame):
        t = t_list_1[frame]
        _ax.set_title(f"t={t:.3f}")
        graph.set_data(u_list_1[frame].reshape(x_num, y_num))
        return (graph,)
    wave_2d_gif = artifacts_dir / "wave_2d.gif"
    _ani = animation.FuncAnimation(_fig, _animate, frames=len(t_list_1), interval=50)
    _ani.save(wave_2d_gif, writer="pillow", dpi=120)
    plt.close(_fig)
    return (wave_2d_gif,)


@app.cell
def _(gif_image, wave_2d_gif):
    gif_image(wave_2d_gif, alt="2次元波動方程式の時間発展")
    return


if __name__ == "__main__":
    app.run()
