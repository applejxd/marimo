import marimo

__generated_with = "0.24.0"
app = marimo.App()


with app.setup:
    from pathlib import Path

    import marimo as mo


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # 有限要素法と KdV 方程式
    legacy/simulation/FEM.ipynb を marimo 向けに整理し、前半では 1 次元 KdV 方程式、後半では NGSolve を使う有限要素法の例を扱います。
    """)
    return


@app.cell
def _():
    import base64

    artifacts_dir = Path(__file__).resolve().parent / "_generated" / "FEM"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    def gif_image(path: Path, alt: str):
        encoded = base64.b64encode(path.read_bytes()).decode()
        return mo.image(f"data:image/gif;base64,{encoded}", alt=alt)

    return artifacts_dir, gif_image


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Python で有限要素法
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## FEM from scratch
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    [KdV 方程式を解く](http://www.cas.cmc.osaka-u.ac.jp/~paoon/Lectures/2018-7Semester-AppliedMath9/14_pde-fem/)。
    marimo 版では静的 HTML に埋め込める GIF を生成するため、空間分割数と反復回数を小さめにしてある。
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### 係数行列の定義
    """)
    return


@app.cell
def _():
    dx, L = 0.02, 2
    N = int(L / dx)

    dt = 0.0002
    return N, dt, dx


@app.cell
def _(N, dx):
    import numpy as np
    from scipy.sparse import dia_array
    _ex = np.ones(N)
    _diagonals = np.array([1 / 6 * _ex, 1 / 6 * _ex, 2 / 3 * _ex, 1 / 6 * _ex, 1 / 6 * _ex])
    _offsets = np.array([-(N - 1), -1, 0, 1, N - 1])
    phi = dia_array((_diagonals, _offsets), shape=(N, N)) * dx
    return dia_array, np, phi


@app.cell
def _(phi):
    type(phi)
    return


@app.cell
def _(np, phi):
    # 行列の端をどれくらい表示するか
    np.set_printoptions(edgeitems=3)
    # 桁数をどうするか
    np.set_printoptions(precision=3)

    # np.ndarray で表示
    phi.toarray()
    return


@app.cell
def _(N, dia_array, np):
    _ex = np.ones(N)
    _diagonals = np.array([-1 / 2 * _ex, 1 / 2 * _ex, -1 / 2 * _ex, 1 / 2 * _ex])
    _offsets = np.array([-(N - 1), -1, 1, N - 1])
    d1 = dia_array((_diagonals, _offsets), shape=(N, N))
    d1.toarray()
    return (d1,)


@app.cell
def _(N, dia_array, dx, np):
    _ex = np.ones(N)
    _diagonals = np.array([-1 * _ex, -1 * _ex, 2 * _ex, -1 * _ex, -1 * _ex])
    _offsets = np.array([-(N - 1), -1, 0, 1, N - 1])
    d2 = dia_array((_diagonals, _offsets), shape=(N, N)) / dx
    d2.toarray()
    return (d2,)


@app.cell
def _(np):
    def p_func(u: np.ndarray) -> np.ndarray:
        u_plus = np.roll(u, -1)
        u_minus = np.roll(u, 1)
        result = (u_plus + u + u_minus) * (u_plus - u_minus) / 6
        return result

    p_func(np.array([1, 2, 3]))
    return (p_func,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### ソルバーを定義
    """)
    return


@app.cell
def _(d1, d2, np, p_func, phi):
    from scipy.sparse.linalg import bicg

    epsilon = 0.022

    def dudt(u: np.ndarray) -> np.ndarray:
        x, _ = bicg(phi, u @ d2, atol=1e-5)
        x = p_func(u) + epsilon ** 2 * x @ d1
        x, _ = bicg(-phi, x, atol=1e-5)
        return x

    return (dudt,)


@app.cell
def _(N, dx, np):
    import matplotlib.pyplot as plt

    u0 = np.array([np.cos(k * np.pi * dx) for k in range(N)])

    plt.plot(range(N), u0)
    return plt, u0


@app.cell
def _(dudt, np):
    def rk4(u: np.ndarray, dt: float) -> np.ndarray:
        k1 = dt * dudt(u)
        k2 = dt * dudt(u + k1 / 2)
        k3 = dt * dudt(u + k2 / 2)
        k4 = dt * dudt(u + k3)
        return u + (k1 + 2 * k2 + 2 * k3 + k4) / 6

    return (rk4,)


@app.cell
def _(dt, rk4, u0):
    from tqdm.auto import tqdm

    u, u_list = u0, [u0]
    save_interval, max_iter = 200, 2000
    for idx in tqdm(range(max_iter)):
        u = rk4(u, dt)
        if idx % save_interval == 0:
            u_list.append(u)
    return max_iter, save_interval, u_list


@app.cell
def _(N, artifacts_dir, max_iter, plt, save_interval, u_list):
    import matplotlib.animation as animation

    fig, ax = plt.subplots()

    ax.set_xlim(0, N)
    ax.set_ylim(-3, 3)

    sol_plt, = ax.plot([], [], label="solution")

    def anim_callback(i):
        ax.set_title(f"Frame {i}")
        sol_plt.set_data(range(N), u_list[i])
        return (sol_plt,)

    kdv_gif = artifacts_dir / "kdv.gif"
    ani = animation.FuncAnimation(fig, anim_callback, frames=int(max_iter / save_interval) + 1)
    ani.save(kdv_gif, writer="pillow")
    plt.close(fig)
    return (kdv_gif,)


@app.cell
def _(gif_image, kdv_gif):
    gif_image(kdv_gif, alt="有限要素法による KdV 方程式の時間発展")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## NGSolve
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    NGSolve セクションは Colab 専用のインストールセルを使わず、
    通常の Python import で実行する。実行には `ngsolve` と `netgen`
    が環境に入っている必要がある。
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ここからは `ngsolve` / `netgen` の既存インストールを前提にする。
    """)
    return


@app.cell
def _():
    import ngsolve

    return (ngsolve,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### 正方領域のポアソン方程式
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    [First NGSolve example](https://docu.ngsolve.org/latest/i-tutorials/unit-1.1-poisson/poisson.html) に沿って正方領域のポアソン方程式を解く。
    $$\begin{aligned}
        &-\Delta u=f\quad\text{@バルク中},\\
        &u=0\quad\text{@下辺および右辺},\\
        &\frac{∂u}{∂n}=0\quad\text{@上辺および左辺}
    \end{aligned}$$
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    メッシュ生成 (maxh はメッシュの最大サイズ)
    """)
    return


@app.cell
def _(ngsolve):
    mesh = ngsolve.Mesh(ngsolve.unit_square.GenerateMesh(maxh=0.2))
    print(type(mesh))
    # number of vertices & elements
    mesh.nv, mesh.ne
    return (mesh,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    メッシュ表示
    """)
    return


@app.cell
def _(mesh):
    from ngsolve.webgui import Draw

    Draw(mesh)
    return (Draw,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    要素空間を定義 (節点・要素の関係、要素の次数）。

    境界条件もここで定義。今回は下辺・右辺にディリクレ境界条件（値固定）。
    """)
    return


@app.cell
def _(mesh, ngsolve):
    fes = ngsolve.H1(mesh, order=2, dirichlet="bottom|right")
    # 自由度
    fes.ndof
    return (fes,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    要素空間の詳細は `help(fes)` で確認可能（内容略）。
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    各種関数を定義：
    - $u$: 最適化に使用する試行関数 (Trial Function)
    - $v$: 弱解を構成するための重み関数 (Test Function)
    - gfu: 有限要素空間上の関数(解)と係数ベクトルを保有
    """)
    return


@app.cell
def _(fes, ngsolve):
    u_1 = fes.TrialFunction()  # symbolic object
    v = fes.TestFunction()  # symbolic object
    gfu = ngsolve.GridFunction(fes)  # solution
    return gfu, u_1, v


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ポアソン方程式の弱解は形状関数$v(\in H^1_0(S))$を用いて
    $$
    \begin{aligned}
        &-Δu=f \Rightarrow∫(-Δ u)vdS=∫fvdS \\
        &\Leftrightarrow
        \oint_{∂S}\left(-\sum_i∂_iu\cdot v\right)dl^i+∫\sum_i∂_iu∂_ivdS=∫fvdS.
    \end{aligned}
    $$
    ただし１つ目の表面項を0にするような形状関数$v$を選ぶ。
    """)
    return


@app.cell
def _(fes, u_1, v):
    from ngsolve import BilinearForm, LinearForm, grad, x
    from ngsolve import dx as ngs_dx

    a = BilinearForm(fes, symmetric=True)
    a += grad(u_1) * grad(v) * ngs_dx
    a.Assemble()

    f = LinearForm(fes)
    f += x * v * ngs_dx
    f.Assemble()
    return a, f, grad, x


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    [Numpy Interface](https://docu.ngsolve.org/latest/how_to/howto_numpy.html?highlight=ngsolve%20la%20basevector) を使用してプロット。

    まずは右辺のベクトル。
    """)
    return


@app.cell
def _(f, plt):
    print(type(f.vec))
    plt.plot(f.vec.FV().NumPy())
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    次に左辺の疎行列
    """)
    return


@app.cell
def _(a, plt):
    import scipy.sparse as sp

    rows,cols,vals = a.mat.COO()
    A = sp.csr_matrix((vals,(rows,cols)))

    plt.spy(A, markersize=1)
    plt.show()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    弱形式の方程式を解いて可視化
    """)
    return


@app.cell
def _(Draw, a, f, fes, gfu):
    gfu.vec.data = a.mat.Inverse(freedofs=fes.FreeDofs()) * f.vec

    Draw(gfu)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    解をベクトルとして可視化
    """)
    return


@app.cell
def _(gfu, plt):
    print(type(gfu.vec))
    plt.plot(gfu.vec.FV().NumPy())
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### 熱方程式
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    [Parabolic model problem](https://docu.ngsolve.org/latest/i-tutorials/unit-3.1-parabolic/parabolic.html)に従って時間依存性のある偏微分方程式である熱方程式を解く
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    熱方程式の例でも、Colab 用の初期化は行わず通常の import だけで実行する。
    """)
    return


@app.cell
def _(ngsolve):
    from netgen.occ import OCCGeometry, Rectangle, X, Y
    shape = Rectangle(2, 2).Face().Move((-1, -1, 0))
    shape.edges.Min(X).name = 'left'
    shape.edges.Max(X).name = 'right'
    shape.edges.Min(Y).name = 'bottom'
    shape.edges.Max(Y).name = 'top'
    mesh_1 = ngsolve.Mesh(OCCGeometry(shape, dim=2).GenerateMesh(maxh=0.25))
    return (mesh_1,)


@app.cell
def _(mesh_1, ngsolve):
    fes_1 = ngsolve.H1(mesh_1, order=3, dirichlet='bottom|right|left|top')
    u_2, v_1 = fes_1.TnT()
    dt_1 = 0.001
    return dt_1, fes_1, u_2, v_1


@app.cell
def _(Draw, mesh_1, ngsolve, x):
    from ngsolve import y
    b = ngsolve.CoefficientFunction((2 * y * (1 - x * x), -2 * x * (1 - y * y)))
    Draw(b, mesh_1, 'wind', vectors={'grid_size': 32}, order=3)
    return b, y


@app.cell
def _(b, fes_1, grad, ngsolve, u_2, v_1):
    a_1 = ngsolve.BilinearForm(fes_1, symmetric=False)
    a_1 += 0.01 * grad(u_2) * grad(v_1) * ngsolve.dx + b * grad(u_2) * v_1 * ngsolve.dx
    a_1.Assemble()

    m = ngsolve.BilinearForm(fes_1, symmetric=False)
    m += u_2 * v_1 * ngsolve.dx
    m.Assemble()
    return a_1, m


@app.cell
def _(a_1, m):
    mstar = m.mat.CreateMatrix()
    print(f'm.mat.nze = {m.mat.nze}, a.mat.nze={a_1.mat.nze}, mstar.nze={mstar.nze}')
    print(type(mstar))
    return (mstar,)


@app.cell
def _(mstar):
    print(f"mstar.nze={mstar.nze}, len(mstar.AsVector())={len(mstar.AsVector())}")
    return


@app.cell
def _(a_1, dt_1, fes_1, m, mstar):
    mstar.AsVector().data = m.mat.AsVector() + dt_1 * a_1.mat.AsVector()
    # corresponds to M* = M + dt * A
    invmstar = mstar.Inverse(freedofs=fes_1.FreeDofs())
    return (invmstar,)


@app.cell
def _(Draw, fes_1, mesh_1, ngsolve, v_1, x, y):
    f_1 = ngsolve.LinearForm(fes_1)
    gaussp = ngsolve.exp(-6 * ((x + 0.5) * (x + 0.5) + y * y)) - ngsolve.exp(-6 * ((x - 0.5) * (x - 0.5) + y * y))
    Draw(gaussp, mesh_1, 'f', deformation=True)
    f_1 += gaussp * v_1 * ngsolve.dx
    f_1.Assemble()
    return (f_1,)


@app.cell
def _(Draw, fes_1, mesh_1, ngsolve, x, y):
    gfu_1 = ngsolve.GridFunction(fes_1)
    gfu_1.Set((1 - y * y) * x)  # note that boundary conditions remain
    Draw(gfu_1, mesh_1, 'u')
    return (gfu_1,)


@app.cell
def _(a_1, dt_1, f_1, gfu_1, ngsolve):
    def TimeStepping(invmstar, initial_cond=None, t0=0, tend=2, nsamples=10):
        if initial_cond is not None:
            gfu_1.Set(initial_cond)
        cnt = 0
        time = t0
        sample_int = int(ngsolve.floor(tend / dt_1 / nsamples) + 1)
        gfut = ngsolve.GridFunction(gfu_1.space, multidim=0)
        gfut.AddMultiDimComponent(gfu_1.vec)
        while time < tend - 0.5 * dt_1:
            res = dt_1 * f_1.vec - dt_1 * a_1.mat * gfu_1.vec
            gfu_1.vec.data = gfu_1.vec.data + invmstar * res
            if cnt % sample_int == 0:
                gfut.AddMultiDimComponent(gfu_1.vec)
            cnt = cnt + 1
            time = cnt * dt_1
        return gfut

    return (TimeStepping,)


@app.cell
def _(TimeStepping, invmstar, x, y):
    gfut = TimeStepping(invmstar, (1 - y * y) * x)
    return (gfut,)


@app.cell
def _(artifacts_dir, gfut, mesh_1, ngsolve):
    import matplotlib.animation as _animation
    import matplotlib.pyplot as _plt
    import numpy as _np

    sample_x = _np.linspace(-1.0, 1.0, 64)
    sample_y = _np.linspace(-1.0, 1.0, 64)
    frames = []
    for component_index in range(len(gfut.vecs)):
        snapshot = ngsolve.GridFunction(gfut.space)
        snapshot.vec.data = gfut.vecs[component_index]
        frame = _np.array(
            [
                [snapshot(mesh_1(x_coord, y_coord)) for x_coord in sample_x]
                for y_coord in sample_y
            ],
            dtype=float,
        )
        frames.append(frame)

    _fig, _ax = _plt.subplots()
    image = _ax.imshow(
        frames[0],
        extent=[sample_x.min(), sample_x.max(), sample_y.min(), sample_y.max()],
        origin="lower",
        cmap="viridis",
    )
    _fig.colorbar(image, ax=_ax)

    def animate(frame_index):
        _ax.set_title(f"sample {frame_index}")
        image.set_data(frames[frame_index])
        return (image,)

    ngsolve_heat_gif = artifacts_dir / "ngsolve_heat.gif"
    _animation.FuncAnimation(_fig, animate, frames=len(frames), interval=150).save(
        ngsolve_heat_gif,
        writer="pillow",
    )
    _plt.close(_fig)
    return (ngsolve_heat_gif,)


@app.cell
def _(gif_image, ngsolve_heat_gif):
    gif_image(ngsolve_heat_gif, alt="NGSolve による熱方程式の時間発展")
    return


if __name__ == "__main__":
    app.run()
