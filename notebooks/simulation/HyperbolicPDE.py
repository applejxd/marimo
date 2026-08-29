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

    波動方程式を 1 次元・2 次元それぞれについて、有限差分で空間を離散化し
    `scipy.integrate.solve_ivp` で時間積分する
    （`legacy/simulation/HyperbolicPDE.ipynb` を marimo 向けに整理したものである）。

    双曲型方程式の特徴は**情報が有限速度 $c$ で伝わり、減衰しない**ことである。
    この notebook では次の 2 つを、波が領域を往復するのに十分な時間まで積分して
    アニメーションで確認する。

    | 節 | 対象 | 空間刻み | 時間範囲 | 境界条件 |
    | --- | --- | --- | --- | --- |
    | 1 次元 | $u_{tt}=c^2u_{xx}$ | $h=0.005$（200 点） | $t\in[0,2]$ | ディリクレ（固定端） |
    | 2 次元 | $u_{tt}=c^2\nabla^2u$ | $h=0.02$（50×50 点） | $t\in[0,1.5]$ | ノイマン（自由端） |

    1 次元では領域長 $L=1$、伝播速度 $c=1$ なので往復周期は $2L/c=2$ である。
    $t=2$ まで積分することで「分裂 → 壁での反転反射 → 初期波形の再現」という
    1 周期がまるごと 1 本のアニメーションに収まる。
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## この notebook の共通部品

    次のセルは、以降のすべてのアニメーションが共有する出力先とヘルパーを用意する。

    - `artifacts_dir`: GIF の保存先 `notebooks/simulation/_generated/HyperbolicPDE/`。
      Git の追跡対象外なので、実行するたびに再生成される。
    - `gif_image(path, alt)`: GIF を base64 の data URI に変換して `mo.image` で
      埋め込む。静的 HTML へ書き出したときも外部ファイルを参照せずに再生できる。
    - `save_gif(fig, update, frames, path, interval)`: `FuncAnimation` を GIF として
      保存し、生成されたファイルサイズ（KiB）を返す。
      `interval` は 1 コマの表示時間（ミリ秒）である。GIF は表示時間を 1/100 秒単位で
      しか保持できないため、要求した値がそのまま使われるとは限らない。
      保存後に GIF を読み直して実際の間隔と再生時間を測り、報告している。
    """)
    return


@app.cell
def _():
    import base64

    import matplotlib.animation as animation
    import matplotlib.pyplot as plt
    from PIL import Image

    artifacts_dir = Path(__file__).resolve().parent / "_generated" / "HyperbolicPDE"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    def gif_image(path: Path, alt: str):
        encoded = base64.b64encode(path.read_bytes()).decode()
        return mo.image(f"data:image/gif;base64,{encoded}", alt=alt)

    def save_gif(fig, update, frames: int, path: Path, interval: int) -> str:
        anim = animation.FuncAnimation(fig, update, frames=frames, interval=interval)
        anim.save(path, writer="pillow")
        plt.close(fig)
        # GIF は間隔を 1/100 秒単位でしか保持できないので、保存後に実測して報告する。
        with Image.open(path) as gif:
            durations = []
            for index in range(gif.n_frames):
                gif.seek(index)
                durations.append(gif.info.get("duration", 0))
        return (
            f"{path.name}: {frames} frames, {durations[0]} ms/frame, "
            f"{sum(durations) / 1000:.1f} s, {path.stat().st_size / 1024:.0f} KiB"
        )

    return artifacts_dir, gif_image, plt, save_gif


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 1 次元の波動方程式
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    波動方程式
    $$
    \frac{∂^2u}{∂t^2}=c^2\frac{∂^2u}{∂x^2}
    $$
    を次の条件で解く。

    - 伝播速度 $c=1$
    - 空間領域 $x\in[0,1]$、空間刻み $h=0.005$（格子点 200 個）
    - 時間範囲 $t\in[0,2]$、アニメーション 1 コマあたり $Δt_{\text{frame}}=0.0125$（161 コマ）
    - 境界条件 $u(0,t)=u(1,t)=0$（両端を固定した弦）

    空間刻みを $h=0.02$ から $h=0.005$ へ細かくしたのは、初期波形の 1 山が
    格子 50 点で表現され、反射のたびに現れる数値分散が目に見えないようにするためである。

    時間刻みについては、情報が 1 ステップで 1 格子分より遠くへ伝わらないという
    CFL 条件
    $$
    k\leq\frac{h}{c}
    $$
    が安定性の目安になる。ここでは固定刻みではなく適応刻みの `RK45` を使うため、
    刻み幅はソルバーが自動的に CFL 条件を満たす範囲へ縮める。
    `max_step` にはコマ間隔を渡し、コマを跨いで補間されないようにしている。
    """)
    return


@app.cell
def _():
    import numpy as np

    wave_speed = 1.0
    dx_1d = 0.005
    x_1d = np.arange(0, 1, dx_1d)
    n_1d = len(x_1d)

    t_max_1d = 2.0
    frame_dt_1d = 0.0125
    t_1d = np.arange(0, t_max_1d + 0.5 * frame_dt_1d, frame_dt_1d)

    f"格子点 {n_1d} 個, コマ数 {len(t_1d)}, CFL 上限 k <= {dx_1d / wave_speed}"
    return dx_1d, frame_dt_1d, n_1d, np, t_1d, t_max_1d, wave_speed, x_1d


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### 初期条件
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    初期条件は、領域の中央 $3/8\leq x\leq5/8$ にだけ台を持つ 1 山の隆起
    $$
    u(x,0)=
    \begin{cases}
    \dfrac{1}{2}\cos(8\pi(x-1/2))+\dfrac{1}{2}\quad(3/8\leq x\leq5/8), \\
    0\quad(\text{otherwise}),
    \end{cases}
    $$
    および初速度ゼロ
    $$
    \frac{∂u}{∂t}(x,0)=0
    $$
    とする。初速度がゼロなので、この山は左右へ振幅半分ずつに分裂して進む
    （ダランベールの解）。
    """)
    return


@app.cell
def _(dx_1d, np, plt, x_1d):
    u0_1d = 0.5 * np.cos(8 * np.pi * (x_1d - 0.5)) + 0.5
    u0_1d[: int(3 / 8 / dx_1d)] = 0
    u0_1d[int(5 / 8 / dx_1d) :] = 0

    _fig, _ax = plt.subplots(figsize=(6.4, 2.6), dpi=100)
    _ax.plot(x_1d, u0_1d)
    _ax.set_xlabel("$x$")
    _ax.set_ylabel("$u(x,0)$")
    _ax.set_title("initial displacement")
    _fig.tight_layout()
    _fig
    return (u0_1d,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### 係数行列
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    2 階の時間微分をそのまま扱う代わりに、速度 $v$ を独立変数として導入し
    1 階の連立系へ書き換える。
    $$
    \frac{∂u}{∂t}(x,t)≡v(x,t),\quad
    \frac{∂v}{∂t}=c^2\frac{∂^2u}{∂x^2}
    $$
    行列で書けば
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
    であり、この $∂_x^2$ を中心差分
    $$
    \frac{∂^2u}{∂x^2}(x_i)\simeq\frac{u_{i+1}-2u_i+u_{i-1}}{h^2}
    $$
    で置き換えたものが係数行列である。

    次のセルの `second_derivative` は、隣接点を指す 2 本のシフト行列
    $S_{-}u|_i=u_{i-1}$、$S_{+}u|_i=u_{i+1}$ から
    $(S_{+}-2I+S_{-})/h^2$ を組み立てる。両端では領域外の $u_{-1}$、$u_{N}$ を
    参照する成分が存在しないため、それらは自動的に $0$ 扱いになる。
    これがそのまま固定端のディリクレ境界条件 $u=0$ に対応する。

    実際に使う行列は $200\times200$ で数字を読んでも分からないため、
    まず $n=6$、$h=1$ の小さな例で構造を確認する。
    """)
    return


@app.cell
def _(np):
    import scipy.sparse as sparse

    def second_derivative(n: int, h: float) -> sparse.csr_matrix:
        """固定端（両端で u=0）の 2 階中心差分行列を返す。"""
        shift_back = sparse.dia_array((np.ones(n), [-1]), shape=(n, n))
        shift_forward = sparse.dia_array((np.ones(n), [1]), shape=(n, n))
        identity = sparse.identity(n)
        return ((shift_forward - 2 * identity + shift_back) / h**2).tocsr()

    np.set_printoptions(edgeitems=3, precision=3, suppress=True)
    second_derivative(6, 1.0).toarray()
    return second_derivative, sparse


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    この $6\times6$ の例で、1 行目に $u_{-1}$ の寄与が無く 2 行目以降と形が違うことが
    固定端の効果である。実際の計算では $n=200$、$h=0.005$ を使うので、対角成分は
    $-2/h^2=-8\times10^4$ という大きな値になる。

    次のセルでは、この 2 階微分行列を左下ブロックに埋め込んだ $2n\times2n$ の
    係数行列 $A$ を作り、非ゼロ成分の配置を `spy` で確認する。
    右上の単位行列ブロックが $∂_tu=v$ を、左下の三重対角ブロックが
    $∂_tv=c^2∂_x^2u$ を表す。
    """)
    return


@app.cell
def _(dx_1d, n_1d, plt, second_derivative, sparse, wave_speed):
    laplacian_1d = second_derivative(n_1d, dx_1d)
    coeff_1d = sparse.bmat(
        [
            [None, sparse.identity(n_1d)],
            [wave_speed**2 * laplacian_1d, None],
        ],
        format="csr",
    )

    _fig, _ax = plt.subplots(figsize=(4.2, 4.2), dpi=100)
    _ax.spy(coeff_1d, markersize=0.4)
    _ax.set_title(f"coefficient matrix {coeff_1d.shape}, nnz={coeff_1d.nnz}")
    _fig.tight_layout()
    _fig
    return (coeff_1d,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### 時間積分
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    状態ベクトルを $[u, v]$ と縦に並べ、`solve_ivp` の `RK45`（陽的ルンゲ＝クッタ法）で
    $t=2$ まで積分する。

    - `t_eval` にコマの時刻を渡し、必要な時刻の値だけを受け取る。
      密な補間器（`dense_output`）を作らないぶんメモリを使わない。
    - `max_step=frame_dt_1d` でコマ間隔より大きな刻みを禁止する。
    - `sol.success` を必ず確認する。刻み幅が下限に達して積分が途中で打ち切られても
      `solve_ivp` は例外を投げないため、確認を省くと発散した結果をそのまま
      可視化してしまう。
    """)
    return


@app.cell
def _(coeff_1d, frame_dt_1d, n_1d, np, t_1d, t_max_1d, u0_1d):
    from scipy.integrate import solve_ivp

    def _rhs(_t, state):
        return coeff_1d @ state

    _sol_1d = solve_ivp(
        _rhs,
        t_span=(0, t_max_1d),
        y0=np.hstack([u0_1d, np.zeros(n_1d)]),
        method="RK45",
        t_eval=t_1d,
        max_step=frame_dt_1d,
        rtol=1e-8,
        atol=1e-10,
    )
    if not _sol_1d.success:
        raise RuntimeError(_sol_1d.message)
    u_1d = _sol_1d.y.T[:, :n_1d]

    f"success={_sol_1d.success}, 右辺評価 {_sol_1d.nfev} 回, u の範囲 [{u_1d.min():.3f}, {u_1d.max():.3f}]"
    return solve_ivp, u_1d


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    $u$ の最小値が $-1$ 近くまで届いている点に注意する。固定端での反射は
    波形を上下反転させるため、縦軸の範囲は $[0,1]$ ではなく $[-1.15,1.15]$ を取る。

    ### 時空間ダイアグラム

    アニメーションの前に、横軸 $x$・縦軸 $t$ で $u(x,t)$ を一枚の画像にする。
    傾き $\pm1/c$ の直線（特性曲線）に沿って山が進み、壁に当たるたびに折り返しながら
    符号が反転する様子が読み取れる。$t=2$ で初期波形へ戻る周期性も確認できる。
    """)
    return


@app.cell
def _(plt, t_1d, u_1d, x_1d):
    _fig, _ax = plt.subplots(figsize=(5.6, 4.0), dpi=100)
    _im = _ax.imshow(
        u_1d,
        origin="lower",
        aspect="auto",
        cmap="RdBu_r",
        vmin=-1,
        vmax=1,
        extent=[x_1d[0], x_1d[-1], t_1d[0], t_1d[-1]],
    )
    _ax.set_xlabel("$x$")
    _ax.set_ylabel("$t$")
    _ax.set_title("space-time diagram of $u(x,t)$")
    _fig.colorbar(_im, ax=_ax)
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### アニメーション

    161 コマを 1 コマ 50 ms で再生するので、再生時間は約 8.1 秒である。
    """)
    return


@app.cell
def _(artifacts_dir, plt, save_gif, t_1d, u_1d, x_1d):
    _fig, _ax = plt.subplots(figsize=(6.4, 3.6), dpi=100)
    _ax.set_xlim(0, 1)
    _ax.set_ylim(-1.15, 1.15)
    _ax.set_xlabel("$x$")
    _ax.set_ylabel("$u$")
    _ax.axhline(0, lw=0.5, color="0.7")
    _line, = _ax.plot([], [], lw=1.6)
    _title = _ax.set_title("1D wave, t = 0.000")
    _fig.tight_layout()

    def _update(frame):
        _line.set_data(x_1d, u_1d[frame])
        _title.set_text(f"1D wave, t = {t_1d[frame]:.3f}")
        return (_line, _title)

    wave_1d_gif = artifacts_dir / "wave_1d.gif"
    save_gif(_fig, _update, len(t_1d), wave_1d_gif, interval=50)
    return (wave_1d_gif,)


@app.cell
def _(gif_image, wave_1d_gif):
    gif_image(wave_1d_gif, alt="1次元波動方程式の時間発展")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    アニメーションで確認できることは次の 3 点である。

    1. $t\lesssim0.12$: 初期の 1 山が振幅 $1/2$ の 2 つの山へ分裂し、左右へ速度 $c=1$ で進む。
    2. $t\simeq0.4$〜$0.6$: 固定端に到達した山が**上下反転して**跳ね返る。
       固定端では常に $u=0$ でなければならないため、入射波を打ち消す反転波が必要になる。
    3. $t=2$: 左右の波が中央で再び重なり、初期波形が復元する（周期 $2L/c=2$）。

    振幅がほとんど減衰していないことが、双曲型と放物型（拡散方程式）の決定的な違いである。
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 2 次元の波動方程式
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    2 次元では
    $$
    \frac{∂^2u}{∂t^2}=c^2\left(\frac{∂^2u}{∂x^2}+\frac{∂^2u}{∂y^2}\right)
    $$
    を正方領域 $[0,1]^2$ で解く。

    - 空間刻み $h=0.02$（$50\times50=2500$ 格子点、未知数は $u$ と $v$ で 5000）
    - 時間範囲 $t\in[0,1.5]$、1 コマ $Δt_{\text{frame}}=0.015$（101 コマ）
    - 境界条件はノイマン条件（法線方向の勾配ゼロ、自由端）

    もとの設定（$h=0.05$ の $20\times20$ 格子、$t\leq0.8$）では格子が粗く、
    円形の波面が階段状に見えていた。格子を $50\times50$ へ、
    時間を壁での反射が 2 回起きる $t=1.5$ まで延ばしている。
    """)
    return


@app.cell
def _(np):
    dx_2d = 0.02
    x_2d = np.arange(0, 1, dx_2d)
    y_2d = np.arange(0, 1, dx_2d)
    nx_2d, ny_2d = len(x_2d), len(y_2d)
    n_2d = nx_2d * ny_2d

    t_max_2d = 1.5
    frame_dt_2d = 0.015
    t_2d = np.arange(0, t_max_2d + 0.5 * frame_dt_2d, frame_dt_2d)

    f"格子 {nx_2d}x{ny_2d}={n_2d} 点, 未知数 {2 * n_2d}, コマ数 {len(t_2d)}"
    return dx_2d, frame_dt_2d, n_2d, nx_2d, ny_2d, t_2d, t_max_2d, x_2d


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### 初期条件

    1 次元と同じ切り出した余弦の山を 2 方向のテンソル積にして 2 次元の隆起を作り、
    $(x,y)\simeq(0.25,0.25)$ と $(0.75,0.75)$ の 2 か所に置く。
    波源を 2 つにすることで、それぞれから広がる円形波が干渉する様子が観察できる。
    """)
    return


@app.cell
def _(dx_2d, np, plt, x_2d):
    _cosine = 0.5 * np.cos(8 * np.pi * (x_2d - 0.5)) + 0.5

    _first = _cosine.copy()
    _first[: int(1 / 8 / dx_2d)] = 0
    _first[int(3 / 8 / dx_2d) :] = 0

    _second = _cosine.copy()
    _second[: int(5 / 8 / dx_2d)] = 0
    _second[int(7 / 8 / dx_2d) :] = 0

    u0_2d = np.tensordot(_first, _first, axes=0) + np.tensordot(_second, _second, axes=0)

    _fig, _ax = plt.subplots(figsize=(4.4, 3.8), dpi=100)
    _im = _ax.imshow(u0_2d, origin="lower", cmap="RdBu_r", vmin=-1, vmax=1, extent=[0, 1, 0, 1])
    _ax.set_xlabel("$x$")
    _ax.set_ylabel("$y$")
    _ax.set_title("initial displacement")
    _fig.colorbar(_im, ax=_ax)
    _fig.tight_layout()
    _fig
    return (u0_2d,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### 係数行列

    2 次元の格子点 $(i,j)$ を `index = i * ny + j` の 1 本のベクトルへ並べ替えると、
    隣接点の参照は 4 本のシフト行列で書ける。

    - `shift_ym` / `shift_yp`: $j\mp1$ を指す。対角から $\mp1$ ずれた成分。
    - `shift_xm` / `shift_xp`: $i\mp1$ を指す。対角から $\mp n_y$ ずれた成分。

    そのままだと $j=0$ の行が前の $x$ 列の $j=n_y-1$ を参照してしまうため、
    境界に当たる行を一度ゼロにしてから**対角成分に 1 を入れ直す**。
    これは領域外の点の値を自分自身の値で置き換えることに相当し、
    法線方向の差分がゼロ、すなわちノイマン境界条件（自由端）になる。
    1 次元で使った固定端と違い、反射のときに波形が反転しない。

    ラプラシアンは
    $$
    \nabla^2u|_{i,j}\simeq\frac{u_{i+1,j}+u_{i-1,j}+u_{i,j+1}+u_{i,j-1}-4u_{i,j}}{h^2}
    $$
    である。
    """)
    return


@app.cell
def _(sparse):
    def neighbor_matrix(n: int, offset: int, blocked: slice | range) -> sparse.csr_matrix:
        """offset だけずれた隣接点を指す行列。境界行は自分自身を指すようにする。"""
        matrix = sparse.lil_matrix(sparse.eye(n, k=offset))
        for index in blocked:
            matrix[index, :] = 0
            matrix[index, index] = 1
        return sparse.csr_matrix(matrix)

    return (neighbor_matrix,)


@app.cell
def _(dx_2d, n_2d, neighbor_matrix, nx_2d, ny_2d, sparse):
    _shift_ym = neighbor_matrix(n_2d, -1, range(0, n_2d, ny_2d))
    _shift_yp = neighbor_matrix(n_2d, 1, range(ny_2d - 1, n_2d, ny_2d))
    _shift_xm = neighbor_matrix(n_2d, -ny_2d, range(ny_2d))
    _shift_xp = neighbor_matrix(n_2d, ny_2d, range((nx_2d - 1) * ny_2d, n_2d))

    laplacian_2d = (
        _shift_ym + _shift_yp + _shift_xm + _shift_xp - 4 * sparse.identity(n_2d)
    ) / dx_2d**2
    coeff_2d = sparse.bmat(
        [[None, sparse.identity(n_2d)], [laplacian_2d, None]],
        format="csr",
    )

    f"laplacian {laplacian_2d.shape} nnz={laplacian_2d.nnz}, coeff {coeff_2d.shape} nnz={coeff_2d.nnz}"
    return coeff_2d, laplacian_2d


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    $5000\times5000$ の行列を `toarray()` で表示すると 200 MB の密行列になってしまうので、
    構造は `spy` で確認する。

    左は係数行列 $A$ 全体で、右上の対角線が $∂_tu=v$ を表す単位行列ブロック、
    左下の帯がラプラシアンのブロックである。右はそのラプラシアンの左上
    $200\times200$ を拡大したもので、非ゼロは 5 本の対角線に並ぶ。
    中央の太い帯が自分自身と $y$ 方向の隣（対角から $\pm1$）、その外側の 2 本が
    $x$ 方向の隣（対角から $\pm n_y=\pm50$）である。
    """)
    return


@app.cell
def _(coeff_2d, laplacian_2d, plt):
    _fig, _axes = plt.subplots(1, 2, figsize=(8.4, 4.2), dpi=100)
    _axes[0].spy(coeff_2d, markersize=0.15)
    _axes[0].set_title(f"coefficient matrix {coeff_2d.shape}")
    _axes[1].spy(laplacian_2d[:200, :200], markersize=1.2)
    _axes[1].set_title("laplacian block (top-left 200x200)")
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### 時間積分

    1 次元と同じく `RK45` で積分する。未知数が 5000 個あるため、右辺評価は
    数千回に達すが、係数行列は 1 行あたり非ゼロが高々 5 個の疎行列なので
    数秒で終わる。
    """)
    return


@app.cell
def _(coeff_2d, frame_dt_2d, n_2d, np, solve_ivp, t_2d, t_max_2d, u0_2d):
    def _rhs(_t, state):
        return coeff_2d @ state

    _sol_2d = solve_ivp(
        _rhs,
        t_span=(0, t_max_2d),
        y0=np.hstack([u0_2d.reshape(-1), np.zeros(n_2d)]),
        method="RK45",
        t_eval=t_2d,
        max_step=frame_dt_2d,
        rtol=1e-6,
        atol=1e-9,
    )
    if not _sol_2d.success:
        raise RuntimeError(_sol_2d.message)
    u_2d = _sol_2d.y.T[:, :n_2d]

    f"success={_sol_2d.success}, 右辺評価 {_sol_2d.nfev} 回, u の範囲 [{u_2d.min():.3f}, {u_2d.max():.3f}]"
    return (u_2d,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### アニメーション

    色は $0$ を白とする発散カラーマップ `RdBu_r` を使い、色の範囲を全コマで
    $[-0.4,0.4]$ に固定する。コマごとに色の範囲を自動調整すると、振幅が小さく
    なった時刻でも同じ濃さで描かれてしまい、減衰や集中の様子が読めなくなるためである。

    範囲を初期振幅の $1$ ではなく $0.4$ に取っているのは、2 次元では円形に広がる
    波の振幅が距離のおよそ $1/\sqrt{r}$ で下がり、$t\gtrsim0.2$ 以降の振幅が
    $0.3$ 程度に落ち着くためである。最初の数コマでは 2 つの隆起が飽和して
    真っ赤に描かれるが、そのぶん以降の波面と干渉模様がはっきり見える。

    101 コマを 1 コマ 80 ms で再生するので、再生時間は約 8.1 秒である。
    """)
    return


@app.cell
def _(artifacts_dir, nx_2d, ny_2d, plt, save_gif, t_2d, u_2d):
    _fig, _ax = plt.subplots(figsize=(4.3, 3.8), dpi=100)
    _im = _ax.imshow(
        u_2d[0].reshape(nx_2d, ny_2d),
        origin="lower",
        cmap="RdBu_r",
        vmin=-0.4,
        vmax=0.4,
        extent=[0, 1, 0, 1],
    )
    _ax.set_xlabel("$x$")
    _ax.set_ylabel("$y$")
    _title = _ax.set_title("2D wave, t = 0.000")
    _fig.colorbar(_im, ax=_ax)
    _fig.tight_layout()

    def _update(frame):
        _im.set_data(u_2d[frame].reshape(nx_2d, ny_2d))
        _title.set_text(f"2D wave, t = {t_2d[frame]:.3f}")
        return (_im, _title)

    wave_2d_gif = artifacts_dir / "wave_2d.gif"
    save_gif(_fig, _update, len(t_2d), wave_2d_gif, interval=80)
    return (wave_2d_gif,)


@app.cell
def _(gif_image, wave_2d_gif):
    gif_image(wave_2d_gif, alt="2次元波動方程式の時間発展")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    アニメーションで確認できることは次の 4 点である。

    1. $t\lesssim0.15$: 2 つの波源からそれぞれ同心円状の波面が広がる。
       広がった円の中心には、進行波の裏返しである窪み（青）が残る。
    2. $t\simeq0.2$〜$0.3$: 2 つの円が中間の対角線上で出会い、山と山が重なって
       振幅が足し合わされる（線形方程式なので重ね合わせが厳密に成り立つ）。
       同じころ、波源に近い壁への到達も始まる。
    3. 壁での反射: 自由端（ノイマン境界）なので**符号を保ったまま**跳ね返る。
       1 次元の固定端で波形が反転したのと対照的で、壁のすぐ内側では
       入射波と反射波が強め合う。
    4. $t\gtrsim0.5$: 反射波どうしが何度も交差し、正方形の対称性を反映した
       格子状の干渉模様が残る。エネルギーは境界から逃げないため、
       個々の波面の振幅が下がっても場全体からは消えない。
    """)
    return


if __name__ == "__main__":
    app.run()
