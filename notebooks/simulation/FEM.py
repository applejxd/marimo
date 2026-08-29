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

    有限要素法（FEM）を 2 通りのやり方で扱う
    （`legacy/simulation/FEM.ipynb` を marimo 向けに整理したものである）。

    | 節 | 内容 | 使うもの |
    | --- | --- | --- |
    | 1 | 1 次元 KdV 方程式を、質量行列・剛性行列を自分で組んで解く | NumPy / SciPy のみ |
    | 2 | 正方領域のポアソン方程式と、風のある熱方程式を解く | NGSolve |

    前半は「有限要素法とは何をしているのか」を、行列を 1 つずつ組み立てて確かめる
    ためのものである。線形の 1 次要素（P1）を周期的な 1 次元メッシュ上に並べ、
    Zabusky と Kruskal が 1965 年にソリトンを発見したときと同じ初期値問題を
    再現する。後半は実用的な FEM ライブラリ NGSolve で 2 次元問題を解く。

    NGSolve の可視化ユーティリティ `ngsolve.webgui.Draw` は WebGL のウィジェットを
    返すため、静的 HTML へ書き出すと空欄になってしまう。この notebook では
    すべての描画を matplotlib で行い、HTML でも図が残るようにしている。
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## この notebook の共通部品

    次のセルは、以降のすべてのアニメーションが共有する出力先とヘルパーを用意する。

    - `artifacts_dir`: GIF の保存先 `notebooks/simulation/_generated/FEM/`。
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
    import numpy as np
    from PIL import Image

    artifacts_dir = Path(__file__).resolve().parent / "_generated" / "FEM"
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

    np.set_printoptions(edgeitems=3, precision=3, suppress=True)
    return artifacts_dir, gif_image, np, plt, save_gif


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 有限要素法をゼロから組む: KdV 方程式
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    [KdV 方程式を有限要素法で解く](http://www.cas.cmc.osaka-u.ac.jp/~paoon/Lectures/2018-7Semester-AppliedMath9/14_pde-fem/)
    に沿って、Zabusky–Kruskal の初期値問題
    $$
    \frac{∂u}{∂t}+u\frac{∂u}{∂x}+δ^2\frac{∂^3u}{∂x^3}=0,\quad
    u(x,0)=\cos(\pi x),\quad δ=0.022
    $$
    を、周期境界条件の区間 $x\in[0,2)$ で解く。

    区間を長さ $h$ の要素に分割し、各節点に「その節点で 1、隣の節点で 0 になる
    三角形の関数」$φ_i(x)$（1 次要素）を置いて $u(x,t)\simeq\sum_iu_i(t)φ_i(x)$ と
    展開する。方程式に $φ_i$ を掛けて積分すると、必要な積分は次の 3 種類だけである。

    - 質量行列 $M_{ij}=\int φ_iφ_j\,dx$: 隣り合う 3 つの節点にまたがる帯行列で、
      成分は $h\cdot(1/6,\,2/3,\,1/6)$。
    - 勾配行列 $B_{ij}=\int φ_jφ_i'\,dx$: 成分は $(1/2,\,0,\,-1/2)$ で $h$ に依らない。
      反対称なので、離散化しても移流がエネルギーを作り出さない。
    - 剛性行列 $K_{ij}=\int φ_i'φ_j'\,dx$: 成分は $(1/h)\cdot(-1,\,2,\,-1)$。
      部分積分により $\int u_{xx}φ_i\,dx=-(Ku)_i$ に対応する。

    3 階微分は、いったん $Mw=Ku$（つまり $w\simeq-u_{xx}$）を解いてから
    $\int u_{xxx}φ_i\,dx=\int wφ_i'\,dx=(Bw)_i$ と 2 段階で評価する。
    非線形項は 1 次要素で厳密に積分でき、
    $$
    \int u\frac{∂u}{∂x}φ_i\,dx=\frac{(u_{i+1}+u_i+u_{i-1})(u_{i+1}-u_{i-1})}{6}
    $$
    となる。まとめると、解くべき常微分方程式は
    $$
    M\dot{u}=-\left[P(u)+δ^2BM^{-1}Ku\right]
    $$
    である。
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### 離散化パラメータ

    | 記号 | 値 | 意味 |
    | --- | --- | --- |
    | $L$ | 2 | 周期区間の長さ |
    | $h$ | 0.01 | 要素の長さ（節点 200 個） |
    | $δ$ | 0.022 | 分散の強さ |
    | $Δt$ | $10^{-4}$ | RK4 の時間刻み |
    | $t_{\max}$ | 10 | 積分の終了時刻 |

    もとの設定は $h=0.02$、$t_{\max}=0.4$（保存 11 コマ）だった。$t=0.4$ は
    波が切り立ち始めた直後にすぎず、この方程式で本当に見たいソリトンの生成も
    再帰現象も始まる前に終わっていた。

    時間の目安は破断時刻 $t_B=1/\pi\simeq0.318$ である。ソリトン列が出そろうのが
    $t\simeq3.6\,t_B\simeq1.15$、初期波形へ戻る再帰時刻が
    $t_R\simeq30.4\,t_B\simeq9.7$ なので、$t_{\max}=10$ まで積分すれば
    「切り立ち → ソリトン生成 → 再帰」が 1 本のアニメーションに収まる。
    格子も $h=0.01$ へ細かくして、幅が $δ$ 程度しかないソリトンを解像する。
    """)
    return


@app.cell
def _(np):
    kdv_length = 2.0
    kdv_dx = 0.01
    kdv_n = int(kdv_length / kdv_dx)
    kdv_epsilon = 0.022

    kdv_dt = 1e-4
    kdv_t_max = 10.0
    kdv_frames = 161
    kdv_save_every = int(round(kdv_t_max / kdv_dt / (kdv_frames - 1)))

    kdv_x = np.arange(kdv_n) * kdv_dx

    (
        f"節点 {kdv_n} 個, 時間ステップ {int(kdv_t_max / kdv_dt)} 回",
        f"{kdv_save_every} ステップごとに保存して {kdv_frames} コマ",
        f"破断時刻 t_B = 1/pi = {1 / np.pi:.3f}, 再帰時刻 30.4 t_B = {30.4 / np.pi:.2f}",
    )
    return (
        kdv_dt,
        kdv_dx,
        kdv_epsilon,
        kdv_frames,
        kdv_length,
        kdv_n,
        kdv_save_every,
        kdv_t_max,
        kdv_x,
    )


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### 係数行列の定義

    3 つの行列はいずれも巡回帯行列なので、`scipy.sparse.dia_array` に対角成分と
    オフセットを渡して組み立てる。オフセット $\pm(N-1)$ の項が、右端と左端を
    つなぐ周期境界条件に対応する。

    実際に使うのは $200\times200$ の行列なので、構造が読めるよう
    まず $N=6$、$h=1$ の小さな例で 3 つを並べて確認する。
    """)
    return


@app.cell
def _(np):
    from scipy.sparse import dia_array

    def mass_matrix(n: int, h: float) -> dia_array:
        """M_ij = ∫ φ_i φ_j dx。1 次要素なら h * (1/6, 2/3, 1/6) の帯行列。"""
        ones = np.ones(n)
        diagonals = np.array([ones / 6, ones / 6, 2 * ones / 3, ones / 6, ones / 6])
        offsets = np.array([-(n - 1), -1, 0, 1, n - 1])
        return dia_array((diagonals, offsets), shape=(n, n)) * h

    def gradient_matrix(n: int) -> dia_array:
        """B_ij = ∫ φ_j φ_i' dx。反対称で h に依らない。"""
        ones = np.ones(n)
        diagonals = np.array([-ones / 2, ones / 2, -ones / 2, ones / 2])
        offsets = np.array([-(n - 1), -1, 1, n - 1])
        return dia_array((diagonals, offsets), shape=(n, n))

    def stiffness_matrix(n: int, h: float) -> dia_array:
        """K_ij = ∫ φ_i' φ_j' dx。(1/h) * (-1, 2, -1) の帯行列。"""
        ones = np.ones(n)
        diagonals = np.array([-ones, -ones, 2 * ones, -ones, -ones])
        offsets = np.array([-(n - 1), -1, 0, 1, n - 1])
        return dia_array((diagonals, offsets), shape=(n, n)) / h

    (
        mass_matrix(6, 1.0).toarray(),
        gradient_matrix(6).toarray(),
        stiffness_matrix(6, 1.0).toarray(),
    )
    return gradient_matrix, mass_matrix, stiffness_matrix


@app.cell
def _(gradient_matrix, kdv_dx, kdv_n, mass_matrix, stiffness_matrix):
    kdv_mass = mass_matrix(kdv_n, kdv_dx).tocsc()
    kdv_gradient = gradient_matrix(kdv_n).tocsr()
    kdv_stiffness = stiffness_matrix(kdv_n, kdv_dx).tocsr()

    f"M {kdv_mass.shape} nnz={kdv_mass.nnz}, B nnz={kdv_gradient.nnz}, K nnz={kdv_stiffness.nnz}"
    return kdv_gradient, kdv_mass, kdv_stiffness


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### 非線形項

    非線形項 $P(u)_i=(u_{i+1}+u_i+u_{i-1})(u_{i+1}-u_{i-1})/6$ は、
    `np.roll` で隣の節点をずらして取り出せば 1 行で書ける。
    `np.roll` が配列の端を巻き戻すので、これがそのまま周期境界条件になる。

    下の例では $u=(1,2,3)$ に対して中央の成分が
    $(3+2+1)(3-1)/6=2$ となることを確認している。
    """)
    return


@app.cell
def _(np):
    def nonlinear_term(u: np.ndarray) -> np.ndarray:
        """∫ u u_x φ_i dx の 1 次要素での厳密な値。"""
        u_plus = np.roll(u, -1)
        u_minus = np.roll(u, 1)
        return (u_plus + u + u_minus) * (u_plus - u_minus) / 6

    nonlinear_term(np.array([1.0, 2.0, 3.0]))
    return (nonlinear_term,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### 右辺の評価

    右辺 $-M^{-1}\left[P(u)+δ^2BM^{-1}Ku\right]$ の評価には、質量行列を係数とする
    連立方程式を 1 回の評価あたり 2 回解く必要がある。$M$ は時間に依らないので、
    毎回反復法で解き直す代わりに `splu` で一度だけ LU 分解し、
    以後は前進後退代入だけで済ませる。

    もとの実装は許容誤差 $10^{-5}$ の `bicg` を使っていた。今回は時間ステップが
    10 万回・右辺評価が 40 万回に増えるため、そのままでは反復法の打ち切り誤差が
    蓄積する。LU 分解なら誤差は丸め誤差の水準で、かつ計測では同じ問題を解くのに
    要する時間が 1/20 になる。

    なお $B$ は反対称なので、$Bw$ と $w^{\mathsf{T}}B$ は符号が逆になる。
    $\int u_{xxx}φ_i\,dx=\int wφ_i'\,dx=\sum_jw_j\int φ_jφ_i'\,dx=(Bw)_i$ なので、
    使うのは `kdv_gradient @ w` の側である。
    """)
    return


@app.cell
def _(kdv_epsilon, kdv_gradient, kdv_mass, kdv_stiffness, nonlinear_term, np):
    from scipy.sparse.linalg import splu

    _mass_lu = splu(kdv_mass)

    def kdv_rhs(u: np.ndarray) -> np.ndarray:
        curvature = _mass_lu.solve(kdv_stiffness @ u)  # w = M^-1 K u ≃ -u_xx
        dispersion = kdv_gradient @ curvature  # (B w)_i ≃ ∫ u_xxx φ_i dx
        return -_mass_lu.solve(nonlinear_term(u) + kdv_epsilon**2 * dispersion)

    f"初期条件での |du/dt| の最大値 = {np.abs(kdv_rhs(np.cos(np.pi * np.arange(200) * 0.01))).max():.3f}"
    return (kdv_rhs,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### 初期条件と時間積分

    初期条件は周期区間 $[0,2)$ 上の $\cos(\pi x)$ である。
    時間積分は古典的な 4 次のルンゲ＝クッタ法（RK4）を固定刻み $Δt=10^{-4}$ で
    回す。分散項の固有値は純虚数なので、虚軸方向に安定領域を持つ RK4 が
    適している（前進オイラー法は虚軸上で必ず発散する）。
    """)
    return


@app.cell
def _(kdv_x, np, plt):
    kdv_u0 = np.cos(np.pi * kdv_x)

    _fig, _ax = plt.subplots(figsize=(6.4, 2.6), dpi=100)
    _ax.plot(kdv_x, kdv_u0)
    _ax.set_xlabel("$x$")
    _ax.set_ylabel("$u(x,0)$")
    _ax.set_title(r"initial condition $\cos(\pi x)$")
    _fig.tight_layout()
    _fig
    return (kdv_u0,)


@app.cell
def _(kdv_rhs, np):
    def rk4_step(u: np.ndarray, dt: float) -> np.ndarray:
        k1 = dt * kdv_rhs(u)
        k2 = dt * kdv_rhs(u + k1 / 2)
        k3 = dt * kdv_rhs(u + k2 / 2)
        k4 = dt * kdv_rhs(u + k3)
        return u + (k1 + 2 * k2 + 2 * k3 + k4) / 6

    # 1 ステップで初期条件がどれだけ動くかの確認。
    f"1 ステップ後の最大変化 = {np.abs(rk4_step(np.cos(np.pi * np.arange(200) * 0.01), 1e-4) - np.cos(np.pi * np.arange(200) * 0.01)).max():.2e}"
    return (rk4_step,)


@app.cell
def _(kdv_dt, kdv_frames, kdv_save_every, kdv_t_max, kdv_u0, np, rk4_step):
    _u = kdv_u0
    _snapshots = [kdv_u0]
    _times = [0.0]
    for _index in range(int(kdv_t_max / kdv_dt)):
        _u = rk4_step(_u, kdv_dt)
        if (_index + 1) % kdv_save_every == 0:
            _snapshots.append(_u)
            _times.append((_index + 1) * kdv_dt)

    kdv_u = np.array(_snapshots)
    kdv_t = np.array(_times)

    (
        f"保存したコマ数 {len(kdv_t)} (想定 {kdv_frames})",
        f"u の範囲 [{kdv_u.min():.3f}, {kdv_u.max():.3f}]",
        f"質量 ∫u dx の変化 {abs(kdv_u[-1].sum() - kdv_u[0].sum()) * 0.01:.2e}",
    )
    return kdv_t, kdv_u


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### 時空間ダイアグラム

    横軸 $x$・縦軸 $t$ で $u(x,t)$ を一枚にすると、$t\simeq0.3$ で波面が切り立ち、
    そこから生まれた複数の筋がそれぞれ一定の傾きで進んでいく様子が見える。
    傾きの逆数が伝播速度で、色が濃い（振幅が大きい）筋ほど速く進む。
    これが「振幅が大きいソリトンほど速い」という KdV の関係である。
    """)
    return


@app.cell
def _(kdv_length, kdv_t, kdv_u, plt):
    _fig, _ax = plt.subplots(figsize=(5.6, 4.4), dpi=100)
    _im = _ax.imshow(
        kdv_u,
        origin="lower",
        aspect="auto",
        cmap="magma",
        vmin=-1,
        vmax=3,
        extent=[0, kdv_length, kdv_t[0], kdv_t[-1]],
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
def _(artifacts_dir, kdv_length, kdv_t, kdv_u, kdv_x, plt, save_gif):
    _fig, _ax = plt.subplots(figsize=(6.4, 3.6), dpi=100)
    _ax.set_xlim(0, kdv_length)
    _ax.set_ylim(-1.4, 3.2)
    _ax.set_xlabel("$x$")
    _ax.set_ylabel("$u$")
    _ax.axhline(0, lw=0.5, color="0.7")
    _line, = _ax.plot([], [], lw=1.4)
    _title = _ax.set_title("KdV (FEM), t = 0.00")
    _fig.tight_layout()

    def _update(frame):
        _line.set_data(kdv_x, kdv_u[frame])
        _title.set_text(f"KdV (FEM), t = {kdv_t[frame]:.2f}")
        return (_line, _title)

    kdv_gif = artifacts_dir / "kdv.gif"
    save_gif(_fig, _update, len(kdv_t), kdv_gif, interval=50)
    return (kdv_gif,)


@app.cell
def _(gif_image, kdv_gif):
    gif_image(kdv_gif, alt="有限要素法による KdV 方程式の時間発展")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    アニメーションで確認できることは次の 4 点である。

    1. $t\lesssim0.3$: 移流項 $uu_x$ だけが効き、余弦波の下り坂が切り立つ。
       分散が無ければ破断時刻 $t_B=1/\pi\simeq0.318$ で勾配が無限大になる。
    2. $t\simeq0.3$〜$0.6$: 切り立った面が $δ^2u_{xxx}$ に耐えきれず、
       幅 $\sim δ$ の細い山が次々に生まれる。振幅は初期値の 2 倍以上に達する。
    3. $t\simeq1$: 8 個前後の山が振幅順に整列する。背の高い山ほど速く進むので、
       追い越しが起こる。**衝突しても波形が崩れず、位相がずれるだけ**で
       すり抜けることが、これらが単なる波ではなくソリトンである証拠。
    4. $t\simeq9.7$: 位相がほぼ揃い、初期の余弦波に近い形が復元する
       （再帰現象）。Fermi–Pasta–Ulam 問題で見つかった再帰と同じ現象で、
       Zabusky と Kruskal がソリトンという概念に到達したきっかけになった。

    最後の出力にある $\int u\,dx$ の変化がほぼゼロであることは、
    質量が保存されている（数値散逸が無い）ことの簡単な確認である。
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## NGSolve で 2 次元問題を解く
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    [NGSolve](https://ngsolve.org/) はメッシュ生成器 Netgen と組になった有限要素法の
    ライブラリで、弱形式を数式に近い形で書くとアセンブルまで自動で行ってくれる。
    このリポジトリでは `specialized` 依存グループに含まれるため、
    `uv sync --all-groups` で導入される（Colab 用の初期化セルは使わない）。

    NGSolve に付属する `Draw` は WebGL のウィジェットを返し、marimo の静的 HTML では
    描画されない。そこで、メッシュの三角形分割を matplotlib の
    `Triangulation` に移し替えるヘルパーと、`CoefficientFunction` や
    `GridFunction` を格子上でサンプリングするヘルパーを用意する。

    - `triangulation(mesh)`: 節点座標と三角形の頂点番号から
      `matplotlib.tri.Triangulation` を作る。`triplot` / `tricontourf` に渡せる。
    - `sample_scalar(func, mesh, xs, ys)`: スカラー場を格子上で評価して
      2 次元配列（`imshow` 用、行が $y$・列が $x$）にする。
    - `sample_vector(func, mesh, xs, ys)`: ベクトル場を格子上で評価して
      2 成分の配列（`quiver` 用）にする。
    """)
    return


@app.cell
def _(np):
    import ngsolve
    from matplotlib.tri import Triangulation

    def triangulation(mesh) -> Triangulation:
        points = np.array([vertex.point for vertex in mesh.vertices])
        triangles = np.array(
            [[vertex.nr for vertex in element.vertices] for element in mesh.Elements(ngsolve.VOL)]
        )
        return Triangulation(points[:, 0], points[:, 1], triangles)

    def sample_scalar(func, mesh, xs, ys) -> np.ndarray:
        return np.array([[func(mesh(px, py)) for px in xs] for py in ys], dtype=float)

    def sample_vector(func, mesh, xs, ys) -> np.ndarray:
        values = np.array([[func(mesh(px, py)) for px in xs] for py in ys], dtype=float)
        return values[:, :, 0], values[:, :, 1]

    f"NGSolve {ngsolve.__version__}"
    return ngsolve, sample_scalar, sample_vector, triangulation


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### 正方領域のポアソン方程式
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    [First NGSolve example](https://docu.ngsolve.org/latest/i-tutorials/unit-1.1-poisson/poisson.html)
    に沿って、単位正方形 $S=[0,1]^2$ 上のポアソン方程式
    $$
    \begin{aligned}
    &-\Delta u=f\quad\text{@バルク中},\\
    &u=0\quad\text{@下辺および右辺},\\
    &\frac{∂u}{∂n}=0\quad\text{@上辺および左辺}
    \end{aligned}
    $$
    を解く。右辺は $f(x,y)=x$ とする。
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    まずメッシュを生成する。`maxh` は要素の最大辺長で、小さくするほど
    細かいメッシュになる。もとの `maxh=0.2` から `maxh=0.12` へ細かくして、
    等高線が角張って見えないようにしている。
    """)
    return


@app.cell
def _(ngsolve, plt, triangulation):
    poisson_mesh = ngsolve.Mesh(ngsolve.unit_square.GenerateMesh(maxh=0.12))
    poisson_tri = triangulation(poisson_mesh)

    _fig, _ax = plt.subplots(figsize=(4.2, 4.0), dpi=100)
    _ax.triplot(poisson_tri, lw=0.5, color="0.35")
    _ax.set_aspect("equal")
    _ax.set_title(f"mesh: {poisson_mesh.nv} vertices, {poisson_mesh.ne} elements")
    _fig.tight_layout()
    _fig
    return poisson_mesh, poisson_tri


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    次に有限要素空間を定義する。`H1` は連続な区分多項式の空間で、
    `order=2` は 2 次要素（各辺の中点にも自由度が付く）を意味する。
    境界条件もここで指定し、`dirichlet="bottom|right"` とした辺の自由度は
    後の `FreeDofs()` から除かれて値が固定される。指定しなかった上辺・左辺には
    何もしなくても自然境界条件（$∂u/∂n=0$）が課される。
    """)
    return


@app.cell
def _(ngsolve, poisson_mesh):
    poisson_fes = ngsolve.H1(poisson_mesh, order=2, dirichlet="bottom|right")
    _free = sum(1 for flag in poisson_fes.FreeDofs() if flag)
    f"全自由度 {poisson_fes.ndof}, うち自由な自由度 {_free}, 固定された自由度 {poisson_fes.ndof - _free}"
    return (poisson_fes,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    有限要素法で使う 3 種類のオブジェクトを用意する。

    - `poisson_trial` ($u$): 解を展開するための試行関数。まだ数値を持たない記号。
    - `poisson_test` ($v$): 弱形式を作るための試験関数（重み関数）。
    - `poisson_gfu`: 有限要素空間上の関数そのもの。係数ベクトル `.vec` を持ち、
      連立方程式を解いた結果がここへ入る。
    """)
    return


@app.cell
def _(ngsolve, poisson_fes):
    poisson_trial = poisson_fes.TrialFunction()
    poisson_test = poisson_fes.TestFunction()
    poisson_gfu = ngsolve.GridFunction(poisson_fes)
    (
        f"trial/test は数値を持たない記号: {type(poisson_trial).__name__}",
        f"gfu は係数ベクトルを持つ有限要素空間上の関数: len(gfu.vec) = {len(poisson_gfu.vec)}",
    )
    return poisson_gfu, poisson_test, poisson_trial


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ポアソン方程式の弱形式は、試験関数 $v\in H^1_0(S)$ を掛けて部分積分すると
    $$
    \begin{aligned}
    &-Δu=f \Rightarrow∫(-Δ u)vdS=∫fvdS \\
    &\Leftrightarrow
    \oint_{∂S}\left(-\sum_i∂_iu\cdot v\right)dl^i+∫\sum_i∂_iu∂_ivdS=∫fvdS.
    \end{aligned}
    $$
    となる。ディリクレ境界では $v=0$、ノイマン境界では $∂u/∂n=0$ なので、
    1 つ目の表面項は消える。残るのは
    $$
    a(u,v)=\int\nabla u\cdot\nabla v\,dS,\quad
    \ell(v)=\int fv\,dS
    $$
    で、これがそのまま `BilinearForm` と `LinearForm` に対応する。
    """)
    return


@app.cell
def _(ngsolve, poisson_fes, poisson_test, poisson_trial):
    from ngsolve import BilinearForm, LinearForm, grad, x
    from ngsolve import dx as ngs_dx

    poisson_a = BilinearForm(poisson_fes, symmetric=True)
    poisson_a += grad(poisson_trial) * grad(poisson_test) * ngs_dx
    poisson_a.Assemble()

    poisson_f = LinearForm(poisson_fes)
    poisson_f += x * poisson_test * ngs_dx
    poisson_f.Assemble()

    f"a: {poisson_a.mat.height}x{poisson_a.mat.width}, 非ゼロ {poisson_a.mat.nze} 個 / f: 長さ {len(poisson_f.vec)}"
    return grad, ngs_dx, poisson_a, poisson_f, x


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    [Numpy Interface](https://docu.ngsolve.org/latest/how_to/howto_numpy.html) を使うと、
    組み上がった右辺ベクトルと左辺行列を NumPy / SciPy へ取り出せる。
    左辺の疎行列は、節点の番号付けを反映した対称なパターンになる。
    """)
    return


@app.cell
def _(plt, poisson_a, poisson_f):
    import scipy.sparse as sp

    _rows, _cols, _vals = poisson_a.mat.COO()
    poisson_matrix = sp.csr_matrix((_vals, (_rows, _cols)))

    _fig, _axes = plt.subplots(1, 2, figsize=(9.0, 3.4), dpi=100)
    _axes[0].plot(poisson_f.vec.FV().NumPy(), lw=0.8)
    _axes[0].set_title("right-hand side vector $f$")
    _axes[0].set_xlabel("degree of freedom")
    _axes[1].spy(poisson_matrix, markersize=0.6)
    _axes[1].set_title(f"stiffness matrix, nnz={poisson_matrix.nnz}")
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    固定されていない自由度についてだけ連立方程式を解き、結果を `tricontourf` で
    描く。下辺と右辺で $u=0$ に固定され、$f=x$ に押し上げられた分布が
    左上に寄っている様子が確認できる。
    """)
    return


@app.cell
def _(np, plt, poisson_a, poisson_f, poisson_fes, poisson_gfu, poisson_mesh, poisson_tri):
    poisson_gfu.vec.data = poisson_a.mat.Inverse(freedofs=poisson_fes.FreeDofs()) * poisson_f.vec
    poisson_values = np.array(
        [poisson_gfu(poisson_mesh(px, py)) for px, py in zip(poisson_tri.x, poisson_tri.y)]
    )

    _fig, _ax = plt.subplots(figsize=(4.6, 4.0), dpi=100)
    _contour = _ax.tricontourf(poisson_tri, poisson_values, levels=24, cmap="viridis")
    _ax.triplot(poisson_tri, lw=0.2, color="w", alpha=0.5)
    _ax.set_aspect("equal")
    _ax.set_title(f"solution $u$, max = {poisson_values.max():.4f}")
    _fig.colorbar(_contour, ax=_ax)
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### 風のある熱方程式（移流拡散方程式）
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    [Parabolic model problem](https://docu.ngsolve.org/latest/i-tutorials/unit-3.1-parabolic/parabolic.html)
    に従って、時間依存の移流拡散方程式
    $$
    \frac{∂u}{∂t}-α\Delta u+b\cdot\nabla u=f
    $$
    を正方領域 $[-1,1]^2$ で解く。$α=0.01$ は熱の拡散しやすさ、
    $b$ は温度を運ぶ「風」である。全周をディリクレ境界とする。

    弱形式を離散化すると、質量行列 $M$ と作用素行列 $A$ を使って
    $$
    M\dot{u}+Au=f
    $$
    になる。時間方向は暗黙オイラー法
    $$
    (M+Δt\,A)u^{n+1}=Mu^n+Δt\,f
    $$
    で進める。この形の利点は、拡散項による強い剛性があっても
    時間刻みを安定性ではなく精度だけで選べることである。
    実装では等価な増分形
    $$
    u^{n+1}=u^n+(M+Δt\,A)^{-1}\left(Δt\,f-Δt\,Au^n\right)
    $$
    を使い、$(M+Δt\,A)^{-1}$ を最初に 1 回だけ作って使い回す。

    もとの設定は `maxh=0.25`、$Δt=10^{-3}$、$t_{\max}=2$ で保存 11 コマだった。
    ここでは `maxh=0.15`、$Δt=5\times10^{-4}$、$t_{\max}=4$ の 81 コマにする。
    風の角速度は領域中央で約 2 なので、1 回転に要する時間は $\pi\simeq3.14$ である。
    $t_{\max}=4$ まで進めることで、1 周を超えて分布が定常状態へ落ち着くところまで
    観察できる。
    """)
    return


@app.cell
def _(ngsolve):
    from netgen.occ import OCCGeometry, Rectangle, X, Y

    _shape = Rectangle(2, 2).Face().Move((-1, -1, 0))
    _shape.edges.Min(X).name = "left"
    _shape.edges.Max(X).name = "right"
    _shape.edges.Min(Y).name = "bottom"
    _shape.edges.Max(Y).name = "top"
    heat_mesh = ngsolve.Mesh(OCCGeometry(_shape, dim=2).GenerateMesh(maxh=0.15))

    f"mesh: {heat_mesh.nv} vertices, {heat_mesh.ne} elements"
    return (heat_mesh,)


@app.cell
def _(heat_mesh, ngsolve):
    heat_fes = ngsolve.H1(heat_mesh, order=3, dirichlet="bottom|right|left|top")
    heat_trial, heat_test = heat_fes.TnT()
    heat_dt = 5e-4
    heat_t_max = 4.0
    heat_frames = 81

    f"自由度 {heat_fes.ndof}, 時間ステップ {int(heat_t_max / heat_dt)} 回, 保存 {heat_frames} コマ"
    return heat_dt, heat_fes, heat_frames, heat_t_max, heat_test, heat_trial


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    風の場は
    $$
    b(x,y)=\left(2y(1-x^2),\,-2x(1-y^2)\right)
    $$
    である。境界上で法線成分がゼロになるので、風は領域から出ていかない。
    原点のまわりを時計回りに回る渦になっており、中心付近では
    $b\simeq(2y,-2x)$、つまり角速度 2 の剛体回転である。
    """)
    return


@app.cell
def _(heat_mesh, ngsolve, np, plt, sample_vector, x):
    from ngsolve import y

    wind = ngsolve.CoefficientFunction((2 * y * (1 - x * x), -2 * x * (1 - y * y)))

    _grid = np.linspace(-0.95, 0.95, 16)
    _bx, _by = sample_vector(wind, heat_mesh, _grid, _grid)
    _mesh_x, _mesh_y = np.meshgrid(_grid, _grid)

    _fig, _ax = plt.subplots(figsize=(4.4, 4.0), dpi=100)
    _ax.quiver(_mesh_x, _mesh_y, _bx, _by, np.hypot(_bx, _by), cmap="viridis")
    _ax.set_aspect("equal")
    _ax.set_xlabel("$x$")
    _ax.set_ylabel("$y$")
    _ax.set_title("wind field $b$")
    _fig.tight_layout()
    _fig
    return wind, y


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    双一次形式 $a(u,v)=\int(α\nabla u\cdot\nabla v+(b\cdot\nabla u)v)\,dS$ と
    質量形式 $m(u,v)=\int uv\,dS$ を組み立てる。移流項があるため $a$ は
    対称ではない（`symmetric=False`）。
    """)
    return


@app.cell
def _(grad, heat_fes, heat_test, heat_trial, ngsolve, wind):
    heat_a = ngsolve.BilinearForm(heat_fes, symmetric=False)
    heat_a += (
        0.01 * grad(heat_trial) * grad(heat_test) * ngsolve.dx
        + wind * grad(heat_trial) * heat_test * ngsolve.dx
    )
    heat_a.Assemble()

    heat_m = ngsolve.BilinearForm(heat_fes, symmetric=False)
    heat_m += heat_trial * heat_test * ngsolve.dx
    heat_m.Assemble()

    f"m.mat.nze = {heat_m.mat.nze}, a.mat.nze = {heat_a.mat.nze}"
    return heat_a, heat_m


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    $M^\ast=M+Δt\,A$ を作って逆行列（実体は疎行列の直接ソルバー）を用意する。
    `AsVector()` で行列を 1 本のベクトルとして扱えるのは、$M$ と $A$ が同じ
    疎パターンを共有しているためである。
    """)
    return


@app.cell
def _(heat_a, heat_dt, heat_fes, heat_m):
    heat_mstar = heat_m.mat.CreateMatrix()
    heat_mstar.AsVector().data = heat_m.mat.AsVector() + heat_dt * heat_a.mat.AsVector()
    _inverse = heat_mstar.Inverse(freedofs=heat_fes.FreeDofs())

    def heat_solve(vec):
        """(M + Δt A)^-1 を適用する。逆作用素は 1 度だけ作って使い回す。"""
        return _inverse * vec

    f"mstar.nze = {heat_mstar.nze}, len(mstar.AsVector()) = {len(heat_mstar.AsVector())}"
    return (heat_solve,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    外力 $f$ は、$(-0.5,0)$ に正・$(0.5,0)$ に負のガウス分布を置いた
    「熱の湧き出しと吸い込み」の対にする。この $f$ は時間に依らないので、
    最終的には風による輸送・拡散・湧き出しが釣り合った定常状態へ落ち着く。
    """)
    return


@app.cell
def _(heat_fes, heat_mesh, heat_test, ngsolve, np, plt, sample_scalar, x, y):
    heat_source_cf = ngsolve.exp(-6 * ((x + 0.5) ** 2 + y * y)) - ngsolve.exp(
        -6 * ((x - 0.5) ** 2 + y * y)
    )
    heat_f = ngsolve.LinearForm(heat_fes)
    heat_f += heat_source_cf * heat_test * ngsolve.dx
    heat_f.Assemble()

    _grid = np.linspace(-0.999, 0.999, 96)
    _values = sample_scalar(heat_source_cf, heat_mesh, _grid, _grid)

    _fig, _ax = plt.subplots(figsize=(4.6, 4.0), dpi=100)
    _im = _ax.imshow(
        _values, origin="lower", cmap="RdBu_r", vmin=-1, vmax=1, extent=[-1, 1, -1, 1]
    )
    _ax.set_aspect("equal")
    _ax.set_title("source term $f$")
    _fig.colorbar(_im, ax=_ax)
    _fig.tight_layout()
    _fig
    return (heat_f,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    初期条件は $u(x,y,0)=(1-y^2)x$ である。左半分が冷たく右半分が暖かい分布で、
    境界上の値がそのままディリクレ境界条件として保持される。
    """)
    return


@app.cell
def _(heat_fes, heat_mesh, ngsolve, np, plt, sample_scalar, x, y):
    heat_gfu = ngsolve.GridFunction(heat_fes)
    heat_gfu.Set((1 - y * y) * x)

    _grid = np.linspace(-0.999, 0.999, 96)
    _values = sample_scalar(heat_gfu, heat_mesh, _grid, _grid)

    _fig, _ax = plt.subplots(figsize=(4.6, 4.0), dpi=100)
    _im = _ax.imshow(
        _values, origin="lower", cmap="RdBu_r", vmin=-1, vmax=1, extent=[-1, 1, -1, 1]
    )
    _ax.set_aspect("equal")
    _ax.set_title("initial condition $u(x,y,0)$")
    _fig.colorbar(_im, ax=_ax)
    _fig.tight_layout()
    _fig
    return (heat_gfu,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### 時間発展

    暗黙オイラー法で $t=4$ まで 8000 ステップ進め、100 ステップごとに
    $96\times96$ の格子へサンプリングして 81 コマぶんの配列を作る。
    可視化用のサンプリングを積分ループの中で済ませておくと、
    大量の `GridFunction` を保持せずにアニメーションを作れる。
    """)
    return


@app.cell
def _(
    heat_a,
    heat_dt,
    heat_f,
    heat_frames,
    heat_gfu,
    heat_mesh,
    heat_solve,
    heat_t_max,
    np,
    sample_scalar,
):
    def time_stepping(gfu, steps: int, sample_every: int, xs, ys):
        """暗黙オイラー法で進めながら、一定間隔で格子上にサンプリングする。"""
        snapshots = [sample_scalar(gfu, heat_mesh, xs, ys)]
        times = [0.0]
        for index in range(steps):
            residual = heat_dt * heat_f.vec - heat_dt * heat_a.mat * gfu.vec
            gfu.vec.data = gfu.vec.data + heat_solve(residual)
            if (index + 1) % sample_every == 0:
                snapshots.append(sample_scalar(gfu, heat_mesh, xs, ys))
                times.append((index + 1) * heat_dt)
        return np.array(snapshots), np.array(times)

    heat_grid = np.linspace(-0.999, 0.999, 96)
    heat_u, heat_t = time_stepping(
        heat_gfu,
        int(heat_t_max / heat_dt),
        int(heat_t_max / heat_dt / (heat_frames - 1)),
        heat_grid,
        heat_grid,
    )

    f"コマ数 {len(heat_t)}, u の範囲 [{heat_u.min():.3f}, {heat_u.max():.3f}]"
    return heat_t, heat_u


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### アニメーション

    81 コマを 1 コマ 100 ms で再生するので、再生時間は約 8.1 秒である。
    色は $0$ を白とする発散カラーマップで、範囲を全コマ $[-1,1]$ に固定する。
    """)
    return


@app.cell
def _(artifacts_dir, heat_t, heat_u, plt, save_gif):
    _fig, _ax = plt.subplots(figsize=(4.6, 4.0), dpi=100)
    _im = _ax.imshow(
        heat_u[0], origin="lower", cmap="RdBu_r", vmin=-1, vmax=1, extent=[-1, 1, -1, 1]
    )
    _ax.set_aspect("equal")
    _ax.set_xlabel("$x$")
    _ax.set_ylabel("$y$")
    _title = _ax.set_title("convection-diffusion, t = 0.00")
    _fig.colorbar(_im, ax=_ax)
    _fig.tight_layout()

    def _update(frame):
        _im.set_data(heat_u[frame])
        _title.set_text(f"convection-diffusion, t = {heat_t[frame]:.2f}")
        return (_im, _title)

    ngsolve_heat_gif = artifacts_dir / "ngsolve_heat.gif"
    save_gif(_fig, _update, len(heat_t), ngsolve_heat_gif, interval=100)
    return (ngsolve_heat_gif,)


@app.cell
def _(gif_image, ngsolve_heat_gif):
    gif_image(ngsolve_heat_gif, alt="NGSolve による移流拡散方程式の時間発展")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    アニメーションで確認できることは次の 3 点である。

    1. $t\lesssim1$: 左右に分かれていた初期分布が風に巻き取られ、
       境界に沿って渦を巻く S 字の腕になる。この段階は移流が支配的で、
       模様はほとんど薄まらない。
    2. $t\simeq1$〜$3$: 腕が中心へ引き込まれて細くなる。細くなるほど
       $\nabla u$ が大きくなるので、拡散項 $α\Delta u$ が効き始めて
       コントラストが急速に落ちる。これは移流が拡散を助ける典型的な効果。
    3. $t\gtrsim3$: 湧き出し $f$・風による輸送・拡散が釣り合い、
       弱い定常パターンへ落ち着く。境界のディリクレ値が変わらないため、
       分布が完全にゼロへ均されることはない。

    ペクレ数（移流と拡散の比）は、領域全体では $|b|\ell/α\simeq2\times1/0.01=200$、
    要素 1 個あたりでも $|b|h/(2α)\simeq2\times0.15/0.02=15$ と移流が優勢である。
    標準ガレルキン法はこの条件で数値振動を起こしやすいのだが、
    3 次要素（`order=3`）を使っているため実効的な格子が細かく、
    実際に解の値は初期条件と同じ $[-1,1]$ の範囲に収まり
    （最終出力の `u の範囲` を参照）、目立った振動は生じていない。
    """)
    return


if __name__ == "__main__":
    app.run()
