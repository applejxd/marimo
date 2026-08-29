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

    時間について 1 階、空間について 2 階以上の微分を含む発展方程式を、
    有限差分で空間を離散化し `scipy.integrate.solve_ivp` で時間積分する
    （`legacy/simulation/ParabolicPDE.ipynb` を marimo 向けに整理したものである）。

    扱う 3 つの方程式は、線形の拡散から非線形・分散性へ段階的に難しくなる。

    | 節 | 方程式 | 空間刻み | 時間範囲 | 見どころ |
    | --- | --- | --- | --- | --- |
    | 拡散方程式 | $u_t=Du_{xx}$ | $h=0.01$（400 点） | $t\in[0,0.1]$ | 解析解（熱核）との一致 |
    | バーガース方程式 | $u_t=-uu_x+Du_{xx}$ | $h=0.005$（200 点） | $t\in[0,2]$ | 衝撃波の形成と粘性減衰 |
    | KdV 方程式 | $u_t=-6uu_x-u_{xxx}$ | $h\simeq0.195$（512 点） | $t\in[0,10]$ | 分散衝撃波とソリトン列 |

    どの節でも、現象が一巡するまで（拡散なら分布の幅が 5 倍、バーガースなら
    振幅が 1/6、KdV ならソリトン列が出そろうまで）積分してからアニメーションに
    している。
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## この notebook の共通部品

    次のセルは、以降のすべてのアニメーションが共有する出力先とヘルパーを用意する。

    - `artifacts_dir`: GIF と `.npz` の保存先
      `notebooks/simulation/_generated/ParabolicPDE/`。Git の追跡対象外なので、
      実行するたびに再生成される。
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

    artifacts_dir = Path(__file__).resolve().parent / "_generated" / "ParabolicPDE"
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
    ## 基本的な放物型: 拡散方程式
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    拡散方程式は
    $$
    \frac{∂u}{∂t}=D\nabla^2u
    $$
    である。ここでは 1 次元で、長さと時間を適切に規格化して $D=1$ として解く。

    初期条件をデルタ関数 $u(x,0)=δ(x)$ に取ると、解は解析的に求まり
    $$
    u(x,t)=\frac{1}{\sqrt{4\pi Dt}}\exp\left(-\frac{x^2}{4Dt}\right)
    $$
    となる（熱核、あるいはグリーン関数）。数値解をこの厳密解と重ねて描くことで、
    離散化と時間積分がどれだけ正確かをその場で確認できる。

    計算領域は $x\in[-2,2]$、空間刻みは $h=0.01$（格子点 400 個）とする。
    $t=0.1$ での分布の幅は $\sqrt{2Dt}\simeq0.45$ なので、境界 $x=\pm2$ は
    十分に遠く、境界条件の影響を受けない。
    """)
    return


@app.cell
def _():
    import numpy as np
    from scipy import sparse

    dx_diff = 0.01
    x_diff = np.arange(-2.0, 2.0, dx_diff)
    n_diff = len(x_diff)

    t_max_diff = 0.1
    frame_dt_diff = 0.001
    # デルタ関数の高さ 1/h は軸に収まらないため、少し発展した時刻から描き始める。
    t_diff = np.arange(0.004, t_max_diff + 0.5 * frame_dt_diff, frame_dt_diff)

    f"格子点 {n_diff} 個, コマ数 {len(t_diff)}, 陽解法の安定条件 k <= h^2/(2D) = {dx_diff**2 / 2}"
    return dx_diff, frame_dt_diff, n_diff, np, sparse, t_diff, t_max_diff, x_diff


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### 係数行列
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    2 階微分を中心差分
    $$
    \frac{∂^2u}{∂x^2}(x_i)\simeq\frac{u_{i+1}-2u_i+u_{i-1}}{h^2}
    $$
    で置き換える。境界は、十分遠方では $u$ が $0$ に減衰するという想定のもと、
    ディリクレ条件
    $$
    u(x_0)=u(x_N)=0
    $$
    を課す。$u$ を適切に平行移動して定義し直せば、境界値は常にこの形にできる。

    次のセルの `derivative_matrices` は、この境界条件に対応する 1 階微分行列
    $(S_{+}-S_{-})/2h$ と 2 階微分行列 $(S_{+}-2I+S_{-})/h^2$ を組で返す
    （1 階微分は後のバーガース方程式で使う）。両端の行では領域外を指す
    シフト成分を落としており、それがディリクレ条件に対応する。

    構造が読み取れるよう、まず $n=6$、$h=1$ の小さな例を表示する。
    """)
    return


@app.cell
def _(np, sparse):
    def derivative_matrices(n: int, h: float) -> tuple[sparse.csr_matrix, sparse.csr_matrix]:
        """両端をディリクレ条件（u=0）とする 1 階・2 階の中心差分行列を返す。"""
        shift_back = sparse.lil_matrix(sparse.eye(n, k=-1))
        shift_back[0, :] = 0
        shift_back = sparse.csr_matrix(shift_back)
        shift_forward = sparse.lil_matrix(sparse.eye(n, k=1))
        shift_forward[-1, :] = 0
        shift_forward = sparse.csr_matrix(shift_forward)
        identity = sparse.identity(n)
        first = (shift_forward - shift_back) / (2 * h)
        second = (shift_forward - 2 * identity + shift_back) / h**2
        return first.tocsr(), second.tocsr()

    np.set_printoptions(edgeitems=3, precision=3, suppress=True)
    derivative_matrices(6, 1.0)[1].toarray()
    return (derivative_matrices,)


@app.cell
def _(derivative_matrices, dx_diff, n_diff):
    _, laplacian_diff = derivative_matrices(n_diff, dx_diff)
    f"{laplacian_diff.shape}, nnz={laplacian_diff.nnz}, 対角成分 = {-2 / dx_diff**2:.0f}"
    return (laplacian_diff,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### 初期条件と時間刻み
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    デルタ関数を格子上で表現するには、中央の 1 点だけに $1/h$ を置く。
    こうすると $\sum_iu_ih=1$ となり、全体の積分が $1$ に規格化される。
    今回は $h=0.01$ なので高さは $100$ である。

    [数値解の不安定性より、時間方向の刻み幅 $k$ は空間方向の刻み幅 $h$ に対して
    十分に小さく取る必要がある](http://www.nibb.ac.jp/miyakohp/asari/htdocs/?page_id=60)。
    具体的には
    $$
    \frac{Dk}{h^2}\leq\frac{1}{2}
    $$
    で、これは差分化した係数行列の固有値を陽解法の安定領域へ収める条件である。
    $h=0.01$、$D=1$ なら $k\leq5\times10^{-5}$ となり、アニメーションのコマ間隔
    $Δt_{\text{frame}}=10^{-3}$ よりずっと小さくなる。適応刻みの `RK45` は
    この制約を自動的に満たすように内部の刻み幅を選ぶため、
    コマ間隔とは独立に安定性が保たれる。
    """)
    return


@app.cell
def _(dx_diff, n_diff, np, plt, x_diff):
    u0_diff = np.zeros(n_diff)
    u0_diff[n_diff // 2] = 1 / dx_diff

    _fig, _ax = plt.subplots(figsize=(6.4, 2.6), dpi=100)
    _ax.plot(x_diff, u0_diff)
    _ax.set_xlim(-0.5, 0.5)
    _ax.set_xlabel("$x$")
    _ax.set_ylabel("$u(x,0)$")
    _ax.set_title(f"discretized delta function (height 1/h = {1 / dx_diff:.0f})")
    _fig.tight_layout()
    _fig
    return (u0_diff,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### 時間積分

    `t_eval` にコマの時刻を渡し、必要な時刻の値だけを受け取る。
    `sol.success` の確認は必須である。刻み幅が下限に達して積分が途中で
    打ち切られても `solve_ivp` は例外を投げないため、確認を省くと
    発散した結果をそのまま可視化してしまう。
    """)
    return


@app.cell
def _(frame_dt_diff, laplacian_diff, t_diff, t_max_diff, u0_diff):
    from scipy.integrate import solve_ivp

    def _rhs(_t, u):
        return laplacian_diff @ u

    _sol = solve_ivp(
        _rhs,
        t_span=(0, t_max_diff),
        y0=u0_diff,
        method="RK45",
        t_eval=t_diff,
        max_step=frame_dt_diff,
        rtol=1e-8,
        atol=1e-10,
    )
    if not _sol.success:
        raise RuntimeError(_sol.message)
    u_diff = _sol.y.T

    f"success={_sol.success}, 右辺評価 {_sol.nfev} 回, ピーク {u_diff[0].max():.3f} -> {u_diff[-1].max():.3f}"
    return solve_ivp, u_diff


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### 可視化

    各コマで数値解（実線）と熱核の厳密解（破線）を重ねる。
    デルタ関数の高さ $1/h=100$ は縦軸に収まらないので、$t=0.004$
    （厳密解のピークが $1/\sqrt{4\pi t}\simeq4.46$）から描き始める。

    97 コマを 1 コマ 80 ms で再生するので、再生時間は約 7.8 秒である。
    """)
    return


@app.cell
def _(artifacts_dir, dx_diff, np, plt, save_gif, t_diff, u_diff, x_diff):
    def _exact(t):
        return np.exp(-(x_diff**2) / (4 * t)) / np.sqrt(4 * np.pi * t)

    _l2_error = np.sqrt(dx_diff) * np.linalg.norm(u_diff[-1] - _exact(t_diff[-1]))

    _fig, _ax = plt.subplots(figsize=(6.4, 3.6), dpi=100)
    _ax.set_xlim(-1.5, 1.5)
    _ax.set_ylim(0, 5)
    _ax.set_xlabel("$x$")
    _ax.set_ylabel("$u$")
    _numeric, = _ax.plot([], [], lw=1.8, label="finite difference")
    _exact_line, = _ax.plot([], [], lw=1.2, ls="--", color="crimson", label="heat kernel")
    _ax.legend(loc="upper right")
    _title = _ax.set_title("diffusion, t = 0.000")
    _fig.tight_layout()

    def _update(frame):
        _numeric.set_data(x_diff, u_diff[frame])
        _exact_line.set_data(x_diff, _exact(t_diff[frame]))
        _title.set_text(f"diffusion, t = {t_diff[frame]:.3f}")
        return (_numeric, _exact_line, _title)

    diffusion_1d_gif = artifacts_dir / "diffusion_1d.gif"
    (
        save_gif(_fig, _update, len(t_diff), diffusion_1d_gif, interval=80),
        f"最終時刻の L2 誤差 = {_l2_error:.2e}",
    )
    return (diffusion_1d_gif,)


@app.cell
def _(diffusion_1d_gif, gif_image):
    gif_image(diffusion_1d_gif, alt="1次元拡散方程式の時間発展")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    数値解と厳密解は目視では区別できず、最終時刻での $L^2$ 誤差も $10^{-5}$ 程度に
    収まる。分布の幅は $\sqrt{2Dt}$ で広がり、ピークは $1/\sqrt{4\pi Dt}$ で
    下がるため、$t$ が 4 倍になるとピークは半分になる。

    ここが双曲型との決定的な違いである。波動方程式では波形が形を保って伝わったが、
    拡散方程式では高周波成分ほど速く減衰し（波数 $k$ の成分は $e^{-Dk^2t}$）、
    どんな初期分布も最終的に滑らかなガウス分布へ近づく。
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 非線形性のある放物型: バーガース方程式
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    バーガース方程式
    $$
    \frac{∂u(x,t)}{∂t}=-u(x,t)\frac{∂u(x,t)}{∂x}+D\frac{∂^2u(x,t)}{∂x^2}
    $$
    は、移流項 $-uu_x$ と拡散項 $Du_{xx}$ の競合を最小構成で表した方程式である。
    移流項は「値が大きいところほど速く動く」ため波形を切り立たせ、
    拡散項はそれを鈍らせる。両者が釣り合った定常構造が衝撃波である。

    次の条件で解く。

    - 拡散係数 $D=0.01$
    - 空間領域 $x\in[0,1]$、空間刻み $h=0.005$（格子点 200 個）
    - 時間範囲 $t\in[0,2]$、1 コマ $Δt_{\text{frame}}=0.01$（201 コマ）
    - 境界条件 $u(0,t)=u(1,t)=0$
    - 初期条件 $u(x,0)=\sin(2\pi x)$

    空間刻みを $h=0.02$ から $h=0.005$ へ細かくしたのは、衝撃波の厚みが
    $\sim D/\max|u|=0.01$ 程度しかないためである。$h=0.02$ では衝撃波が
    格子 1 個分より薄くなり、切り立った部分に非物理的な振動が出る。
    $h=0.005$ なら遷移層に 2〜4 点が入り、滑らかに解像される。
    """)
    return


@app.cell
def _(np):
    diffusion_const = 0.01
    dx_burgers = 0.005
    x_burgers = np.arange(0, 1, dx_burgers)
    n_burgers = len(x_burgers)

    t_max_burgers = 2.0
    frame_dt_burgers = 0.01
    t_burgers = np.arange(0, t_max_burgers + 0.5 * frame_dt_burgers, frame_dt_burgers)

    u0_burgers = np.sin(2 * np.pi * x_burgers)

    f"格子点 {n_burgers} 個, コマ数 {len(t_burgers)}, 衝撃波の厚み ~ D/max|u| = {diffusion_const}"
    return (
        diffusion_const,
        dx_burgers,
        frame_dt_burgers,
        n_burgers,
        t_burgers,
        t_max_burgers,
        u0_burgers,
        x_burgers,
    )


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    右辺は拡散方程式と同じ差分行列から組み立てる。異なるのは移流項が
    `-u * (first @ u)` という $u$ の二次形式になっている点だけで、
    ここが非線形性の出どころである。行列とベクトルの積が毎回必要になるため、
    右辺の評価回数は拡散方程式より増える。

    出力は $t=0$ における 2 つの項の大きさである。移流項が $\pi\simeq3.14$、
    拡散項が $4\pi^2D\simeq0.39$ で、初期状態では移流項が 1 桁大きく、
    波形を切り立たせる働きが優勢だと分かる。切り立って $u_{xx}$ が
    大きくなると拡散項が追いつき、そこで衝撃波の厚みが決まる。
    """)
    return


@app.cell
def _(derivative_matrices, diffusion_const, dx_burgers, n_burgers, np, u0_burgers):
    _first, _second = derivative_matrices(n_burgers, dx_burgers)

    def burgers_rhs(_t, u):
        return -u * (_first @ u) + diffusion_const * (_second @ u)

    # 両端の行は片側のシフト成分を落としてあるので、大きさの比較は内点で行う。
    _advection = (-u0_burgers * (_first @ u0_burgers))[1:-1]
    _diffusion = (diffusion_const * (_second @ u0_burgers))[1:-1]
    (
        f"t=0 の内点での移流項の最大値 {np.abs(_advection).max():.2f}",
        f"拡散項の最大値 {np.abs(_diffusion).max():.2f}",
    )
    return (burgers_rhs,)


@app.cell
def _(burgers_rhs, frame_dt_burgers, solve_ivp, t_burgers, t_max_burgers, u0_burgers):
    _sol = solve_ivp(
        burgers_rhs,
        t_span=(0, t_max_burgers),
        y0=u0_burgers,
        method="RK45",
        t_eval=t_burgers,
        max_step=frame_dt_burgers,
        rtol=1e-8,
        atol=1e-10,
    )
    if not _sol.success:
        raise RuntimeError(_sol.message)
    u_burgers = _sol.y.T

    f"success={_sol.success}, 右辺評価 {_sol.nfev} 回, 振幅 {abs(u_burgers[0]).max():.3f} -> {abs(u_burgers[-1]).max():.3f}"
    return (u_burgers,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### 時空間ダイアグラム

    横軸 $x$・縦軸 $t$ で $u(x,t)$ を一枚にすると、$x\simeq0.5$ に向かって
    等値線が集まり、そこから上へ細い界面がまっすぐ伸びる様子が見える。
    この界面が衝撃波である。
    """)
    return


@app.cell
def _(plt, t_burgers, u_burgers, x_burgers):
    _fig, _ax = plt.subplots(figsize=(5.6, 4.0), dpi=100)
    _im = _ax.imshow(
        u_burgers,
        origin="lower",
        aspect="auto",
        cmap="RdBu_r",
        vmin=-1,
        vmax=1,
        extent=[x_burgers[0], x_burgers[-1], t_burgers[0], t_burgers[-1]],
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

    201 コマを 1 コマ 50 ms で再生するので、再生時間は約 10 秒である。
    """)
    return


@app.cell
def _(artifacts_dir, plt, save_gif, t_burgers, u_burgers, x_burgers):
    _fig, _ax = plt.subplots(figsize=(6.4, 3.6), dpi=100)
    _ax.set_xlim(0, 1)
    _ax.set_ylim(-1.05, 1.05)
    _ax.set_xlabel("$x$")
    _ax.set_ylabel("$u$")
    _ax.axhline(0, lw=0.5, color="0.7")
    _line, = _ax.plot([], [], lw=1.6)
    _title = _ax.set_title("Burgers, t = 0.000")
    _fig.tight_layout()

    def _update(frame):
        _line.set_data(x_burgers, u_burgers[frame])
        _title.set_text(f"Burgers, t = {t_burgers[frame]:.3f}")
        return (_line, _title)

    burgers_1d_gif = artifacts_dir / "burgers_1d.gif"
    save_gif(_fig, _update, len(t_burgers), burgers_1d_gif, interval=50)
    return (burgers_1d_gif,)


@app.cell
def _(burgers_1d_gif, gif_image):
    gif_image(burgers_1d_gif, alt="バーガース方程式の時間発展")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    アニメーションは次の 3 つの局面に分かれる。

    1. $t\lesssim0.2$: 正弦波の下り坂が切り立っていく。$u$ が大きい部分が
       速く右へ動くためで、粘性が無ければ $t=1/(2\pi)\simeq0.16$ で
       勾配が無限大になる（波の突っ立ち）。
    2. $t\simeq0.2$〜$0.5$: 勾配が $1/D$ 程度で頭打ちになり、厚み $\sim D$ の
       衝撃波が $x\simeq0.5$ に立つ。ここで移流と拡散が釣り合っている。
    3. $t\gtrsim0.5$: 衝撃波を通じてエネルギーが散逸し、三角波状の波形が
       全体として減衰する。$t=2$ では振幅が初期の 1/6 程度まで落ちる。

    振幅が単調に減る点は拡散方程式と同じだが、減衰の主役が
    「なだらかにならす拡散」ではなく「非線形で作られた急峻な界面での散逸」に
    変わっているのが非線形性の効果である。
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 非線形と 3 階微分を含む方程式: KdV 方程式
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    [python で学ぶ計算物理](http://www.physics.okayama-u.ac.jp/~otsuki/lecture/CompPhys2/pde/kdv.html)
    から移植した設定で、KdV 方程式
    $$
    \frac{∂u(x,t)}{∂t}+6u(x,t)\frac{∂u(x,t)}{∂x}+\frac{∂^3u(x,t)}{∂x^3}=0
    $$
    を初期値問題として解く。空間方向は周期境界条件である。

    バーガース方程式の拡散項 $Du_{xx}$ が、ここでは分散項 $-u_{xxx}$ に
    置き換わっている。拡散項がエネルギーを散逸させたのに対し、分散項は
    エネルギーを保存したまま波数ごとに位相速度を変える。その結果、
    切り立った波面は「鈍る」のではなく**振動する列に分解**され、
    最終的に振幅と速度が対応した孤立波（ソリトン）の列になる。

    次の条件で解く。

    - 空間領域 $x\in[0,100)$、格子点 $n_x=512$（$h\simeq0.195$）
    - 時間範囲 $t\in[0,10]$、コマ数 161
    - 初期条件 $u(x,0)=\sin(2\pi x/100)$

    もとの設定は $n_x=200$、$t\leq4$ だったが、この格子では分散項が解像できず、
    $t\simeq3.17$ で `solve_ivp` が
    `Required step size is less than spacing between numbers.` を返して
    積分が破綻していた。`sol.success` を確認していなかったため、
    アニメーションの後半には発散した値が描かれていた。
    $n_x=512$ にすると最後まで安定に積分でき、$n_x=1024$ で計算した場合と
    ソリトンの本数も位置も一致する（尖頭の高さだけは格子に依存し、
    $n_x=512$ で 2.97、$n_x=1024$ で 2.83 と数 % 変わる）。
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### 空間差分の係数行列

    周期境界条件なので、`np.roll` で単位行列の列を巡回させるだけで
    隣接点を指す行列が作れる。左端の 1 つ手前が右端になる。
    3 階微分には 5 点ステンシル
    $$
    \frac{∂^3f}{∂x^3}(x_i)\simeq\frac{f_{i+2}-2f_{i+1}+2f_{i-1}-f_{i-2}}{2h^3}
    $$
    を使う。$1/h^3$ という因子のため、$h$ を半分にすると係数は 8 倍になり、
    陽解法が必要とする時間刻みも急激に小さくなる。
    これが KdV の数値計算を重くしている理由である。
    """)
    return


@app.cell
def _(np, sparse):
    def make_differential_ops(nx: int, dx: float):
        """周期境界条件の 1 階・2 階・3 階中心差分行列を返す。"""
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
    右辺 $-6uu_x-u_{xxx}$ を関数として定義する。`solve_ivp` の `args` を通じて
    差分行列を渡す形にしてあるので、格子数を変えて収束を確かめるときも
    この関数はそのまま使える。
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
    ### 時間積分

    刻み幅の上限を `max_step` で縛らず、`RK45` の適応制御に任せる。
    5 点ステンシルの 3 階微分行列の固有値は純虚数で、絶対値は最大で
    $2.6/h^3\simeq3.5\times10^2$ に達す。ソルバーはこれに合わせて
    刻みを $10^{-3}$ 程度まで詰め、$t\in[0,10]$ の積分に約 3100 ステップを要す。

    結果は再利用できるよう `.npz` にも保存する
    （`.npz` は Git の追跡対象外である）。
    """)
    return


@app.cell
def _(artifacts_dir, make_differential_ops, np, solve_ivp):
    nx_kdv = 512
    length_kdv = 100.0
    x_kdv_grid = np.linspace(0, length_kdv, nx_kdv, endpoint=False)
    dx_kdv = x_kdv_grid[1] - x_kdv_grid[0]
    u0_kdv = np.sin(x_kdv_grid * (2.0 * np.pi / length_kdv))

    op_df1, _, op_df3 = make_differential_ops(nx_kdv, dx_kdv)

    t_max_kdv = 10.0
    t_kdv_grid = np.linspace(0, t_max_kdv, 161)
    _sol = solve_ivp(
        f_kdv,
        (0, t_max_kdv),
        u0_kdv,
        args=(op_df1, op_df3),
        method="RK45",
        t_eval=t_kdv_grid,
        rtol=1e-8,
        atol=1e-10,
    )
    if not _sol.success:
        raise RuntimeError(_sol.message)
    u_tx_kdv_raw = _sol.y.T

    kdv_dataset_path = artifacts_dir / "kdv_solve_ivp.npz"
    np.savez(kdv_dataset_path, x=x_kdv_grid, t=t_kdv_grid, u_tx=u_tx_kdv_raw)

    (
        f"success={_sol.success}, 右辺評価 {_sol.nfev} 回, dx={dx_kdv:.3f}",
        f"u の範囲 [{u_tx_kdv_raw.min():.2f}, {u_tx_kdv_raw.max():.2f}]",
        f"保存先 {kdv_dataset_path.name} ({kdv_dataset_path.stat().st_size / 1024:.0f} KiB)",
    )
    return (kdv_dataset_path,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### アニメーション

    保存した `.npz` を読み直してから描画する。こうしておくと、
    積分をやり直さずに図だけを作り直せる。

    161 コマを 1 コマ 50 ms で再生するので、再生時間は約 8.1 秒である。
    """)
    return


@app.cell
def _(artifacts_dir, kdv_dataset_path, np, plt, save_gif):
    with np.load(kdv_dataset_path) as _npz:
        x_kdv = _npz["x"]
        t_kdv = _npz["t"]
        u_tx_kdv = _npz["u_tx"]

    _fig, _ax = plt.subplots(figsize=(6.4, 3.6), dpi=100)
    _ax.set_xlim(0, 100)
    _ax.set_ylim(-1.5, 3.2)
    _ax.set_xlabel("$x$")
    _ax.set_ylabel("$u(x)$")
    _ax.axhline(0, lw=0.5, color="0.7")
    _line, = _ax.plot([], [], lw=1.2, color="tab:blue")
    _title = _ax.set_title("KdV, t = 0.00")
    _fig.tight_layout()

    def _update(frame):
        _line.set_data(x_kdv, u_tx_kdv[frame])
        _title.set_text(f"KdV, t = {t_kdv[frame]:.2f}")
        return (_line, _title)

    kdv_gif = artifacts_dir / "kdv_solve_ivp.gif"
    save_gif(_fig, _update, len(t_kdv), kdv_gif, interval=50)
    return (kdv_gif,)


@app.cell
def _(gif_image, kdv_gif):
    gif_image(kdv_gif, alt="KdV 方程式の時間発展")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    アニメーションは次の 3 つの局面に分かれる。

    1. $t\lesssim2.5$: 移流項 $-6uu_x$ により、正弦波の下り坂が
       バーガース方程式と同じように切り立つ。
    2. $t\simeq2.5$〜$4$: 勾配が大きくなると分散項 $-u_{xxx}$ が効き始め、
       切り立った面の後ろに振動列が生まれる。これが分散衝撃波（波打つ段波）で、
       バーガース方程式で見た「鈍った 1 枚の界面」との違いが最もはっきりする場面。
    3. $t\gtrsim5$: 振動列が個別のソリトンに分離する。振幅の大きいソリトンほど
       速く進むため、列は左ほど背が低く右ほど背が高い階段状に整列する。
       ソリトン $u=(c/2)\,\mathrm{sech}^2(\sqrt{c}(x-ct)/2)$ は振幅と速度が
       比例するという KdV 特有の関係を反映している。

    3 つの節を通して、同じ移流項 $uu_x$ が
    「拡散項と組めば衝撃波」「分散項と組めばソリトン」という
    まったく違う漸近形へ落ち着くことが確認できる。
    """)
    return


if __name__ == "__main__":
    app.run()
