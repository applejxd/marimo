import marimo

__generated_with = "0.24.0"
app = marimo.App()

with app.setup:
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # 曲線フィッティングと RANSAC

    観測点へ曲線を当てはめる問題を、次の 2 軸で整理して扱う。

    | | 陽関数 $y=f(x)$ | 陰関数 $f(x,y)=0$ |
    | --- | --- | --- |
    | 外れ値なし | 最小二乗法（`numpy.polyfit` / `scipy.optimize`） | 直交距離回帰（`scipy.odr`） |
    | 外れ値あり | RANSAC ＋ 多項式フィッティング | RANSAC ＋ 代数的当てはめ |

    陽関数は $y$ 方向の残差だけを測るため、縦に伸びた形や、1 つの $x$ に複数の $y$ が
    対応する曲線（円・楕円）を扱えない。陰関数にはその制約が無い代わりに、係数を
    定数倍しても同じ曲線を表すというスケールの不定性があり、$\|a\|=1$ のような
    正規化が必要になる。この違いが、後半の誤差共分散行列と `scipy.odr` の自明解の
    話に直結する。

    各節では、推定した値を生成に使った真値と突き合わせて確認する。
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## この notebook の共通部品

    冒頭の `with app.setup` ブロック（marimo 上では折りたたまれている）で
    `marimo` `matplotlib.pyplot` `numpy` を読み込んでいる。`scipy` は使う節でのみ
    読み込む。

    次のセルは、以降の節が共有するヘルパーを定義する。

    - `show_matrix(matrix, title)`: 行列をカラーバー付きのヒートマップで表示する。
      共分散行列や構造行列の確認に使う。
    - `rotation_matrix(degrees)`: 反時計回りの 2 次元回転行列を返す。
    - `describe_conic(params)`: 基底 $[x^2, 2xy, y^2, 2x, 2y, 1]$ に対する係数から、
      楕円の中心・半軸長・半軸の向き（度）を復元する。推定結果を真値と比べるために使う。
      楕円にならない係数のときは `None` を返す。
    - `implicit_grid(basis_func, points, params, step)`: 陰関数 $f(x,y)$ の値を格子上で
      評価し、`plt.contour` へ渡せる 3 つの格子を返す。
    - `iteration_budget(inlier_ratio, sample_size, target_prob)`: RANSAC の試行回数の
      目安を返す。式の意味は RANSAC の節で説明する。

    乱数は、データを作るセルごとに `np.random.default_rng(seed)` で生成器を作る。
    marimo はセルを依存グラフの順に、必要なものだけ再実行するため、`np.random.seed`
    による大域的な種の設定では実行のたびに結果が変わりうるからである。
    """)
    return


@app.cell
def _():
    def show_matrix(matrix, title):
        """行列をヒートマップで表示する。

        :param matrix: 表示する 2 次元配列
        :param title: 図のタイトル
        """
        figure, axes = plt.subplots(figsize=(4.2, 3.4))
        image = axes.imshow(matrix)
        axes.set_title(title)
        figure.colorbar(image, ax=axes)
        figure.tight_layout()
        plt.show()

    def rotation_matrix(degrees):
        """反時計回りの回転行列を返す。

        :param degrees: 回転角（度）
        :return: 形状 (2, 2) の回転行列
        """
        radians = np.deg2rad(degrees)
        return np.array(
            [[np.cos(radians), -np.sin(radians)], [np.sin(radians), np.cos(radians)]]
        )

    def describe_conic(params):
        """円錐曲線の係数から中心・半軸長・半軸の向きを復元する。

        :param params: 基底 [x^2, 2xy, y^2, 2x, 2y, 1] に対する係数 (6,)
        :return: (中心 (2,), 半軸長 (2,), 半軸の向き [度] (2,))。楕円でなければ None
        """
        coeff_a, coeff_b, coeff_c, coeff_d, coeff_e, coeff_f = params
        quadratic = np.array([[coeff_a, coeff_b], [coeff_b, coeff_c]])
        if np.isclose(np.linalg.det(quadratic), 0.0):
            return None
        center = np.linalg.solve(quadratic, -np.array([coeff_d, coeff_e]))
        constant = coeff_f + coeff_d * center[0] + coeff_e * center[1]
        values, vectors = np.linalg.eigh(quadratic)
        if np.any(-constant / values <= 0.0):
            return None
        semi_axes = np.sqrt(-constant / values)
        angles = np.degrees(np.arctan2(vectors[1], vectors[0]))
        # 半軸の向きは 180 度の周期を持つので (-90, 90] へ畳む。
        return center, semi_axes, (angles + 90.0) % 180.0 - 90.0

    def implicit_grid(basis_func, points, params, step=0.01):
        """陰関数の値を点群の外接矩形上の格子で評価する。

        :param basis_func: 基底関数
        :param points: 形状 (N, 2) の点群。格子の範囲を決めるために使う
        :param params: 基底に対する係数
        :param step: 格子の刻み幅
        :return: (x 格子, y 格子, 陰関数の値の格子)
        """
        x_range = np.arange(points[:, 0].min(), points[:, 0].max(), step)
        y_range = np.arange(points[:, 1].min(), points[:, 1].max(), step)
        x_mesh, y_mesh = np.meshgrid(x_range, y_range)
        bases = np.array(
            [basis * np.ones_like(x_mesh) for basis in basis_func((x_mesh, y_mesh))]
        )
        return x_mesh, y_mesh, np.einsum("i,ijk->jk", params, bases)

    def iteration_budget(inlier_ratio, sample_size, target_prob=0.9999, cap=100000):
        """RANSAC の試行回数の目安 N = ln(1-p) / ln(1-e^n) を返す。

        :param inlier_ratio: 1 点を引いたときに inlier である確率 e
        :param sample_size: 1 回の試行で引く点数 n
        :param target_prob: RANSAC を成功させたい確率 p
        :param cap: 試行回数の上限
        :return: 試行回数（1 以上 cap 以下の整数）。cap に達しない限り、目標確率を
            下回らないよう切り上げる。cap で打ち切った場合は目標確率を保証しない
        """
        failure_prob = 1.0 - inlier_ratio**sample_size
        if failure_prob <= 0.0:
            # 全点が inlier なら 1 回引けば十分。
            return 1
        if np.isclose(failure_prob, 1.0):
            # inlier 率が低すぎて式が発散する場合は上限で打ち切る。
            return cap
        exact = np.log(1.0 - target_prob) / np.log(failure_prob)
        return int(np.clip(np.ceil(exact), 1, cap))

    return (
        describe_conic,
        implicit_grid,
        iteration_budget,
        rotation_matrix,
        show_matrix,
    )


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 外れ値なしのフィッティング

    ### 陽関数の場合

    観測を $y_i = f(x_i; a) + \varepsilon_i$ と置き、係数 $a$ を残差二乗和の最小化で
    決める。

    $$
    \hat{a} = \arg\min_{a} \sum_{i=1}^{N} \bigl( y_i - f(x_i; a) \bigr)^2
    $$

    $f$ が係数について線形な多項式なら、計画行列 $X_{ij} = x_i^{\,m-j}$ を使って
    正規方程式 $X^\top X a = X^\top y$ を解くだけで閉じた形で求まる。

    検証用のデータは 2 次関数に正規分布の誤差を足して作る。真の係数は
    $(a, b, c) = (1, 0, 0)$、誤差の標準偏差は $0.2$ である。
    """)
    return


@app.cell
def _():
    quad_sample_size = 1000
    quad_true_coeffs = np.array([1.0, 0.0, 0.0])
    quad_noise_sd = 0.2
    _rng = np.random.default_rng(0)
    quad_x = np.linspace(0, 10, quad_sample_size)
    quad_y = np.polyval(quad_true_coeffs, quad_x) + _rng.normal(
        0.0, quad_noise_sd, quad_sample_size
    )
    quad_points = np.vstack([quad_x, quad_y]).T
    plt.scatter(quad_x, quad_y, s=0.5)
    plt.xlim(0, 3.5)
    plt.ylim(-1, 11)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"y = x^2 + N(0, {quad_noise_sd}^2), {quad_sample_size} points")
    plt.show()
    return quad_points, quad_sample_size, quad_true_coeffs, quad_x, quad_y


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    代表的な API は次の 3 つで、いずれも同じ最小二乗解へ到達するが、扱える
    モデルの範囲と共分散行列の正規化の規約が異なる。

    | 関数 | 当てはめられるモデル | 共分散行列 |
    | --- | --- | --- |
    | [`numpy.polyfit`](https://numpy.org/doc/stable/reference/generated/numpy.polyfit.html) | 多項式のみ | `cov=True` で $\chi^2/(N-m)$ 倍済みの値を返す |
    | [`scipy.optimize.curve_fit`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html) | $y=f(x, a)$ の形なら任意 | 既定（`absolute_sigma=False`）で同じ正規化 |
    | [`scipy.optimize.leastsq`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.leastsq.html) | 残差ベクトルを自分で書く | `cov_x` は正規化前の $(X^\top X)^{-1}$ |

    ここで $m$ は係数の数、$\chi^2$ は残差二乗和である。`leastsq` の `cov_x` だけは
    スケールが違うので、他と比べるには $\chi^2/(N-m)$ を掛ける必要がある。
    この 3 点は後のセルで実際に突き合わせて確認する。
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### numpy.polyfit を使う方法

    多項式に限れば最も簡便である。`deg` に次数、`cov=True` に指定すると係数と
    共分散行列を返す。共分散行列の対角成分が各係数の分散で、$x^2$ の係数ほど
    小さく、定数項ほど大きい。$x^2$ の項は $x$ が大きい領域で値が大きく効くため、
    同じ観測誤差でも係数が動く余地が小さいからである。
    """)
    return


@app.cell
def _(quad_true_coeffs, quad_x, quad_y, show_matrix):
    polyfit_coeffs, polyfit_cov = np.polyfit(quad_x, quad_y, deg=2, cov=True)
    print(f"推定係数 (a, b, c) = {np.round(polyfit_coeffs, 6)}")
    print(f"真の係数 (a, b, c) = {quad_true_coeffs}")
    print(f"係数の標準偏差     = {np.round(np.sqrt(np.diag(polyfit_cov)), 6)}")
    show_matrix(polyfit_cov, "numpy.polyfit covariance")
    return polyfit_coeffs, polyfit_cov


@app.cell
def _(polyfit_coeffs, quad_x, quad_y):
    plt.scatter(quad_x, quad_y, s=0.5, label="observation")
    plt.plot(quad_x, np.polyval(polyfit_coeffs, quad_x), color="orange", label="polyfit")
    plt.xlim(0, 3.5)
    plt.ylim(-1, 11)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.show()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### scipy.optimize.curve_fit を使う方法

    `numpy.polyfit` と違い、モデル関数 $f(x, a)$ を自分で書けるので多項式以外も
    当てはめられる。ここでは比較のために同じ 2 次関数を渡す。内部は非線形最小二乗
    （Levenberg-Marquardt 系）だが、モデルが係数について線形なので `polyfit` と
    同じ解へ収束する。
    """)
    return


@app.cell
def _(polyfit_coeffs, polyfit_cov, quad_x, quad_y, show_matrix):
    import scipy.optimize as spo

    def quad_model(x, coeff_a, coeff_b, coeff_c):
        """2 次関数 y = a x^2 + b x + c。curve_fit へ渡すモデル関数。"""
        return coeff_a * x**2 + coeff_b * x + coeff_c

    curve_fit_coeffs, curve_fit_cov = spo.curve_fit(quad_model, quad_x, quad_y)
    print(f"推定係数 = {np.round(curve_fit_coeffs, 6)}")
    print(f"polyfit との係数の最大差   = {np.max(np.abs(curve_fit_coeffs - polyfit_coeffs)):.3e}")
    print(f"polyfit との共分散の最大差 = {np.max(np.abs(curve_fit_cov - polyfit_cov)):.3e}")
    show_matrix(curve_fit_cov, "scipy.optimize.curve_fit covariance")
    return curve_fit_coeffs, quad_model, spo


@app.cell
def _(curve_fit_coeffs, quad_model, quad_x, quad_y):
    plt.scatter(quad_x, quad_y, s=0.5, label="observation")
    plt.plot(quad_x, quad_model(quad_x, *curve_fit_coeffs), color="orange", label="curve_fit")
    plt.xlim(0, 3.5)
    plt.ylim(-1, 11)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.show()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### scipy.optimize.leastsq を使う方法

    残差ベクトルそのものを渡す形なので、二乗和以外の重み付けや、$x$ と $y$ を
    対等に扱う残差も書ける。返り値は `full_output=True` のとき
    `(係数, cov_x, infodict, mesg, ier)` の 5 要素のタプルで、`cov_x` は正規化前の
    共分散である。

    次のセルでは `cov_x` に $\chi^2/(N-m)$ を掛け、`polyfit` の共分散と一致すること
    を確認する。
    """)
    return


@app.cell
def _(
    polyfit_cov,
    quad_model,
    quad_points,
    quad_sample_size,
    show_matrix,
    spo,
):
    def quad_residuals(coeffs, points):
        """2 次関数の y 方向残差ベクトルを返す。

        :param coeffs: 係数 (a, b, c)
        :param points: 形状 (N, 2) の観測点
        :return: 形状 (N,) の残差 f(x) - y
        """
        return quad_model(points[:, 0], *coeffs) - points[:, 1]

    _output = spo.leastsq(quad_residuals, (0.0, 0.0, 0.0), args=quad_points, full_output=True)
    leastsq_coeffs, leastsq_cov_x = _output[0], _output[1]
    leastsq_chi2 = float(np.sum(quad_residuals(leastsq_coeffs, quad_points) ** 2))
    leastsq_cov = leastsq_cov_x * leastsq_chi2 / (quad_sample_size - 3)
    print(f"推定係数 = {np.round(leastsq_coeffs, 6)}")
    print(f"chi^2 / (N - m) = {leastsq_chi2 / (quad_sample_size - 3):.6f}")
    print(f"正規化前の cov_x の対角 = {np.round(np.diag(leastsq_cov_x), 8)}")
    print(f"polyfit との共分散の最大差（正規化後） = {np.max(np.abs(leastsq_cov - polyfit_cov)):.3e}")
    show_matrix(leastsq_cov, "scipy.optimize.leastsq covariance (scaled)")
    return (leastsq_coeffs,)


@app.cell
def _(leastsq_coeffs, quad_model, quad_x, quad_y):
    plt.scatter(quad_x, quad_y, s=0.5, label="observation")
    plt.plot(quad_x, quad_model(quad_x, *leastsq_coeffs), color="orange", label="leastsq")
    plt.xlim(0, 3.5)
    plt.ylim(-1, 11)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.show()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### 陰関数の場合

    円のように 1 つの $x$ に 2 つの $y$ が対応する曲線は、$y=f(x)$ の形では表せない。
    $f(x,y)=0$ の形にしたうえで、点から曲線までの距離を残差とする。この定式化は
    直交距離回帰（ODR, Orthogonal Distance Regression）と呼ばれ、
    [`scipy.odr`](https://docs.scipy.org/doc/scipy/reference/odr.html) で扱える。

    検証用のデータは、中心 $(1, 2)$・半径 $1.3$ の円に、標準偏差 $0.02$ の正規分布
    誤差を $x$ と $y$ の両方へ足して作る。円は
    $$
    f(x, y) = (x - c_x)^2 + (y - c_y)^2 - r^2 = 0
    $$
    と書け、パラメータ $\beta=(c_x, c_y, r)$ について同次でない。この点が後半の
    円錐曲線との違いになる。
    """)
    return


@app.cell
def _():
    circle_sample_size = 500
    circle_true_center = np.array([1.0, 2.0])
    circle_true_radius = 1.3
    circle_noise_sd = 0.02
    _rng = np.random.default_rng(1)
    _theta = np.linspace(0, 2 * np.pi, circle_sample_size)
    _errors = circle_noise_sd * _rng.normal(0.0, 1.0, (circle_sample_size, 2))
    circle_points = (
        np.vstack(
            [
                circle_true_radius * np.cos(_theta) + circle_true_center[0],
                circle_true_radius * np.sin(_theta) + circle_true_center[1],
            ]
        ).T
        + _errors
    )
    plt.scatter(circle_points[:, 0], circle_points[:, 1], s=0.5)
    plt.axis("equal")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"circle: center={tuple(circle_true_center)}, r={circle_true_radius}")
    plt.show()
    return circle_points, circle_true_center, circle_true_radius


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    `odr.Model` に `implicit=True` を渡すと、モデル関数の値そのものを 0 へ近づける
    問題として解く。`odr.Data(points.T, y=1)` の `y=1` は「陰関数なので目標値の配列は
    無い」ことを示す約束事である。`beta0` は初期値で、ここでは原点中心・半径 1 から
    始める。
    """)
    return


@app.cell
def _(circle_points, circle_true_center, circle_true_radius, show_matrix):
    from scipy import odr

    def circle_residual(beta, point):
        """円の陰関数 (x-cx)^2 + (y-cy)^2 - r^2。

        :param beta: (中心 x, 中心 y, 半径)
        :param point: 形状 (2, N) の点群
        :return: 形状 (N,) の陰関数の値
        """
        return (point[0] - beta[0]) ** 2 + (point[1] - beta[1]) ** 2 - beta[2] ** 2

    _model = odr.Model(circle_residual, implicit=True)
    _data = odr.Data(circle_points.T, y=1)
    circle_odr_result = odr.ODR(_data, _model, beta0=[0.0, 0.0, 1.0]).run()
    circle_beta = circle_odr_result.beta
    print(f"推定 中心 = {np.round(circle_beta[:2], 5)} / 真値 = {circle_true_center}")
    print(f"推定 半径 = {abs(circle_beta[2]):.5f} / 真値 = {circle_true_radius}")
    print(f"中心の誤差 = {np.linalg.norm(circle_beta[:2] - circle_true_center):.5f}")
    print(f"停止理由 = {circle_odr_result.stopreason}")
    show_matrix(circle_odr_result.cov_beta, "scipy.odr cov_beta (unscaled)")
    return circle_beta, circle_residual, odr


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    推定した陰関数を格子上で評価してヒートマップにすると、値が 0 になる部分空間が
    推定された曲線になる。`imshow` は行列の行方向を上から下へ描くので、散布図と
    向きを揃えるには `origin="lower"` と `extent` の指定が要る。
    """)
    return


@app.cell
def _(circle_beta, circle_points, circle_residual):
    _step = 0.01
    _x_range = np.arange(circle_points[:, 0].min(), circle_points[:, 0].max(), _step)
    _y_range = np.arange(circle_points[:, 1].min(), circle_points[:, 1].max(), _step)
    circle_x_mesh, circle_y_mesh = np.meshgrid(_x_range, _y_range)
    circle_z_mesh = circle_residual(circle_beta, (circle_x_mesh, circle_y_mesh))
    _extent = (_x_range[0], _x_range[-1], _y_range[0], _y_range[-1])
    plt.imshow(circle_z_mesh, origin="lower", extent=_extent)
    plt.colorbar(label="f(x, y)")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("implicit function value")
    plt.show()
    return circle_x_mesh, circle_y_mesh, circle_z_mesh


@app.cell
def _(circle_points, circle_x_mesh, circle_y_mesh, circle_z_mesh):
    plt.scatter(circle_points[:, 0], circle_points[:, 1], s=0.5, color="blue", label="observation")
    plt.contour(circle_x_mesh, circle_y_mesh, circle_z_mesh, [0], colors="orange")
    plt.axis("equal")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.show()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## RANSAC

    最小二乗法は、全観測が同じ誤差分布に従うことを前提にしている。外れ値が
    混ざると、二乗和の最小化はその 1 点を減らすために解を大きく引きずられる。

    RANSAC (RANdom SAmple Consensus) は次を繰り返す。

    1. モデルを一意に決めるのに必要な最小個数の点を無作為に選ぶ
    2. その点だけでモデルを当てはめる
    3. 全点との距離を測り、閾値以内の点（inlier）を数える
    4. inlier が最多だった候補を採用し、最後にその inlier だけで再フィットする

    ### 試行回数の見積もり

    1 点を引いたときそれが inlier である確率を $e$、1 回の試行で引く点数を $n$ と
    すると、1 回の試行で全点が inlier である確率は $e^n$ である。よって $N$ 回の
    試行でそれが一度も起こらない確率、すなわち RANSAC が失敗する確率は
    $(1-e^n)^N$ になる。成功確率を $p$ にしたければ
    $$
    1 - p = (1 - e^n)^N
    \quad\Longleftrightarrow\quad
    N = \frac{\ln(1-p)}{\ln(1-e^n)}
    $$
    だけ繰り返せばよい。$e$ は事前には分からないので、散布図を見た大まかな値から
    始め、inlier 数の最大値を更新するたびに $e \simeq$ (inlier 数)/(全点数) として
    $N$ を計算し直すと早期に打ち切れる。共通部品の `iteration_budget` がこの式に
    あたる。$N$ は試行回数なので、右辺を切り上げて整数にする。切り捨てると
    成功確率が目標を下回る。なお $e$ が小さいと $N$ は急激に増えるため、
    `iteration_budget` は上限（既定 10 万回）で打ち切る。上限に達した場合は
    目標の成功確率を保証しない。
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### RANSAC（陽関数）

    #### 直線フィッティング

    外れ値を含む状況を作るため、誤差分布として裾の重いコーシー分布を使う。
    確率密度は
    $$
    p(x) = \frac{1}{\pi (1 + x^2)}
    $$
    で、平均も分散も定義されない。$|x|$ が大きい側の密度が正規分布より桁違いに
    大きいため、標本には明確な外れ値が混ざる。
    """)
    return


@app.cell
def _():
    def cauchy_pdf(x):
        """標準コーシー分布の確率密度。"""
        return 1.0 / np.pi / (1.0 + x * x)

    _x = np.linspace(-5, 5, 100)
    plt.plot(_x, cauchy_pdf(_x), label="Cauchy")
    plt.plot(_x, np.exp(-(_x**2) / 2) / np.sqrt(2 * np.pi), label="Normal", linestyle="--")
    plt.xlabel("x")
    plt.ylabel("density")
    plt.legend()
    plt.show()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    真の直線 $y = x$ に、コーシー分布に従う誤差を $1/10$ に縮めて足したものを
    観測とする。
    """)
    return


@app.cell
def _():
    line_sample_size = 1000
    line_true_coeffs = np.array([1.0, 0.0])
    _rng = np.random.default_rng(2)
    line_x = np.linspace(0, 10, line_sample_size)
    line_y = np.polyval(line_true_coeffs, line_x) + _rng.standard_cauchy(line_sample_size) / 10
    line_points = np.vstack([line_x, line_y]).T
    plt.scatter(line_x, line_y, s=0.5)
    plt.ylim(-1, 11)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("y = x + Cauchy/10")
    plt.show()
    return line_points, line_sample_size, line_true_coeffs


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    inlier である確率 $e$ は本来は事前に分からない。ここでは散布図の見た目から
    $e=0.9$ と仮に置く。直線のパラメータは 2 個なので、1 回の試行で引く点数は
    2 点でよい。成功確率は 99.99% を狙う。この見積もりが妥当だったかは、
    後で実際の inlier 率と突き合わせて確認する。
    """)
    return


@app.cell
def _(iteration_budget):
    line_inlier_ratio = 0.9
    line_sample_points = 2
    line_budget = iteration_budget(line_inlier_ratio, line_sample_points)
    print(f"e={line_inlier_ratio}, n={line_sample_points} のときの試行回数 N = {line_budget}")
    return line_budget, line_inlier_ratio, line_sample_points


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    試行ごとに 2 点を重複なく選び、形状 (試行数, 点数, 座標) のテンソルにまとめる。
    こうすると以降の処理を試行方向へまとめて書ける。
    """)
    return


@app.cell
def _(line_budget, line_points, line_sample_points, line_sample_size):
    _rng = np.random.default_rng(3)
    line_index_mat = np.array(
        [
            _rng.choice(line_sample_size, size=line_sample_points, replace=False)
            for _ in range(line_budget)
        ]
    )
    line_samples = line_points[line_index_mat]
    print(f"インデックス行列の形状 = {line_index_mat.shape}")
    print(f"サンプルの形状 (試行数, 点数, 座標) = {line_samples.shape}")
    return (line_samples,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    各試行のサンプルへ `numpy.polyfit` で 1 次式を当てはめ、候補の係数を作る。
    """)
    return


@app.cell
def _(line_samples):
    line_candidates = np.array(
        [np.polyfit(sample[:, 0], sample[:, 1], 1) for sample in line_samples]
    )
    print(f"候補の係数の形状 (試行数, 係数) = {line_candidates.shape}")
    print(f"候補の係数 (傾き, 切片) =\n{np.round(line_candidates, 4)}")
    return (line_candidates,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    候補ごとに全点の $y$ 方向残差を測り、閾値 $0.3$ 以内の点を inlier とする。
    inlier が最も多い候補を採用する。
    """)
    return


@app.cell
def _(
    iteration_budget,
    line_budget,
    line_candidates,
    line_inlier_ratio,
    line_points,
    line_sample_points,
):
    line_distance_th = 0.3
    _predicted = np.array([np.polyval(coeffs, line_points[:, 0]) for coeffs in line_candidates])
    _residuals = np.abs(line_points[:, 1][None, :] - _predicted)
    line_inlier_masks = _residuals < line_distance_th
    line_inlier_counts = line_inlier_masks.sum(axis=1)
    line_best_index = int(np.argmax(line_inlier_counts))
    print(f"各候補の inlier 数 = {line_inlier_counts}")
    print(f"最良の候補 = {line_best_index} 番目（inlier {line_inlier_counts[line_best_index]} 点）")
    _actual_ratio = line_inlier_counts[line_best_index] / len(line_points)
    print(f"想定した inlier 率 e = {line_inlier_ratio} / 実際の inlier 率 = {_actual_ratio:.3f}")
    print(f"実際の inlier 率なら必要な試行回数 N = "
          f"{iteration_budget(_actual_ratio, line_sample_points)}（実行したのは {line_budget} 回）")
    return line_best_index, line_inlier_masks


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    実際の inlier 率は見積もった $e=0.9$ より低い。$e$ を大きく見積もると $N$ は
    小さく出るので、そのぶん成功確率は目標の 99.99% を下回る。実務では、
    inlier 数の最大値を更新するたびに $e$ を測り直して $N$ を計算し直す。
    後半の `ImplicitRansac.execute` はその形で実装してある。
    """)
    return


@app.cell
def _(line_best_index, line_candidates, line_inlier_masks, line_points):
    line_best_coeffs = line_candidates[line_best_index]
    line_best_mask = line_inlier_masks[line_best_index]
    plt.scatter(
        line_points[~line_best_mask, 0], line_points[~line_best_mask, 1], s=0.5, color="blue", label="outlier"
    )
    plt.scatter(
        line_points[line_best_mask, 0], line_points[line_best_mask, 1], s=0.5, color="orange", label="inlier"
    )
    plt.plot(
        line_points[:, 0], np.polyval(line_best_coeffs, line_points[:, 0]), color="green", label="best sample"
    )
    plt.ylim(-1, 11)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.show()
    return line_best_coeffs, line_best_mask


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    採用した候補は 2 点だけで決めた直線なので、そのままでは inlier 全体の情報を
    使えていない。最後に inlier だけで最小二乗を解き直す。次の図では、2 点から
    決めた直線（緑）と再フィットした直線（赤）を重ねる。
    """)
    return


@app.cell
def _(line_best_coeffs, line_best_mask, line_points, line_true_coeffs):
    line_refined_coeffs = np.polyfit(line_points[line_best_mask, 0], line_points[line_best_mask, 1], 1)
    print(f"真の係数     (傾き, 切片) = {line_true_coeffs}")
    print(f"2 点での係数 (傾き, 切片) = {np.round(line_best_coeffs, 5)}")
    print(f"再フィット後 (傾き, 切片) = {np.round(line_refined_coeffs, 5)}")
    print(f"真値との差 : {np.abs(line_best_coeffs - line_true_coeffs).max():.5f} -> "
          f"{np.abs(line_refined_coeffs - line_true_coeffs).max():.5f}")
    plt.scatter(
        line_points[~line_best_mask, 0], line_points[~line_best_mask, 1], s=0.5, color="blue", label="outlier"
    )
    plt.scatter(
        line_points[line_best_mask, 0], line_points[line_best_mask, 1], s=0.5, color="orange", label="inlier"
    )
    plt.plot(
        line_points[:, 0], np.polyval(line_best_coeffs, line_points[:, 0]), color="green", label="best sample"
    )
    plt.plot(
        line_points[:, 0], np.polyval(line_refined_coeffs, line_points[:, 0]), color="red", label="refit on inliers"
    )
    plt.ylim(-1, 11)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.show()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### 2 次曲線フィッティング

    ここまでの手順は次数を変えるだけでそのまま使えるので、関数にまとめる。
    """)
    return


@app.cell
def _(iteration_budget):
    def ransac_polyfit(points, degree, distance_th, inlier_ratio, seed=0, target_prob=0.9999):
        """多項式に対する RANSAC。

        :param points: 形状 (N, 2) の観測点
        :param degree: 多項式の次数。1 回の試行では degree+1 点を引く
        :param distance_th: inlier と見なす y 方向残差の閾値
        :param inlier_ratio: inlier の割合の見積もり。試行回数の算出に使う
        :param seed: 乱数の種
        :param target_prob: RANSAC を成功させたい確率
        :return: (最良候補の係数, inlier マスク, inlier で再フィットした係数, 試行回数)
        """
        rng = np.random.default_rng(seed)
        sample_points = degree + 1
        budget = iteration_budget(inlier_ratio, sample_points, target_prob)
        samples = points[
            np.array(
                [rng.choice(len(points), size=sample_points, replace=False) for _ in range(budget)]
            )
        ]
        candidates = np.array(
            [np.polyfit(sample[:, 0], sample[:, 1], degree) for sample in samples]
        )
        predicted = np.array([np.polyval(coeffs, points[:, 0]) for coeffs in candidates])
        masks = np.abs(points[:, 1][None, :] - predicted) < distance_th
        best = int(np.argmax(masks.sum(axis=1)))
        refined = np.polyfit(points[masks[best], 0], points[masks[best], 1], degree)
        return candidates[best], masks[best], refined, budget

    return (ransac_polyfit,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    真の曲線を $y=x^2$ に変え、同じくコーシー分布の誤差を足す。2 次曲線は
    パラメータが 3 個なので 1 回の試行で 3 点を引く。$e=0.9$ のままだと
    $e^n$ が小さくなって試行回数が増えるため、ここでは控えめに $e=0.6$ と
    見積もる。
    """)
    return


@app.cell
def _():
    parabola_sample_size = 1000
    parabola_true_coeffs = np.array([1.0, 0.0, 0.0])
    _rng = np.random.default_rng(4)
    parabola_x = np.linspace(0, 10, parabola_sample_size)
    parabola_y = (
        np.polyval(parabola_true_coeffs, parabola_x)
        + _rng.standard_cauchy(parabola_sample_size) / 10
    )
    parabola_points = np.vstack([parabola_x, parabola_y]).T
    plt.scatter(parabola_x, parabola_y, s=0.5)
    plt.xlim(0, 3.5)
    plt.ylim(-1, 11)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("y = x^2 + Cauchy/10")
    plt.show()
    return parabola_points, parabola_true_coeffs


@app.cell
def _(parabola_points, parabola_true_coeffs, ransac_polyfit):
    parabola_best, parabola_mask, parabola_refined, parabola_budget = ransac_polyfit(
        parabola_points, degree=2, distance_th=0.3, inlier_ratio=0.6, seed=5
    )
    print(f"試行回数 = {parabola_budget}, inlier = {int(parabola_mask.sum())} / {len(parabola_points)}")
    print(f"真の係数     = {parabola_true_coeffs}")
    print(f"3 点での係数 = {np.round(parabola_best, 5)}")
    print(f"再フィット後 = {np.round(parabola_refined, 5)}")
    print(f"真値との差 : {np.abs(parabola_best - parabola_true_coeffs).max():.5f} -> "
          f"{np.abs(parabola_refined - parabola_true_coeffs).max():.5f}")
    return parabola_best, parabola_mask, parabola_refined


@app.cell
def _(parabola_best, parabola_mask, parabola_points, parabola_refined):
    plt.scatter(
        parabola_points[~parabola_mask, 0], parabola_points[~parabola_mask, 1], s=0.5, color="blue", label="outlier"
    )
    plt.scatter(
        parabola_points[parabola_mask, 0], parabola_points[parabola_mask, 1], s=0.5, color="orange", label="inlier"
    )
    plt.plot(
        parabola_points[:, 0], np.polyval(parabola_best, parabola_points[:, 0]), color="green", label="best sample"
    )
    plt.plot(
        parabola_points[:, 0], np.polyval(parabola_refined, parabola_points[:, 0]), color="red", label="refit on inliers"
    )
    plt.xlim(0, 3.5)
    plt.ylim(-1, 11)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.show()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### RANSAC（陰関数）

    #### 基底関数と構造行列

    陰関数を基底 $\xi_k(x,y)$ の線形結合で書く。

    $$
    f(x, y) = \sum_{k=1}^{m} a_k \xi_k(x, y) = (\xi, a) = 0
    $$

    直線なら $\xi = (x,\ y,\ 1)$ で $m=3$、円錐曲線なら
    $\xi = (x^2,\ 2xy,\ y^2,\ 2x,\ 2y,\ 1)$ で $m=6$ である。交差項と 1 次項に $2$ を
    付けておくと、$2$ 次の項の係数がそのまま対称行列の成分になり、後で中心や半軸長を
    求めるときに扱いやすい。

    係数を定数倍しても同じ曲線を表すので、$\|a\|=1$ と正規化して最小二乗を解く。

    $$
    \hat{a} = \arg\min_{\|a\|=1} \frac{1}{N} \sum_{i=1}^{N} (\xi_i, a)^2
    = \arg\min_{\|a\|=1} (a, M a),
    \qquad
    M = \frac{1}{N} \sum_{i=1}^{N} \xi_i \xi_i^\top
    $$

    右辺はレイリー商なので、解は $M$ の最小固有値に対応する固有ベクトルである。
    この $M$ を構造行列（モーメント行列）と呼ぶ。点数で割っても固有ベクトルは
    変わらないが、点数に依らないスケールにしておくと後の共分散の式と揃う。
    $M$ は対称なので `numpy.linalg.eigh` を使えば実固有値が昇順で得られる。

    観測に誤差が無ければ全点が厳密に $(\xi_i, a)=0$ を満たすため、$M$ は $a$ 方向を
    零空間に持つ特異行列（$\det M = 0$、半正定値）になる。誤差があると最小固有値が
    正へ持ち上がり、正定値行列になる。
    """)
    return


@app.cell
def _():
    def linear_basis(point):
        """直線 ax + by + c = 0 の基底 [x, y, 1]。

        :param point: (x, y)。x と y は配列でもよい
        :return: 基底の値のリスト。定数項は x*0+1 として形状を揃える
        """
        x, y = point
        return [x, y, x * 0 + 1]

    def quad_basis(point):
        """円錐曲線の基底 [x^2, 2xy, y^2, 2x, 2y, 1]。

        :param point: (x, y)。x と y は配列でもよい
        :return: 基底の値のリスト
        """
        x, y = point
        return [x**2, 2 * x * y, y**2, 2 * x, 2 * y, x * 0 + 1]

    return linear_basis, quad_basis


@app.cell
def _():
    ring_sample_size = 500
    ring_noise_sd = 0.02
    _theta = np.linspace(0, 2 * np.pi, ring_sample_size)
    ring_clean_points = np.vstack([np.cos(_theta), np.sin(_theta)]).T
    _rng = np.random.default_rng(7)
    ring_points = ring_clean_points + ring_noise_sd * _rng.normal(
        0.0, 1.0, (ring_sample_size, 2)
    )
    plt.scatter(ring_points[:, 0], ring_points[:, 1], s=0.5)
    plt.axis("equal")
    plt.xlim(-2, 2)
    plt.ylim(-2, 2)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("unit circle + normal noise")
    plt.show()
    return ring_clean_points, ring_points, ring_sample_size


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    構造行列は、1 点ごとの $\xi\xi^\top$ を全点で平均したものである。
    まず 1 点分を見ると、ランク 1 の対称行列になっている。
    """)
    return


@app.cell
def _(quad_basis, ring_points, show_matrix):
    _xi = quad_basis(ring_points[20])
    _single = np.outer(_xi, _xi)
    print(f"xi = {np.round(_xi, 4)}")
    print(f"rank(xi xi^T) = {np.linalg.matrix_rank(_single)}")
    show_matrix(_single, "xi xi^T for a single point")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    全点で平均を取って構造行列を作り、その最小固有ベクトルを係数とする。
    誤差なしの点群と誤差ありの点群で行列式を比べると、上で述べた特異性の違いが
    数値として見える。
    """)
    return


@app.cell
def _(
    quad_basis,
    ring_clean_points,
    ring_points,
    ring_sample_size,
    show_matrix,
):
    ring_design = np.array([quad_basis(point) for point in ring_points])
    ring_moment = ring_design.T @ ring_design / ring_sample_size
    _clean_design = np.array([quad_basis(point) for point in ring_clean_points])
    _clean_moment = _clean_design.T @ _clean_design / ring_sample_size
    print(f"det(M) 誤差なし = {np.linalg.det(_clean_moment):.3e}")
    print(f"det(M) 誤差あり = {np.linalg.det(ring_moment):.3e}")
    try:
        np.linalg.cholesky(ring_moment)
    except np.linalg.LinAlgError as error:
        print(f"誤差ありの M は正定値ではない: {error}")
    else:
        print("誤差ありの M はコレスキー分解でき、正定値である")
    ring_eigenvalues, _eigenvectors = np.linalg.eigh(ring_moment)
    ring_params = _eigenvectors[:, 0]
    if ring_params[np.argmax(np.abs(ring_params))] < 0:
        ring_params = -ring_params
    _truth = np.array([1.0, 0.0, 1.0, 0.0, 0.0, -1.0]) / np.sqrt(3.0)
    print(f"固有値 = {np.round(ring_eigenvalues, 6)}")
    print(f"推定した a = {np.round(ring_params, 5)}")
    print(f"真の単位円 x^2 + y^2 - 1 = 0 に対応する a = {np.round(_truth, 5)}")
    print(f"最大差 = {np.max(np.abs(ring_params - _truth)):.5f}")
    show_matrix(ring_moment, "structure matrix M")
    return (ring_params,)


@app.cell
def _(implicit_grid, quad_basis, ring_params, ring_points):
    ring_x_mesh, ring_y_mesh, ring_z_mesh = implicit_grid(quad_basis, ring_points, ring_params)
    _extent = (ring_x_mesh[0, 0], ring_x_mesh[0, -1], ring_y_mesh[0, 0], ring_y_mesh[-1, 0])
    plt.imshow(ring_z_mesh, origin="lower", extent=_extent)
    plt.colorbar(label="f(x, y)")
    plt.scatter(ring_points[:, 0], ring_points[:, 1], s=0.5, color="blue")
    plt.contour(ring_x_mesh, ring_y_mesh, ring_z_mesh, [0], colors="orange")
    plt.axis("equal")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("fitted implicit function")
    plt.show()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### 幾何距離の近似と自動微分

    RANSAC で inlier を数えるには、点と曲線の距離が要る。しかし代数的な残差
    $(\xi, a)$ は距離ではない。曲線の勾配が急な場所では、同じ $(\xi, a)$ でも実際の
    距離は小さいからである。1 次のテイラー展開で幾何距離の 2 乗を近似したものが
    Sampson 誤差である。

    $$
    E(x) = \frac{(\xi(x), a)^2}{(a,\ V_0[\xi(x)]\ a)},
    \qquad
    V_0[\xi] = J J^\top,
    \qquad
    J_{k\ell} = \frac{\partial \xi_k}{\partial x_\ell}
    $$

    $V_0[\xi]$ は基底のヤコビアンから決まる正規化共分散行列で、分母がちょうど勾配に
    よる補正にあたる。ヤコビアンは基底ごとに手で微分してもよいが、基底を差し替える
    たびに書き直すことになる。そこで、値と微分係数を同時に持つ数（二重数, Jet）を
    定義し、基底関数へそのまま流し込む前進モード自動微分で求める。

    $$
    (a + b\varepsilon)(c + d\varepsilon) = ac + (ad + bc)\varepsilon,
    \qquad \varepsilon^2 = 0
    $$

    この規則を演算子オーバーロードで実装すれば、`quad_basis` を書き換えずに
    ヤコビアンが得られる。
    """)
    return


@app.class_definition
class Jet:
    """値 a と微分係数ベクトル v を同時に運ぶ二重数。

    Jet(x, (1, 0)) と Jet(y, (0, 1)) を基底関数へ渡すと、返ってきた Jet の
    v が (∂ξ/∂x, ∂ξ/∂y) になる。
    """

    def __init__(self, a, v):
        self.a = a
        self.v = np.asarray(v, dtype=float)

    def __str__(self):
        return f"{self.a}+{self.v}eps"

    def _coerce(self, other):
        """定数を微分係数 0 の Jet へ揃える。"""
        return other if isinstance(other, Jet) else Jet(float(other), np.zeros_like(self.v))

    def __add__(self, other):
        other = self._coerce(other)
        return Jet(self.a + other.a, self.v + other.v)

    def __sub__(self, other):
        other = self._coerce(other)
        return Jet(self.a - other.a, self.v - other.v)

    def __mul__(self, other):
        other = self._coerce(other)
        return Jet(self.a * other.a, self.a * other.v + self.v * other.a)

    def __truediv__(self, other):
        other = self._coerce(other)
        return Jet(self.a / other.a, self.v / other.a - self.a * other.v / other.a**2)

    def __pow__(self, other):
        other = self._coerce(other)
        value = self.a**other.a
        base_diff = other.a * self.a ** (other.a - 1) * self.v
        # 底が 0 のとき log が発散するので、指数側の微分は落とす。
        exp_diff = 0.0 if np.isclose(self.a, 0.0) else value * other.v * np.log(np.abs(self.a))
        return Jet(value, base_diff + exp_diff)

    def __radd__(self, other):
        return self.__add__(other)

    def __rsub__(self, other):
        # 求めたいのは other - self であって self - other ではない。
        return self._coerce(other).__sub__(self)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __rtruediv__(self, other):
        return self._coerce(other).__truediv__(self)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    定義した `Jet` を使ってヤコビアンと正規化共分散行列を組み立てる。
    円錐曲線の基底なら解析的なヤコビアンも書けるので、両者が一致することを
    確認しておく。あわせて、定数を左に置いた減算（`__rsub__`）が符号を取り違えて
    いないことも確かめる。
    """)
    return


@app.cell
def _(quad_basis):
    def jet_jacobian(basis_func, point):
        """基底関数のヤコビアン ∂ξ_k/∂x_l を自動微分で求める。

        :param basis_func: 基底関数
        :param point: 形状 (2,) の点
        :return: 形状 (基底の数, 2) のヤコビアン
        """
        jets = [Jet(point[index], np.eye(len(point))[index]) for index in range(len(point))]
        outputs = basis_func(jets)
        return np.array(
            [
                (output if isinstance(output, Jet) else Jet(float(output), np.zeros(len(point)))).v
                for output in outputs
            ]
        )

    def normalized_cov(basis_func, point):
        """正規化共分散行列 V0[ξ] = J J^T。

        :param basis_func: 基底関数
        :param point: 形状 (2,) の点
        :return: 形状 (基底の数, 基底の数) の対称行列
        """
        jacobian = jet_jacobian(basis_func, point)
        return jacobian @ jacobian.T

    _point = np.array([1.7, -0.4])
    _x, _y = _point
    _analytic = np.array(
        [[2 * _x, 0.0], [2 * _y, 2 * _x], [0.0, 2 * _y], [2.0, 0.0], [0.0, 2.0], [0.0, 0.0]]
    )
    _diff = np.max(np.abs(jet_jacobian(quad_basis, _point) - _analytic))
    print(f"自動微分と解析解のヤコビアンの最大差 = {_diff:.3e}")
    print(f"5 - Jet(2, (1, 0)) = {5 - Jet(2.0, (1.0, 0.0))}")
    return (normalized_cov,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### RANSAC クラス

    前節までの部品をまとめる。基底関数を差し替えるだけで直線にも円錐曲線にも使える。

    - `fit(points)`: 構造行列の最小固有ベクトルを返す。固有ベクトルの符号は不定なので、
      絶対値が最大の成分が正になるよう揃える。第 2 返り値は残差二乗和
      $\sum_i (\xi_i, a)^2$ である。
    - `sampson_errors(design, cov_stack, params)`: 全点の Sampson 誤差をまとめて計算する。
      $V_0[\xi]$ は点だけで決まり係数に依存しないので、`execute` の冒頭で一度だけ
      作って使い回す。これで 1 試行あたりの計算が numpy の行列演算だけで済む。
    - `execute(points, distance_th)`: 試行を繰り返して最良の inlier 集合を選び、最後に
      inlier だけで再フィットする。inlier 数の最大値を更新するたびに試行回数の上限を
      計算し直して早期に打ち切る。
    """)
    return


@app.cell
def _(iteration_budget, normalized_cov):
    class ImplicitRansac:
        """陰関数 (ξ, a) = 0 を RANSAC で当てはめる。"""

        def __init__(self, basis_func, target_prob=0.9999, seed=0):
            """
            :param basis_func: 基底関数
            :param target_prob: RANSAC を成功させたい確率
            :param seed: 乱数の種
            """
            self.basis_func = basis_func
            self.target_prob = target_prob
            self.seed = seed

        def fit(self, points):
            """点群へ代数的最小二乗を当てはめる。

            :param points: 形状 (N, 2) の点群
            :return: (単位ベクトルへ正規化した係数, 残差二乗和)
            """
            design = np.array([self.basis_func(point) for point in points])
            moment = design.T @ design / len(design)
            values, vectors = np.linalg.eigh(moment)
            params = vectors[:, 0]
            if params[np.argmax(np.abs(params))] < 0:
                params = -params
            return params, values[0] * len(design)

        @staticmethod
        def sampson_errors(design, cov_stack, params):
            """全点の Sampson 誤差（幾何距離の 2 乗の近似）を返す。

            :param design: 形状 (N, m) の基底の値
            :param cov_stack: 形状 (N, m, m) の正規化共分散行列
            :param params: 形状 (m,) の係数
            :return: 形状 (N,) の Sampson 誤差
            """
            numerator = (design @ params) ** 2
            denominator = np.einsum("i,nij,j->n", params, cov_stack, params)
            with np.errstate(divide="ignore", invalid="ignore"):
                errors = numerator / denominator
            return np.where(np.isfinite(errors), errors, np.inf)

        def execute(self, points, distance_th, inlier_ratio=0.1):
            """RANSAC を実行する。

            :param points: 形状 (N, 2) の点群
            :param distance_th: inlier と見なす幾何距離の閾値
            :param inlier_ratio: inlier 率の初期見積もり
            :return: (係数, inlier マスク, inlier での残差二乗和, 試行回数)
            """
            rng = np.random.default_rng(self.seed)
            design = np.array([self.basis_func(point) for point in points])
            cov_stack = np.array([normalized_cov(self.basis_func, point) for point in points])
            sample_points = design.shape[1] - 1
            budget = iteration_budget(inlier_ratio, sample_points, self.target_prob)
            best_mask = np.zeros(len(points), dtype=bool)
            iterations = 0
            while iterations < budget:
                index = rng.choice(len(points), size=sample_points, replace=False)
                params, _ = self.fit(points[index, :])
                mask = self.sampson_errors(design, cov_stack, params) < distance_th**2
                iterations = iterations + 1
                if mask.sum() > best_mask.sum():
                    best_mask = mask
                    budget = min(
                        budget, iteration_budget(mask.mean(), sample_points, self.target_prob)
                    )
            best_params, residual = self.fit(points[best_mask, :])
            return best_params, best_mask, residual, iterations

    return (ImplicitRansac,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### 直線での動作確認

    実装の確認のため、まず陽関数の節と同じコーシー誤差付きの直線データへ適用する。
    直線の基底は 3 個なので、1 回の試行で引く点は 2 点である。
    """)
    return


@app.cell
def _(ImplicitRansac, linear_basis):
    implicit_line_sample_size = 1000
    _rng = np.random.default_rng(8)
    _x = np.linspace(0, 10, implicit_line_sample_size)
    _y = _x + _rng.standard_cauchy(implicit_line_sample_size) / 10
    implicit_line_points = np.vstack([_x, _y]).T
    implicit_line_params, implicit_line_mask, _residual, _iterations = ImplicitRansac(
        linear_basis, seed=9
    ).execute(implicit_line_points, distance_th=0.3)
    _slope = -implicit_line_params[0] / implicit_line_params[1]
    _intercept = -implicit_line_params[2] / implicit_line_params[1]
    print(f"試行回数 = {_iterations}")
    print(f"inlier = {int(implicit_line_mask.sum())} / {implicit_line_sample_size}")
    print(f"係数 a = {np.round(implicit_line_params, 5)}")
    print(f"y = ax + b の形へ直すと (傾き, 切片) = ({_slope:.5f}, {_intercept:.5f})")
    print("真の直線は y = x なので (傾き, 切片) = (1.0, 0.0)")
    return implicit_line_mask, implicit_line_params, implicit_line_points


@app.cell
def _(
    implicit_grid,
    implicit_line_mask,
    implicit_line_params,
    implicit_line_points,
    linear_basis,
):
    plt.scatter(
        implicit_line_points[~implicit_line_mask, 0],
        implicit_line_points[~implicit_line_mask, 1],
        s=0.5,
        color="blue",
        label="outlier",
    )
    plt.scatter(
        implicit_line_points[implicit_line_mask, 0],
        implicit_line_points[implicit_line_mask, 1],
        s=0.5,
        color="orange",
        label="inlier",
    )
    plt.contour(
        *implicit_grid(linear_basis, implicit_line_points, implicit_line_params),
        [0],
        colors="green",
    )
    plt.ylim(-1, 11)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.show()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### 楕円での動作確認

    陽関数では扱えない形の例として、傾いた楕円を当てはめる。データは半軸長
    $(2, 3)$ の楕円を $-30^\circ$ 回転し、中心を $(1, 2)$ へ平行移動して作る。
    誤差はコーシー分布を $1/100$ に縮めたものである。
    """)
    return


@app.cell
def _(rotation_matrix):
    ellipse_sample_size = 500
    ellipse_true_center = np.array([1.0, 2.0])
    ellipse_true_semi_axes = np.array([2.0, 3.0])
    ellipse_true_angle = -30.0
    _theta = np.linspace(0, 2 * np.pi, ellipse_sample_size)
    _base = np.vstack(
        [
            ellipse_true_semi_axes[0] * np.cos(_theta),
            ellipse_true_semi_axes[1] * np.sin(_theta),
        ]
    ).T
    ellipse_clean_points = _base @ rotation_matrix(ellipse_true_angle).T + ellipse_true_center
    _rng = np.random.default_rng(10)
    ellipse_points = ellipse_clean_points + 0.01 * _rng.standard_cauchy((ellipse_sample_size, 2))
    plt.scatter(ellipse_points[:, 0], ellipse_points[:, 1], s=0.5)
    plt.axis("equal")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("rotated ellipse + Cauchy noise")
    plt.show()
    return (
        ellipse_clean_points,
        ellipse_points,
        ellipse_sample_size,
        ellipse_true_angle,
        ellipse_true_center,
        ellipse_true_semi_axes,
    )


@app.cell
def _(
    ImplicitRansac,
    describe_conic,
    ellipse_points,
    ellipse_sample_size,
    ellipse_true_angle,
    ellipse_true_center,
    ellipse_true_semi_axes,
    quad_basis,
):
    ellipse_params, ellipse_mask, ellipse_residual, _iterations = ImplicitRansac(
        quad_basis, seed=11
    ).execute(ellipse_points, distance_th=0.07)
    _center, _semi_axes, _angles = describe_conic(ellipse_params)
    print(f"試行回数 = {_iterations}")
    print(f"inlier = {int(ellipse_mask.sum())} / {ellipse_sample_size}")
    print(f"係数 a = {np.round(ellipse_params, 5)}")
    print(f"推定 中心   = {np.round(_center, 4)} / 真値 = {ellipse_true_center}")
    print(f"推定 半軸長 = {np.round(_semi_axes, 4)} / 真値 = {ellipse_true_semi_axes}")
    print(f"推定 向き   = {np.round(_angles, 3)} 度 / 真値 = {ellipse_true_angle} 度とその直交方向")
    print(f"中心の誤差   = {np.linalg.norm(_center - ellipse_true_center):.5f}")
    print(f"半軸長の誤差 = {np.max(np.abs(_semi_axes - ellipse_true_semi_axes)):.5f}")
    return ellipse_mask, ellipse_params, ellipse_residual


@app.cell
def _(ellipse_mask, ellipse_params, ellipse_points, implicit_grid, quad_basis):
    plt.scatter(
        ellipse_points[~ellipse_mask, 0],
        ellipse_points[~ellipse_mask, 1],
        s=0.5,
        color="blue",
        label="outlier",
    )
    plt.scatter(
        ellipse_points[ellipse_mask, 0],
        ellipse_points[ellipse_mask, 1],
        s=0.5,
        color="orange",
        label="inlier",
    )
    plt.contour(*implicit_grid(quad_basis, ellipse_points, ellipse_params), [0], colors="green")
    plt.axis("equal")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.show()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### 推定した係数の誤差共分散行列

    $\|a\|=1$ と正規化しているので、$\hat{a}$ は単位球上を動く。誤差が小さい範囲で
    1 次まで見れば、その揺らぎは $\hat{a}$ に直交する $m-1$ 次元の接平面に限られる。
    したがって線形化した共分散行列はランク $m-1$ に落ち、$\hat{a}$ 方向の分散は 0 に
    なる（厳密には球面上の変位に 2 次の法線方向成分が残るので、この方向の分散は
    完全な 0 ではない。後のモンテカルロで実際の大きさを測る）。この近似のもとで、
    代数的最小二乗の解の共分散は次で与えられる。

    $$
    V[\hat{a}] = \frac{\sigma^2}{N}\ \bar{M}^{-}\ \bar{N}\ \bar{M}^{-},
    \qquad
    \bar{M} = \frac{1}{N}\sum_i \xi_i \xi_i^\top,
    \qquad
    \bar{N} = \frac{1}{N}\sum_i \bigl(\hat{a},\ V_0[\xi_i]\,\hat{a}\bigr)\ \xi_i \xi_i^\top
    $$

    $\bar{M}^{-}$ は最小固有値の方向（$\hat{a}$ 自身の方向）を除いたランク $m-1$ の
    擬似逆行列である。$\sigma^2$ は観測誤差の分散で、Sampson 誤差の和から推定できる。

    $$
    \hat{\sigma}^2 = \frac{1}{N-(m-1)}\ \sum_i
    \frac{(\xi_i,\ \hat{a})^2}{\bigl(\hat{a},\ V_0[\xi_i]\,\hat{a}\bigr)}
    $$

    この式は誤差が有限の分散を持つことを前提にしている。楕円のデータに載せたのは
    分散を持たないコーシー分布であり、さらに RANSAC が選んだ inlier はデータに依存
    する切り出し方なので、次のセルで得る $\hat{\sigma}$ は「inlier に選ばれた点の
    曲線からのばらつきの尺度」であって、元の観測誤差の標準偏差ではない。式そのものの
    妥当性は、その後の正規分布誤差によるモンテカルロで確かめる。
    陽関数の節で使った $\chi^2/(N-m)\,(X^\top X)^{-1}$ をそのまま持ち込むことは
    できない。理由は 2 つある。第 1 に $(\xi, a)$ は幾何距離ではないので、
    $\chi^2$ をそのまま誤差の分散の推定に使えない。第 2 に $\|a\|=1$ の制約を無視して
    $(X^\top X)^{-1}$ を取ると、ほとんど零空間である $\hat{a}$ 方向の固有値の逆数が
    混入する。次のセルで両方の式を計算し、モンテカルロで求めた経験的な共分散と
    突き合わせる。
    """)
    return


@app.cell
def _(normalized_cov):
    def rank_pseudo_inverse(matrix, rank):
        """対称行列の、大きい方から rank 個の固有値だけを使った擬似逆行列。

        :param matrix: 対称行列
        :param rank: 残す固有値の数
        :return: 擬似逆行列
        """
        values, vectors = np.linalg.eigh(matrix)
        kept = np.argsort(values)[::-1][:rank]
        return sum(np.outer(vectors[:, i], vectors[:, i]) / values[i] for i in kept)

    def algebraic_covariance(basis_func, points, params):
        """代数的最小二乗で求めた係数の誤差共分散行列を推定する。

        :param basis_func: 基底関数
        :param points: 形状 (N, 2) の点群
        :param params: 形状 (m,) の推定した係数
        :return: (共分散行列 (m, m), 推定した観測誤差の分散)
        """
        design = np.array([basis_func(point) for point in points])
        cov_stack = np.array([normalized_cov(basis_func, point) for point in points])
        weights = np.einsum("i,nij,j->n", params, cov_stack, params)
        sigma_sq = float(np.sum((design @ params) ** 2 / weights)) / (
            len(points) - (len(params) - 1)
        )
        moment = design.T @ design / len(points)
        weighted_moment = (design.T * weights) @ design / len(points)
        moment_pinv = rank_pseudo_inverse(moment, len(params) - 1)
        covariance = sigma_sq / len(points) * moment_pinv @ weighted_moment @ moment_pinv
        return covariance, sigma_sq

    return (algebraic_covariance,)


@app.cell
def _(
    algebraic_covariance,
    ellipse_mask,
    ellipse_params,
    ellipse_points,
    ellipse_residual,
    quad_basis,
    show_matrix,
):
    ellipse_inlier_points = ellipse_points[ellipse_mask, :]
    ellipse_cov, ellipse_sigma_sq = algebraic_covariance(
        quad_basis, ellipse_inlier_points, ellipse_params
    )
    _design = np.array([quad_basis(point) for point in ellipse_inlier_points])
    ellipse_naive_cov = (
        np.linalg.inv(_design.T @ _design)
        * ellipse_residual
        / (len(ellipse_inlier_points) - _design.shape[1])
    )
    print(f"inlier 数 = {len(ellipse_inlier_points)}")
    print(f"Sampson 誤差から求めた sigma = {np.sqrt(ellipse_sigma_sq):.5f}")
    print(f"共分散の対角     = {np.diag(ellipse_cov)}")
    print(f"旧来の式の対角   = {np.diag(ellipse_naive_cov)}")
    print(f"cond(X^T X)      = {np.linalg.cond(_design.T @ _design):.3e}")
    print(f"a 方向の分散     = {ellipse_params @ ellipse_cov @ ellipse_params:.3e}（線形化により 0）")
    show_matrix(ellipse_cov, "covariance of the conic coefficients")
    return (ellipse_inlier_points,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    式が正しいかどうかは、同じ真の楕円から誤差だけを引き直したデータを大量に作り、
    推定値のばらつきを直接測れば確かめられる。ここでは外れ値の影響を切り離すため、
    標準偏差 $0.01$ の正規分布誤差だけを乗せた点群を 1000 組作り、それぞれに代数的
    最小二乗を当てはめて経験的な共分散行列を求める。理論式の側は、そのうちの
    1 組だけを使い、その 1 組から当てはめた係数と $\hat{\sigma}$ で計算する。
    実務で手元にあるのは 1 組だけだからである。符号の不定性を揃えるためにだけ、
    誤差なしの点群から求めた係数を基準に使う。

    あわせて、経験的な共分散の基準係数方向の成分も測る。線形化では 0 になる量で、
    実際にどの程度小さいかがそのまま近似の妥当性を示す。
    """)
    return


@app.cell
def _(algebraic_covariance, ellipse_clean_points, quad_basis):
    montecarlo_trials = 1000
    montecarlo_noise_sd = 0.01
    _reference_design = np.array([quad_basis(point) for point in ellipse_clean_points])
    _values, _vectors = np.linalg.eigh(
        _reference_design.T @ _reference_design / len(ellipse_clean_points)
    )
    montecarlo_reference = _vectors[:, 0]
    _rng = np.random.default_rng(12)
    _estimates = []
    for _trial in range(montecarlo_trials):
        _noisy = ellipse_clean_points + montecarlo_noise_sd * _rng.standard_normal(
            ellipse_clean_points.shape
        )
        _design = np.array([quad_basis(point) for point in _noisy])
        _eigenvalues, _eigenvectors = np.linalg.eigh(_design.T @ _design / len(_noisy))
        _estimate = _eigenvectors[:, 0]
        if _estimate @ montecarlo_reference < 0:
            _estimate = -_estimate
        _estimates.append(_estimate)
        if _trial == 0:
            montecarlo_single = _noisy
            montecarlo_single_params = _estimate
    montecarlo_empirical = np.cov(np.array(_estimates).T)
    montecarlo_theory, montecarlo_sigma_sq = algebraic_covariance(
        quad_basis, montecarlo_single, montecarlo_single_params
    )
    _normal_variance = montecarlo_reference @ montecarlo_empirical @ montecarlo_reference
    print(f"試行数 = {montecarlo_trials}, 誤差の標準偏差 = {montecarlo_noise_sd}")
    print(f"1 組から推定した sigma = {np.sqrt(montecarlo_sigma_sq):.6f}")
    print(f"経験的な共分散の跡 = {np.trace(montecarlo_empirical):.6e}")
    print(f"理論式の共分散の跡 = {np.trace(montecarlo_theory):.6e}")
    print(f"跡の比（理論/経験） = {np.trace(montecarlo_theory) / np.trace(montecarlo_empirical):.4f}")
    print(f"対角成分の比       = {np.round(np.diag(montecarlo_theory) / np.diag(montecarlo_empirical), 4)}")
    print(f"経験的な共分散の基準係数方向の分散 = {_normal_variance:.3e}（跡の "
          f"{_normal_variance / np.trace(montecarlo_empirical):.1e} 倍）")
    return (
        montecarlo_empirical,
        montecarlo_single,
        montecarlo_single_params,
        montecarlo_theory,
    )


@app.cell
def _(
    montecarlo_empirical,
    montecarlo_single,
    montecarlo_single_params,
    montecarlo_theory,
    quad_basis,
    show_matrix,
):
    _design = np.array([quad_basis(point) for point in montecarlo_single])
    _chi2 = float(np.sum((_design @ montecarlo_single_params) ** 2))
    _naive = (
        np.linalg.inv(_design.T @ _design) * _chi2 / (len(montecarlo_single) - _design.shape[1])
    )
    print(f"旧来の式 / 経験的な共分散 の対角比 = {np.round(np.diag(_naive) / np.diag(montecarlo_empirical), 1)}")
    show_matrix(montecarlo_empirical, "empirical covariance (1000 trials)")
    show_matrix(montecarlo_theory, "theoretical covariance (single sample)")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### scipy.odr で同じ楕円を当てはめる

    前半の円では `scipy.odr` がそのまま使えた。しかし円錐曲線の陰関数
    $(\xi, a) = 0$ は係数について同次であり、$a = 0$ が残差 0 の解になってしまう。
    `scipy.odr` はスケールの拘束を持たないため、素直に渡すとこの自明解へ収束する。
    """)
    return


@app.cell
def _(ellipse_inlier_points, odr, quad_basis):
    def conic_residual(beta, point):
        """円錐曲線の陰関数 (ξ, a)。

        :param beta: 形状 (6,) の係数
        :param point: 形状 (2, N) の点群
        :return: 形状 (N,) の陰関数の値
        """
        return np.dot(beta, quad_basis(point))

    odr_model = odr.Model(conic_residual, implicit=True)
    odr_data = odr.Data(ellipse_inlier_points.T, y=1)
    odr_free_result = odr.ODR(odr_data, odr_model, beta0=[1 / 6] * 6).run()
    print(f"推定した beta = {np.round(odr_free_result.beta, 10)}")
    print(f"|beta| = {np.linalg.norm(odr_free_result.beta):.3e}")
    print(f"残差二乗和 = {odr_free_result.sum_square:.3e}")
    print(f"停止理由 = {odr_free_result.stopreason}")
    return odr_data, odr_model


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    自明解を避けるには、スケールを外から固定すればよい。`ODR` の `ifixb` は
    「値 0 の成分を固定し、正の成分を自由にする」指定なので、代数解で絶対値が最大の
    成分をその値のまま固定し、残り 5 個を動かす。こうして得た解を単位ベクトルへ
    正規化すれば、代数的最小二乗の解と比べられる。
    """)
    return


@app.cell
def _(
    describe_conic,
    ellipse_params,
    ellipse_true_center,
    ellipse_true_semi_axes,
    odr,
    odr_data,
    odr_model,
):
    _anchor = int(np.argmax(np.abs(ellipse_params)))
    _ifixb = [1] * len(ellipse_params)
    _ifixb[_anchor] = 0
    odr_fixed_result = odr.ODR(
        odr_data, odr_model, beta0=list(ellipse_params), ifixb=_ifixb
    ).run()
    odr_beta = odr_fixed_result.beta / np.linalg.norm(odr_fixed_result.beta)
    if odr_beta @ ellipse_params < 0:
        odr_beta = -odr_beta
    _angle = np.degrees(np.arccos(np.clip(abs(odr_beta @ ellipse_params), 0.0, 1.0)))
    _center, _semi_axes, _angles = describe_conic(odr_beta)
    print(f"固定した成分 = {_anchor} 番目")
    print(f"停止理由 = {odr_fixed_result.stopreason}")
    print(f"ODR の係数   = {np.round(odr_beta, 5)}")
    print(f"代数解の係数 = {np.round(ellipse_params, 5)}")
    print(f"2 つの係数ベクトルのなす角 = {_angle:.4f} 度")
    print(f"ODR の 中心   = {np.round(_center, 4)} / 真値 = {ellipse_true_center}")
    print(f"ODR の 半軸長 = {np.round(_semi_axes, 4)} / 真値 = {ellipse_true_semi_axes}")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## まとめ

    - 陽関数の最小二乗は `numpy.polyfit` `scipy.optimize.curve_fit`
      `scipy.optimize.leastsq` のいずれでも同じ解に到達する。違いは扱えるモデルの
      範囲と共分散行列の正規化で、`leastsq` の `cov_x` だけは $\chi^2/(N-m)$ を
      掛けないと他と揃わない。
    - 陰関数は 1 つの $x$ に複数の $y$ が対応する曲線を扱えるが、係数にスケールの
      不定性がある。同次形のまま `scipy.odr` へ渡すと $a=0$ の自明解に落ちるので、
      $\|a\|=1$ の正規化か、成分の 1 つを固定する拘束が要る。
    - 外れ値があると最小二乗は破綻する。RANSAC は最小個数の点で候補を作り、
      inlier が最多の候補を選ぶ。試行回数は $N=\ln(1-p)/\ln(1-e^n)$ で見積もり、
      inlier 率の更新に合わせて縮められる。
    - 陰関数の RANSAC では、代数的残差ではなく Sampson 誤差で距離を測る。分母に
      現れる正規化共分散行列 $V_0[\xi]$ は、二重数による自動微分で基底関数から
      機械的に得られる。
    - 正規化した係数の共分散行列は、誤差が小さい範囲の一次近似ではランク $m-1$ に
      落ちる。擬似逆行列を使った式は 1000 回のモンテカルロで得た経験的な共分散と
      一致する。
    """)
    return


if __name__ == "__main__":
    app.run()
