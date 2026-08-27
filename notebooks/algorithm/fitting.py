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
    <a href="https://colab.research.google.com/github/applejxd/colaboratory/blob/master/algorithm/fitting.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 外れ値なしフィッティング
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 陽関数の場合
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    2次関数+正規分布誤差で検証
    """)
    return


@app.cell
def _():
    import matplotlib.pyplot as plt
    import numpy as np

    np.random.seed(42)
    point_num: int = 1000
    _errors = np.random.normal(0, 2, (point_num,))
    x = np.linspace(0, 10, point_num)
    # 誤差
    y = x ** 2 + _errors / 10
    points = np.vstack([x, y]).T
    plt.scatter(x, y, s=0.5)
    plt.xlim(0, 3.5)
    plt.ylim(-1, 11)
    plt.show()
    return np, plt, points, x, y


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    陽関数フィッティング。
    代表的な Python ライブラリを使う方法は以下の3つ：
    1. [numpy.polyfit](https://numpy.org/doc/stable/reference/generated/numpy.polyfit.html)
    2. [scipy.optimize.curve_fit](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html)
    3. [scipy.optimize.leastsq](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.leastsq.html)

    それぞれを使った方法を示す。
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### numpy.polyfit を使う方法
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    多項式フィッティングを行う場合に簡便である。
    """)
    return


@app.cell
def _(np, plt, points):
    params, _cov_mat = np.polyfit(points[:, 0], points[:, 1], deg=2, cov=True)
    # 誤差共分散行列を可視化
    plt.imshow(_cov_mat)
    return (params,)


@app.cell
def _(params, plt, x, y):
    def quad_func(x, a, b, c):
        return a*x**2+b*x+c

    plt.scatter(x, y, s=0.5)
    plt.plot(x, quad_func(x, *params), color="orange")
    plt.xlim(0, 3.5)
    plt.ylim(-1, 11)
    plt.show()
    return (quad_func,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### scipy.optimize.curve_fit を使う方法
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    numpy.polyfit と比較して、多項式以外のフィッティングが可能である。
    """)
    return


@app.cell
def _(plt, quad_func, x, y):
    import scipy.optimize as spo

    p_opt, p_cov = spo.curve_fit(quad_func, x, y)
    # 誤差共分散行列の可視化
    plt.imshow(p_cov)
    return p_opt, spo


@app.cell
def _(p_opt, plt, quad_func, x, y):
    plt.scatter(x, y, s=0.5)
    plt.plot(x, quad_func(x, *p_opt), color="orange")
    plt.xlim(0, 3.5)
    plt.ylim(-1, 11)
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### scipy.optimize.leastsq を使う方法
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    フィッティングの損失関数の設定が可能である。
    """)
    return


@app.cell
def _(plt, points, quad_func, spo):
    def loss_func(params, points):
        y_est = quad_func(points[:, 0], *params)
        return y_est - points[:, 1]
    result = spo.leastsq(loss_func, (0, 0, 0), args=points, full_output=True)
    params_1 = result[0]
    _cov_mat = result[1]
    plt.imshow(_cov_mat)
    return (params_1,)


@app.cell
def _(params_1, plt, quad_func, x, y):
    plt.scatter(x, y, s=0.5)
    plt.plot(x, quad_func(x, *params_1), color='orange')
    plt.xlim(0, 3.5)
    plt.ylim(-1, 11)
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 陰関数の場合
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    円+正規分布誤差で検証
    """)
    return


@app.cell
def _(np, plt):
    point_num_1 = 500
    _errors = 0.02 * np.random.normal(0, 1, (point_num_1, 2))
    _theta = np.linspace(0, 2 * np.pi, point_num_1)
    radius, _center = (1.3, (1, 2))
    x_1 = radius * np.cos(_theta) + _center[0] + _errors[:, 0]
    y_1 = radius * np.sin(_theta) + _center[1] + _errors[:, 1]
    points_1 = np.vstack([x_1, y_1]).T
    plt.scatter(points_1[:, 0], points_1[:, 1], s=0.5)
    plt.axis('equal')
    plt.show()
    return (points_1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### scipy.odr を使う方法
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    [ODR (Orthogonal Distance Regression)](https://pota.hatenablog.jp/entry/2014/10/31/033326)を実施
    """)
    return


@app.cell
def _(plt, points_1):
    from scipy import odr

    def circ_func(beta, x):
        return (x[0] - beta[0]) ** 2 + (x[1] - beta[1]) ** 2 - beta[2] ** 2
    _model = odr.Model(circ_func, implicit=True)
    _data = odr.Data(points_1.T, y=1)
    _solver = odr.ODR(_data, _model, beta0=[0, 0, 1])
    result_1 = _solver.run()
    result_1.pprint()
    plt.imshow(result_1.cov_beta)
    return circ_func, odr, result_1


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    推定結果の陰関数のヒートマップを作成(0となる部分空間が推定ライン)
    """)
    return


@app.cell
def _(circ_func, np, plt, points_1, result_1):
    _diff = 0.01
    _x_range = np.arange(np.min(points_1[:, 0]), np.max(points_1[:, 0]), _diff)
    _y_range = np.arange(np.min(points_1[:, 1]), np.max(points_1[:, 1]), _diff)
    X, Y = np.meshgrid(_x_range, _y_range)
    Z = circ_func(result_1.beta, (X, Y))
    plt.imshow(Z)
    plt.colorbar()
    return X, Y, Z


@app.cell
def _(X, Y, Z, plt, points_1):
    plt.scatter(points_1[:, 0], points_1[:, 1], s=0.5, color='blue')
    plt.contour(X, Y, Z, [0], colors='orange')
    plt.axis('equal')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## RANSAC
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### RANSAC (陽関数)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### 直線フィッティング
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    誤差分布として外れ値が大きいコーシー分布を仮定
    """)
    return


@app.cell
def _(np, plt):
    def cauchy(x: np.ndarray):
        return 1 / np.pi / (1 + x * x)
    x_2 = np.linspace(-5, 5, 100)
    y_2 = cauchy(x_2)
    plt.plot(x_2, y_2)
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    サンプルデータ生成
    """)
    return


@app.cell
def _(np, plt):
    point_num_2: int = 1000
    _errors = np.random.standard_cauchy(point_num_2)
    x_3 = np.linspace(0, 10, point_num_2)
    y_3 = x_3 + _errors / 10
    points_2 = np.vstack([x_3, y_3]).T
    plt.scatter(x_3, y_3, s=0.5)
    plt.ylim(-1, 11)
    plt.show()
    return point_num_2, points_2, x_3, y_3


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    iteration の回数の目安は以下の通り：

    1. データ分布から、1回のサンプリングで inlier を引き当てる確率 $e$ を仮定
    2. 1回の試行でのサンプル数を $n$ とすると、それらのサンプルが全て inlier である確率は $e^n$
    3. 一方で$N$回の試行で 2. が1回も生じない場合(RANSAC が失敗する場合)の確率は$(1-e^n)^N$
    4. つまり RANSAC をほぼ確実(確率$p\sim1$)で成功させたい場合は
    $$
    1-p=(1-e^n)^N⇔N=\frac{\ln(1-p)}{\ln(1-e^n)}
    $$
    くらいの$N$に設定して反復する。
    4. 更に早期終了を目指す場合は、最大の inlier 数更新のタイミングで $e\sim$  (inlier とした数)/(データ数) として$N$を更新
    """)
    return


@app.cell
def _(np):
    # データの目視確認より 9/10 は対象データと想定
    _e = 9 / 10
    # 直線のパラメータは2個 → サンプリング数は2個で十分
    _n = 2
    # 99.99% の確率で RANSAC を成功させる
    _p = 0.9999
    # zero division error 対策
    max_iteration = int(np.log(1 - _p) / _) if not np.isclose((_ := np.log(1 - _e ** _n)), 0) else 100000.0
    print(max_iteration)
    return (max_iteration,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    乱数生成
    """)
    return


@app.cell
def _(max_iteration, np, point_num_2: int):
    index_mat = np.random.randint(0, point_num_2, (max_iteration, 2))
    print(index_mat)
    return (index_mat,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ランダムサンプリング
    """)
    return


@app.cell
def _(index_mat, np, x_3, y_3):
    np.array([x_3[index_mat], y_3[index_mat]]).shape
    return


@app.cell
def _(index_mat, np, x_3, y_3):
    # 転置前の成分数は (axis, trials, points)
    # 転置後の成分数は (trials, points, axis)
    samples_tensor = np.array([x_3[index_mat], y_3[index_mat]]).transpose(1, 2, 0)
    print(samples_tensor.shape)
    print(samples_tensor)
    return (samples_tensor,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    numpy.polyfit で傾きと切片計算
    """)
    return


@app.cell
def _(np, samples_tensor):
    params_tensor = np.array([np.polyfit(sample[:, 0], sample[:, 1], 1) 
                              for sample in samples_tensor])
    return (params_tensor,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    inlier の数を計算 → 最も良い候補を選択
    """)
    return


@app.cell
def _(np, params_tensor, points_2):
    def polynomial(params, points):
        max_degree = len(params)
        terms = np.array([params[degree] * points[:, 0] ** (max_degree - degree - 1) for degree in range(max_degree)])
        return np.sum(terms, axis=0)

    def _get_distances(params, points):
        y_est = polynomial(params, points)
        distances = np.abs(points[:, 1] - y_est)
        return distances
    _distances_tensor = np.array([_get_distances(params, points_2) for params in params_tensor])
    _distance_th = 0.3
    inlier_bool_tensor = np.array([distances < _distance_th for distances in _distances_tensor])
    _inlier_num_array = np.array([np.sum(inlier_bool) for inlier_bool in inlier_bool_tensor])
    best_idx = np.argmax(_inlier_num_array)
    print(best_idx)
    return best_idx, inlier_bool_tensor, polynomial


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    結果を確認
    """)
    return


@app.cell
def _(
    best_idx,
    inlier_bool_tensor,
    np,
    params_tensor,
    plt,
    point_num_2: int,
    points_2,
    polynomial,
):
    params_2 = params_tensor[best_idx]
    inlier_idx = np.arange(point_num_2)[inlier_bool_tensor[best_idx]]
    outlier_idx = np.arange(point_num_2)[~inlier_bool_tensor[best_idx]]
    plt.scatter(points_2[inlier_idx, 0], points_2[inlier_idx, 1], s=0.5, color='orange')
    plt.scatter(points_2[outlier_idx, 0], points_2[outlier_idx, 1], s=0.5, color='blue')
    plt.plot(points_2[:, 0], polynomial(params_2, points_2), color='green')
    plt.ylim(-1, 11)
    plt.show()
    return inlier_idx, outlier_idx


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    再フィッティング
    """)
    return


@app.cell
def _(inlier_idx, np, outlier_idx, plt, points_2, polynomial):
    _refined_params = np.polyfit(points_2[inlier_idx, 0], points_2[inlier_idx, 1], 1)
    plt.scatter(points_2[inlier_idx, 0], points_2[inlier_idx, 1], s=0.5, color='orange')
    plt.scatter(points_2[outlier_idx, 0], points_2[outlier_idx, 1], s=0.5, color='blue')
    plt.plot(points_2[:, 0], polynomial(_refined_params, points_2), color='green')
    plt.plot(points_2[:, 0], polynomial(_refined_params, points_2), color='red')
    plt.ylim(-1, 11)
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### 2次曲線フィッティング
    """)
    return


@app.cell
def _(np, plt):
    point_num_3: int = 1000
    _errors = np.random.standard_cauchy(point_num_3)
    x_4 = np.linspace(0, 10, point_num_3)
    y_4 = x_4 ** 2 + _errors / 10
    points_3 = np.vstack([x_4, y_4]).T
    plt.scatter(x_4, y_4, s=0.5)
    plt.xlim(0, 3.5)
    plt.ylim(-1, 11)
    plt.show()
    return point_num_3, points_3, x_4, y_4


@app.cell
def _(np):
    _e = 6 / 10
    _n = 3
    _p = 0.9999
    max_iteration_1 = int(np.log(1 - _p) / _) if not np.isclose((_ := np.log(1 - _e ** _n)), 0) else 100000.0
    print(max_iteration_1)
    return (max_iteration_1,)


@app.cell
def _(max_iteration_1, np, point_num_3: int, x_4, y_4):
    index_mat_1 = np.random.randint(0, point_num_3, (max_iteration_1, 3))
    samples_tensor_1 = np.array([x_4[index_mat_1], y_4[index_mat_1]]).transpose(1, 2, 0)
    # 転置前の成分数は (axis, trials, points)
    # 転置後の成分数は (trials, points, axis)
    print(samples_tensor_1.shape)
    return (samples_tensor_1,)


@app.cell
def _(np, samples_tensor_1):
    params_tensor_1 = np.array([np.polyfit(sample[:, 0], sample[:, 1], 2) for sample in samples_tensor_1])
    return (params_tensor_1,)


@app.cell
def _(np, params_tensor_1, points_3):
    def _get_distances(params, points):
        y_est = params[0] * points[:, 0] ** 2 + params[1] * points[:, 0] + params[2]
        distances = np.abs(points[:, 1] - y_est)
        return distances
    _distances_tensor = np.array([_get_distances(params, points_3) for params in params_tensor_1])
    _distance_th = 0.3
    inlier_bool_tensor_1 = np.array([distances < _distance_th for distances in _distances_tensor])
    _inlier_num_array = np.array([np.sum(inlier_bool) for inlier_bool in inlier_bool_tensor_1])
    best_idx_1 = np.argmax(_inlier_num_array)
    print(best_idx_1)
    return best_idx_1, inlier_bool_tensor_1


@app.cell
def _(
    best_idx_1,
    inlier_bool_tensor_1,
    np,
    params_tensor_1,
    plt,
    point_num_3: int,
    points_3,
):
    params_3 = params_tensor_1[best_idx_1]
    inlier_idx_1 = np.arange(point_num_3)[inlier_bool_tensor_1[best_idx_1]]
    outlier_idx_1 = np.arange(point_num_3)[~inlier_bool_tensor_1[best_idx_1]]
    plt.scatter(points_3[inlier_idx_1, 0], points_3[inlier_idx_1, 1], s=0.5, color='orange')
    plt.scatter(points_3[outlier_idx_1, 0], points_3[outlier_idx_1, 1], s=0.5, color='blue')
    y_est = params_3[0] * points_3[:, 0] ** 2 + params_3[1] * points_3[:, 0] + params_3[2]
    plt.plot(points_3[:, 0], y_est, color='green')
    plt.xlim(0, 3.5)
    plt.ylim(-1, 11)
    plt.show()
    return inlier_idx_1, outlier_idx_1, params_3


@app.cell
def _(inlier_idx_1, np, outlier_idx_1, params_3, plt, points_3, polynomial):
    _refined_params = np.polyfit(points_3[inlier_idx_1, 0], points_3[inlier_idx_1, 1], 2)
    y_refined = _refined_params[0] * points_3[:, 0] ** 2 + _refined_params[1] * points_3[:, 0] + _refined_params[2]
    plt.scatter(points_3[inlier_idx_1, 0], points_3[inlier_idx_1, 1], s=0.5, color='orange')
    plt.scatter(points_3[outlier_idx_1, 0], points_3[outlier_idx_1, 1], s=0.5, color='blue')
    plt.plot(points_3[:, 0], polynomial(params_3, points_3), color='green')
    plt.plot(points_3[:, 0], y_refined, color='red')
    plt.xlim(0, 3.5)
    plt.ylim(-1, 11)
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### RANSAC (陰関数)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### 円の例
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    基底関数を定義
    """)
    return


@app.cell
def _():
    def linear_basis(point):
        x, y = point
        return [x, y, x * 0 + 1]

    def quad_basis(point):
        x, y = point
        return [x**2, 2*x*y, y**2, 2*x, 2*y, x * 0 + 1]

    return linear_basis, quad_basis


@app.cell
def _(np, plt):
    point_num_4 = 500
    _errors = 0.02 * np.random.normal(0, 1, (point_num_4, 2))
    _theta = np.linspace(0, 2 * np.pi, point_num_4)
    x_5 = np.cos(_theta) + _errors[:, 0]
    y_5 = np.sin(_theta) + _errors[:, 1]
    points_4 = np.vstack([x_5, y_5]).T
    plt.scatter(points_4[:, 0], points_4[:, 1], s=0.5)
    plt.axis('equal')
    plt.xlim(-2, 2)
    plt.ylim(-2, 2)
    plt.show()
    return (points_4,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    構造行列の計算例
    """)
    return


@app.cell
def _(np, plt, points_4, quad_basis):
    xi = quad_basis(points_4[20])
    mat = np.outer(xi, xi)
    plt.imshow(mat)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    誤差がある系だと正定値行列。誤差がない系だと det=0 で非負定値。
    """)
    return


@app.cell
def _(np, plt, points_4, quad_basis):
    xi_list = np.array([quad_basis(point) for point in points_4])
    structure_mat = np.mean(np.array([np.outer(xi, xi) for xi in xi_list]), 0)
    plt.imshow(structure_mat)
    print(f'det={np.linalg.det(structure_mat)}')
    try:
        np.linalg.cholesky(structure_mat)
    except np.linalg.LinAlgError as error:
        print(error)
    else:
        print('Matrix is positive definite')
    e_value, e_vec = np.linalg.eig(structure_mat)
    print(f'e_value={e_value}')
    params_4 = e_vec[:, np.argmin(e_value)]
    print(f'params={params_4}')
    return (params_4,)


@app.cell
def _(np, params_4, plt, points_4, quad_basis):
    _diff = 0.01
    _x_range = np.arange(np.min(points_4[:, 0]), np.max(points_4[:, 0]), _diff)
    _y_range = np.arange(np.min(points_4[:, 1]), np.max(points_4[:, 1]), _diff)
    X_1, Y_1 = np.meshgrid(_x_range, _y_range)
    basises = np.array([basis * np.ones_like(X_1) for basis in quad_basis((X_1, Y_1))])
    Z_1 = np.einsum('i,ijk->jk', params_4, basises)
    plt.imshow(Z_1)
    plt.colorbar()
    return X_1, Y_1, Z_1


@app.cell
def _(X_1, Y_1, Z_1, plt, points_4):
    plt.scatter(points_4[:, 0], points_4[:, 1], s=0.5, color='blue')
    plt.contour(X_1, Y_1, Z_1, [0], colors='orange')
    plt.axis('equal')
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### RANSAC クラスの作成
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    自動微分を実施するための Jet を定義
    """)
    return


@app.cell
def _(np, quad_basis):
    class Jet:

        def __init__(self, a: float, v):
            self.a = a
            self.v = np.array(v)

        def __str__(self):
            return f'{self.a}+{self.v}'

        def __add__(self, other):
            if isinstance(other, (int, float)):
                other = Jet(other, np.zeros(len(self.v)))
            return Jet(self.a + other.a, self.v + other.v)

        def __sub__(self, other):
            if isinstance(other, (int, float)):
                other = Jet(other, np.zeros(len(self.v)))
            return Jet(self.a - other.a, self.v - other.v)

        def __mul__(self, other):
            if isinstance(other, (int, float)):
                other = Jet(other, np.zeros(len(self.v)))
            return Jet(self.a * other.a, self.a * other.v + self.v * other.a)

        def __truediv__(self, other):
            if isinstance(other, (int, float)):
                other = Jet(other, np.zeros(len(self.v)))
            return Jet(self.a / other.a, self.v / other.a - self.a * other.v / other.a ** 2)

        def __pow__(self, other):
            if isinstance(other, (int, float)):
                other = Jet(other, np.zeros(len(self.v)))
            value = self.a ** other.a
            base_diff = other.a * self.a ** (other.a - 1) * self.v  # 底が0, 負の場合に注意
            if np.isclose(self.a, 0):
                exp_diff = 0
            else:
                exp_diff = value * other.v * np.log(np.abs(self.a))
            return Jet(value, base_diff + exp_diff)

        def __radd__(self, other):
            return self.__add__(other)

        def __rsub__(self, other):
            return self.__sub__(other)

        def __rmul__(self, other):
            return self.__mul__(other)
    x_6 = Jet(1, (1, 0))
    y_6 = Jet(2, (0, 1))
    print(quad_basis((x_6, y_6))[1])
    return (Jet,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    前セクションまでの処理を踏まえて、RANSAC を行うための一般的なクラスを定義する
    """)
    return


@app.cell
def _(Jet, np):
    import random
    from typing import List

    import numpy.linalg as LA
    from tqdm import tqdm

    class RANSAC:

        def __init__(self, basis_func, target_prob=0.9999):
            self.basis_func = basis_func
            self.target_prob = target_prob

        def fit(self, points):
            xi_list = np.array([self.basis_func(point) for point in points])
            mat_list = np.array([np.outer(xi, xi) for xi in xi_list])
            mat = np.mean(mat_list, axis=0)
            if not np.isclose(np.linalg.det(mat), 0):
                print('This fitting is statistics mode.')
            e_value, e_vec = LA.eig(mat)
            idx = np.argmin(e_value)
            return (e_vec[:, idx], e_value[idx] * mat_list.shape[0])

        def jet_basis(self, point: np.ndarray) -> List[Jet]:
            """Jet 型の基底関数
      # 固有ベクトルは列毎に格納されていることに注意
            :param point: データ値(規格化済)
            :return:
            """
            jets = [Jet(point[idx], np.eye(len(point))[idx]) for idx in range(len(point))]
            basis_list = self.basis_func(jets)
            return [basis + Jet(0, np.zeros_like(point)) for basis in basis_list]

        def get_cov_mat(self, point: np.ndarray) -> np.ndarray:
            """正規化共分散行列

            :param point: データ値(規格化済)
            :return: 正規化共分散行列
            """
            xi = self.jet_basis(point)
            return np.array([[np.dot(xi[i].v, xi[j].v) for j in range(len(xi))] for i in range(len(xi))])

        def sampson_error(self, point: np.ndarray, params: np.ndarray):
            _cov_mat = self.get_cov_mat(point)
            denominator = np.dot(params, _cov_mat @ params)
            xi = self.basis_func(point)
            numerator = np.dot(xi, params) ** 2
            return numerator / denominator if not np.isclose(denominator, 0) else np.inf

        def criteria(self, inlier_prob: float, degree: int) -> int:
            """RANSAC の終了条件を計算

            :param inlier_prob: inlier の割合/確率
            :param degree: フィッティング次数/サンプル数
            :return: RANSAC の反復回数の目安
            """
            return int(np.log(1 - self.target_prob) / _) if not np.isclose((_ := np.log(1 - inlier_prob ** degree)), 0) else 100000.0

        def execute(self, points, distance_th=0.1):
            data_num = points.shape[0]
            degree = len(self.basis_func(points[0, :])) - 1
            inlier_prob = 0.1
            max_iteration = self.criteria(inlier_prob, degree)
            iteration = 0
            best_inlier_bool = [False for _ in range(data_num)]
            best_inlier_num = 0
            best_params = []
            with tqdm(total=max_iteration) as pbar:
                while iteration < max_iteration:
                    rand_idx = random.sample(range(data_num), degree)
                    params, _ = self.fit(points[rand_idx, :])
                    distance_sq = np.array([self.sampson_error(point, params) for point in points])
                    inlier_bool = distance_sq < distance_th ** 2
                    inlier_num = np.sum(inlier_bool)
                    iteration = iteration + 1
                    pbar.update(1)
                    if inlier_num > best_inlier_num:
                        max_iteration = min(max_iteration, self.criteria(inlier_num / data_num, degree))
                        best_inlier_bool = inlier_bool
                        best_inlier_num = inlier_num
                        best_params = params
                        pbar.total = max_iteration
                        pbar.refresh()
            best_params, loss = self.fit(points[best_inlier_bool, :])
            print(f'params={best_params}')
            print(f'fitness={best_inlier_num / data_num}')
            return (best_params, best_inlier_bool, loss)

    return (RANSAC,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### 線形フィッティング
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    実装の確認のために再度線形フィッティングを実施。
    上と同様にサンプルデータを生成。
    """)
    return


@app.cell
def _(np, plt):
    point_num_5: int = 1000
    _errors = np.random.standard_cauchy(point_num_5)
    x_7 = np.linspace(0, 10, point_num_5)
    y_7 = x_7 + _errors / 10
    points_5 = np.vstack([x_7, y_7]).T
    plt.scatter(x_7, y_7, s=0.5)
    plt.ylim(-1, 11)
    plt.show()
    return point_num_5, points_5


@app.cell
def _(RANSAC, linear_basis, points_5):
    linear_estimator = RANSAC(linear_basis)
    params_5, inlier_bools, loss = linear_estimator.execute(points_5, distance_th=0.3)
    return inlier_bools, params_5


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    陰関数プロットのためのヘルパー関数を定義
    """)
    return


@app.cell
def _(np):
    def get_contour(basis_func, points, params, diff):
        _x_range = np.arange(np.min(points[:, 0]), np.max(points[:, 0]), diff)  # グリッド生成
        _y_range = np.arange(np.min(points[:, 1]), np.max(points[:, 1]), diff)
        x_mesh, y_mesh = np.meshgrid(_x_range, _y_range)
        basises = np.array([basis * np.ones_like(x_mesh) for basis in basis_func((x_mesh, y_mesh))])
        z_mesh = np.einsum('i,ijk->jk', params, basises)
        return (x_mesh, y_mesh, z_mesh)

    return (get_contour,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    結果をプロット
    """)
    return


@app.cell
def _(
    get_contour,
    inlier_bools,
    linear_basis,
    np,
    params_5,
    plt,
    point_num_5: int,
    points_5,
):
    inlier_idx_2 = np.arange(point_num_5)[inlier_bools]
    outlier_idx_2 = np.arange(point_num_5)[~inlier_bools]
    plt.scatter(points_5[outlier_idx_2, 0], points_5[outlier_idx_2, 1], s=0.5, color='blue')
    plt.scatter(points_5[inlier_idx_2, 0], points_5[inlier_idx_2, 1], s=0.5, color='orange')
    plt.contour(*get_contour(linear_basis, points_5, params_5, 0.01), [0], colors='green')
    plt.ylim(-1, 11)
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### 2次曲線フィッティング
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    サンプルデータ生成
    """)
    return


@app.cell
def _(np, plt):
    _center = np.array([1, 2])
    scales = np.array([2, 3])
    rad = np.deg2rad(-30)
    point_num_6 = 500
    _errors = 0.01 * np.random.standard_cauchy((point_num_6, 2))
    _theta = np.linspace(0, 2 * np.pi, point_num_6)
    x_8 = _center[0] + scales[0] * np.cos(_theta) + _errors[:, 0]
    y_8 = _center[1] + scales[1] * np.sin(_theta) + _errors[:, 1]
    points_6 = np.vstack([x_8, y_8]).T
    rot_mat = np.array([[np.cos(rad), -np.sin(rad)], [np.sin(rad), np.cos(rad)]])
    points_6 = points_6 @ rot_mat
    plt.scatter(points_6[:, 0], points_6[:, 1], s=0.5)
    plt.xlim((-3, 3))
    plt.ylim((-2, 6))
    plt.show()
    return point_num_6, points_6


@app.cell
def _(RANSAC, points_6, quad_basis):
    quad_estimator = RANSAC(quad_basis)
    params_6, inlier_bools_1, loss_1 = quad_estimator.execute(points_6, distance_th=0.07)
    return inlier_bools_1, loss_1, params_6


@app.cell
def _(
    get_contour,
    inlier_bools_1,
    np,
    params_6,
    plt,
    point_num_6,
    points_6,
    quad_basis,
):
    inlier_idx_3 = np.arange(point_num_6)[inlier_bools_1]
    outlier_idx_3 = np.arange(point_num_6)[~inlier_bools_1]
    plt.scatter(points_6[outlier_idx_3, 0], points_6[outlier_idx_3, 1], s=0.5, color='blue')
    plt.scatter(points_6[inlier_idx_3, 0], points_6[inlier_idx_3, 1], s=0.5, color='orange')
    plt.contour(*get_contour(quad_basis, points_6, params_6, 0.01), [0], colors='green')
    plt.xlim((-3, 3))
    plt.ylim((-2, 6))
    plt.show()
    return (inlier_idx_3,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    誤差共分散行列
    $$
    V\equiv E[\delta a^T\delta a]
    =\frac{\chi^2}{n-m}(X^TX)^{-1}
    $$
    """)
    return


@app.cell
def _(inlier_bools_1, inlier_idx_3, loss_1, np, plt, points_6, quad_basis):
    basis_list = np.array([quad_basis(point) for point in points_6[inlier_idx_3, :]])
    variance = loss_1 / (np.sum(inlier_bools_1) - basis_list.shape[1])
    param_cov = np.linalg.inv(basis_list.T @ basis_list) * variance
    plt.imshow(param_cov)
    return (param_cov,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    // TODO: 出力パラメータがおかしい(小さすぎる・自明な解?)
    """)
    return


@app.cell
def _(inlier_idx_3, np, odr, plt, points_6, quad_basis):
    _model = odr.Model(lambda beta, x: np.dot(beta, quad_basis(x)), implicit=True)
    _data = odr.Data(points_6[inlier_idx_3, :].T, y=1)
    _solver = odr.ODR(_data, _model, beta0=[1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6])
    result_2 = _solver.run()
    result_2.pprint()
    plt.imshow(result_2.cov_beta)
    return (result_2,)


@app.cell
def _(param_cov, result_2):
    print(param_cov / result_2.cov_beta)
    return


if __name__ == "__main__":
    app.run()
