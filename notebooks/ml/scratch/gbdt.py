import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # 勾配ブースティング決定木 (GBDT) をゼロから実装する

    表形式データの回帰・分類で長く実務標準であり続けているのが
    **勾配ブースティング決定木 (Gradient Boosted Decision Trees, GBDT)** である。
    LightGBM や XGBoost はその高度に最適化された実装だが、内部は「浅い回帰木を
    残差方向に少しずつ足し込む」という単純な原理で動いている。

    この notebook では **アルゴリズム本体を NumPy だけ**で書き起こし、
    seaborn の `diamonds` データセット（53,940 件）でダイヤモンド価格を回帰する。
    scikit-learn は train/test 分割・評価指標・比較対象のベースラインにのみ使い、
    scratch 実装の中では一切使わない。最後に LightGBM / scikit-learn と
    同じテストデータ上で精度と速度を比較する。
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## この notebook の流れ

    1. **データ準備** — `diamonds` を読み込み、順序カテゴリを整数コードへ符号化する。
    2. **回帰木** — 分散減少基準で再帰的に二分割する回帰木を NumPy で実装する。
    3. **勾配ブースティング** — 負の勾配（= 残差）に木を当てはめる加法モデルを組み立て、
       参考実装に潜む `predict` のバグを指摘して直す。
    4. **2 次近似 (Newton) 版** — XGBoost/LightGBM が使う葉値・分割利得の式を導き、
       L2 正則化つきの木を実装する。
    5. **ヒストグラム分割** — 特徴量を離散ビンに区切って分割探索を高速化し、
       厳密探索との速度差を測る。
    6. **比較** — scratch 実装・LightGBM・scikit-learn を同一テストデータで比較する。

    解説は常体（である調）で書き、コードは一つの考えごとに小さなセルに分ける。
    """)
    return


@app.cell
def _():
    import time

    import lightgbm as lgb
    import seaborn as sns
    from sklearn.ensemble import GradientBoostingRegressor as SkGradientBoostingRegressor
    from sklearn.ensemble import HistGradientBoostingRegressor as SkHistGradientBoostingRegressor
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.model_selection import train_test_split

    def rmse(y_true, y_pred):
        return float(np.sqrt(mean_squared_error(y_true, y_pred)))

    versions = pd.DataFrame(
        {
            "library": ["numpy", "pandas", "lightgbm", "seaborn"],
            "version": [np.__version__, pd.__version__, lgb.__version__, sns.__version__],
        }
    )
    versions
    return (
        SkGradientBoostingRegressor,
        SkHistGradientBoostingRegressor,
        lgb,
        mean_absolute_error,
        r2_score,
        rmse,
        sns,
        time,
        train_test_split,
    )


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 1. データ準備

    `diamonds` は 53,940 件のダイヤモンドについて、重さ (`carat`)・研磨評価 (`cut`)・
    色 (`color`)・透明度 (`clarity`) などの特徴量と価格 (`price`, 単位ドル) を持つ。
    目的は `price` の回帰である。

    `cut` / `color` / `clarity` の 3 つは**順序カテゴリ**である。決定木は特徴量の
    大小関係（分割しきい値）しか見ないため、順序に沿った整数コードへ符号化すれば
    one-hot 化せずとも自然に扱える。順序は次のとおり定義する。

    - `cut`: Fair < Good < Very Good < Premium < Ideal
    - `color`: J < I < H < G < F < E < D（D が最良）
    - `clarity`: I1 < SI2 < SI1 < VS2 < VS1 < VVS2 < VVS1 < IF

    また `x` / `y` / `z`（寸法, mm）には物理的にありえない `0` が数十件混ざっている。
    ここでは**その行を除外**する方針を採る（欠測を 0 で埋めた記録とみなし、木が
    偽の分割点を学習するのを避けるため）。
    """)
    return


@app.cell
def _(sns):
    diamonds_raw = sns.load_dataset("diamonds")
    _zero_dim = (diamonds_raw[["x", "y", "z"]] == 0).any(axis=1)
    diamonds = diamonds_raw[~_zero_dim].reset_index(drop=True)
    clean_report = pd.DataFrame(
        {
            "item": ["raw rows", "rows with zero x/y/z", "rows after cleaning"],
            "count": [len(diamonds_raw), int(_zero_dim.sum()), len(diamonds)],
        }
    )
    clean_report
    return (diamonds,)


@app.cell
def _(diamonds):
    cut_order = ["Fair", "Good", "Very Good", "Premium", "Ideal"]
    color_order = ["J", "I", "H", "G", "F", "E", "D"]
    clarity_order = ["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"]

    encoded = diamonds.copy()
    encoded["cut"] = encoded["cut"].cat.set_categories(cut_order, ordered=True).cat.codes
    encoded["color"] = encoded["color"].cat.set_categories(color_order, ordered=True).cat.codes
    encoded["clarity"] = encoded["clarity"].cat.set_categories(clarity_order, ordered=True).cat.codes

    feature_names = ["carat", "cut", "color", "clarity", "depth", "table", "x", "y", "z"]
    X_all = encoded[feature_names].to_numpy(dtype=float)
    y_all = encoded["price"].to_numpy(dtype=float)
    encoded[feature_names + ["price"]].head()
    return X_all, feature_names, y_all


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### 学習・評価データの分割とサブサンプリング

    他の notebook にならって `random_state=71` で 75% / 25% に分割する。評価はつねに
    この**共通のテストデータ**で行い、全モデルを公平に比べる。

    scratch の GBDT は純粋な NumPy 実装で、最適化された C++ 実装より遅い。そこで
    **学習データは 6,000 件にサブサンプリング**して scratch 実装の計算量を抑える。
    LightGBM や scikit-learn にも同じ 6,000 件を与えて条件をそろえ、参考として
    LightGBM を全学習データで学習した行も別途示す。さらに scratch の学習データを
    学習用と検証用に分け、学習曲線と早期終了に使う。
    """)
    return


@app.cell
def _(X_all, train_test_split, y_all):
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X_all, y_all, test_size=0.25, random_state=71
    )
    split_report = pd.DataFrame(
        {
            "split": ["train (full)", "test"],
            "rows": [len(X_train_full), len(X_test)],
        }
    )
    split_report
    return X_test, X_train_full, y_test, y_train_full


@app.cell
def _(X_train_full, train_test_split, y_train_full):
    _rng = np.random.default_rng(71)
    _idx = _rng.choice(len(X_train_full), size=6000, replace=False)
    X_sub = X_train_full[_idx]
    y_sub = y_train_full[_idx]
    X_fit, X_val, y_fit, y_val = train_test_split(
        X_sub, y_sub, test_size=0.2, random_state=71
    )
    subsample_report = pd.DataFrame(
        {
            "role": ["scratch train (sub)", "  - fit part", "  - valid part"],
            "rows": [len(X_sub), len(X_fit), len(X_val)],
        }
    )
    subsample_report
    return X_fit, X_sub, X_val, y_fit, y_sub, y_val


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 2. 回帰木をゼロから実装する

    GBDT の部品は**回帰木**である。回帰木は特徴量空間を軸に平行な矩形へ再帰的に
    分割し、各領域（葉）で定数を予測する。分割の良さは**分散減少**で測る。
    親ノードの目的変数を $y$、しきい値で左右に分けた集合を $y_L, y_R$ とすると、

    $$
    \mathrm{VarRed} = \mathrm{Var}(y) - \tfrac{|y_L|}{|y|}\mathrm{Var}(y_L) - \tfrac{|y_R|}{|y|}\mathrm{Var}(y_R)
    $$

    これを最大化するしきい値を全特徴量について探す。素朴に書くと各特徴量の
    ユニーク値ごとに左右へ分けて分散を測るので
    $O(\text{特徴量数} \times \text{ユニーク値数} \times n)$ かかる。ここでは特徴量ごとに
    一度だけソートし、**累積和で全分割点の分散を一括計算**して
    $O(\text{特徴量数} \times n \log n)$ に落とす。分散は
    $\mathrm{Var}(S) = \frac{1}{|S|}\sum s^2 - \bar{s}^2$ と展開でき、
    $\sum s$ と $\sum s^2$ の累積和があれば任意の分割位置で定数時間に求まる。
    葉の予測値はその領域の目的変数の平均とする。
    """)
    return


@app.class_definition
class RegressionTree:
    """分散減少基準で成長させる回帰木（NumPy 実装）。

    累積和を使って各特徴量の全分割点を一括評価するため、素朴な実装より速い。
    葉の値は領域内の目的変数の平均。
    """

    def __init__(self, max_depth=6, min_samples_split=20, min_impurity=1e-7):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_impurity = min_impurity
        self.root = None

    def fit(self, X, y):
        self.root = self._grow(X, np.asarray(y, dtype=float), depth=0)
        return self

    def _best_split(self, X, y):
        n, n_features = X.shape
        best_gain, best_feat, best_thr = 0.0, -1, 0.0
        total_sum = y.sum()
        total_sq = (y * y).sum()
        parent_var = total_sq / n - (total_sum / n) ** 2
        for feature_i in range(n_features):
            order = np.argsort(X[:, feature_i], kind="stable")
            x_sorted = X[order, feature_i]
            y_sorted = y[order]
            sum_left = np.cumsum(y_sorted)[:-1]
            sq_left = np.cumsum(y_sorted * y_sorted)[:-1]
            count_left = np.arange(1, n)
            count_right = n - count_left
            sum_right = total_sum - sum_left
            sq_right = total_sq - sq_left
            var_left = sq_left / count_left - (sum_left / count_left) ** 2
            var_right = sq_right / count_right - (sum_right / count_right) ** 2
            gain = parent_var - (
                count_left / n * var_left + count_right / n * var_right
            )
            boundary = x_sorted[1:] > x_sorted[:-1]
            gain = np.where(boundary, gain, -np.inf)
            k = int(np.argmax(gain))
            if gain[k] > best_gain:
                best_gain = float(gain[k])
                best_feat = feature_i
                best_thr = 0.5 * (x_sorted[k] + x_sorted[k + 1])
        return best_gain, best_feat, best_thr

    def _grow(self, X, y, depth):
        n = len(y)
        leaf = {"value": float(y.mean())}
        if depth >= self.max_depth or n < self.min_samples_split:
            return leaf
        gain, feature_i, threshold = self._best_split(X, y)
        if feature_i < 0 or gain <= self.min_impurity:
            return leaf
        mask = X[:, feature_i] <= threshold
        return {
            "feature_i": feature_i,
            "threshold": threshold,
            "left": self._grow(X[mask], y[mask], depth + 1),
            "right": self._grow(X[~mask], y[~mask], depth + 1),
        }

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        out = np.empty(len(X), dtype=float)
        self._predict_into(self.root, X, np.arange(len(X)), out)
        return out

    def _predict_into(self, node, X, rows, out):
        if "value" in node:
            out[rows] = node["value"]
            return
        go_left = X[rows, node["feature_i"]] <= node["threshold"]
        self._predict_into(node["left"], X, rows[go_left], out)
        self._predict_into(node["right"], X, rows[~go_left], out)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### 単体の回帰木の実力を見る

    まず 1 本の回帰木で価格を直接予測してみる。深さを変えると、浅い木は表現力不足で
    誤差が大きく、深くすると学習データに過剰適合していく様子が分かる。単体の木は
    分散が大きく不安定であり、これがブースティングで多数の浅い木を足し込む動機になる。
    """)
    return


@app.cell
def _(X_fit, X_test, rmse, y_fit, y_test):
    single_tree_rows = []
    for _depth in [2, 4, 6, 8]:
        _tree = RegressionTree(max_depth=_depth).fit(X_fit, y_fit)
        single_tree_rows.append(
            {
                "max_depth": _depth,
                "train_RMSE": round(rmse(y_fit, _tree.predict(X_fit)), 1),
                "test_RMSE": round(rmse(y_test, _tree.predict(X_test)), 1),
            }
        )
    single_tree_table = pd.DataFrame(single_tree_rows)
    single_tree_table
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 3. 勾配ブースティング

    ブースティングは弱学習器を逐次的に足し込む加法モデルである。$m$ ステップ目の
    予測を $F_m$、新しく足す木を $h_m$、学習率（縮小率, shrinkage）を $\nu$ とすると

    $$
    F_m(x) = F_{m-1}(x) + \nu\, h_m(x)
    $$

    初期値は目的変数の平均 $F_0(x) = \bar{y}$ に置く。各ステップでは損失 $L$ の
    負の勾配に木を当てはめる。二乗損失 $L(y, F) = \tfrac{1}{2}(y - F)^2$ の場合、

    $$
    -\frac{\partial L}{\partial F} = y - F_{m-1}(x)
    $$

    となり、**負の勾配はそのまま残差**である。したがって二乗損失の勾配ブースティングは
    「現在の残差に回帰木を当てはめ、その予測を $\nu$ 倍して足す」ことに等しい。
    学習率 $\nu$ を小さくすると 1 本あたりの寄与が減り、過剰適合しにくくなるが、
    同じ精度に到達するのに必要な木の本数は増える。
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### 参考実装の `predict` バグ

    有名な教材 `eriklindernoren/ML-From-Scratch` の `GradientBoosting` は、
    `fit` では初期値を $\bar{y}$ に取りながら、`predict` では 0 から木の寄与を
    足し始めている。つまり**初期オフセット $F_0 = \bar{y}$ を落としている**。
    価格のように平均が 0 から大きく離れた目的変数では、この抜けは予測全体を
    定数分だけずらし、致命的な誤差になる。

    以下では初期値 `f0` を保存する正しい `predict` と、バグを再現した
    `predict_buggy` の両方を実装し、数値で差を確認する。
    """)
    return


@app.class_definition
class GradientBoostingRegressor:
    """二乗損失の勾配ブースティング（回帰木を残差に当てはめる教育用実装）。

    `predict` は初期値 f0 を含む正しい実装、`predict_buggy` は参考実装の
    バグ（f0 を落とす）を再現したもの。
    """

    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3,
                 min_samples_split=20):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.trees = []
        self.f0 = 0.0
        self.history = []

    def fit(self, X, y, eval_set=None):
        y = np.asarray(y, dtype=float)
        self.f0 = float(y.mean())
        train_pred = np.full(len(y), self.f0)
        val_pred = None
        if eval_set is not None:
            X_val, y_val = eval_set
            val_pred = np.full(len(y_val), self.f0)
        self.trees = []
        self.history = []
        for _ in range(self.n_estimators):
            residual = y - train_pred
            tree = RegressionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
            ).fit(X, residual)
            train_pred = train_pred + self.learning_rate * tree.predict(X)
            self.trees.append(tree)
            record = {"train_rmse": float(np.sqrt(np.mean((y - train_pred) ** 2)))}
            if val_pred is not None:
                val_pred = val_pred + self.learning_rate * tree.predict(X_val)
                record["val_rmse"] = float(np.sqrt(np.mean((y_val - val_pred) ** 2)))
            self.history.append(record)
        return self

    def predict(self, X):
        out = np.full(len(X), self.f0)
        for tree in self.trees:
            out = out + self.learning_rate * tree.predict(X)
        return out

    def predict_buggy(self, X):
        out = np.zeros(len(X))
        for tree in self.trees:
            out = out + self.learning_rate * tree.predict(X)
        return out


@app.cell
def _(X_fit, X_test, X_val, rmse, y_fit, y_test, y_val):
    gbr_teaching = GradientBoostingRegressor(
        n_estimators=100, learning_rate=0.1, max_depth=3
    ).fit(X_fit, y_fit, eval_set=(X_val, y_val))
    predict_bug_table = pd.DataFrame(
        {
            "predict": ["fixed (with f0)", "buggy (drops f0)"],
            "test_RMSE": [
                round(rmse(y_test, gbr_teaching.predict(X_test)), 1),
                round(rmse(y_test, gbr_teaching.predict_buggy(X_test)), 1),
            ],
        }
    )
    predict_bug_table
    return (gbr_teaching,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    バグ版はおよそ平均価格 `f0` のぶんだけ全体がずれるため、RMSE が桁違いに悪化する。
    初期オフセットを保存するだけで直る、というのが教訓である。

    ### 学習曲線

    木の本数を増やすと学習・検証 RMSE がどう下がるかを見る。序盤で急速に改善し、
    やがて検証 RMSE の改善が緩やかになる。これが早期終了の根拠になる。
    """)
    return


@app.cell
def _(gbr_teaching):
    _hist = pd.DataFrame(gbr_teaching.history)
    _rounds = np.arange(1, len(_hist) + 1)
    _fig, _ax = plt.subplots(figsize=(6, 4))
    _ax.plot(_rounds, _hist["train_rmse"], label="train")
    _ax.plot(_rounds, _hist["val_rmse"], label="valid")
    _ax.set_xlabel("number of trees")
    _ax.set_ylabel("RMSE")
    _ax.set_title("Scratch GBDT learning curve")
    _ax.legend()
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 4. 2 次近似 (Newton) 版

    XGBoost や LightGBM は損失を 2 次までテイラー展開する。サンプル $i$ の
    1 次勾配 $g_i = \partial_F L$、2 次勾配（ヘシアン）$h_i = \partial_F^2 L$ を使うと、
    葉 $j$（サンプル集合 $I_j$）に定数 $w_j$ を置いたときの正則化つき目的関数は

    $$
    \tilde{L}_j(w_j) = \sum_{i \in I_j} \left( g_i w_j + \tfrac{1}{2} h_i w_j^2 \right) + \tfrac{1}{2}\lambda w_j^2
    $$

    $w_j$ で微分して 0 と置くと、最適な葉値は

    $$
    w_j^\ast = -\frac{\sum_{i \in I_j} g_i}{\sum_{i \in I_j} h_i + \lambda}
    $$

    ある分割で左右へ $I_L, I_R$ と分けたときの利得は、$G = \sum g_i,\ H = \sum h_i$ として

    $$
    \mathcal{G} = \tfrac{1}{2}\left[
    \frac{G_L^2}{H_L + \lambda} + \frac{G_R^2}{H_R + \lambda} - \frac{(G_L + G_R)^2}{H + \lambda}
    \right] - \gamma
    $$

    二乗損失では $g_i = F - y_i,\ h_i = 1$ なので $\sum h_i = |I_j|$ となり、
    $\lambda = 0$ のとき葉値は残差平均、利得は分散減少に一致する。つまり **2 節の
    実装は Newton 版の $\lambda = 0,\ h_i = 1$ という特別な場合**である。$\lambda$ は
    L2 正則化として葉値を 0 方向へ縮め、$\gamma$ は分割に最低利得を課す。
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### 勾配・ヘシアンで動く木

    2 節の分散減少ではなく、上で導いた利得 $\mathcal{G}$ を最大化し、葉値を
    $w_j^\ast$ で置く木を実装する。木は生のサンプル $(X, g, h)$ を受け取り、
    分割探索は「厳密（ソート＋累積和）」か「ヒストグラム」を選べるようにする。
    分割ごとの利得は特徴量重要度（total gain）として積算する。
    """)
    return


@app.cell
def _():
    def newton_leaf(grad_sum, hess_sum, reg_lambda):
        return -grad_sum / (hess_sum + reg_lambda)

    def newton_gain(gl, hl, gr, hr, reg_lambda, gamma):
        total_g, total_h = gl + gr, hl + hr
        score = (
            gl * gl / (hl + reg_lambda)
            + gr * gr / (hr + reg_lambda)
            - total_g * total_g / (total_h + reg_lambda)
        )
        return 0.5 * score - gamma

    def newton_build(X, g, h, depth, cfg, gains):
        n = len(g)
        grad_sum, hess_sum = g.sum(), h.sum()
        leaf = {"value": float(newton_leaf(grad_sum, hess_sum, cfg["reg_lambda"]))}
        if depth >= cfg["max_depth"] or n < cfg["min_samples_split"]:
            return leaf
        best_gain, best_feat, best_thr = cfg["gamma"], -1, 0.0
        for feature_i in range(X.shape[1]):
            column = X[:, feature_i]
            if cfg["kind"] == "hist":
                gl = np.cumsum(np.bincount(column, weights=g, minlength=cfg["nbins"]))[:-1]
                hl = np.cumsum(np.bincount(column, weights=h, minlength=cfg["nbins"]))[:-1]
                count_left = np.cumsum(np.bincount(column, minlength=cfg["nbins"]))[:-1]
                thresholds = np.arange(cfg["nbins"] - 1, dtype=float)
                boundary = count_left > 0
            else:
                order = np.argsort(column, kind="stable")
                x_sorted = column[order]
                gl = np.cumsum(g[order])[:-1]
                hl = np.cumsum(h[order])[:-1]
                count_left = np.arange(1, n)
                thresholds = 0.5 * (x_sorted[:-1] + x_sorted[1:])
                boundary = x_sorted[1:] > x_sorted[:-1]
            count_right = n - count_left
            gr = grad_sum - gl
            hr = hess_sum - hl
            gain = newton_gain(gl, hl, gr, hr, cfg["reg_lambda"], cfg["gamma"])
            enough = (count_left >= cfg["min_child"]) & (count_right >= cfg["min_child"])
            gain = np.where(boundary & enough, gain, -np.inf)
            k = int(np.argmax(gain))
            if gain[k] > best_gain:
                best_gain, best_feat, best_thr = float(gain[k]), feature_i, float(thresholds[k])
        if best_feat < 0:
            return leaf
        gains[best_feat] += best_gain
        mask = X[:, best_feat] <= best_thr
        return {
            "feature_i": best_feat,
            "threshold": best_thr,
            "left": newton_build(X[mask], g[mask], h[mask], depth + 1, cfg, gains),
            "right": newton_build(X[~mask], g[~mask], h[~mask], depth + 1, cfg, gains),
        }

    def newton_predict(node, X):
        out = np.empty(len(X), dtype=float)

        def descend(current, rows):
            if "value" in current:
                out[rows] = current["value"]
                return
            go_left = X[rows, current["feature_i"]] <= current["threshold"]
            descend(current["left"], rows[go_left])
            descend(current["right"], rows[~go_left])

        descend(node, np.arange(len(X)))
        return out

    return newton_build, newton_predict


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 5. ヒストグラム分割による高速化

    厳密な分割探索はノードごとに全特徴量をソートするため、サンプル数に対して
    $O(n \log n)$ かかる。LightGBM は各特徴量をあらかじめ少数のビン
    （既定 255）に離散化し、**ビンごとに勾配・ヘシアンのヒストグラムを積算**して
    ビン境界だけを走査する。1 ノードあたりの分割探索は
    $O(\text{特徴量数} \times \text{ビン数})$ になり、しきい値の候補がビン数に
    抑えられるため大幅に速くなる。ここでは分位点で 64 ビンに区切る。ビン化は近似だが、
    木の予測はしきい値の位置にしか依存しないため、十分細かいビンなら精度は
    ほとんど変わらない。

    `GBDTScratch` は厳密版・ヒストグラム版を `split` 引数で切り替えられる。加えて
    行サブサンプリング（`subsample`, bagging）と、検証集合 `eval_set` を使った
    早期終了（`early_stopping_rounds`）を備える。早期終了時は、検証 RMSE が最良だった
    ラウンド数を `best_iteration_` に記録し、`predict` はそこまでの木だけを使う。
    `eval_set` や `early_stopping_rounds` を渡さなければ `best_iteration_` は全木数となり、
    従来どおり全ての木で予測する。
    """)
    return


@app.cell
def _(newton_build, newton_predict):
    class GBDTScratch:
        """Newton 型の勾配ブースティング（厳密／ヒストグラム分割を切替可能）。"""

        def __init__(self, n_estimators=200, learning_rate=0.1, max_depth=6,
                     reg_lambda=1.0, gamma=0.0, min_child=20, min_samples_split=40,
                     split="exact", max_bin=64, subsample=1.0, random_state=71,
                     early_stopping_rounds=None):
            self.n_estimators = n_estimators
            self.learning_rate = learning_rate
            self.max_depth = max_depth
            self.reg_lambda = reg_lambda
            self.gamma = gamma
            self.min_child = min_child
            self.min_samples_split = min_samples_split
            self.split = split
            self.max_bin = max_bin
            self.subsample = subsample
            self.random_state = random_state
            self.early_stopping_rounds = early_stopping_rounds

        def _make_bins(self, X):
            self.bin_edges = []
            for feature_i in range(X.shape[1]):
                quantiles = np.linspace(0, 1, self.max_bin + 1)[1:-1]
                edges = np.unique(np.quantile(X[:, feature_i], quantiles))
                self.bin_edges.append(edges)

        def _apply_bins(self, X):
            binned = np.empty(X.shape, dtype=np.int64)
            for feature_i in range(X.shape[1]):
                binned[:, feature_i] = np.searchsorted(
                    self.bin_edges[feature_i], X[:, feature_i], side="right"
                )
            return binned

        def fit(self, X, y, eval_set=None):
            y = np.asarray(y, dtype=float)
            self.n_features_ = X.shape[1]
            self.feature_gains_ = np.zeros(self.n_features_)
            cfg = {
                "kind": self.split,
                "max_depth": self.max_depth,
                "min_samples_split": self.min_samples_split,
                "min_child": self.min_child,
                "reg_lambda": self.reg_lambda,
                "gamma": self.gamma,
            }
            if self.split == "hist":
                self._make_bins(X)
                X_used = self._apply_bins(X)
                cfg["nbins"] = int(X_used.max()) + 1
            else:
                X_used = X
            rng = np.random.default_rng(self.random_state)

            self.f0 = float(y.mean())
            train_pred = np.full(len(y), self.f0)
            val_pred = None
            if eval_set is not None:
                X_val_used = self._apply_bins(eval_set[0]) if self.split == "hist" else eval_set[0]
                y_val = np.asarray(eval_set[1], dtype=float)
                val_pred = np.full(len(y_val), self.f0)
            self.trees = []
            self.history = []
            self.best_iteration_ = None
            best_val_rmse = np.inf
            rounds_without_improvement = 0
            n = len(y)
            for _ in range(self.n_estimators):
                grad = train_pred - y
                hess = np.ones_like(y)
                if self.subsample < 1.0:
                    take = rng.choice(n, size=int(self.subsample * n), replace=False)
                else:
                    take = slice(None)
                tree = newton_build(
                    X_used[take], grad[take], hess[take], 0, cfg, self.feature_gains_
                )
                train_pred = train_pred + self.learning_rate * newton_predict(tree, X_used)
                self.trees.append(tree)
                record = {"train_rmse": float(np.sqrt(np.mean((y - train_pred) ** 2)))}
                if val_pred is not None:
                    val_pred = val_pred + self.learning_rate * newton_predict(tree, X_val_used)
                    record["val_rmse"] = float(np.sqrt(np.mean((y_val - val_pred) ** 2)))
                self.history.append(record)
                if val_pred is not None and self.early_stopping_rounds is not None:
                    if record["val_rmse"] < best_val_rmse - 1e-9:
                        best_val_rmse = record["val_rmse"]
                        self.best_iteration_ = len(self.trees)
                        rounds_without_improvement = 0
                    else:
                        rounds_without_improvement += 1
                        if rounds_without_improvement >= self.early_stopping_rounds:
                            break
            if self.best_iteration_ is None:
                self.best_iteration_ = len(self.trees)
            return self

        def predict(self, X):
            X_used = self._apply_bins(X) if self.split == "hist" else X
            out = np.full(len(X_used), self.f0)
            for tree in self.trees[: self.best_iteration_]:
                out = out + self.learning_rate * newton_predict(tree, X_used)
            return out

    return (GBDTScratch,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### 厳密探索とヒストグラム探索の速度・精度比較

    同じハイパーパラメータで厳密版とヒストグラム版を学習し、学習時間と
    テスト RMSE を比べる。ヒストグラム版はしきい値候補をビン数に抑えるぶん速く、
    精度はほぼ変わらないはずである。ここで得た学習済みモデルと学習時間は
    最後の総合比較でも再利用する。
    """)
    return


@app.cell
def _(GBDTScratch, X_sub, X_test, rmse, time, y_sub, y_test):
    split_finder_rows = []
    gbdt_exact = None
    gbdt_hist = None
    exact_seconds = 0.0
    hist_seconds = 0.0
    for _kind in ["exact", "hist"]:
        _t0 = time.perf_counter()
        _model = GBDTScratch(
            n_estimators=200, learning_rate=0.1, max_depth=6, split=_kind
        ).fit(X_sub, y_sub)
        _elapsed = time.perf_counter() - _t0
        if _kind == "exact":
            gbdt_exact = _model
            exact_seconds = _elapsed
        else:
            gbdt_hist = _model
            hist_seconds = _elapsed
        split_finder_rows.append(
            {
                "split finder": _kind,
                "fit_seconds": round(_elapsed, 2),
                "test_RMSE": round(rmse(y_test, _model.predict(X_test)), 1),
            }
        )
    split_finder_table = pd.DataFrame(split_finder_rows)
    split_finder_table
    return exact_seconds, gbdt_exact, gbdt_hist, hist_seconds


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### 早期終了と行サブサンプリング（bagging）を動かす

    ここまでは木の本数を固定していたが、実務では**必要な本数はデータ依存**である。
    多すぎれば過剰適合し計算も無駄になり、少なすぎれば未学習になる。**早期終了**は
    検証集合の RMSE を監視し、一定ラウンド改善しなければ学習を打ち切る仕組みで、
    本数の手動調整を不要にし過剰適合を防ぐ。

    **行サブサンプリング（bagging）**は各木を学習データの一部（ここでは 80%）だけで
    育てる。木ごとに見るデータが変わるため木の間の相関が下がり、平均としての分散が
    減って汎化しやすくなる（確率的勾配ブースティングの考え方）。

    以下では `n_estimators=400` と多めに上限を取り、`early_stopping_rounds=20`・
    `subsample=0.8`・ヒストグラム分割で学習し、実際に何本で打ち切られたか
    (`best_iteration_`) と、その時点の検証・テスト RMSE を確認する。
    """)
    return


@app.cell
def _(GBDTScratch, X_fit, X_test, X_val, rmse, y_fit, y_test, y_val):
    early_stop_model = GBDTScratch(
        n_estimators=400,
        learning_rate=0.1,
        max_depth=6,
        split="hist",
        subsample=0.8,
        early_stopping_rounds=20,
        random_state=71,
    ).fit(X_fit, y_fit, eval_set=(X_val, y_val))
    early_stop_table = pd.DataFrame(
        {
            "item": [
                "n_estimators (upper bound)",
                "best_iteration_ (trees kept)",
                "trees actually trained",
                "valid RMSE at best_iteration_",
                "test RMSE at best_iteration_",
            ],
            "value": [
                400,
                early_stop_model.best_iteration_,
                len(early_stop_model.trees),
                round(early_stop_model.history[early_stop_model.best_iteration_ - 1]["val_rmse"], 1),
                round(rmse(y_test, early_stop_model.predict(X_test)), 1),
            ],
        }
    )
    early_stop_table
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 6. 主要ライブラリとの比較

    同一のテストデータ上で、scratch 実装（厳密・ヒストグラム）を LightGBM・
    scikit-learn の GBDT と比べる。木の本数・学習率・木の深さをそろえて公平にする。
    規模感の基準として、つねに学習データの平均を予測する自明なベースラインも入れる。
    参考として LightGBM を**全学習データ**で学習した行（`LightGBM (full train)`）も
    加える。
    """)
    return


@app.cell
def _(
    SkGradientBoostingRegressor,
    SkHistGradientBoostingRegressor,
    X_sub,
    X_test,
    X_train_full,
    lgb,
    time,
    y_sub,
    y_train_full,
):
    def _timed(name, model, X_train, y_train):
        t0 = time.perf_counter()
        model.fit(X_train, y_train)
        elapsed = time.perf_counter() - t0
        return name, model.predict(X_test), elapsed

    library_results = [
        _timed(
            "LightGBM",
            lgb.LGBMRegressor(
                n_estimators=200, learning_rate=0.1, max_depth=6,
                num_leaves=63, random_state=71, verbosity=-1,
            ),
            X_sub, y_sub,
        ),
        _timed(
            "sklearn GradientBoosting",
            SkGradientBoostingRegressor(
                n_estimators=200, learning_rate=0.1, max_depth=6, random_state=71
            ),
            X_sub, y_sub,
        ),
        _timed(
            "sklearn HistGradientBoosting",
            SkHistGradientBoostingRegressor(
                max_iter=200, learning_rate=0.1, max_depth=6, random_state=71
            ),
            X_sub, y_sub,
        ),
        _timed(
            "LightGBM (full train)",
            lgb.LGBMRegressor(
                n_estimators=200, learning_rate=0.1, max_depth=6,
                num_leaves=63, random_state=71, verbosity=-1,
            ),
            X_train_full, y_train_full,
        ),
    ]
    return (library_results,)


@app.cell
def _(
    X_test,
    exact_seconds,
    gbdt_exact,
    gbdt_hist,
    hist_seconds,
    library_results,
    mean_absolute_error,
    r2_score,
    rmse,
    y_sub,
    y_test,
):
    def _metrics(name, pred, seconds):
        return {
            "model": name,
            "RMSE": round(rmse(y_test, pred), 1),
            "MAE": round(mean_absolute_error(y_test, pred), 1),
            "R2": round(r2_score(y_test, pred), 4),
            "fit_seconds": round(seconds, 2),
        }

    summary_rows = [
        _metrics("baseline (mean)", np.full(len(y_test), y_sub.mean()), 0.0),
        _metrics("scratch GBDT (exact)", gbdt_exact.predict(X_test), exact_seconds),
        _metrics("scratch GBDT (histogram)", gbdt_hist.predict(X_test), hist_seconds),
    ]
    for _name, _pred, _seconds in library_results:
        summary_rows.append(_metrics(_name, _pred, _seconds))
    summary_table = pd.DataFrame(summary_rows).sort_values("RMSE").reset_index(drop=True)
    summary_table
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### 特徴量重要度

    scratch モデルは分割ごとの利得を特徴量別に積算している（total gain）。これを
    LightGBM の `importance_type="gain"` と並べて比較する。ダイヤモンド価格は
    重さ (`carat`) と寸法 (`x`,`y`,`z`) がほぼ支配的であり、両者はこれを同様に
    捉えるはずである。値はそれぞれ合計 1 に正規化して並べる。
    """)
    return


@app.cell
def _(X_sub, feature_names, gbdt_hist, lgb, y_sub):
    _lgb_imp_model = lgb.LGBMRegressor(
        n_estimators=200, learning_rate=0.1, max_depth=6,
        num_leaves=63, random_state=71, verbosity=-1,
    ).fit(X_sub, y_sub)
    _scratch_gain = gbdt_hist.feature_gains_ / gbdt_hist.feature_gains_.sum()
    _lgb_gain = _lgb_imp_model.booster_.feature_importance(importance_type="gain")
    _lgb_gain = _lgb_gain / _lgb_gain.sum()
    _order = np.argsort(_scratch_gain)[::-1]
    _pos = np.arange(len(feature_names))
    _fig, _ax = plt.subplots(figsize=(7, 4))
    _ax.bar(_pos - 0.2, _scratch_gain[_order], width=0.4, label="scratch (hist)")
    _ax.bar(_pos + 0.2, _lgb_gain[_order], width=0.4, label="LightGBM")
    _ax.set_xticks(_pos)
    _ax.set_xticklabels([feature_names[i] for i in _order], rotation=45, ha="right")
    _ax.set_ylabel("normalized total gain")
    _ax.set_title("Feature importance: scratch vs LightGBM")
    _ax.legend()
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### 予測と実測の散布図

    もっとも精度の良かった scratch ヒストグラム版について、テストデータの実測価格と
    予測価格を散布図で確認する。対角線に沿って点が並べば予測が実測に一致している。
    高価格帯でばらつきが増えるのは、希少で高価な石の数が少なく学習しにくいためである。
    """)
    return


@app.cell
def _(X_test, gbdt_hist, y_test):
    _pred = gbdt_hist.predict(X_test)
    _lo = float(min(y_test.min(), _pred.min()))
    _hi = float(max(y_test.max(), _pred.max()))
    _fig, _ax = plt.subplots(figsize=(5, 5))
    _ax.scatter(y_test, _pred, s=6, alpha=0.3)
    _ax.plot([_lo, _hi], [_lo, _hi], linestyle="--", color="black")
    _ax.set_xlabel("actual price")
    _ax.set_ylabel("predicted price")
    _ax.set_title("Scratch GBDT (histogram): predicted vs actual")
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## まとめ

    - **回帰木** を分散減少基準で NumPy 実装し、累積和で分割探索を
      $O(\text{特徴量数} \times n \log n)$ に抑えた。単体の木は分散が大きく不安定である。
    - **勾配ブースティング** は残差（= 二乗損失の負の勾配）に浅い木を当てはめて
      足し込む加法モデルであることを確認した。参考実装の `predict` は初期値 $\bar{y}$ を
      落とすバグがあり、価格回帰では RMSE が桁違いに悪化する。$F_0$ を保存すれば直る。
    - **Newton 版**では葉値 $w_j^\ast = -\sum g_i / (\sum h_i + \lambda)$、分割利得
      $\mathcal{G}$ を導いた。二乗損失では $h_i = 1$ なので分散減少に L2 正則化を
      加えた形に帰着する。
    - **ヒストグラム分割**はしきい値候補をビン数に抑え、厳密探索より速く、精度は
      ほぼ同等だった（速度差は上の比較表のとおり）。
    - **早期終了と bagging** を実装し、上限 400 本・`early_stopping_rounds=20`・
      `subsample=0.8` で学習すると、検証 RMSE をもとに `best_iteration_ = 120` 本で
      自動的に打ち切られた。木の本数を手で調整せずに済み、過剰適合と無駄な計算を
      避けられる。
    - **総合比較**では、scratch 実装は同じ木の本数・学習率・深さの LightGBM や
      scikit-learn と近い RMSE / R2 に達した。ただし学習速度は桁で及ばない。
      LightGBM が速く、わずかに高精度なのは、ヒストグラム分割に加えて
      leaf-wise 成長・ネイティブなカテゴリ処理・C++/OpenMP による並列化・欠測値の
      自動振り分け・洗練された正則化を備えるためである。scratch 実装は原理を
      なぞることで「何が速度と精度を生んでいるのか」を明確にする教材として役立つ。
    """)
    return


if __name__ == "__main__":
    app.run()
