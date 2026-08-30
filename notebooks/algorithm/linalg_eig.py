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
    # 固有値問題と特異値分解

    行列の固有値を数値的に求める方法を、素朴なべき乗法から実用的な QR 法まで
    順に実装する。5 次以上の代数方程式には解の公式が無いため、固有値計算は
    本質的に反復法になる。後半では前処理（ハウスホルダー変換・ギブンス回転）
    による高速化と、固有値問題に帰着させる特異値分解を扱う。
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 固有値計算
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    アーベル・ルフィニの定理より5次以上の代数方程式には公式が存在しない。よって5x5行列の特性方程式は直接的に解けないため、反復法を使用する。
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    このノートブックでは、数値計算の流れを追いやすくするために **4x4 / 4x5 の代表例** と
    **小さめの反復上限** を使う。特に後半のハウスホルダー変換 + ギブンス回転の節では、
    各アクティブブロックの QR 反復回数を上限付きにして、HTML エクスポートでも
    実行時間が発散しないようにしている。
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### べき乗法
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ランダム行列を作成
    """)
    return


@app.cell
def _():
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns

    # 表示の精度を設定
    np.set_printoptions(precision=3, suppress=False)
    plt.rcParams['figure.figsize'] = (5, 2)
    np.random.seed(42)

    N = 4

    # A = np.random.rand(N, N)
    # A = A.T @ A
    A = np.array([
        [16., -1., 1., 2.],
        [2., 12., 1., -1.],
        [1., 3., -24., 2.],
        [4., -2., 1., 20.]
    ])

    print("A=")
    sns.heatmap(A, annot=True, cmap='coolwarm')
    plt.show()
    return A, N, np, plt, sns


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    べき乗法は行列 $A$ のべき乗を初期ベクトルに作用させて、絶対値最大の固有値と
    その固有ベクトルを求める方法である。

    $A$ の固有ベクトル $\vec{v}_i$ が基底をなすとき、初期ベクトルは
    $$
    \vec{x}^{(0)}=\sum_i c_i\vec{v}_i
    $$
    と展開できる。これに $A$ を $k$ 回作用させると
    $$
    A^k\vec{x}^{(0)}=\lambda_1^k\left(c_1\vec{v}_1
    +\sum_{i\geq2}c_i\left(\frac{\lambda_i}{\lambda_1}\right)^k\vec{v}_i\right)
    $$
    となる。$|\lambda_1|>|\lambda_2|\geq\cdots$ であれば第 2 項以降は
    $|\lambda_2/\lambda_1|^k$ で減衰するので、$A^k\vec{x}^{(0)}$ の向きは $\vec{v}_1$ に収束する。
    これが $A(A^{\infty}\vec{x})=\lambda(A^{\infty}\vec{x})$ となることの中身である。

    前提は次の 2 つ。

    1. 絶対値最大の固有値がただ 1 つであること。縮退していたり複素共役対だったりすると向きが定まらない。
    2. 初期ベクトルが $c_1\neq0$ を満たすこと。乱数で選べばほぼ確実に満たされる。

    収束は線形で、比 $|\lambda_2/\lambda_1|$ が 1 に近いほど遅い。

    実装では毎回 $\vec{x}$ を正規化するため、固有値の推定にはレイリー商
    $\lambda\simeq(\vec{x},A\vec{x})$ を使う。停止条件の `np.dot(v, v) - lambda_ ** 2` は
    $\|\vec{x}\|=1$ のとき
    $$
    \|A\vec{x}-\lambda\vec{x}\|^2=\|A\vec{x}\|^2-\lambda^2
    $$
    に一致するので、残差ノルムの 2 乗を見ていることになる。
    """)
    return


@app.cell
def _(A, N, np):
    def power_iteration(A_, eps=1e-6, max_iter=100):
      A = A_.copy()
      x = (_ := np.random.rand(N)) / np.linalg.norm(_)
      lambda_ = 0
      for i in range(max_iter):
        # べき乗：反復毎に最大固有値の固有ベクトル成分が大きく
        v = A @ x
        # 最終的に(x,Ax)=(x,λx)=λとなるはず
        lambda_ = np.dot(x, v)
        x = v / np.linalg.norm(v)
        if np.dot(v, v) - lambda_ ** 2 < eps:
          print(f"early stop: {i=}")
          break
      return lambda_, x

    lambda_, x = power_iteration(A)

    print(f"{lambda_=:.3}")
    print(f"{x=}")
    print(f"{A@x-lambda_*x=}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### QR 分解
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    最も基本的な QR 分解はグラム・シュミットの直交化を利用
    """)
    return


@app.cell
def _(A, np, plt, sns):
    def gram_schmidt(vec_in_):
        vec_in = vec_in_.copy()
        vec_out = np.zeros_like(vec_in)
        for k in range(vec_in.shape[1]):
            u = vec_in[:, k]
            for j in range(k):
                u = u - np.dot(vec_in[:, k], vec_out[:, j]) * vec_out[:, j]
            vec_out[:, k] = u / np.linalg.norm(u)
        return vec_out
    _Q = gram_schmidt(A)
    print('Q @ Q.T=')
    sns.heatmap(_Q @ _Q.T, annot=True, cmap='coolwarm')
    plt.show()
    return (gram_schmidt,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    対象の正則行列$A$の各列を一次独立なベクトルとみなして、グラム・シュミットで直交化。

    直交化したベクトルを並べたものを行列$Q$として、$A=QR$となるように行列$R$を構成。

    $R$ の成分は $Q$ の列 $\vec{q}_j$ と $A$ の列 $\vec{a}_k$ から
    $$
    R_{jk}=(\vec{q}_j,\vec{a}_k)\quad(j<k),\qquad
    R_{kk}=\left\|\vec{a}_k-\sum_{j<k}(\vec{q}_j,\vec{a}_k)\vec{q}_j\right\|
    $$
    で決まる。グラム・シュミットの構成上 $\vec{a}_k$ は
    $\vec{q}_1,\dots,\vec{q}_k$ の張る空間に含まれるので、$j>k$ では $R_{jk}=0$、
    つまり $R$ は上三角行列になる。

    なおここで実装した古典グラム・シュミットは、列がほぼ平行なとき桁落ちで直交性を失う。
    実用のライブラリでは修正グラム・シュミットか、後述のハウスホルダー変換を使う。
    """)
    return


@app.cell
def _(A, gram_schmidt, np, plt, sns):
    def QR_decomp(A_):
        A = A_.copy()
        _Q = gram_schmidt(A)
        _R = np.zeros_like(A)
        for j in range(len(A)):
            for k in range(j, len(A)):
                if j == k:
                    a_k = A[:, k]
                    for col_idx in range(j):
                        a_k = a_k - np.dot(A[:, k], _Q[:, col_idx]) * _Q[:, col_idx]
                    _R[j, k] = np.linalg.norm(a_k)
                else:
                    _R[j, k] = np.dot(A[:, k], _Q[:, j])
        return (_Q, _R)
    _Q, _R = QR_decomp(A)
    print('Q=')
    sns.heatmap(_Q, annot=True, cmap='coolwarm')
    plt.show()
    print('R=')
    sns.heatmap(_R, annot=True, cmap='coolwarm')
    plt.show()
    print('A-QR=')
    sns.heatmap(A - _Q @ _R, annot=True, cmap='coolwarm')
    plt.show()
    return (QR_decomp,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    QR 法は QR 分解を繰り返して固有値を対角成分に持つ行列を作成。

    $A_k=Q_kR_k, A_{k+1}=R_kQ_k=Q_k^{-1}A_kQ_k=Q_k^TA_kQ_k$を繰り返して、直交変換。

    各ステップは直交行列による相似変換なので、固有値は変わらない。
    固有値がすべて実で絶対値が異なる（$|\lambda_1|>\cdots>|\lambda_n|$）とき、
    $A_k$ の下三角成分は
    $$
    (A_k)_{ij}=O\left(\left|\frac{\lambda_i}{\lambda_j}\right|^k\right)\quad(i>j)
    $$
    で減衰し、$A_k$ は固有値を絶対値の大きい順に対角へ並べた上三角行列へ収束する。
    これは互いに直交する部分空間でべき乗法を同時に実行していることに相当する。

    絶対値が等しい固有値や複素共役対があると 2x2 のブロックが対角に残る（実シューア分解）。
    また収束は隣り合う固有値の比 $|\lambda_{i+1}/\lambda_i|$ で決まる線形収束なので、
    比が 1 に近いと非常に遅い。次節の原点シフトはこの比を人為的に小さくする工夫である。
    """)
    return


@app.cell
def _(A, QR_decomp, np, plt, sns):
    def QR_iteration(A_, eps=1e-06, max_iter=100):
        A = A_.copy()
        for i in range(max_iter):
            _Q, _R = QR_decomp(A)
            A = _R @ _Q
            if np.tril(np.abs(A), -1).max() < eps:
                print(f'early stop: i={i!r}')
                break
        return A
    diag_A = QR_iteration(A)
    print('diag_A=')
    sns.heatmap(diag_A, annot=True, cmap='coolwarm')
    plt.show()
    return QR_iteration, diag_A


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    numpy の結果と突き合わせて検算する。ここで扱う $A$ は非対称なので、
    対称行列専用の `np.linalg.eigh`（下三角しか参照しない）ではなく `np.linalg.eig` を使う。

    また反復法が返す固有値の並び順は手法ごとに違うため、
    比較の際は numpy 側を反復法の並び順に対応付けてから差を取る。
    """)
    return


@app.cell
def _(A, diag_A, np):
    def ordered_eig(A_, reference=None):
        # A は非対称なので、対称行列専用の eigh ではなく eig を使う
        evalues, evectors = np.linalg.eig(A_)
        if reference is None:
            idx = np.abs(evalues).argsort()[::-1]  # 固有値の絶対値が大きい順に並べる
        else:
            # 反復法が返す並び順に、最も近い固有値を 1 対 1 で対応付ける
            rest = list(range(len(evalues)))
            idx = []
            for value in reference:
                nearest = min(rest, key=lambda j: abs(evalues[j] - value))
                rest.remove(nearest)
                idx.append(nearest)
            idx = np.array(idx)
        return (evalues[idx], evectors[:, idx])
    _evalues_np, _ = ordered_eig(A, np.diag(diag_A))
    print(f'np.diag(diag_A)={np.diag(diag_A)!r}')
    print(f'evalues_np={_evalues_np!r}')
    print(f'residuals={np.diag(diag_A) - _evalues_np}')
    return (ordered_eig,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 原点シフト
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    QR 法の収束速度は隣り合う固有値の比 $|\lambda_{i+1}/\lambda_i|$ で決まる。
    そこで原点をずらす量（シフト）$\mu_k$ を導入し
    $$
    A_k-\mu_k I=Q_kR_k,\quad
    A_{k+1}=R_kQ_k+\mu_kI
    $$
    と反復する。$A_{k+1}=Q_k^T A_k Q_k$ は変わらないので固有値は保存されるが、
    収束を支配する比は
    $$
    \left|\frac{\lambda_{i+1}-\mu_k}{\lambda_i-\mu_k}\right|
    $$
    に置き換わる。$\mu_k$ をどれか 1 つの固有値に近づけるほどこの比は小さくなり、
    右下隅の劣対角成分が急速に 0 へ向かう。

    ウィルキンソンの移動法は$A_k$右下隅の2x2小行列の固有値のうち、
    $A_k$の右下隅の値に近いほうを$\mu_k$として利用。
    2x2 行列の固有値は特性方程式
    $$
    \lambda^2-(\mathrm{tr}A)\lambda+\det A=0
    $$
    から
    $$
    \lambda=\frac{\mathrm{tr}A\pm\sqrt{(\mathrm{tr}A)^2-4\det A}}{2}
    $$
    と閉じた形で書ける。次のセルの `minimal_eigh` はこの公式そのもので、
    `b` がトレース、`c` が行列式、`D` が判別式に対応する。
    判別式が負（複素共役対）の場合は実数の範囲では扱えないため、
    ここでは固有値がすべて実である行列を前提にしている。
    """)
    return


@app.cell
def _(A, np):
    def minimal_eigh(A):
      if A.shape != (2, 2):
        raise ValueError("A must be 2x2 matrix")
      b = np.trace(A)
      c = np.linalg.det(A)
      D = b**2-4*c
      return np.array([(b+np.sqrt(D))/2, (b-np.sqrt(D))/2])

    def wilkinson_shift(A):
      evalues = minimal_eigh(A[-2:, -2:])
      idx = np.argmin(np.abs(evalues-A[-1,-1]))
      return evalues[idx]

    print(f"{wilkinson_shift(A)=:.2f}")
    return (wilkinson_shift,)


@app.cell
def _(A, QR_decomp, np, ordered_eig, plt, sns, wilkinson_shift):
    def shifted_QR_iteration(A_, eps=1e-06, max_iter=100):
        A = A_.copy()
        N = A.shape[0]
        for i in range(max_iter):
            mu = wilkinson_shift(A)
            _Q, _R = QR_decomp(A - mu * np.eye(N))
            A = _R @ _Q + mu * np.eye(N)
            if np.tril(np.abs(A), -1).max() < eps:
                print(f'early stop: i={i!r}')
                break
        return A
    diag_A_1 = shifted_QR_iteration(A)
    print('diag_A=')
    sns.heatmap(diag_A_1, annot=True, cmap='coolwarm')
    plt.show()
    # シフトを入れると固有値の並び順は絶対値順にならないので、numpy 側を対応付ける
    _evalues_np, _ = ordered_eig(A, np.diag(diag_A_1))
    print(f'residuals={np.diag(diag_A_1) - _evalues_np}')
    return (diag_A_1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 逆反復法
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    使用する変数を整理
    """)
    return


@app.cell
def _(A, diag_A_1, np, plt, sns):
    print('A=')
    sns.heatmap(A, annot=True, cmap='coolwarm')
    plt.show()
    evalues = np.diag(diag_A_1)
    print(f'evalues={evalues!r}')
    return (evalues,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    逆反復法は近似固有値$\lambda'$に対する$((A-\lambda'E)^{-1})^\infty$を固有ベクトルとして計算する方法。

    $A$ の固有値を $\lambda_i$ とすると $(A-\lambda'E)^{-1}$ の固有値は
    $1/(\lambda_i-\lambda')$ であり、固有ベクトルは $A$ のものと共通である。
    $\lambda'$ を目的の固有値 $\lambda$ の近くに取れば $1/(\lambda-\lambda')$ だけが突出して大きくなるので、
    この行列にべき乗法を適用すれば目的の固有ベクトルだけが残る。
    べき乗法が絶対値最大の固有値しか取れないのに対し、逆反復法は $\lambda'$ を変えることで
    任意の固有値の固有ベクトルを取り出せる。

    $\lambda'$ が固有値に近いほど $A-\lambda'E$ は特異に近づき連立方程式は悪条件になるが、
    誤差もまた目的の固有ベクトルの方向へ増幅されるため、固有ベクトルの精度は落ちない。

    計算の過程でレイリー商による計算
    $$
    R(\vec{x}^{(k)})=\frac{(\vec{x}^{(k)},(A-\lambda'E)^{-1}\vec{x}^{(k)})}{(\vec{x}^{(k)},\vec{x}^{(k)})}
    =\frac{1}{\lambda-\lambda'}\equiv\mu
    $$
    で固有値の高精度化が図れる。$\mu$ を逆に解くと $\lambda=\lambda'+1/\mu$ となるので、
    実装ではこれで $\lambda'$ を毎回更新している（レイリー商反復）。
    レイリー商は正規化前の $\vec{x}^{(k)}$ との内積で計算する必要がある。
    先に $\vec{x}^{(k)}$ を更新してしまうと $\|\vec{v}\|$ を計算することになり符号が落ちる。

    シフトが固有値へ近づくにつれ収束は加速する。対称行列では 3 次収束するが、
    ここで扱う $A$ は非対称なので一般には 2 次収束である。
    停止判定には残差 $\|A\vec{x}-\lambda\vec{x}\|$ を使う。
    初期ベクトルは単位ベクトルのままだが、シフトに QR 法で求めた高精度な固有値を使うため、
    2 回の反復で機械精度に達する。
    """)
    return


@app.cell
def _(A, evalues, np, plt, sns):
    def inverse_iteration(A, evalues_, eps=1e-10, max_iter=100):
        evalues = evalues_.copy()
        N = A.shape[0]
        evectors = np.eye(N)
        for i in range(N):
            x = evectors[:, i]
            for _ in range(max_iter):
                v = np.linalg.solve(A - evalues[i] * np.eye(N), x)
                # レイリー商は正規化前の x で計算する（先に x を更新すると符号が落ちる）
                mu = np.dot(x, v)
                x = v / np.linalg.norm(v)
                evalues[i] = evalues[i] + 1 / mu
                if np.linalg.norm(A @ x - evalues[i] * x) < eps:
                    break
            evectors[:, i] = x
        return (evalues, evectors)
    evalues_1, evectors = inverse_iteration(A, evalues)
    lambda_mat = np.diag(evalues_1)
    _residuals = A @ evectors - evectors @ lambda_mat
    print('residuals=')
    sns.heatmap(_residuals, annot=True, cmap='coolwarm')
    plt.show()
    return evalues_1, evectors


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    計算結果を numpy による結果と比較
    """)
    return


@app.cell
def _(A, evalues_1, evectors, ordered_eig, plt, sns):
    evalues_np_1, _evectors_np = ordered_eig(A, evalues_1)
    for idx in range(_evectors_np.shape[1]):
        if _evectors_np[0, idx] * evectors[0, idx] < 0:
            _evectors_np[:, idx] = _evectors_np[:, idx] * -1
    print(f'evalues={evalues_1!r}')
    print(f'evalues_np={evalues_np_1!r}')
    print(f'evectors={evectors!r}')
    print(f'evectors_np={_evectors_np!r}')
    print(f'evalues-evalues_np={evalues_1 - evalues_np_1!r}')
    print('residuals=')
    sns.heatmap(evectors - _evectors_np, annot=True, cmap='coolwarm')
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 固有値計算の高速化
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### ハウスホルダー変換
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    対角成分より2以上左の成分が0となっている行列をヘッセンベルグ行列という。
    QR 分解の前処理としてハウスホルダー変換と呼ばれる直交・鏡映変換で、
    対象行列をヘッセンベルグ行列へ変換する。

    単位ベクトル $\vec{u}$ に対して
    $$
    P=I-2\vec{u}\vec{u}^T
    $$
    は $\vec{u}$ に直交する超平面に関する鏡映を表す。$P^T=P$ かつ $P^TP=P^2=I$ なので
    直交行列であり、$P^{-1}=P^T=P$ が成り立つ。
    したがって $PAP$ は相似変換であり固有値を変えない。

    ベクトル $\vec{b}$ を第 1 軸へ写す（$P\vec{b}=s\vec{e}_1$）には
    $$
    s=\mp\|\vec{b}\|,\quad
    \vec{u}=\frac{\vec{b}-s\vec{e}_1}{\|\vec{b}-s\vec{e}_1\|}
    $$
    と取ればよい。符号は $s=-\mathrm{sign}(b_1)\|\vec{b}\|$ と $b_1$ の逆に選ぶ。
    同符号にすると $\vec{b}-s\vec{e}_1$ の第 1 成分で桁落ちが起きるためである。

    第 $k$ 列に対する $P_k$ は先頭 $k+1$ 行・列を単位行列のままにしてあるので、
    相似変換 $P_kHP_k$ は既に作った 0 を壊さない。この制約があるため上三角までは落とせず、
    劣対角成分が 1 本残るヘッセンベルグ形が限界になる
    （上三角にできるなら反復無しで固有値が求まってしまう）。$A$ が対称なら結果は三重対角行列になる。

    利点は計算量である。密行列の QR 分解は 1 回 $O(n^3)$ かかるが、
    ヘッセンベルグ行列なら消すべき成分が $n-1$ 個しかないので $O(n^2)$ で済む。
    さらに $RQ$ もヘッセンベルグ形を保つので、この形は反復の間ずっと維持される。
    前処理自体は $O(n^3)$ だが 1 回で終わる。
    """)
    return


@app.cell
def _(A, N, np, plt, sns):
    def householder(b):
        e = np.zeros_like(b)
        e[0] = 1
        s = -np.sign(b[0]) * np.linalg.norm(b)  # 桁落ち対策で異符号に
        u = (_ := (b - s * e)) / np.linalg.norm(_)
        _P = np.eye(len(b)) - 2 * np.outer(u, u)
        return _P

    def hessenberg(A):
        _H = A.copy()
        _P = np.eye(N)
        for k in range(N - 2):
            x = _H[k + 1:, k]
            Pk = np.eye(N)
            Pk[k + 1:, k + 1:] = householder(x)  # ハウスホルダー変換
            _H = Pk @ _H @ Pk
            _P = _P @ Pk
        return (_H, _P)
    _H, _P = hessenberg(A)
    print('P.T @ P=')
    sns.heatmap(_P.T @ _P, annot=True, cmap='coolwarm')
    plt.show()
    print('H=')
    sns.heatmap(_H, annot=True, cmap='coolwarm')
    plt.show()
    print('P.T@A@P-H=')
    sns.heatmap(_P.T @ A @ _P - _H, annot=True, cmap='coolwarm')
    plt.show()
    return (hessenberg,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ヘッセンベルグ化した $H=P^TAP$ に QR 法を適用しても、
    相似変換なので元の $A$ と同じ固有値が得られる。
    """)
    return


@app.cell
def _(A, QR_iteration, hessenberg, np, ordered_eig, plt, sns):
    _H, _ = hessenberg(A)
    H_qr = QR_iteration(_H)
    print('QR 反復後の H=')
    sns.heatmap(H_qr, annot=True, cmap='coolwarm')
    plt.show()
    _evalues_np, _ = ordered_eig(A, np.diag(H_qr))
    print(f'residuals={np.diag(H_qr) - _evalues_np}')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### ギブンス回転
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ギブンス回転は 2 つの座標軸が張る平面内での回転で、狙った 1 成分だけを 0 にする直交変換である。
    第 $j$ 列の劣対角成分 $a_{j+1,j}$ を消すには、$j, j+1$ 行だけに作用する行列
    $$
    P_j=\begin{pmatrix}\cos\theta&\sin\theta\\-\sin\theta&\cos\theta\end{pmatrix},\quad
    \cos\theta=\frac{a_{jj}}{r},\quad
    \sin\theta=\frac{a_{j+1,j}}{r},\quad
    r=\sqrt{a_{jj}^2+a_{j+1,j}^2}
    $$
    を左から掛ければよい。実際
    $$
    P_j\begin{pmatrix}a_{jj}\\a_{j+1,j}\end{pmatrix}
    =\begin{pmatrix}r\\0\end{pmatrix}
    $$
    となる（$r$ が 0 に近い場合は既に消えているので何もしない）。

    ハウスホルダー変換が 1 回で列全体を消すのに対し、ギブンス回転は 1 成分ずつ消す。
    密行列では割に合わないが、ヘッセンベルグ行列は消すべき劣対角成分が $n-1$ 個しかないため、
    $n-1$ 回の回転だけで上三角化でき $O(n^2)$ で QR 分解が完了する。

    $R=P_{n-2}\cdots P_0A$ であり、各 $P_j$ は直交行列なので
    $Q=(P_{n-2}\cdots P_0)^T$ とすれば $A=QR$ となる。
    実装が返す `_P.T` がこの $Q$ に当たる。
    """)
    return


@app.cell
def _(A, hessenberg, np, plt, sns):
    def gibbens_rotation(A, j, eps=1e-06):
        _P = np.eye(A.shape[0])
        sin, cos = (0, 0)
        norm = np.linalg.norm(A[j:j + 2, j])
        if norm > eps:
            sin, cos = (A[j + 1, j] / norm, A[j, j] / norm)
        _P[j:j + 2, j:j + 2] = np.array([[cos, sin], [-sin, cos]])
        return _P

    def gibbens_QR(A):
        N = A.shape[0]
        _R, _P = (A.copy(), np.eye(N))
        for j in range(N - 1):
            P_ = gibbens_rotation(_R, j)
            _R = P_ @ _R
            _P = P_ @ _P
        return (_P.T, _R)
    _H, _P = hessenberg(A)
    _Q, _R = gibbens_QR(_H)
    print('Q.T @ Q=')
    sns.heatmap(_Q.T @ _Q, annot=True, cmap='coolwarm')
    plt.show()
    print('R=')
    sns.heatmap(_R, annot=True, cmap='coolwarm')
    plt.show()
    print('P.T@A@P-QR=')
    sns.heatmap(_P.T @ A @ _P - _Q @ _R, annot=True, cmap='coolwarm')
    plt.show()
    return (gibbens_QR,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### デフレーション付き QR 反復
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ここまでの部品を組み合わせる。ハウスホルダー変換でヘッセンベルグ化し、
    ギブンス回転による QR 分解でシフト付き QR 反復を回す。

    仕上げがデフレーションである。劣対角成分 $H_{m,m-1}$ が十分小さくなったらこれを 0 とみなす。
    このとき $H$ はブロック上三角になり、$H_{mm}$ がそのまま固有値として確定する。
    以降は左上の $m\times m$ 部分だけを相手にすればよく、
    アクティブブロックの次元が 1 ずつ縮んでいく。これを $m=0$ まで繰り返せば全固有値が求まる。

    シフトには右下隅の値 $s=H_{mm}$ をそのまま使う（レイリーシフト）。
    ウィルキンソンシフトほど頑健ではないが、平方根の計算が要らず実装が簡単である。

    複素共役対を持つ行列では 1x1 まで潰れず反復が止まらないため、
    ブロックごとの反復回数に上限 `max_block_iter` を設けて打ち切る。
    実用の実装ではこの場合 2x2 ブロックを残す（実シューア形）。
    """)
    return


@app.cell
def _(A, gibbens_QR, hessenberg, np, ordered_eig, plt, sns):
    def householder_QR_iteration(A_, eps=1e-06, max_block_iter=40):
        _H, _ph = hessenberg(A_)
        m = _H.shape[0] - 1
        while m > 0:
            if np.abs(_H[m, m - 1]) < eps:
                _H[m, m - 1] = 0
                m = m - 1
                continue

            active_dim = m + 1
            converged = False
            for _iter in range(max_block_iter):
                s = _H[m, m]
                _Q, _R = gibbens_QR(_H[:active_dim, :active_dim] - s * np.eye(active_dim))
                _H[:active_dim, :active_dim] = _R @ _Q + s * np.eye(active_dim)
                if np.abs(_H[m, m - 1]) < eps:
                    _H[m, m - 1] = 0
                    converged = True
                    break

            if not converged:
                print(f'bounded stop: active_dim={active_dim}, iterations={max_block_iter}')
                break
        return _H
    _H = householder_QR_iteration(A)
    print('H=')
    sns.heatmap(_H, annot=True, cmap='coolwarm')
    plt.show()
    _evalues_np, _ = ordered_eig(A, np.diag(_H))
    print(f'residuals={np.diag(_H) - _evalues_np}')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 特異値分解
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 右特異ベクトル
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    行列 $A$ に対して行列 $A^TA$ を考える。
    明らかに $(A^TA)^T=A^TA$ なので対称行列。
    [スペクトル定理](https://ja.wikipedia.org/wiki/%E3%82%B9%E3%83%9A%E3%82%AF%E3%83%88%E3%83%AB%E5%AE%9A%E7%90%86)より固有値は常に実数である。

    また行列 $A$ の非負定値の条件は任意の $\vec{x}\neq\vec{0}$ に対して $\vec{x}^TA\vec{x}\geq0$。
    よって任意の $A, \vec{x}\neq\vec{0}$ に対して $\vec{x}^T(A^TA)\vec{x}=||A\vec{x}||^2\geq0$。
    これより固有値は常に非負である。そのため固有値の平方根も常に実数である。

    これは $AA^T$ でも同様に成立。

    さらに $A^TA\vec{v}=w\vec{v}$ の両辺に左から $A$ を掛けると
    $$
    (AA^T)(A\vec{v})=w(A\vec{v})
    $$
    となる。$w\neq0$ なら $A\vec{v}\neq\vec{0}$ なので、$A^TA$ と $AA^T$ は
    非零固有値を完全に共有する。$A$ が $N\times M$ 行列のとき $A^TA$ は $M\times M$、
    $AA^T$ は $N\times N$ とサイズが違うが、余分な分は固有値 0 が埋めているだけである。
    非零固有値の個数は $\mathrm{rank}A$ に等しく、高々 $\min(N,M)$ 個である。
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    実際に確かめるために乱数生成の行列でこれを計算。
    """)
    return


@app.cell
def _(np, plt, sns):
    np.set_printoptions(precision=3, suppress=False)
    plt.rcParams['figure.figsize'] = (5, 2)
    N_1, M = (4, 5)
    A_1 = np.random.rand(N_1, M)
    # 表示の精度を設定
    sns.heatmap(A_1, annot=True, cmap='coolwarm')
    return (A_1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    $A^TA$ は対称行列
    """)
    return


@app.cell
def _(A_1, sns):
    ATA = A_1.T @ A_1
    sns.heatmap(ATA, annot=True, cmap='coolwarm')
    return (ATA,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    固有値はすべて実数で非負（数値誤差に注意）
    """)
    return


@app.cell
def _(ATA, sns):
    import numpy.linalg as la

    w_right, V = la.eigh(ATA)
    # 固有値が大きい順
    w_right, V = w_right[::-1], V[:, ::-1]
    print(f"{w_right=}")
    sns.heatmap(V, annot=True, cmap='coolwarm')
    return V, la, w_right


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    固有ベクトルは$V$の各列が対応。
    検算で $\vec{v}^T(A^TA)\vec{v}-w$ を計算。
    """)
    return


@app.cell
def _(ATA, V, np, sns, w_right):
    _residuals = V.T @ ATA @ V - np.diag(w_right)
    sns.heatmap(_residuals, annot=True, cmap='coolwarm')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    $V$ 行列は直交行列(ユニタリー行列)で[右特異ベクトル](https://manabitimes.jp/math/1280#5)
    """)
    return


@app.cell
def _(V, sns):
    # 検算。ユニタリなら単位行列に近くなる。
    sns.heatmap(V.T @ V, annot=True, cmap='coolwarm')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 左特異ベクトル
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    同様に $AA^T$ について計算し左特異ベクトル$U$を求める
    """)
    return


@app.cell
def _(A_1, la, sns):
    AAT = A_1 @ A_1.T
    w_left, U = la.eigh(AAT)
    # 固有値が大きい順
    w_left, U = (w_left[::-1], U[:, ::-1])
    print(f'w_left={w_left!r}')
    sns.heatmap(U, annot=True, cmap='coolwarm')
    return AAT, U, w_left


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    左特異ベクトルを検算。$U^T(AA^T)U$ が固有値の対角行列になり、
    $U$ 自身は直交行列（$U^TU=I$）になる。
    """)
    return


@app.cell
def _(AAT, U, np, sns, w_left):
    _residuals = U.T @ AAT @ U - np.diag(w_left)
    sns.heatmap(_residuals, annot=True, cmap='coolwarm')
    return


@app.cell
def _(U, sns):
    sns.heatmap(U.T @ U, annot=True, cmap='coolwarm')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 特異値分解
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    数値誤差の対策。ほぼ0は0とする。
    """)
    return


@app.cell
def _(np, w_left, w_right):
    w_right_1 = np.array([w if abs(w) > 1e-10 else 0 for w in w_right])
    w_left_1 = np.array([w if abs(w) > 1e-10 else 0 for w in w_left])
    print(f'w_right={w_right_1!r}')
    print(f'w_left={w_left_1!r}')
    return w_left_1, w_right_1


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    $A^TA, AA^T$ の固有値の平方根は特異値
    """)
    return


@app.cell
def _(np, w_left_1, w_right_1):
    sigma_right = np.sqrt(w_right_1)
    sigma_left = np.sqrt(w_left_1)
    print(f'sigma_right={sigma_right!r}')
    print(f'sigma_left={sigma_left!r}')
    return sigma_left, sigma_right


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    特異値による行列 $\Sigma$ を作成。$A$ が $N\times M$ なら $\Sigma$ も $N\times M$ で、
    左上の $\min(N,M)$ 個の対角成分に特異値を並べ、残りは 0 で埋める。
    実装で `sigma_right` と `sigma_left` の短いほうを選んでいるのは、
    共有される非零特異値の個数が $\min(N,M)$ を超えないためである。
    """)
    return


@app.cell
def _(A_1, np, sigma_left, sigma_right):
    Sigma = np.zeros_like(A_1)
    sigma = sigma_right if len(sigma_right) < len(sigma_left) else sigma_left
    Sigma[:len(sigma), :len(sigma)] = np.diag(sigma)
    print(f'Sigma={Sigma!r}')
    return Sigma, sigma


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    固有ベクトルには符号の不定性があるためそのままでは$A=U\Sigma V^T$を満たすとは限らない。

    $\Sigma$の成分は非負なので$(AV)_i=\lambda_iU_i, \lambda_i\geq0$。
    そこでこの[条件を満たすように符号の反転をする](
    https://math.stackexchange.com/questions/4844816/discrepancies-in-custom-svd-implementation-compared-to-np-linalg-svd-sign-issu)。

    つまり $A\vec{v}_i=\sigma_i\vec{u}_i$（$\sigma_i\geq0$）が特異値分解の定義であり、
    $\vec{v}_i$ と $\vec{u}_i$ を独立に対角化で求めると符号の整合が取れない。
    そこで $(A\vec{v}_i,\vec{u}_i)$ の符号を見て、負なら $\vec{u}_i$ を反転させる。
    """)
    return


@app.cell
def _(A_1, U, V, np, sns):
    U_1 = np.array([np.sign(np.dot(A_1 @ V[:, idx], U[:, idx])) * U[:, idx] for idx in range(U.shape[1])]).T
    sns.heatmap(U_1, annot=True, cmap='coolwarm')
    return (U_1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    [特異値分解](https://manabitimes.jp/math/1280#5) $A=U\Sigma V^T$ の検算。
    """)
    return


@app.cell
def _(A_1, Sigma, U_1, V, sns):
    _residuals = A_1 - U_1 @ Sigma @ V.T
    sns.heatmap(_residuals, annot=True, cmap='coolwarm')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    numpy の標準機能でも特異値分解。
    """)
    return


@app.cell
def _(A_1, la):
    U2, sigma2, V2h = la.svd(A_1)
    print(sigma2)
    return U2, V2h, sigma2


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    特異ベクトルは符号の不定性があるため、対角化から求めた結果と一致するとは限らない。
    """)
    return


@app.cell
def _(U2, U_1, V, V2h, plt, sigma, sigma2, sns):
    print(f'sigma2-sigma={sigma2 - sigma!r}')
    sns.heatmap(U_1 - U2, annot=True, cmap='coolwarm')
    plt.show()
    sns.heatmap(V.T - V2h, annot=True, cmap='coolwarm')
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    特異分解$A=U\Sigma V^T$はきちんと満たす。
    """)
    return


@app.cell
def _(A_1, U2, V2h, np, sigma_left, sigma_right, sns):
    Sigma2 = np.zeros_like(A_1)
    sigma2_1 = sigma_right if len(sigma_right) < len(sigma_left) else sigma_left
    Sigma2[:len(sigma2_1), :len(sigma2_1)] = np.diag(sigma2_1)
    _residuals = A_1 - U2 @ Sigma2 @ V2h
    sns.heatmap(_residuals, annot=True, cmap='coolwarm')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 補足

    ここでは $A^TA$ と $AA^T$ を明示的に作って対角化したが、
    この方法は条件数が 2 乗になるため小さい特異値の精度が落ちる。
    `np.linalg.svd` を含む実用の実装では、$A$ を直接二重対角化してから
    QR 法系のアルゴリズムを適用する。

    特異値分解の主な用途は次の 2 つ。

    1. 低ランク近似。特異値の大きい順に $k$ 個だけ残した $A_k=U_k\Sigma_kV_k^T$ は、
       ランク $k$ の行列の中でフロベニウスノルム誤差を最小にする（エッカート・ヤングの定理）。
       主成分分析や画像圧縮はこれを使う。
    2. 擬似逆行列。$A^+=V\Sigma^+U^T$（$\Sigma^+$ は非零特異値の逆数を取り転置したもの）は、
       正方でない行列や特異な行列に対する最小二乗解を与える。
    """)
    return


if __name__ == "__main__":
    app.run()
