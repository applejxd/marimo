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
    <a href="https://colab.research.google.com/github/applejxd/colaboratory/blob/master/algorithm/linalg_eig.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
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
    べき乗法では対象の行列$A$のべき乗で最大固有値とその固有ベクトルを求める。

    $A(A^{\infty}\vec{x})=\lambda(A^{\infty}\vec{x})$となることを利用する。
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

    $A_k=Q_kR_k, A_{k+1}=R_kQ_k=Q_k^{-1}AQ_k=Q_k^TAQ_k$を繰り返して、直交変換。
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


@app.cell
def _(A, diag_A, np):
    def ordered_eigh(A):
        evalues, evectors = np.linalg.eigh(A)
        idx = np.abs(evalues).argsort()[::-1]  # 固有値の絶対値が大きい順に並べる
        evalues = evalues[idx]
        evectors = evectors[:, idx]
        return (evalues, evectors)
    evalues_np, _evectors_np = ordered_eigh(A)
    print(f'np.diag(diag_A)={np.diag(diag_A)!r}')
    print(f'evalues_np={evalues_np!r}')
    print(f'residuals={np.diag(diag_A) - evalues_np}')
    return evalues_np, ordered_eigh


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 原点シフト
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    QR 法の収束速度は近似固有値と最小固有値の比で決定されるため、
    最小固有値を小さくすると収束が早くなる。
    そこで最小固有値（の近似値）を$\mu_k$として
    $$
    A_k-\mu_k I=Q_kR_k,\quad
    A_{k+1}=R_kQ_k+\mu_kI
    $$
    と反復する。

    ウィルキンソンの移動法は$A_k$右下隅の2x2小行列の固有値のうち、
    $A_k$の右下隅の値に近いほうを$\mu_k$として利用。
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
def _(A, QR_decomp, evalues_np, np, plt, sns, wilkinson_shift):
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
    print(f'residuals={np.diag(diag_A_1) - evalues_np}')
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

    計算の過程でレイリー商による計算
    $$
    R(\vec{x}^{(k)})=\frac{(\vec{x}^{(k)},(A-\lambda'E)^{-1}\vec{x}^{(k)})}{(\vec{x}^{(k)},\vec{x}^{(k)})}
    =\frac{1}{\lambda-\lambda'}\equiv\mu
    $$
    で固有値の高精度化が図れる。
    """)
    return


@app.cell
def _(A, evalues, np, plt, sns):
    def inverse_iteration(A, evalues_, eps=1e-06, max_iter=100):
        evalues = evalues_.copy()
        N = A.shape[0]
        evectors = np.eye(N)
        for i in range(N):
            for _ in range(max_iter):
                v = np.linalg.solve(A - evalues[i] * np.eye(N), evectors[:, i])
                evectors[:, i] = v / np.linalg.norm(v)
                mu = np.dot(evectors[:, i], v)
                evalues[i] = evalues[i] + 1 / mu
                if np.linalg.norm(v) - mu ** 2 < eps:
                    break
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
def _(A, evalues_1, evectors, ordered_eigh, plt, sns):
    evalues_np_1, _evectors_np = ordered_eigh(A)
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


@app.cell
def _(A, QR_iteration, np, ordered_eigh, plt, sns):
    A_ = QR_iteration(A)
    evalues_np_2, evectors_1 = ordered_eigh(A_)
    print('A_=')
    sns.heatmap(A_, annot=True, cmap='coolwarm')
    plt.show()
    print(f'residuals={np.diag(A_) - evalues_np_2}')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### ギブンス回転
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


@app.cell
def _(A, gibbens_QR, hessenberg, np, plt, sns):
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
    return U, w_left


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    左特異ベクトルを検算
    """)
    return


@app.cell
def _(ATA, V, np, sns, w_right):
    _residuals = V.T @ ATA @ V - np.diag(w_right)
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
    特異値による行列 $\Sigma$ を作成
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


if __name__ == "__main__":
    app.run()
