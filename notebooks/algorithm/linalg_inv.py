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
    <a href="https://colab.research.google.com/github/applejxd/colaboratory/blob/master/algorithm/linalg_inv.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 線形方程式の求解
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 直接法
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### LU分解
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    LU分解$PA=LU$を求める。
    $P$は置換行列$PP^T=I$、$L$は対角成分1の下三角行列、$U$は上三角行列。

    LU分解を求める過程には部分ピボット選択付きガウスの消去法から求める外積形式ガウス法や、$PA=LU$から直接成分を比較して求める内積形式ガウス法が存在。ここでは[外積形式ガウス法](https://cedddnav.org/raspi4a/)を使用。
    この方法は部分ピボット選択付きガウスの消去法を$L^{-1}PA=U$と比較して$P,L,U$を求める。$P$は最終的な部分ピボット選択全体。$L$は行基本変形の積の逆行列であり、各行基本変形の逆行列が非対角成分の符号を反転させたものであることを利用。

    $PA\vec{x}=LU\vec{x}=\vec{b}⇔LU\vec{x}=P^T\vec{b}$であるので
    、$L\vec{y}=P^T\vec{b}, U\vec{x}=\vec{y}$として代入操作で解けば、
    これは前進・後退代入に対応。

    ランダムに生成した正方行列で検証。numpy の LU 分解の機能を使う。
    （非正方行列は[疑似逆行列の計算法](https://www.momoyama-usagi.com/entry/math-linear-algebra-ap08)を適用したものとする。）
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
    A = np.random.rand(N, N)

    print("A=")
    sns.heatmap(A, annot=True, cmap='coolwarm')
    plt.show()
    return A, N, np, plt, sns


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    部分ピボット選択付きガウスの消去法で LU 分解を実装
    """)
    return


@app.cell
def _(A, np, plt, sns):
    def LU_decomposition(A):
        N = A.shape[0]
        P, L, U = (np.eye(N), np.eye(N), A.copy())
        for k in range(N - 1):
            p = np.argmax(abs(U[k:, k])) + k
            if np.isclose(U[p, k], 0):
                raise ValueError('Singular matrix')
            if p != k:
                P[[k, p], :] = P[[p, k], :]
                U[[k, p], :] = U[[p, k], :]
                if k > 0:
                    L[[k, p], :k] = L[[p, k], :k]
            for _i in range(k + 1, N):
                L[_i, k] = U[_i, k] / U[k, k]
                U[_i, :] = U[_i, :] - L[_i, k] * U[k, :]
        return (P, L, U)
    P, L, U = LU_decomposition(A)
    print('P=')
    sns.heatmap(P, annot=True, cmap='coolwarm')
    plt.show()
    print('L=')
    sns.heatmap(L, annot=True, cmap='coolwarm')
    plt.show()
    print('U=')
    sns.heatmap(U, annot=True, cmap='coolwarm')
    plt.show()
    return L, P, U


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    分解が正しいか検算
    """)
    return


@app.cell
def _(A, L, P, U, plt, sns):
    residuals = P @ A - L @ U

    print("PA-LU=")
    sns.heatmap(residuals, annot=True, cmap='coolwarm')
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    scipy で検算。置換行列$P$は一致しない場合がある。
    """)
    return


@app.cell
def _(A, L, P, U, plt, sns):
    import scipy.linalg as linalg
    P2, L2, U2 = linalg.lu(A)

    print("δP=")
    sns.heatmap(P-P2, annot=True, cmap='coolwarm')
    plt.show()

    print("δL=")
    sns.heatmap(L-L2, annot=True, cmap='coolwarm')
    plt.show()

    print("δU=")
    sns.heatmap(U-U2, annot=True, cmap='coolwarm')
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ここから$A\vec{x}=\vec{b}$を解くために$\vec{b}$を生成
    """)
    return


@app.cell
def _(N, np):
    b = np.random.rand(N)
    print(f"{b=}")
    return (b,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    $L\vec{y}=P^T\vec{b}$ を解く。
    前進代入は$y_i=b'_i-\sum_{j=0}^{i-1}l_{ij}y_j$。
    """)
    return


@app.cell
def _(L, P, b, np):
    def forward_substitute(P, L, b):
        N = len(b)
        y = np.zeros(N)
        for _i in range(N):
            y[_i] = (P.T @ b)[_i] - L[_i, :_i] @ y[:_i]
        return y
    y = forward_substitute(P, L, b)
    print(f'y={y!r}')
    return forward_substitute, y


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    $U\vec{x}=\vec{y}$ を解く。
    後退代入は$x_i=(y_i-\sum_{j=i+1}^nu_{ij}x_j)/u_{ii}$。
    """)
    return


@app.cell
def _(U, np, y):
    def backward_substitute(U, y):
        N = len(y)
        x = np.zeros(N)
        for _i in range(N - 1, -1, -1):
            x[_i] = (y[_i] - U[_i, _i + 1:] @ x[_i + 1:]) / U[_i, _i]
        return x
    x = backward_substitute(U, y)
    print(f'x={x!r}')
    return backward_substitute, x


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    解の検算を実施
    """)
    return


@app.cell
def _(A, b, x):
    print(f"{A@x-b=}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    逆行列もLU分解から求まる。$AA^{-1}=I$なので$A\vec{x}_i=\vec{e}_i$を解けばいい。これは他の手法でも同じ。
    """)
    return


@app.cell
def _(A, L, N, P, U, backward_substitute, forward_substitute, np, plt, sns):
    A_inv = np.zeros((N, N))
    for _i in range(N):
        e_i = np.zeros(N)
        e_i[_i] = 1
        y_i = forward_substitute(P, L, e_i)
        x_i = backward_substitute(U, y_i)
        A_inv[:, _i] = x_i
    print('A_inv=')
    sns.heatmap(A_inv, annot=True, cmap='coolwarm')
    plt.show()
    print('AA_inv=')
    sns.heatmap(A @ A_inv, annot=True, cmap='coolwarm')
    # 検算
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### コレスキー分解法
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    解くべき連立一次方程式$A\vec{x}=\vec{b}$の係数行列$A$が非負定値対称行列ならコレスキー分解$A=LL^T$が可能なので、これを利用。
    公式は成分比較で求める。

    対称性は$A$と比較して$L$の成分が半分であることから必要。
    非不定値は平方根の計算のために必要。

    不定値なら$(A^TA)\vec{x}=A^T\vec{b}$を解く。$A^TA$は必ず対称で非負定値(特異値分解の部分を参照)。
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    コレスキー分解は対称正定値行列$A$に対して舌三角行列$L$で$A=LL^T$と分解して、前進・後退代入で$A\vec{x}=\vec{b}$を解く。
    あるいは$(AA^T)\vec{x}=(LL^T)\vec{x}=A^T\vec{b}$とする。
    """)
    return


@app.cell
def _(A, np, plt, sns):
    def cholesky_decomposition(A):
        L = np.zeros_like(A)
        N = A.shape[0]
        for _i in range(N):
            for j in range(_i + 1):
                if _i == j:
                    L[_i, j] = np.sqrt(A[_i, _i] - np.sum(L[_i, :j] ** 2))
                else:
                    L[_i, j] = (A[_i, j] - np.sum(L[_i, :j] * L[j, :j])) / L[j, j]
        return L
    L_1 = cholesky_decomposition(A.T @ A)
    print('L=')
    sns.heatmap(L_1, annot=True, cmap='coolwarm')
    plt.show()
    print('LL^T-A^TA=')
    sns.heatmap(L_1 @ L_1.T - A.T @ A, annot=True, cmap='coolwarm')
    plt.show()
    return (L_1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    線形方程式$A\vec{x}=\vec{b}$の解法はLU分解と同様
    """)
    return


@app.cell
def _(A, L_1, b, np):
    def forward_substitute_1(L, b):
        N = len(b)
        y = np.zeros(N)
        for _i in range(N):
            y[_i] = (b[_i] - L[_i, :_i] @ y[:_i]) / L[_i, _i]
        return y

    def backward_substitute_1(LT, y):
        N = len(y)
        x = np.zeros(N)
        for _i in range(N - 1, -1, -1):
            x[_i] = (y[_i] - LT[_i, _i + 1:] @ x[_i + 1:]) / LT[_i, _i]
        return x
    y_1 = forward_substitute_1(L_1, A.T @ b)
    x_1 = backward_substitute_1(L_1.T, y_1)
    print(f'A@x-b={A @ x_1 - b!r}')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 修正コレスキー分解
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    平方根の計算を避ける修正コレスキー分解ではスケール行列$D'$を抜き出して$A=LL^T≡(L'D')(L'D')^T=L'D'D'^TL'^T\equiv L'DL'^T$とする。
    $L'$の対角成分はすべて1である。

    成分比較から分解の公式を求める。
    """)
    return


@app.cell
def _(A, np, plt, sns):
    def modified_cholesky_decomposition(A):
        L, D = (np.zeros_like(A), np.zeros_like(A))
        N = A.shape[0]
        for _i in range(N):
            for j in range(_i + 1):
                if _i == j:
                    L[_i, _i] = 1
                    D[_i, _i] = A[_i, _i] - np.sum(L[_i, :_i] ** 2 * D[:_i, :_i])
                else:
                    L[_i, j] = (A[_i, j] - np.sum(L[_i, :j] * L[j, :j] * D[:j, :j])) / D[j, j]
        return (L, D)
    L_2, D = modified_cholesky_decomposition(A.T @ A)
    print('L=')
    sns.heatmap(L_2, annot=True, cmap='coolwarm')
    plt.show()
    print('D=')
    sns.heatmap(D, annot=True, cmap='coolwarm')
    plt.show()
    print('L D L^T - A^T A=')
    sns.heatmap(L_2 @ D @ L_2.T - A.T @ A, annot=True, cmap='coolwarm')
    plt.show()
    return D, L_2


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    線形方程式$A\vec{x}=\vec{b}$の解法はコレスキー分解と同様
    """)
    return


@app.cell
def _(A, D, L_2, b, np):
    def forward_substitute_2(L, b):
        N = len(b)
        y = np.zeros(N)
        for _i in range(N):
            y[_i] = (b[_i] - L[_i, :_i] @ y[:_i]) / L[_i, _i]
        return y

    def backward_substitute_2(LT, y):
        N = len(y)
        x = np.zeros(N)
        for _i in range(N - 1, -1, -1):
            x[_i] = (y[_i] - LT[_i, _i + 1:] @ x[_i + 1:]) / LT[_i, _i]
        return x
    y_2 = forward_substitute_2(L_2 @ D, A.T @ b)
    x_2 = backward_substitute_2(L_2.T, y_2)
    print(f'A@x-b={A @ x_2 - b!r}')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 間接法
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    間接法は反復更新で解に収束させることで解く。
    [大きく定常法と非定常法に分けられる](http://nkl.cc.u-tokyo.ac.jp/13n/SolverIterative.pdf)。
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 定常法
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    定常法の収束に対する十分条件は狭義行・列対角優位性なので、ランダム生成した行列$A$では収束しない可能性があることに気を付ける。

    このためアルゴリズムチェックでは手動で狭義対角優位な行列を設定する。
    """)
    return


@app.cell
def _(np, plt, sns):
    N_1 = 5
    A_1 = np.zeros((N_1, N_1))
    np.fill_diagonal(A_1, 5)
    for _i in range(N_1 - 1):
        A_1[_i, _i + 1] = 2
        A_1[_i + 1, _i] = 2
    print('A=')
    sns.heatmap(A_1, annot=True, cmap='coolwarm')
    plt.show()
    return A_1, N_1


@app.cell
def _(N_1, np):
    b_1 = np.random.rand(N_1)
    print(f'b={b_1!r}')
    return (b_1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    対象の行列$A$を下三角$L$・対角$D$・上三角$U$に分解。つまり$A=D+L+U$。
    """)
    return


@app.cell
def _(A_1, np, plt, sns):
    L_3 = np.tril(A_1, -1)
    D_1 = np.diag(np.diag(A_1))
    U_1 = np.triu(A_1, 1)
    print('L=')
    sns.heatmap(L_3, annot=True, cmap='coolwarm')
    plt.show()
    print('D=')
    sns.heatmap(D_1, annot=True, cmap='coolwarm')
    plt.show()
    print('U=')
    sns.heatmap(U_1, annot=True, cmap='coolwarm')
    plt.show()
    return D_1, L_3, U_1


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### ヤコビ法
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    反復の漸化式は
    $$
    \vec{x}^{(k+1)}=D^{-1}(\vec{b}-(L+U)\vec{x}^{(k)})
    $$
    """)
    return


@app.cell
def _(np):
    def jacobi_step(L, D, U, x, b):
      Dinv = np.diag(1/np.diag(D))
      return Dinv @ (b - (L + U) @ x)

    return (jacobi_step,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    この反復で$A\vec{x}=\vec{b}$を満たす$\vec{x}$を求める。
    """)
    return


@app.cell
def _(A_1, D_1, L_3, U_1, b_1, jacobi_step, np):
    _max_step = 100
    _eps = 1e-06
    x_3 = np.ones(A_1.shape[0])
    for _idx, _step in enumerate(range(_max_step)):
        _x_new = jacobi_step(L_3, D_1, U_1, x_3, b_1)
        if np.linalg.norm(_x_new - x_3) < _eps:
            break
        x_3 = _x_new
    print(f'idx={_idx!r}')
    print(f'x={x_3!r}')
    print(f'A@x-b={A_1 @ x_3 - b_1!r}')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### ガウス・ザイデル法
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    反復の漸化式は
    $$
    \vec{x}^{(k+1)}=D^{-1}(\vec{b}-L\vec{x}^{(k+1)}-U\vec{x}^{(k)}).
    $$
    $L$は下三角なので$x_i^{(k+1)}$の計算で$x_{0,...,i-1}^{(k+1)}$を使用することになる。
    """)
    return


@app.cell
def _(np):
    def gauss_seidel_step(L, D, U, x, b):
        Dinv = np.diag(1 / np.diag(D))
        _x_new = np.zeros_like(x)
        for _i in range(len(x)):
            _x_new[_i] = Dinv[_i, _i] * (b[_i] - L[_i, :_i] @ _x_new[:_i] - U[_i, _i:] @ x[_i:])
        return _x_new

    return (gauss_seidel_step,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    この反復で$A\vec{x}=\vec{b}$を満たす$\vec{x}$を求める。
    """)
    return


@app.cell
def _(A_1, D_1, L_3, U_1, b_1, gauss_seidel_step, np):
    _max_step = 100
    _eps = 1e-06
    x_4 = np.ones(A_1.shape[0])
    for _idx, _step in enumerate(range(_max_step)):
        _x_new = gauss_seidel_step(L_3, D_1, U_1, x_4, b_1)
        if np.linalg.norm(_x_new - x_4) < _eps:
            break
        x_4 = _x_new
    print(f'idx={_idx!r}')
    print(f'x={x_4!r}')
    print(f'A@x-b={A_1 @ x_4 - b_1!r}')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### SOR 法
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    反復の漸化式は
    $$
    \vec{x}^{(k+1)}=(1-\omega)\vec{x}^{(k)}+\omega D^{-1}(\vec{b}-L\vec{x}^{(k+1)}-U\vec{x}^{(k)}).
    $$
    $L$は下三角なので$x_i^{(k+1)}$の計算で$x_{0,...,i-1}^{(k+1)}$を使用することになる。

    $\omega=1$でガウス・ザイデル法と一致する。
    """)
    return


@app.cell
def _(np):
    def sor_step(L, D, U, x, b, omega):
        Dinv = np.diag(1 / np.diag(D))
        _x_new = np.zeros_like(x)
        for _i in range(len(x)):
            xi = Dinv[_i, _i] * (b[_i] - L[_i, :_i] @ _x_new[:_i] - U[_i, _i:] @ x[_i:])
            _x_new[_i] = (1 - omega) * x[_i] + omega * xi
        return _x_new

    return (sor_step,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    この反復で$A\vec{x}=\vec{b}$を満たす$\vec{x}$を求める。
    ただし収束の必要条件などから$1<\omega<2$。
    """)
    return


@app.cell
def _(A_1, D_1, L_3, U_1, b_1, np, sor_step):
    _max_step = 100
    _eps = 1e-06
    omega = 1.22
    x_5 = np.ones(A_1.shape[0])
    for _idx, _step in enumerate(range(_max_step)):
        _x_new = sor_step(L_3, D_1, U_1, x_5, b_1, omega)
        if np.linalg.norm(_x_new - x_5) < _eps:
            break
        x_5 = _x_new
    print(f'idx={_idx!r}')
    print(f'x={x_5!r}')
    print(f'A@x-b={A_1 @ x_5 - b_1!r}')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 非定常法
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### 共役勾配法
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    まずはランダム行列を生成する
    """)
    return


@app.cell
def _(np, plt, sns):
    N_2 = 4
    A_2 = np.random.rand(N_2, N_2)
    b_2 = np.random.rand(N_2)
    print('A=')
    sns.heatmap(A_2, annot=True, cmap='coolwarm')
    plt.show()
    print(f'b={b_2!r}')
    return A_2, b_2


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    共役勾配法は対称正定値行列$A$に対して2次形式の最適化問題
    $$
    \min_{\vec{x}}f(\vec{x})
    \equiv
    \min_{\vec{x}}\left(\frac{1}{2}(\vec{x},A\vec{x})-(\vec{x},\vec{b})\right)
    $$
    に変形して適用。

    A が対称正定値でなければ $A^TA$ を $A$ として処理する。
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    暫定解$\vec{x}_k$に対し探索方向$\vec{p}_k$で$f(\vec{x})$が最小になる点は$\vec{r}_k\equiv\vec{b}-A\vec{x}_k$として
    $$
    \vec{x}_{k+1}=\vec{x}_k+\frac{(\vec{p}_k,\vec{r}_k)}{(\vec{p}_k,A\vec{p}_k)}\vec{p}_k.
    $$
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    勾配方向$\vec{p}$は共役の条件$(\vec{p}_{k+1},A\vec{p}_k)=0$から決定する。
    """)
    return


@app.cell
def _(A_2, b_2, np):
    _max_step = 100
    _eps = 1e-06
    ATA = A_2.T @ A_2
    x_6 = np.ones(ATA.shape[0])
    p = A_2.T @ b_2 - ATA @ x_6
    r = A_2.T @ b_2 - ATA @ x_6
    for _idx, _step in enumerate(range(_max_step)):
        alpha = np.dot(p, r) / np.dot(p, ATA @ p)
        x_6 = x_6 + alpha * p
        r = r - alpha * ATA @ p
        if np.linalg.norm(r) < _eps:
            break
        else:
            beta = -np.dot(r, ATA @ p) / np.dot(p, ATA @ p)
            p = r + beta * p
    print(f'idx={_idx!r}')
    print(f'x={x_6!r}')
    print(f'ATA@x-A.T@b={ATA @ x_6 - A_2.T @ b_2!r}')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    共役勾配法は$\dim\vec{b}$回の反復で$A\vec{x}=\vec{b}$の解が得られることが保証されている。
    """)
    return


if __name__ == "__main__":
    app.run()
