import marimo

__generated_with = "0.24.0"
app = marimo.App()


with app.setup:
    import marimo as mo


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # 数値積分法
    legacy/simulation/Integration.ipynb を marimo 向けに整理し、円周率の積分表現を題材に各手法の精度を比較する。
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    数値積分法の精度を円周率の計算
    $$
        I\equiv\int_a^b dx f(x)
        =\int_{0}^1dx\frac{4}{1+x^2}=\pi
    $$
    で比較する。
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 台形公式
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    台形公式は積分を単純な差分で表したもので
    $$
    I\simeq\frac{\Delta x}{2}\left(f(x_0)+2\sum_{k=1}^{n-1}f(x_k)+f(x_n)\right),\quad
    \Delta x\equiv\frac{x_n - x_0}{n},\quad x_k\equiv x_0 + k\Delta x
    $$
    である。
    """)
    return


@app.cell
def _():
    import numpy as np

    #: 被積分関数
    def integrand(x: float) -> float:
        return 4 / (1 + x**2)

    return integrand, np


@app.cell
def _(integrand):
    from typing import Callable

    def trapezoid(integrand: Callable[[float], float], 
                  x_0: float, x_n: float, num: int) -> float:
        # 分割幅
        dx: float = 1./num
        # 台形公式
        sum: float = dx / 2 * (integrand(x_0) + integrand(x_n))
        for k in range(1, num):
            x_k = x_0 + k * dx
            sum += dx * integrand(x_k)
        return sum

    print(f"pi = {trapezoid(integrand, 0, 1, 100):.10f}")
    return Callable, trapezoid


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## シンプソン公式
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    シンプソン公式は 2 次のニュートン・コーツ公式（2 次のラグランジュ補間による積分）で
    $$
    I\simeq\frac{\Delta x}{3}\left(f(x_0)+4\sum_{k=1,3,\cdots}^{2n-1}f(x_k)+2\sum_{k=2,4,\cdots}^{2n-2}f(x_k)+f(x_{2n})\right),\quad
    \Delta x\equiv\frac{x_{2n}-x_0}{2n},\quad
    x_k\equiv x_0 + k\Delta x
    $$
    となる。
    """)
    return


@app.cell
def _(Callable, integrand, np):
    def simpson(integrand: Callable[[float], float], 
                x_0: float, x_2n: float, num: int) -> float:
        dx = (x_2n - x_0) / (2 * num)
        sum = dx / 3 * (integrand(x_0) + integrand(x_2n))
    
        #: 奇数の場合
        for k in np.arange(1, 2*num, 2):
            x_k = x_0 + k * dx
            sum += dx * 4/3 * integrand(x_k)
        #: 偶数の場合
        for k in np.arange(2, 2*num, 2):
            x_k = x_0 + k * dx
            sum += dx * 2/3 * integrand(x_k)
        return sum

    print(f"pi = {simpson(integrand, 0, 1, 50):.15f}")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## ロンバーグ法
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ロンバーグ法は収束値が真の積分値であるような収束列を用いて、補外を行うことで積分値を求めるアルゴリズムである。
    このような収束列を台形公式によって
    $$
    \begin{aligned}
    &T_0^n\equiv\frac{\Delta x}{2}\left(f(x_0)+2\sum_{k=1}^{2^n-1}f(x_k)+f(x_{2^n})\right)\overset{n\rightarrow\infty}{\rightarrow}I,\\
    &\Delta x\equiv\frac{x_{2^n}-x_0}{2^n},\quad x_k\equiv k\Delta x
    \end{aligned}
    $$
    のように作る。
    上の添字は分割数の添字であり、下の添字は補外回数の添字である。
    ロンバーグ法は各 $T^n_0$ に対して
    $$
    T^{n+1}_m \equiv\frac{4^m T^{n+1}_{m-1} - T^n_{m-1}}{4^m-1}
    $$
    を計算する。
    計算は
    $$
    T_0^0\overset{\text{分割}}{\rightarrow}
    T_1^0\overset{\text{補外}}{\rightarrow}
    T_1^1\overset{\text{分割}}{\rightarrow}
    T_2^0\overset{\text{補外}}{\rightarrow}
    T_2^1\overset{\text{補外}}{\rightarrow}
    T_2^2\overset{\text{分割}}{\rightarrow}
    \cdots
    $$
    の順に行い更新幅が一定以下になるまで行う。
    """)
    return


@app.cell
def _(Callable, integrand, trapezoid):
    def romberg(integrand: Callable[[float], float], num: int) -> float:
        epsilon: float = 1e-10
        #: インデックスは下添字
        T_list: list = [trapezoid(integrand, 0, 1, 1)]
        for up_index in range(1, num+1):
            #: T_0 のリスト更新
            T_0: float = trapezoid(integrand, 0, 1, 2 ** up_index)
            # 終了判定
            if abs(T_0 - T_list[-1]) < epsilon:
                return T_0

            #: 2^up_index 分割に対する Romberg 補外
            T_tmp_list: list = [T_0]
            for low_index in range(1, up_index+1):
                T_tmp_list.append(
                    (4. ** low_index * T_tmp_list[low_index-1] - T_list[low_index-1])
                    / (4. ** low_index - 1.)
                    )
                #: 終了判定
                if low_index > 1:
                    if abs(T_tmp_list[-1] - T_tmp_list[-2]) < epsilon:
                        return T_tmp_list[-1]
            #: リスト更新
            T_list = T_tmp_list
        return T_list[-1]
            
    print(f"pi = {romberg(integrand, 6):.15f}")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## ガウス・ルジャンドル積分
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ガウス・ルジャンドル積分はルジャンドル多項式のゼロ点
    $$
    x_1\leq x_2\leq\cdots\leq x_k\leq\cdots\leq x_N\quad
    \text{s.t.}\quad P_N(x_k)=0
    $$
    を利用する。
    ルジャンドル多項式はボネの漸化式
    $$
    P_0(x)=1,\quad
    P_1(x)=x,\quad
    (n+1)P_{n+1}(x)=(2n+1)xP_n(x)-nP_{n-1}(x)
    $$
    によって生成される。
    """)
    return


@app.cell
def _():
    def legendre(n: int, x: float) -> float:
        '''
        ルジャンドル多項式
        '''
        if n == 0:
            return 1
        elif n == 1:
            return x
    
        p: list[float] = [0., 1., x]

        for k in range(n-1):
            p[0] = p[1]
            p[1] = p[2]
            # ボネの漸化式
            p[2] = ((2*k+3) * x * p[1] - (k+1) * p[0]) / (k+2)
        return p[2]

    print(f"P_10(0.148874) = {legendre(10, 0.148874):.4e}")
    return (legendre,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ルジャンドル多項式のゼロ点は二分法で求める:

    1. 解が含まれる領域 $[a,b]$ を事前に指定する
    2. 中点を利用して領域を $[a,(a+b)/2]$ と $[(a+b)/2,b]$ の2つに分割する
    3. 解を含む領域を中間値の定理で判定する
    4. 解を含む領域を選択して新たに計算する領域として繰り返す
    5. 領域幅が一定以下になったら終了

    中間値の定理から領域 $[a,b]$ に対して $f(a)f(b)<0$ なら、この領域 $[a,b]$ に $f(x)=0$ の解が含まれる事がわかる。
    """)
    return


@app.cell
def _(Callable, legendre):
    def bisection(func: Callable[[float],float],
                  left: float, right: float) -> float:
        '''
        二分法のアルゴリズム
        '''
        # 二分法で解けない場合
        if func(left) * func(right) >= 0:
            raise Exception('Cannot be solved by bisection method!')

        # 数値精度
        epsilon: float = 1e-10

        middle: float = 0.
        while True:
            # 中点計算
            middle = (left + right)/2.
            # 中間値の定理から解を含む区間を選択
            if func(left) * func(middle) < 0:
                right = middle
            else:
                left = middle
            # 終了判定
            if abs(func(left) - func(right)) < epsilon:
                break
    
        middle = (left+right)/2.
        return middle

    print(f"x = {bisection(lambda x: legendre(3, x), 0.1, 1):.4f} s.t. P_3(x) < 1e-10")
    return (bisection,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    $n$ 次ルジャンドル関数の正のゼロ点 $x_k$ は
    $$
    \begin{aligned}
    &n=2m\quad\text{($n$ is even)}\\
    &n=2m+1\quad\text{($n$ is odd)}
    \end{aligned}
    $$
    とすると
    $$
    \sin\left(\frac{n-1-2k}{2n+1}\pi\right)
    < x_k <
    \sin\left(\frac{n+1-2k}{2n+1}\pi\right)
    $$
    にある。
    また$-x_k$もゼロ点であることが知られている。
    $n$ が奇数の場合は $x=0$ もゼロ点になる。
    この性質を利用してルジャンドル関数のゼロ点を二分法によって求めることができる。
    """)
    return


@app.cell
def _(bisection, legendre, np):
    import math

    def legendre_zeros(num: int) -> list:
        # 初期化
        m: int = 0
        zeros: list[float ] = []
        if num % 2 == 0:
            m = num / 2
        else:
            m = (num - 1) / 2
            zeros.append(0)

        # 二分法の区間算出
        sections: list[float] = []
        for k in np.arange(0, m+1, 1):
            boundary: float = math.sin((num-(2*m+1)+2*k)/(2*num+1) * math.pi)
            if boundary > 0:
                sections.append(boundary)
            else:
                sections.append(1e-10)

        # ゼロ点の計算
        for k in range(0, int(m)):
            zero_tmp: float = bisection(lambda x: legendre(num, x), 
                                        sections[k], sections[k+1])
            zeros.append(zero_tmp)
            zeros.insert(0, -zero_tmp)
        return zeros
    print(legendre_zeros(5))
    return (legendre_zeros,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    差分近似
    $$
    I\equiv\int_{-1}^1dx \tilde{f}(x)
    \simeq\sum_{k=1}^N w_k \tilde{f}(x_k),\quad
    w_k\equiv 2\left[\sum_{l=0}^{N-1}(2l+1)\left[P_l(x_k)\right]^2\right]^{-1}
    $$
    の係数を計算する。
    """)
    return


@app.cell
def _(legendre, legendre_zeros):
    def legendre_coeff(num: int) -> list:
        zeros: list[float] = legendre_zeros(num)

        coeff: list[float] = []
        for k in range(0, num):
            sum: float = 0
            for ell in range(0, num):
                sum += (2 * ell + 1) * legendre(ell, zeros[k]) ** 2
            coeff.append(2/sum)
        return coeff
    legendre_coeff(5)
    return (legendre_coeff,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    積分を変形
    $$
        I=\int_{0}^1dx\frac{4}{1+x^2}
        =\frac{1}{2}\int_{-1}^1dx\frac{4}{1+x^2}
        =\pi
    $$
    してガウス・ルジャンドル積分を実施。
    """)
    return


@app.cell
def _(Callable, integrand, legendre_coeff, legendre_zeros):
    def gauss_legendre(integrand: Callable[[float], float], num: int) -> float:
        zeros: list[float] = legendre_zeros(num)
        coeff: list[float] = legendre_coeff(num)

        ans: float = 0
        for k in range(0,num):
            ans += coeff[k] * integrand(zeros[k])
    
        return ans
    
    print(f"pi = {gauss_legendre(integrand, 20)/2:.15f}")
    return


if __name__ == "__main__":
    app.run()
