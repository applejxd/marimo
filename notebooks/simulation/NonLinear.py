import marimo

__generated_with = "0.24.0"
app = marimo.App()


with app.setup:
    import marimo as mo


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # 非線形方程式の数値解法

    1 変数および多変数の非線形方程式 $f(x)=0$ を数値的に解く。
    解を含む区間を狭めていく方法（二分法・はさみうち法）と、
    微分を使って収束を速める方法（ニュートン法・割線法）を、
    同じ例題に対して比較する。
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 例題
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    3次方程式
    $$
    x^3+6x^2+21x+32=0
    $$
    の解は
    $$
    x=-9^{1/3}+3^{1/3}-2,\ -9^{1/3}e^{i2\pi/3}+3^{1/3}e^{i4\pi/3}-2,\ -9^{1/3}e^{i4\pi/3}+3^{1/3}e^{i2\pi/3}-2
    $$
    または
    $$
    x\sim-2.64,\ -1.68-3.05i,\ -1.68+3.05i
    $$
    """)
    return


@app.cell
def _():
    def non_linear_func(x: float) -> float:
        return x**3 + 6 * x**2 + 21 * x + 32

    def derivative_func(x: float) -> float:
        return 3 * x**2 + 12 * x + 21
    x_real = -2.637834253
    return derivative_func, non_linear_func, x_real


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 解を含む区分を狭めていく方法
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    - 初期値として解を含む区分を与える必要がある
    - ヤコビ行列を必要としない
    - 確実に収束する
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### 2分法
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    1. 解が含まれる領域 $[a,b]$ を事前に指定する
    2. 中点を利用して領域を $[a,(a+b)/2]$ と $[(a+b)/2,b]$ の2つに分割する
    3. 解を含む領域を中間値の定理で判定する
    4. 解を含む領域を選択して新たに計算する領域として繰り返す
    5. 領域幅が一定以下になったら終了

    中間値の定理から領域 $[a,b]$ に対して $f(a)f(b)<0$ なら、この領域 $[a,b]$ に $f(x)=0$ の解が含まれる事がわかる。
    """)
    return


@app.cell
def _():
    from typing import Callable

    def bisection(func: Callable[[float],float],
                  left: float, right: float) -> float:
        '''
        二分法のアルゴリズム
        '''
        # 二分法で解けない場合
        if func(left) * func(right) >= 0:
            raise Exception('Cannot be solved by bisection method!')

        # 代数式の数値精度
        delta: float = 1e-10
        # 解の数値精度
        epsilon: float = 5e-6

        middle: float = 0.
        count: int = 0
        while True:
            # 中点計算
            middle = (left + right)/2.
            count += 1

            # 中間値の定理から解を含む区間を選択
            if func(left) * func(middle) < 0:
                right = middle
            else:
                left = middle

            # 終了判定
            if (abs(func(middle)) < delta):
                print(f"finished: f(x)={abs(func(middle)):.6e} (count={count})")
                break
            if (abs(left - right) < epsilon):
                print(f"finished: |left - right|={abs(left - right):.6e} (count={count})")
                break
    
        middle = (left+right)/2.
        return middle

    return Callable, bisection


@app.cell
def _(bisection, non_linear_func, x_real):
    _x = bisection(non_linear_func, -3.0, 0)
    print(f'Δx = {_x - x_real:.6e}, f(x)={non_linear_func(_x):.6e}')
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### はさみうち法
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    はさみうち法では2分法のアレンジとして、交点を中点ではなく直線近似の解を用いる。
    $$
    y(x)\equiv\frac{f(a)-f(b)}{a-b}(x-a)+f(a)
    $$
    とした場合の
    $$
    y(c)=0 ⇔ c=-f(a)\frac{a-b}{f(a)-f(b)}+a=\frac{bf(a)-af(b)}{f(a)-f(b)}
    $$
    """)
    return


@app.cell
def _(Callable):
    def squeeze(func: Callable[[float],float], 
                left: float, right: float) -> float:
        '''
        はさみうち法のアルゴリズム
        '''
        # 二分法で解けない場合
        if func(left) * func(right) >= 0:
            raise Exception('Cannot be solved by squeeze method!')

        # 代数式の数値精度
        delta: float = 1e-10
        # 解の数値精度
        epsilon: float = 5e-6

        middle: float = 0.
        count: int = 0
        while True:
            # 中点計算
            f_left, f_right = func(left), func(right)
            middle = (right*f_left - left*f_right)/(f_left - f_right)
            count += 1

            # 中間値の定理から解を含む区間を選択
            if func(left) * func(middle) < 0:
                right = middle
            else:
                left = middle

            # 終了判定
            if (abs(func(middle)) < delta):
                print(f"finished: f(x)={abs(func(middle)):.6e} (count={count})")
                break
            if (abs(left - right) < epsilon):
                print(f"finished: |left - right|={abs(left - right):.6e} (count={count})")
                break
    
        f_left, f_right = func(left), func(right)
        middle = (right*f_left - left*f_right)/(f_left - f_right)
        return middle

    return (squeeze,)


@app.cell
def _(non_linear_func, squeeze, x_real):
    _x = squeeze(non_linear_func, -3.0, 0)
    print(f'Δx = {_x - x_real:.6e}, f(x)={non_linear_func(_x):.6e}')
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## ヤコビ行列を使う方法
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### ニュートン・ラフソン法
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    非線形関数の$x=a$周りのテーラー展開を用いて
    $$
    f(x)=f(a)+f'(a)(x-a)+\mathcal{O}(\epsilon)=0
    ⇔x=a-\frac{f(a)}{f'(a)}+\mathcal{O}(\epsilon)
    $$
    と解を1次近似する。
    """)
    return


@app.cell
def _(Callable):
    def newton1d(func: Callable[[float], float],
                 derivative: Callable[[float], float], seed: float) -> float:
        '''
        1D のニュートン・ラフソン法
        '''
        # 代数式の数値精度
        delta: float = 1e-10

        result: float = seed
        count: int = 0
        while True:
            # 解の更新
            result = result - func(result) / derivative(result)
            count += 1

            # 終了判定
            if (abs(func(result)) < delta):
                print(f"finished: f(x)={abs(func(result)):.4e} (count={count})")
                break
    
        return result

    return (newton1d,)


@app.cell
def _(derivative_func, newton1d, non_linear_func, x_real):
    _x = newton1d(non_linear_func, derivative_func, 0)
    print(f'Δx = {_x - x_real:.6e}, f(x)={non_linear_func(_x):.6e}')
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## ヤコビ行列を使用しない方法
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### 割線法
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    割線法はニュートン法に対し、はさみうち法のような直線近似の解を用いる。
    逐次近似の直前2つの解$(x_0, f(x_0))$と$(x_1, f(x_1))$に対し
    $$
    y(x)\equiv\frac{f(x_0)-f(x_1)}{x_0-x_1}(x-x_0)+f(x_0)
    $$
    とした場合の
    $$
    y(x_2)=0 ⇔ x_2=-f(x_0)\frac{x_0-x_1}{f(x_0)-f(x_1)}+x_0=\frac{x_1f(x_0)-x_0f(x_1)}{f(x_0)-f(x_1)}
    $$
    を使用する。
    """)
    return


@app.cell
def _(Callable):
    def secant(func: Callable[[float],float], 
                x_0: float, x_1: float) -> float:
        '''
        割線法のアルゴリズム
        '''
        # 代数式の数値精度
        delta: float = 1e-10

        result: float = 0
        count: int = 0
        while True:
            # 解の更新
            result = (x_1*func(x_0)-x_0*func(x_1))/(func(x_0)-func(x_1))
            count += 1

            # 終了判定
            if (abs(func(result)) < delta):
                print(f"finished: f(x)={abs(func(result)):.4e} (count={count})")
                break
            else:
                x_0 = x_1
                x_1 = result

        return result

    return (secant,)


@app.cell
def _(non_linear_func, secant, x_real):
    _x = secant(non_linear_func, -3.0, 0)
    print(f'Δx = {_x - x_real:.6e}, f(x)={non_linear_func(_x):.6e}')
    return


if __name__ == "__main__":
    app.run()
