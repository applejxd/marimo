import marimo

__generated_with = "0.24.0"
app = marimo.App()


with app.setup:
    import marimo as mo


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # 常微分方程式の初期値問題
    legacy/simulation/ODE_IVP.ipynb を marimo 向けに整理し、軌道計算を題材に各種 ODE 積分法の性質を比較します。
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 前準備（万有引力の働く系）
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    万有引力が働く系
    \begin{equation}
        m\ddot{\vec{r}}=-G\frac{Mm}{r^2}\vec{e}_r
        \Leftrightarrow
        \ddot{\vec{r}}=-\frac{GM}{r^2}\vec{e}_r
    \end{equation}
    を考える。
    簡便のため $GM=1$ となるように単位系を取る。
    このとき系を直交座標系に書き換えると
    \begin{equation}
        \ddot{x}=-\frac{x}{(x^2+y^2)^{5/2}},\quad
        \ddot{y}=-\frac{y}{(x^2+y^2)^{5/2}}
    \end{equation}
    となる。
    更にハミルトン系へと書き換えると
    \begin{align}
        &\dot{q}_x=p_x,\quad
        \dot{p}_x=-\frac{q_x}{(q_x^2+q_y^2)^{3/2}},\\
        &\dot{q}_y=p_y,\quad
        \dot{p}_y=-\frac{q_y}{(q_x^2+q_y^2)^{3/2}}
    \end{align}
    となる。
    エネルギーは運動の積分で
    \begin{equation}
        E=\frac{1}{2}(p_x^2+p_y^2)-\frac{1}{(q_x^2+q_y^2)^{1/2}}
    \end{equation}
    である。
    初期条件に $(q_x,q_y,p_x,p_y)=(1,0,0,1)$ を選ぶと
    $E = -1/2$ である。
    """)
    return


@app.cell
def _():
    from typing import Callable, List

    from numpy import ndarray

    def dqxdt(x: ndarray) -> float:
        return x[2]

    def dqydt(x: ndarray) -> float:
        return x[3]

    def dpxdt(x: ndarray) -> float:
        return -x[0]/(x[0]**2 + x[1]**2)**1.5

    def dpydt(x: ndarray) -> float:
        return -x[1]/(x[0]**2 + x[1]**2)**1.5

    # 運動方程式
    dxdt: List[Callable[[ndarray], float]] = [dqxdt,dqydt,dpxdt,dpydt]
    return Callable, List, dxdt, ndarray


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 一段階法（陽的解法）

    一段階法は実装が容易。ただし非シンプレクティックで硬い方程式系に不適。
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### オイラー法
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    オイラー法は単純な差分近似
    \begin{equation}
        \vec{x}(t+dt)
        \simeq\vec{x}(t)+\frac{d\vec{x}}{dt}(t,\vec{x}(t))dt
    \end{equation}
    """)
    return


@app.cell
def _(Callable, List, ndarray):
    import math

    import numpy as np

    def euler(dxdt: List[Callable[[ndarray], float]],
              init: ndarray, t_end: float) -> ndarray:
        # 次元チェック
        if len(dxdt) != len(init):
            raise Exception("Dimension error!")

        # ソルバーのパラメータ
        dt: float = 1e-2

        # 初期状態
        x_list: ndarray = np.array([init])
        e_list: ndarray = np.array([(init[2]**2+init[3]**2)/2
                                    -(init[0]**2+init[1]**2)**(-1/2)])
        # ソルバー実施
        steps: List[int] = range(1, math.floor(t_end/dt), 1)
        for t in steps:
            # オイラー法
            x_next: ndarray = np.zeros(len(init))
            for k in range(len(init)):
                x_next[k] = x_list[-1][k] + dxdt[k](x_list[-1]) * dt
            # 結果を追加
            x_list = np.vstack((x_list,x_next))
            e_list = np.append(e_list,
                (x_next[2]**2+x_next[3]**2)/2-(x_next[0]**2+x_next[1]**2)**(-1/2))
        
        # 位置・運動量・エネルギーを戻り値として戻す
        return np.vstack((x_list.T, e_list))

    return euler, math, np


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    軌道を計算して表示。誤差によって軌道が広がる。
    """)
    return


@app.cell
def _(dxdt, euler, ndarray, np):
    import matplotlib.pyplot as plt

    ans: ndarray = euler(dxdt, np.array([1,0,0,1]), 100)
    plt.plot(ans[0], ans[1])
    return ans, plt


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    誤差でエネルギーが上昇。
    """)
    return


@app.cell
def _(ans, plt):
    plt.plot(ans[4])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### ホイン法
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ホイン法は次数 2 の 1 段階法。
    \begin{equation}
    \vec{k}_1\equiv\frac{d\vec{x}}{dt}(t,\vec{x}(t)),\quad
    \vec{k}_2\equiv\frac{d\vec{x}}{dt}\left(t+dt,\vec{x}(t)+\vec{k}_1dt\right)
    \end{equation}
    とすれば
    \begin{equation}
        \vec{x}(t+dt)
        \simeq\vec{x}(t)
        +(\vec{k}_1+\vec{k}_2)\frac{dt}{2}
    \end{equation}
    """)
    return


@app.cell
def _(Callable, List, math, ndarray, np):
    def heun(dxdt: List[Callable[[ndarray], float]],
              init: ndarray, t_end: float) -> ndarray:
        # 次元チェック
        if len(dxdt) != len(init):
            raise Exception("Dimension error!")

        # ソルバーのパラメータ
        dt: float = 1e-2
        steps: List[int] = range(1, math.floor(t_end/dt), 1)

        # 初期状態
        x_list: ndarray = np.array([init])
        e_list: ndarray = np.array([(init[2]**2+init[3]**2)/2
                                    -(init[0]**2+init[1]**2)**(-1/2)])
        # ソルバー実施
        for t in steps:
            # ホイン法
            k: ndarray = np.zeros(len(init))
            for d in range(len(init)):
                k[d] = x_list[-1][d] + dxdt[d](x_list[-1]) * dt
        
            x_next: ndarray = np.zeros(len(init))
            for d in range(len(init)):
                x_next[d] = x_list[-1][d] + dt/2 * (dxdt[d](x_list[-1])+dxdt[d](k))
        
            # 結果を追加
            x_list = np.vstack((x_list,x_next))
            e_list = np.append(e_list,
                (x_next[2]**2+x_next[3]**2)/2-(x_next[0]**2+x_next[1]**2)**(-1/2))
        
        # 位置・運動量・エネルギーを戻り値として戻す
        return np.vstack((x_list.T, e_list))

    return (heun,)


@app.cell
def _(dxdt, heun, ndarray, plt):
    ans_1: ndarray = heun(dxdt, [1, 0, 0, 1], 100)
    plt.plot(ans_1[0], ans_1[1])
    return (ans_1,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    オイラー法と比較して誤差が低減する
    """)
    return


@app.cell
def _(ans_1, plt):
    plt.ticklabel_format(useOffset=False)
    plt.plot(ans_1[4])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### ルンゲ・クッタ法
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ルンゲ・クッタ法は次数 4 の 1 段階法。
    \begin{align}
        &\vec{k}_1\equiv\frac{d\vec{x}}{dt}\left(t,\vec{x}\right),\\
        &\vec{k}_2\equiv\frac{d\vec{x}}{dt}\left(t+\frac{dt}{2},\vec{x}+\vec{k}_1\frac{dt}{2}\right),\\
        &\vec{k}_3\equiv\frac{d\vec{x}}{dt}\left(t+\frac{dt}{2},\vec{x}+\vec{k}_2\frac{dt}{2}\right),\\
        &\vec{k}_4\equiv\frac{d\vec{x}}{dt}\left(t+dt,\vec{x}+\vec{k}_3dt\right),
    \end{align}
    として
    \begin{equation}
        \vec{x}(t+dt)
        \simeq\vec{x}(t)+(\vec{k}_1+2\vec{k}_2+2\vec{k}_3+\vec{k}_4)\frac{dt}{6}
    \end{equation}
    """)
    return


@app.cell
def _(Callable, List, math, ndarray, np):
    def rk4(dxdt: List[Callable[[ndarray], float]],
            init: ndarray, t_end: float) -> ndarray:
        # 次元チェック
        if len(dxdt) != len(init):
            raise Exception("Dimension error!")

        # ソルバーのパラメータ
        dt: float = 1e-2
        steps: List[int] = range(1, math.floor(t_end/dt), 1)

        # 初期状態
        x_list: ndarray = np.array([init])
        e_list: ndarray = np.array([(init[2]**2+init[3]**2)/2
                                    -(init[0]**2+init[1]**2)**(-1/2)])
        # ソルバー実施
        for t in steps:
            # ルンゲ・クッタ法
            k: ndarray = np.zeros((4, len(init)))
            for d in range(len(init)):
                k[0,d] = dxdt[d](x_list[-1])
            for d in range(len(init)):
                k[1,d] = dxdt[d](x_list[-1]+dt/2*k[0])
            for d in range(len(init)):
                k[2,d] = dxdt[d](x_list[-1]+dt/2*k[1])
            for d in range(len(init)):
                k[3,d] = dxdt[d](x_list[-1]+dt*k[2])
    
            x_next: ndarray = np.zeros(len(init))
            for d in range(len(init)):
                x_next[d] = x_list[-1][d] + dt * (k[0,d]+2*k[1,d]+2*k[2,d]+k[3,d])/6
        
            # 結果を追加
            x_list = np.vstack((x_list,x_next))
            e_list = np.append(e_list,
                (x_next[2]**2+x_next[3]**2)/2-(x_next[0]**2+x_next[1]**2)**(-1/2))
        
        # 位置・運動量・エネルギーを戻り値として戻す
        return np.vstack((x_list.T, e_list))

    return (rk4,)


@app.cell
def _(dxdt, ndarray, plt, rk4):
    ans_2: ndarray = rk4(dxdt, [1, 0, 0, 1], 100)
    plt.plot(ans_2[0], ans_2[1])
    return (ans_2,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    誤差が更に低減するが、誤差の増大方向が逆
    """)
    return


@app.cell
def _(ans_2, plt):
    plt.ticklabel_format(useOffset=False)
    plt.plot(ans_2[4])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 線形多段法

    線形多段法は一段階法と比べて導出が容易で計算量が少ない。
    一方で計算開始前に一段階法などで数回計算を行う必要がある。
    また一段階法と同様に非シンプレクティックで硬い方程式に不適。
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### アダムス・バッシュフォース法（陽的解法）
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    アダムス・バッシュフォース法は補外による陽的線形多段法。4次の公式は
    \begin{equation}
        \vec{x}(t+dt)
        \simeq \vec{x}(t)+\left(-\frac{9}{24}\vec{x}'(t-3dt)+\frac{37}{24}\vec{x}'(t-2dt)-\frac{59}{24}\vec{x}'(t-dt)+\frac{55}{24}\vec{x}'(t)\right)dt
    \end{equation}
    """)
    return


@app.cell
def _(Callable, List, math, ndarray, np):
    def adams_bashforth(dxdt: List[Callable[[ndarray], float]],
                        init: ndarray, t_end: float) -> ndarray:
        # 次元チェック
        if len(dxdt) != len(init):
            raise Exception("Dimension error!")

        # ソルバーのパラメータ
        dt: float = 1e-2

        # 初期状態
        x_list: ndarray = np.array([init])
        e_list: ndarray = np.array([(init[2]**2+init[3]**2)/2
                                    -(init[0]**2+init[1]**2)**(-1/2)])
        # 多段法の準備
        for t in range(1,4):
            # オイラー法
            x_next: ndarray = np.zeros(len(init))
            for k in range(len(init)):
                x_next[k] = x_list[-1][k] + dxdt[k](x_list[-1]) * dt
            # 結果を追加
            x_list = np.vstack((x_list,x_next))
            e_list = np.append(e_list,
                (x_next[2]**2+x_next[3]**2)/2-(x_next[0]**2+x_next[1]**2)**(-1/2))

        # ソルバー実施
        steps: List[int] = range(5, math.floor(t_end/dt), 1)
        for t in steps:
            # アダムス・バッシュフォース法
            x_next: ndarray = np.zeros(len(init))
            for k in range(len(init)):
                x_next[k] = x_list[-1][k] + (
                    - 9 * dxdt[k](x_list[-4])
                    +37 * dxdt[k](x_list[-3])
                    -59 * dxdt[k](x_list[-2])
                    +55 * dxdt[k](x_list[-1])) * dt/24
    
            # 結果を追加
            x_list = np.vstack((x_list,x_next))
            e_list = np.append(e_list,
                (x_next[2]**2+x_next[3]**2)/2-(x_next[0]**2+x_next[1]**2)**(-1/2))
        
        # 位置・運動量・エネルギーを戻り値として戻す
        return np.vstack((x_list.T, e_list))

    return (adams_bashforth,)


@app.cell
def _(
    adams_bashforth,
    dxdt,
    ndarray,
    np,
    plt,
):
    ans_3: ndarray = adams_bashforth(dxdt, np.array([1, 0, 0, 1]), 100)
    plt.plot(ans_3[0], ans_3[1])
    return (ans_3,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    前準備を除いたエネルギーをプロット
    """)
    return


@app.cell
def _(ans_3, plt):
    plt.ticklabel_format(useOffset=False)
    plt.plot(ans_3[4, 10:])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### アダムス・ムルトン法（陰的解法）
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    アダムス・ムルトン法は補間による陰的線形多段法。4次の公式は
    \begin{equation}
        \vec{x}(t+dt)
        \simeq \vec{x}(t)+\left(\frac{1}{24}\vec{x}'(t-2dt)-\frac{5}{24}\vec{x}'(t-dt)-\frac{19}{24}\vec{x}'(t)+\frac{9}{24}\vec{x}'(t+dt)\right)dt.
    \end{equation}
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### アダムス法（予測子修正子法）
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    アダムス法（予測子修正子法）は以下の通り：
    1. アダムス・バッシュフォース法（予測子）で $\vec{x}(t)$ から $\tilde{\vec{x}}(t+dt)$ を計算
    2. $\tilde{\vec{x}}(t+dt)$ から $\vec{x}'(t+dt)$ を計算
    3. アダムス・ムルトン法（修正子）で $\tilde{\vec{x}}(t+dt)$ を $\vec{x}(t+dt)$に修正
    4. 修正の差が一定値以下なら終了.そうでないなら 2. に戻って修正結果を再修正.
    """)
    return


@app.cell
def _(Callable, List, math, ndarray, np):
    def adams(dxdt: List[Callable[[ndarray], float]],
                        init: ndarray, t_end: float) -> ndarray:
        # 次元チェック
        if len(dxdt) != len(init):
            raise Exception("Dimension error!")

        # ソルバーのパラメータ
        dt: float = 1e-2
        iter_max: int = 10

        # 初期状態
        x_list: ndarray = np.array([init])
        e_list: ndarray = np.array([(init[2]**2+init[3]**2)/2
                                    -(init[0]**2+init[1]**2)**(-1/2)])
        # 多段法の準備
        for t in range(1,4):
            # オイラー法
            x_next: ndarray = np.zeros(len(init))
            for k in range(len(init)):
                x_next[k] = x_list[-1][k] + dxdt[k](x_list[-1]) * dt
            # 結果を追加
            x_list = np.vstack((x_list,x_next))
            e_list = np.append(e_list,
                (x_next[2]**2+x_next[3]**2)/2-(x_next[0]**2+x_next[1]**2)**(-1/2))

        # ソルバー実施
        steps: List[int] = range(5, math.floor(t_end/dt), 1)
        for t in steps:
            # アダムス・バッシュフォース法
            x_next: ndarray = np.zeros(len(init))
            for k in range(len(init)):
                x_next[k] = x_list[-1][k] + (
                    - 9 * dxdt[k](x_list[-4])
                    +37 * dxdt[k](x_list[-3])
                    -59 * dxdt[k](x_list[-2])
                    +55 * dxdt[k](x_list[-1])) * dt/24
        
            for i in range(0,iter_max):
                # 微分値を計算
                x_prime: ndarray = np.zeros(len(init))
                for k in range(len(init)):
                    x_prime[k] = dxdt[k](x_next)

                # アダムス・ムルトン法
                x_nnext: ndarray = np.zeros(len(init))
                for k in range(len(init)):
                    x_nnext[k] = x_list[-1][k] + (
                              dxdt[k](x_list[-3])
                        - 5 * dxdt[k](x_list[-2])
                        +19 * dxdt[k](x_list[-1])
                        + 9 * x_prime[k]) * dt/24
    
                diff: float = np.linalg.norm(x_next-x_nnext, ord=2)
                x_next = x_nnext

                if diff < 1e-10:
                    break
        
            # 結果を追加
            x_list = np.vstack((x_list,x_next))
            e_list = np.append(e_list,
                (x_next[2]**2+x_next[3]**2)/2-(x_next[0]**2+x_next[1]**2)**(-1/2))
        
        # 位置・運動量・エネルギーを戻り値として戻す
        return np.vstack((x_list.T, e_list))

    return (adams,)


@app.cell
def _(adams, dxdt, ndarray, np, plt):
    ans_4: ndarray = adams(dxdt, np.array([1, 0, 0, 1]), 100)
    plt.plot(ans_4[0], ans_4[1])
    return (ans_4,)


@app.cell
def _(ans_4, plt):
    plt.ticklabel_format(useOffset=False)
    plt.plot(ans_4[4, 10:])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## シンプレクティック数値積分

    ハミルトン系の保存量を近似的に維持する。
    陽公式はリープフロッグ法をベースとした合成公式。
    陰公式はガウス・ルジャンドル積分を利用。
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### リープ・フロッグ法（陽的解法）
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    リープ・フロッグ法は中点差分による2段階法で2次のシンプレクティック数値積分。
    \begin{align}
        &\vec{q}(t+2dt)\simeq\vec{q}(t)+\vec{p}(t+dt)\cdot2dt,\\
        &\vec{p}(t+2dt)\simeq\vec{p}(t)+\frac{d\vec{p}}{dt}(t+dt)\cdot2dt
    \end{align}
    分離可能なハミルトン系では
    \begin{align}
        &\vec{q}(t+dt/2)\simeq\vec{q}(t)+\frac{d\vec{q}}{dt}(p(t))dt/2,\\
        &\vec{p}(t+dt)\simeq\vec{p}(t)+\frac{d\vec{p}}{dt}(q(t+dt/2))dt,\\
        &\vec{q}(t+dt)\simeq\vec{q}(t+dt/2)+\frac{d\vec{q}}{dt}(p(t+dt))dt/2
    \end{align}
    となる。
    """)
    return


@app.cell
def _(Callable, List, math, ndarray, np):
    def leap_flog(dxdt: List[Callable[[ndarray], float]],
              init: ndarray, t_end: float) -> ndarray:
        # 次元チェック
        if len(dxdt) != len(init):
            raise Exception("Dimension error!")

        # ソルバーのパラメータ
        dt: float = 1e-2
        steps: List[int] = range(1, math.floor(t_end/dt), 1)

        # 初期状態
        x_list: ndarray = np.array([init])
        e_list: ndarray = np.array([(init[2]**2+init[3]**2)/2
                                    -(init[0]**2+init[1]**2)**(-1/2)])
        # ソルバー実施
        for t in steps:
            # リープ・フロッグ法
            x_next: ndarray = np.zeros(len(init))

            x_next[0] = x_list[-1,0]+dxdt[0](x_list[-1])*dt/2
            x_next[1] = x_list[-1,1]+dxdt[1](x_list[-1])*dt/2

            x_next[2] = x_list[-1,2]+dxdt[2](x_next)*dt
            x_next[3] = x_list[-1,3]+dxdt[3](x_next)*dt

            x_next[0] = x_next[0]+dxdt[0](x_next)*dt/2
            x_next[1] = x_next[1]+dxdt[1](x_next)*dt/2
    
            # 結果を追加
            x_list = np.vstack((x_list,x_next))
            e_list = np.append(e_list,
                (x_next[2]**2+x_next[3]**2)/2-(x_next[0]**2+x_next[1]**2)**(-1/2))
        
        # 位置・運動量・エネルギーを戻り値として戻す
        return np.vstack((x_list.T, e_list))

    return (leap_flog,)


@app.cell
def _(dxdt, leap_flog, ndarray, plt):
    ans_5: ndarray = leap_flog(dxdt, [1, 0, 0, 1], 100)
    plt.plot(ans_5[0], ans_5[1])
    return (ans_5,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    エネルギーの誤差は振動。エネルギーが近似的に保存される。
    """)
    return


@app.cell
def _(ans_5, plt):
    plt.ticklabel_format(useOffset=False)
    plt.plot(ans_5[4])
    return


if __name__ == "__main__":
    app.run()
