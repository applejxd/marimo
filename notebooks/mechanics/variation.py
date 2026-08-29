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
    # 変分原理のデモ
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    調和振動子の作用（汎関数）
    $$
        S[q]
        =\int dt \left(\frac{1}{2}m\dot{q}(t)^2-\frac{1}{2}kq(t)^2\right)
    $$
    に対する変分原理を最適化問題として解く。
    """)
    return


@app.cell
def _():
    import matplotlib.pyplot as plt
    import numpy as np

    t_list = np.arange(0, 10, 0.01)
    dt = t_list[1] - t_list[0]
    print(f"len(t_list)={len(t_list)}, dt = {dt}")
    q_first, q_last = -5, 5
    q_list = np.linspace(q_first, q_last, len(t_list)-2)

    plt.plot(t_list, np.hstack([q_first, q_list, q_last]))
    plt.xlabel("t")
    plt.ylabel("q(t)")
    plt.show()
    return dt, np, plt, q_first, q_last, q_list, t_list


@app.cell
def _(dt, np, plt, q_first, q_last, q_list, t_list):
    def dq_list(q_list: np.ndarray) -> np.ndarray:
        q_list_all = np.hstack([q_first, q_list, q_last])
        result = np.array([(q_list_all[idx+1] - q_list_all[idx-1]) / (2 * dt)
                            for idx in range(1, len(q_list_all)-1)])
        return result

    print(f"len(dq_list)={len(dq_list(q_list))}")
    plt.plot(t_list[1:-1], dq_list(q_list))
    plt.xlabel("t")
    plt.ylabel("dq/dt")
    plt.show()
    return (dq_list,)


@app.cell
def _(dq_list, dt, np, q_list):
    m, k = 1, 1

    def action(q_list: np.ndarray) -> float:
        kinetic_energy = 0.5 * m * np.sum(dq_list(q_list)**2) * dt
        potential_energy = 0.5 * k * np.sum(q_list**2) * dt
        return kinetic_energy - potential_energy

    print(f"action = {action(q_list)}")
    return action, k, m


@app.cell
def _(dt, k, m, np, plt, q_first, q_last, q_list, t_list):
    def ddq_list(q_list: np.ndarray) -> np.ndarray:
        q_list_all = np.hstack([q_first, q_list, q_last])
        result = np.array([(q_list_all[idx+1] - 2*q_list_all[idx] + q_list_all[idx-1]) / dt
                            for idx in range(1, len(q_list_all)-1)])
        return result

    def eom(q_list: np.ndarray) -> np.ndarray:
        return m * ddq_list(q_list) - k * q_list

    plt.plot(t_list[1:-1], eom(q_list))
    plt.xlabel("t")
    plt.ylabel("eom")
    plt.show()
    return (eom,)


@app.cell
def _(action, eom, plt, q_list, t_list):
    from scipy.optimize import minimize

    def minimize_callback(xk) -> bool:
        print(f"action={action(xk)}")

    res = minimize(action, q_list, method="CG", callback=minimize_callback)
    plt.plot(t_list[1:-1], eom(res.x))
    plt.xlabel("t")
    plt.ylabel("eom")
    plt.show()
    return (res,)


@app.cell
def _(np, plt, q_first, q_last, res, t_list):
    plt.plot(t_list, np.hstack([q_first, res.x, q_last]))
    return


if __name__ == "__main__":
    app.run()
