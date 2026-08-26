import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    from qulacs import QuantumCircuit, QuantumState
    from qulacs.gate import RZ, H, X, Z, to_matrix_gate
    from qulacs.state import inner_product

    return H, QuantumCircuit, QuantumState, RZ, X, Z, inner_product, mo, np, plt, to_matrix_gate


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Grover のアルゴリズム

    Jupyter / Colab マジックを除去し、量子状態の分布を静止画として追えるように整理した。
    目標状態は $|11\ldots1\rangle$ とする。
    """)
    return


@app.cell
def _(np, plt):
    def distribution_figure(state, nqubits: int, title: str):
        figure, axis = plt.subplots(figsize=(7, 3))
        amplitudes = np.abs(state.get_vector())
        axis.bar(range(2**nqubits), amplitudes)
        axis.set_xlabel("basis state")
        axis.set_ylabel("|amplitude|")
        axis.set_title(title)
        return figure

    return (distribution_figure,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 量子状態を初期化
    """)
    return


@app.cell
def _(QuantumState, distribution_figure):
    nqubits = 5
    zero_state = QuantumState(nqubits)
    zero_state.set_zero_state()
    distribution_figure(zero_state, nqubits, "zero state")
    return nqubits, zero_state


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Hadamard 変換で一様状態にする
    """)
    return


@app.cell
def _(H, QuantumCircuit):
    def make_hadamard(nqubits: int):
        hadamard = QuantumCircuit(nqubits)
        for qubit in range(nqubits):
            hadamard.add_gate(H(qubit))
        return hadamard

    return (make_hadamard,)


@app.cell
def _(distribution_figure, make_hadamard, nqubits, zero_state):
    hadamard = make_hadamard(nqubits)
    uniform_state = zero_state.copy()
    hadamard.update_quantum_state(uniform_state)
    distribution_figure(uniform_state, nqubits, "uniform superposition")
    return hadamard, uniform_state


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Oracle を作る
    """)
    return


@app.cell
def _(QuantumCircuit, Z, to_matrix_gate):
    def make_oracle(nqubits: int):
        oracle = QuantumCircuit(nqubits)
        controlled_z = to_matrix_gate(Z(nqubits - 1))
        for qubit in range(nqubits - 1):
            controlled_z.add_control_qubit(qubit, 1)
        oracle.add_gate(controlled_z)
        return oracle

    return (make_oracle,)


@app.cell
def _(QuantumState, distribution_figure, make_oracle, nqubits, uniform_state):
    oracle = make_oracle(nqubits)
    oracle_state = uniform_state.copy()
    oracle.update_quantum_state(oracle_state)
    distribution_figure(oracle_state, nqubits, "after oracle")
    return (oracle,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 拡散変換
    """)
    return


@app.cell
def _(H, QuantumCircuit, RZ, X, Z, np, to_matrix_gate):
    def make_diffusion(nqubits: int):
        diffusion = QuantumCircuit(nqubits)
        for qubit in range(nqubits):
            diffusion.add_gate(H(qubit))
        diffusion.add_gate(to_matrix_gate(RZ(nqubits - 1, 2 * np.pi)))
        diffusion.add_gate(X(nqubits - 1))
        controlled_z = to_matrix_gate(Z(nqubits - 1))
        for qubit in range(nqubits - 1):
            controlled_z.add_control_qubit(qubit, 0)
        diffusion.add_gate(controlled_z)
        diffusion.add_gate(X(nqubits - 1))
        for qubit in range(nqubits):
            diffusion.add_gate(H(qubit))
        return diffusion

    return (make_diffusion,)


@app.cell
def _(QuantumState, distribution_figure, hadamard, make_diffusion, nqubits, oracle):
    diffusion = make_diffusion(nqubits)
    one_step_state = QuantumState(nqubits)
    one_step_state.set_zero_state()
    hadamard.update_quantum_state(one_step_state)
    oracle.update_quantum_state(one_step_state)
    diffusion.update_quantum_state(one_step_state)
    distribution_figure(one_step_state, nqubits, "after one Grover iteration")
    return (diffusion,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 反復で正答確率が上がる様子
    """)
    return


@app.cell
def _(QuantumState, diffusion, hadamard, inner_product, make_hadamard, nqubits, np, oracle, plt):
    target_state = QuantumState(nqubits)
    target_state.set_computational_basis(2**nqubits - 1)
    state_iter = QuantumState(nqubits)
    state_iter.set_zero_state()
    hadamard.update_quantum_state(state_iter)

    probabilities = []
    snapshots = []
    for _iteration in range(4):
        oracle.update_quantum_state(state_iter)
        diffusion.update_quantum_state(state_iter)
        probabilities.append(np.linalg.norm(inner_product(state_iter, target_state)))
        snapshots.append(np.abs(state_iter.get_vector()))

    figure_iter, axes_iter = plt.subplots(2, 2, figsize=(12, 6), sharey=True)
    for axis_iter, iteration_label, amplitudes, probability in zip(axes_iter.ravel(), range(1, 5), snapshots, probabilities, strict=True):
        axis_iter.bar(range(2**nqubits), amplitudes)
        axis_iter.set_title(f"k={iteration_label}, success={probability:.3f}")
    figure_iter.tight_layout()
    figure_iter
    return probabilities


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 10 量子ビットでの反復回数依存性
    """)
    return


@app.cell
def _(QuantumState, inner_product, make_diffusion, make_hadamard, make_oracle, np, plt):
    nqubits_large = 10
    target_state_large = QuantumState(nqubits_large)
    target_state_large.set_computational_basis(2**nqubits_large - 1)
    hadamard_large = make_hadamard(nqubits_large)
    oracle_large = make_oracle(nqubits_large)
    diffusion_large = make_diffusion(nqubits_large)
    state_large = QuantumState(nqubits_large)
    state_large.set_zero_state()
    hadamard_large.update_quantum_state(state_large)

    success_curve = []
    for _iteration in range(30):
        oracle_large.update_quantum_state(state_large)
        diffusion_large.update_quantum_state(state_large)
        success_curve.append(np.linalg.norm(inner_product(state_large, target_state_large)))

    max_iteration = int(np.argmax(success_curve)) + 1
    print(f"maximal probability {success_curve[max_iteration - 1]:.5e} is obtained at k = {max_iteration}")
    figure_curve, axis_curve = plt.subplots(figsize=(7, 4))
    axis_curve.plot(np.arange(1, 31), success_curve, "o-")
    axis_curve.set_xlabel("iteration")
    axis_curve.set_ylabel("success probability")
    axis_curve.set_title("10-qubit Grover search")
    figure_curve
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 量子ビット数と最適反復回数の関係
    """)
    return


@app.cell
def _(QuantumState, inner_product, make_diffusion, make_hadamard, make_oracle, np, plt):
    scaling_rows = []
    for nqubits_scaling in range(6, 17, 2):
        target_state_scaling = QuantumState(nqubits_scaling)
        target_state_scaling.set_computational_basis(2**nqubits_scaling - 1)
        hadamard_scaling = make_hadamard(nqubits_scaling)
        oracle_scaling = make_oracle(nqubits_scaling)
        diffusion_scaling = make_diffusion(nqubits_scaling)
        state_scaling = QuantumState(nqubits_scaling)
        state_scaling.set_zero_state()
        hadamard_scaling.update_quantum_state(state_scaling)

        best_probability = 0.0
        best_iteration = 0
        for _iteration_count in range(1, 1001):
            oracle_scaling.update_quantum_state(state_scaling)
            diffusion_scaling.update_quantum_state(state_scaling)
            success_probability = np.linalg.norm(inner_product(state_scaling, target_state_scaling))
            if success_probability >= best_probability:
                best_probability = success_probability
                best_iteration = _iteration_count
            else:
                break
        scaling_rows.append((nqubits_scaling, best_iteration, best_probability))
        print(
            f"nqubits={nqubits_scaling}, num_iter={best_iteration}, suc_prob={best_probability:.5e}"
        )

    scaling_array = np.array(scaling_rows)
    figure_scale, axis_scale = plt.subplots(figsize=(7, 4))
    axis_scale.semilogy(scaling_array[:, 0], scaling_array[:, 1], "o-", label="experiment")
    axis_scale.semilogy(scaling_array[:, 0], 0.05 * 2 ** scaling_array[:, 0], "-", label="$\\propto N=2^n$")
    axis_scale.semilogy(scaling_array[:, 0], 2 ** (0.5 * scaling_array[:, 0]), "-", label="$\\propto \\sqrt{N}=2^{n/2}$")
    axis_scale.set_xlabel("n, # of qubits")
    axis_scale.set_ylabel("k, # of iterations")
    axis_scale.legend(fontsize=10)
    figure_scale
    return


if __name__ == "__main__":
    app.run()
