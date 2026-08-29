import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import base64
    import importlib
    import sys
    from pathlib import Path

    import japanize_matplotlib  # noqa: F401
    import marimo as mo
    import numpy as np

    notebook_dir = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
    repo_root = notebook_dir.parents[1]
    output_dir = repo_root / "data" / "others" / "matplotlib"
    output_dir.mkdir(parents=True, exist_ok=True)

    original_sys_path = list(sys.path)
    sys.path = [
        entry
        for entry in sys.path
        if Path(entry or ".").resolve() != notebook_dir.resolve()
    ]
    try:
        importlib.import_module("matplotlib")
        animation = importlib.import_module("matplotlib.animation")
        plt = importlib.import_module("matplotlib.pyplot")
        tableau_colors = importlib.import_module("matplotlib.colors").TABLEAU_COLORS
    finally:
        sys.path = original_sys_path

    return animation, base64, mo, np, output_dir, plt, tableau_colors


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # matplotlib のデモ

    matplotlib の主要な作図パターンを一通り試す。静止画に加えて GIF アニメーションも
    生成し、静的 HTML にそのまま埋め込める形で出力する。
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 静止画
    """)
    return


@app.cell
def _(np, output_dir, plt):
    x_basic = np.linspace(0, 2 * np.pi, 100)
    fig_basic, ax_basic = plt.subplots(figsize=(7, 4))
    ax_basic.plot(x_basic, np.sin(x_basic), label="sin 波")
    ax_basic.plot(x_basic, np.cos(x_basic), label="cos 波")
    ax_basic.set_xlim(0, 2 * np.pi)
    ax_basic.set_ylim(-1.1, 1.1)
    ax_basic.legend(loc="upper left")
    ax_basic.set_title("sin / cos")
    plot_path = output_dir / "plot.png"
    fig_basic.savefig(plot_path, dpi=300, bbox_inches="tight")
    fig_basic
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 色使い

    デフォルト配色と、色相環を使ってシャッフルした配色を比較する。
    """)
    return


@app.cell
def _(np, plt):
    x_lines = np.linspace(0, 10, 100)
    fig_default, ax_default = plt.subplots(figsize=(7, 5))
    for _default_line_index in range(30):
        ax_default.plot(x_lines, _default_line_index * np.ones(len(x_lines)))
    ax_default.set_title("matplotlib のデフォルト配色")
    fig_default
    return (x_lines,)


@app.cell
def _(np, plt):
    color_count = 100
    color_labels = np.arange(color_count)
    rng_colors = np.random.default_rng(0)
    rng_colors.shuffle(color_labels)

    def get_color(seed: int):
        return plt.get_cmap("hsv")(color_labels[seed] / color_count)[:3]

    return (get_color,)


@app.cell
def _(get_color, np, plt, x_lines):
    fig_hsv, ax_hsv = plt.subplots(figsize=(7, 5))
    for _hsv_line_index in range(30):
        ax_hsv.plot(
            x_lines,
            _hsv_line_index * np.ones(len(x_lines)),
            color=get_color(_hsv_line_index),
        )
    ax_hsv.set_title("色相環を使った配色")
    fig_hsv
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 動画

    アニメーションは GIF として保存し、その内容をノートに埋め込む。
    """)
    return


@app.cell
def _():
    frame_count = 100
    time_start = 0.0
    time_end = 10.0
    wave_length = 1.0
    velocity = 0.5
    return frame_count, time_end, time_start, velocity, wave_length


@app.cell
def _(animation, frame_count, np, output_dir, plt, time_end, time_start, velocity, wave_length):
    fig_wave, ax_wave = plt.subplots(figsize=(7, 4))
    x_wave = np.linspace(0, 2 * np.pi, 1000)
    ax_wave.set_xlim(0, 2 * np.pi)
    ax_wave.set_ylim(-1.1, 1.1)
    line_sin, = ax_wave.plot([], [], c="tab:blue", label="sin 波")
    line_cos, = ax_wave.plot([], [], c="tab:orange", label="cos 波")
    ax_wave.legend(loc="upper right")

    def animate(frame: int):
        time_value = time_start + (time_end - time_start) / frame_count * frame
        ax_wave.set_title(f"frame={frame}")
        line_sin.set_data(
            x_wave,
            np.sin(2 * np.pi / wave_length * (x_wave - velocity * time_value)),
        )
        line_cos.set_data(
            x_wave,
            np.cos(2 * np.pi / wave_length * (x_wave - velocity * time_value)),
        )
        return line_sin, line_cos

    wave_animation = animation.FuncAnimation(fig_wave, animate, frames=frame_count, interval=100)
    gif_path = output_dir / "animation.gif"
    wave_animation.save(gif_path, writer="pillow", dpi=120)
    return gif_path


@app.cell
def _(base64, gif_path, mo):
    gif_base64 = base64.b64encode(gif_path.read_bytes()).decode("ascii")
    mo.md(
        f"""
        ### GIF プレビュー
        <img src="data:image/gif;base64,{gif_base64}" alt="sin と cos のアニメーション" />
        """
    )
    return


if __name__ == "__main__":
    app.run()
