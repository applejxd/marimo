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
    # カルマンフィルタ

    ノイズの乗った観測から状態を推定するカルマンフィルタを、まず自前実装で
    仕組みを追い、次に非線形系へ拡張した EKF（拡張カルマンフィルタ）を扱う。
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## この notebook の共通部品

    次のセルは、以降のすべてのアニメーションが共有する出力先とヘルパーを用意する。

    - `artifacts_dir`: GIF の保存先 `notebooks/algorithm/_generated/kalman/`。
      Git の追跡対象外なので、実行するたびに再生成される。
    - `gif_image(path, alt)`: GIF を base64 の data URI に変換して `mo.image` で
      埋め込む。静的 HTML へ書き出したときも外部ファイルを参照せずに再生できる。
    - `save_gif(fig, update, frames, path, interval)`: `FuncAnimation` を GIF として
      保存する。`frames` にはコマ数（`int`）かコマ番号の並び（`range` など）を渡す。
      `interval` は 1 コマの表示時間（ミリ秒）である。GIF は表示時間を 1/100 秒単位で
      しか保持できないため、要求した値がそのまま使われるとは限らない。保存後に
      GIF を読み直し、実際の間隔・再生時間・ファイルサイズを測って返す。

    以降の GIF はすべて `interval=100`（10 コマ/秒）に揃えてある。挙動を目で追う
    には 20 コマ/秒では速すぎるため、意図的に落としてある。
    """)
    return


@app.cell
def _(mo):
    import base64
    from pathlib import Path

    import matplotlib.animation as animation
    import matplotlib.pyplot as plt
    from PIL import Image

    artifacts_dir = Path(__file__).resolve().parent / "_generated" / "kalman"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    def gif_image(path: Path, alt: str):
        encoded = base64.b64encode(path.read_bytes()).decode()
        return mo.image(f"data:image/gif;base64,{encoded}", alt=alt)

    def save_gif(fig, update, frames, path: Path, interval: int) -> str:
        frame_list = list(range(frames)) if isinstance(frames, int) else list(frames)
        anim = animation.FuncAnimation(fig, update, frames=frame_list, interval=interval)
        anim.save(path, writer="pillow")
        plt.close(fig)
        # GIF は間隔を 1/100 秒単位でしか保持できないので、保存後に実測して報告する。
        with Image.open(path) as gif:
            durations = []
            for index in range(gif.n_frames):
                gif.seek(index)
                durations.append(gif.info.get("duration", 0))
        return (
            f"{path.name}: {len(frame_list)} frames, {durations[0]} ms/frame, "
            f"{sum(durations) / 1000:.1f} s, {path.stat().st_size / 1024:.0f} KiB"
        )

    return artifacts_dir, gif_image, save_gif


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Kalman Filters from scratch
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 使用する運動モデル
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    円運動モデル
    $$
    \begin{pmatrix}
    x \\ y
    \end{pmatrix}
    =
    \begin{pmatrix}
    \cos\theta \\
    \sin\theta
    \end{pmatrix}
    $$
    """)
    return


@app.cell
def _():
    import numpy as np

    np.random.seed(42)
    dt = 2 * np.pi / 100
    _t = np.arange(0, 2 * np.pi, dt)
    x = np.sin(_t)
    y = np.cos(_t)
    points = np.vstack((x, y)).T
    return dt, np, points


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    配色を定義。
    詳細は [matplotlib - List of named colors](https://matplotlib.org/stable/gallery/color/named_colors.html#tableau-palette) を参照。
    """)
    return


@app.cell
def _():
    from matplotlib.colors import TABLEAU_COLORS

    blue = TABLEAU_COLORS["tab:blue"]
    orange = TABLEAU_COLORS["tab:orange"]
    green = TABLEAU_COLORS["tab:green"]
    red = TABLEAU_COLORS["tab:red"]
    return blue, green, orange, red


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    軌跡のアニメーション作成。まずはパレット作成。
    """)
    return


@app.cell
def _():
    import matplotlib.pyplot as plt_circle

    circle_fig, circle_ax = plt_circle.subplots()
    circle_ax.set_aspect('equal')
    circle_ax.set_xlim(-1.3, 1.3)
    circle_ax.set_ylim(-1.3, 1.3)
    return circle_ax, circle_fig, plt_circle


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    アニメーション描画
    """)
    return


@app.cell
def _(artifacts_dir, blue, circle_ax, circle_fig, gif_image, orange, points, save_gif):
    circle_traj_plt, = circle_ax.plot([], [], c=blue, label='Trajectory')
    circle_cur_pos_plt, = circle_ax.plot([], [], c=orange, marker='o', markersize=5, label='Current Position')
    circle_ax.legend(loc='upper right', fontsize='x-small')

    def circle_anim_callback(i):
        circle_ax.set_title(f'Frame {i}')
        circle_traj_plt.set_data(points[:i, 0], points[:i, 1])
        circle_cur_pos_plt.set_data([points[i, 0]], [points[i, 1]])

    circle_gif_path = artifacts_dir / "circle_motion.gif"
    print(save_gif(circle_fig, circle_anim_callback, points.shape[0], circle_gif_path, 100))
    move_animation = gif_image(circle_gif_path, "円運動する真値の軌跡")
    return (move_animation,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    再生
    """)
    return


@app.cell
def _(move_animation):
    move_animation
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### EKF
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    まずは正規分布ノイズを与えて疑似観測データを作成
    """)
    return


@app.cell
def _(np, points):
    noise_sigma = 0.4
    circle_obs_rng = np.random.default_rng(43)
    circle_obs_noise = circle_obs_rng.normal(0, noise_sigma ** 2, (points.shape[0], 2))
    points_obs = points + circle_obs_noise
    return noise_sigma, points_obs


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    EKF は非線形運動モデルを線形近似して Kalman Filter を適用するだけ。
    Kalman Filter の公式は[ここ](https://ja.wikipedia.org/wiki/%E3%82%AB%E3%83%AB%E3%83%9E%E3%83%B3%E3%83%95%E3%82%A3%E3%83%AB%E3%82%BF%E3%83%BC)を参照。
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    今回の EKF での運動モデルの近似式は次の通り。
    $$
    \begin{pmatrix}
    x_{k+1} \\ y_{k+1}
    \end{pmatrix}
    =
    \begin{pmatrix}
    \cos\theta_{k+1} \\ \sin\theta_{k+1}
    \end{pmatrix}
    =
    \begin{pmatrix}
    \cos(\theta_k+\delta\theta) \\ \sin(\theta_k+\delta\theta)
    \end{pmatrix}
    =
    \begin{pmatrix}
    1 & -\delta\theta \\
    \delta\theta & 1
    \end{pmatrix}
    \begin{pmatrix}
    x_k \\ y_k
    \end{pmatrix}
    +\frac{1}{2}
    \begin{pmatrix}
     \cos\theta_0\delta\theta^2 \\
     \sin\theta_0\delta\theta^2
    \end{pmatrix}
    +\mathcal{O}(\delta\theta^3)
    $$
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    誤差絶対値の上限値を用いて次の近似をする。
    $$
    \begin{pmatrix}
    x_{k+1} \\ y_{k+1}
    \end{pmatrix}
    \simeq
    \begin{pmatrix}
    1 & -\delta\theta \\
    \delta\theta & 1
    \end{pmatrix}
    \begin{pmatrix}
    x_k \\ y_k
    \end{pmatrix}
    +w_k,\quad
    w_k\sim N\left(0,\frac{1}{2}\delta\theta^2I\right)
    $$
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    観測モデルは$(x,y)$変数の多変量正規分布とする。
    $$
    \begin{pmatrix}
    x_k' \\ y_k'
    \end{pmatrix}
    =
    \begin{pmatrix}
    x_k \\ y_k
    \end{pmatrix}
    +v_k,\quad
    v_k\sim N(0,\sigma^2 I)
    $$
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    差分化の刻み幅（FPS）は短いほど拡張カルマンフィルタの精度は良くなる。
    """)
    return


@app.cell
def _(np):
    class EKF:

        def __init__(self, state: np.ndarray, state_error: np.ndarray, pred_mat: np.ndarray, pred_error: np.ndarray, obs_mat: np.ndarray, obs_error: np.ndarray):
            self.state = state
            self.state_error = state_error
            self.pred_error = pred_error
            self.obs_error = obs_error
            self.pred_mat = pred_mat
            self.obs_mat = obs_mat

        def predict(self):
            self.state = self.pred_mat @ self.state
            self.state_error = self.pred_mat @ self.state_error @ self.pred_mat.T + self.pred_error

        def update(self, observation: np.ndarray):
            innovation = observation - self.obs_mat @ self.state
            innovation_cov = self.obs_mat @ self.state_error @ self.obs_mat.T + self.obs_error
            kalman_gain = self.state_error @ self.obs_mat @ np.linalg.inv(innovation_cov)
            self.state = self.state + kalman_gain @ innovation
            self.state_error = (np.eye(len(self.state)) - kalman_gain) @ self.state_error  # カルマンゲインを計算  # フィルタリング

    return (EKF,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    EKF 前準備
    """)
    return


@app.cell
def _(EKF, dt, noise_sigma, np, points_obs):
    # 初期状態作成
    init_state = points_obs[0, :]
    init_state_error = np.array([[noise_sigma ** 2, 0], [0, noise_sigma ** 2]])
    pred_mat = np.array([[1, dt], [-dt, 1]])
    # EKF モデル設定 (時計回り回転)
    _pred_error = np.array([[0.5 * dt ** 2, 0], [0, 0.5 * dt ** 2]])
    obs_mat = np.eye(2)
    _obs_error = np.array([[noise_sigma ** 2, 0], [0, noise_sigma ** 2]])
    # EKF インスタンス生成
    kf = EKF(state=init_state, state_error=init_state_error, pred_mat=pred_mat, pred_error=_pred_error, obs_mat=obs_mat, obs_error=_obs_error)
    return init_state, init_state_error, kf


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    EKF 実行
    """)
    return


@app.cell
def _(init_state, init_state_error, kf, np, points_obs):
    # 初期化
    _pred_states = [np.array([0, 1])]
    kf_states = [init_state]
    kf_errors = [init_state_error]
    for _i in range(0, points_obs.shape[0] - 1):
    # 逐次ステップ計算
        _pred_state = _pred_states[-1] @ kf.pred_mat
        _pred_states.append(_pred_state)  # フィルタなし予測
        kf.predict()
        kf.update(points_obs[_i + 1, :])
        kf_states.append(kf.state)
        kf_errors.append(kf.state_error)  # カルマンフィルタ
    _pred_states = np.vstack(_pred_states)
    kf_states = np.vstack(kf_states)
    # np.ndarray に変換
    kf_errors = np.dstack(kf_errors)  # データ保存
    return kf_errors, kf_states


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    誤差共分散行列を楕円で表すためのヘルパー関数
    """)
    return


@app.cell
def _(np):
    def cov2ellipse(cov: np.ndarray):
        eig_val, eig_vec = np.linalg.eigh(cov)
        angle = np.arctan2(eig_vec[1, 0], eig_vec[0, 0])
        width = np.sqrt(eig_val[0])
        height = np.sqrt(eig_val[1])
        return width, height, angle

    return (cov2ellipse,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### 可視化
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    結果を描画。まずはパレット作成。
    """)
    return


@app.cell
def _():
    import matplotlib.pyplot as plt_ekf

    ekf_fig, ekf_ax = plt_ekf.subplots()
    ekf_ax.set_aspect('equal')
    ekf_ax.set_xlim(-1.3, 1.3)
    ekf_ax.set_ylim(-1.3, 1.3)
    return ekf_ax, ekf_fig, plt_ekf


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    データ・軌跡のレイヤー作成
    """)
    return


@app.cell
def _(blue, ekf_ax, green, orange, red):
    gt_traj_plt, = ekf_ax.plot([], [], c=blue, label="GT Trajectory")
    gt_pos_plt, = ekf_ax.plot(
        [], [], c=orange, marker="o", markersize=5, label="True Point"
    )

    obs_pos_scat = ekf_ax.scatter([], [], s=5, c=red, label="Observation")

    kf_pos_plt, = ekf_ax.plot(
        [], [], c=green, marker="o", markersize=5, label="Kalman Filter"
    )
    return gt_pos_plt, gt_traj_plt, kf_pos_plt, obs_pos_scat


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    誤差範囲の描画レイヤー作成
    """)
    return


@app.cell
def _(ekf_ax, green, noise_sigma, red):
    from matplotlib.patches import Ellipse

    # 観測誤差範囲
    obs_ellipse = Ellipse(
        xy=(0, 1), width=noise_sigma, height=noise_sigma, angle=0,
        color=red, alpha=0.1, animated=True)
    ekf_ax.add_patch(obs_ellipse)

    # 推定誤差範囲
    kalman_ellipse = Ellipse(
        xy=(0, 1), width=noise_sigma, height=noise_sigma, angle=0,
        color=green, alpha=0.3, animated=True)
    ekf_ax.add_patch(kalman_ellipse)
    return Ellipse, kalman_ellipse, obs_ellipse


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    GT・予測データ・観測データ・フュージョン結果を誤差分布付きで描画
    """)
    return


@app.cell
def _(
    artifacts_dir,
    cov2ellipse,
    ekf_ax,
    ekf_fig,
    gif_image,
    gt_pos_plt,
    gt_traj_plt,
    kalman_ellipse,
    kf_errors,
    kf_pos_plt,
    kf_states,
    obs_ellipse,
    obs_pos_scat,
    points,
    points_obs,
    save_gif,
):
    ekf_ax.legend(loc='upper right', fontsize='x-small')

    def ekf_anim_callback(i):
        ekf_ax.set_title(f'Frame {i}')
        gt_traj_plt.set_data(points[: i + 1, 0], points[: i + 1, 1])
        gt_pos_plt.set_data([points[i, 0]], [points[i, 1]])
        obs_pos_scat.set_offsets(points_obs[: i + 1, :])
        obs_ellipse.set_center(points_obs[i, :])
        kf_pos_plt.set_data([kf_states[i, 0]], [kf_states[i, 1]])
        width, height, angle = cov2ellipse(kf_errors[:, :, i])
        kalman_ellipse.set_center([kf_states[i, 0], kf_states[i, 1]])
        kalman_ellipse.set_width(width)
        kalman_ellipse.set_height(height)
        kalman_ellipse.set_angle(angle)

    ekf_gif_path = artifacts_dir / "ekf.gif"
    print(save_gif(ekf_fig, ekf_anim_callback, len(points_obs), ekf_gif_path, 100))
    ekf_animation = gif_image(ekf_gif_path, "EKF による円運動の推定")
    return (ekf_animation,)


@app.cell
def _(ekf_animation):
    ekf_animation
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### UKF
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    https://inzkyk.xyz/kalman_filter/unscented_kalman_filter/#section:10.5
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    シグマ点
    $$
    \begin{aligned}
    &\vec{s}^0=\vec{\mu},\quad
    w_m^0=\frac{\lambda}{n+\lambda},\quad
    w_c^0=\frac{\lambda}{n+\lambda}+(1-\alpha^2+\beta), \\
    &\vec{s}^i=\vec{\mu}+\sqrt{n+\lambda}\sqrt{\Sigma}_{\lceil i\rceil}, \quad
    \vec{s}^{i+n}=\vec{\mu}-\sqrt{n+\lambda}\sqrt{\Sigma}_{\lceil i\rceil}, \\
    &w_m^i=w_c^i=\frac{1}{2(n+\lambda)},\quad
    i=1,2,\ldots,n\,\\
    &\kappa\leq0,\quad\alpha\in[0,1],\quad
    \lambda=\alpha^2(n+\kappa)-n,\quad\beta=2
    \end{aligned}
    $$
    """)
    return


@app.cell
def _(np):
    from typing import Callable

    class UKF:

        def __init__(self, state: np.ndarray, state_error: np.ndarray, pred_map: Callable[[np.ndarray], np.ndarray], pred_error: np.ndarray, obs_map: Callable[[np.ndarray], np.ndarray], obs_error: np.ndarray, alpha: float=0.1, beta: float=2, kappa: float=-1):
            self.state = state
            self.state_error = state_error
            self.pred_error = pred_error
            self.obs_error = obs_error
            self.pred_map = pred_map
            self.obs_map = obs_map
            n = state.shape[0]
            self.lambda_ = alpha ** 2 * (n + kappa) - n
            self.w_m = np.full(2 * n + 1, 1 / (2 * (n + self.lambda_)))
            self.w_c = np.full(2 * n + 1, 1 / (2 * (n + self.lambda_)))
            self.w_m[0] = self.lambda_ / (n + self.lambda_)
            self.w_c[0] = self.lambda_ / (n + self.lambda_) + (1 - alpha ** 2 + beta)
            self.sigmas = self.get_sigmas()

        def predict(self):
            self.sigmas = self.get_sigmas()
            self.sigmas = np.array([self.pred_map(sigma) for sigma in self.sigmas])
            self.state, self.state_error = self.u_transform(self.sigmas, self.pred_error)

        def update(self, observation: np.ndarray):
            obs_sigmas = np.array([self.obs_map(sigma) for sigma in self.sigmas])
            z_mu, z_error = self.u_transform(obs_sigmas, self.obs_error)
            xz_error = np.zeros((len(self.state), len(z_mu)))
            for idx in range(len(self.w_c)):
                xz_error = xz_error + self.w_c[idx] * np.outer(self.sigmas[idx, :] - self.state, obs_sigmas[idx, :] - z_mu)
            kalman_gain = xz_error @ np.linalg.inv(z_error)
            self.state = self.state + kalman_gain @ (observation - z_mu)
            self.state_error = self.state_error - kalman_gain @ z_error @ kalman_gain.T

        def u_transform(self, sigmas, noise_cov):
            mu = 0
            for idx in range(len(self.w_m)):
                mu = mu + self.w_m[idx] * sigmas[idx, :]
            cov_mat = np.zeros_like(noise_cov)
            for idx in range(len(self.w_c)):
                cov_mat = cov_mat + self.w_c[idx] * np.outer(sigmas[idx, :] - mu, sigmas[idx, :] - mu)
            cov_mat = cov_mat + noise_cov
            return (mu, cov_mat)

        def get_sigmas(self):
            n = len(self.state)
            sqrt_mat = np.linalg.cholesky((n + self.lambda_) * self.state_error)
            sigma_plus = self.state + sqrt_mat
            sigma_minus = self.state - sqrt_mat
            sigmas = np.vstack((self.state, sigma_plus, sigma_minus))
            return sigmas

    return (UKF,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    UKF の設定
    """)
    return


@app.cell
def _(UKF, dt, noise_sigma, np, points_obs):
    init_state_1 = points_obs[0, :]
    init_state_error_1 = np.array([[noise_sigma ** 2, 0], [0, noise_sigma ** 2]])

    def pred_map(x):
        return np.array([[np.cos(dt), np.sin(dt)], [-np.sin(dt), np.cos(dt)]]) @ x

    def obs_map(x):
        return x

    pred_error = np.array([[0.5 * dt ** 2, 0], [0, 0.5 * dt ** 2]])
    obs_error = np.array([[noise_sigma ** 2, 0], [0, noise_sigma ** 2]])
    kf_1 = UKF(
        state=init_state_1,
        state_error=init_state_error_1,
        pred_map=pred_map,
        pred_error=pred_error,
        obs_map=obs_map,
        obs_error=obs_error,
    )
    return init_state_1, init_state_error_1, kf_1, pred_map


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    UKF 実行
    """)
    return


@app.cell
def _(init_state_1, init_state_error_1, kf_1, np, points_obs, pred_map):
    _pred_states = [np.array([0, 1])]
    pred_sigmas = [np.array([[0, 1]])]
    kf_states_1 = [init_state_1]
    kf_errors_1 = [init_state_error_1]
    for _i in range(0, points_obs.shape[0] - 1):
        _pred_state = pred_map(_pred_states[-1])
        _pred_states.append(_pred_state)
        kf_1.predict()
        kf_1.update(points_obs[_i + 1, :])
        pred_sigmas.append(np.array(kf_1.sigmas))
        kf_states_1.append(kf_1.state)
        kf_errors_1.append(kf_1.state_error)
    _pred_states = np.vstack(_pred_states)
    kf_states_1 = np.vstack(kf_states_1)
    kf_errors_1 = np.dstack(kf_errors_1)
    return kf_errors_1, kf_states_1


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### 可視化
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    結果を描画。まずはパレット作成。
    """)
    return


@app.cell
def _():
    import matplotlib.pyplot as plt_ukf

    ukf_fig, ukf_ax = plt_ukf.subplots()
    ukf_ax.set_aspect('equal')
    ukf_ax.set_xlim(-1.3, 1.3)
    ukf_ax.set_ylim(-1.3, 1.3)
    return plt_ukf, ukf_ax, ukf_fig


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    データ・軌跡のレイヤー作成
    """)
    return


@app.cell
def _(blue, green, orange, red, ukf_ax):
    gt_traj_plt_1, = ukf_ax.plot([], [], c=blue, label='GT Trajectory')
    gt_pos_plt_1, = ukf_ax.plot([], [], c=orange, marker='o', markersize=5, label='True Point')
    obs_pos_scat_1 = ukf_ax.scatter([], [], s=5, c=red, label='Observation')
    kf_pos_plt_1, = ukf_ax.plot([], [], c=green, marker='o', markersize=5, label='Kalman Filter')
    return gt_pos_plt_1, gt_traj_plt_1, kf_pos_plt_1, obs_pos_scat_1


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    誤差範囲の描画レイヤー作成
    """)
    return


@app.cell
def _(Ellipse, green, noise_sigma, red, ukf_ax):
    obs_ellipse_1 = Ellipse(xy=(0, 1), width=noise_sigma, height=noise_sigma, angle=0, color=red, alpha=0.1, animated=True)
    ukf_ax.add_patch(obs_ellipse_1)
    # 観測誤差範囲
    kalman_ellipse_1 = Ellipse(xy=(0, 1), width=noise_sigma, height=noise_sigma, angle=0, color=green, alpha=0.3, animated=True)
    # 推定誤差範囲
    ukf_ax.add_patch(kalman_ellipse_1)
    return kalman_ellipse_1, obs_ellipse_1


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    GT・予測データ・観測データ・フュージョン結果を誤差分布付きで描画
    """)
    return


@app.cell
def _(
    artifacts_dir,
    cov2ellipse,
    gif_image,
    gt_pos_plt_1,
    gt_traj_plt_1,
    kalman_ellipse_1,
    kf_errors_1,
    kf_pos_plt_1,
    kf_states_1,
    obs_ellipse_1,
    obs_pos_scat_1,
    points,
    points_obs,
    save_gif,
    ukf_ax,
    ukf_fig,
):
    ukf_ax.legend(loc='upper right', fontsize='x-small')

    def ukf_anim_callback(i):
        ukf_ax.set_title(f'Frame {i}')
        gt_traj_plt_1.set_data(points[: i + 1, 0], points[: i + 1, 1])
        gt_pos_plt_1.set_data([points[i, 0]], [points[i, 1]])
        obs_pos_scat_1.set_offsets(points_obs[: i + 1, :])
        obs_ellipse_1.set_center(points_obs[i, :])
        kf_pos_plt_1.set_data([kf_states_1[i, 0]], [kf_states_1[i, 1]])
        width, height, angle = cov2ellipse(kf_errors_1[:, :, i])
        kalman_ellipse_1.set_center([kf_states_1[i, 0], kf_states_1[i, 1]])
        kalman_ellipse_1.set_width(width)
        kalman_ellipse_1.set_height(height)
        kalman_ellipse_1.set_angle(angle)

    ukf_gif_path = artifacts_dir / "ukf.gif"
    print(save_gif(ukf_fig, ukf_anim_callback, len(points_obs), ukf_gif_path, 100))
    ukf_animation = gif_image(ukf_gif_path, "UKF による円運動の推定")
    return (ukf_animation,)


@app.cell
def _(ukf_animation):
    ukf_animation
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Kalman Filter by FilterPy
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    [The Documentation](https://filterpy.readthedocs.io/en/latest/)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### UKF
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    filterpy で UKF のインスタンスを作成。
    API は[ここ](https://filterpy.readthedocs.io/en/latest/kalman/UnscentedKalmanFilter.html)。
    """)
    return


@app.cell
def _(dt, noise_sigma, np, points_obs):
    from filterpy import kalman

    def pred_map_1(x, dt):
        return x @ np.array([[np.cos(dt), -np.sin(dt)], [np.sin(dt), np.cos(dt)]])

    def obs_map_1(x):
        return x

    sigma_points = kalman.MerweScaledSigmaPoints(2, alpha=0.1, beta=2.0, kappa=-1)
    kf_2 = kalman.UnscentedKalmanFilter(
        dim_x=2, dim_z=2, dt=dt, hx=obs_map_1, fx=pred_map_1, points=sigma_points
    )
    kf_2.x = points_obs[0, :]
    kf_2.P = kf_2.P * noise_sigma ** 2
    kf_2.R = np.array([[noise_sigma ** 2, 0], [0, noise_sigma ** 2]])
    kf_2.Q = np.array([[0.5 * dt ** 2, 0], [0, 0.5 * dt ** 2]])
    return kalman, kf_2


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    UKF 実行
    """)
    return


@app.cell
def _(kf_2, np, points_obs):
    kf_states_2 = [kf_2.x]
    kf_errors_2 = [kf_2.P]
    for _i in range(0, points_obs.shape[0] - 1):
        kf_2.predict()
        kf_2.update(points_obs[_i + 1, :])
        kf_states_2.append(kf_2.x)
        kf_errors_2.append(kf_2.P)
    kf_states_2 = np.vstack(kf_states_2)
    kf_errors_2 = np.dstack(kf_errors_2)
    return kf_errors_2, kf_states_2


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### 可視化
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    結果を描画。まずはパレット作成。
    """)
    return


@app.cell
def _():
    import matplotlib.pyplot as plt_filterpy

    filterpy_fig, filterpy_ax = plt_filterpy.subplots()
    filterpy_ax.set_aspect('equal')
    filterpy_ax.set_xlim(-1.3, 1.3)
    filterpy_ax.set_ylim(-1.3, 1.3)
    return filterpy_ax, filterpy_fig, plt_filterpy


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    データ・軌跡のレイヤー作成
    """)
    return


@app.cell
def _(blue, filterpy_ax, green, orange, red):
    gt_traj_plt_2, = filterpy_ax.plot([], [], c=blue, label='GT Trajectory')
    gt_pos_plt_2, = filterpy_ax.plot([], [], c=orange, marker='o', markersize=5, label='True Point')
    filterpy_obs_pos_scat = filterpy_ax.scatter([], [], s=5, c=red, label='Observation')
    kf_pos_plt_2, = filterpy_ax.plot([], [], c=green, marker='o', markersize=5, label='Kalman Filter')
    return filterpy_obs_pos_scat, gt_pos_plt_2, gt_traj_plt_2, kf_pos_plt_2


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    誤差範囲の描画レイヤー作成
    """)
    return


@app.cell
def _(Ellipse, filterpy_ax, green, noise_sigma, red):
    obs_ellipse_2 = Ellipse(xy=(0, 1), width=noise_sigma, height=noise_sigma, angle=0, color=red, alpha=0.1, animated=True)
    filterpy_ax.add_patch(obs_ellipse_2)
    # 観測誤差範囲
    kalman_ellipse_2 = Ellipse(xy=(0, 1), width=noise_sigma, height=noise_sigma, angle=0, color=green, alpha=0.3, animated=True)
    # 推定誤差範囲
    filterpy_ax.add_patch(kalman_ellipse_2)
    return kalman_ellipse_2, obs_ellipse_2


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    GT・予測データ・観測データ・フュージョン結果を誤差分布付きで描画
    """)
    return


@app.cell
def _(
    artifacts_dir,
    cov2ellipse,
    filterpy_ax,
    filterpy_fig,
    filterpy_obs_pos_scat,
    gif_image,
    gt_pos_plt_2,
    gt_traj_plt_2,
    kalman_ellipse_2,
    kf_errors_2,
    kf_pos_plt_2,
    kf_states_2,
    obs_ellipse_2,
    points,
    points_obs,
    save_gif,
):
    filterpy_ax.legend(loc='upper right', fontsize='x-small')

    def filterpy_anim_callback(i):
        filterpy_ax.set_title(f'Frame {i}')
        gt_traj_plt_2.set_data(points[: i + 1, 0], points[: i + 1, 1])
        gt_pos_plt_2.set_data([points[i, 0]], [points[i, 1]])
        filterpy_obs_pos_scat.set_offsets(points_obs[: i + 1, :])
        obs_ellipse_2.set_center(points_obs[i, :])
        kf_pos_plt_2.set_data([kf_states_2[i, 0]], [kf_states_2[i, 1]])
        width, height, angle = cov2ellipse(kf_errors_2[:, :, i])
        kalman_ellipse_2.set_center([kf_states_2[i, 0], kf_states_2[i, 1]])
        kalman_ellipse_2.set_width(width)
        kalman_ellipse_2.set_height(height)
        kalman_ellipse_2.set_angle(angle)

    filterpy_gif_path = artifacts_dir / "filterpy_ekf.gif"
    print(save_gif(filterpy_fig, filterpy_anim_callback, len(points_obs), filterpy_gif_path, 100))
    filterpy_animation = gif_image(filterpy_gif_path, "FilterPy の EKF による円運動の推定")
    return (filterpy_animation,)


@app.cell
def _(filterpy_animation):
    filterpy_animation
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 二重振り子 by FilterPy
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### データ作成
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    [二重振り子のラグランジアン](https://monologue-physics.hatenablog.com/entry/2021/09/22/155932)
    $$
    \begin{aligned}
    L&=\frac{m_1}{2}(\dot{x_1}^2+\dot{y_1}^2)+\frac{m_2}{2}(\dot{x_2}^2+\dot{y_2}^2)-m_1gy_1-m_2gy_2 \\
    &=
    \frac{1}{2}(m_1+m_2)l_1^2\dot{\theta_1}^2
    +\frac{1}{2}m_2l_2^2\dot{\theta_2}^2
    +m_2l_1l_2\dot{\theta_1}\dot{\theta_2}\cos(\theta_1-\theta_2)
    +(m_1+m_2)gl_1\cos(\theta_1)+m_2gl_2\cos(\theta_2)
    \end{aligned}
    $$
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    $\theta_1$ についてのラグランジュ方程式
    $$
    (m_1+m_2)l_1^2\ddot{\theta_1}+(m_1+m_2)gl_1\sin(\theta_1)+m_2l_1l_2\left(\ddot{\theta_2}\cos(\theta_1-\theta_2)+\dot{\theta_2}^2\sin(\theta_1-\theta_2)\right)=0
    $$
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    $\theta_2$ についてのラグランジュ方程式
    $$
    m_2l_2^2\ddot{\theta_2}+m_2l_1l_2\left(\ddot{\theta_1}\cos(\theta_1-\theta_2)-\dot{\theta_1}^2\sin(\theta_1-\theta_2)\right)+m_2gl_2\sin(\theta_2)=0
    $$
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    数値計算するには連立一次方程式である[正準方程式](https://www.aihara.co.jp/~taiji/pendula-equations/present-node2.html)の方が望ましい。
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    $\vec{\theta}\equiv(\theta_1, \theta_2)^T$ に対する正準方程式
    $$
    \dot{\vec{\theta}}
    =\frac{1}{(m_1+m_2)l_1^2m_2l_2^2-m_2^2l_1^2l_2^2\cos(\theta_1-\theta_2)}
    \begin{pmatrix}
    m_2l_2^2p_1-m_2l_1l_2\cos(\theta_1-\theta_2)p_2 \\
    -m_2l_1l_2\cos(\theta_1-\theta_2)+(m_1+m_2)l_1^2p_2
    \end{pmatrix}
    $$
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    $\vec{p}\equiv(p_1,p_2)^T$に対する正準方程式
    $$
    \dot{\vec{p}}=
    \begin{pmatrix}
    -m_2l_1l_2\sin(\theta_1-\theta_2)\dot{\theta_1}\dot{\theta_2}-(m_1+m_2)gl_1\sin(\theta_1) \\
    m_2l_1l_2\sin(\theta_1-\theta_2)\dot{\theta_1}\dot{\theta_2}-m_2gl_2\sin(\theta_2) \\
    \end{pmatrix}
    $$
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    簡便のため $m_1=m_2=l_1=l_2=1$ として次を得る。
    $$
    \begin{aligned}
    &\dot{\vec{\theta}}
    =\frac{1}{2-\cos(\theta_1-\theta_2)}
    \begin{pmatrix}
    p_1-\cos(\theta_1-\theta_2)p_2 \\
    -\cos(\theta_1-\theta_2)+2p_2
    \end{pmatrix} \\
    &\dot{\vec{p}}=
    \begin{pmatrix}
    -\sin(\theta_1-\theta_2)\dot{\theta_1}\dot{\theta_2}-2g\sin(\theta_1) \\
    \sin(\theta_1-\theta_2)\dot{\theta_1}\dot{\theta_2}-g\sin(\theta_2) \\
    \end{pmatrix}
    \end{aligned}
    $$
    """)
    return


@app.cell
def _(np):
    def dp_fmap(x: np.ndarray, dt: float):
        theta1, theta2, p1, p2 = x
        cos = np.cos(theta1 - theta2)
        sin = np.sin(theta1 - theta2)

        d_theta1 = (p1 - cos * p2) / (2-cos)
        d_theta2 = -cos + 2 * p2 / (2-cos)

        g = 9.8
        d_p1 = -sin * d_theta1 * d_theta2 - 2 * g * np.sin(theta1)
        d_p2 = sin * d_theta1 * d_theta2 - g * np.sin(theta2)

        theta1_next = theta1 + d_theta1 * dt
        theta2_next = theta2 + d_theta2 * dt
        p1_next = p1 + d_p1 * dt
        p2_next = p2 + d_p2 * dt

        return np.array([theta1_next, theta2_next, p1_next, p2_next])

    def dp_hmap(x: np.ndarray):
        # 運動量は使用しない
        theta1, theta2, _, _ = x

        x1 = np.sin(theta1)
        y1 = -np.cos(theta1)
        x2 = x1 + np.sin(theta2)
        y2 = y1 - np.cos(theta2)

        return np.array([x1, y1, x2, y2])

    return dp_fmap, dp_hmap


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    観測データを生成
    """)
    return


@app.cell
def _(dp_fmap, dp_hmap, noise_sigma, np):
    dt_1 = 0.02
    times = np.arange(0, 10, dt_1)
    x_list = [np.array([np.pi / 2, np.pi / 2, 0, 0])]
    for _t in times:
        x_list.append(dp_fmap(x_list[-1], dt_1))
    x_list = np.vstack(x_list)
    dp_obs_rng = np.random.default_rng(44)
    dp_obs_noise = dp_obs_rng.normal(0, noise_sigma ** 2, (len(x_list), 4))
    x_list = np.array([dp_hmap(x) for x in x_list])
    obs_list = np.array([x + dp_obs_noise[i] for i, x in enumerate(x_list)])
    return dt_1, obs_list, x_list


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    可視化。まずはパレット作成。
    """)
    return


@app.cell
def _(blue, orange, red):
    import matplotlib.pyplot as plt_dp

    dp_fig, dp_ax = plt_dp.subplots()
    dp_ax.set_aspect('equal')
    dp_ax.set_xlim(-2.3, 2.3)
    dp_ax.set_ylim(-2.3, 1.5)
    dp_traj_plt, = dp_ax.plot([], [], c=blue, label="Trajectory")
    dp_cur_pos_plt, = dp_ax.plot(
        [], [], c=orange, marker="o", markersize=5,
        label="Current Position")
    dp_obs_pos_scat = dp_ax.scatter([], [], s=5, c=red, label="Observation")
    dp_ax.legend(loc="upper right", fontsize="x-small")
    return dp_ax, dp_cur_pos_plt, dp_fig, dp_obs_pos_scat, dp_traj_plt, plt_dp


@app.cell
def _(
    Ellipse,
    artifacts_dir,
    dp_ax,
    dp_cur_pos_plt,
    dp_fig,
    dp_obs_pos_scat,
    dp_traj_plt,
    gif_image,
    noise_sigma,
    obs_list,
    red,
    save_gif,
    x_list,
):
    obs_ellipse_3 = Ellipse(xy=(0, 1), width=noise_sigma, height=noise_sigma, angle=0, color=red, alpha=0.1, animated=True)
    dp_ax.add_patch(obs_ellipse_3)

    def dp_anim_callback(i):
        dp_ax.set_title(f'Frame {i}')
        dp_traj_plt.set_data(x_list[:i, 2], x_list[:i, 3])
        dp_cur_pos_plt.set_data([0, *x_list[i, [0, 2]]], [0, *x_list[i, [1, 3]]])
        dp_obs_pos_scat.set_offsets(obs_list[: i + 1, [2, 3]])
        obs_ellipse_3.set_center(obs_list[i, [2, 3]])

    skip = 4
    dp_gif_path = artifacts_dir / "double_pendulum.gif"
    print(save_gif(dp_fig, dp_anim_callback, range(0, len(obs_list), skip), dp_gif_path, 100))
    dp_animation = gif_image(dp_gif_path, "二重振り子の真値と観測")
    return (dp_animation,)


@app.cell
def _(dp_animation):
    dp_animation
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### UKF by filterpy
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    UKF モデル作成
    """)
    return


@app.cell
def _(dp_fmap, dp_hmap, dt_1, kalman, noise_sigma, np, obs_list):
    _sigma_points = kalman.MerweScaledSigmaPoints(4, alpha=0.1, beta=2.0, kappa=-1)
    kf_3 = kalman.UnscentedKalmanFilter(dim_x=4, dim_z=4, dt=dt_1, hx=dp_hmap, fx=dp_fmap, points=_sigma_points)
    kf_3.x = obs_list[0, :]
    kf_3.P = kf_3.P * noise_sigma ** 2
    kf_3.R = noise_sigma ** 2 * np.eye(4)
    kf_3.Q = 2 * dt_1 ** 2 * np.eye(4)
    return (kf_3,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    UKF 実行
    """)
    return


@app.cell
def _(dp_hmap, kf_3, np, obs_list):
    kf_states_3 = [kf_3.x]
    kf_errors_3 = [kf_3.P]
    for _i in range(0, obs_list.shape[0] - 1):
        kf_3.predict()
        kf_3.update(obs_list[_i + 1, :])
        dx_dtheta = np.array([[np.cos(kf_3.x[0]), 0], [np.sin(kf_3.x[1]), 0], [np.cos(kf_3.x[0]), np.cos(kf_3.x[1])], [np.sin(kf_3.x[1]), np.sin(kf_3.x[1])]])
        kf_states_3.append(dp_hmap(kf_3.x))
        kf_errors_3.append(dx_dtheta @ kf_3.P[:2, :2] @ dx_dtheta.T)
    kf_states_3 = np.vstack(kf_states_3)
    kf_errors_3 = np.dstack(kf_errors_3)
    return kf_errors_3, kf_states_3


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    結果の描画。まずはパレット作成。
    """)
    return


@app.cell
def _():
    import matplotlib.pyplot as plt_dp_ukf

    dp_ukf_fig, dp_ukf_ax = plt_dp_ukf.subplots()
    dp_ukf_ax.set_aspect('equal')

    # 描画範囲
    dp_ukf_ax.set_xlim(-2.3, 2.3)
    dp_ukf_ax.set_ylim(-2.3, 2)
    return dp_ukf_ax, dp_ukf_fig, plt_dp_ukf


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    レイヤー作成
    """)
    return


@app.cell
def _(blue, dp_ukf_ax, green, orange, red):
    # 軌跡描画のレイヤー
    gt_traj_plt_3, = dp_ukf_ax.plot([], [], c=blue, label='GT Trajectory')
    gt_pos_plt_3, = dp_ukf_ax.plot([], [], c=orange, marker='o', markersize=5, label='True Point')
    obs_pos_scat_3 = dp_ukf_ax.scatter([], [], s=5, c=red, label='Observation')
    kf_pos_plt_3, = dp_ukf_ax.plot([], [], c=green, marker='o', markersize=5, label='Kalman Filter')
    return gt_pos_plt_3, gt_traj_plt_3, kf_pos_plt_3, obs_pos_scat_3


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    誤差範囲描画のレイヤー作成
    """)
    return


@app.cell
def _(Ellipse, dp_ukf_ax, green, noise_sigma, red):
    # 観測誤差範囲
    obs_ellipse_4 = Ellipse(xy=(0, 1), width=noise_sigma, height=noise_sigma, angle=0, color=red, alpha=0.1, animated=True)
    dp_ukf_ax.add_patch(obs_ellipse_4)
    kalman_ellipse_3 = Ellipse(xy=(0, 1), width=noise_sigma, height=noise_sigma, angle=0, color=green, alpha=0.3, animated=True)
    # 推定誤差範囲
    dp_ukf_ax.add_patch(kalman_ellipse_3)
    return kalman_ellipse_3, obs_ellipse_4


@app.cell
def _(
    artifacts_dir,
    cov2ellipse,
    dp_ukf_ax,
    dp_ukf_fig,
    gif_image,
    gt_pos_plt_3,
    gt_traj_plt_3,
    kalman_ellipse_3,
    kf_errors_3,
    kf_pos_plt_3,
    kf_states_3,
    obs_ellipse_4,
    obs_list,
    obs_pos_scat_3,
    save_gif,
    x_list,
):
    dp_ukf_ax.legend(loc='upper right', fontsize='x-small')

    def dp_ukf_anim_callback(i):
        dp_ukf_ax.set_title(f'Frame {i}')
        gt_traj_plt_3.set_data(x_list[: i + 1, 2], x_list[: i + 1, 3])
        gt_pos_plt_3.set_data([0, *x_list[i, [0, 2]]], [0, *x_list[i, [1, 3]]])
        obs_pos_scat_3.set_offsets(obs_list[: i + 1, [2, 3]])
        obs_ellipse_4.set_center(obs_list[i, [2, 3]])
        kf_pos_plt_3.set_data([kf_states_3[i, 2]], [kf_states_3[i, 3]])
        width, height, angle = cov2ellipse(kf_errors_3[2:, 2:, i])
        kalman_ellipse_3.set_center([kf_states_3[i, 2], kf_states_3[i, 3]])
        kalman_ellipse_3.set_width(width)
        kalman_ellipse_3.set_height(height)
        kalman_ellipse_3.set_angle(angle)

    dp_ukf_skip = 4
    dp_ukf_gif_path = artifacts_dir / "double_pendulum_ukf.gif"
    print(
        save_gif(
            dp_ukf_fig,
            dp_ukf_anim_callback,
            range(0, len(obs_list), dp_ukf_skip),
            dp_ukf_gif_path,
            100,
        )
    )
    dp_ukf_animation = gif_image(dp_ukf_gif_path, "UKF による二重振り子の推定")
    return (dp_ukf_animation,)


@app.cell
def _(dp_ukf_animation):
    dp_ukf_animation
    return


if __name__ == "__main__":
    app.run()
