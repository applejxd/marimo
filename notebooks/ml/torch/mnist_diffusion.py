import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")

with app.setup:
    import base64
    import io
    import math
    import time

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from PIL import Image


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # MNIST で拡散モデル

    拡散モデルは、画像へ少しずつノイズを加える **forward process** と、その逆をたどって
    ノイズから画像を作る **reverse process** の組で定義されます。この notebook では
    条件付き denoising diffusion probabilistic model（DDPM）を MNIST で学習し、
    ラベルを指定して数字を生成します。

    生成品質は見た目の印象に頼らず、別途学習した MNIST 分類器で
    「指定したラベル通りの数字が生成されたか」を測って確認します。
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 共通部品と再現性

    冒頭の `with app.setup` ブロックで、notebook 全体から使うライブラリを読み込んでいます。
    marimo、Matplotlib、NumPy、pandas、PyTorch に加え、生成過程を GIF にするため
    `base64`、`io`、Pillow の `Image`、ステップ埋め込みの計算に使う `math`、
    学習時間の計測に使う `time` を読み込んでいます。

    次のセルでは乱数 seed を 42 に固定し、CUDA、MPS、CPU の順に利用可能な device を選びます。
    """)
    return


@app.cell
def _():
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        device = torch.device("cuda")
    elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"selected device: {device}")
    pd.DataFrame({"item": ["torch", "device", "seed"], "value": [torch.__version__, str(device), seed]})
    return device


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## データ

    torchvision の MNIST を 28×28 の tensor として読み込み、`x * 2 - 1` で画素値を
    $[-1, 1]$ へ変換します。0を中心に対称な値域へ揃える前処理で、ネットワークの入力と
    出力の尺度が偏らないようにするためです。

    数字の形と書き癖を十分に学習させるため、学習用60,000件をすべて使い、batch size は256です。
    後半の品質評価では、学習に使っていない評価用10,000件で分類器の正解率を確認します。
    """)
    return


@app.cell
def _():
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms

    transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x * 2.0 - 1.0)])
    train_dataset = datasets.MNIST("./data", train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST("./data", train=False, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False, num_workers=0)
    print(f"train images: {len(train_dataset)}, test images: {len(test_dataset)}")
    return test_loader, train_dataset, train_loader


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    学習データの先頭10件を表示し、値域の変換後も数字が正しく読み取れることを確認します。
    表示時は `(x + 1) / 2` で $[0, 1]$ へ戻しています。
    """)
    return


@app.cell
def _(train_dataset):
    _fig, _axes = plt.subplots(2, 5, figsize=(8, 4))
    for _idx, _axis in enumerate(_axes.ravel()):
        _image, _label = train_dataset[_idx]
        _axis.imshow(((_image.squeeze(0) + 1.0) / 2.0), cmap="gray")
        _axis.set_title(f"label={_label}")
        _axis.axis("off")
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 拡散スケジュール

    forward process は各ステップで分散 $\beta_t$ のノイズを加えます。

    $$
    q(x_t \mid x_{t-1}) = \mathcal{N}\!\left(x_t;\ \sqrt{1-\beta_t}\,x_{t-1},\ \beta_t I\right)
    $$

    ステップを逐次適用しなくても、累積係数 $\bar{\alpha}_t$ を使えば任意の $t$ の状態を
    1回の計算で得られます。学習時はこの式でノイズ画像を作ります。

    $$
    x_t = \sqrt{\bar{\alpha}_t}\,x_0 + \sqrt{1-\bar{\alpha}_t}\,\varepsilon,
    \qquad \bar{\alpha}_t = \prod_{s=1}^{t}\left(1-\beta_s\right)
    $$

    ここで重要なのは**終端 $\bar{\alpha}_T$ が十分に 0 へ近いこと**です。生成は標準正規分布
    $x_T \sim \mathcal{N}(0, I)$ から始めるため、$\bar{\alpha}_T$ が大きいと
    「学習した終端分布」と「生成の開始分布」がずれ、逆過程がノイズのままになります。

    $\beta_t$ は $10^{-4}$ から $0.02$ までの線形スケジュールとし、ステップ数は400にします。
    次のセルで定義する `q_sample` が上の $x_t$ の式そのものです。
    """)
    return


@app.cell
def _():
    num_steps = 400
    betas = torch.linspace(1e-4, 2e-2, num_steps)
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)

    def q_sample(x0, t, noise):
        alpha_bars_local = alpha_bars.to(x0.device)
        t_local = t.to(x0.device)
        sqrt_alpha_bar = alpha_bars_local[t_local].view(-1, 1, 1, 1).sqrt()
        sqrt_one_minus = (1.0 - alpha_bars_local[t_local]).view(-1, 1, 1, 1).sqrt()
        return sqrt_alpha_bar * x0 + sqrt_one_minus * noise

    return alpha_bars, alphas, betas, num_steps, q_sample


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    スケジュールが妥当かを、$\bar{\alpha}_t$ と、元画像の残存量 $\sqrt{\bar{\alpha}_t}$、
    ノイズ量 $\sqrt{1-\bar{\alpha}_t}$ の3つで確認します。表の最終行で
    $\sqrt{\bar{\alpha}_T}$ が十分小さくなっていれば、生成をガウスノイズから始めてよい状態です。
    """)
    return


@app.cell
def _(alpha_bars, num_steps):
    _checkpoints = [0, 99, 199, 299, num_steps - 1]
    schedule_frame = pd.DataFrame(
        {
            "t": [_t + 1 for _t in _checkpoints],
            "alpha_bar": [round(alpha_bars[_t].item(), 6) for _t in _checkpoints],
            "signal_amplitude": [round(alpha_bars[_t].sqrt().item(), 4) for _t in _checkpoints],
            "noise_amplitude": [round((1.0 - alpha_bars[_t]).sqrt().item(), 4) for _t in _checkpoints],
        }
    )
    schedule_frame
    return


@app.cell
def _(alpha_bars, num_steps):
    _steps = np.arange(1, num_steps + 1)
    _fig, _ax = plt.subplots(figsize=(6, 4))
    _ax.plot(_steps, alpha_bars.numpy(), label=r"$\bar{\alpha}_t$")
    _ax.plot(_steps, alpha_bars.sqrt().numpy(), label=r"$\sqrt{\bar{\alpha}_t}$ (signal)")
    _ax.plot(_steps, (1.0 - alpha_bars).sqrt().numpy(), label=r"$\sqrt{1-\bar{\alpha}_t}$ (noise)")
    _ax.set_xlabel("diffusion step t")
    _ax.set_ylabel("amplitude")
    _ax.set_title("Forward diffusion schedule")
    _ax.legend()
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    実際に1枚の画像へ `q_sample` を適用し、$t$ が進むほど数字が見えなくなることを確かめます。
    左端が元画像 $x_0$、右へ進むほどノイズが強く、$t=400$ ではほぼ純粋なノイズになります。
    """)
    return


@app.cell
def _(num_steps, q_sample, train_dataset):
    image, label = train_dataset[0]
    noisy_versions = []
    for _step in [0, 99, 199, 299, num_steps - 1]:
        _noise = torch.randn_like(image).unsqueeze(0)
        _noised = q_sample(image.unsqueeze(0), torch.tensor([_step]), _noise)[0]
        noisy_versions.append((_step + 1, _noised.squeeze(0)))
    return image, label, noisy_versions


@app.cell
def _(image, label, noisy_versions):
    _fig, _axes = plt.subplots(1, len(noisy_versions) + 1, figsize=(12, 3))
    _axes[0].imshow(((image.squeeze(0) + 1.0) / 2.0), cmap="gray")
    _axes[0].set_title(f"x0 label={label}")
    _axes[0].axis("off")
    for _axis, (_step, _noised) in zip(_axes[1:], noisy_versions, strict=False):
        _axis.imshow(((_noised + 1.0) / 2.0), cmap="gray")
        _axis.set_title(f"t={_step}")
        _axis.axis("off")
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## デノイザ

    学習するのは「ノイズ画像 $x_t$ に含まれるノイズ $\varepsilon$ を当てるネットワーク」
    $\varepsilon_\theta(x_t, t, y)$ です。入力はノイズ画像・ステップ番号・ラベルの3つで、
    出力は入力と同じ形のノイズ推定です。

    ステップ番号は整数のスカラーなので、そのまま入力すると表現力が足りません。
    Transformer と同じ **sinusoidal embedding** で周波数の異なる正弦・余弦へ展開し、
    細かい変化と大きな変化を同時に表せる形にしてから MLP に通します。
    """)
    return


@app.cell
def _():
    def timestep_embedding(t, dim):
        half = dim // 2
        freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device) / half)
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        return torch.cat([torch.cos(args), torch.sin(args)], dim=1)

    return timestep_embedding


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    本体は畳み込みの UNet です。全結合の MLP は画素の隣接関係を構造として持たないため、
    同じ位置関係を毎回学び直す必要があります。畳み込みなら重みを共有したまま局所パターンを
    扱えるので、同じ学習量でも輪郭を捉えやすくなります。ここでは残差ブロックで特徴を抽出しつつ
    2回ダウンサンプリングし（28→14→7）、skip connection をつないで元の解像度へ戻します。

    - `ResidualBlock`：入力は特徴マップと埋め込みベクトル。畳み込み2層と GroupNorm で
      変換し、途中でステップ・ラベルの埋め込みをチャンネル方向のバイアスとして加算します。
      出力は指定チャンネル数の特徴マップです。
    - `Denoiser`：入力はノイズ画像 `x`、ステップ `t`、ラベル `labels`。
      出力は `x` と同じ形のノイズ推定です。ラベル埋め込みは11種類あり、
      末尾の `null_label`（=10）が「条件なし」を表します。これが後述の
      classifier-free guidance で使う無条件側の入力になります。
    """)
    return


@app.cell
def _(timestep_embedding):
    class ResidualBlock(nn.Module):
        def __init__(self, in_channels, out_channels, embed_dim):
            super().__init__()
            self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
            self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
            self.norm1 = nn.GroupNorm(8, out_channels)
            self.norm2 = nn.GroupNorm(8, out_channels)
            self.embed = nn.Linear(embed_dim, out_channels)
            self.skip = (
                nn.Conv2d(in_channels, out_channels, 1)
                if in_channels != out_channels
                else nn.Identity()
            )

        def forward(self, x, embedding):
            h = F.silu(self.norm1(self.conv1(x)))
            h = h + self.embed(embedding).unsqueeze(-1).unsqueeze(-1)
            h = F.silu(self.norm2(self.conv2(h)))
            return h + self.skip(x)

    class Denoiser(nn.Module):
        def __init__(self, base=64, embed_dim=128, num_labels=10):
            super().__init__()
            self.embed_dim = embed_dim
            self.null_label = num_labels
            self.time_mlp = nn.Sequential(
                nn.Linear(embed_dim, embed_dim), nn.SiLU(), nn.Linear(embed_dim, embed_dim)
            )
            self.label_embedding = nn.Embedding(num_labels + 1, embed_dim)
            self.stem = nn.Conv2d(1, base, 3, padding=1)
            self.down1 = ResidualBlock(base, base, embed_dim)
            self.down2 = ResidualBlock(base, base * 2, embed_dim)
            self.middle = ResidualBlock(base * 2, base * 2, embed_dim)
            self.up2 = ResidualBlock(base * 4, base, embed_dim)
            self.up1 = ResidualBlock(base * 2, base, embed_dim)
            self.head = nn.Sequential(
                nn.GroupNorm(8, base), nn.SiLU(), nn.Conv2d(base, 1, 3, padding=1)
            )

        def forward(self, x, t, labels):
            embedding = self.time_mlp(timestep_embedding(t, self.embed_dim))
            embedding = embedding + self.label_embedding(labels)
            h0 = self.stem(x)
            h1 = self.down1(h0, embedding)
            h2 = self.down2(F.avg_pool2d(h1, 2), embedding)
            h3 = self.middle(F.avg_pool2d(h2, 2), embedding)
            u2 = self.up2(torch.cat([F.interpolate(h3, scale_factor=2), h2], dim=1), embedding)
            u1 = self.up1(torch.cat([F.interpolate(u2, scale_factor=2), h1], dim=1), embedding)
            return self.head(u1)

    return Denoiser


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 学習

    モデルの生成から学習までを1つのセルにまとめ、学習済みモデルを `trained_model` として
    返します。モデルを作るセルと学習するセルを分けると、marimo の依存グラフ上は
    「学習によってモデルが変化したこと」が表現できず、生成セルが未学習のモデルを
    使ってしまう可能性があるためです。

    1回の更新では、バッチごとにステップ $t$ を一様乱数で選び、`q_sample` でノイズ画像を作り、
    加えたノイズを当てる二乗誤差を最小化します。

    $$
    \mathcal{L} = \mathbb{E}_{x_0, t, \varepsilon}
    \left\lVert \varepsilon - \varepsilon_\theta(x_t, t, y) \right\rVert^2
    $$

    このとき10%の確率でラベルを `null_label` に置き換えます。条件付きと条件なしの
    両方を1つのモデルに学習させる **classifier-free guidance** の準備で、
    生成時に条件の効き方を調節できるようになります。

    最適化は Adam、学習率は $2\times10^{-4}$、エポック数は15です。
    損失の目安として、ノイズを全く予測できないモデルの誤差はおよそ **1.0** です。
    学習後の値がこれを大きく下回っていることが、モデルが機能している最低条件になります。
    """)
    return


@app.cell
def _(Denoiser, device, num_steps, q_sample, train_loader):
    trained_model = Denoiser().to(device)
    optimizer = torch.optim.Adam(trained_model.parameters(), lr=2e-4)
    history_rows = []
    training_start = time.time()
    for epoch in range(1, 16):
        trained_model.train()
        running_loss = 0.0
        running_items = 0
        for _batch, _labels in train_loader:
            _batch = _batch.to(device)
            _labels = _labels.to(device)
            timestep = torch.randint(0, num_steps, (_batch.shape[0],), device=device)
            noise = torch.randn_like(_batch)
            noisy_batch = q_sample(_batch, timestep, noise)
            conditioned_labels = _labels.clone()
            conditioned_labels[torch.rand(_batch.shape[0], device=device) < 0.1] = (
                trained_model.null_label
            )
            loss = F.mse_loss(trained_model(noisy_batch, timestep, conditioned_labels), noise)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * _batch.shape[0]
            running_items += _batch.shape[0]
        history_rows.append(
            {
                "epoch": epoch,
                "train_loss": running_loss / running_items,
                "elapsed_sec": round(time.time() - training_start, 1),
            }
        )
    history_frame = pd.DataFrame(history_rows)
    return history_frame, trained_model


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    学習したモデルの規模を確認します。次の表は奇数エポックの損失の抜粋で、
    全エポックの推移はその下の対数軸のグラフから読み取れます。
    """)
    return


@app.cell
def _(trained_model):
    _parameter_count = sum(_p.numel() for _p in trained_model.parameters())
    pd.DataFrame(
        {
            "item": ["parameters", "optimizer", "learning rate", "epochs"],
            "value": [f"{_parameter_count / 1e6:.2f}M", "Adam", "2e-4", 15],
        }
    )
    return


@app.cell
def _(history_frame):
    history_frame[history_frame["epoch"] % 2 == 1].reset_index(drop=True)
    return


@app.cell
def _(history_frame):
    _fig, _ax = plt.subplots(figsize=(6, 4))
    _ax.plot(history_frame["epoch"], history_frame["train_loss"], marker="o", label="train loss")
    _ax.axhline(1.0, color="tab:red", linestyle="--", label="uninformed baseline")
    _ax.set_yscale("log")
    _ax.set_xlabel("epoch")
    _ax.set_ylabel("MSE on predicted noise")
    _ax.set_title("Diffusion training loss")
    _ax.legend()
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## サンプリング

    生成は $x_T \sim \mathcal{N}(0, I)$ から始め、$t = T$ から $t = 1$ へ逆向きに進めます。
    各ステップの平均は、推定ノイズ $\hat{\varepsilon}$ を使って次で与えられます。

    $$
    \mu_\theta(x_t, t) = \frac{1}{\sqrt{\alpha_t}}
    \left(x_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}}\,\hat{\varepsilon}\right)
    $$

    加えるノイズの分散には、$\beta_t$ ではなく事後分散 $\tilde{\beta}_t$ を使います。
    終盤のステップで過剰なノイズが乗るのを防ぎ、輪郭がはっきりします。

    $$
    \tilde{\beta}_t = \frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_t}\,\beta_t
    $$

    推定ノイズは、条件付きと条件なしの2つの出力を重み $w$ で外挿して作ります（$w=2$ を使用）。
    $w$ を大きくするほどラベルへの忠実度が上がり、多様性は下がります。

    $$
    \hat{\varepsilon} = \varepsilon_\theta(x_t, t, \varnothing)
    +w\left(\varepsilon_\theta(x_t, t, y) - \varepsilon_\theta(x_t, t, \varnothing)\right)
    $$

    条件付きと条件なしは1つのバッチへまとめて1回で推論し、ステップ数400ぶんの
    呼び出し回数を半分に抑えています。`sample_images` の入力は生成したいラベル列と
    guidance 強度、戻り値は次の3つ組です。

    - 最終的な生成画像（ラベル数 × 1 × 28 × 28）
    - `snapshots`：代表ステップの更新直後の状態。静止画の一覧表示に使います
    - `frames`：8ステップおきの中間結果。GIF のフレームになります
    """)
    return


@app.cell
def _(alpha_bars, alphas, betas, device, num_steps, trained_model):
    @torch.no_grad()
    def sample_images(labels, guidance_scale=2.0):
        trained_model.eval()
        x = torch.randn(len(labels), 1, 28, 28, device=device)
        labels = labels.to(device)
        null_labels = torch.full_like(labels, trained_model.null_label)
        snapshot_steps = {299, 199, 99, 49, 0}
        snapshots = []
        frames = []
        for step in reversed(range(num_steps)):
            timestep = torch.full((len(labels),), step, device=device, dtype=torch.long)
            paired_noise = trained_model(
                torch.cat([x, x]),
                torch.cat([timestep, timestep]),
                torch.cat([labels, null_labels]),
            )
            cond_noise, uncond_noise = paired_noise.chunk(2)
            eps = uncond_noise + guidance_scale * (cond_noise - uncond_noise)
            alpha = alphas[step].to(device)
            alpha_bar = alpha_bars[step].to(device)
            beta = betas[step].to(device)
            mean = (x - beta / torch.sqrt(1.0 - alpha_bar) * eps) / torch.sqrt(alpha)
            if step > 0:
                alpha_bar_prev = alpha_bars[step - 1].to(device)
                posterior_var = beta * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar)
                x = mean + torch.sqrt(posterior_var) * torch.randn_like(x)
            else:
                x = mean
            if step in snapshot_steps:
                snapshots.append((step, x.detach().cpu()))
            if step % 8 == 0:
                frames.append(x.detach().cpu())
        return x.detach().cpu(), snapshots, frames

    return sample_images


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 生成

    0から9のラベルを2巡分、合計20枚を生成します。同じラベルでも初期ノイズが異なるため、
    書き方の違う数字が得られます。
    """)
    return


@app.cell
def _(sample_images):
    labels = torch.tensor(list(range(10)) + list(range(10)))
    generated, snapshots, denoising_frames = sample_images(labels)
    print(f"generated batch shape: {tuple(generated.shape)}, gif frames: {len(denoising_frames)}")
    return denoising_frames, generated, labels, snapshots


@app.cell
def _(generated, labels):
    _fig, _axes = plt.subplots(2, 10, figsize=(12, 3))
    for _axis, _image, _label in zip(_axes.ravel(), generated, labels, strict=False):
        _axis.imshow(((_image.squeeze(0) + 1.0) / 2.0), cmap="gray")
        _axis.set_title(str(_label.item()))
        _axis.axis("off")
    _fig.suptitle("Generated digits conditioned on labels 0-9", y=1.04)
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    逆過程の途中経過を段階ごとに並べます。各行のラベル $t$ は、そのステップの更新を終えた
    直後の状態を表し、最下行の $t=0$ が最終的な生成画像です。上の行ほどノイズが強く、
    $t$ が100を切ったあたりから急速に輪郭が定まります。
    """)
    return


@app.cell
def _(snapshots):
    _fig, _axes = plt.subplots(len(snapshots), 10, figsize=(12, 1.3 * len(snapshots)))
    for _row, (_step, _batch) in enumerate(snapshots):
        for _col in range(10):
            _axes[_row, _col].imshow(((_batch[_col, 0] + 1.0) / 2.0), cmap="gray")
            _axes[_row, _col].set_xticks([])
            _axes[_row, _col].set_yticks([])
            if _col == 0:
                _axes[_row, _col].set_ylabel(f"t={_step}", rotation=0, labelpad=24, va="center")
    _fig.suptitle("Denoising snapshots for labels 0-9", y=0.99)
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 生成過程の GIF

    `frames` を1枚のグリッド画像へ並べ、アニメーション GIF にします。marimo の HTML 出力は
    外部ファイルを参照できないため、GIF を base64 で埋め込んだ `img` 要素として表示します。
    1フレームあたり80ミリ秒、最後のフレームだけ1.5秒表示して、完成した数字を確認しやすくしています。
    """)
    return


@app.cell
def _(denoising_frames):
    _pil_frames = []
    for _batch in denoising_frames:
        _images = ((_batch[:, 0].clamp(-1.0, 1.0) + 1.0) / 2.0 * 255.0).to(torch.uint8).numpy()
        _canvas = np.full((2 * 28 + 3 * 2, 10 * 28 + 11 * 2), 255, dtype=np.uint8)
        for _index, _image in enumerate(_images):
            _row, _column = divmod(_index, 10)
            _top = 2 + _row * 30
            _left = 2 + _column * 30
            _canvas[_top : _top + 28, _left : _left + 28] = _image
        _pil_frames.append(Image.fromarray(_canvas, mode="L"))

    _buffer = io.BytesIO()
    _pil_frames[0].save(
        _buffer,
        format="GIF",
        save_all=True,
        append_images=_pil_frames[1:],
        duration=[80] * (len(_pil_frames) - 1) + [1500],
        loop=0,
    )
    _gif_base64 = base64.b64encode(_buffer.getvalue()).decode("ascii")
    mo.Html(
        f"""
        <img
          src="data:image/gif;base64,{_gif_base64}"
          alt="ラベル 0 から 9 の MNIST 画像を生成する逆拡散過程"
          style="width: 100%; image-rendering: pixelated;"
        />
        """
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## ラベル忠実度の評価

    見た目の印象だけでは判断できないため、MNIST 分類器を別に学習し、
    **指定したラベル通りの数字が生成されたか**を定量的に測ります。ここで測れるのは
    ラベルへの忠実度だけで、多様性や学習データの丸暗記の有無は別途の検証が必要です。

    分類器の入力は28×28の画像、出力は10クラスのlogitです。畳み込み2層と全結合2層の
    小さな構成で、学習データ60,000件を2エポック学習します。
    """)
    return


@app.cell
def _(device, train_loader):
    class Classifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(1, 32, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Flatten(),
                nn.Linear(64 * 7 * 7, 128),
                nn.ReLU(),
                nn.Linear(128, 10),
            )

        def forward(self, x):
            return self.net(x)

    classifier = Classifier().to(device)
    _optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-3)
    classifier.train()
    for _ in range(2):
        for _batch, _labels in train_loader:
            _loss = F.cross_entropy(classifier(_batch.to(device)), _labels.to(device))
            _optimizer.zero_grad()
            _loss.backward()
            _optimizer.step()
    classifier.eval()
    return classifier


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    まず分類器そのものを、学習に使っていない評価用10,000件で確認します。これは実データに
    対する健全性の確認であり、生成画像のような分布外の入力での信頼性まで保証するものでは
    ありません。以降の指標は「この分類器がどう判定したか」であって、人間による正誤判定では
    ない点に注意してください。
    """)
    return


@app.cell
def _(classifier, device, test_loader):
    _correct = 0
    _total = 0
    with torch.no_grad():
        for _batch, _labels in test_loader:
            _prediction = classifier(_batch.to(device)).argmax(dim=1).cpu()
            _correct += (_prediction == _labels).sum().item()
            _total += _labels.numel()
    classifier_accuracy = _correct / _total
    print(f"classifier test accuracy: {classifier_accuracy:.4f}")
    return classifier_accuracy


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    次に、ラベル0から9をそれぞれ20枚ずつ、合計200枚を生成し、分類器の予測と
    指定ラベルが一致する割合を測ります。`label_accuracy` が1に近ければ、
    指定した数字として認識される形で生成できていることになります。
    `mean_target_probability` は**指定ラベルに対して**分類器が与えた確率の平均です。
    予測クラスの最大確率ではなく指定ラベルの確率を見ることで、
    別の数字を自信たっぷりに描いた場合を見逃さないようにしています。
    """)
    return


@app.cell
def _(classifier, classifier_accuracy, device, sample_images):
    evaluation_labels = torch.arange(10).repeat(20)
    evaluation_images, _, _ = sample_images(evaluation_labels)
    with torch.no_grad():
        _logits = classifier(evaluation_images.to(device))
    evaluation_predictions = _logits.argmax(dim=1).cpu()
    label_accuracy = (evaluation_predictions == evaluation_labels).float().mean().item()
    _probabilities = _logits.softmax(dim=1).cpu()
    mean_target_probability = _probabilities[
        torch.arange(len(evaluation_labels)), evaluation_labels
    ].mean().item()
    pd.DataFrame(
        {
            "metric": [
                "classifier test accuracy",
                "generated label accuracy",
                "generated mean target probability",
            ],
            "value": [
                round(classifier_accuracy, 4),
                round(label_accuracy, 4),
                round(mean_target_probability, 4),
            ],
        }
    )
    return evaluation_labels, evaluation_predictions


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    最後にラベルごとの内訳を確認します。特定の数字だけ生成に失敗していないか、
    偏りを見つけるための表です。各ラベル20枚ずつの1回の試行なので、
    値が20であることは「その試行で失敗が無かった」ことを意味します。
    """)
    return


@app.cell
def _(evaluation_labels, evaluation_predictions):
    per_label_frame = pd.DataFrame(
        {
            "label": list(range(10)),
            "correct_of_20": [
                int(((evaluation_labels == _digit) & (evaluation_predictions == _digit)).sum().item())
                for _digit in range(10)
            ],
        }
    )
    per_label_frame
    return


if __name__ == "__main__":
    app.run()
