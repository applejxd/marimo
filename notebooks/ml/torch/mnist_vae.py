import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")

with app.setup:
    import io

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import torch
    import torch.nn as nn
    import torch.nn.functional as F


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # MNIST で VAE

    TensorBoard と Colab 依存を外し、学習曲線・再構成画像・潜在空間散布図を notebook 内に収めました。
    乱数シードとデバイス選択を明示し、チェックポイントは欠落ファイルに依存しないよう in-memory round-trip で確認します。
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 共通部品と再現性

    冒頭の `with app.setup` ブロックで marimo、Matplotlib、NumPy、pandas、PyTorch を
    notebook 全体から使えるようにしています。続くセルでは乱数 seed を 42 に固定し、
    CUDA、MPS、CPU の順で利用可能な device を選びます。
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

    torchvision の MNIST を `[0, 1]` の 28×28 tensor として読み込みます。数字ごとの
    多様性を十分に学習するため、学習用60,000件と評価用10,000件をすべて使用し、
    batch size はそれぞれ256、512とします。次の2セルで loader を作成し、入力例を確認します。
    """)
    return


@app.cell
def _():
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms

    transform = transforms.ToTensor()
    train_dataset = datasets.MNIST("./data", train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST("./data", train=False, download=True, transform=transform)
    train_loader = DataLoader(
        train_dataset,
        batch_size=256,
        shuffle=True,
        num_workers=0,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=512,
        shuffle=False,
        num_workers=0,
    )
    return test_loader, train_dataset, train_loader


@app.cell
def _(train_dataset):
    _fig, _axes = plt.subplots(2, 5, figsize=(8, 4))
    for _idx, _axis in enumerate(_axes.ravel()):
        _image, _label = train_dataset[_idx]
        _axis.imshow(_image.squeeze(0), cmap="gray")
        _axis.set_title(f"label={_label}")
        _axis.axis("off")
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## VAE の構造

    encoder は784次元の画像から16次元の潜在分布の平均 $\mu$ と対数分散
    $\log \sigma^2$ を推定し、decoder は16次元の潜在変数を784個の画素logitへ戻します。
    潜在変数は再パラメータ化 trick

    $$
    q_\phi(z\mid x)=\mathcal{N}\!\left(\mu_\phi(x),
    \operatorname{diag}(\sigma_\phi^2(x))\right),\qquad
    z=\mu_\phi(x)+\sigma_\phi(x)\odot\epsilon,\quad
    \epsilon\sim\mathcal{N}(0,I)
    $$

    で sampling します。これにより sampling を含む処理でも encoder へ勾配を伝播できます。
    """)
    return


@app.cell
def _():
    class VAE(nn.Module):
        def __init__(self, x_dim=784, h_dim1=256, h_dim2=128, z_dim=16):
            super().__init__()
            self.fc1 = nn.Linear(x_dim, h_dim1)
            self.fc2 = nn.Linear(h_dim1, h_dim2)
            self.fc_mu = nn.Linear(h_dim2, z_dim)
            self.fc_logvar = nn.Linear(h_dim2, z_dim)
            self.fc3 = nn.Linear(z_dim, h_dim2)
            self.fc4 = nn.Linear(h_dim2, h_dim1)
            self.fc5 = nn.Linear(h_dim1, x_dim)

        def encode(self, x):
            h = F.relu(self.fc1(x))
            h = F.relu(self.fc2(h))
            return self.fc_mu(h), self.fc_logvar(h)

        def reparameterize(self, mu, logvar):
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std

        def decode(self, z):
            h = F.relu(self.fc3(z))
            h = F.relu(self.fc4(h))
            return self.fc5(h)

        def forward(self, x):
            x = x.view(-1, 784)
            mu, logvar = self.encode(x)
            z = self.reparameterize(mu, logvar)
            logits = self.decode(z)
            return logits, mu, logvar

    return VAE


@app.cell
def _(VAE, device):
    model = VAE().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    return model, optimizer


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 損失関数と学習

    最小化する negative ELBO は、画素の再構成誤差と潜在分布を標準正規分布へ近づける
    KL divergence の和です。

    $$
    \begin{aligned}
    \mathcal{L}
      &= \operatorname{BCE}(x,\hat{x}) \\
      &\quad + D_{\mathrm{KL}}\!\left(q_\phi(z\mid x)\,\Vert\,\mathcal{N}(0,I)\right)
    \end{aligned}
    $$

    $$
    D_{\mathrm{KL}}
      = -\frac{1}{2}\sum_j
        \left(1+\log\sigma_j^2-\mu_j^2-\sigma_j^2\right)
    $$

    BCE と KL は batch 内で加算し、履歴では dataset 件数で割った1画像あたりの値を表示します。
    Adam（learning rate `1e-3`）で20 epoch学習し、各 epoch後にtest ELBOと、潜在平均から
    復元した画像のBCE・画素平均MSEを計算します。

    RTX 3070・seed 42でtest data 10,000件を共通にして旧条件を一度ローカルで再実行した結果、
    元の2次元潜在空間・20,000件・batch size 128・8 epochの再構成MSEは`0.04490`、
    新しい16次元潜在空間・60,000件・batch size 256・20 epochでは`0.01388`で、
    約69.1%低下しました。後半は改善幅が小さくなるため、品質と実行時間のバランスから
    20 epochで止めています。単独ページのbuild時間は同じRTX 3070で約4分で、
    CPUでは実行時間が大幅に長くなります。
    この設定選択にもtest dataを使ったため最終値はわずかに楽観的であり、deviceやbackendに
    よって値は多少変動します。
    """)
    return


@app.cell
def _():
    def loss_function(logits, x, mu, logvar):
        reconstruction = F.binary_cross_entropy_with_logits(logits, x.view(-1, 784), reduction="sum")
        kl_divergence = -0.5 * torch.sum(1 + logvar - logvar.exp() - mu.pow(2))
        return reconstruction + kl_divergence

    def evaluate(model, device, loader):
        model.eval()
        total_loss = 0.0
        deterministic_bce = 0.0
        deterministic_mse = 0.0
        with torch.no_grad():
            for batch, _ in loader:
                batch = batch.to(device)
                logits, mu, logvar = model(batch)
                total_loss += loss_function(logits, batch, mu, logvar).item()
                deterministic_logits = model.decode(mu)
                deterministic_bce += F.binary_cross_entropy_with_logits(
                    deterministic_logits,
                    batch.view(-1, 784),
                    reduction="sum",
                ).item()
                deterministic_mse += F.mse_loss(
                    torch.sigmoid(deterministic_logits),
                    batch.view(-1, 784),
                    reduction="sum",
                ).item()
        sample_count = len(loader.dataset)
        return {
            "test_loss": total_loss / sample_count,
            "reconstruction_bce": deterministic_bce / sample_count,
            "reconstruction_mse": deterministic_mse / (sample_count * 784),
        }

    return evaluate, loss_function


@app.cell
def _(device, evaluate, loss_function, model, optimizer, test_loader, train_loader):
    history_rows = []
    for epoch in range(1, 21):
        model.train()
        total_train_loss = 0.0
        for _batch, _ in train_loader:
            _batch = _batch.to(device)
            optimizer.zero_grad()
            logits, _mu, _logvar = model(_batch)
            loss = loss_function(logits, _batch, _mu, _logvar)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
        history_rows.append(
            {
                "epoch": epoch,
                "train_loss": total_train_loss / len(train_loader.dataset),
                **evaluate(model, device, test_loader),
            }
        )
    history_frame = pd.DataFrame(history_rows)
    return history_frame


@app.cell
def _(history_frame):
    history_frame.loc[
        (history_frame["epoch"] == 1) | (history_frame["epoch"] % 5 == 0)
    ]
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    最終 epoch の `test_loss` は sampling を含む negative ELBO、
    `reconstruction_bce` と `reconstruction_mse` は潜在平均 $\mu$ を decode した
    決定論的な再構成品質です。MSE は全画像・全784画素の平均なので、小さいほど入力画像を
    忠実に復元できています。
    """)
    return


@app.cell
def _(history_frame):
    final_quality_metrics = history_frame.tail(1)[
        [
            "epoch",
            "test_loss",
            "reconstruction_bce",
            "reconstruction_mse",
        ]
    ]
    final_quality_metrics
    return (final_quality_metrics,)


@app.cell
def _(history_frame):
    _fig, (_loss_ax, _mse_ax) = plt.subplots(1, 2, figsize=(11, 4))
    _loss_ax.plot(history_frame["epoch"], history_frame["train_loss"], label="train")
    _loss_ax.plot(history_frame["epoch"], history_frame["test_loss"], label="test")
    _loss_ax.set_xlabel("epoch")
    _loss_ax.set_ylabel("negative ELBO")
    _loss_ax.set_title("VAE loss per sample")
    _loss_ax.legend()
    _mse_ax.plot(
        history_frame["epoch"],
        history_frame["reconstruction_mse"],
        color="tab:green",
    )
    _mse_ax.set_xlabel("epoch")
    _mse_ax.set_ylabel("MSE per pixel")
    _mse_ax.set_title("Deterministic reconstruction")
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 再構成結果

    評価用 loader の先頭8枚を encoder に通し、sampling ノイズを加えず潜在平均 $\mu$ を
    decoder へ渡します。入力を上段、決定論的な再構成を下段に並べて、数字の形がどの程度
    保たれたかを比較します。
    """)
    return


@app.cell
def _(device, model, test_loader):
    preview_batch, preview_labels = next(iter(test_loader))
    preview_batch = preview_batch[:8].to(device)
    preview_labels = preview_labels[:8]
    with torch.no_grad():
        preview_mu, _ = model.encode(preview_batch.view(-1, 784))
        reconstruction_logits = model.decode(preview_mu)
        reconstructions = torch.sigmoid(reconstruction_logits).view(-1, 1, 28, 28).cpu()
    return preview_batch, preview_labels, reconstructions


@app.cell
def _(preview_batch, preview_labels, reconstructions):
    _fig, _axes = plt.subplots(2, 8, figsize=(12, 3))
    _originals = preview_batch.cpu()
    for _idx in range(8):
        _axes[0, _idx].imshow(_originals[_idx, 0], cmap="gray")
        _axes[0, _idx].set_title(f"x={preview_labels[_idx].item()}")
        _axes[0, _idx].axis("off")
        _axes[1, _idx].imshow(reconstructions[_idx, 0], cmap="gray")
        _axes[1, _idx].axis("off")
    _fig.text(0.01, 0.72, "input", rotation=90, va="center")
    _fig.text(0.01, 0.28, "reconstruction", rotation=90, va="center")
    _fig.tight_layout(rect=(0.03, 0, 1, 1))
    _fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 16次元潜在空間の2次元可視化

    評価用画像5,000枚について、sampling 後の $z$ ではなく encoder が出力した平均 $\mu$ を
    取り出します。再構成に使う潜在表現は16次元のまま保持し、散布図に限って中心化した
    潜在平均をSVDで情報量の多い2軸へ射影します。数字labelで色分けし、同じ数字が近くへ
    配置されるか、異なる数字の領域がどのようにつながるかを確認します。
    """)
    return


@app.cell
def _(device, model, test_loader):
    latent_images = []
    latent_points = []
    latent_labels = []
    with torch.no_grad():
        for _batch, _labels in test_loader:
            _batch = _batch.to(device)
            _mu, _ = model.encode(_batch.view(-1, 784))
            latent_images.append(_batch.cpu())
            latent_points.append(_mu.cpu())
            latent_labels.append(_labels)
            if sum(item.shape[0] for item in latent_points) >= 5_000:
                break
    latent_images = torch.cat(latent_images, dim=0)[:5_000]
    latent_points = torch.cat(latent_points, dim=0)[:5_000].numpy()
    latent_labels = torch.cat(latent_labels, dim=0)[:5_000].numpy()
    centered_latent_points = latent_points - latent_points.mean(axis=0)
    _, _, latent_components = np.linalg.svd(
        centered_latent_points,
        full_matrices=False,
    )
    latent_projection = centered_latent_points @ latent_components[:2].T
    return latent_images, latent_labels, latent_points, latent_projection


@app.cell
def _(latent_labels, latent_projection):
    _fig, _ax = plt.subplots(figsize=(6, 5))
    _scatter = _ax.scatter(
        latent_projection[:, 0],
        latent_projection[:, 1],
        c=latent_labels,
        cmap="tab10",
        s=10,
    )
    _ax.set_title("SVD projection of latent mean vectors")
    _ax.set_xlabel("component 1")
    _ax.set_ylabel("component 2")
    _fig.colorbar(_scatter, ax=_ax)
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## checkpoint の round-trip

    学習済み `state_dict` をメモリ上の buffer へ保存し、新しい `VAE` instance へ読み戻します。
    全 parameter の最大絶対差と、同じ潜在平均から得た再構成 logit の最大絶対差を測り、
    保存前後のモデルが同じ計算結果を返すことを確認します。
    """)
    return


@app.cell
def _(VAE, device, model, preview_batch):
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    buffer.seek(0)
    reloaded_model = VAE().to(device)
    reloaded_model.load_state_dict(torch.load(buffer, map_location=device))
    reloaded_model.eval()
    model.eval()
    max_param_diff = max(
        (model.state_dict()[name] - reloaded_model.state_dict()[name]).abs().max().item()
        for name in model.state_dict()
    )
    with torch.no_grad():
        original_mu, _ = model.encode(preview_batch.view(-1, 784))
        reloaded_mu, _ = reloaded_model.encode(preview_batch.view(-1, 784))
        original_logits = model.decode(original_mu)
        reloaded_logits = reloaded_model.decode(reloaded_mu)
    roundtrip_metrics = pd.DataFrame(
        {
            "metric": ["max_param_diff", "deterministic_recon_max_abs_diff"],
            "value": [
                max_param_diff,
                (original_logits - reloaded_logits).abs().max().item(),
            ],
        }
    )
    print(roundtrip_metrics.to_string(index=False))
    return reloaded_model, roundtrip_metrics


@app.cell
def _(roundtrip_metrics):
    roundtrip_metrics
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 潜在空間を数字として読み解く

    一様な座標 grid を decode するだけでは、各画像がどの数字に対応するのか分かりにくいため、
    test data を encode した潜在平均から数字ごとの重心を求めます。上段は各クラスの
    重心に最も近い入力画像、中段はその潜在表現をVAEで再構成した画像です。下段では
    5,000枚から求めた数字1の重心から数字7の重心までを直線補間し、潜在空間上で形が
    連続的に変化する様子を示します。

    $$
    z(t)=(1-t)z_1+t z_7,\qquad 0\leq t\leq 1
    $$
    """)
    return


@app.cell
def _(latent_images, latent_labels, latent_points, reloaded_model):
    class_centroids = np.stack(
        [latent_points[latent_labels == digit].mean(axis=0) for digit in range(10)]
    )
    representative_indices = np.array(
        [
            np.flatnonzero(latent_labels == digit)[
                np.linalg.norm(
                    latent_points[latent_labels == digit] - class_centroids[digit],
                    axis=1,
                ).argmin()
            ]
            for digit in range(10)
        ]
    )
    representative_inputs = latent_images[representative_indices]
    representative_latents = latent_points[representative_indices]
    interpolation_weights = np.linspace(0.0, 1.0, 10)
    interpolation_latents = np.stack(
        [
            (1.0 - weight) * class_centroids[1] + weight * class_centroids[7]
            for weight in interpolation_weights
        ]
    )
    visualization_latents = np.concatenate(
        [representative_latents, interpolation_latents],
        axis=0,
    )
    model_device = next(reloaded_model.parameters()).device
    with torch.no_grad():
        visualization_images = torch.sigmoid(
            reloaded_model.decode(
                torch.tensor(
                    visualization_latents,
                    dtype=torch.float32,
                    device=model_device,
                )
            )
        ).cpu()
    class_prototypes = visualization_images[:10]
    interpolation_images = visualization_images[10:]
    return (
        class_prototypes,
        interpolation_images,
        interpolation_weights,
        representative_inputs,
    )


@app.cell
def _(
    class_prototypes,
    interpolation_images,
    interpolation_weights,
    representative_inputs,
):
    _fig, _axes = plt.subplots(3, 10, figsize=(14, 5.5))
    for _digit, (_axis, _image) in enumerate(
        zip(_axes[0], representative_inputs, strict=True)
    ):
        _axis.imshow(_image.view(28, 28), cmap="gray", vmin=0.0, vmax=1.0)
        _axis.set_title(str(_digit))
        _axis.axis("off")

    for _axis, _image in zip(_axes[1], class_prototypes, strict=True):
        _axis.imshow(_image.view(28, 28), cmap="gray", vmin=0.0, vmax=1.0)
        _axis.axis("off")

    for _axis, _image, _weight in zip(
        _axes[2],
        interpolation_images,
        interpolation_weights,
        strict=True,
    ):
        _axis.imshow(_image.view(28, 28), cmap="gray", vmin=0.0, vmax=1.0)
        _axis.set_title(f"{_weight:.2f}")
        _axis.axis("off")

    _fig.text(0.01, 0.79, "representative input", rotation=90, va="center")
    _fig.text(0.01, 0.49, "reconstruction", rotation=90, va="center")
    _fig.text(0.01, 0.18, "1 to 7", rotation=90, va="center")
    _fig.suptitle("Representative digits, reconstructions, and latent interpolation")
    _fig.tight_layout(rect=(0.03, 0, 1, 0.95))
    _fig
    return


if __name__ == "__main__":
    app.run()
