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
    # AutoEncoder から VAE へ

    画像を圧縮して復元するだけなら通常の AutoEncoder（AE）で十分です。一方、未知の潜在点から
    新しい画像を生成するには、潜在空間のどこを decode すればよいかも学習する必要があります。
    この notebook では同じ条件で AE と Variational AutoEncoder（VAE）を学習し、
    **再構成性能と生成可能性は別の目標**であることを確かめます。
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
    ## まず AutoEncoder を基準にする

    AE は784次元の画像を16次元のベクトル $z$ へ圧縮し、decoder で784個の画素logitへ
    戻します。再構成誤差だけを最小化するため、入力を圧縮・復元する用途には適しています。
    ただし、潜在ベクトルの分布には制約がなく、学習データのない領域をdecodeした結果は保証されません。

    $$
    z=f_\phi(x),\qquad \hat{x}=g_\theta(z)
    $$

    公平に比べるため、AE と VAE は同じ16次元、同じ256→128の隠れ層、同じdecoder構造を
    使用します。VAEだけが潜在分布を表すための平均・対数分散の2つの出力headを持ちます。
    """)
    return


@app.cell
def _():
    class AutoEncoder(nn.Module):
        def __init__(self, x_dim=784, h_dim1=256, h_dim2=128, z_dim=16):
            super().__init__()
            self.fc1 = nn.Linear(x_dim, h_dim1)
            self.fc2 = nn.Linear(h_dim1, h_dim2)
            self.fc_z = nn.Linear(h_dim2, z_dim)
            self.fc3 = nn.Linear(z_dim, h_dim2)
            self.fc4 = nn.Linear(h_dim2, h_dim1)
            self.fc5 = nn.Linear(h_dim1, x_dim)

        def encode(self, x):
            h = F.relu(self.fc1(x))
            h = F.relu(self.fc2(h))
            return self.fc_z(h)

        def decode(self, z):
            h = F.relu(self.fc3(z))
            h = F.relu(self.fc4(h))
            return self.fc5(h)

        def forward(self, x):
            z = self.encode(x.view(-1, 784))
            return self.decode(z), z

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

    return AutoEncoder, VAE


@app.cell
def _(AutoEncoder, VAE, device):
    ae_model = AutoEncoder().to(device)
    vae_model = VAE().to(device)
    ae_optimizer = torch.optim.Adam(ae_model.parameters(), lr=1e-3)
    vae_optimizer = torch.optim.Adam(vae_model.parameters(), lr=1e-3)
    model_sizes = pd.DataFrame(
        {
            "model": ["AE", "VAE"],
            "parameters": [
                sum(parameter.numel() for parameter in ae_model.parameters()),
                sum(parameter.numel() for parameter in vae_model.parameters()),
            ],
        }
    )
    model_sizes
    return ae_model, ae_optimizer, vae_model, vae_optimizer


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## VAE：潜在空間にも学習目標を与える

    VAE の encoder は、点 $z$ ではなく平均 $\mu$ と対数分散 $\log \sigma^2$ を推定します。
    標準正規分布から得たノイズを使う再パラメータ化により、samplingを含む処理にも
    backpropagationできます。

    $$
    \begin{aligned}
    q_\phi(z\mid x)
      &=\mathcal{N}\!\left(\mu_\phi(x),
        \operatorname{diag}(\sigma_\phi^2(x))\right) \\
    z
      &=\mu_\phi(x)+\sigma_\phi(x)\odot\epsilon,
        \qquad \epsilon\sim\mathcal{N}(0,I)
    \end{aligned}
    $$

    最小化する negative ELBO は、画素の再構成誤差と潜在分布を標準正規分布へ近づける
    KL divergence の和です。AEには前者しかなく、VAEは後者と再構成品質を両立させます。

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

    両モデルを同じmini-batch、Adam（learning rate `1e-3`）、20 epochで学習します。
    BCEとKLはbatch内で加算し、履歴ではdataset件数で割った1画像あたりの値を表示します。
    比較には同じtest dataを使いますが、これはモデル選択を目的とした独立評価ではなく、
    2つの学習目標の違いを確認する教材上の比較です。
    """)
    return


@app.cell
def _():
    def vae_loss_components(logits, x, mu, logvar):
        reconstruction = F.binary_cross_entropy_with_logits(logits, x.view(-1, 784), reduction="sum")
        kl_divergence = -0.5 * torch.sum(1 + logvar - logvar.exp() - mu.pow(2))
        return reconstruction, kl_divergence

    def evaluate_models(ae_model, vae_model, device, loader):
        ae_model.eval()
        vae_model.eval()
        totals = {
            "ae_reconstruction_bce": 0.0,
            "ae_reconstruction_mse": 0.0,
            "vae_test_negative_elbo": 0.0,
            "vae_reconstruction_bce": 0.0,
            "vae_reconstruction_mse": 0.0,
            "vae_kl": 0.0,
        }
        with torch.no_grad():
            for batch, _ in loader:
                batch = batch.to(device)
                ae_logits, _ = ae_model(batch)
                vae_logits, mu, logvar = vae_model(batch)
                vae_bce, vae_kl = vae_loss_components(
                    vae_logits,
                    batch,
                    mu,
                    logvar,
                )
                vae_deterministic_logits = vae_model.decode(mu)
                totals["ae_reconstruction_bce"] += F.binary_cross_entropy_with_logits(
                    ae_logits,
                    batch.view(-1, 784),
                    reduction="sum",
                ).item()
                totals["ae_reconstruction_mse"] += F.mse_loss(
                    torch.sigmoid(ae_logits),
                    batch.view(-1, 784),
                    reduction="sum",
                ).item()
                totals["vae_test_negative_elbo"] += (vae_bce + vae_kl).item()
                totals["vae_kl"] += vae_kl.item()
                totals["vae_reconstruction_bce"] += F.binary_cross_entropy_with_logits(
                    vae_deterministic_logits,
                    batch.view(-1, 784),
                    reduction="sum",
                ).item()
                totals["vae_reconstruction_mse"] += F.mse_loss(
                    torch.sigmoid(vae_deterministic_logits),
                    batch.view(-1, 784),
                    reduction="sum",
                ).item()
        sample_count = len(loader.dataset)
        return {
            key: value / (sample_count * 784)
            if key.endswith("_mse")
            else value / sample_count
            for key, value in totals.items()
        }

    return evaluate_models, vae_loss_components


@app.cell
def _(
    ae_model,
    ae_optimizer,
    device,
    evaluate_models,
    test_loader,
    train_loader,
    vae_loss_components,
    vae_model,
    vae_optimizer,
):
    history_rows = []
    for epoch in range(1, 21):
        ae_model.train()
        vae_model.train()
        ae_train_bce = 0.0
        vae_train_negative_elbo = 0.0
        for _batch, _ in train_loader:
            _batch = _batch.to(device)
            ae_optimizer.zero_grad()
            _ae_logits, _ = ae_model(_batch)
            _ae_loss = F.binary_cross_entropy_with_logits(
                _ae_logits,
                _batch.view(-1, 784),
                reduction="sum",
            )
            _ae_loss.backward()
            ae_optimizer.step()
            ae_train_bce += _ae_loss.item()

            vae_optimizer.zero_grad()
            _vae_logits, _mu, _logvar = vae_model(_batch)
            _vae_bce, _vae_kl = vae_loss_components(
                _vae_logits,
                _batch,
                _mu,
                _logvar,
            )
            _vae_loss = _vae_bce + _vae_kl
            _vae_loss.backward()
            vae_optimizer.step()
            vae_train_negative_elbo += _vae_loss.item()
        history_rows.append(
            {
                "epoch": epoch,
                "ae_train_bce": ae_train_bce / len(train_loader.dataset),
                "vae_train_negative_elbo": (
                    vae_train_negative_elbo / len(train_loader.dataset)
                ),
                **evaluate_models(
                    ae_model,
                    vae_model,
                    device,
                    test_loader,
                ),
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
    ## 再構成性能の比較

    AEはencodeした点、VAEはsamplingノイズを加えない平均 $\mu$ をdecodeして比較します。
    MSEは全画像・全784画素の平均で、小さいほど入力を忠実に復元しています。
    VAEのKL項は生成可能な潜在空間を作るための制約なので、再構成指標だけならAEが有利でも
    VAEの失敗を意味しません。履歴の`vae_test_negative_elbo`はsamplingした $z$ で計算し、
    比較表の`reconstruction_bce`は決定論的な`decode(mu)`で計算するため、後者とKLの和には
    一致しません。
    """)
    return


@app.cell
def _(history_frame):
    _last_epoch = history_frame.iloc[-1]
    final_quality_metrics = pd.DataFrame(
        {
            "model": ["AE", "VAE"],
            "reconstruction_bce": [
                _last_epoch["ae_reconstruction_bce"],
                _last_epoch["vae_reconstruction_bce"],
            ],
            "reconstruction_mse": [
                _last_epoch["ae_reconstruction_mse"],
                _last_epoch["vae_reconstruction_mse"],
            ],
            "kl_per_sample": [np.nan, _last_epoch["vae_kl"]],
        }
    )
    final_quality_metrics
    return (final_quality_metrics,)


@app.cell
def _(history_frame):
    _fig, (_loss_ax, _mse_ax) = plt.subplots(1, 2, figsize=(11, 4))
    _loss_ax.plot(
        history_frame["epoch"],
        history_frame["ae_reconstruction_bce"],
        label="AE reconstruction BCE",
    )
    _loss_ax.plot(
        history_frame["epoch"],
        history_frame["vae_reconstruction_bce"],
        label="VAE reconstruction BCE",
    )
    _loss_ax.set_xlabel("epoch")
    _loss_ax.set_ylabel("BCE per sample")
    _loss_ax.set_title("Deterministic reconstruction")
    _loss_ax.legend()
    _mse_ax.plot(
        history_frame["epoch"],
        history_frame["ae_reconstruction_mse"],
        label="AE",
    )
    _mse_ax.plot(
        history_frame["epoch"],
        history_frame["vae_reconstruction_mse"],
        label="VAE",
    )
    _mse_ax.set_xlabel("epoch")
    _mse_ax.set_ylabel("MSE per pixel")
    _mse_ax.set_title("Reconstruction error")
    _mse_ax.legend()
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 再構成結果

    評価用loaderの先頭8枚について、入力、AE、VAEの決定論的な再構成を並べます。
    この図は既知画像を圧縮・復元する能力の比較であり、新しい画像を生成する能力はまだ測っていません。
    """)
    return


@app.cell
def _(ae_model, device, test_loader, vae_model):
    preview_batch, preview_labels = next(iter(test_loader))
    preview_batch = preview_batch[:8].to(device)
    preview_labels = preview_labels[:8]
    with torch.no_grad():
        preview_ae_latents = ae_model.encode(preview_batch.view(-1, 784))
        preview_vae_mu, _ = vae_model.encode(preview_batch.view(-1, 784))
        ae_reconstructions = torch.sigmoid(
            ae_model.decode(preview_ae_latents)
        ).view(-1, 1, 28, 28).cpu()
        vae_reconstructions = torch.sigmoid(
            vae_model.decode(preview_vae_mu)
        ).view(-1, 1, 28, 28).cpu()
    return (
        ae_reconstructions,
        preview_batch,
        preview_labels,
        vae_reconstructions,
    )


@app.cell
def _(ae_reconstructions, preview_batch, preview_labels, vae_reconstructions):
    _fig, _axes = plt.subplots(3, 8, figsize=(12, 4.5))
    _originals = preview_batch.cpu()
    for _idx in range(8):
        _axes[0, _idx].imshow(_originals[_idx, 0], cmap="gray")
        _axes[0, _idx].set_title(f"x={preview_labels[_idx].item()}")
        _axes[0, _idx].axis("off")
        _axes[1, _idx].imshow(ae_reconstructions[_idx, 0], cmap="gray")
        _axes[1, _idx].axis("off")
        _axes[2, _idx].imshow(vae_reconstructions[_idx, 0], cmap="gray")
        _axes[2, _idx].axis("off")
    _fig.text(0.01, 0.82, "input", rotation=90, va="center")
    _fig.text(0.01, 0.51, "AE", rotation=90, va="center")
    _fig.text(0.01, 0.20, "VAE", rotation=90, va="center")
    _fig.tight_layout(rect=(0.03, 0, 1, 1))
    _fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 潜在分布：VAEを使う理由

    評価用画像5,000枚をencodeします。AEの潜在点には分布の制約がありません。VAEでは
    各 $q_\phi(z\mid x)$ を目標prior $\mathcal{N}(0,I)$ へ近づけるため、集約した分布も
    0中心・標準偏差1に近いか確認できます。VAEの集約分散には、平均 $\mu$ のばらつきだけでなく
    各画像の $\sigma^2$ も含めます。
    """)
    return


@app.cell
def _(ae_model, device, test_loader, vae_model):
    ae_latent_batches = []
    latent_images = []
    latent_labels = []
    vae_logvar_batches = []
    vae_mu_batches = []
    ae_model.eval()
    vae_model.eval()
    with torch.no_grad():
        for _batch, _labels in test_loader:
            _batch = _batch.to(device)
            _flat_batch = _batch.view(-1, 784)
            _ae_latents = ae_model.encode(_flat_batch)
            _vae_mu, _vae_logvar = vae_model.encode(_flat_batch)
            ae_latent_batches.append(_ae_latents.cpu())
            latent_images.append(_batch.cpu())
            latent_labels.append(_labels)
            vae_logvar_batches.append(_vae_logvar.cpu())
            vae_mu_batches.append(_vae_mu.cpu())
            if sum(item.shape[0] for item in vae_mu_batches) >= 5_000:
                break
    ae_latent_points = torch.cat(ae_latent_batches, dim=0)[:5_000].numpy()
    latent_images = torch.cat(latent_images, dim=0)[:5_000]
    latent_labels = torch.cat(latent_labels, dim=0)[:5_000].numpy()
    vae_logvar_points = torch.cat(vae_logvar_batches, dim=0)[:5_000].numpy()
    vae_latent_points = torch.cat(vae_mu_batches, dim=0)[:5_000].numpy()

    ae_latent_mean = ae_latent_points.mean(axis=0)
    ae_latent_std = ae_latent_points.std(axis=0)
    vae_aggregate_mean = vae_latent_points.mean(axis=0)
    vae_aggregate_variance = (
        np.exp(vae_logvar_points) + np.square(vae_latent_points)
    ).mean(axis=0) - np.square(vae_aggregate_mean)
    vae_aggregate_std = np.sqrt(vae_aggregate_variance)
    latent_moment_summary = pd.DataFrame(
        {
            "representation": ["AE codes", "VAE aggregate q(z)"],
            "mean_abs_average": [
                np.abs(ae_latent_mean).mean(),
                np.abs(vae_aggregate_mean).mean(),
            ],
            "std_average": [
                ae_latent_std.mean(),
                vae_aggregate_std.mean(),
            ],
            "std_min": [
                ae_latent_std.min(),
                vae_aggregate_std.min(),
            ],
            "std_max": [
                ae_latent_std.max(),
                vae_aggregate_std.max(),
            ],
        }
    )
    return (
        ae_latent_points,
        latent_images,
        latent_labels,
        latent_moment_summary,
        vae_latent_points,
    )


@app.cell
def _(latent_moment_summary):
    latent_moment_summary
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 同じ標準正規乱数から生成する

    両decoderへ同じ $z\sim\mathcal{N}(0,I)$ を入力します。AEはこの分布から生成するよう
    学習していないため、結果が不自然でも再構成器としての失敗ではありません。VAEではKL項により
    学習時の潜在分布とこのpriorを近づけるため、ラベルや入力画像なしで数字らしい画像を生成できます。
    これが再構成だけでなく生成を目的とするときにVAEを選ぶ中心的な理由です。
    """)
    return


@app.cell
def _(ae_model, device, vae_model):
    prior_generator = torch.Generator().manual_seed(43)
    prior_latents = torch.randn((10, 16), generator=prior_generator).to(device)
    with torch.no_grad():
        ae_prior_samples = torch.sigmoid(
            ae_model.decode(prior_latents)
        ).view(-1, 1, 28, 28).cpu()
        vae_prior_samples = torch.sigmoid(
            vae_model.decode(prior_latents)
        ).view(-1, 1, 28, 28).cpu()
    return ae_prior_samples, vae_prior_samples


@app.cell
def _(ae_prior_samples, vae_prior_samples):
    _fig, _axes = plt.subplots(2, 10, figsize=(14, 3.2))
    for _index in range(10):
        _axes[0, _index].imshow(
            ae_prior_samples[_index, 0],
            cmap="gray",
            vmin=0.0,
            vmax=1.0,
        )
        _axes[1, _index].imshow(
            vae_prior_samples[_index, 0],
            cmap="gray",
            vmin=0.0,
            vmax=1.0,
        )
        _axes[0, _index].axis("off")
        _axes[1, _index].axis("off")
    _fig.text(0.01, 0.72, "AE", rotation=90, va="center")
    _fig.text(0.01, 0.27, "VAE", rotation=90, va="center")
    _fig.suptitle("Decode the same samples from N(0, I)")
    _fig.tight_layout(rect=(0.03, 0, 1, 0.92))
    _fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## VAE潜在平均のSVD・t-SNE可視化

    VAEの潜在平均 $\mu$ を2次元に射影し、数字の局所的なまとまりを補助的に確認します。
    SVDは分散が最大の2軸を使う線形手法、t-SNEは近傍関係を強調する非線形手法です。
    どちらも16次元のprior整合性を証明するものではなく、t-SNEのクラスタ間距離や配置にも
    直接的な意味はありません。後段の補間は2次元座標ではなく元の16次元空間で行います。
    """)
    return


@app.cell
def _(vae_latent_points):
    from sklearn.manifold import TSNE

    centered_latent_points = (
        vae_latent_points - vae_latent_points.mean(axis=0)
    )
    _, latent_singular_values, latent_components = np.linalg.svd(
        centered_latent_points,
        full_matrices=False,
    )
    latent_projection = centered_latent_points @ latent_components[:2].T
    svd_explained_variance = (
        np.square(latent_singular_values[:2]).sum()
        / np.square(latent_singular_values).sum()
    )
    tsne_projection = TSNE(
        n_components=2,
        perplexity=30,
        learning_rate="auto",
        init="pca",
        random_state=42,
    ).fit_transform(centered_latent_points)
    return (
        latent_projection,
        svd_explained_variance,
        tsne_projection,
    )


@app.cell
def _(
    latent_labels,
    latent_projection,
    svd_explained_variance,
    tsne_projection,
):
    from matplotlib.colors import BoundaryNorm

    _fig, (_svd_ax, _tsne_ax) = plt.subplots(
        1,
        2,
        figsize=(12, 5),
        layout="constrained",
    )
    _color_norm = BoundaryNorm(np.arange(-0.5, 10.5), ncolors=10)
    _scatter = _svd_ax.scatter(
        latent_projection[:, 0],
        latent_projection[:, 1],
        c=latent_labels,
        cmap="tab10",
        norm=_color_norm,
        s=10,
    )
    _svd_ax.set_title(f"SVD ({svd_explained_variance:.1%} variance)")
    _svd_ax.set_xlabel("component 1")
    _svd_ax.set_ylabel("component 2")
    _tsne_ax.scatter(
        tsne_projection[:, 0],
        tsne_projection[:, 1],
        c=latent_labels,
        cmap="tab10",
        norm=_color_norm,
        s=10,
    )
    _tsne_ax.set_title("t-SNE")
    _tsne_ax.set_xlabel("dimension 1")
    _tsne_ax.set_ylabel("dimension 2")
    _fig.colorbar(
        _scatter,
        ax=[_svd_ax, _tsne_ax],
        ticks=range(10),
        shrink=0.9,
        pad=0.02,
    )
    _fig.suptitle("VAE latent mean vectors")
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
def _(VAE, device, preview_batch, vae_model):
    buffer = io.BytesIO()
    torch.save(vae_model.state_dict(), buffer)
    buffer.seek(0)
    reloaded_vae_model = VAE().to(device)
    reloaded_vae_model.load_state_dict(torch.load(buffer, map_location=device))
    reloaded_vae_model.eval()
    vae_model.eval()
    max_param_diff = max(
        (
            vae_model.state_dict()[name]
            - reloaded_vae_model.state_dict()[name]
        ).abs().max().item()
        for name in vae_model.state_dict()
    )
    with torch.no_grad():
        original_mu, _ = vae_model.encode(preview_batch.view(-1, 784))
        reloaded_mu, _ = reloaded_vae_model.encode(preview_batch.view(-1, 784))
        original_logits = vae_model.decode(original_mu)
        reloaded_logits = reloaded_vae_model.decode(reloaded_mu)
    roundtrip_metrics = pd.DataFrame(
        {
            "metric": ["max_param_diff", "deterministic_recon_max_abs_diff"],
            "value": [
                max_param_diff,
                (original_logits - reloaded_logits).abs().max().item(),
            ],
        }
    )
    return reloaded_vae_model, roundtrip_metrics


@app.cell
def _(roundtrip_metrics):
    roundtrip_metrics
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## クラス重心と16次元空間での補間

    AEの潜在点とVAEの潜在平均について、数字ごとの重心をそれぞれ求めます。上段はVAE重心に
    最も近い入力例、続く2段は各モデルが自身のクラス重心をdecodeした結果です。最後の2段では、
    それぞれの元の16次元空間で数字1から7の重心までを直線補間します。

    $$
    z(t)=(1-t)z_1+t z_7,\qquad 0\leq t\leq 1
    $$
    """)
    return


@app.cell
def _(
    ae_latent_points,
    ae_model,
    latent_images,
    latent_labels,
    vae_model,
    vae_latent_points,
):
    ae_class_centroids = np.stack(
        [
            ae_latent_points[latent_labels == digit].mean(axis=0)
            for digit in range(10)
        ]
    )
    vae_class_centroids = np.stack(
        [
            vae_latent_points[latent_labels == digit].mean(axis=0)
            for digit in range(10)
        ]
    )
    representative_indices = np.array(
        [
            np.flatnonzero(latent_labels == digit)[
                np.linalg.norm(
                    vae_latent_points[latent_labels == digit]
                    - vae_class_centroids[digit],
                    axis=1,
                ).argmin()
            ]
            for digit in range(10)
        ]
    )
    representative_inputs = latent_images[representative_indices]
    interpolation_weights = np.linspace(0.0, 1.0, 10)
    ae_interpolation_latents = np.stack(
        [
            (1.0 - weight) * ae_class_centroids[1]
            + weight * ae_class_centroids[7]
            for weight in interpolation_weights
        ]
    )
    vae_interpolation_latents = np.stack(
        [
            (1.0 - weight) * vae_class_centroids[1]
            + weight * vae_class_centroids[7]
            for weight in interpolation_weights
        ]
    )
    model_device = next(vae_model.parameters()).device
    with torch.no_grad():
        ae_class_prototypes = torch.sigmoid(
            ae_model.decode(
                torch.tensor(ae_class_centroids, dtype=torch.float32, device=model_device)
            )
        ).cpu()
        vae_class_prototypes = torch.sigmoid(
            vae_model.decode(
                torch.tensor(vae_class_centroids, dtype=torch.float32, device=model_device)
            )
        ).cpu()
        ae_interpolation_images = torch.sigmoid(
            ae_model.decode(
                torch.tensor(
                    ae_interpolation_latents,
                    dtype=torch.float32,
                    device=model_device,
                )
            )
        ).cpu()
        vae_interpolation_images = torch.sigmoid(
            vae_model.decode(
                torch.tensor(
                    vae_interpolation_latents,
                    dtype=torch.float32,
                    device=model_device,
                )
            )
        ).cpu()
    return (
        ae_class_prototypes,
        ae_interpolation_images,
        interpolation_weights,
        representative_inputs,
        vae_class_prototypes,
        vae_interpolation_images,
    )


@app.cell
def _(
    ae_class_prototypes,
    ae_interpolation_images,
    interpolation_weights,
    representative_inputs,
    vae_class_prototypes,
    vae_interpolation_images,
):
    _fig, _axes = plt.subplots(5, 10, figsize=(14, 8))
    for _digit, (_axis, _image) in enumerate(
        zip(_axes[0], representative_inputs, strict=True)
    ):
        _axis.imshow(_image.view(28, 28), cmap="gray", vmin=0.0, vmax=1.0)
        _axis.set_title(str(_digit))
        _axis.axis("off")

    for _axis, _image in zip(_axes[1], ae_class_prototypes, strict=True):
        _axis.imshow(_image.view(28, 28), cmap="gray", vmin=0.0, vmax=1.0)
        _axis.axis("off")

    for _axis, _image in zip(_axes[2], vae_class_prototypes, strict=True):
        _axis.imshow(_image.view(28, 28), cmap="gray", vmin=0.0, vmax=1.0)
        _axis.axis("off")

    for _axis, _image, _weight in zip(
        _axes[3],
        ae_interpolation_images,
        interpolation_weights,
        strict=True,
    ):
        _axis.imshow(_image.view(28, 28), cmap="gray", vmin=0.0, vmax=1.0)
        _axis.set_title(f"{_weight:.2f}")
        _axis.axis("off")

    for _axis, _image in zip(
        _axes[4],
        vae_interpolation_images,
        strict=True,
    ):
        _axis.imshow(_image.view(28, 28), cmap="gray", vmin=0.0, vmax=1.0)
        _axis.axis("off")

    _fig.text(0.01, 0.84, "representative input", rotation=90, va="center")
    _fig.text(0.01, 0.67, "AE centroid", rotation=90, va="center")
    _fig.text(0.01, 0.50, "VAE centroid", rotation=90, va="center")
    _fig.text(0.01, 0.33, "AE: 1 to 7", rotation=90, va="center")
    _fig.text(0.01, 0.16, "VAE: 1 to 7", rotation=90, va="center")
    _fig.suptitle("Class centroids and interpolation in each 16D latent space")
    _fig.tight_layout(rect=(0.03, 0, 1, 0.96))
    _fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## まとめ

    AEは既知画像の圧縮と再構成に適し、再構成誤差だけならVAEより小さくなることがあります。
    VAEはKL項によって潜在分布を既知のpriorへ接続するため、再構成とのトレードオフを払いながら
    $z\sim\mathcal{N}(0,I)$ から新しい画像を生成できます。今回のようにAEの補間も滑らかに見える
    場合があるため、補間やt-SNEだけでは両者を区別できません。既知のpriorから生成したいならVAE、
    再構成だけが目的ならAEが自然な選択です。
    """)
    return


if __name__ == "__main__":
    app.run()
