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


@app.cell
def _():
    from torch.utils.data import DataLoader, Subset
    from torchvision import datasets, transforms

    transform = transforms.ToTensor()
    train_dataset = datasets.MNIST("./data", train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST("./data", train=False, download=True, transform=transform)
    train_subset = Subset(train_dataset, list(range(10_000)))
    test_subset = Subset(test_dataset, list(range(2_000)))
    train_loader = DataLoader(train_subset, batch_size=128, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_subset, batch_size=256, shuffle=False, num_workers=0)
    return test_loader, test_subset, train_loader, train_subset


@app.cell
def _(train_subset):
    _fig, _axes = plt.subplots(2, 5, figsize=(8, 4))
    for _idx, _axis in enumerate(_axes.ravel()):
        _image, _label = train_subset[_idx]
        _axis.imshow(_image.squeeze(0), cmap="gray")
        _axis.set_title(f"label={_label}")
        _axis.axis("off")
    _fig.tight_layout()
    _fig
    return


@app.cell
def _():
    class VAE(nn.Module):
        def __init__(self, x_dim=784, h_dim1=256, h_dim2=128, z_dim=2):
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


@app.cell
def _():
    def loss_function(logits, x, mu, logvar):
        reconstruction = F.binary_cross_entropy_with_logits(logits, x.view(-1, 784), reduction="sum")
        kl_divergence = -0.5 * torch.sum(1 + logvar - logvar.exp() - mu.pow(2))
        return reconstruction + kl_divergence

    def evaluate(model, device, loader):
        model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for batch, _ in loader:
                batch = batch.to(device)
                logits, mu, logvar = model(batch)
                total_loss += loss_function(logits, batch, mu, logvar).item()
        return total_loss / len(loader.dataset)

    return evaluate, loss_function


@app.cell
def _(device, evaluate, loss_function, model, optimizer, test_loader, train_loader):
    history_rows = []
    for epoch in range(1, 5):
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
        history_rows.append({"epoch": epoch, "train_loss": total_train_loss / len(train_loader.dataset), "test_loss": evaluate(model, device, test_loader)})
    history_frame = pd.DataFrame(history_rows)
    return history_frame


@app.cell
def _(history_frame):
    history_frame
    return


@app.cell
def _(history_frame):
    _fig, _ax = plt.subplots(figsize=(6, 4))
    _ax.plot(history_frame["epoch"], history_frame["train_loss"], label="train")
    _ax.plot(history_frame["epoch"], history_frame["test_loss"], label="test")
    _ax.set_xlabel("epoch")
    _ax.set_ylabel("negative ELBO")
    _ax.set_title("VAE loss per sample")
    _ax.legend()
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(device, model, test_loader):
    preview_batch, preview_labels = next(iter(test_loader))
    preview_batch = preview_batch[:8].to(device)
    preview_labels = preview_labels[:8]
    with torch.no_grad():
        reconstruction_logits, _, _ = model(preview_batch)
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
    _axes[0, 0].set_ylabel("input")
    _axes[1, 0].set_ylabel("recon")
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(device, model, test_loader):
    latent_points = []
    latent_labels = []
    with torch.no_grad():
        for _batch, _labels in test_loader:
            _batch = _batch.to(device)
            _mu, _ = model.encode(_batch.view(-1, 784))
            latent_points.append(_mu.cpu())
            latent_labels.append(_labels)
            if sum(item.shape[0] for item in latent_points) >= 500:
                break
    latent_points = torch.cat(latent_points, dim=0)[:500].numpy()
    latent_labels = torch.cat(latent_labels, dim=0)[:500].numpy()
    return latent_labels, latent_points


@app.cell
def _(latent_labels, latent_points):
    _fig, _ax = plt.subplots(figsize=(6, 5))
    _scatter = _ax.scatter(latent_points[:, 0], latent_points[:, 1], c=latent_labels, cmap="tab10", s=10)
    _ax.set_title("Latent mean vectors")
    _ax.set_xlabel("z₀")
    _ax.set_ylabel("z₁")
    _fig.colorbar(_scatter, ax=_ax)
    _fig.tight_layout()
    _fig
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


@app.cell
def _(np, reloaded_model):
    latent_x = np.linspace(-2.0, 2.0, 8)
    latent_y = np.linspace(-2.0, 2.0, 8)
    latent_grid = np.stack(np.meshgrid(latent_x, latent_y), axis=-1).reshape(-1, 2)
    with torch.no_grad():
        samples = torch.sigmoid(reloaded_model.decode(torch.tensor(latent_grid, dtype=torch.float32, device=next(reloaded_model.parameters()).device))).cpu()
    return samples


@app.cell
def _(samples):
    _fig, _axes = plt.subplots(8, 8, figsize=(8, 8))
    for _axis, _image in zip(_axes.ravel(), samples, strict=False):
        _axis.imshow(_image.view(28, 28), cmap="gray")
        _axis.axis("off")
    _fig.suptitle("Decoded latent grid", y=0.92)
    _fig.tight_layout()
    _fig
    return


if __name__ == "__main__":
    app.run()
