import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")

with app.setup:
    import base64
    import io

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

    条件付き生成の意図は残しつつ、UNet を小さな MLP デノイザへ置き換えて学習を実用的な長さにしています。
    ラベル埋め込みと classifier-free guidance を使い、forward noising・学習曲線・生成結果を notebook 内で確認します。
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

    transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x * 2.0 - 1.0)])
    train_dataset = datasets.MNIST("./data", train=True, download=True, transform=transform)
    train_subset = Subset(train_dataset, list(range(3_000)))
    train_loader = DataLoader(train_subset, batch_size=128, shuffle=True, num_workers=0)
    return train_loader, train_subset


@app.cell
def _(train_subset):
    _fig, _axes = plt.subplots(2, 5, figsize=(8, 4))
    for _idx, _axis in enumerate(_axes.ravel()):
        _image, _label = train_subset[_idx]
        _axis.imshow(((_image.squeeze(0) + 1.0) / 2.0), cmap="gray")
        _axis.set_title(f"label={_label}")
        _axis.axis("off")
    _fig.tight_layout()
    _fig
    return


@app.cell
def _():
    num_steps = 30
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


@app.cell
def _(q_sample, train_subset):
    image, label = train_subset[0]
    noisy_versions = []
    for _step in [0, 9, 19, 29]:
        _noise = torch.randn_like(image).unsqueeze(0)
        _noised = q_sample(image.unsqueeze(0), torch.tensor([_step]), _noise)[0]
        noisy_versions.append((_step + 1, _noised.squeeze(0)))
    return image, label, noisy_versions


@app.cell
def _(image, label, noisy_versions):
    _fig, _axes = plt.subplots(1, 5, figsize=(12, 3))
    _axes[0].imshow(((image.squeeze(0) + 1.0) / 2.0), cmap="gray")
    _axes[0].set_title(f"x₀ label={label}")
    _axes[0].axis("off")
    for _axis, (_step, _noised) in zip(_axes[1:], noisy_versions, strict=False):
        _axis.imshow(((_noised + 1.0) / 2.0), cmap="gray")
        _axis.set_title(f"t={_step}")
        _axis.axis("off")
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(num_steps):
    class Denoiser(nn.Module):
        def __init__(self, hidden_dim=256, null_label=10):
            super().__init__()
            self.null_label = null_label
            self.time_mlp = nn.Sequential(nn.Linear(1, 64), nn.SiLU(), nn.Linear(64, 64))
            self.label_embedding = nn.Embedding(11, 64)
            self.net = nn.Sequential(nn.Linear(28 * 28 + 64 + 64, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 28 * 28))

        def forward(self, x, t, labels):
            x_flat = x.view(x.shape[0], -1)
            t_embed = self.time_mlp((t.float() / num_steps).unsqueeze(1))
            label_embed = self.label_embedding(labels)
            return self.net(torch.cat([x_flat, t_embed, label_embed], dim=1)).view_as(x)

    return Denoiser


@app.cell
def _(Denoiser, device):
    model = Denoiser().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    return model, optimizer


@app.cell
def _(alpha_bars, betas, device, model, num_steps):
    @torch.no_grad()
    def sample(labels, guidance_scale=2.0):
        model.eval()
        x = torch.randn(len(labels), 1, 28, 28, device=device)
        labels = labels.to(device)
        null_labels = torch.full_like(labels, model.null_label)
        snapshots = []
        denoising_frames = []
        for step in reversed(range(num_steps)):
            timestep = torch.full((len(labels),), step, device=device, dtype=torch.long)
            cond_noise = model(x, timestep, labels)
            uncond_noise = model(x, timestep, null_labels)
            eps = uncond_noise + guidance_scale * (cond_noise - uncond_noise)
            alpha = (1.0 - betas[step]).to(device)
            alpha_bar = alpha_bars[step].to(device)
            noise = torch.randn_like(x) if step > 0 else torch.zeros_like(x)
            x = (x - (1 - alpha) / torch.sqrt(1 - alpha_bar) * eps) / torch.sqrt(alpha) + torch.sqrt(betas[step]).to(device) * noise
            denoising_frames.append(x.detach().cpu())
            if step in {29, 19, 9, 0}:
                snapshots.append((step + 1, x.detach().cpu()))
        return x.detach().cpu(), snapshots, denoising_frames

    return sample


@app.cell
def _(alphas, device, model, num_steps, optimizer, q_sample, train_loader):
    history_rows = []
    for epoch in range(1, 4):
        model.train()
        running_loss = 0.0
        running_items = 0
        for _batch, _labels in train_loader:
            _batch = _batch.to(device)
            _labels = _labels.to(device)
            timestep = torch.randint(0, num_steps, (_batch.shape[0],), device=device)
            noise = torch.randn_like(_batch)
            noisy_batch = q_sample(_batch, timestep, noise)
            drop_mask = torch.rand(_batch.shape[0], device=device) < 0.1
            conditioned_labels = _labels.clone()
            conditioned_labels[drop_mask] = model.null_label
            pred_noise = model(noisy_batch, timestep, conditioned_labels)
            loss = F.mse_loss(pred_noise, noise)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * _batch.shape[0]
            running_items += _batch.shape[0]
        history_rows.append({"epoch": epoch, "train_loss": running_loss / running_items})
    history_frame = pd.DataFrame(history_rows)
    return history_frame


@app.cell
def _(history_frame):
    history_frame
    return


@app.cell
def _(history_frame):
    _fig, _ax = plt.subplots(figsize=(6, 4))
    _ax.plot(history_frame["epoch"], history_frame["train_loss"], marker="o")
    _ax.set_xlabel("epoch")
    _ax.set_ylabel("MSE on predicted noise")
    _ax.set_title("Diffusion training loss")
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(sample):
    labels = torch.tensor(list(range(10)) + list(range(10)))
    generated, snapshots, denoising_frames = sample(labels)
    print(f"generated batch shape: {tuple(generated.shape)}")
    return denoising_frames, generated, labels, snapshots


@app.cell
def _(generated, labels):
    _fig, _axes = plt.subplots(2, 10, figsize=(12, 3))
    for _axis, _image, _label in zip(_axes.ravel(), generated, labels, strict=False):
        _axis.imshow(((_image.squeeze(0) + 1.0) / 2.0), cmap="gray")
        _axis.set_title(str(_label.item()))
        _axis.axis("off")
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(snapshots):
    _fig, _axes = plt.subplots(len(snapshots), 10, figsize=(12, 5))
    for _row, (_step, _batch) in enumerate(snapshots):
        for _col in range(10):
            _axes[_row, _col].imshow(((_batch[_col, 0] + 1.0) / 2.0), cmap="gray")
            _axes[_row, _col].axis("off")
            if _col == 0:
                _axes[_row, _col].set_ylabel(f"t={_step}")
    _fig.suptitle("Denoising snapshots for labels 0-9", y=0.95)
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(denoising_frames):
    _pil_frames = []
    for _batch in denoising_frames:
        _images = (
            ((_batch[:, 0].clamp(-1.0, 1.0) + 1.0) / 2.0 * 255.0)
            .to(torch.uint8)
            .numpy()
        )
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
        duration=120,
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


if __name__ == "__main__":
    app.run()
