import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")

with app.setup:
    import copy
    import importlib.util
    from pathlib import Path

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
    # MNIST 認識モデルを PyTorch で構築して変換

    この版では **TorchScript → ONNX Runtime → OpenVINO** の保守しやすい経路に整理し、
    同じ入力に対する出力差を比較します。
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

    mean, scale = 0.1307, 0.3081
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((mean,), (scale,))])
    train_dataset = datasets.MNIST("./data", train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST("./data", train=False, download=True, transform=transform)
    train_subset = Subset(train_dataset, list(range(6_000)))
    test_subset = Subset(test_dataset, list(range(1_000)))
    train_loader = DataLoader(train_subset, batch_size=128, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_subset, batch_size=256, shuffle=False, num_workers=0)
    return mean, scale, test_loader, train_loader, train_subset


@app.cell
def _(mean, scale, train_subset):
    _fig, _axes = plt.subplots(2, 5, figsize=(8, 4))
    for _idx, _axis in enumerate(_axes.ravel()):
        _image, _label = train_subset[_idx]
        _axis.imshow(_image.squeeze(0) * scale + mean, cmap="gray")
        _axis.set_title(f"label={_label}")
        _axis.axis("off")
    _fig.tight_layout()
    _fig
    return


@app.cell
def _():
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(1, 32, 3, 1)
            self.conv2 = nn.Conv2d(32, 64, 3, 1)
            self.dropout1 = nn.Dropout(0.25)
            self.dropout2 = nn.Dropout(0.5)
            self.fc1 = nn.Linear(9216, 128)
            self.fc2 = nn.Linear(128, 10)

        def forward(self, x):
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            x = F.max_pool2d(x, 2)
            x = self.dropout1(x)
            x = torch.flatten(x, 1)
            x = F.relu(self.fc1(x))
            x = self.dropout2(x)
            return F.log_softmax(self.fc2(x), dim=1)

    return Net


@app.cell
def _(Net, device):
    model = Net().to(device)
    optimizer = torch.optim.Adadelta(model.parameters(), lr=1.0)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.7)
    return model, optimizer, scheduler


@app.cell
def _():
    def evaluate(model, device, loader):
        model.eval()
        total_loss = 0.0
        correct = 0
        with torch.no_grad():
            for _batch, _target in loader:
                _batch = _batch.to(device)
                _target = _target.to(device)
                logits = model(_batch)
                total_loss += F.nll_loss(logits, _target, reduction="sum").item()
                correct += logits.argmax(dim=1).eq(_target).sum().item()
        return total_loss / len(loader.dataset), correct / len(loader.dataset)

    return evaluate


@app.cell
def _(device, evaluate, model, optimizer, scheduler, test_loader, train_loader):
    history_rows = []
    for epoch in range(1, 3):
        model.train()
        running_loss = 0.0
        for _batch, _target in train_loader:
            _batch = _batch.to(device)
            _target = _target.to(device)
            optimizer.zero_grad()
            logits = model(_batch)
            loss = F.nll_loss(logits, _target)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * _batch.size(0)
        test_loss, test_accuracy = evaluate(model, device, test_loader)
        history_rows.append({"epoch": epoch, "train_loss": running_loss / len(train_loader.dataset), "test_loss": test_loss, "test_accuracy": test_accuracy})
        scheduler.step()
    history_frame = pd.DataFrame(history_rows)
    return history_frame


@app.cell
def _(history_frame):
    history_frame
    return


@app.cell
def _(history_frame):
    _fig, _axes = plt.subplots(1, 2, figsize=(10, 4))
    _axes[0].plot(history_frame["epoch"], history_frame["train_loss"], label="train")
    _axes[0].plot(history_frame["epoch"], history_frame["test_loss"], label="test")
    _axes[0].legend()
    _axes[0].set_title("NLL loss")
    _axes[1].plot(history_frame["epoch"], history_frame["test_accuracy"], marker="o")
    _axes[1].set_ylim(0.0, 1.0)
    _axes[1].set_title("Test accuracy")
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(device, model, test_loader):
    sample_batch, sample_labels = next(iter(test_loader))
    sample_input = sample_batch[:8].to(device)
    sample_labels = sample_labels[:8]
    with torch.no_grad():
        native_logits = model(sample_input).cpu()
    return native_logits, sample_input, sample_labels


@app.cell
def _(mean, native_logits, sample_input, sample_labels, scale):
    native_predictions = native_logits.argmax(dim=1).numpy()
    _fig, _axes = plt.subplots(2, 4, figsize=(8, 4))
    for _axis, _image, _pred, _label in zip(_axes.ravel(), sample_input.cpu(), native_predictions, sample_labels, strict=False):
        _axis.imshow(_image.squeeze(0) * scale + mean, cmap="gray")
        _axis.set_title(f"pred={_pred}, true={_label.item()}")
        _axis.axis("off")
    _fig.tight_layout()
    _fig
    return native_predictions


@app.cell
def _(model, native_logits, sample_input, sample_labels):
    artifact_dir = Path(__file__).with_name("mnist_convert_artifacts")
    artifact_dir.mkdir(exist_ok=True)
    cpu_model = copy.deepcopy(model).to("cpu").eval()
    cpu_input = sample_input.cpu()
    script_path = artifact_dir / "mnist_cnn_script.pt"
    scripted_model = torch.jit.script(cpu_model)
    scripted_model.save(script_path)
    scripted_model = torch.jit.load(script_path)
    with torch.no_grad():
        native_logits_cpu = cpu_model(cpu_input)
        torchscript_logits = scripted_model(cpu_input)
    comparison = [{"runtime": "torch", "max_abs_diff": 0.0}, {"runtime": "torchscript", "max_abs_diff": float((native_logits_cpu - torchscript_logits).abs().max().item())}]
    prediction_table = pd.DataFrame({"true": sample_labels.numpy(), "torch": native_logits_cpu.argmax(dim=1).numpy(), "torchscript": torchscript_logits.argmax(dim=1).numpy()})
    return artifact_dir, comparison, cpu_input, cpu_model, native_logits_cpu, prediction_table


@app.cell
def _(artifact_dir, comparison, cpu_input, cpu_model, native_logits_cpu, prediction_table):
    onnx_path = artifact_dir / "mnist_cnn.onnx"
    _script_path = artifact_dir / "mnist_cnn_script.pt"
    has_onnx = importlib.util.find_spec("onnx") is not None and importlib.util.find_spec("onnxruntime") is not None
    if has_onnx:
        import onnx
        import onnxruntime

        torch.onnx.export(
            cpu_model,
            cpu_input,
            str(onnx_path),
            input_names=["input"],
            output_names=["logits"],
            dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
            opset_version=18,
            dynamo=False,
        )
        onnx_model = onnx.load(str(onnx_path))
        onnx.checker.check_model(onnx_model)
        ort_session = onnxruntime.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        onnx_logits = ort_session.run(None, {ort_session.get_inputs()[0].name: cpu_input.numpy().astype(np.float32)})[0]
        comparison.append({"runtime": "onnxruntime", "max_abs_diff": float(np.max(np.abs(onnx_logits - native_logits_cpu.numpy())))})
        prediction_table["onnxruntime"] = onnx_logits.argmax(axis=1)
    has_openvino = importlib.util.find_spec("openvino") is not None and onnx_path.exists()
    if has_openvino:
        import openvino as ov

        ov_model = ov.convert_model(str(onnx_path))
        compiled_model = ov.Core().compile_model(ov_model, "AUTO")
        ov_logits = compiled_model([cpu_input.numpy().astype(np.float32)])[compiled_model.output(0)]
        comparison.append({"runtime": "openvino", "max_abs_diff": float(np.max(np.abs(ov_logits - native_logits_cpu.numpy())))})
        prediction_table["openvino"] = ov_logits.argmax(axis=1)
    comparison_frame = pd.DataFrame(comparison)
    notes = []
    if not has_onnx:
        notes.append("Install the model-conversion dependency group to run the ONNX Runtime comparison.")
    if not has_openvino:
        notes.append("OpenVINO comparison is skipped unless `openvino` is available.")
    note_frame = pd.DataFrame({"note": notes}) if notes else pd.DataFrame({"note": ["All supported conversion runtimes were executed."]})
    print(comparison_frame.to_string(index=False))
    print(note_frame.to_string(index=False))
    if _script_path.exists():
        _script_path.unlink()
    if onnx_path.exists():
        onnx_path.unlink()
    if artifact_dir.exists() and not any(artifact_dir.iterdir()):
        artifact_dir.rmdir()
    mo.vstack([comparison_frame, prediction_table, note_frame])
    return


if __name__ == "__main__":
    app.run()
