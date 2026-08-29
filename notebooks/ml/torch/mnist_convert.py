import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")

with app.setup:
    import copy
    import importlib
    import importlib.metadata
    import importlib.util
    import os
    import platform
    import time
    from pathlib import Path

    import japanize_matplotlib  # noqa: F401
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

    小さな畳み込みネットワークを PyTorch で学習し、**TorchScript → ONNX Runtime →
    OpenVINO → ExecuTorch** の順に変換して、次の 2 点を確認する。

    1. **出力の一致**：どの変換先でも、学習済みモデルと同じ logits・同じ予測が得られるか
    2. **推論時間の違い**：同じ重み・同じ入力でも、実行するプラットフォーム
       （CPU か GPU か、どの推論 runtime か、どの backend か）で速度がどれだけ変わるか

    GPU を使えるかは runtime によって違う。ONNX Runtime は同じ ONNX ファイルから CPU と
    CUDA の session を両方作れるので、**モデルを固定したまま実行先だけを CPU と GPU で切り替えた
    比較**ができる。OpenVINO の GPU プラグインは Intel 製 GPU 専用、ExecuTorch の XNNPACK
    backend は CPU 専用なので、この 2 つは CPU での比較になる。

    後半では batch size を変えながら 1 回の推論にかかる時間を測り、差が生じる理由を実測値と
    対応付けて整理する。
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 共通部品と再現性

    冒頭の `with app.setup` ブロック（marimo 上では折りたたまれている）で、notebook 全体から
    使う標準ライブラリと marimo・Matplotlib・NumPy・pandas・PyTorch を import している。

    次のセルでは乱数 seed を 42 に固定し、CUDA・MPS・CPU の順に利用可能な device を選ぶ。
    ここで選んだ `device` は**学習にのみ**使う。変換と比較は、どの環境でも同じ手順になるよう
    CPU 上の重みから行う。
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
    return (device,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 実行環境の記録

    推論時間はハードウェアとスレッド数に強く依存するため、比較の前提として実行環境を表に
    残す。このページに載っている数値は、下の表の環境で測ったものである。

    - 入力：なし（実行中のプロセスから取得）
    - 出力：`available` = 変換先ライブラリの有無、`environment_frame` = 環境情報の `DataFrame`
    - `torch.get_num_threads()` は PyTorch が CPU 演算に使うスレッド数である。ONNX Runtime・
      OpenVINO・ExecuTorch はそれぞれ独自のスレッドプールを持ち、既定値も異なる
    """)
    return


@app.cell
def _():
    def package_version(name):
        if importlib.util.find_spec(name) is None:
            return "not installed"
        for _distribution in importlib.metadata.packages_distributions().get(name, [name]):
            try:
                return f"{_distribution} {importlib.metadata.version(_distribution)}"
            except importlib.metadata.PackageNotFoundError:
                continue
        return "unknown"

    available = {
        name: importlib.util.find_spec(name) is not None
        for name in ("onnx", "onnxruntime", "openvino", "executorch")
    }
    return available, package_version


@app.cell
def _(device, package_version):
    def cpu_name():
        if importlib.util.find_spec("openvino") is not None:
            _openvino = importlib.import_module("openvino")
            try:
                return str(_openvino.Core().get_property("CPU", "FULL_DEVICE_NAME")).strip()
            except Exception:
                pass
        return platform.processor() or platform.machine()

    def runtime_devices():
        rows = []
        if importlib.util.find_spec("onnxruntime") is not None:
            _onnxruntime = importlib.import_module("onnxruntime")
            rows.append(("ONNX Runtime の provider", ", ".join(_onnxruntime.get_available_providers())))
        if importlib.util.find_spec("openvino") is not None:
            _openvino = importlib.import_module("openvino")
            rows.append(("OpenVINO の device", ", ".join(_openvino.Core().available_devices)))
        return rows

    environment_frame = pd.DataFrame(
        [
            ("OS", platform.platform()),
            ("CPU", cpu_name()),
            ("論理コア数", str(os.cpu_count())),
            ("torch スレッド数", str(torch.get_num_threads())),
            ("学習 device", str(device)),
            ("GPU", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "なし"),
            ("torch", torch.__version__),
            ("onnxruntime", package_version("onnxruntime")),
            ("openvino", package_version("openvino")),
            ("executorch", package_version("executorch")),
            *runtime_devices(),
        ],
        columns=["item", "value"],
    )
    environment_frame
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## データ

    torchvision の MNIST を `data/` へ download し、`ToTensor` で `[0, 1]` に変換したあと、
    平均 0.1307・標準偏差 0.3081 で正規化する。この notebook の目的は精度の追求ではなく
    変換経路の比較なので、学習は先頭 6,000 件、評価は先頭 1,000 件だけを使う。

    - 前処理：`Normalize((0.1307,), (0.3081,))`（MNIST 全画素の平均と標準偏差）
    - 出力：`train_loader`（batch size 128・shuffle あり）、`test_loader`（batch size 256・順序固定）
    - 次のセルでは、正規化を戻して学習データの先頭 10 枚を表示する
    """)
    return


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


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## モデル

    畳み込み 2 層と全結合 2 層の小さな CNN である。28×28 の入力は `conv1` で 26×26、`conv2` で
    24×24 になり、`max_pool2d` で 12×12 へ縮小されるため、全結合層の入力は
    64 × 12 × 12 = 9,216 次元になる。出力は 10 クラスの対数確率である。

    `F.max_pool2d(x, 2, 2)` と **stride を明示**している点だけ、素朴な実装と異なる。stride を
    省略しても既定で kernel size と同じ 2 になり計算結果は変わらないが、後半で使う ExecuTorch の
    XNNPACK backend は stride 引数が省略された `max_pool2d` を解釈できず、変換が失敗する。
    """)
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
            x = F.max_pool2d(x, 2, 2)
            x = self.dropout1(x)
            x = torch.flatten(x, 1)
            x = F.relu(self.fc1(x))
            x = self.dropout2(x)
            return F.log_softmax(self.fc2(x), dim=1)

    return (Net,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 学習の準備

    最適化には Adadelta（学習率 1.0）を使い、`StepLR` で 1 epoch ごとに学習率を 0.7 倍する。
    損失は `log_softmax` 出力に対応する negative log likelihood である。続く 2 セルで、最適化器と
    評価関数を用意する。

    `evaluate()` は指定した device 上でモデルを評価する。

    - 入力：`model`（評価するモデル）、`device`、`loader`（`DataLoader`）
    - 処理：`eval()` と `no_grad()` の下で全 batch を推論し、損失の合計と正解数を集計する
    - 出力：`(平均損失, 正解率)` のタプル
    """)
    return


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

    return (evaluate,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 学習

    2 epoch だけ学習する。各 epoch の終わりに評価用データで損失と正解率を測り、
    `history_frame` へ記録する。続く 2 セルで、その記録を表と図で確認する。
    """)
    return


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
    return (history_frame,)


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


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 変換前の基準となる推論

    評価用データの先頭 8 枚を、以降のすべての比較で共通の入力として使う。まず学習に使った
    device 上のモデルで推論し、その予測を図で確認する。

    - 出力：`sample_input`（8×1×28×28 の tensor・学習 device 上）、`sample_labels`（正解ラベル）、
      `native_logits`（CPU へ戻した logits）
    """)
    return


@app.cell
def _(device, model, test_loader):
    sample_batch, sample_labels = next(iter(test_loader))
    sample_input = sample_batch[:8].to(device)
    sample_labels = sample_labels[:8]
    model.eval()
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
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 変換の共通準備

    変換は CPU 上の重みから行う。学習済みモデルを `deepcopy` して CPU へ移し、`eval()` で
    dropout を無効にした `cpu_model` を、すべての変換と測定の元にする。

    - 出力：`artifact_dir` = notebook と同じ場所に作る一時ディレクトリ（最後のセルで削除する）、
      `cpu_model` = CPU 上の推論用モデル、`cpu_input` = 8 枚の入力、
      `native_logits_cpu` = 比較の基準になる logits
    """)
    return


@app.cell
def _(model, sample_input):
    artifact_dir = Path(__file__).with_name("mnist_convert_artifacts")
    artifact_dir.mkdir(exist_ok=True)
    cpu_model = copy.deepcopy(model).to("cpu").eval()
    cpu_input = sample_input.cpu()
    with torch.no_grad():
        native_logits_cpu = cpu_model(cpu_input)
    return artifact_dir, cpu_input, cpu_model, native_logits_cpu


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 1. TorchScript

    `torch.jit.script` は Python で書いたモデル定義を TorchScript の中間表現へ変換する。Python
    インタプリタなしで実行できる形式になるが、呼び出される演算そのものは PyTorch と同じ
    ATen カーネルである。

    - 入力：`cpu_model`、`cpu_input`
    - 処理：script 化してファイルへ保存し、読み直してから推論する（保存と読み込みを経由する
      ことで、実際に配布する形と同じ経路を測る）
    - 出力：`scripted_model`、`torchscript_logits`、`torchscript_setup_ms` = 変換にかかった時間 [ms]
    """)
    return


@app.cell
def _(artifact_dir, cpu_input, cpu_model):
    script_path = artifact_dir / "mnist_cnn_script.pt"
    _start = time.perf_counter()
    torch.jit.script(cpu_model).save(script_path)
    scripted_model = torch.jit.load(script_path)
    torchscript_setup_ms = (time.perf_counter() - _start) * 1e3
    with torch.no_grad():
        torchscript_logits = scripted_model(cpu_input)
    print(f"TorchScript: {torchscript_setup_ms:.0f} ms, {script_path.stat().st_size / 1e6:.2f} MB")
    return scripted_model, torchscript_logits, torchscript_setup_ms


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 2. ONNX と ONNX Runtime（CPU と CUDA）

    ONNX は framework に依存しないグラフ表現で、ONNX Runtime はそれを実行する推論エンジンである。
    `dynamic_axes` で batch 次元を可変にしているため、1 つのファイルでどの batch size も実行できる。

    ここが他の 3 経路と違う点として、**同じ ONNX ファイルから CPU と GPU の session を両方作れる**。
    ONNX Runtime は演算の実装を execution provider（EP）として差し替える設計で、`onnxruntime-gpu`
    package には `CPUExecutionProvider` と `CUDAExecutionProvider` の両方が含まれるためである。
    グラフも重みも共通なので、**同じモデルを CPU で動かすか GPU で動かすか**だけを比べられる。

    - 入力：`cpu_model`、`cpu_input`（グラフを trace するための例）
    - 処理：opset 18 で export → `onnx.checker` で妥当性を検査 → CPU EP の session を作成し、
      CUDA EP が使えれば同じファイルから GPU 用の session も作成
    - 出力：`onnx_logits` / `onnx_cuda_logits`、`ort_session` / `ort_cuda_session`、
      `onnx_setup_ms` / `onnx_cuda_setup_ms`
    - session 作成時にグラフ最適化（定数の畳み込み、演算の融合、メモリ確保の計画）が走るため、
      `onnx_setup_ms` には export だけでなく最適化の時間も含まれる。`onnx_cuda_setup_ms` は
      export 済みのファイルから session を作る時間だけである
    - `session.run()` は host 上の NumPy 配列を受け取って返すため、CUDA EP の測定値には
      host と device の間の転送が含まれる。PyTorch の「転送込み」と同じ土俵の値である
    """)
    return


@app.cell
def _(artifact_dir, available, cpu_input, cpu_model):
    onnx_path = artifact_dir / "mnist_cnn.onnx"
    onnx_logits = None
    onnx_cuda_logits = None
    onnx_cuda_fp32_logits = None
    ort_session = None
    ort_cuda_session = None
    onnx_setup_ms = float("nan")
    onnx_cuda_setup_ms = float("nan")
    if available["onnx"]:
        import onnx

        _start = time.perf_counter()
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
        onnx.checker.check_model(onnx.load(str(onnx_path)))
        if available["onnxruntime"]:
            import onnxruntime

            ort_session = onnxruntime.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
            onnx_setup_ms = (time.perf_counter() - _start) * 1e3
            _input_name = ort_session.get_inputs()[0].name
            _input_values = cpu_input.numpy().astype(np.float32)
            onnx_logits = ort_session.run(None, {_input_name: _input_values})[0]
            print(f"ONNX Runtime: {onnx_setup_ms:.0f} ms, provider={ort_session.get_providers()[0]}")
            if "CUDAExecutionProvider" in onnxruntime.get_available_providers() and torch.cuda.is_available():
                _cuda_start = time.perf_counter()
                ort_cuda_session = onnxruntime.InferenceSession(str(onnx_path), providers=["CUDAExecutionProvider"])
                onnx_cuda_setup_ms = (time.perf_counter() - _cuda_start) * 1e3
                onnx_cuda_logits = ort_cuda_session.run(None, {_input_name: _input_values})[0]
                _fp32_session = onnxruntime.InferenceSession(
                    str(onnx_path),
                    providers=[("CUDAExecutionProvider", {"use_tf32": 0})],
                )
                onnx_cuda_fp32_logits = _fp32_session.run(None, {_input_name: _input_values})[0]
                print(f"ONNX Runtime CUDA: {onnx_cuda_setup_ms:.0f} ms, provider={ort_cuda_session.get_providers()[0]}")
    return (
        onnx_cuda_fp32_logits,
        onnx_cuda_logits,
        onnx_cuda_setup_ms,
        onnx_logits,
        onnx_path,
        onnx_setup_ms,
        ort_cuda_session,
        ort_session,
    )


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 3. OpenVINO

    OpenVINO は Intel 系ハードウェア向けに最適化された推論 runtime である。ONNX ファイルを中間表現へ
    変換し、`compile_model` の時点で対象デバイス向けのカーネルを生成する。

    GPU で動かすこともできるが、対象は Intel の内蔵 GPU や Arc であり、NVIDIA GPU は扱えない。
    実際にこの環境の `available_devices` は `CPU` だけである（「実行環境の記録」の表を参照）。
    そのため OpenVINO は CPU での比較に限定する。

    - 入力：前のセルで書き出した `onnx_path`
    - 処理：`convert_model` → `compile_model("CPU")`。CPU の推論 runtime どうしを比べるのが目的
      なので device を明示し、実際に使われた device を `EXECUTION_DEVICES` から記録して確認する
    - 出力：`openvino_logits`、`compiled_model`、`openvino_device`、`openvino_setup_ms`
    """)
    return


@app.cell
def _(available, cpu_input, onnx_path):
    openvino_logits = None
    compiled_model = None
    openvino_device = "なし"
    openvino_setup_ms = float("nan")
    if available["openvino"] and onnx_path.exists():
        import openvino as ov

        _start = time.perf_counter()
        compiled_model = ov.Core().compile_model(ov.convert_model(str(onnx_path)), "CPU")
        openvino_setup_ms = (time.perf_counter() - _start) * 1e3
        openvino_device = ", ".join(compiled_model.get_property("EXECUTION_DEVICES"))
        openvino_logits = compiled_model([cpu_input.numpy().astype(np.float32)])[compiled_model.output(0)]
        print(f"OpenVINO: {openvino_setup_ms:.0f} ms, device={openvino_device}")
    return compiled_model, openvino_device, openvino_logits, openvino_setup_ms


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 4. ExecuTorch

    ExecuTorch はスマートフォンや組み込み機器での実行を想定した PyTorch の runtime である。ここまでの
    3 つと違い、**入力 shape ごとに事前コンパイル（AOT）した `.pte` を作る**点が特徴である。実行時に
    グラフを組み替えないぶん、必要なメモリを静的に決められる。

    変換は 3 段階である。

    1. `torch.export.export`：モデルを ATen 演算のグラフへ変換する（batch size は固定）
    2. `to_edge_transform_and_lower`：edge 方言へ落とし、`partitioner` に渡した backend が扱える
       部分グラフを delegate として切り出す
    3. `to_executorch`：メモリ計画まで済ませた `.pte` を生成する

    `build_executorch_method()` はこの一連の処理をまとめた関数である。

    - 入力：`module`（CPU 上の `nn.Module`）、`sample`（この shape 専用に固めるための入力）、
      `partition`（CPU 向けの XNNPACK backend へ delegate するなら `True`、参照実装の
      portable kernels のままにするなら `False`）、`tag`（ファイル名の識別子）
    - 出力：`(実行できる method, `.pte` のパス, import から method 読み込みまでの時間 [ms])` のタプル
    """)
    return


@app.cell
def _(artifact_dir, available):
    def build_executorch_method(module, sample, partition=True, tag="xnnpack"):
        start = time.perf_counter()
        from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
        from executorch.exir import to_edge_transform_and_lower
        from executorch.runtime import Runtime

        exported = torch.export.export(module, (sample,))
        partitioners = [XnnpackPartitioner()] if partition else []
        program = to_edge_transform_and_lower(exported, partitioner=partitioners).to_executorch()
        path = artifact_dir / f"mnist_cnn_{tag}_{sample.shape[0]}.pte"
        path.write_bytes(program.buffer)
        method = Runtime.get().load_program(str(path)).load_method("forward")
        setup_ms = (time.perf_counter() - start) * 1e3
        return method, path, setup_ms

    executorch_ready = available["executorch"]
    return build_executorch_method, executorch_ready


@app.cell
def _(build_executorch_method, cpu_input, cpu_model, executorch_ready):
    executorch_logits = None
    executorch_setup_ms = float("nan")
    if executorch_ready:
        _method, _path, executorch_setup_ms = build_executorch_method(cpu_model, cpu_input)
        executorch_logits = _method.execute([cpu_input])[0].detach()
        print(f"ExecuTorch (XNNPACK): {executorch_setup_ms:.0f} ms, {_path.stat().st_size / 1e6:.2f} MB")
    return executorch_logits, executorch_setup_ms


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 出力の一致

    基準となる `cpu_model` の logits と、各変換先の logits の最大絶対差を比べる。予測ラベルは
    どの runtime でも完全に一致するが、演算の順序や融合の仕方、そして**使う数値形式**が
    runtime ごとに違うため、差がちょうど 0 になるとは限らない。ONNX Runtime は CPU EP と
    CUDA EP を別々に検査するので、**GPU で実行したときに何がどれだけ変わるか**も確認できる。

    一致は文章で主張するだけでなく、`assert` で検査する。将来 runtime や環境が変わって出力が
    ずれた場合、このセルが失敗して page の生成自体が止まるため、誤った主張が公開されない。

    ただし **GPU だけは許容誤差を分けている**。Ampere 世代以降の NVIDIA GPU では、ONNX Runtime の
    CUDA EP が畳み込みに既定で **TF32** を使う。TF32 は指数部を float32 と同じにしたまま仮数部を
    10 bit に減らす形式で、行列積を大幅に速くする代わりに精度が落ちる。実際、provider option
    `use_tf32: 0` を渡して TF32 を無効にすると、差は CPU と同じ水準まで小さくなる。下の表では
    既定と TF32 無効の両方を並べて、この違いを確認できるようにした。

    - 許容誤差：float32 で計算する runtime は `1e-4` 未満、TF32 を使う CUDA EP の既定は `5e-3` 未満
    - どの runtime でも**予測ラベルは基準と完全一致**することを要求する
    - 出力：`comparison_frame` = runtime ごとの最大絶対差と許容値、`prediction_table` = 8 枚の予測、
      `note_frame` = 利用できず比較から外した runtime の一覧
    """)
    return


@app.cell
def _(available, executorch_logits, native_logits_cpu, onnx_cuda_fp32_logits, onnx_cuda_logits, onnx_logits, openvino_logits, sample_labels, torchscript_logits):
    def to_array(values):
        if values is None:
            return None
        if isinstance(values, torch.Tensor):
            return values.detach().numpy()
        return np.asarray(values)

    reference_logits = native_logits_cpu.detach().numpy()
    reference_predictions = reference_logits.argmax(axis=1)
    float32_tolerance = 1e-4
    tf32_tolerance = 5e-3
    candidate_logits = {
        "torch (CPU)": (reference_logits, float32_tolerance),
        "torchscript": (to_array(torchscript_logits), float32_tolerance),
        "onnxruntime (CPU)": (to_array(onnx_logits), float32_tolerance),
        "onnxruntime (CUDA, 既定=TF32)": (to_array(onnx_cuda_logits), tf32_tolerance),
        "onnxruntime (CUDA, TF32 無効)": (to_array(onnx_cuda_fp32_logits), float32_tolerance),
        "openvino": (to_array(openvino_logits), float32_tolerance),
        "executorch": (to_array(executorch_logits), float32_tolerance),
    }
    comparison_rows = []
    prediction_table = pd.DataFrame({"true": sample_labels.numpy()})
    for _name, (_values, _tolerance) in candidate_logits.items():
        if _values is None:
            continue
        if _values.shape != reference_logits.shape:
            raise AssertionError(f"{_name} の出力 shape が基準と一致しません: {_values.shape}")
        if not np.isfinite(_values).all():
            raise AssertionError(f"{_name} の出力に有限でない値が含まれています")
        _max_abs_diff = float(np.max(np.abs(_values - reference_logits)))
        _predictions = _values.argmax(axis=1)
        if _max_abs_diff >= _tolerance:
            raise AssertionError(f"{_name} の最大絶対差 {_max_abs_diff:.3e} が許容値 {_tolerance:.0e} を超えました")
        if not np.array_equal(_predictions, reference_predictions):
            raise AssertionError(f"{_name} の予測ラベルが基準と一致しません")
        comparison_rows.append({"runtime": _name, "max_abs_diff": _max_abs_diff, "許容値": _tolerance})
        prediction_table[_name] = _predictions
    comparison_frame = pd.DataFrame(comparison_rows)
    _notes = [f"{_name} が見つからないため比較から除外しました" for _name, _found in available.items() if not _found]
    if available["openvino"] and not available["onnx"]:
        _notes.append("openvino は ONNX が無く入力ファイルを作れないため比較から除外しました")
    note_frame = pd.DataFrame({"note": _notes or ["すべての変換先を実行しました"]})
    mo.vstack([comparison_frame, prediction_table, note_frame])
    return comparison_frame, note_frame, prediction_table


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 推論時間の測り方

    同じ重み・同じ入力に対して、batch size を 1・8・64・256 と変えながら 1 回の推論にかかる時間を
    測る。測定条件は次のとおりである。

    - **warmup を「5 回以上、かつ合計 0.3 秒以上」実行してから測る**。初回の呼び出しには
      cuDNN のアルゴリズム選択、メモリアリーナの確保、遅延初期化といった一度きりのコストが
      含まれるためである。加えて GPU は負荷がかかるまでクロックが上がらないので、測定を始める前に
      1 秒間だけ空回ししておく
    - 各測定の直前に 0.1 秒待つ。ONNX Runtime や XNNPACK のスレッドプールは仕事が終わった
      あともしばらく spin して CPU を占有するため、直前に測った runtime の影響を受けないようにする
    - 30 回測って**中央値**を採る。平均だと OS のスケジューリングによる外れ値に引きずられる
      （1 回が 5 ms を下回る条件では、複数回をまとめて計測して 1 回あたりに割り戻す。
      また 1 つの条件が 2.5 秒を超える場合は途中で打ち切る）
    - GPU を使う条件（PyTorch の 2 つと ONNX Runtime CUDA）は、CPU の runtime より先に測る。
      CPU 側のスレッドプールの影響を受けにくくするためである
    - PyTorch の GPU 実行は非同期なので、`synchronize()` で完了を待ってから時刻を読む。
      ONNX Runtime の `run()` は結果を host へ返して戻るため、同期は不要である
    - PyTorch の GPU は「計算のみ」と「入力の転送と結果の取り出しを含む」の 2 通りを測る。実際の
      アプリケーションで支払うのは後者である。ONNX Runtime CUDA は NumPy 配列を受け渡す API のため
      常に転送込みで、PyTorch の「転送込み」と比べるのが公平である
    - ExecuTorch は batch size ごとに `.pte` を作り直す（AOT で shape を固定するため）。
      ONNX Runtime と OpenVINO は batch 次元を可変にしたモデルを使い回す

    続く 4 セルで、測定用のヘルパー、測定に使う入力、GPU 用のモデルと GPU の warm-up を用意する。
    `measure_ms()` は `call()` を繰り返し実行し、1 回あたりの所要時間の中央値 [ms] を返す。
    """)
    return


@app.cell
def _(device):
    def synchronize():
        if device.type == "cuda":
            torch.cuda.synchronize()
        elif device.type == "mps" and hasattr(torch, "mps"):
            torch.mps.synchronize()

    def measure_ms(call, repeats=30, warmup=5, min_warmup_s=0.3, target_sample_s=0.005, max_total_s=2.5, settle_s=0.1):
        time.sleep(settle_s)
        warm_start = time.perf_counter()
        warm_count = 0
        while warm_count < warmup or time.perf_counter() - warm_start < min_warmup_s:
            call()
            warm_count += 1
        average_s = (time.perf_counter() - warm_start) / warm_count
        inner = max(1, min(200, int(target_sample_s / max(average_s, 1e-9))))
        samples = []
        loop_start = time.perf_counter()
        while len(samples) < repeats:
            start = time.perf_counter()
            for _ in range(inner):
                call()
            samples.append((time.perf_counter() - start) * 1e3 / inner)
            if len(samples) >= 5 and time.perf_counter() - loop_start > max_total_s:
                break
        return float(np.median(samples))

    return measure_ms, synchronize


@app.cell
def _(test_loader):
    bench_batch_sizes = (1, 8, 64, 256)
    bench_pool = torch.cat([_images for _images, _labels in test_loader])[: max(bench_batch_sizes)].contiguous()
    print(f"benchmark pool: {tuple(bench_pool.shape)}")
    return bench_batch_sizes, bench_pool


@app.cell
def _(device, model):
    device_model = copy.deepcopy(model).to(device).eval()
    return (device_model,)


@app.cell
def _(bench_pool, device, device_model, synchronize):
    if device.type != "cpu":
        _warm_input = bench_pool.to(device)
        _warm_start = time.perf_counter()
        _warm_iterations = 0
        while time.perf_counter() - _warm_start < 1.0:
            with torch.no_grad():
                device_model(_warm_input)
            _warm_iterations += 1
        synchronize()
        gpu_warmup_note = f"{device.type} を {_warm_iterations} 回空回ししました"
    else:
        gpu_warmup_note = "GPU が無いため warm-up は不要です"
    print(gpu_warmup_note)
    return (gpu_warmup_note,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    次のセルが測定の本体である。batch size ごとに入力を切り出し、利用できる runtime を順に測って
    `latency_frame` に記録する。

    - 出力：`latency_frame` = 列 `batch_size`・`runtime`・`latency_ms`（1 回の推論時間の中央値）・
      `per_image_us`（1 画像あたりに換算した時間）
    - GPU が無い環境では GPU の行が現れず、CPU の runtime だけが比較される
    """)
    return


@app.cell
def _(bench_batch_sizes, bench_pool, build_executorch_method, compiled_model, cpu_model, device, device_model, executorch_ready, gpu_warmup_note, measure_ms, ort_cuda_session, ort_session, scripted_model, synchronize):
    print(f"benchmark 開始（{gpu_warmup_note}）")
    latency_rows = []
    gpu_labels = set()
    for _size in bench_batch_sizes:
        _batch_cpu = bench_pool[:_size].contiguous()
        _batch_np = _batch_cpu.numpy().astype(np.float32)
        _cases = {}

        def _eager(_tensor=_batch_cpu):
            with torch.no_grad():
                cpu_model(_tensor)

        def _script(_tensor=_batch_cpu):
            with torch.no_grad():
                scripted_model(_tensor)

        _cases["PyTorch eager (CPU)"] = _eager
        _cases["TorchScript (CPU)"] = _script
        if ort_session is not None:
            _input_name = ort_session.get_inputs()[0].name
            _cases["ONNX Runtime (CPU)"] = lambda _values=_batch_np, _name=_input_name: ort_session.run(None, {_name: _values})
        if compiled_model is not None:
            _cases["OpenVINO (CPU)"] = lambda _values=_batch_np: compiled_model([_values])[compiled_model.output(0)]
        if executorch_ready:
            _method, _path, _ = build_executorch_method(cpu_model, _batch_cpu)
            _cases["ExecuTorch XNNPACK (CPU)"] = lambda _tensor=_batch_cpu, _runner=_method: _runner.execute([_tensor])
        if ort_cuda_session is not None:
            _cuda_input_name = ort_cuda_session.get_inputs()[0].name
            _cases["ONNX Runtime (CUDA) 転送込み"] = lambda _values=_batch_np, _name=_cuda_input_name: ort_cuda_session.run(None, {_name: _values})
            gpu_labels.add("ONNX Runtime (CUDA) 転送込み")
        if device.type != "cpu":
            _batch_device = _batch_cpu.to(device)

            def _gpu_compute(_tensor=_batch_device):
                with torch.no_grad():
                    device_model(_tensor)
                synchronize()

            def _gpu_end_to_end(_tensor=_batch_cpu):
                with torch.no_grad():
                    device_model(_tensor.to(device)).cpu()
                synchronize()

            _cases[f"PyTorch ({device.type}) 計算のみ"] = _gpu_compute
            _cases[f"PyTorch ({device.type}) 転送込み"] = _gpu_end_to_end
            gpu_labels.update({f"PyTorch ({device.type}) 計算のみ", f"PyTorch ({device.type}) 転送込み"})
        _measure_order = sorted(_cases, key=lambda _name: 0 if _name in gpu_labels else 1)
        _measured = {_label: measure_ms(_cases[_label]) for _label in _measure_order}
        for _label in _cases:
            _median_ms = _measured[_label]
            latency_rows.append({"batch_size": _size, "runtime": _label, "latency_ms": _median_ms, "per_image_us": _median_ms / _size * 1e3})
    latency_frame = pd.DataFrame(latency_rows)
    return (latency_frame,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 測定結果

    上の表が 1 回の推論にかかる時間 [ms]、下の表が 1 画像あたりに換算した時間 [µs] である。
    行が runtime、列 `batch=N` が batch size である。`batch=64` なら、64 枚を 1 回の呼び出しで
    まとめて推論したときの時間を表す。
    """)
    return


@app.cell
def _(latency_frame):
    runtime_order = list(dict.fromkeys(latency_frame["runtime"]))

    def pivot_by_batch(values, digits):
        table = latency_frame.pivot(index="runtime", columns="batch_size", values=values).reindex(runtime_order).round(digits)
        table.columns = [f"batch={_size}" for _size in table.columns]
        return table.rename_axis(index="runtime", columns=None).reset_index()

    latency_pivot = pivot_by_batch("latency_ms", 3)
    per_image_pivot = pivot_by_batch("per_image_us", 1)
    mo.vstack([mo.md("**1 回の推論時間 [ms]**"), latency_pivot, mo.md("**1 画像あたり [µs]**"), per_image_pivot])
    return


@app.cell
def _(latency_frame):
    _fig, _axes = plt.subplots(1, 2, figsize=(11, 4))
    for _name, _group in latency_frame.groupby("runtime"):
        _sorted = _group.sort_values("batch_size")
        _axes[0].plot(_sorted["batch_size"], _sorted["latency_ms"], marker="o", label=_name)
        _axes[1].plot(_sorted["batch_size"], _sorted["per_image_us"], marker="o", label=_name)
    for _axis, _title, _ylabel in ((_axes[0], "Latency per call", "ms"), (_axes[1], "Latency per image", "us")):
        _axis.set_xscale("log", base=2)
        _axis.set_yscale("log")
        _axis.set_xlabel("batch size")
        _axis.set_ylabel(_ylabel)
        _axis.set_title(_title)
        _axis.grid(True, which="both", alpha=0.3)
    _axes[1].legend(fontsize=8, loc="center left", bbox_to_anchor=(1.02, 0.5))
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## backend を外すとどうなるか（ExecuTorch）

    ExecuTorch の速度は delegate する backend に強く依存する。`partition=False` で変換すると、
    どの環境でも動くことを優先した参照実装（portable kernels）だけで実行される。同じ `.pte` 形式・
    同じ重みでも、XNNPACK へ delegate した場合と桁違いの差が出ることを確認する。

    - 入力：`cpu_model`、`cpu_input`（8 枚）
    - 処理：`partition=False` で `.pte` を作り、5 回測って中央値を採る（非常に遅いため回数を
      減らしている）
    - 出力：`backend_frame` = backend ごとの推論時間と、XNNPACK に対する倍率
    """)
    return


@app.cell
def _(build_executorch_method, cpu_input, cpu_model, executorch_ready, latency_frame, measure_ms):
    backend_frame = pd.DataFrame({"note": ["executorch が無いため測定していません"]})
    if executorch_ready:
        _method, _path, _setup_ms = build_executorch_method(cpu_model, cpu_input, partition=False, tag="portable")
        _portable_ms = measure_ms(lambda: _method.execute([cpu_input]), repeats=5, warmup=2)
        _matched = latency_frame[(latency_frame["batch_size"] == 8) & (latency_frame["runtime"] == "ExecuTorch XNNPACK (CPU)")]
        _xnnpack_ms = float(_matched["latency_ms"].iloc[0])
        backend_frame = pd.DataFrame(
            [
                {"backend": "XNNPACK delegate", "latency_ms": round(_xnnpack_ms, 3), "XNNPACK 比": 1.0},
                {"backend": "portable kernels", "latency_ms": round(_portable_ms, 3), "XNNPACK 比": round(_portable_ms / _xnnpack_ms, 1)},
            ]
        )
    backend_frame
    return (backend_frame,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 一度きりのコスト

    推論時間の比較には現れないが、変換・最適化・コンパイルにかかる時間も配置先を選ぶ材料に
    なる。ここまでのセルで測った所要時間をまとめる。TorchScript は保存と読み込み、ONNX は
    export と session 作成、ExecuTorch は `.pte` の書き出しと method の読み込みまでを含み、いずれも
    「実行できる状態になるまで」を測っている。OpenVINO と ONNX Runtime の CUDA session だけは
    前のセルで書き出した ONNX を入力とするため、ONNX export の時間は含まない。ExecuTorch は
    batch size ごとにこの時間を
    払う点と、この表の値には初回だけ発生する module の import 時間も含まれる点に注意する。
    """)
    return


@app.cell
def _(executorch_setup_ms, onnx_cuda_setup_ms, onnx_setup_ms, openvino_device, openvino_setup_ms, torchscript_setup_ms):
    setup_frame = pd.DataFrame(
        [
            {"step": "TorchScript script + save + load", "milliseconds": round(torchscript_setup_ms, 1), "備考": "shape 非依存"},
            {"step": "ONNX export + checker + CPU session", "milliseconds": round(onnx_setup_ms, 1), "備考": "batch 可変"},
            {"step": "ONNX Runtime CUDA session", "milliseconds": round(onnx_cuda_setup_ms, 1), "備考": "export 済み ONNX から"},
            {"step": "OpenVINO convert + compile", "milliseconds": round(openvino_setup_ms, 1), "備考": f"device={openvino_device}"},
            {"step": "ExecuTorch export + lower (batch=8)", "milliseconds": round(executorch_setup_ms, 1), "備考": "shape ごとに必要"},
        ]
    )
    setup_frame
    return (setup_frame,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## なぜプラットフォームで差が出るのか

    測定結果の傾向と、その理由を整理する。数値は上の表を参照。

    ### batch size が小さいと GPU は速くならない

    batch size が 1 のとき、GPU の推論時間は CPU の runtime と同程度かむしろ遅くなる。この
    領域では実際の計算よりも、**カーネル起動（数十 µs × 層数）と PCIe 経由のデータ転送**という
    固定費が支配的だからである。batch size を増やすと、この固定費が多数の画像で薄まり、かつ数千の
    演算器を同時に使えるようになるため、1 画像あたりの時間が急激に下がる。「転送込み」は
    host から device への転送・結果の取り出し・同期まで含めた実運用に近い値で、「計算のみ」との差は
    1 ms 未満である。この差は測定ごとのばらつきと同程度なので、batch size に対する傾向として
    読み取ることはできない。

    ### 推論専用 runtime は大きい batch で効く

    ONNX Runtime・OpenVINO・ExecuTorch(XNNPACK) は、モデルを読み込んだ時点でグラフ全体を最適化
    する。

    - 定数の畳み込みと、Conv と ReLU のような連続する演算の融合により、中間 tensor の書き戻しが減る
    - 畳み込みに適したメモリ配置へあらかじめ変換し、重みも並べ替えておける
    - 実行時に確保するメモリの計画を事前に立てられる

    一方 PyTorch eager は、呼び出しのたびに Python から 1 演算ずつ dispatch し、中間 tensor を都度
    確保する。柔軟さと引き換えのオーバーヘッドで、計算量が増える大きい batch では推論専用 runtime が
    eager の数倍速くなる（正確な倍率は測定のたびに変わるため、上の表の値を参照）。

    ただし**小さい batch では逆転しうる**点に注意する。上の表の batch size 1 の行が示すと
    おり、推論専用 runtime が必ず eager より速いわけではなく、この環境では ExecuTorch が eager より
    遅く測れている。1 回の呼び出しあたりの固定費（入力の受け渡し、
    動的 shape の解決、自前のスレッドプールへの仕事の割り当て）が、実際の計算より大きくなるためである。
    同じ理由で、どの runtime が最速かは batch size によって入れ替わる。

    ### 同じ ONNX でも実行先で速度も数値も変わる

    ONNX Runtime は同じファイル・同じ重みのまま EP を差し替えるだけで実行先を変えられる。
    上の表の `ONNX Runtime (CPU)` と `ONNX Runtime (CUDA) 転送込み` は、**モデル以外の条件を
    揃えたうえで CPU と GPU を比べた値**である。CUDA EP は host との転送を毎回含むため、小さい
    batch では CPU EP に勝てず、batch を大きくするほど有利になる。

    変わるのは速度だけではない。CUDA EP は既定で TF32 を使うため、出力の一致表で見たとおり
    CPU EP より 1 桁以上大きい差が出る。予測ラベルは変わらなかったが、**実行先を変えると
    数値も変わりうる**ことは、回帰テストの許容誤差を決めるときに効いてくる。

    なお GPU を使えるかどうかは runtime に依存する。OpenVINO の GPU プラグインは Intel 製 GPU
    専用で、この環境の NVIDIA GPU は対象外である。ExecuTorch の XNNPACK backend も CPU 専用である。
    「GPU 対応」と言うときは、**どの runtime のどの backend が、どのベンダの GPU に対応するか**まで
    確認する必要がある。

    ### TorchScript は eager とほとんど変わらない

    TorchScript は Python インタプリタを介さずに実行できるが、呼び出すカーネルは eager と同じ
    ATen 実装である。したがって差が出るのは、Python 側のオーバーヘッドが相対的に大きい小さい batch
    だけで、畳み込みの計算そのものが支配的になる大きい batch では eager とほぼ同じになる。
    「script 化すれば速くなる」わけではなく、TorchScript の主目的は Python 非依存の配布形式である
    ことが読み取れる。

    ### ExecuTorch は backend 次第

    ExecuTorch は runtime 自体を小さく保ち、演算の実装は backend に委ねる設計である。参照実装の
    portable kernels は可搬性を最優先しているため 2 桁以上遅く、XNNPACK へ delegate して初めて他の
    CPU runtime と同じ土俵に乗る。ExecuTorch の性能を評価するときは、**どの backend へ
    delegate されたか**を必ず確認する必要がある。

    ### 静的 shape か動的 shape か

    ExecuTorch は AOT の時点で shape を固定し、メモリ計画まで決めてしまう。実行時の判断が
    減るぶん組み込み機器で扱いやすい反面、batch size ごとに `.pte` を作り直す必要がある。
    ONNX Runtime と OpenVINO は batch 次元を可変にできるが、その代わり実行のたびに shape に
    応じた処理が入る。用途が「固定 shape で繰り返し推論する端末」なのか「入力サイズが変わる
    サーバ」なのかで、選ぶべき経路が変わる。

    ### 測定値を読むときの注意

    ここまでの数値は「実行環境の記録」の表にあるマシンで測ったものである。CPU の世代と利用可能な
    SIMD 命令、割り当てスレッド数、GPU の種類、OS（この環境は WSL2）によって傾向は変わる。
    特に **1 画像あたりの時間は batch size に対して単調に減るとは限らない**。上の表でも、
    CPU の runtime のいくつかは途中で悪化している。小さい batch では固定費が、
    大きい batch ではメモリ帯域とキャッシュの効き方が効くためである。自分の配置環境で、実際に使う
    batch size で測り直すことが重要である。
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 生成物の削除

    比較に使った `.pt`・`.onnx`・`.pte` を削除し、一時ディレクトリを片付ける。measurement 用の
    セルより後に実行されるよう、測定結果を入力として受け取っている。
    """)
    return


@app.cell
def _(artifact_dir, backend_frame, latency_frame, setup_frame):
    removed_files = []
    if artifact_dir.exists():
        for _artifact in sorted(artifact_dir.iterdir()):
            if _artifact.is_file():
                _artifact.unlink()
                removed_files.append(_artifact.name)
        if not any(artifact_dir.iterdir()):
            artifact_dir.rmdir()
    pd.DataFrame({"removed": removed_files or ["削除対象なし"]})
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## まとめ

    - 学習済みの重みは、TorchScript・ONNX Runtime・OpenVINO・ExecuTorch のいずれへ変換しても
      予測ラベルが一致した。CPU で実行する限り logits の差も float32 の丸め誤差の範囲である
    - ただし CUDA EP は既定で TF32 を使うため、GPU では差が 1 桁以上大きくなる。**実行先を
      変えると数値も変わりうる**ので、許容誤差は runtime と device ごとに決める必要がある
    - 一方で推論時間は、同じモデルでもプラットフォームによって桁単位で変わる。決めるのは
      「CPU か GPU か」だけではなく、**batch size・推論 runtime・delegate する backend**の
      組み合わせである
    - GPU は batch をまとめられる用途で効く。1 件ずつ低遅延で返す用途では、CPU の推論専用
      runtime のほうが速く、この環境では batch size 1 で ONNX Runtime の CPU EP が
      GPU の条件より速く応答した
    - GPU を使えるかは runtime 依存である。ONNX Runtime は EP を差し替えるだけで CPU と CUDA を
      切り替えられるが、OpenVINO の GPU は Intel 製 GPU 専用、ExecuTorch の XNNPACK は CPU 専用である
    - 端末側で動かすなら ExecuTorch のように AOT で shape を固定する経路が向くが、backend を
      delegate しなければ性能は出ない
    """)
    return


if __name__ == "__main__":
    app.run()
