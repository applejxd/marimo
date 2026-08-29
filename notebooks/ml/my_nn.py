import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")

with app.setup:
    from collections import OrderedDict
    from dataclasses import dataclass

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # 手書きニューラルネットによる数字分類

    ライブラリの自動微分に頼らず、順伝播と逆伝播を NumPy だけで実装して数字を分類する。
    レイヤは Sigmoid・Affine・Softmax-with-Loss の3種類で、それぞれ `forward` と
    `backward` を持つ。`backward` を数式から導いて実装することがこの notebook の主眼である。

    データは scikit-learn 同梱の `load_digits()` を使う。8×8 と粗いが追加の
    ダウンロードが要らず、逆伝播の実装を確かめるには十分な規模である。
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## データ

    scikit-learn の `load_digits()` は 8×8 の手書き数字1,797枚で、各画素は 0 から 16 の
    整数値である。これを16で割って $[0, 1]$ に収め、64次元のベクトルとして入力する。

    ラベルは0から9の整数だが、交差エントロピーの計算では正解クラスだけが1のベクトルが
    必要なので、`np.eye(10)[labels]` で one-hot 表現へ変換する。

    データは**学習・検証・テストの3つ**に分ける。まず75対25で学習用と保留分に分け、
    保留分をさらに半分ずつ検証用とテスト用にする。学習曲線の監視や初期値スケールの比較には
    検証データを使い、テストデータは最後に一度だけ使う。同じデータで何度も判断すると、
    その data に合わせ込んだ結果を汎化性能と取り違えるためである。
    分割はいずれも `stratify` で各クラスの比率を保つ。
    """)
    return


@app.cell
def _():
    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split

    digits = load_digits()
    features = digits.data.astype(np.float64) / 16.0
    labels = digits.target.astype(np.int64)
    x_train, x_holdout, y_train_labels, y_holdout_labels = train_test_split(features, labels, test_size=0.25, random_state=71, stratify=labels)
    x_validation, x_test, y_validation_labels, y_test_labels = train_test_split(x_holdout, y_holdout_labels, test_size=0.5, random_state=71, stratify=y_holdout_labels)
    y_train = np.eye(10)[y_train_labels]
    y_validation = np.eye(10)[y_validation_labels]
    print(f"train: {x_train.shape}, validation: {x_validation.shape}, test: {x_test.shape}, pixel range: [{features.min():.2f}, {features.max():.2f}]")
    return digits, x_test, x_train, x_validation, y_test_labels, y_train, y_validation


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    入力画像の先頭10枚を表示する。8×8という粗さでも数字が判別できることを確認する。
    """)
    return


@app.cell
def _(digits):
    _fig, _axes = plt.subplots(2, 5, figsize=(8, 4))
    for _idx, _axis in enumerate(_axes.ravel()):
        _axis.imshow(digits.images[_idx], cmap="gray_r")
        _axis.set_title(f"label={digits.target[_idx]}")
        _axis.axis("off")
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## レイヤの実装

    以降のレイヤはすべて**バッチを行方向に並べた規約**で書く。入力 $X$ は
    $N \times D_{\text{in}}$ 行列で、$n$ 行目が $n$ 番目のサンプルである。教科書でよく見る
    $Y = WX + b$（サンプルを列に並べる規約）とは転置の関係にあるので、
    式に現れる行列の順序と転置の位置が入れ替わる点に注意する。

    各レイヤは次の2つのメソッドを持つ。

    - `forward(x)`：入力から出力を計算し、逆伝播で必要になる中間結果を自身に保存する
    - `backward(dout)`：出力側から来た勾配 $\partial L/\partial Y$ を受け取り、
      入力側へ渡す勾配 $\partial L/\partial X$ を返す。重みを持つレイヤは
      自身のパラメータの勾配も保存する
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Sigmoid レイヤ

    シグモイド関数と、その導関数が出力だけで書けることを使う。

    $$
    \sigma(x) = \frac{1}{1+e^{-x}}, \qquad
    \sigma'(x) = \frac{e^{-x}}{\left(1+e^{-x}\right)^2} = \sigma(x)\left(1-\sigma(x)\right)
    $$

    このレイヤは要素ごとの写像なので、入力がベクトルでも行列でも同じ式で書ける。
    $Y = \sigma(X)$ とすると、逆伝播は要素ごとの積（アダマール積）になる。

    $$
    \frac{\partial L}{\partial X}
    = \frac{\partial L}{\partial Y} \odot Y \odot \left(1 - Y\right)
    $$

    導関数が $\sigma$ の**出力**だけで書けるので、`forward` で `self.out` を保存しておけば
    `backward` は入力を持たなくても計算できる。入力は $N \times D$ の配列、
    出力は同じ形の配列である。
    """)
    return


@app.cell
def _():
    class Sigmoid:
        def __init__(self):
            self.out = None

        def forward(self, x):
            self.out = 1.0 / (1.0 + np.exp(-x))
            return self.out

        def backward(self, dout):
            return dout * self.out * (1.0 - self.out)

    return Sigmoid


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Affine レイヤ

    バッチを行に並べた規約では、順伝播は次の形になる。$W$ は
    $D_{\text{in}} \times D_{\text{out}}$、$b$ は $D_{\text{out}}$ 次元で、
    $b$ は全サンプルへ同じ値が足される（ブロードキャスト）。

    $$
    Y = XW + \mathbf{1}_N b^{\top},
    \qquad X \in \mathbb{R}^{N \times D_{\text{in}}},\ Y \in \mathbb{R}^{N \times D_{\text{out}}}
    $$

    $\partial L/\partial Y$ を $\mathrm{d}Y$ と書くと、連鎖律から3つの勾配が得られる。
    $W$ の勾配と $b$ の勾配はバッチ方向に和を取る形になる。

    $$
    \frac{\partial L}{\partial X} = \mathrm{d}Y\,W^{\top}, \qquad
    \frac{\partial L}{\partial W} = X^{\top}\mathrm{d}Y, \qquad
    \frac{\partial L}{\partial b} = \sum_{n=1}^{N} \mathrm{d}Y_{n,:}
    $$

    形を追うと転置の位置が確認できる。$\mathrm{d}Y$ は $N \times D_{\text{out}}$、
    $W^{\top}$ は $D_{\text{out}} \times D_{\text{in}}$ なので積は $X$ と同じ
    $N \times D_{\text{in}}$、$X^{\top}\mathrm{d}Y$ は $W$ と同じ
    $D_{\text{in}} \times D_{\text{out}}$ になる。

    `forward` は入力 $X$ を保存する。$W$ の勾配計算に必要だからである。
    出力は $N \times D_{\text{out}}$ の配列、`backward` の戻り値は $N \times D_{\text{in}}$ の配列で、
    副作用として `dw` と `db` が更新される。
    """)
    return


@app.cell
def _():
    @dataclass
    class Affine:
        w: np.ndarray
        b: np.ndarray
        x: np.ndarray | None = None
        dw: np.ndarray | None = None
        db: np.ndarray | None = None

        def forward(self, x):
            self.x = x
            return x @ self.w + self.b

        def backward(self, dout):
            self.dw = self.x.T @ dout
            self.db = np.sum(dout, axis=0)
            return dout @ self.w.T

    return Affine


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Softmax と交差エントロピー

    softmax は指数関数を使うため、入力が大きいと `exp` があふれる。分子と分母に同じ
    定数を掛けても値が変わらない性質を使い、**行ごとの最大値**を引いてから指数を取る。
    引いた後の最大の logit が0になるので、指数の最大値が $e^0=1$ に収まる。

    $$
    y_{nk} = \frac{\exp\left(a_{nk} - C_n\right)}{\sum_{i} \exp\left(a_{ni} - C_n\right)},
    \qquad C_n = \max_{i} a_{ni}
    $$

    交差エントロピーは対数を取るため、確率が0に近いと $\log y$ が $-\infty$ へ、
    したがって損失 $-\log y$ が $+\infty$ へ発散する。softmax の出力を経由してから
    対数を取ると、小さい確率が丸めで0になった時点で計算が壊れる。logit から
    log-sum-exp を使って対数確率を直接求めれば、この丸めを経由しない。
    $a_{nk} - C_n \le 0$ なので指数はあふれず、対数の引数も1以上になる。
    バッチに対してはサンプルごとの値の平均を損失とする。

    $$
    \log y_{nk} = \left(a_{nk} - C_n\right) - \log \sum_{i} \exp\left(a_{ni} - C_n\right)
    $$

    $$
    L = -\frac{1}{N}\sum_{n=1}^{N}\sum_{k} t_{nk} \log y_{nk}
    $$

    softmax と交差エントロピーを1つのレイヤにまとめる利点は、逆伝播が次のように
    引き算だけになることである。個別に微分すると商の微分が現れるが、合成すると打ち消し合う。
    損失をバッチ平均で定義したので、勾配も $N$ で割る。微小量を足して対数を守る実装だと
    この式は近似になるが、log-sum-exp で計算しているので厳密に成り立つ。

    $$
    \frac{\partial L}{\partial A} = \frac{Y - T}{N}
    $$

    `forward` の入力は Affine の出力（logit）と one-hot の正解で、出力はスカラーの損失である。
    `backward` の戻り値は logit と同じ $N \times 10$ の配列である。
    """)
    return


@app.cell
def _():
    def softmax(x):
        shifted = x - np.max(x, axis=1, keepdims=True)
        exp_x = np.exp(shifted)
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def cross_entropy_error(logits, y_true):
        shifted = logits - np.max(logits, axis=1, keepdims=True)
        log_probs = shifted - np.log(np.sum(np.exp(shifted), axis=1, keepdims=True))
        return -np.mean(np.sum(y_true * log_probs, axis=1))

    @dataclass
    class SoftmaxWithLoss:
        loss: float | None = None
        y_pred: np.ndarray | None = None
        y_true: np.ndarray | None = None

        def forward(self, x, y_true):
            self.y_true = y_true
            self.y_pred = softmax(x)
            self.loss = cross_entropy_error(x, y_true)
            return self.loss

        def backward(self, dout=1.0):
            return dout * (self.y_pred - self.y_true) / self.y_true.shape[0]

    return SoftmaxWithLoss


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## ネットワークの組み立て

    Affine → Sigmoid → Affine → Sigmoid → Affine の5層を `OrderedDict` に順番に積む。
    64次元の入力を64、32と変換し、最後に10クラスの logit を出す。
    `predict` はこの順に `forward` を呼ぶだけ、`gradient` は逆順に `backward` を呼ぶだけである。

    **重みの初期値スケールは学習の進み方を大きく左右する。** ここでは Xavier の初期化を使い、
    層ごとに入力次元 $D_{\text{in}}$ に応じて標準偏差を変える。

    $$
    W \sim \mathcal{N}\!\left(0, \frac{1}{D_{\text{in}}}\right),
    \qquad \text{標準偏差} = \frac{1}{\sqrt{D_{\text{in}}}}
    $$

    入力次元が大きいほど和を取る項数が増えるので、1項あたりの大きさを小さくして
    レイヤを通したときの分散を揃える、という考え方である。この notebook の構成では
    $D_{\text{in}}=64$ の層で 0.125、$D_{\text{in}}=32$ の層で 0.177 になる。
    後の節で、これを定数 0.05 に固定した場合と比較する。

    実装上の注意が1つある。`self.layers` の各 Affine は `self.params` の配列を
    **同じオブジェクトとして**参照している。そのため学習ループの
    `params[key] -= lr * grad[key]` という**その場での**更新はレイヤ側にも反映される。
    ここを `params[key] = params[key] - lr * grad[key]` と書くと新しい配列に束ね直され、
    レイヤは古い配列を参照したままになり、まったく学習が進まない。
    """)
    return


@app.cell
def _(Affine, Sigmoid, SoftmaxWithLoss):
    class MyNet:
        def __init__(self, input_size, hidden_sizes, output_size, rng):
            def xavier(fan_in, fan_out):
                return np.sqrt(1.0 / fan_in) * rng.standard_normal((fan_in, fan_out))

            self.params = {
                "W1": xavier(input_size, hidden_sizes[0]),
                "b1": np.zeros(hidden_sizes[0]),
                "W2": xavier(hidden_sizes[0], hidden_sizes[1]),
                "b2": np.zeros(hidden_sizes[1]),
                "W3": xavier(hidden_sizes[1], output_size),
                "b3": np.zeros(output_size),
            }
            self.layers = OrderedDict([
                ("Affine1", Affine(self.params["W1"], self.params["b1"])),
                ("Sigmoid1", Sigmoid()),
                ("Affine2", Affine(self.params["W2"], self.params["b2"])),
                ("Sigmoid2", Sigmoid()),
                ("Affine3", Affine(self.params["W3"], self.params["b3"])),
            ])
            self.last_layer = SoftmaxWithLoss()

        def predict(self, x):
            for layer in self.layers.values():
                x = layer.forward(x)
            return x

        def loss(self, x, y_true):
            return self.last_layer.forward(self.predict(x), y_true)

        def accuracy(self, x, y_true):
            logits = self.predict(x)
            return float(np.mean(np.argmax(logits, axis=1) == np.argmax(y_true, axis=1)))

        def gradient(self, x, y_true):
            self.loss(x, y_true)
            dout = self.last_layer.backward()
            for layer in reversed(list(self.layers.values())):
                dout = layer.backward(dout)
            return {
                "W1": self.layers["Affine1"].dw,
                "b1": self.layers["Affine1"].db,
                "W2": self.layers["Affine2"].dw,
                "b2": self.layers["Affine2"].db,
                "W3": self.layers["Affine3"].dw,
                "b3": self.layers["Affine3"].db,
            }

    return MyNet


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 勾配確認

    逆伝播は式を手で導いて実装しているので、間違えても損失がそれらしく下がってしまい、
    誤りに気付けないことがある。そこで、解析的に求めた勾配を数値微分と突き合わせる。
    中心差分は次の式である。

    $$
    \frac{\partial L}{\partial \theta} \simeq \frac{L(\theta + h) - L(\theta - h)}{2h}
    $$

    打ち切り誤差は $O(h^2)$ で減るが、$h$ を小さくしすぎると引き算の桁落ちが効いて
    かえって悪化する。ここでは $h=10^{-5}$ を使う。

    全パラメータを数値微分すると重いので、各パラメータ配列から5箇所ずつ抜き出して比べる。
    抜き取り検査なので実装全体の正しさを証明するものではないが、
    誤差が十分小さければ実装が正しいという確信は大きく高まる。
    絶対誤差と相対誤差の両方を表示する。
    """)
    return


@app.cell
def _(MyNet, x_train, y_train):
    _check_rng = np.random.default_rng(0)
    _check_net = MyNet(input_size=x_train.shape[1], hidden_sizes=[8, 6], output_size=10, rng=_check_rng)
    _x_check = x_train[:5]
    _y_check = y_train[:5]
    _analytic = _check_net.gradient(_x_check, _y_check)

    _h = 1e-5
    _rows = []
    for _name, _param in _check_net.params.items():
        _flat = _param.ravel()
        _positions = _check_rng.choice(_flat.size, size=min(5, _flat.size), replace=False)
        _worst_absolute = 0.0
        _worst_relative = 0.0
        for _position in _positions:
            _original = _flat[_position]
            _flat[_position] = _original + _h
            _loss_plus = _check_net.loss(_x_check, _y_check)
            _flat[_position] = _original - _h
            _loss_minus = _check_net.loss(_x_check, _y_check)
            _flat[_position] = _original
            _numeric = (_loss_plus - _loss_minus) / (2.0 * _h)
            _exact = _analytic[_name].ravel()[_position]
            _absolute = abs(_numeric - _exact)
            _worst_absolute = max(_worst_absolute, _absolute)
            _worst_relative = max(_worst_relative, _absolute / max(1e-12, abs(_numeric) + abs(_exact)))
        _rows.append({
            "parameter": _name,
            "checked_entries": len(_positions),
            "max_absolute_error": f"{_worst_absolute:.2e}",
            "max_relative_error": f"{_worst_relative:.2e}",
        })
    gradient_check_frame = pd.DataFrame(_rows)
    gradient_check_frame
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 学習

    確率的勾配降下法で学習する。毎回、学習データから重複なしで64件を選び、
    その勾配で全パラメータを更新する。学習率は0.5、反復回数は3,000回である。
    1,347件の学習データに対して 3000 × 64 / 1347 ≒ 143 エポックに相当する。

    $$
    \theta \leftarrow \theta - \eta \frac{\partial L}{\partial \theta}, \qquad \eta = 0.5
    $$

    初回と20回ごとに、そのミニバッチの損失と、学習データ・検証データ全体の正解率を記録する。
    損失は更新後に測っているので、その反復の結果を表す。
    比較の目安として、10クラスを当てずっぽうに答えるときの交差エントロピーは
    $\log 10 \approx 2.303$、正解率は0.1である。
    """)
    return


@app.cell
def _(MyNet, x_train, x_validation, y_train, y_validation):
    rng = np.random.default_rng(71)
    network = MyNet(input_size=x_train.shape[1], hidden_sizes=[64, 32], output_size=y_train.shape[1], rng=rng)
    history_rows = []
    for iteration in range(1, 3001):
        batch_indices = rng.choice(len(x_train), size=64, replace=False)
        x_batch = x_train[batch_indices]
        y_batch = y_train[batch_indices]
        gradients = network.gradient(x_batch, y_batch)
        for key in gradients:
            network.params[key] -= 0.5 * gradients[key]
        if iteration == 1 or iteration % 20 == 0:
            history_rows.append({"iteration": iteration, "loss": network.loss(x_batch, y_batch), "train_accuracy": network.accuracy(x_train, y_train), "validation_accuracy": network.accuracy(x_validation, y_validation)})

    history_frame = pd.DataFrame(history_rows)
    return history_frame, network


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    記録の末尾5行である。損失が当てずっぽうの2.303から大きく下がり、
    学習データと検証データの正解率が揃って上がっていれば、学習は成功している。
    """)
    return


@app.cell
def _(history_frame):
    history_frame.tail()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    左はミニバッチ損失、右は正解率の推移である。ミニバッチ損失は64件だけで測るため
    上下に振れるが、全体の傾向は下降する。右の図で学習用と検証用の曲線が
    大きく離れていなければ、極端な過学習は起きていない。
    """)
    return


@app.cell
def _(history_frame):
    _fig, _axes = plt.subplots(1, 2, figsize=(10, 4))
    _axes[0].plot(history_frame["iteration"], history_frame["loss"], marker="o", markersize=3)
    _axes[0].axhline(np.log(10.0), color="tab:red", linestyle="--", label="chance level")
    _axes[0].set_xlabel("iteration")
    _axes[0].set_title("mini-batch loss")
    _axes[0].legend()
    _axes[1].plot(history_frame["iteration"], history_frame["train_accuracy"], label="train")
    _axes[1].plot(history_frame["iteration"], history_frame["validation_accuracy"], label="validation")
    _axes[1].set_xlabel("iteration")
    _axes[1].set_ylim(0.0, 1.0)
    _axes[1].legend()
    _axes[1].set_title("accuracy")
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 初期値スケールの影響

    初期化の話を実測で確かめる。同じ構成・同じ学習率で、重みの標準偏差だけを
    Xavier（$1/\sqrt{D_{\text{in}}}$）と定数 0.05 に変えて比較する。

    定数 0.05 は $D_{\text{in}}=64$ の層に対して Xavier の 0.125 の半分以下である。
    重みが小さいと、Affine の逆伝播 $\mathrm{d}X = \mathrm{d}Y\,W^{\top}$ で小さな $W$ が
    掛かるため、入口へ向かうほど勾配が縮む。順伝播でも Affine の出力が0付近に集中し、
    シグモイドの出力が0.5付近に固まって、サンプルごとの差が出にくくなる。

    この条件で実際に効いたのは**収束の速さ**だった。反復を十分に取ればどちらも同程度まで
    学習するが、初期の伸びが大きく違う。次のセルでは50反復ごとに検証正解率を測り、
    正解率0.9に最初に到達した反復と、学習途中の目安として600反復時点の正解率を比べる。
    ここでもテストデータは使わない。
    """)
    return


@app.cell
def _(Affine, Sigmoid, SoftmaxWithLoss, x_train, x_validation, y_train, y_validation):
    def train_with_scale(scale_mode, iterations=3000, learning_rate=0.5, seed=71):
        local_rng = np.random.default_rng(seed)

        def make_weight(fan_in, fan_out):
            scale = np.sqrt(1.0 / fan_in) if scale_mode == "xavier" else 0.05
            return scale * local_rng.standard_normal((fan_in, fan_out))

        params = {
            "W1": make_weight(x_train.shape[1], 64),
            "b1": np.zeros(64),
            "W2": make_weight(64, 32),
            "b2": np.zeros(32),
            "W3": make_weight(32, 10),
            "b3": np.zeros(10),
        }
        layers = [
            Affine(params["W1"], params["b1"]),
            Sigmoid(),
            Affine(params["W2"], params["b2"]),
            Sigmoid(),
            Affine(params["W3"], params["b3"]),
        ]
        last_layer = SoftmaxWithLoss()
        names = ["W1", "b1", "W2", "b2", "W3", "b3"]
        curve = []
        for step in range(1, iterations + 1):
            indices = local_rng.choice(len(x_train), size=64, replace=False)
            activation = x_train[indices]
            for layer in layers:
                activation = layer.forward(activation)
            last_layer.forward(activation, y_train[indices])
            grad = last_layer.backward()
            for layer in reversed(layers):
                grad = layer.backward(grad)
            affine_layers = [layer for layer in layers if isinstance(layer, Affine)]
            for position, layer in enumerate(affine_layers):
                params[names[2 * position]] -= learning_rate * layer.dw
                params[names[2 * position + 1]] -= learning_rate * layer.db
            if step == 1 or step % 50 == 0:
                probe = x_validation
                for layer in layers:
                    probe = layer.forward(probe)
                curve.append((step, float(np.mean(np.argmax(probe, axis=1) == np.argmax(y_validation, axis=1)))))
        return curve

    scale_curves = {mode: train_with_scale(mode) for mode in ["xavier", "fixed_0.05"]}

    def _summarize(curve):
        reached = [step for step, value in curve if value >= 0.9]
        at_600 = next((value for step, value in curve if step == 600), float("nan"))
        return reached[0] if reached else None, at_600, curve[-1][1]

    scale_summary = pd.DataFrame(
        [
            {
                "init": mode,
                "iterations_to_reach_0.9": _summarize(curve)[0],
                "validation_accuracy_at_600": round(_summarize(curve)[1], 4),
                "final_validation_accuracy": round(_summarize(curve)[2], 4),
            }
            for mode, curve in scale_curves.items()
        ]
    )
    scale_summary
    return scale_curves


@app.cell
def _(scale_curves):
    _fig, _ax = plt.subplots(figsize=(6, 4))
    for _name, _curve in scale_curves.items():
        _ax.plot([_point[0] for _point in _curve], [_point[1] for _point in _curve], label=_name)
    _ax.axhline(0.1, color="tab:red", linestyle="--", label="chance level")
    _ax.set_xlabel("iteration")
    _ax.set_ylabel("validation accuracy")
    _ax.set_ylim(0.0, 1.0)
    _ax.set_title("Effect of weight initialization scale")
    _ax.legend()
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    曲線の立ち上がりの差がそのまま学習の速さの差である。定数0.05の側は序盤ほとんど
    当てずっぽうの水準（0.1）に留まり、後から追いついてくる。反復回数が足りないまま
    打ち切ると、実装が正しくても「学習できていない」という誤った結論になりかねない。
    初期値スケールは、モデルの表現力ではなく最適化の効きやすさを左右する設定である。
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 評価

    ここまでの判断はすべて検証データで行った。最後に、一度も参照していない
    テストデータ225件で汎化性能を測る。scikit-learn の `accuracy_score` を使うので、
    自前実装の `network.accuracy` とは独立した計算になり、突き合わせの意味もある。
    """)
    return


@app.cell
def _(network, x_test, y_test_labels):
    from sklearn.metrics import accuracy_score, confusion_matrix

    test_predictions = np.argmax(network.predict(x_test), axis=1)
    accuracy = accuracy_score(y_test_labels, test_predictions)
    confusion = confusion_matrix(y_test_labels, test_predictions)
    metrics_frame = pd.DataFrame({"metric": ["test_accuracy"], "value": [round(float(accuracy), 4)]})
    return confusion, metrics_frame


@app.cell
def _(metrics_frame):
    metrics_frame
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    混同行列で、どの数字をどの数字と取り違えたかを見る。行が正解、列が予測で、
    対角成分が正解数である。対角から外れた位置の数字が、形の似た組み合わせに
    偏っていないかを確認する。
    """)
    return


@app.cell
def _(confusion):
    _fig, _ax = plt.subplots(figsize=(6, 5))
    _image = _ax.imshow(confusion, cmap="Blues")
    for _row in range(confusion.shape[0]):
        for _col in range(confusion.shape[1]):
            _ax.text(_col, _row, confusion[_row, _col], ha="center", va="center", fontsize=8)
    _ax.set_xticks(range(10))
    _ax.set_yticks(range(10))
    _ax.set_xlabel("predicted")
    _ax.set_ylabel("true")
    _ax.set_title("Confusion matrix")
    _fig.colorbar(_image, ax=_ax)
    _fig.tight_layout()
    _fig
    return


if __name__ == "__main__":
    app.run()
