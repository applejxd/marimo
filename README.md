# marimo による科学技術 notebook

科学技術計算と機械学習のサンプルを、実行可能な
[marimo](https://marimo.io/) notebook として [`notebooks/`](notebooks/) で
管理しています。

## 実行環境

依存関係の管理には [uv](https://docs.astral.sh/uv/) を使用します。
対応する Python のバージョンは 3.11 です。

```shell
uv sync --all-groups
```

`gpu` グループでは PyTorch と TensorFlow をインストールします。すべての notebook から
サイトを生成する場合は CUDA 対応 GPU の使用を想定していますが、CUDA を利用できない
環境では notebook が明示的に CPU を選択します。`specialized` グループには NGSolve や
Open3D などの大規模なパッケージやネイティブパッケージが含まれます。
`notebooks/others/cpp.py` の実行には OpenMP 対応の C++ コンパイラも必要です。

API キーなどの秘密情報を必要とする notebook はありません。公開データセットは固定した
URL から取得し、Git の追跡対象外である `data/` ディレクトリへキャッシュします。

## notebook の実行と編集

```shell
uv run --all-groups marimo edit notebooks/algorithm/GradientDescent.py
uv run --all-groups marimo run notebooks/algorithm/GradientDescent.py
```

## 検証

```shell
uv run ruff check .
uv run marimo check --strict notebooks
uv run python scripts/check_math.py
uv run python scripts/check_site.py
```

## サイトの生成

`build_site.py` は marimo ファイルを再帰的に自動検出します。`notebooks/` 以下へ
notebook を追加すると生成される index にも自動で追加されるため、notebook の一覧を
手動で管理する必要はありません。

```shell
uv run python scripts/build_site.py
python -m http.server --directory site
```

export 時にはすべての notebook を実際に実行します。プロセスが 0 以外で終了した場合、
marimo が `some cells failed to execute` を出力した場合、または生成した HTML に
例外を示す文字列が含まれる場合は build を失敗させ、該当する出力を削除します。
また、すべてのソースと HTML の SHA-256 hash を
`site/notebooks-manifest.json` へ記録します。Pages workflow は、古い生成物や編集された
生成物を検出した場合に deploy を拒否します。notebook を実行せず、自動検出と index
生成だけを確認する場合は、次のコマンドを使用します。

```shell
uv run python scripts/build_site.py --discover-only
```

生成した `site/` ディレクトリは Git の管理対象です。GitHub Actions はこの静的ファイルを
GitHub Pages へ upload するだけで、モデルの実行やデータセットの download は行わず、
秘密情報も必要としません。初回 deploy 時に Pages の自動有効化が許可されていない場合は、
**Settings → Pages → Source** を **GitHub Actions** に設定してください。

notebook のソースを変更した push でも deploy workflow が起動します。変更した notebook
の HTML が再生成されるまでは manifest の検査が失敗するため、古いページが公開されることは
ありません。
