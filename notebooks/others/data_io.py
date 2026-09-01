import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import json
    import statistics
    import time
    from pathlib import Path

    import dask.dataframe as dd
    import marimo as mo
    import numpy as np
    import orjson
    import pandas as pd
    import polars as pl
    import pyarrow

    notebook_dir = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
    repo_root = notebook_dir.parents[1]
    data_dir = repo_root / "data" / "others" / "data_io"
    data_dir.mkdir(parents=True, exist_ok=True)
    return dd, json, mo, np, orjson, pd, pl, pyarrow, statistics, time


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # データ入出力の高速化

    表データと JSON の読み書きを、形式とライブラリを変えて実測する。

    計算をいくら速くしても、その手前でデータを読むのに時間がかかっていては意味がない。
    実務では**計算の最適化に手を付ける前に、まず入出力を見るべき**ことが多い。
    このノートで扱うのは形式とライブラリの選択だけで、
    並列化や GPU 化の話は出てこない（それらは「Python 高速化メモ」で扱う）。

    比べるのは次の 2 つ。

    1. **表データ** — CSV と Parquet を、pandas・polars・dask で読む
    2. **JSON** — `json` と `orjson` で読み書きする

    絶対値は環境に強く依存するので、注目すべきは**手法どうしの比**である。
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 計測の共通部品

    計測ヘルパーを 2 つ定義する。

    - `timed_call(label, func, *args)`: `func` を 1 回だけ呼び、経過秒数を返す。
    - `benchmark(label, func, repeats=7, warmup_s=0.5)`: `warmup_s` 秒ぶん回し続けてから
      `repeats` 回計測し、中央値と最小値を返す。

    ウォームアップを挟むのは、初回にファイルがページキャッシュへ載る分や、
    ライブラリの遅延初期化を計測から外すためである。
    どちらも `time.perf_counter`（単調増加・高分解能）を使う。
    """)
    return


@app.cell
def _(statistics, time):
    def timed_call(label: str, func, *args, **kwargs):
        start = time.perf_counter()
        value = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        print(f"[{label}] {elapsed:.3g} s")
        return value, elapsed

    def benchmark(label: str, func, repeats: int = 7, warmup_s: float = 0.5):
        deadline = time.perf_counter() + warmup_s
        while True:
            func()
            if time.perf_counter() >= deadline:
                break
        samples = []
        for _ in range(repeats):
            start = time.perf_counter()
            func()
            samples.append(time.perf_counter() - start)
        median_s = statistics.median(samples)
        best_s = min(samples)
        print(f"[{label}] median {median_s * 1e3:.3f} ms / min {best_s * 1e3:.3f} ms (n={repeats})")
        return {"case": label, "median_ms": median_s * 1e3, "min_ms": best_s * 1e3}

    return benchmark, timed_call


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 実行環境

    計測値の意味は環境で変わるため、CPU 数と主要ライブラリのバージョンを記録しておく。
    CSV のパーサは複数コアを使う実装があるので、CPU 数は効いてくる。
    """)
    return


@app.cell
def _(orjson, pd, pl, pyarrow):
    import os

    import dask

    pd.DataFrame([
        {"item": "CPU count", "value": str(os.cpu_count())},
        {"item": "pandas", "value": pd.__version__},
        {"item": "polars", "value": pl.__version__},
        {"item": "pyarrow", "value": pyarrow.__version__},
        {"item": "dask", "value": dask.__version__},
        {"item": "orjson", "value": orjson.__version__},
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 表データ：CSV と Parquet

    題材は 200,000 行 4 列（`carat` / `depth` / `table` / `price`）で、
    CSV と Parquet の両方で `data/others/data_io/` へ書き出す。

    読み込みで比べるのは次の 6 通り。

    | 手法 | 中身 |
    | --- | --- |
    | `pandas.read_csv` | 既定の C エンジン |
    | `pandas.read_csv(engine="pyarrow")` | Arrow のパーサ。複数コアを使う |
    | `pandas.read_parquet` | 列指向・型付き・圧縮済みの形式 |
    | `polars.read_csv` | Rust 実装。並列パース |
    | `polars.read_parquet` | 同上 |
    | `dask.dataframe.read_csv().compute()` | タスクグラフを組んで分割実行 |

    注目してほしいのは 2 点。

    1. **CSV をやめると速くなる。** CSV はテキストなので、読むたびに数値へ変換し直し、
       型を推定し直す必要がある。Parquet は型と列の境界を持っているのでその作業が要らない。
       ファイルサイズも小さくなるので、`size_MiB` 列と併せて見る。
    2. **同じ CSV でもパーサで変わる。** pandas の既定は単一コアの C 実装だが、
       `engine="pyarrow"` と polars は複数コアでパースする。

    dask はこの規模では pandas より遅い。タスクグラフの構築という固定コストが乗るためで、
    有利になるのはメモリに載らない大きさや、分割して並列処理できる場合である。
    """)
    return


@app.cell
def _(benchmark, data_dir, dd, np, pd, pl):
    rng_data = np.random.default_rng(0)
    TABLE_ROWS = 200_000
    csv_path = data_dir / "diamonds.csv"
    parquet_path = data_dir / "diamonds.parquet"
    table_frame = pd.DataFrame({
        "carat": rng_data.uniform(0.2, 2.5, size=TABLE_ROWS).round(3),
        "depth": rng_data.uniform(55.0, 70.0, size=TABLE_ROWS).round(3),
        "table": rng_data.uniform(50.0, 70.0, size=TABLE_ROWS).round(3),
        "price": rng_data.integers(300, 18000, size=TABLE_ROWS),
    })
    table_frame.to_csv(csv_path, index=False)
    table_frame.to_parquet(parquet_path)

    table_cases = [
        ("pandas.read_csv (c)", "csv", lambda: pd.read_csv(csv_path)),
        (
            "pandas.read_csv (pyarrow)",
            "csv",
            lambda: pd.read_csv(csv_path, engine="pyarrow"),
        ),
        ("pandas.read_parquet", "parquet", lambda: pd.read_parquet(parquet_path)),
        ("polars.read_csv", "csv", lambda: pl.read_csv(csv_path)),
        ("polars.read_parquet", "parquet", lambda: pl.read_parquet(parquet_path)),
        ("dask.read_csv", "csv", lambda: dd.read_csv(csv_path).compute()),
    ]
    table_rows = []
    for _label, _fmt, _reader in table_cases:
        _result = benchmark(_label, _reader, repeats=3, warmup_s=0.1)
        _path = csv_path if _fmt == "csv" else parquet_path
        table_rows.append({
            "task": _label,
            "format": _fmt,
            "elapsed_ms": _result["min_ms"],
            "size_MiB": _path.stat().st_size / 2**20,
        })
    _frame = pd.DataFrame(table_rows)
    _frame["speedup_vs_pandas_csv"] = _frame["elapsed_ms"].iloc[0] / _frame["elapsed_ms"]
    _frame.round(3)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## JSON：標準ライブラリと orjson

    5,000 要素の配列（各要素は `id` / `value` / `square`）を題材に、
    `json` と `orjson` を読み書きの両方で比べる。
    `orjson` は C 実装で、この規模でも確実に速い。

    こちらは 1 回だけの計測なので `timed_call` を使う。
    表データほど所要時間が長くないため、外乱の影響は相対的に大きい点に注意する。
    """)
    return


@app.cell
def _(data_dir, json, np, orjson, pd, timed_call):
    rng_json = np.random.default_rng(1)
    json_path = data_dir / "large-file.json"
    json_payload = [
        {"id": int(index), "value": float(value), "square": float(value**2)}
        for index, value in enumerate(rng_json.normal(size=5000))
    ]
    json_path.write_text(json.dumps(json_payload, ensure_ascii=False), encoding="utf-8")

    _, json_elapsed = timed_call(
        "json.loads", lambda: json.loads(json_path.read_text(encoding="utf-8"))
    )
    data_orjson, orjson_elapsed = timed_call(
        "orjson.loads", lambda: orjson.loads(json_path.read_bytes())
    )
    _, dump_elapsed = timed_call(
        "json.dumps",
        lambda: (data_dir / "large-file.standard.json").write_text(
            json.dumps(data_orjson), encoding="utf-8"
        ),
    )
    _, orjson_dump_elapsed = timed_call(
        "orjson.dumps",
        lambda: (data_dir / "large-file.orjson.json").write_bytes(orjson.dumps(data_orjson)),
    )
    pd.DataFrame([
        {"task": "json.loads", "elapsed_s": json_elapsed, "group": "read"},
        {"task": "orjson.loads", "elapsed_s": orjson_elapsed, "group": "read"},
        {"task": "json.dumps", "elapsed_s": dump_elapsed, "group": "write"},
        {"task": "orjson.dumps", "elapsed_s": orjson_dump_elapsed, "group": "write"},
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## まとめ

    実務で効く順に並べると次のようになる。

    1. **形式を変える。** CSV から Parquet へ移すのが最も効く。
       読み込みが速くなるうえ、ファイルサイズも小さくなる。
       型が保存されるので、読み込み後の型変換も要らなくなる。
    2. **パーサを変える。** CSV のままでも `engine="pyarrow"` や polars にすれば
       複数コアでパースできる。既存のコードを大きく変えずに済む。
    3. **JSON は `orjson` にする。** 標準ライブラリからの置き換えは
       `loads` / `dumps` の名前が同じなので容易である。

    逆に **dask はこの規模では効かない**。分散処理の枠組みは、
    メモリに載らない大きさになって初めて元が取れる。
    """)
    return


if __name__ == "__main__":
    app.run()
