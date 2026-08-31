import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import asyncio
    import atexit
    import concurrent.futures as cf
    import ctypes
    import hashlib
    import json
    import os
    import socket
    import statistics
    import subprocess
    import sys
    import time
    import warnings
    from functools import partial
    from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
    from multiprocessing import get_context
    from pathlib import Path
    from threading import Thread

    import dask.dataframe as dd
    import httpx
    import marimo as mo
    import numpy as np
    import orjson
    import pandas as pd
    import polars as pl
    import requests
    from numba import njit, prange
    from tqdm.contrib.concurrent import thread_map

    notebook_dir = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
    repo_root = notebook_dir.parents[1]
    data_dir = repo_root / "data" / "others" / "rapid_python"
    data_dir.mkdir(parents=True, exist_ok=True)
    warnings.filterwarnings(
        "ignore",
        message=r"resource_tracker: There appear to be \d+ leaked semaphore objects to clean up at shutdown",
        category=UserWarning,
    )
    warnings.filterwarnings(
        "ignore",
        category=UserWarning,
        module=r"multiprocessing\.resource_tracker",
    )

    return (
        SimpleHTTPRequestHandler,
        Thread,
        ThreadingHTTPServer,
        asyncio,
        atexit,
        cf,
        ctypes,
        data_dir,
        dd,
        get_context,
        hashlib,
        httpx,
        json,
        mo,
        njit,
        np,
        orjson,
        os,
        partial,
        pd,
        pl,
        prange,
        requests,
        socket,
        statistics,
        subprocess,
        sys,
        thread_map,
        time,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Python 高速化メモ

    Python のコードを速くする手立てを、JIT コンパイル・GPU・並列処理・データ入出力の
    4 つの軸で実測しながら比較する。

    このノートブックの計測はすべて 1 台のマシンで同一プロセス内で行う。
    絶対値は環境に強く依存するので、注目すべきは**手法どうしの比**と、
    どこで頭打ちになるかである。数値は下の「実行環境」セルの構成で得たものである。
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 計測の共通部品

    以降のセルで使う計測ヘルパーを 2 つ定義する。

    - `timed_call(label, func, *args)`: `func` を **1 回だけ**呼び、経過秒数を返す。
      コンパイル時間のように「初回」に意味がある計測に使う。
    - `benchmark(label, func, repeats=7, warmup_s=0.5)`: `warmup_s` 秒ぶん回し続けてから
      `repeats` 回計測し、**中央値と最小値**を返す。ウォームアップは JIT のコンパイルと、
      GPU のクロックを省電力状態から引き上げるためのもので、後者の効き方は
      GPU セクションの「計測の罠」で実測する。

    どちらも経過時間の計測に `time.perf_counter`（単調増加・高分解能）を使う。
    中央値と最小値を両方出すのは、外乱の混入を見分けるためである。
    両者が大きく離れていれば他のプロセスや電力制御の影響を疑う。
    """)
    return


@app.cell
def _(np, statistics, time):
    def timed_call(label: str, func, *args, **kwargs):
        start = time.perf_counter()
        value = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        print(f"[{label}] {elapsed:.3g} s")
        return value, elapsed

    def benchmark(label: str, func, repeats: int = 7, warmup_s: float = 0.5):
        # ウォームアップ。JIT のコンパイルを済ませるだけでなく、GPU のクロックを
        # 省電力状態から引き上げるために一定時間まわし続ける（下の「計測の罠」を参照）
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
        return {
            "case": label,
            "median_ms": median_s * 1e3,
            "min_ms": best_s * 1e3,
        }

    return benchmark, timed_call


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 実行環境

    計測値の意味は環境で変わるため、CPU 数と主要ライブラリのバージョンを記録しておく。
    GPU の情報はここでは取らず、GPU セクションの先頭で調べる。こうすると
    GPU を一度も触っていない状態のプロセス生成コストを測れるので、
    GPU を使った後との比較ができる（「fork のコスト」の節）。
    """)
    return


@app.cell
def _(os, pd):
    import cuda.core as ccore
    import cupy as cp
    import numba
    import numba_cuda
    import torch
    import triton
    from numba import cuda as nbcuda

    # ここでは import とバージョン取得だけを行い、CUDA の初期化はしない。
    # 初期化すると後段の fork が高くつくため、GPU セクションの先頭まで遅らせる。
    environment_rows = [
        {"item": "CPU count", "value": str(os.cpu_count())},
        {"item": "numba", "value": numba.__version__},
        {"item": "numba-cuda", "value": numba_cuda.__version__},
        {"item": "cupy", "value": cp.__version__},
        {"item": "triton", "value": triton.__version__},
        {"item": "torch", "value": torch.__version__},
        {"item": "cuda.core", "value": ccore.__version__},
    ]
    pd.DataFrame(environment_rows)
    return ccore, cp, nbcuda, torch, triton


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Numba による JIT コンパイル

    ### 題材：非線形な縮約

    JIT の効き方を見るために、1 次元配列に対する非線形な縮約
    $\sum_i (\sin^2 x_i + \sqrt{x_i})$ を使う。要素数は 100 万。

    この 1 つの題材で、次の 4 つをまとめて比べられる。

    - 素の Python ループ
    - NumPy によるベクトル化
    - Numba の **遅延コンパイル**（型指定なし）と**事前コンパイル**（シグネチャ指定）
    - Numba の **並列化**（`parallel=True`）

    `@njit` は Python の関数を LLVM 経由でネイティブコードへコンパイルする。
    型指定を書かない遅延コンパイルでは、最初の呼び出しで引数の型を見てから
    コンパイルするため、初回だけコンパイル時間が乗る。
    デコレータに `"float64(float64[:])"` のようにシグネチャを書く事前コンパイルでは、
    デコレータ評価の時点でコンパイルが終わるので、初回の呼び出しから速い。
    表の `lazy_njit_first` と `typed_njit_first` の差がそのコンパイル時間である。

    `*_first` の行はコンパイル時間を見るためのものなので 1 回だけ測る。
    それ以外の行は `benchmark` でウォームアップしてから複数回測った定常状態の値である。
    最初の 1 回だけを測ると、配列の初回確保や並列版のスレッド起動が混ざって
    実力より遅く出る。
    """)
    return


@app.cell
def _(benchmark, np, njit, pd, prange, timed_call):
    import math

    REDUCE_SIZE = 1_000_000
    reduce_input = np.linspace(0.1, 10.0, REDUCE_SIZE)

    def reduce_python(values):
        total = 0.0
        for value in values:
            total += math.sin(value) ** 2 + math.sqrt(value)
        return total

    def reduce_numpy(values):
        return float((np.sin(values) ** 2 + np.sqrt(values)).sum())

    @njit(cache=False)
    def reduce_lazy(values):
        total = 0.0
        for index in range(values.size):
            total += math.sin(values[index]) ** 2 + math.sqrt(values[index])
        return total

    @njit("float64(float64[:])", cache=False)
    def reduce_eager(values):
        total = 0.0
        for index in range(values.size):
            total += math.sin(values[index]) ** 2 + math.sqrt(values[index])
        return total

    @njit("float64(float64[:])", cache=False, parallel=True, fastmath=True)
    def reduce_parallel(values):
        total = 0.0
        for index in prange(values.size):
            total += math.sin(values[index]) ** 2 + math.sqrt(values[index])
        return total

    answer_py, elapsed_py = timed_call("python loop", reduce_python, reduce_input)
    answer_lazy, lazy_first = timed_call("lazy njit 1st (compile included)", reduce_lazy, reduce_input)
    answer_eager, eager_first = timed_call("typed njit 1st", reduce_eager, reduce_input)

    # 定常状態はウォームアップしてから複数回測る（1 回だけだと初回の確保やスレッド起動が乗る）
    steady = {
        row["case"]: row["min_ms"] / 1e3
        for row in [
            benchmark("python loop", lambda: reduce_python(reduce_input), repeats=3, warmup_s=0.1),
            benchmark("numpy vectorised", lambda: reduce_numpy(reduce_input), repeats=3, warmup_s=0.1),
            benchmark("lazy njit", lambda: reduce_lazy(reduce_input), repeats=3, warmup_s=0.1),
            benchmark("typed njit", lambda: reduce_eager(reduce_input), repeats=3, warmup_s=0.1),
            benchmark("njit parallel", lambda: reduce_parallel(reduce_input), repeats=3, warmup_s=0.1),
        ]
    }
    answer_np = reduce_numpy(reduce_input)
    answer_reduce_par = reduce_parallel(reduce_input)
    pd.DataFrame([
        {"case": case, "answer": answer, "elapsed_s": elapsed, "speedup_vs_python": steady["python loop"] / elapsed}
        for case, answer, elapsed in [
            ("python", answer_py, steady["python loop"]),
            ("numpy", answer_np, steady["numpy vectorised"]),
            ("lazy_njit_first", answer_lazy, lazy_first),
            ("lazy_njit_steady", answer_lazy, steady["lazy njit"]),
            ("typed_njit_first", answer_eager, eager_first),
            ("typed_njit_steady", answer_eager, steady["typed njit"]),
            ("njit_parallel", answer_reduce_par, steady["njit parallel"]),
        ]
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    読み方の要点は 3 つ。

    1. **`lazy_njit_first` だけが突出して遅い。** これがコンパイル時間で、
       2 回目以降は消える。事前コンパイルではデコレータの評価時にコンパイルが
       済んでいるため、1 回目から 2 回目と同じ速さになる。
    2. **NumPy より njit のほうが速い。** NumPy は `np.sin(values)`、`np.sqrt(values)` と
       段階ごとに中間配列を作り、そのたびにメモリを往復する。
       njit のループは 1 要素を読んだらそのまま計算して足し込むので、中間配列が要らない。
       **ベクトル化は速いが最速ではない**、という点は覚えておく価値がある。
    3. **`answer` 列が全手法で一致している。** 最適化で意味が変わっていない証拠になる。
       `fastmath=True` は浮動小数点の結合則の並べ替えを許すため、
       並列版の値は下位桁で他とわずかに異なりうる。
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 計測の罠：コンパイラが計算そのものを消す

    次は 2 重ループの総和を素の Python・`@njit`・`@njit(parallel=True, fastmath=True)` で比べる。
    ただし**ループ本体の選び方に注意がいる**。

    最初のセルはループ本体を `total += i - j` にしている。この総和は $i$ と $j$ の対称性から
    恒等的に 0 で、LLVM はループ全体を定数へ畳んでしまう。
    結果として `@njit` 版の所要時間は反復回数に依存しなくなり、
    見かけ上ありえない速度が出る。`4000 * 4000 = 1600` 万回の反復が数マイクロ秒、
    毎秒 1000 Giter を超える計算になる。1 サイクルに 1 反復こなせたとしても
    数 GHz の CPU では毎秒数 Giter が上限なので、**反復していない**ことは数字だけで分かる。
    これは高速化ではなく計算が消えたことを意味する。

    2 番目のセルはループ本体を `total += (i * j) % 7` に変えている。
    剰余が入ると閉じた形に畳めないので、実際にループが回った上での速度が測れる。
    ベンチマークを書くときは、**結果が入力に依存し、かつ閉じた形を持たない**本体を選ぶ。
    """)
    return


@app.cell
def _(njit, pd, prange, timed_call):
    def folded_sum(size: int):
        total = 0
        for i in range(size):
            for j in range(size):
                total += i - j
        return total

    @njit("int64(int64)", cache=False)
    def folded_sum_njit(size):
        total = 0
        for i in range(size):
            for j in range(size):
                total += i - j
        return total

    @njit("int64(int64)", cache=False, parallel=True, fastmath=True)
    def folded_sum_parallel(size):
        total = 0
        for i in prange(size):
            for j in range(size):
                total += i - j
        return total

    loop_size = 4000
    _, folded_python = timed_call("folded python", folded_sum, loop_size)
    _, folded_njit = timed_call("folded njit", folded_sum_njit, loop_size)
    _, folded_par = timed_call("folded parallel", folded_sum_parallel, loop_size)
    pd.DataFrame([
        {
            "case": case,
            "elapsed_s": elapsed,
            "Giter_per_s": loop_size**2 / elapsed / 1e9,
        }
        for case, elapsed in [
            ("python", folded_python),
            ("njit", folded_njit),
            ("njit parallel+fastmath", folded_par),
        ]
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    `njit` 列の `Giter_per_s` が現実のメモリ帯域・命令発行率から桁違いに外れていれば、
    畳み込みが起きた印である。次は畳み込めない本体で測り直す。
    """)
    return


@app.cell
def _(njit, pd, prange, timed_call):
    def mod_sum(size: int):
        total = 0
        for i in range(size):
            for j in range(size):
                total += (i * j) % 7
        return total

    @njit("int64(int64)", cache=False)
    def mod_sum_njit(size):
        total = 0
        for i in range(size):
            for j in range(size):
                total += (i * j) % 7
        return total

    @njit("int64(int64)", cache=False, parallel=True, fastmath=True)
    def mod_sum_parallel(size):
        total = 0
        for i in prange(size):
            for j in range(size):
                total += (i * j) % 7
        return total

    mod_size = 4000
    answer_python, elapsed_python = timed_call("mod python", mod_sum, mod_size)
    answer_njit, elapsed_njit = timed_call("mod njit", mod_sum_njit, mod_size)
    answer_par, elapsed_par = timed_call("mod parallel", mod_sum_parallel, mod_size)
    pd.DataFrame([
        {
            "case": case,
            "answer": answer,
            "elapsed_s": elapsed,
            "Giter_per_s": mod_size**2 / elapsed / 1e9,
            "speedup_vs_python": elapsed_python / elapsed,
        }
        for case, answer, elapsed in [
            ("python", answer_python, elapsed_python),
            ("njit", answer_njit, elapsed_njit),
            ("njit parallel+fastmath", answer_par, elapsed_par),
        ]
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    こちらでも `answer` 列は 3 手法で一致する。
    整数演算なので `fastmath=True` は効かず、効いているのは `prange` による並列化だけである。
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 並列処理

    ### GIL とスレッド・プロセスの使い分け

    CPython のバイトコードは GIL（グローバルインタプリタロック）に守られて実行されるため、
    **純粋な Python コードはスレッドを増やしても速くならない**。
    プロセスを分ければインタプリタごと分かれるので GIL を回避できる。

    題材は前半でも使った二重ループ `(i * j) % 7` の総和（素の Python 版）を 4 本。
    これを逐次・スレッド 4 本・プロセス 4 個で走らせて比べる。

    プロセスの生成には `multiprocessing` の `fork` を使う。ここで
    **`Process` と `ProcessPoolExecutor` の違いに注意がいる**。

    - `fork` の `Process(target=...)` は、親のメモリをそのまま子へ引き継ぐので
      関数を pickle しない。marimo のセル内で定義した関数もそのまま渡せる。
    - `ProcessPoolExecutor` は start method に関わらず、投入する callable と引数を
      キュー経由で pickle する。したがって**セル内で定義した関数は fork でも渡せない**。

    次のセルが `Process` を直接使っているのはこのためである。fork は Linux でのみ使える。
    """)
    return


@app.cell
def _(cf, get_context, pd, time):
    def gil_bound_work(size):
        total = 0
        for i in range(size):
            for j in range(size):
                total += (i * j) % 7
        return total

    GIL_TASKS = [1600] * 4

    def measure(func):
        start = time.perf_counter()
        func()
        return time.perf_counter() - start

    def run_sequential():
        return [gil_bound_work(size) for size in GIL_TASKS]

    def run_threads():
        with cf.ThreadPoolExecutor(max_workers=4) as executor:
            return list(executor.map(gil_bound_work, GIL_TASKS))

    def run_processes():
        context = get_context("fork")
        processes = [
            context.Process(target=gil_bound_work, args=(size,)) for size in GIL_TASKS
        ]
        for process in processes:
            process.start()
        for process in processes:
            process.join()
            process.close()

    gil_sequential = measure(run_sequential)
    gil_threads = measure(run_threads)
    gil_processes = measure(run_processes)
    pd.DataFrame([
        {"strategy": "sequential", "elapsed_s": gil_sequential, "speedup": 1.0},
        {
            "strategy": "ThreadPoolExecutor(4)",
            "elapsed_s": gil_threads,
            "speedup": gil_sequential / gil_threads,
        },
        {
            "strategy": "fork Process x4",
            "elapsed_s": gil_processes,
            "speedup": gil_sequential / gil_processes,
        },
    ]).round(3)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    スレッドはほとんど速くならず、プロセスだけが効く。これが GIL の姿である。

    ただしプロセス 4 個でも 4 倍には届かない。プロセスの生成自体にコストがかかるためで、
    その内訳は下の「fork のコスト」の節で測る。

    ### 例外：GIL を解放するライブラリ

    「CPU バウンドならスレッドは無駄」は**純 Python のコードに限った話**である。
    C で書かれた拡張は、計算に入る前に GIL を解放できる。
    その間 Python 側は別のスレッドを動かせるので、スレッドでも並列化できる。

    `hashlib.pbkdf2_hmac`（OpenSSL 実装）がその例で、鍵導出の計算中は GIL を手放す。
    次のセルでは反復回数を 200,000〜300,000 と変えた 6 個のタスクを、
    スレッドプールとプロセスプールの両方へ投げて比べる。
    **この題材はスレッドでもプロセスでも並列化できる。**
    NumPy・SciPy・PyTorch の重い演算も同じ性質を持つものが多い。

    プロセス側では `max_workers` を 1, 2, 4, 8 と振る。読み方は 2 点。

    - **タスク 1 個あたりの所要時間は、プロセス生成のコストより十分大きくする必要がある。**
      反復回数を 1 桁小さくすると 1 タスクが数十 ms になり、プロセスを起こす時間の方が
      大きくなってワーカーを増やしても速くならない。
    - 6 個しかタスクが無いので頭打ちになる。8 まで増やすと 4 のときより遅くなることもある。
      その内訳は下の「fork のコスト」の節で測る。

    ここで `ProcessPoolExecutor` を使えるのは、worker が
    `partial(hashlib.pbkdf2_hmac, ...)` という C 関数の partial で pickle 可能だからである。
    `fork` を選んでいる理由は pickle ではなく**起動コストの低さ**で、`spawn` でも動作する。

    **このセルは GPU セクションより前に置いてある。** プロセス生成のコストは
    親プロセスの大きさに影響されるため、GPU の初期化を挟む前後で比較できるようにしている。
    ここでは比較の基準として、仕事をしない空のプールの起動・終了時間も測っておく。
    """)
    return


@app.cell
def _(cf, get_context, hashlib, os, partial, pd, time):
    cpu_tasks = [200_000, 220_000, 240_000, 260_000, 280_000, 300_000]
    digest_worker = partial(hashlib.pbkdf2_hmac, "sha256", b"marimo", b"parallel", dklen=32)

    _start = time.perf_counter()
    [digest_worker(task) for task in cpu_tasks]
    digest_sequential = time.perf_counter() - _start

    _start = time.perf_counter()
    with cf.ThreadPoolExecutor(max_workers=6) as _thread_pool:
        list(_thread_pool.map(digest_worker, cpu_tasks))
    digest_threads = time.perf_counter() - _start
    print(
        f"pbkdf2: sequential {digest_sequential * 1e3:.0f} ms"
        f" / threads(6) {digest_threads * 1e3:.0f} ms"
        f" -> {digest_sequential / digest_threads:.2f}x (GIL を解放している証拠)"
    )

    worker_rows = []
    for _process_workers in [1, 2, 4, 8]:
        _start = time.perf_counter()
        with cf.ProcessPoolExecutor(
            max_workers=_process_workers, mp_context=get_context("fork")
        ) as _process_executor:
            list(_process_executor.map(digest_worker, cpu_tasks))
        worker_rows.append(
            {"max_workers": _process_workers, "elapsed_s": time.perf_counter() - _start}
        )
    print(f"CPU count is {os.cpu_count()}, tasks = {len(cpu_tasks)}")

    def empty_pool_seconds(max_workers=8):
        start = time.perf_counter()
        with cf.ProcessPoolExecutor(
            max_workers=max_workers, mp_context=get_context("fork")
        ) as executor:
            list(executor.map(int, range(max_workers)))
        return time.perf_counter() - start

    fork_cost_before_cuda = empty_pool_seconds()
    print(f"empty pool of 8 workers: {fork_cost_before_cuda * 1e3:.0f} ms")
    _frame = pd.DataFrame(worker_rows)
    _frame["speedup_vs_sequential"] = digest_sequential / _frame["elapsed_s"]
    _frame
    return cpu_tasks, digest_worker, empty_pool_seconds, fork_cost_before_cuda, worker_rows


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### スレッドプール：I/O バウンドな仕事

    I/O 待ちの間は GIL が解放されるので、こちらはスレッドで並列化できる。

    外部サイトへ依存しないよう、ノートブック内でローカルの HTTP サーバを立て、
    そこに置いた 6 個の JSON を取得する。次のセルはサーバの起動までを行う。
    ポートは 0 番を bind して OS に空きポートを選ばせ、
    ハンドラはアクセスログを抑止している。プロセス終了時に確実に落とすため
    `atexit` にも後始末を登録する。

    ハンドラには **50 ms の人為的な待ち**を入れてある。同一ホストのファイル配信は
    応答が速すぎて 1 件あたり数 ms しかかからず、そのままでは
    スレッドプールの生成コストの方が大きくなって並列化の効果が見えないためである。
    ネットワーク越しの API 呼び出しではこの程度の待ちが普通に発生する。
    """)
    return


@app.cell
def _(
    SimpleHTTPRequestHandler,
    Thread,
    ThreadingHTTPServer,
    atexit,
    data_dir,
    json,
    np,
    partial,
    requests,
    socket,
    time,
):
    RESPONSE_DELAY_S = 0.05
    http_dir = data_dir / "http_payloads"
    http_dir.mkdir(exist_ok=True)
    rng_payload = np.random.default_rng(42)
    for _index in range(6):
        _payload = {"id": _index, "values": rng_payload.normal(size=8).round(4).tolist()}
        (http_dir / f"payload_{_index}.json").write_text(json.dumps(_payload), encoding="utf-8")

    class QuietHandler(SimpleHTTPRequestHandler):
        def log_message(self, format, *args):
            return

        def do_GET(self):
            # 実在のサービス並みの待ちを模す。これが無いと応答が速すぎて
            # スレッドプールの効果よりプールの生成コストの方が大きくなる
            time.sleep(RESPONSE_DELAY_S)
            super().do_GET()

    with socket.socket() as _sock:
        _sock.bind(("127.0.0.1", 0))
        port = _sock.getsockname()[1]
    http_server = ThreadingHTTPServer(
        ("127.0.0.1", port), partial(QuietHandler, directory=str(http_dir))
    )
    server_thread = Thread(target=http_server.serve_forever, daemon=True)
    server_thread.start()
    thread_urls = [f"http://127.0.0.1:{port}/payload_{index}.json" for index in range(6)]
    requests.get(thread_urls[0], timeout=5).raise_for_status()

    server_state = {"closed": False}

    def cleanup_server():
        if server_state["closed"]:
            return
        http_server.shutdown()
        http_server.server_close()
        server_thread.join(timeout=1)
        server_state["closed"] = True

    atexit.register(cleanup_server)
    print(f"serving {len(thread_urls)} payloads on 127.0.0.1:{port} "
          f"with {RESPONSE_DELAY_S * 1e3:.0f} ms artificial delay")
    return cleanup_server, thread_urls


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    `max_workers` を 1, 2, 4, 8 と変えて 6 件の取得にかかる時間を測る。

    1 件あたり 50 ms の待ちがあるので、逐次なら約 300 ms かかる。
    スレッドを増やすと待ち時間が重なり、「6 件をワーカー数で割った切り上げ回数」×50 ms へ
    近づく。ワーカーが 6 を超えても、タスクが 6 個しかないので改善は止まる。

    純 Python の CPU 計算では効かなかったスレッドが I/O ではそのまま効くのは、
    `requests` が応答を待つ間 GIL を手放すからである。
    """)
    return


@app.cell
def _(cf, pd, requests, thread_urls, timed_call):
    def fetch_status(url: str):
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        return {
            "url": url.rsplit("/", 1)[-1],
            "status_code": response.status_code,
            "bytes": len(response.content),
        }

    def fetch_all(urls, max_workers: int):
        with cf.ThreadPoolExecutor(max_workers=max_workers) as _thread_executor:
            return list(_thread_executor.map(fetch_status, urls))

    timing_rows = []
    for _thread_workers in [1, 2, 4, 8]:
        _statuses, _elapsed = timed_call(
            f"thread pool max_workers={_thread_workers}",
            fetch_all,
            thread_urls,
            _thread_workers,
        )
        timing_rows.append({"max_workers": _thread_workers, "elapsed_s": _elapsed})
    _frame = pd.DataFrame(timing_rows)
    _frame["speedup_vs_1"] = _frame["elapsed_s"].iloc[0] / _frame["elapsed_s"]
    _frame
    return (fetch_status,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### asyncio：スレッドを使わない I/O 並列

    I/O 並列にはもう 1 つ主要な選択肢がある。asyncio である。
    スレッドを増やす代わりに、**1 本のスレッドの上でイベントループが
    待ち状態のタスクを切り替える**。OS スレッドを作らないので、
    同時接続数が数千規模になっても破綻しにくい。

    比較を公平にするため、両方とも `httpx` を使う。
    `httpx` は同期 API と非同期 API がほぼ同じ形なので、
    差が並列化の仕組みだけになる。

    - 同期 + `ThreadPoolExecutor`: `httpx.Client` を 8 スレッドで使う
    - 非同期 + `asyncio.gather`: `httpx.AsyncClient` で 6 件を同時に投げる

    marimo のセルでは `await` をそのまま書ける（セルが `async def` になる）。
    逆に `asyncio.run()` は使えない。marimo 自身がイベントループの中で
    セルを実行しているため、`asyncio.run() cannot be called from a running event loop`
    になる。同じ理由で Jupyter でも `asyncio.run()` は使えない。
    """)
    return


@app.cell
async def _(asyncio, cf, httpx, pd, thread_urls, time):
    def httpx_sync_fetch(client, url):
        response = client.get(url)
        response.raise_for_status()
        return len(response.content)

    async def httpx_async_fetch(client, url):
        response = await client.get(url)
        response.raise_for_status()
        return len(response.content)

    def fetch_with_threads():
        with httpx.Client(timeout=5.0) as client:
            with cf.ThreadPoolExecutor(max_workers=8) as executor:
                return list(executor.map(lambda url: httpx_sync_fetch(client, url), thread_urls))

    async def fetch_with_asyncio():
        async with httpx.AsyncClient(timeout=5.0) as client:
            return await asyncio.gather(
                *[httpx_async_fetch(client, url) for url in thread_urls]
            )

    def measure_sync(func, repeats=3):
        func()  # ウォームアップ
        samples = []
        for _ in range(repeats):
            start = time.perf_counter()
            func()
            samples.append(time.perf_counter() - start)
        return min(samples)

    async def measure_async(func, repeats=3):
        await func()  # ウォームアップ
        samples = []
        for _ in range(repeats):
            start = time.perf_counter()
            await func()
            samples.append(time.perf_counter() - start)
        return min(samples)

    sync_elapsed = measure_sync(fetch_with_threads)
    async_elapsed = await measure_async(fetch_with_asyncio)
    _sync_sizes = fetch_with_threads()
    _async_sizes = await fetch_with_asyncio()

    io_rows = [
        {"strategy": "httpx sync + ThreadPoolExecutor(8)", "elapsed_s": sync_elapsed},
        {"strategy": "httpx async + asyncio.gather", "elapsed_s": async_elapsed},
    ]
    print(f"bytes match: {list(_sync_sizes) == list(_async_sizes)}")
    _frame = pd.DataFrame(io_rows)
    _frame["speedup_vs_threads"] = sync_elapsed / _frame["elapsed_s"]
    _frame
    return (io_rows,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    どちらも 6 件を同時に投げるので、逐次なら 300 ms かかるところが大きく下がる。
    ただし 1 件ぶんの待ち（50 ms）ちょうどにはならない。
    クライアントの生成、接続の確立、スレッドプールやイベントループの立ち上げが
    上乗せされるためである。
    **この規模では両者に大きな差は出ない。** asyncio が効いてくるのは、
    同時接続が数百〜数千に増えてスレッドの生成とコンテキストスイッチが
    無視できなくなってからである。

    選択の基準は速度ではなく、コードの形になる。
    asyncio は呼び出し側まで `async` に染まるため、
    既存の同期コードへ後から入れるのは難しい。
    スレッドプールは既存の関数をそのまま渡せる。

    どちらもウォームアップしてから 3 回測り、最小値を採っている。
    1 回だけ測るとクライアントの生成やスレッドプールの起動が乗り、
    先に測ったほうが不利になる。

    `tqdm.contrib.concurrent.thread_map` は `ThreadPoolExecutor` と進捗バーを
    ひとまとめにした薄いラッパで、`map` と同じ感覚で書ける。
    静的 HTML には進捗バーが残ってしまうため `disable=True` にしている。
    次のセルは取得結果そのものを表示し、あわせて HTTP サーバを停止する。
    """)
    return


@app.cell
def _(cleanup_server, fetch_status, io_rows, pd, thread_map, thread_urls):
    _ = io_rows  # asyncio の計測が終わってからサーバを止める
    thread_map_rows = thread_map(fetch_status, thread_urls, max_workers=8, disable=True)
    cleanup_server()
    pd.DataFrame(thread_map_rows)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### プロセス間でデータを渡す

    プロセスはメモリ空間を共有しないので、結果を受け取るには明示的な仕組みがいる。
    代表的な 2 つを示す。

    - **共有メモリ** (`Value` / `Array`): 固定長の値・配列を共有メモリ上に確保する。
      `lock=False` はロックを付けない指定で、書き込みが競合しない使い方に限られる。
      子プロセスが書き換えた結果を親がそのまま読める。
    - **パイプ** (`Pipe`): 片方向（`duplex=False`）の接続を作り、
      子が `send` した Python オブジェクトを親が `recv` で受け取る。
      pickle 経由なので任意のオブジェクトを渡せるが、その分のコストがかかる。

    パイプでは**親側が送信端を閉じる**ことが重要である。閉じ忘れると
    送信端が開いたままになり、受信側が終端を検出できない。
    """)
    return


@app.cell
def _(get_context, pd):
    fork_context = get_context("fork")

    def shared_memory_target(number, array):
        number.value = 3.1415927
        for index in range(len(array)):
            array[index] = -array[index]

    shared_number = fork_context.Value("d", 0.0, lock=False)
    shared_array = fork_context.Array("i", range(10), lock=False)
    _process = fork_context.Process(target=shared_memory_target, args=(shared_number, shared_array))
    _process.start()
    _process.join()
    _process.close()
    pd.DataFrame([
        {"shared_number": shared_number.value, "shared_array": list(shared_array)}
    ])
    return (fork_context,)


@app.cell
def _(fork_context, pd):
    def pipe_target(index, send_end):
        send_end.send({"index": index, "negated": -index})
        send_end.close()

    _endpoints = []
    _processes = []
    for _pipe_index in range(4):
        _recv_end, _send_end = fork_context.Pipe(duplex=False)
        _pipe_process = fork_context.Process(target=pipe_target, args=(_pipe_index, _send_end))
        _pipe_process.start()
        _send_end.close()  # 親側の送信端は閉じる
        _endpoints.append(_recv_end)
        _processes.append(_pipe_process)
    pipe_messages = [_recv_end.recv() for _recv_end in _endpoints]
    for _recv_end in _endpoints:
        _recv_end.close()
    for _pipe_process in _processes:
        _pipe_process.join()
        _pipe_process.close()
    pd.DataFrame(pipe_messages)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## GPU で計算する 6 つの方法

    ### 比較の題材と計測の約束

    題材はもっとも単純な要素ごとの加算 `c[i] = a[i] + b[i]` で、
    要素数は $2^{24}$（float32 で 1 配列あたり 64 MiB）。
    この演算は 1 要素あたり浮動小数点演算 1 回に対して 12 バイトの読み書きが発生するため、
    完全に**メモリ帯域律速**である。したがって、どの方式で書いても
    正しく書けている限り速度はほぼ同じになるはずで、
    それが確認できるかどうかが比較の目的である。

    比較するのは次の 6 方式。

    | 方式 | カーネルの記述言語 | 位置づけ |
    | --- | --- | --- |
    | numba-cuda | Python | Python のサブセットをそのまま CUDA カーネルへ |
    | CuPy 要素演算 | 書かない | NumPy 互換 API。裏で融合カーネルを生成 |
    | CuPy RawKernel | CUDA C++ | C++ の文字列を実行時コンパイル |
    | PyTorch | 書かない | 実際に最も広く使われる経路 |
    | Triton | Python (DSL) | ブロック単位で書く。タイル分割は自動 |
    | cuda-python | CUDA C++ | NVIDIA 公式バインディング。最も低水準 |

    計測は次の 2 通りを分ける。

    - **kernel**: 入力が既に GPU 上にある状態でカーネルだけを実行し、同期して測る。
    - **+transfer**: ホストからの転送・カーネル実行・ホストへの回収をすべて含めて測る。

    GPU の実行は非同期なので、**同期（`synchronize` / `stream.sync()`）を挟まないと
    計測値はカーネルの投入時間になってしまう**。以降のすべての計測で同期を入れている。
    さらに、次節で実測するとおり GPU のクロックは待たされると落ちるため、
    どの計測もウォームアップを済ませてから行う。

    最初のセルで GPU の有無を判定する。判定をノートブックの先頭ではなくここへ置くのは、
    GPU を一度も触っていない状態と触った後とでプロセス生成のコストを比べるためである
    （結果は「fork のコスト」の節で見る）。
    GPU が使えない環境では、以降の計測は行わず欠測として表示する。
    """)
    return


@app.cell
def _(nbcuda, pd, torch):
    # ノートブックが GPU に触れるのはここが最初
    gpu_ready = nbcuda.is_available() and torch.cuda.is_available()
    if gpu_ready:
        # numba-cuda 0.30 以降は str、numba 内蔵の CUDA ターゲットは bytes を返す
        _raw_name = nbcuda.get_current_device().name
        gpu_name = _raw_name.decode() if isinstance(_raw_name, bytes) else str(_raw_name)
    else:
        gpu_name = "N/A"
    pd.DataFrame([{"item": "GPU", "value": gpu_name}, {"item": "gpu_ready", "value": str(gpu_ready)}])
    return (gpu_ready,)


@app.cell
def _(np):
    VECTOR_SIZE = 1 << 24
    THREADS_PER_BLOCK = 256
    BLOCKS_PER_GRID = (VECTOR_SIZE + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK
    left_host = np.ones(VECTOR_SIZE, dtype=np.float32)
    right_host = np.full(VECTOR_SIZE, 2.0, dtype=np.float32)
    # 読み 2 本 + 書き 1 本。帯域の実効値を出すために使う
    MOVED_GIGABYTES = 3 * left_host.nbytes / 1e9

    CUDA_SOURCE = r"""
    extern "C" __global__
    void add_kernel(const float* a, const float* b, float* c, int n) {
        int i = blockDim.x * blockIdx.x + threadIdx.x;
        if (i < n) c[i] = a[i] + b[i];
    }
    """
    print(f"{VECTOR_SIZE=}, 1 配列あたり {left_host.nbytes / 2**20:.0f} MiB")
    return (
        BLOCKS_PER_GRID,
        CUDA_SOURCE,
        MOVED_GIGABYTES,
        THREADS_PER_BLOCK,
        VECTOR_SIZE,
        left_host,
        right_host,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 計測の罠：GPU のクロックは待たされると落ちる

    GPU には電力管理があり、仕事が来ない状態が続くとクロックを下げた省電力状態へ移る。
    このノートブックは GPU セクションの手前で 10 秒以上 CPU の計測をしているので、
    そのまま測ると**最初に測った方式だけが極端に遅く出る**。

    次のセルで、その落ち込みを実際に測る。手順は 3 段階。

    1. カーネルを 1 回実行してコンパイルを済ませる。
    2. `time.sleep` で GPU を遊ばせ、省電力状態へ落とす。
    3. 落ちた直後（cold）と、しばらく回して復帰させた後（warm）で同じカーネルを測る。

    `nvidia-smi` から取れる性能状態（`pstate`）とメモリクロックも併記する。
    `P8` が最も低い省電力状態、`P0` / `P2` が高負荷時の状態である。

    これが `benchmark` の `warmup_s` を設けている理由で、以降の計測はすべて
    ウォームアップ後の値である。**GPU のマイクロベンチマークでは、
    計測前に数百 ms 以上まわしてクロックを上げておく必要がある。**
    """)
    return


@app.cell
def _(
    BLOCKS_PER_GRID,
    THREADS_PER_BLOCK,
    gpu_ready,
    left_host,
    nbcuda,
    np,
    pd,
    right_host,
    statistics,
    subprocess,
    time,
):
    GPU_IDLE_SECONDS = 15.0

    def gpu_power_state():
        try:
            completed = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=pstate,clocks.mem,power.draw",
                    "--format=csv,noheader",
                ],
                capture_output=True,
                text=True,
                timeout=10,
                check=True,
            )
            return completed.stdout.strip().splitlines()[0]
        except (OSError, subprocess.SubprocessError, IndexError):
            return "N/A"

    clock_rows = []
    if not gpu_ready:
        gpu_warm = False
        clock_rows.append({"phase": "gpu unavailable", "nvidia_smi": "N/A", "kernel_ms": np.nan})
    else:
        @nbcuda.jit
        def clock_probe_add(a, b, c):
            index = nbcuda.grid(1)
            if index < c.size:
                c[index] = a[index] + b[index]

        probe_left = nbcuda.to_device(left_host)
        probe_right = nbcuda.to_device(right_host)
        probe_out = nbcuda.device_array_like(left_host)

        def probe_once():
            clock_probe_add[BLOCKS_PER_GRID, THREADS_PER_BLOCK](probe_left, probe_right, probe_out)
            nbcuda.synchronize()

        def probe_median(repeats=5):
            samples = []
            for _ in range(repeats):
                start = time.perf_counter()
                probe_once()
                samples.append(time.perf_counter() - start)
            return statistics.median(samples) * 1e3

        probe_once()  # ここでコンパイルを済ませる
        time.sleep(GPU_IDLE_SECONDS)  # GPU を遊ばせて省電力状態へ落とす
        clock_rows.append(
            {"phase": f"after {GPU_IDLE_SECONDS:.0f}s idle", "nvidia_smi": gpu_power_state(),
             "kernel_ms": np.nan}
        )
        clock_rows.append(
            {"phase": "cold kernel", "nvidia_smi": gpu_power_state(), "kernel_ms": probe_median()}
        )
        _warm_deadline = time.perf_counter() + 2.0
        while time.perf_counter() < _warm_deadline:
            probe_once()
        clock_rows.append(
            {"phase": "warm kernel", "nvidia_smi": gpu_power_state(), "kernel_ms": probe_median()}
        )
        gpu_warm = True
    pd.DataFrame(clock_rows)
    return (gpu_warm,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 基準：NumPy による CPU 実行

    まず比較の基準として、同じ加算を NumPy で CPU 実行する。
    """)
    return


@app.cell
def _(benchmark, left_host, right_host):
    cpu_row = benchmark("numpy (CPU)", lambda: left_host + right_host)
    cpu_row
    return (cpu_row,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### numba-cuda

    `@cuda.jit` は Python の関数を CUDA カーネルへコンパイルする。
    `cuda.grid(1)` が「ブロック番号 × ブロックサイズ + スレッド番号」を返すので、
    それを配列の添字として使い、範囲外を `if` で弾く。
    起動は `kernel[ブロック数, ブロックあたりスレッド数](引数...)` と書く。

    **numba の内蔵 CUDA ターゲットは非推奨になり、開発は NVIDIA が管理する
    `numba-cuda` パッケージへ移った。** `numba-cuda` を入れると `from numba import cuda`
    の解決先が入れ替わり、`numba.cuda.__file__` は `numba_cuda/numba/cuda/__init__.py` を指す。
    書き方は変わらないが、新しい CUDA ツールキットへの追随はこちらでしか行われない。
    実際、numba 内蔵の実装は新しめの CUDA と組み合わせると PTX のバージョン不整合
    （`CUDA_ERROR_UNSUPPORTED_PTX_VERSION`）で動かないことがある。

    転送込みの計測では、毎回 `to_device` で送り直し `copy_to_host` で回収する。
    """)
    return


@app.cell
def _(
    BLOCKS_PER_GRID,
    THREADS_PER_BLOCK,
    benchmark,
    gpu_ready,
    gpu_warm,
    left_host,
    nbcuda,
    np,
    right_host,
):
    numba_rows = []
    # gpu_warm を条件に含めることで、クロックを上げるセルの後に実行されるようにする
    if not (gpu_ready and gpu_warm):
        numba_rows.append({"case": "numba-cuda", "median_ms": np.nan, "min_ms": np.nan})
        numba_checksum = None
    else:
        @nbcuda.jit
        def numba_add(a, b, c):
            index = nbcuda.grid(1)
            if index < c.size:
                c[index] = a[index] + b[index]

        numba_left = nbcuda.to_device(left_host)
        numba_right = nbcuda.to_device(right_host)
        numba_out = nbcuda.device_array_like(left_host)

        def numba_kernel_only():
            numba_add[BLOCKS_PER_GRID, THREADS_PER_BLOCK](numba_left, numba_right, numba_out)
            nbcuda.synchronize()

        def numba_with_transfer():
            a = nbcuda.to_device(left_host)
            b = nbcuda.to_device(right_host)
            c = nbcuda.device_array_like(left_host)
            numba_add[BLOCKS_PER_GRID, THREADS_PER_BLOCK](a, b, c)
            return c.copy_to_host()

        numba_rows.append(benchmark("numba-cuda kernel", numba_kernel_only))
        numba_rows.append(benchmark("numba-cuda +transfer", numba_with_transfer))
        numba_checksum = numba_out.copy_to_host()[:3]
    print(f"{numba_checksum=}")
    return (numba_rows,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### CuPy：要素演算と RawKernel

    CuPy は NumPy 互換の API を GPU 上で提供する。`a + b` と書くだけで
    要素ごとの加算カーネルが生成・キャッシュされるため、カーネルを 1 行も書かなくてよい。

    より細かい制御が要るときは `cp.RawKernel` に CUDA C++ の文字列を渡す。
    ここでは `CUDA_SOURCE` に定義した `add_kernel` をそのまま使う。
    同じ C++ ソースを後段の cuda-python でも使うので、**同一のカーネルを
    2 つの異なるホスト API から起動している**ことになる。

    `RawKernel` の呼び出し規約は `kernel(grid, block, args)` で、
    `grid` と `block` はタプル、`args` は引数のタプルである。
    スカラ引数は `np.int32(...)` のように型を明示しないと C 側の `int` と食い違う。
    """)
    return


@app.cell
def _(
    BLOCKS_PER_GRID,
    CUDA_SOURCE,
    THREADS_PER_BLOCK,
    VECTOR_SIZE,
    benchmark,
    cp,
    gpu_ready,
    gpu_warm,
    left_host,
    np,
    right_host,
):
    cupy_rows = []
    if not (gpu_ready and gpu_warm):
        cupy_rows.append({"case": "cupy", "median_ms": np.nan, "min_ms": np.nan})
        cupy_checksum = None
    else:
        cupy_left = cp.asarray(left_host)
        cupy_right = cp.asarray(right_host)
        cupy_out = cp.empty_like(cupy_left)
        cupy_raw_kernel = cp.RawKernel(CUDA_SOURCE, "add_kernel")

        def cupy_elementwise():
            result = cupy_left + cupy_right
            cp.cuda.Stream.null.synchronize()
            return result

        def cupy_raw():
            cupy_raw_kernel(
                (BLOCKS_PER_GRID,),
                (THREADS_PER_BLOCK,),
                (cupy_left, cupy_right, cupy_out, np.int32(VECTOR_SIZE)),
            )
            cp.cuda.Stream.null.synchronize()

        def cupy_with_transfer():
            a = cp.asarray(left_host)
            b = cp.asarray(right_host)
            return cp.asnumpy(a + b)

        cupy_rows.append(benchmark("cupy elementwise", cupy_elementwise))
        cupy_rows.append(benchmark("cupy RawKernel", cupy_raw))
        cupy_rows.append(benchmark("cupy +transfer", cupy_with_transfer))
        cupy_checksum = cp.asnumpy(cupy_out[:3])
    print(f"{cupy_checksum=}")
    return (cupy_rows,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### PyTorch のテンソル演算

    Python から GPU を使う経路として実際に最も広く使われているのは PyTorch である。
    `torch.add(a, b, out=c)` と書くだけで、CuPy と同じように要素ごとの加算カーネルが走る。
    カーネルを書かずに済み、既存の深層学習のコードとそのまま噛み合う。

    ここでは `out=` を指定して出力先を使い回している。指定しないと呼び出しのたびに
    テンソルを確保するので、確保の時間まで測ってしまう。
    """)
    return


@app.cell
def _(benchmark, gpu_ready, gpu_warm, left_host, np, right_host, torch):
    torch_rows = []
    if not (gpu_ready and gpu_warm):
        torch_rows.append({"case": "torch", "median_ms": np.nan, "min_ms": np.nan})
        torch_checksum = None
    else:
        torch_left = torch.from_numpy(left_host).cuda()
        torch_right = torch.from_numpy(right_host).cuda()
        torch_out = torch.empty_like(torch_left)

        def torch_kernel_only():
            torch.add(torch_left, torch_right, out=torch_out)
            torch.cuda.synchronize()

        def torch_with_transfer():
            a = torch.from_numpy(left_host).cuda()
            b = torch.from_numpy(right_host).cuda()
            return (a + b).cpu().numpy()

        torch_rows.append(benchmark("torch kernel", torch_kernel_only))
        torch_rows.append(benchmark("torch +transfer", torch_with_transfer))
        torch_checksum = torch_out[:3].cpu().numpy()
    print(f"{torch_checksum=}")
    return (torch_rows,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Triton

    Triton は GPU カーネル用の Python DSL である。numba-cuda や CUDA C++ が
    **1 スレッドの処理**を書くのに対し、Triton は **1 ブロックが担当するタイル**を書く。
    `tl.arange(0, BLOCK)` でブロック内のオフセット列を作り、
    `tl.load` / `tl.store` でまとめて読み書きする。スレッドへの割り当てとベクトル化は
    コンパイラが決める。

    引数のポインタには PyTorch のテンソルをそのまま渡せる。
    `BLOCK` は `tl.constexpr` なのでコンパイル時定数として扱われ、
    値ごとに別のカーネルがコンパイルされる。

    Triton は `torch` の依存として入るが、単独のパッケージとしても宣言している。
    """)
    return


@app.cell
def _(
    THREADS_PER_BLOCK,
    VECTOR_SIZE,
    benchmark,
    gpu_ready,
    gpu_warm,
    left_host,
    np,
    right_host,
    torch,
    triton,
):
    import triton.language as tl

    triton_rows = []
    if not (gpu_ready and gpu_warm):
        triton_rows.append({"case": "triton", "median_ms": np.nan, "min_ms": np.nan})
        triton_checksum = None
    else:
        @triton.jit
        def triton_add(a_ptr, b_ptr, c_ptr, n, BLOCK: tl.constexpr):
            offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
            mask = offsets < n
            a = tl.load(a_ptr + offsets, mask=mask)
            b = tl.load(b_ptr + offsets, mask=mask)
            tl.store(c_ptr + offsets, a + b, mask=mask)

        triton_left = torch.from_numpy(left_host).cuda()
        triton_right = torch.from_numpy(right_host).cuda()
        triton_out = torch.empty_like(triton_left)
        triton_grid = (triton.cdiv(VECTOR_SIZE, THREADS_PER_BLOCK),)

        def triton_kernel_only():
            triton_add[triton_grid](
                triton_left, triton_right, triton_out, VECTOR_SIZE, BLOCK=THREADS_PER_BLOCK
            )
            torch.cuda.synchronize()

        triton_rows.append(benchmark("triton kernel", triton_kernel_only))
        triton_checksum = triton_out[:3].cpu().numpy()
    print(f"{triton_checksum=}")
    return (triton_rows,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### cuda-python (cuda.core)

    `cuda-python` は NVIDIA 公式の Python バインディングで、2 層に分かれている。

    - `cuda.bindings`: CUDA Driver / Runtime API をほぼそのまま写した薄い層。
    - `cuda.core`: その上の Python らしい層。`Device` / `Program` / `Buffer` /
      `LaunchConfig` / `launch` を提供する。

    ここでは `cuda.core` を使い、CuPy RawKernel と**同じ CUDA C++ ソース**を
    NVRTC で cubin へコンパイルし、`launch` で起動する。
    `ProgramOptions(arch=...)` に `sm_86` のようなアーキテクチャを渡す必要があるので、
    `Device().arch` から組み立てる。

    ホスト側のメモリはページロック（pinned）で確保する。
    `Buffer.copy_from` / `copy_to` は Buffer どうしの転送しか受け付けないため、
    NumPy 配列と直接やり取りはできない。ポインタを `ctypes` 経由で NumPy 配列として
    見る小さなヘルパー `host_view` を用意している。
    この手間がそのまま「最も低水準である」ことの中身である。
    """)
    return


@app.cell
def _(
    BLOCKS_PER_GRID,
    CUDA_SOURCE,
    THREADS_PER_BLOCK,
    VECTOR_SIZE,
    benchmark,
    ccore,
    ctypes,
    gpu_ready,
    gpu_warm,
    left_host,
    np,
    right_host,
):
    cuda_python_rows = []
    if not (gpu_ready and gpu_warm):
        cuda_python_rows.append({"case": "cuda-python", "median_ms": np.nan, "min_ms": np.nan})
        cuda_python_checksum = None
    else:
        cuda_device = ccore.Device()
        cuda_device.set_current()
        cuda_stream = cuda_device.create_stream()
        cuda_program = ccore.Program(
            CUDA_SOURCE,
            "c++",
            ccore.ProgramOptions(arch=f"sm_{cuda_device.arch}", std="c++17"),
        )
        cuda_kernel = cuda_program.compile("cubin").get_kernel("add_kernel")

        def host_view(buffer):
            pointer = ctypes.cast(int(buffer.handle), ctypes.POINTER(ctypes.c_float))
            return np.ctypeslib.as_array(pointer, shape=(VECTOR_SIZE,))

        cuda_nbytes = left_host.nbytes
        cuda_pinned = ccore.LegacyPinnedMemoryResource()
        cuda_host_left = cuda_pinned.allocate(cuda_nbytes)
        cuda_host_right = cuda_pinned.allocate(cuda_nbytes)
        cuda_host_out = cuda_pinned.allocate(cuda_nbytes)
        host_view(cuda_host_left)[:] = left_host
        host_view(cuda_host_right)[:] = right_host

        cuda_left = cuda_device.allocate(cuda_nbytes, stream=cuda_stream)
        cuda_right = cuda_device.allocate(cuda_nbytes, stream=cuda_stream)
        cuda_out = cuda_device.allocate(cuda_nbytes, stream=cuda_stream)
        cuda_host_left.copy_to(cuda_left, stream=cuda_stream)
        cuda_host_right.copy_to(cuda_right, stream=cuda_stream)
        cuda_stream.sync()
        cuda_config = ccore.LaunchConfig(grid=(BLOCKS_PER_GRID,), block=(THREADS_PER_BLOCK,))

        def cuda_python_kernel_only():
            ccore.launch(
                cuda_stream,
                cuda_config,
                cuda_kernel,
                cuda_left,
                cuda_right,
                cuda_out,
                np.int32(VECTOR_SIZE),
            )
            cuda_stream.sync()

        def cuda_python_with_transfer():
            cuda_host_left.copy_to(cuda_left, stream=cuda_stream)
            cuda_host_right.copy_to(cuda_right, stream=cuda_stream)
            ccore.launch(
                cuda_stream,
                cuda_config,
                cuda_kernel,
                cuda_left,
                cuda_right,
                cuda_out,
                np.int32(VECTOR_SIZE),
            )
            cuda_out.copy_to(cuda_host_out, stream=cuda_stream)
            cuda_stream.sync()

        cuda_python_rows.append(benchmark("cuda-python kernel", cuda_python_kernel_only))
        cuda_python_rows.append(benchmark("cuda-python +transfer", cuda_python_with_transfer))
        cuda_python_checksum = host_view(cuda_host_out)[:3].copy()
    print(f"{cuda_python_checksum=}")
    return (cuda_python_rows,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 結果

    6 方式の結果を実効メモリ帯域とともに並べる。`bandwidth_GB_per_s` は
    「読み 2 本 + 書き 1 本」の 3 配列分のバイト数を所要時間で割った値である。

    読み方の要点は 3 つ。

    1. **kernel 同士はほぼ横並びになる。** 帯域律速の演算なので、
       どの方式で書いても GPU の実効帯域が上限になる。方式の選択は速度ではなく
       記述性・依存関係・既存コードとの相性で決めるべきである。
    2. **転送を含めると CPU より遅くなりうる。** PCIe の帯域は GPU の
       デバイスメモリ帯域より 1 桁小さい。加算 1 回のためにデータを往復させると
       転送だけで時間が埋まる。GPU が効くのは、データを GPU 上に置いたまま
       複数の演算を重ねられる場合である。
    3. **`+transfer` の行は横並びで比べてはいけない。** 条件をそろえていない。
       numba-cuda・CuPy・PyTorch の行は、毎回デバイスメモリを確保し直し、
       ページング可能な（pageable）ホストメモリから転送している。
       PyTorch はさらに結果のテンソルも毎回確保するので、この 3 つの中でも条件が違う。
       cuda-python の行はデバイス側もホスト側も確保済みで、しかもホスト側は
       ページロック済み（pinned）である。pinned メモリは DMA で直接転送できるため
       中間バッファへのコピーが要らず、確保の手間と引き換えに転送が大きく速くなる。
       この差がそのまま数値に出ている。
    """)
    return


@app.cell
def _(
    MOVED_GIGABYTES,
    cpu_row,
    cuda_python_rows,
    cupy_rows,
    numba_rows,
    pd,
    torch_rows,
    triton_rows,
):
    all_rows = [
        cpu_row,
        *numba_rows,
        *cupy_rows,
        *torch_rows,
        *triton_rows,
        *cuda_python_rows,
    ]
    summary = pd.DataFrame(all_rows)
    summary["bandwidth_GB_per_s"] = MOVED_GIGABYTES / (summary["min_ms"] / 1e3)
    summary["speedup_vs_cpu"] = cpu_row["min_ms"] / summary["min_ms"]
    summary.round(3)
    return (summary,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### カーネル融合：演算を 1 つにまとめる

    上の比較で方式による差が出なかったのは、題材が加算 1 回だったからである。
    実際のコードは `a * b + a - b * b` のように複数の演算を繋げる。
    ここで方式による大きな差が出る。

    NumPy 互換の API（CuPy の要素演算、PyTorch のテンソル演算）は、
    式を**演算ごとに別々のカーネル**へ分解する。上の式なら

    1. `t1 = a * b`（$a$, $b$ を読み $t_1$ を書く）
    2. `t2 = t1 + a`
    3. `t3 = b * b`
    4. `t4 = t2 - t3`

    の 4 回カーネルを起動し、そのたびに中間配列をメモリへ書き出して読み直す。
    素朴に数えれば配列 11〜12 本ぶんの読み書きになる。

    **融合**すると、1 要素につき $a$ と $b$ を 1 回読んで計算し、結果を 1 回書くだけで済む。
    読み書きは配列 3 本ぶんに減る。帯域律速の演算なので、この比がそのまま速度比になる。

    融合させる方法を 2 つ比べる。

    - `cupy.fuse()`: デコレータを付けるだけで、CuPy が式を 1 つのカーネルへまとめる。
    - `torch.compile()`: PyTorch が計算グラフを追跡し、**Triton のカーネルを自動生成**する。
      手書きの Triton と同じ土俵に自動で載る。初回はコンパイルが走るため、
      ウォームアップの時間を別に測って表示する。
    """)
    return


@app.cell
def _(VECTOR_SIZE, benchmark, cp, gpu_ready, gpu_warm, left_host, np, right_host, time, torch):
    fusion_rows = []
    if not (gpu_ready and gpu_warm):
        fusion_rows.append({"case": "fusion", "median_ms": np.nan, "min_ms": np.nan})
        fusion_checksums = None
        torch_compile_warmup_s = np.nan
    else:
        fuse_left = cp.asarray(left_host)
        fuse_right = cp.asarray(right_host)

        def cupy_eager_expression():
            result = fuse_left * fuse_right + fuse_left - fuse_right * fuse_right
            cp.cuda.Stream.null.synchronize()
            return result

        @cp.fuse()
        def cupy_fused_expression(a, b):
            return a * b + a - b * b

        def cupy_fused():
            result = cupy_fused_expression(fuse_left, fuse_right)
            cp.cuda.Stream.null.synchronize()
            return result

        torch_fuse_left = torch.from_numpy(left_host).cuda()
        torch_fuse_right = torch.from_numpy(right_host).cuda()

        def torch_eager_expression():
            result = (
                torch_fuse_left * torch_fuse_right
                + torch_fuse_left
                - torch_fuse_right * torch_fuse_right
            )
            torch.cuda.synchronize()
            return result

        compiled_expression = torch.compile(lambda a, b: a * b + a - b * b)
        _warmup_start = time.perf_counter()
        compiled_expression(torch_fuse_left, torch_fuse_right)
        torch.cuda.synchronize()
        torch_compile_warmup_s = time.perf_counter() - _warmup_start

        def torch_compiled():
            result = compiled_expression(torch_fuse_left, torch_fuse_right)
            torch.cuda.synchronize()
            return result

        fusion_rows.append(benchmark("cupy eager (4 kernels)", cupy_eager_expression))
        fusion_rows.append(benchmark("cupy.fuse", cupy_fused))
        fusion_rows.append(benchmark("torch eager (4 kernels)", torch_eager_expression))
        fusion_rows.append(benchmark("torch.compile (triton)", torch_compiled))
        fusion_checksums = [
            float(cupy_eager_expression()[0]),
            float(cupy_fused()[0]),
            float(torch_eager_expression()[0].item()),
            float(torch_compiled()[0].item()),
        ]
    print(f"{fusion_checksums=}")
    print(f"torch.compile warmup: {torch_compile_warmup_s:.1f} s")
    print(f"array size: {VECTOR_SIZE} elements")
    return (fusion_rows,)


@app.cell
def _(cupy_rows, fusion_rows, pd):
    # vector add の CuPy 要素演算は「読み 2 + 書き 1」の 3 パス。これを基準に、
    # 融合前後の所要時間が何パス相当かを実測から逆算する
    reference_ms = next(row["min_ms"] for row in cupy_rows if row["case"] == "cupy elementwise")
    fusion_summary = pd.DataFrame(fusion_rows)
    fusion_summary["array_passes"] = 3 * fusion_summary["min_ms"] / reference_ms
    fusion_summary["speedup_vs_eager"] = (
        fusion_summary["min_ms"].iloc[0] / fusion_summary["min_ms"]
    )
    fusion_summary.round(2)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    `array_passes` は、vector add の CuPy 要素演算（読み 2 本 + 書き 1 本 = 3 パス）を
    基準にして、所要時間から逆算した「配列を何本ぶん読み書きしたか」の推定値である。
    帯域律速なので、時間の比がそのままメモリ往復量の比になる。

    融合版が 3 に近い値になれば、**中間配列がすべて消えた**ことの裏付けになる。
    eager 版は式を 4 つのカーネルへ分けるので、素朴に数えれば 11〜12 パスになる。
    実測がそれより少し小さく出るのは、直前のカーネルが書いた中間配列の一部が
    キャッシュに残っていて、メモリまで往復しなくて済むためである。

    実務上の指針は単純である。**要素ごとの演算を並べるときは融合する。**
    CuPy なら `cupy.fuse`、PyTorch なら `torch.compile` を付けるだけでよく、
    カーネルを手で書く必要はない。
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 演算律速の題材：行列積

    ここまでの題材はすべて帯域律速だった。今度は逆に、**演算が支配的な**題材を見る。

    $N \times N$ の行列積は
    浮動小数点演算が $2N^3$ 回、転送するデータが $3N^2$ 要素である。
    比を取ると 1 要素あたりの演算数は $N$ に比例して増える。
    つまり **$N$ を大きくするほど、転送のコストが計算に埋もれる**。
    これが「GPU が効く仕事」の典型である。

    比較するのは NumPy（裏で CPU の BLAS が動き、複数コアを使う）と
    `torch.matmul`（裏で cuBLAS が動く）。どちらも**自分でカーネルを書かない**点が重要で、
    行列積のような定番の演算は、手書きより十分に最適化されたライブラリを呼ぶほうが速い。

    GPU 側は kernel のみと転送込みの両方を測る。
    vector add では転送込みが CPU に負けたが、行列積ではどうなるかを見る。
    """)
    return


@app.cell
def _(benchmark, gpu_ready, gpu_warm, np, pd, torch):
    MATMUL_SIZES = [1024, 2048, 4096]
    matmul_rows = []
    matmul_rng = np.random.default_rng(0)
    for _size in MATMUL_SIZES:
        _flops = 2 * _size**3
        _left = matmul_rng.random((_size, _size), dtype=np.float32)
        _right = matmul_rng.random((_size, _size), dtype=np.float32)

        def _cpu_matmul(a=_left, b=_right):
            return a @ b

        _cpu = benchmark(f"numpy matmul N={_size}", _cpu_matmul, repeats=3, warmup_s=0.3)
        _row = {
            "N": _size,
            "cpu_ms": _cpu["min_ms"],
            "cpu_TFLOPS": _flops / (_cpu["min_ms"] / 1e3) / 1e12,
        }

        if gpu_ready and gpu_warm:
            _device_left = torch.from_numpy(_left).cuda()
            _device_right = torch.from_numpy(_right).cuda()

            def _gpu_kernel(a=_device_left, b=_device_right):
                torch.matmul(a, b)
                torch.cuda.synchronize()

            def _gpu_with_transfer(a=_left, b=_right):
                return (torch.from_numpy(a).cuda() @ torch.from_numpy(b).cuda()).cpu().numpy()

            _kernel = benchmark(f"cublas matmul N={_size}", _gpu_kernel, repeats=3, warmup_s=0.3)
            _transfer = benchmark(
                f"cublas matmul +transfer N={_size}", _gpu_with_transfer, repeats=3, warmup_s=0.3
            )
            _row.update({
                "gpu_kernel_ms": _kernel["min_ms"],
                "gpu_kernel_TFLOPS": _flops / (_kernel["min_ms"] / 1e3) / 1e12,
                "gpu_transfer_ms": _transfer["min_ms"],
                "speedup_kernel": _cpu["min_ms"] / _kernel["min_ms"],
                "speedup_with_transfer": _cpu["min_ms"] / _transfer["min_ms"],
            })
            del _device_left, _device_right
            torch.cuda.empty_cache()
        matmul_rows.append(_row)
    pd.DataFrame(matmul_rows).round(2)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    vector add との違いがはっきり出る。

    | 題材 | 1 要素あたりの演算 | kernel のみ | 転送込み |
    | --- | --- | --- | --- |
    | vector add | 定数（1/12 FLOP/byte） | GPU が大きく勝つ | **GPU が負ける** |
    | 行列積 | $N$ に比例して増える | GPU が大きく勝つ | **GPU が勝つ** |

    同じ GPU、同じ転送経路でも、**題材の演算強度が結論を反転させる**。
    「GPU は速い／遅い」と一般化できず、
    転送量あたりどれだけ計算するかで判断するしかない。

    ただし `speedup_with_transfer` の列は $N$ を大きくしても増えない。
    理論上は計算が $N^3$、転送が $N^2$ なので $N$ とともに有利になるはずだが、
    ここでは 2 つの要因がそれを打ち消している。

    - `cpu_TFLOPS` 列のとおり、CPU 側も $N$ が大きいほど効率が上がる（比較相手が強くなる）。
    - `gpu_transfer_ms` の伸びが $N^2$ より速い。$N$ を 1024 から 2048 へ倍にした区間は
      転送量に比例して増えるが、4096 ではそれ以上に跳ねる。
      この転送はページング可能なホストメモリを使い、毎回デバイスメモリを確保し直すため、
      大きなサイズでステージングと再確保のコストが効いてくる。

    転送を本気で速くするならページロックメモリと確保の使い回しが要る。
    それでも**行列積では転送を含めても GPU が勝つ**という結論は変わらない。

    もう 1 つ読み取れるのは、`cpu_TFLOPS` と `gpu_kernel_TFLOPS` の差である。
    どちらも人が書いたループではなく、BLAS と cuBLAS という
    高度に最適化されたライブラリの数字である。
    行列積・FFT・畳み込みのような定番の演算では、
    **自分でカーネルを書かないことが最も効く最適化**になる。
    このノートで見てきたカーネルの書き方は、
    ライブラリに無い演算を書くときの手段だと位置づけるのが正しい。
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 落とし穴：fork のコストは親プロセスの大きさで決まる

    上の表でワーカーを 8 に増やしても頭打ちになっている。
    6 タスクしか無いことに加えて、**プロセスの生成そのものが重い**からである。

    `fork` は親のアドレス空間を子へ引き継ぐ。実データはコピーオンライトで共有されるが、
    ページテーブルの複製と各種ランタイムの後処理は親のメモリマップの大きさに比例する。
    このノートブックの親プロセスは torch・CuPy・dask・pandas などを読み込んでいるため、
    仕事をしない空のプールを起こして畳むだけでも無視できない時間がかかる。

    次のセルで 6 つの条件を測る。1 から 4 は別プロセスを立てて 1 つずつ条件を積み上げ、
    5 と 6 はこのノートブック自身のプロセスで測る。

    1. 何も import しない最小の Python
    2. torch を import しただけ
    3. さらに CuPy・dask・pandas も import
    4. さらに GPU 上に配列を 1 つ作って CUDA コンテキストを実際に生成
    5. このノートブックのプロセス、GPU セクションより前
    6. このノートブックのプロセス、GPU セクションを通した後

    4 で `torch.cuda.is_available()` ではなく `torch.zeros(1, device="cuda")` を使うのは、
    **`is_available()` は CUDA を初期化しないから**である。PyTorch はこの判定を NVML で
    行うため、呼んでも `torch.cuda.is_initialized()` は `False` のままになる。

    よく「CUDA を初期化したプロセスから fork するな」と言われる。子プロセスの中で
    CUDA を使えないことは常に成り立つ制約である。しかし**コストの面では、
    3 と 4 の差は 1 から 3 での増え方に比べてはるかに小さく、実行ごとの揺らぎと
    同程度である**。効いているのは CUDA の初期化ではなく、
    **親プロセスが抱えるメモリマップの大きさ**である。1 から 3 へ import を積むだけで
    空プールの生成は数倍になり、5 から 6 の増加も、GPU セクションで確保した
    デバイスメモリとページロックメモリを含む footprint 全体の増加による。
    """)
    return


@app.cell
def _(empty_pool_seconds, fork_cost_before_cuda, pd, subprocess, summary, sys):
    _ = summary  # GPU セクションを通した後に測るための依存

    FORK_PROBE = """
import concurrent.futures as cf
import time
from multiprocessing import get_context
{setup}
start = time.perf_counter()
with cf.ProcessPoolExecutor(max_workers=8, mp_context=get_context("fork")) as executor:
    list(executor.map(int, range(8)))
print(time.perf_counter() - start)
"""

    def probe_fork(setup: str):
        completed = subprocess.run(
            [sys.executable, "-c", FORK_PROBE.format(setup=setup)],
            capture_output=True,
            text=True,
            timeout=600,
            check=True,
        )
        return float(completed.stdout.strip().splitlines()[-1])

    fork_rows = [
        {"parent": "1. bare python", "empty_pool_ms": probe_fork("") * 1e3},
        {"parent": "2. + import torch", "empty_pool_ms": probe_fork("import torch") * 1e3},
        {
            "parent": "3. + import cupy, dask, pandas",
            "empty_pool_ms": probe_fork("import cupy, dask.dataframe, pandas, torch") * 1e3,
        },
        {
            "parent": "4. + CUDA context created",
            "empty_pool_ms": probe_fork(
                "import cupy, dask.dataframe, pandas, torch;"
                ' torch.zeros(1, device="cuda"); torch.cuda.synchronize()'
            )
            * 1e3,
        },
        {"parent": "5. this notebook, before GPU", "empty_pool_ms": fork_cost_before_cuda * 1e3},
        {"parent": "6. this notebook, after GPU", "empty_pool_ms": empty_pool_seconds() * 1e3},
    ]
    pd.DataFrame(fork_rows).round(1)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## データの読み書き

    最後に入出力を比べる。題材は次の 2 つで、いずれもこのセルで生成する。

    - 表データ: 200,000 行 4 列（`carat` / `depth` / `table` / `price`）を
      CSV と Parquet の両方で `data/others/rapid_python/` へ書き出す。
    - JSON: 5,000 要素の配列（各要素は `id` / `value` / `square`）。

    表データの読み込みで比べるのは次の 6 通り。

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
       ファイルサイズも小さくなるので、後で `size_MiB` 列と併せて見る。
    2. **同じ CSV でもパーサで変わる。** pandas の既定は単一コアの C 実装だが、
       `engine="pyarrow"` と polars は複数コアでパースする。

    dask はこの規模では pandas より遅い。タスクグラフの構築という固定コストが乗るためで、
    有利になるのはメモリに載らない大きさや、分割して並列処理できる場合である。

    JSON 側では `json` と `orjson` を読み書きの両方で比べる。
    `orjson` は C 実装で、この規模でも確実に速い。
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
    続いて JSON の読み書きを `json` と `orjson` で比べる。
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


if __name__ == "__main__":
    app.run()
