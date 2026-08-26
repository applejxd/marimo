import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import atexit
    import concurrent.futures as cf
    import hashlib
    import json
    import socket
    import sys
    import time
    import warnings
    from functools import partial
    from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
    from multiprocessing import Array, Manager, Process, Value, get_context
    from pathlib import Path
    from threading import Thread

    import dask.dataframe as dd
    import marimo as mo
    import numpy as np
    import orjson
    import pandas as pd
    import requests
    from numba import cuda, njit, prange
    from tqdm.contrib.concurrent import process_map, thread_map

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

    return Array, Manager, Path, Process, SimpleHTTPRequestHandler, Thread, ThreadingHTTPServer, Value, atexit, cf, cuda, data_dir, dd, get_context, hashlib, json, mo, njit, np, orjson, partial, pd, prange, process_map, requests, socket, sys, thread_map, time, warnings


@app.cell
def _(SimpleHTTPRequestHandler, Thread, ThreadingHTTPServer, atexit, data_dir, json, np, partial, requests, socket, time):
    def timed_call(label: str, func, *args, **kwargs):
        start = time.perf_counter()
        value = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        print(f"[{label}] {elapsed:.3g} s")
        return value, elapsed

    http_dir = data_dir / "http_payloads"
    http_dir.mkdir(exist_ok=True)
    rng_payload = np.random.default_rng(42)
    for index in range(6):
        payload = {"id": index, "values": rng_payload.normal(size=8).round(4).tolist()}
        (http_dir / f"payload_{index}.json").write_text(json.dumps(payload), encoding="utf-8")

    class QuietHandler(SimpleHTTPRequestHandler):
        def log_message(self, format, *args):
            return

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    http_server = ThreadingHTTPServer(("127.0.0.1", port), partial(QuietHandler, directory=str(http_dir)))
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
    return cleanup_server, thread_urls, timed_call


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Python 高速化メモ

    Colab マジックを除去し、CPU / GPU / 並列処理 / データロードの例を
    marimo でそのまま headless 実行できるように整理した。
    """)
    return


@app.cell
def _(mo, njit, np, pd, sys, timed_call):
    sys.setrecursionlimit(100000)

    def ack(m: int, n: int):
        if m == 0:
            return n + 1
        if n == 0:
            return ack(m - 1, 1)
        return ack(m - 1, ack(m, n - 1))

    _, base_elapsed = timed_call("ack(3, 10)", ack, 3, 10)

    @njit(cache=False)
    def lazy_ack(m, n):
        if m == 0:
            return n + 1
        if n == 0:
            return lazy_ack(m - 1, 1)
        return lazy_ack(m - 1, lazy_ack(m, n - 1))

    _, lazy_first = timed_call("lazy njit first", lazy_ack, 3, 10)
    _, lazy_second = timed_call("lazy njit second", lazy_ack, 3, 10)

    @njit("int32(int32, int32)", cache=False)
    def eager_ack(m, n):
        if m == 0:
            return n + 1
        if n == 0:
            return eager_ack(m - 1, 1)
        return eager_ack(m - 1, eager_ack(m, n - 1))

    _, eager_first = timed_call("typed njit first", eager_ack, 3, 10)
    _, eager_second = timed_call("typed njit second", eager_ack, 3, 10)
    pd.DataFrame([
        {"case": "python", "elapsed_s": base_elapsed},
        {"case": "lazy_njit_first", "elapsed_s": lazy_first},
        {"case": "lazy_njit_second", "elapsed_s": lazy_second},
        {"case": "typed_njit_first", "elapsed_s": eager_first},
        {"case": "typed_njit_second", "elapsed_s": eager_second},
    ])
    return


@app.cell
def _(njit, pd, prange, timed_call):
    def double_sum(size: int):
        total = 0
        for i in range(size):
            for j in range(size):
                total += i - j
        return total

    @njit("int32(int32)", cache=False)
    def double_sum_njit(size):
        total = 0
        for i in range(size):
            for j in range(size):
                total += i - j
        return total

    @njit("int32(int32)", cache=False, parallel=True, fastmath=True)
    def double_sum_fast(size):
        total = 0
        for i in prange(size):
            for j in range(size):
                total += i - j
        return total

    size = 4000
    answer_python, elapsed_python = timed_call("double_sum python", double_sum, size)
    answer_njit, elapsed_njit = timed_call("double_sum njit", double_sum_njit, size)
    answer_fast, elapsed_fast = timed_call("double_sum fast", double_sum_fast, size)
    pd.DataFrame([
        {"case": "python", "answer": answer_python, "elapsed_s": elapsed_python},
        {"case": "njit", "answer": answer_njit, "elapsed_s": elapsed_njit},
        {"case": "parallel_fastmath", "answer": answer_fast, "elapsed_s": elapsed_fast},
    ])
    return


@app.cell
def _(cuda, np, pd, timed_call):
    gpu_rows = []
    if not cuda.is_available():
        gpu_rows.append({"backend": "cuda", "status": "not available", "elapsed_s": None})
    else:
        @cuda.jit
        def add_kernel(a, b, c):
            index = cuda.grid(1)
            if index < c.size:
                c[index] = a[index] + b[index]

        def add_arrays_gpu(a, b, threads_per_block=256):
            blocks = (a.size + threads_per_block - 1) // threads_per_block
            out = cuda.to_device(np.zeros_like(a))
            add_kernel[blocks, threads_per_block](cuda.to_device(a), cuda.to_device(b), out)
            return out.copy_to_host()

        array_size = 2_000_000
        left = np.ones(array_size, dtype=np.float32)
        right = np.ones(array_size, dtype=np.float32)
        _, cpu_elapsed = timed_call("cpu vector add", lambda: left + right)
        gpu_rows.append({"backend": "cpu", "status": "ok", "elapsed_s": cpu_elapsed})
        try:
            _, gpu_elapsed = timed_call("gpu vector add", add_arrays_gpu, left, right)
        except Exception as exc:
            gpu_rows.append(
                {
                    "backend": "cuda",
                    "status": f"error: {exc.__class__.__name__}",
                    "elapsed_s": None,
                }
            )
        else:
            gpu_rows.append({"backend": "cuda", "status": "ok", "elapsed_s": gpu_elapsed})
    pd.DataFrame(gpu_rows)
    return

@app.cell
def _(cf, get_context, hashlib, mo, partial, pd, timed_call):
    mo.md("## concurrent.futures での高速化")
    cpu_tasks = [60_000, 65_000, 70_000, 75_000, 80_000, 85_000]
    _digest_worker_executor = partial(hashlib.pbkdf2_hmac, "sha256", b"marimo", b"parallel", dklen=32)
    worker_rows = []
    for _process_workers in [1, 2, 4]:
        start = __import__("time").perf_counter()
        with cf.ProcessPoolExecutor(max_workers=_process_workers, mp_context=get_context("fork")) as _process_executor:
            list(_process_executor.map(_digest_worker_executor, cpu_tasks))
        worker_rows.append({"max_workers": _process_workers, "elapsed_s": __import__("time").perf_counter() - start})
    print(f"CPU count is {__import__('os').cpu_count()}")
    pd.DataFrame(worker_rows)
    return (cpu_tasks,)


@app.cell
def _(cf, pd, requests, thread_urls, timed_call):
    def fetch_status(url: str):
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        return {"url": url.rsplit("/", 1)[-1], "status_code": response.status_code, "bytes": len(response.content)}

    def fetch_all(urls, max_workers: int):
        with cf.ThreadPoolExecutor(max_workers=max_workers) as _thread_executor:
            return list(_thread_executor.map(fetch_status, urls))

    timing_rows = []
    last_statuses = None
    for _thread_workers in [1, 2, 4, 8]:
        statuses, elapsed = timed_call(
            f"thread pool max_workers={_thread_workers}",
            fetch_all,
            thread_urls,
            _thread_workers,
        )
        timing_rows.append({"max_workers": _thread_workers, "elapsed_s": elapsed})
        last_statuses = statuses
    pd.DataFrame(last_statuses)
    pd.DataFrame(timing_rows)
    return (fetch_status,)


@app.cell
def _(cleanup_server, fetch_status, get_context, pd, thread_map, thread_urls):
    thread_map_rows = thread_map(fetch_status, thread_urls, max_workers=8, disable=True)
    _fork_context = get_context("fork")

    def shared_memory_target(number, array):
        number.value = 3.1415927
        for index in range(len(array)):
            array[index] = -array[index]

    shared_number = _fork_context.Value("d", 0.0, lock=False)
    shared_array = _fork_context.Array("i", range(10), lock=False)
    process = _fork_context.Process(target=shared_memory_target, args=(shared_number, shared_array))
    process.start()
    process.join()
    process.close()

    def pipe_target(index, send_end):
        send_end.send({"index": index, "negated": -index})
        send_end.close()

    _pipe_endpoints = []
    _pipe_processes = []
    for _pipe_index in range(4):
        _recv_end, _send_end = _fork_context.Pipe(duplex=False)
        _pipe_process = _fork_context.Process(target=pipe_target, args=(_pipe_index, _send_end))
        _pipe_process.start()
        _send_end.close()
        _pipe_endpoints.append(_recv_end)
        _pipe_processes.append(_pipe_process)
    pipe_messages = [_recv_end.recv() for _recv_end in _pipe_endpoints]
    for _recv_end in _pipe_endpoints:
        _recv_end.close()
    for _pipe_process in _pipe_processes:
        _pipe_process.join()
        _pipe_process.close()

    cleanup_server()
    pd.DataFrame(thread_map_rows)
    pd.DataFrame([
        {"shared_number": shared_number.value, "shared_array_head": list(shared_array)[:5], "pipe_messages": str(pipe_messages)}
    ])
    return


@app.cell
def _(data_dir, dd, json, np, orjson, pd, timed_call):
    rng_data = np.random.default_rng(0)
    csv_path = data_dir / "diamonds.csv"
    json_path = data_dir / "large-file.json"
    csv_frame = pd.DataFrame({
        "carat": rng_data.uniform(0.2, 2.5, size=20000).round(3),
        "depth": rng_data.uniform(55.0, 70.0, size=20000).round(3),
        "table": rng_data.uniform(50.0, 70.0, size=20000).round(3),
        "price": rng_data.integers(300, 18000, size=20000),
    })
    csv_frame.to_csv(csv_path, index=False)
    json_payload = [
        {"id": int(index), "value": float(value), "square": float(value**2)}
        for index, value in enumerate(rng_data.normal(size=5000))
    ]
    json_path.write_text(json.dumps(json_payload, ensure_ascii=False), encoding="utf-8")

    _, pandas_elapsed = timed_call("pandas.read_csv", pd.read_csv, csv_path)
    _, dask_elapsed = timed_call("dask.read_csv", lambda: dd.read_csv(csv_path).compute())
    _, json_elapsed = timed_call("json.load", lambda: json.loads(json_path.read_text(encoding="utf-8")))
    data_orjson, orjson_elapsed = timed_call("orjson.loads", lambda: orjson.loads(json_path.read_bytes()))
    _, dump_elapsed = timed_call(
        "json.dump",
        lambda: (data_dir / "large-file.standard.json").write_text(json.dumps(data_orjson), encoding="utf-8"),
    )
    _, orjson_dump_elapsed = timed_call(
        "orjson.dumps",
        lambda: (data_dir / "large-file.orjson.json").write_bytes(orjson.dumps(data_orjson)),
    )
    pd.DataFrame([
        {"task": "pandas.read_csv", "elapsed_s": pandas_elapsed},
        {"task": "dask.read_csv", "elapsed_s": dask_elapsed},
        {"task": "json.load", "elapsed_s": json_elapsed},
        {"task": "orjson.loads", "elapsed_s": orjson_elapsed},
        {"task": "json.dump", "elapsed_s": dump_elapsed},
        {"task": "orjson.dumps", "elapsed_s": orjson_dump_elapsed},
    ])
    return


if __name__ == "__main__":
    app.run()
