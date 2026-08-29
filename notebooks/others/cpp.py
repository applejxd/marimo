import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import shlex
    import shutil
    import subprocess
    import textwrap
    from pathlib import Path

    import marimo as mo

    notebook_dir = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
    repo_root = notebook_dir.parents[1]
    work_dir = repo_root / "data" / "others" / "cpp"
    work_dir.mkdir(parents=True, exist_ok=True)
    compiler_path = shutil.which("g++")

    return Path, compiler_path, mo, shlex, subprocess, textwrap, work_dir


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # C++ コードの実行方法

    notebook 自身が C++ のソースを書き出し、`subprocess.run()` でコンパイルして
    実行する。標準出力もコンパイルエラーもその場で確認できる。
    計算時間の計測に加えて、OpenMP（`#pragma omp`）と OpenACC（`#pragma acc`）による
    並列化も扱う。いずれも g++ 単体でコンパイルできる。
    """)
    return


@app.cell
def _(compiler_path, shlex, subprocess, textwrap, work_dir):
    def run_command(command_parts):
        return subprocess.run(
            command_parts,
            cwd=work_dir,
            capture_output=True,
            text=True,
            check=False,
        )

    def compile_and_run(name: str, source: str, extra_flags=()):
        source_path = work_dir / f"{name}.cpp"
        binary_path = work_dir / name
        source_path.write_text(textwrap.dedent(source).strip() + "\n", encoding="utf-8")

        if compiler_path is None:
            return {
                "compile_command": "g++ not found",
                "compile_returncode": 127,
                "compile_stdout": "",
                "compile_stderr": "g++ is not installed.",
                "run_returncode": None,
                "run_stdout": "",
                "run_stderr": "",
                "source_path": source_path,
            }

        compile_command = [compiler_path, str(source_path), "-std=c++17", "-O2", "-o", str(binary_path), *extra_flags]
        compile_result = run_command(compile_command)
        report = {
            "compile_command": " ".join(shlex.quote(part) for part in compile_command),
            "compile_returncode": compile_result.returncode,
            "compile_stdout": compile_result.stdout,
            "compile_stderr": compile_result.stderr,
            "run_returncode": None,
            "run_stdout": "",
            "run_stderr": "",
            "source_path": source_path,
        }
        if compile_result.returncode == 0:
            run_result = run_command([str(binary_path)])
            report["run_returncode"] = run_result.returncode
            report["run_stdout"] = run_result.stdout
            report["run_stderr"] = run_result.stderr
        return report

    def format_report(title: str, report):
        body = [
            f"### {title}",
            f"- source: `{report['source_path'].name}`",
            f"- compile command: `{report['compile_command']}`",
            f"- compile return code: `{report['compile_returncode']}`",
        ]
        if report["compile_stdout"]:
            body.append("```text\n" + report["compile_stdout"].rstrip() + "\n```")
        if report["compile_stderr"]:
            body.append("```text\n" + report["compile_stderr"].rstrip() + "\n```")
        if report["run_returncode"] is not None:
            body.append(f"- run return code: `{report['run_returncode']}`")
        if report["run_stdout"]:
            body.append("```text\n" + report["run_stdout"].rstrip() + "\n```")
        if report["run_stderr"]:
            body.append("```text\n" + report["run_stderr"].rstrip() + "\n```")
        return "\n".join(body)

    return compile_and_run, format_report, run_command


@app.cell
def _(Path, compiler_path, format_report, mo, run_command):
    if compiler_path is None:
        compiler_summary = mo.md("g++ が見つからないため、以降のサンプルはコンパイル結果の説明のみになる。")
    else:
        version_result = run_command([compiler_path, "--version"])
        report = {
            "compile_command": f"{compiler_path} --version",
            "compile_returncode": version_result.returncode,
            "compile_stdout": version_result.stdout,
            "compile_stderr": version_result.stderr,
            "run_returncode": None,
            "run_stdout": "",
            "run_stderr": "",
            "source_path": Path(compiler_path),
        }
        compiler_summary = mo.md(format_report("コンパイラ確認", report))
    compiler_summary
    return


@app.cell
def _():
    hello_world_source = r'''
    #include <iostream>

    int main() {
      std::cout << "Hello World!" << std::endl;
      return 0;
    }
    '''
    return (hello_world_source,)


@app.cell
def _(compile_and_run, format_report, hello_world_source, mo):
    hello_report = compile_and_run("hello_world", hello_world_source)
    mo.md(format_report("Hello World", hello_report))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 計算時間計測
    """)
    return


@app.cell
def _():
    chrono_source = r'''
    #include <chrono>
    #include <iostream>
    #include <vector>

    int main() {
        auto start = std::chrono::system_clock::now();
        std::vector<int> values;
        constexpr int N = 1000 * 1000;
        for (int i = 0; i < N; ++i) {
            values.push_back(i);
        }
        auto end = std::chrono::system_clock::now();
        auto msec = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        std::cout << msec << " milli sec" << std::endl;
        return 0;
    }
    '''
    return (chrono_source,)


@app.cell
def _(chrono_source, compile_and_run, format_report, mo):
    chrono_report = compile_and_run("chrono_vector", chrono_source)
    mo.md(format_report("chrono による計測", chrono_report))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## OpenMP
    """)
    return


@app.cell
def _():
    openmp_probe_source = r'''
    #include <cstdio>
    #ifdef _OPENMP
    #include <omp.h>
    #endif

    int main() {
    #ifdef _OPENMP
        std::printf("OpenMP enabled\n");
    #else
        std::printf("OpenMP unavailable\n");
    #endif
        return 0;
    }
    '''
    return (openmp_probe_source,)


@app.cell
def _(compile_and_run, format_report, mo, openmp_probe_source):
    openmp_probe_report = compile_and_run("openmp_probe", openmp_probe_source, extra_flags=("-fopenmp",))
    mo.md(format_report("OpenMP の有効化確認", openmp_probe_report))
    return


@app.cell
def _():
    openmp_parallel_source = r'''
    #include <chrono>
    #include <iostream>
    #include <vector>
    #ifdef _OPENMP
    #include <omp.h>
    #endif

    int main() {
        auto start = std::chrono::system_clock::now();
        constexpr int N = 1000 * 1000;
        std::vector<int> values(N);
    #pragma omp parallel for
        for (int i = 0; i < N; ++i) {
            values[i] = i;
        }
        auto end = std::chrono::system_clock::now();
        auto msec = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        std::cout << msec << " milli sec" << std::endl;
        return 0;
    }
    '''
    return (openmp_parallel_source,)


@app.cell
def _(compile_and_run, format_report, mo, openmp_parallel_source):
    openmp_parallel_report = compile_and_run("openmp_parallel", openmp_parallel_source, extra_flags=("-fopenmp",))
    mo.md(format_report("OpenMP による並列化", openmp_parallel_report))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## OpenACC

    OpenACC のサポート有無は環境依存なので、コンパイルログをそのまま残す。
    """)
    return


@app.cell
def _():
    openacc_source = r'''
    #include <chrono>
    #include <iostream>
    #include <vector>
    #ifdef _OPENACC
    #include <openacc.h>
    #endif

    int main() {
        auto start = std::chrono::system_clock::now();
        constexpr int N = 1000 * 1000;
        std::vector<int> values(N);
    #pragma acc data copy(values)
        {
    #pragma acc kernels
            for (int i = 0; i < N; ++i) {
                values[i] = i;
            }
        }
        auto end = std::chrono::system_clock::now();
        auto msec = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        std::cout << msec << " milli sec" << std::endl;
        return 0;
    }
    '''
    return (openacc_source,)


@app.cell
def _(compile_and_run, format_report, mo, openacc_source):
    openacc_report = compile_and_run("openacc_parallel", openacc_source, extra_flags=("-fopenacc",))
    mo.md(format_report("OpenACC の確認", openacc_report))
    return


if __name__ == "__main__":
    app.run()
