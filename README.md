# Scientific notebooks with marimo

Scientific and machine-learning examples maintained as executable
[marimo](https://marimo.io/) notebooks. The original Jupyter notebooks are retained under
[`legacy/`](legacy/) for migration review; active notebooks live under [`notebooks/`](notebooks/).
[`migration-map.json`](migration-map.json) records the few renamed notebooks whose original names
would shadow imported Python packages.

## Environment

The project uses [uv](https://docs.astral.sh/uv/) and Python 3.11.

```shell
uv sync --all-groups
```

The `gpu` group installs PyTorch and TensorFlow. A CUDA-capable GPU is expected for the complete
site build, although notebooks select CPU explicitly when CUDA is unavailable. `specialized`
contains large or native packages such as NGSolve and Open3D. A C++ compiler with OpenMP support
is required by `notebooks/others/cpp.py`.

No notebook requires an API key or other secret. Public datasets are downloaded from fixed URLs
and cached under the ignored `data/` directory.

## Run and edit

```shell
uv run --all-groups marimo edit notebooks/algorithm/GradientDescent.py
uv run --all-groups marimo run notebooks/algorithm/GradientDescent.py
```

## Validate

```shell
uv run python scripts/check_migration.py
uv run ruff check .
uv run marimo check --strict notebooks
uv run python scripts/check_site.py
```

## Build the site

`build_site.py` discovers marimo files recursively. Adding a notebook under `notebooks/` is enough
to add it to the generated index; no hand-maintained list exists.

```shell
uv run python scripts/build_site.py
python -m http.server --directory site
```

Every notebook is executed during export. A non-zero process status, marimo's
`some cells failed to execute` warning, or an exception marker in generated HTML fails the build
and removes the affected output. The build also writes `site/notebooks-manifest.json` with SHA-256
hashes of every source and HTML file; the Pages workflow rejects stale or edited generated output.
To inspect discovery and index generation without executing:

```shell
uv run python scripts/build_site.py --discover-only
```

The generated `site/` directory is committed. GitHub Actions only uploads those static files to
GitHub Pages; it does not run models, download datasets, or need secrets. For the first deployment,
set **Settings → Pages → Source** to **GitHub Actions** if automatic enablement is not permitted.
Pushes that change notebook sources also run the deployment workflow; its manifest check rejects
the push's site artifact until the affected HTML has been rebuilt.
