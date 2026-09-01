# AGENTS.md

## このファイルの目的

- AI エージェント向けの実務ルールを定義する。
- 導入手順や背景は README.md にある。ここには実行手順と安全境界だけを書く。

## プロジェクト概要

- プロジェクト名: scientific-marimo-notebooks
- 概要: 科学技術計算と機械学習の marimo notebook を実行し、静的 HTML を GitHub Pages へ公開する。

## 技術スタック

- Language: Python 3.11 のみ（`pyproject.toml` で `>=3.11,<3.12` に固定）
- Notebook: marimo
- Package Manager: uv
- Linter: ruff（`line-length = 100`、`select = ["E4", "E7", "E9", "F", "I"]`）

## 最重要コマンド

```shell
# Install（specialized/gpu グループを含む。NGSolve などはここに入る）
uv sync --all-groups

# 検証（すべて通ること）
uv run ruff check .
uv run marimo check --strict notebooks
uv run python scripts/check_math.py
uv run python scripts/check_site.py

# 編集・実行
uv run --all-groups marimo edit notebooks/<path>.py

# ビルド（変更した notebook だけを再実行する）
uv run python scripts/build_site.py --notebook <path>.py

# ビルド（全 39 冊を再実行。GPU 込みで 30 分以上かかる）
uv run python scripts/build_site.py
```

## 変更時の必須チェック

1. 上の検証コマンド 5 つをすべて通す。
2. 変更した notebook を `build_site.py --notebook` で再ビルドする。
3. 生成された `site/**.html` と `site/notebooks-manifest.json` を一緒にコミットする。
   CI は notebook を実行せず、コミット済みの HTML を配信するだけである
   （`.github/workflows/pages.yml`）。manifest の SHA-256 が一致しないと deploy が拒否される。

## ディレクトリ責務

- `notebooks/`: 公開する marimo notebook。`build_site.py` が再帰的に自動検出するため、
  一覧を手で管理する必要はない。
- `site/`: 生成物。Git の管理対象なので、notebook を変更したら必ず再生成してコミットする。
- `scripts/`: 検証とビルド。`_generated/`・`*.npz`・`data/` は `.gitignore` 済み。
- ファイル名を変えたときは `site/` の旧 HTML を削除し、`build_site.py --notebook` で
  再生成する（manifest は部分ビルドでも旧エントリを落とすようになっている）。

## notebook の規約

- 文体は常体（である調）。敬体（ですます調）は使わない。
- 数式: display math は `$$` のみを使い、`$$` は単独行に置く。数式ブロック内で
  行頭を「`-` `+` `*` または数字＋ピリオド」＋空白にしない
  （Markdown のリストと解釈されて数式が割れる。`check_math.py` は行頭の
  `^\s*(?:[-+*]|\d+\.)\s` を検出する）。`\[ \]` と `\begin{equation}` 系も拒否される。
- marimo の変数: 名前は notebook 全体で一意にする。`_` 始まりの変数はセルローカルになり
  他セルから参照できない。
- セル出力: `<... object at 0x...>` のような生 repr や tqdm の進捗を残さない。
  値を確認するなら f-string で意味のある形にする。
- 可視化: WebGL 系ウィジェット（`ngsolve.webgui.Draw` など）は静的 HTML で描画されず
  文字列だけが残る。matplotlib で描き直す。
- 数値計算: `solve_ivp` は `sol.success` を必ず検査する。失敗しても例外は出ず、
  `dense_output` の外挿値がそのまま図になる。
- アニメーション: 見せたい現象が一巡するまで積分する。GIF のコマ間隔は 1/100 秒単位に
  切り捨てられるため、`interval` は 50/80/100 のような値にし、保存後に実測して報告する。
  カラースケールと軸範囲は全コマで固定する。

## 安全境界（Always / Ask first / Never）

- Always:
  - 変更後に検証コマンドを実行し、その出力を根拠として示す。
  - 実行していないチェックの結果を推測で書かない。
- Ask first:
  - 依存関係の追加・変更（`pyproject.toml` / `uv.lock`）
  - CI 設定（`.github/workflows/`）の変更
  - 全 notebook の再ビルド（GPU 込みで 30 分以上かかる）
  - notebook のファイル名変更（`site/` の旧 HTML 削除と再生成を伴う）
- Never:
  - 秘密情報（API キー、`.env`、トークン）をコミットしない。秘密情報を必要とする
    notebook は 1 つも無い。必要になった時点で設計を見直す。
  - `site/` を手で編集しない（必ず `build_site.py` で生成する）。
  - notebook を変更したまま `site/` を更新せずにコミットしない。
  - 破壊的操作（履歴改変、force push、大量削除）を無断実行しない。

## コミット

- Conventional Commits 形式。件名は 72 文字以内。本文は既存コミットに合わせて英語で書く。
- 本文は `- Motivation:` / `- Change:` / `- Impact:` の 3 点。
- 数値を書くのは実測したときだけ。見込みや推定は書かない。
- 対象ファイルをパスで明示してステージする（`git add .` / `git add -A` は使わない）。

## 不明点があるとき

- 推測で進めず確認する。`TODO: 要確認` を明示して報告する。
