import marimo

__generated_with = "0.24.0"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # マイニングの仕組み

    Bitcoin のブロックハッシュを、ブロックエクスプローラで公開されている
    ヘッダ情報だけから再計算する。実際に採掘された
    [Block 395791](https://explorer.btc.com/btc/block/395791)
    （直前は
    [Block 395790](https://explorer.btc.com/btc/block/395790)）
    を題材に、次の 3 点を順に確認する。

    1. ヘッダ 6 フィールドを連結してダブル SHA-256 を取ると、公開されている
       ブロックハッシュと一致すること
    2. そのハッシュが `bits` から復元される目標値を下回っており、
       プルーフ・オブ・ワークの条件を満たしていること
    3. 採掘とは、条件を満たす `nonce` を総当たりで探す作業であること

    参考:
    [Block hashing algorithm](https://en.bitcoin.it/wiki/Block_hashing_algorithm) /
    [ビットコインのブロックのハッシュ値を計算してみる](https://qiita.com/ryskchy/items/a862139e9521942248fa#%E5%8F%82%E8%80%83%E6%96%87%E7%8C%AE)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## ブロックヘッダの構造

    ブロックヘッダは次の 6 フィールドを連結した固定長 80 バイトである。
    ブロック本体（取引の集合）はハッシュの入力に直接は含まれず、
    マークルルートを経由して反映される。

    | フィールド | サイズ | 内容 |
    | --- | --- | --- |
    | `version` | 4 バイト | ブロックのバージョン番号 |
    | `prev_block` | 32 バイト | 直前のブロックのハッシュ値 |
    | `merkle_root` | 32 バイト | ブロック内の全取引を二分木でまとめた根 |
    | `timestamp` | 4 バイト | 採掘時刻（UTC の Unix 秒） |
    | `bits` | 4 バイト | 目標値の圧縮表現 |
    | `nonce` | 4 バイト | 採掘者が総当たりで探した値 |

    ブロックハッシュは、このヘッダに SHA-256 を 2 回適用した値である。

    $$
    H = \mathrm{SHA256}(\mathrm{SHA256}(\text{header}))
    $$

    `prev_block` がヘッダに含まれるため、過去のブロックを 1 つでも書き換えると
    それ以降のハッシュがすべて変わる。これが連鎖の改竄耐性の根拠になっている。
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 入力データ

    Block 395791 のヘッダ 6 フィールドを、エクスプローラの表示どおりに書き写す。
    整数リテラルはすべて人間が読む向き（ビッグエンディアン）であり、
    ヘッダのバイト列そのものではない。`timestamp` だけは可読性のため文字列で
    保持し、後段で Unix 秒へ変換する。
    """)
    return


@app.cell
def _():
    version = 0x4
    prev_block = 0x0000000000000000005629EF6B683F8F6301C7E6F8E796E7C58702A079DB14E8
    merkle_root = 0xEFB8011CB97B5F1599B2E18F200188F1B8207DA2884392672F92AC7985534EEB
    timestamp = "2016-01-30 13:23:09"
    bits = 0x180928F0
    nonce = 0x56591FC2

    print(f"version={version}")
    print(f"timestamp={timestamp} (UTC)")
    print(f"bits=0x{bits:08x}, nonce=0x{nonce:08x} ({nonce})")
    return bits, merkle_root, nonce, prev_block, timestamp, version


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## フォーマットの調整

    ### エンディアンの反転

    ヘッダの各フィールドはリトルエンディアンのバイト列として連結される。
    一方、エクスプローラが表示するハッシュ値はビッグエンディアンなので、
    `int.to_bytes(サイズ, "little")` でバイト順を反転させてから 16 進文字列に
    直す。`version` は 4 バイト、`prev_block` と `merkle_root` は 32 バイトである。
    """)
    return


@app.cell
def _(merkle_root, prev_block, version):
    version_h = version.to_bytes(4, "little").hex()
    prev_block_h = prev_block.to_bytes(32, "little").hex()
    merkle_root_h = merkle_root.to_bytes(32, "little").hex()

    print(f"version_h={version_h}")
    print(f"prev_block_h={prev_block_h}")
    print(f"merkle_root_h={merkle_root_h}")
    return merkle_root_h, prev_block_h, version_h


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### タイムスタンプの数値化

    エクスプローラの表示時刻は UTC である。`strptime` が返すのはタイムゾーン情報を
    持たない naive な `datetime` なので、`datetime(1970, 1, 1)` との差を取ることで
    そのまま UTC 基準の Unix 秒として解釈する。得られた秒数を 4 バイトの
    リトルエンディアンに直す。
    """)
    return


@app.cell
def _(timestamp):
    from datetime import datetime

    timestamp_s = int(
        (datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S") - datetime(1970, 1, 1)).total_seconds()
    )
    timestamp_h = timestamp_s.to_bytes(4, "little").hex()

    print(f"timestamp_s={timestamp_s}")
    print(f"timestamp_h={timestamp_h}")
    return (timestamp_h,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### bits と nonce

    `bits` と `nonce` はいずれも 4 バイトである。他のフィールドと同様に
    リトルエンディアンへ反転する。
    """)
    return


@app.cell
def _(bits, nonce):
    bits_h = bits.to_bytes(4, "little").hex()
    nonce_h = nonce.to_bytes(4, "little").hex()

    print(f"bits_h={bits_h}")
    print(f"nonce_h={nonce_h}")
    return bits_h, nonce_h


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## ブロックハッシュの再計算

    6 フィールドを定義順に連結すると、80 バイト（16 進で 160 文字）のヘッダが
    できる。これをバイト列へ戻してダブル SHA-256 を取る。

    ダイジェストはリトルエンディアンで並んでいるため、エクスプローラの表示と
    比べるにはバイト順を反転（`[::-1]`）する必要がある。
    """)
    return


@app.cell
def _(bits_h, merkle_root_h, nonce_h, prev_block_h, timestamp_h, version_h):
    from hashlib import sha256

    header_hex = version_h + prev_block_h + merkle_root_h + timestamp_h + bits_h + nonce_h
    header_bin = bytes.fromhex(header_hex)

    digest = sha256(sha256(header_bin).digest()).digest()
    block_hash = digest[::-1].hex()

    print(f"header_hex={header_hex}")
    print(f"header size={len(header_bin)} bytes")
    print(f"digest (little endian)={digest.hex()}")
    print(f"block hash (big endian)={block_hash}")
    return block_hash, header_bin, sha256


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    エクスプローラが公開している Block 395791 のハッシュ値と突き合わせ、
    一致を機械的に確かめる。
    """)
    return


@app.cell
def _(block_hash):
    expected_hash = "000000000000000003a0343cc001d21b97d15e97e665b68c790b98c871cf0731"

    print(f"expected={expected_hash}")
    print(f"computed={block_hash}")
    print(f"match={block_hash == expected_hash}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 目標値と難易度

    ブロックハッシュの先頭に 0 が並ぶのは偶然ではなく、目標値（target）を
    下回るハッシュが見つかるまで `nonce` を試し続けた結果である。目標値は
    `bits` に圧縮されており、上位 1 バイトを指数 $e$、下位 3 バイトを係数 $c$
    として次のように復元する。

    $$
    \text{target} = c \times 256^{e - 3}
    $$

    採掘の成功条件は、ブロックハッシュをビッグエンディアンの整数と見たときに
    次が成り立つことである。

    $$
    \mathrm{int}(H) < \text{target}
    $$

    難易度は、Bitcoin で許される最も緩い目標値
    $\text{target}_1 = \mathtt{0xffff} \times 256^{\mathtt{0x1d} - 3}$
    との比で表す。

    $$
    \text{difficulty} = \frac{\text{target}_1}{\text{target}}
    $$

    値が大きいほど条件が厳しく、必要な試行回数の期待値も比例して増える。
    """)
    return


@app.cell
def _(bits, block_hash):
    exponent = bits >> 24
    coefficient = bits & 0xFFFFFF
    target = coefficient * 256 ** (exponent - 3)
    target_1 = 0xFFFF * 256 ** (0x1D - 3)

    print(f"exponent=0x{exponent:02x}, coefficient=0x{coefficient:06x}")
    print(f"target ={target:064x}")
    print(f"hash   ={block_hash}")
    print(f"hash < target: {int(block_hash, 16) < target}")
    print(f"difficulty={target_1 / target:,.2f}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 採掘の実演

    採掘は、ヘッダの先頭 76 バイトを固定したまま末尾 4 バイトの `nonce` を
    変えながらダブル SHA-256 を計算し、目標値を下回る値を探す作業である。

    Block 395791 の難易度では期待試行回数が桁違いに大きく、そのままでは
    再現できない。ここでは条件を「ハッシュの先頭 22 ビットが 0」まで緩めて
    探索の過程だけを再現する。期待試行回数は $2^{22} \simeq 4.2 \times 10^6$ 回である。

    実物の採掘との違いは条件の厳しさだけで、手続きは同じである。`nonce` の
    4 バイトを使い切った場合は、`timestamp` やコインベース取引を変えて
    マークルルートを更新し、別の探索空間をやり直すことになる。
    """)
    return


@app.cell
def _(header_bin, sha256):
    demo_bits = 22
    demo_target = 1 << (256 - demo_bits)
    demo_prefix = header_bin[:76]

    demo_nonce = None
    demo_hash = None
    for candidate in range(1 << 25):
        candidate_digest = sha256(
            sha256(demo_prefix + candidate.to_bytes(4, "little")).digest()
        ).digest()
        if int.from_bytes(candidate_digest[::-1], "big") < demo_target:
            demo_nonce = candidate
            demo_hash = candidate_digest[::-1].hex()
            break

    print(f"required leading zero bits={demo_bits}")
    if demo_nonce is None:
        print("not found within the search range")
    else:
        print(f"found nonce={demo_nonce} (0x{demo_nonce:08x})")
        print(f"hash={demo_hash}")
        print(f"trials={demo_nonce + 1:,}")
    return


if __name__ == "__main__":
    app.run()
