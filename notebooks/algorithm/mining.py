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
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    [Block 395791](https://explorer.btc.com/btc/block/395791)
    と
    [Block 395790](https://explorer.btc.com/btc/block/395790)
    の内容の検証

    - [参考サイト1](https://en.bitcoin.it/wiki/Block_hashing_algorithm)
    - [参考サイト2](https://qiita.com/ryskchy/items/a862139e9521942248fa#%E5%8F%82%E8%80%83%E6%96%87%E7%8C%AE)
    """)
    return


@app.cell
def _():
    # ブロックのバージョン
    version = 0x4
    # 前のブロックのハッシュ値
    prev_block = 0x0000000000000000005629ef6b683f8f6301c7e6f8e796e7c58702a079db14e8
    # マークルルート
    markle_root = 0xefb8011cb97b5f1599b2e18f200188f1b8207da2884392672f92ac7985534eeb
    timestamp = "2016-01-30 13:23:09"
    bits = 0x180928f0
    nonce = 0x56591fc2
    return bits, markle_root, nonce, prev_block, timestamp, version


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    フォーマットを調整
    """)
    return


@app.cell
def _(markle_root, prev_block, version):
    version_h = version.to_bytes(4, "little").hex()
    print(version_h)

    prev_block_h = prev_block.to_bytes(32, "little").hex()
    print(prev_block_h)

    markle_root_h = markle_root.to_bytes(32, "little").hex()
    print(markle_root_h)
    return markle_root_h, prev_block_h, version_h


@app.cell
def _(timestamp):
    from datetime import datetime

    timestamp_s = int((datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")-datetime(1970,1,1)).total_seconds())
    timestamp_h = timestamp_s.to_bytes(4, "little").hex()
    print(timestamp_h)
    return (timestamp_h,)


@app.cell
def _(bits, nonce):
    bits_h = bits.to_bytes(4, "little").hex()
    print(bits_h)

    nonce_h = nonce.to_bytes(4, "little").hex()
    print(nonce_h)
    return bits_h, nonce_h


@app.cell
def _(bits_h, markle_root_h, nonce_h, prev_block_h, timestamp_h, version_h):
    from binascii import hexlify, unhexlify
    from hashlib import sha256

    header_hex = (version_h + prev_block_h + markle_root_h + timestamp_h + bits_h + nonce_h)
    print(header_hex)

    header_bin = unhexlify(header_hex)
    hash = sha256(sha256(header_bin).digest()).digest()

    print(hexlify(hash).decode("utf-8"))
    print(hexlify(hash[::-1]).decode("utf-8"))
    return


if __name__ == "__main__":
    app.run()
