import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    from pathlib import Path

    import cv2
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import requests

    notebook_dir = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
    repo_root = notebook_dir.parents[1]
    data_dir = repo_root / "data" / "cv" / "opencv_tutorial"
    data_dir.mkdir(parents=True, exist_ok=True)

    opencv_ref = "4.11.0"
    asset_urls = {
        "lena.jpg": f"https://raw.githubusercontent.com/opencv/opencv/{opencv_ref}/samples/data/lena.jpg",
        "haarcascade_frontalface_default.xml": f"https://raw.githubusercontent.com/opencv/opencv/{opencv_ref}/data/haarcascades/haarcascade_frontalface_default.xml",
        "haarcascade_eye.xml": f"https://raw.githubusercontent.com/opencv/opencv/{opencv_ref}/data/haarcascades/haarcascade_eye.xml",
    }

    return asset_urls, cv2, data_dir, mo, np, plt, requests


@app.cell
def _(data_dir, requests):
    def ensure_download(file_name: str, url: str):
        destination = data_dir / file_name
        if destination.exists():
            return destination
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        destination.write_bytes(response.content)
        return destination

    return (ensure_download,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # OpenCV チュートリアル

    OpenCV の基本的な画像処理を一通り試す。サンプル画像は固定 URL から取得して
    ローカルにキャッシュするので、実行のたびにダウンロードし直すことはない。
    """)
    return


@app.cell
def _(asset_urls, ensure_download):
    downloaded_assets = {
        file_name: ensure_download(file_name, url)
        for file_name, url in asset_urls.items()
    }
    return (downloaded_assets,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 画像の表示

    OpenCV は BGR、matplotlib は RGB が既定なので、その違いを確認する。
    """)
    return


@app.cell
def _(cv2, downloaded_assets, plt):
    image_bgr = cv2.imread(str(downloaded_assets["lena.jpg"]))
    if image_bgr is None:
        raise RuntimeError("Failed to load lena.jpg")
    figure_bgr, axis_bgr = plt.subplots(figsize=(4, 4))
    axis_bgr.imshow(image_bgr)
    axis_bgr.set_title("BGR のまま matplotlib へ渡した例")
    axis_bgr.axis("off")
    figure_bgr
    return (image_bgr,)


@app.cell
def _(cv2, image_bgr, plt):
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    figure_rgb, axis_rgb = plt.subplots(figsize=(4, 4))
    axis_rgb.imshow(image_rgb)
    axis_rgb.set_title("RGB へ変換後")
    axis_rgb.axis("off")
    figure_rgb
    return image_rgb


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Haar-cascade による顔・眼検出
    """)
    return


@app.cell
def _(cv2, downloaded_assets):
    face_cascade = cv2.CascadeClassifier(str(downloaded_assets["haarcascade_frontalface_default.xml"]))
    eye_cascade = cv2.CascadeClassifier(str(downloaded_assets["haarcascade_eye.xml"]))
    if face_cascade.empty() or eye_cascade.empty():
        raise RuntimeError("Failed to load Haar cascade assets.")
    return eye_cascade, face_cascade


@app.cell
def _(cv2, eye_cascade, face_cascade, image_bgr, plt):
    image_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    annotated_bgr = image_bgr.copy()
    faces = face_cascade.detectMultiScale(image_gray, scaleFactor=1.3, minNeighbors=5)
    for x_coord, y_coord, width, height in faces:
        cv2.rectangle(annotated_bgr, (x_coord, y_coord), (x_coord + width, y_coord + height), (255, 0, 0), 2)
        roi_gray = image_gray[y_coord:y_coord + height, x_coord:x_coord + width]
        roi_bgr = annotated_bgr[y_coord:y_coord + height, x_coord:x_coord + width]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for eye_x, eye_y, eye_w, eye_h in eyes:
            cv2.rectangle(roi_bgr, (eye_x, eye_y), (eye_x + eye_w, eye_y + eye_h), (0, 255, 0), 2)

    figure_detect, axis_detect = plt.subplots(figsize=(4, 4))
    axis_detect.imshow(cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB))
    axis_detect.set_title(f"faces={len(faces)}")
    axis_detect.axis("off")
    figure_detect
    return


if __name__ == "__main__":
    app.run()
