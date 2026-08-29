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
    data_dir = repo_root / "data" / "cv" / "opencv_calibration"
    data_dir.mkdir(parents=True, exist_ok=True)

    opencv_ref = "4.11.0"
    image_names = [f"left{index:02d}.jpg" for index in range(1, 15) if index != 10]
    base_url = f"https://raw.githubusercontent.com/opencv/opencv/{opencv_ref}/samples/data"

    return base_url, cv2, data_dir, image_names, mo, np, plt, requests


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
    # OpenCV カメラキャリブレーション

    チェスボードを撮影した画像からカメラの内部パラメータと歪み係数を推定し、
    歪み補正までを一通り行う。サンプル画像は固定 URL から取得して `data/` に
    キャッシュするので、実行のたびにダウンロードし直すことはない。
    """)
    return


@app.cell
def _(base_url, ensure_download, image_names):
    calibration_images = [
        ensure_download(image_name, f"{base_url}/{image_name}")
        for image_name in image_names
    ]
    return (calibration_images,)


@app.cell
def _(np):
    object_points_template = np.zeros((6 * 7, 3), np.float32)
    object_points_template[:, :2] = np.mgrid[0:7, 0:6].T.reshape(-1, 2)
    return (object_points_template,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## チェスボードコーナー検出
    """)
    return


@app.cell
def _(calibration_images, cv2, object_points_template, plt):
    corner_refinement = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        30,
        0.001,
    )
    image_points = []
    object_points = []
    annotated_images = []
    annotated_names = []
    image_size = None

    for image_path in calibration_images:
        image_bgr = cv2.imread(str(image_path))
        if image_bgr is None:
            raise RuntimeError(f"Failed to load calibration image: {image_path}")
        image_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(image_gray, (7, 6), None)
        if not found:
            continue
        refined_corners = cv2.cornerSubPix(image_gray, corners, (11, 11), (-1, -1), corner_refinement)
        object_points.append(object_points_template.copy())
        image_points.append(refined_corners)
        image_size = image_gray.shape[::-1]
        annotated_bgr = cv2.drawChessboardCorners(image_bgr.copy(), (7, 6), refined_corners, found)
        annotated_images.append(cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB))
        annotated_names.append(image_path.name)

    if not image_points or image_size is None:
        raise RuntimeError("Chessboard corners were not detected in the calibration set.")

    cols = 3
    rows = (len(annotated_images) + cols - 1) // cols
    figure_detect, axes_detect = plt.subplots(rows, cols, figsize=(12, 4 * rows))
    axes_flat = axes_detect.ravel() if hasattr(axes_detect, "ravel") else [axes_detect]
    for _gallery_axis, image_rgb, image_name in zip(axes_flat, annotated_images, annotated_names, strict=False):
        _gallery_axis.imshow(image_rgb)
        _gallery_axis.set_title(image_name)
        _gallery_axis.axis("off")
    for _unused_gallery_axis in axes_flat[len(annotated_images):]:
        _unused_gallery_axis.axis("off")
    figure_detect.tight_layout()
    figure_detect
    return image_points, image_size, object_points


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## カメラ行列の推定
    """)
    return


@app.cell
def _(cv2, image_points, image_size, object_points):
    _, camera_matrix, distortion_coeffs, rotation_vectors, translation_vectors = cv2.calibrateCamera(
        object_points,
        image_points,
        image_size,
        None,
        None,
    )
    print(camera_matrix)
    return camera_matrix, distortion_coeffs, rotation_vectors, translation_vectors


@app.cell
def _(calibration_images, camera_matrix, cv2, distortion_coeffs):
    sample_image = cv2.imread(str(calibration_images[-1]))
    if sample_image is None:
        raise RuntimeError(f"Failed to load sample image: {calibration_images[-1]}")
    image_height, image_width = sample_image.shape[:2]
    optimal_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
        camera_matrix,
        distortion_coeffs,
        (image_width, image_height),
        1,
        (image_width, image_height),
    )
    print(optimal_camera_matrix)
    return optimal_camera_matrix, roi, sample_image


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 歪み補正の比較
    """)
    return


@app.cell
def _(camera_matrix, cv2, distortion_coeffs, optimal_camera_matrix, plt, roi, sample_image):
    undistorted_direct = cv2.undistort(sample_image, camera_matrix, distortion_coeffs, None, optimal_camera_matrix)
    roi_x, roi_y, roi_w, roi_h = roi
    undistorted_direct = undistorted_direct[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]

    map_x, map_y = cv2.initUndistortRectifyMap(
        camera_matrix,
        distortion_coeffs,
        None,
        optimal_camera_matrix,
        (sample_image.shape[1], sample_image.shape[0]),
        cv2.CV_32FC1,
    )
    undistorted_remap = cv2.remap(sample_image, map_x, map_y, cv2.INTER_LINEAR)
    undistorted_remap = undistorted_remap[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]

    figure_undistort, axes_undistort = plt.subplots(1, 3, figsize=(15, 4))
    axes_undistort[0].imshow(cv2.cvtColor(sample_image, cv2.COLOR_BGR2RGB))
    axes_undistort[0].set_title("original")
    axes_undistort[1].imshow(cv2.cvtColor(undistorted_direct, cv2.COLOR_BGR2RGB))
    axes_undistort[1].set_title("cv2.undistort")
    axes_undistort[2].imshow(cv2.cvtColor(undistorted_remap, cv2.COLOR_BGR2RGB))
    axes_undistort[2].set_title("cv2.remap")
    for _undistort_axis in axes_undistort:
        _undistort_axis.axis("off")
    figure_undistort.tight_layout()
    figure_undistort
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## キャリブレーション誤差
    """)
    return


@app.cell
def _(camera_matrix, cv2, distortion_coeffs, image_points, object_points, rotation_vectors, translation_vectors):
    mean_error = 0.0
    for index in range(len(object_points)):
        reprojected, _ = cv2.projectPoints(
            object_points[index],
            rotation_vectors[index],
            translation_vectors[index],
            camera_matrix,
            distortion_coeffs,
        )
        error = cv2.norm(image_points[index], reprojected, cv2.NORM_L2) / len(reprojected)
        mean_error += error
    print(f"total error: {mean_error / len(object_points):.6f}")
    return


if __name__ == "__main__":
    app.run()
