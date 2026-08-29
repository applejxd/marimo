import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import base64
    import copy
    import importlib
    import sys
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    notebook_dir = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
    repo_root = notebook_dir.parents[1]
    output_dir = repo_root / "data" / "others" / "open3d"
    output_dir.mkdir(parents=True, exist_ok=True)

    original_sys_path = list(sys.path)
    sys.path = [
        entry
        for entry in sys.path
        if Path(entry or ".").resolve() != notebook_dir.resolve()
    ]
    try:
        o3d = importlib.import_module("open3d")
    finally:
        sys.path = original_sys_path

    return base64, copy, mo, np, o3d, output_dir, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Open3D による点群レジストレーション

    合成した 2 つの点群の位置合わせを、Open3D の ICP で行う。GUI を開かずに
    offscreen renderer で静止画を書き出すため、そのまま静的 HTML へ載る。
    offscreen renderer が使えない場合は、理由を表示したうえで matplotlib に切り替える。
    """)
    return


@app.cell
def _(np, o3d):
    rng_points = np.random.default_rng(7)
    theta = np.linspace(0.0, 2.0 * np.pi, 700)
    spiral = np.column_stack([
        0.6 * np.cos(theta),
        0.6 * np.sin(theta),
        np.linspace(-0.8, 0.8, theta.size),
    ])
    plane = rng_points.uniform(-0.8, 0.8, size=(900, 3))
    plane[:, 2] = 0.15 * plane[:, 0] - 0.1 * plane[:, 1]
    sphere = rng_points.normal(size=(700, 3))
    sphere = sphere / np.linalg.norm(sphere, axis=1, keepdims=True) * 0.35
    sphere += np.array([0.9, 0.2, -0.2])

    source_points = np.vstack([spiral, plane, sphere])
    source_points += 0.01 * rng_points.normal(size=source_points.shape)

    ground_truth = np.eye(4)
    ground_truth[:3, :3] = o3d.geometry.get_rotation_matrix_from_xyz((0.35, -0.2, 0.45))
    ground_truth[:3, 3] = np.array([0.35, -0.18, 0.12])

    target_points = source_points @ ground_truth[:3, :3].T + ground_truth[:3, 3]
    target_points += 0.008 * rng_points.normal(size=target_points.shape)

    source_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(source_points))
    target_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(target_points))
    return ground_truth, source_cloud, target_cloud


@app.cell
def _(o3d):
    def preprocess_point_cloud(point_cloud, voxel_size: float):
        point_cloud_down = point_cloud.voxel_down_sample(voxel_size)
        point_cloud_down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2.0, max_nn=30)
        )
        point_cloud_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            point_cloud_down,
            o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5.0, max_nn=100),
        )
        return point_cloud_down, point_cloud_fpfh

    return (preprocess_point_cloud,)


@app.cell
def _(o3d, preprocess_point_cloud, source_cloud, target_cloud):
    voxel_size = 0.08
    source_down, source_fpfh = preprocess_point_cloud(source_cloud, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target_cloud, voxel_size)
    distance_threshold = voxel_size * 1.5
    ransac_result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down,
        target_down,
        source_fpfh,
        target_fpfh,
        True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3,
        [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold),
        ],
        o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999),
    )
    return ransac_result, voxel_size


@app.cell
def _(o3d, ransac_result, source_cloud, target_cloud, voxel_size):
    source_cloud.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2.0, max_nn=30)
    )
    target_cloud.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2.0, max_nn=30)
    )
    icp_result = o3d.pipelines.registration.registration_icp(
        source_cloud,
        target_cloud,
        voxel_size,
        ransac_result.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
    )
    return (icp_result,)


@app.cell
def _(base64, copy, mo, np, o3d, output_dir, plt):
    def render_point_clouds(title: str, filename: str, point_clouds, colors):
        colored_clouds = []
        for point_cloud, color in zip(point_clouds, colors, strict=True):
            painted = copy.deepcopy(point_cloud)
            painted.paint_uniform_color(color)
            colored_clouds.append(painted)

        image_path = output_dir / filename
        backend_name = "open3d-offscreen"
        try:
            renderer = o3d.visualization.rendering.OffscreenRenderer(960, 720)
            renderer.scene.set_background([1.0, 1.0, 1.0, 1.0])
            material = o3d.visualization.rendering.MaterialRecord()
            material.shader = "defaultUnlit"
            material.point_size = 5.0
            for geometry_index, point_cloud in enumerate(colored_clouds):
                renderer.scene.add_geometry(f"pcd-{geometry_index}", point_cloud, material)
            bbox = renderer.scene.bounding_box
            center = bbox.get_center()
            extent = max(float(np.linalg.norm(bbox.get_extent())), 1.0)
            eye = center + np.array([1.3, -1.5, 1.0]) * extent
            renderer.setup_camera(55.0, center, eye, [0.0, 0.0, 1.0])
            image = renderer.render_to_image()
            o3d.io.write_image(str(image_path), image)
            del renderer
        except (OSError, RuntimeError) as exc:
            backend_name = f"matplotlib fallback ({exc.__class__.__name__}: {exc})"
            figure = plt.figure(figsize=(8, 6))
            axis = figure.add_subplot(111, projection="3d")
            for point_cloud, color in zip(colored_clouds, colors, strict=True):
                points = np.asarray(point_cloud.points)
                axis.scatter(points[:, 0], points[:, 1], points[:, 2], s=1.5, c=[color], alpha=0.8)
            axis.set_title(title)
            axis.set_xlabel("x")
            axis.set_ylabel("y")
            axis.set_zlabel("z")
            figure.tight_layout()
            figure.savefig(image_path, dpi=200, bbox_inches="tight")
            plt.close(figure)
        encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
        return mo.md(
            f"""
            ### {title}
            - renderer: `{backend_name}`
            <img src="data:image/png;base64,{encoded}" alt="{title}" />
            """
        )

    return (render_point_clouds,)


@app.cell
def _(render_point_clouds, source_cloud):
    render_point_clouds("ソース点群", "source_cloud.png", [source_cloud], [[0.95, 0.55, 0.15]])
    return


@app.cell
def _(render_point_clouds, source_cloud, target_cloud):
    render_point_clouds(
        "初期位置の 2 点群",
        "initial_alignment.png",
        [source_cloud, target_cloud],
        [[0.95, 0.55, 0.15], [0.15, 0.55, 0.95]],
    )
    return


@app.cell
def _(copy, icp_result, render_point_clouds, source_cloud, target_cloud):
    aligned_source = copy.deepcopy(source_cloud)
    aligned_source.transform(icp_result.transformation)
    render_point_clouds(
        "レジストレーション後",
        "registered_alignment.png",
        [aligned_source, target_cloud],
        [[0.95, 0.55, 0.15], [0.15, 0.55, 0.95]],
    )
    return


@app.cell
def _(ground_truth, icp_result, np):
    print("Ground truth transformation")
    print(np.array2string(ground_truth, precision=4))
    print("Estimated transformation")
    print(np.array2string(icp_result.transformation, precision=4))
    print(f"fitness={icp_result.fitness:.4f}, inlier_rmse={icp_result.inlier_rmse:.4f}")
    return


if __name__ == "__main__":
    app.run()
