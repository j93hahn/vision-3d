from pathlib import Path
import numpy as np
import open3d as o3d
from render import (
    compute_intrinsics,
    compute_extrinsics,
    as_homogeneous,
    homogenize
)


def load_horse_pointcloud():
    root = Path("./data")
    fname = str(root / "horse.obj")
    mesh = o3d.io.read_triangle_mesh(fname, True)
    # sample a point cloud from the surface of the mesh
    pcloud = mesh.sample_points_uniformly(500)
    points = np.asarray(pcloud.points)
    return points


def pnp_calibration(pts_2d, pts_3d):
    # your code here
    return P


def main():
    pts_3d = load_horse_pointcloud()
    pts_3d = as_homogeneous(pts_3d)

    img_w, img_h = 600, 450
    fov = 90
    eye = np.array([2, 0, 0.5])
    front = np.array([-1, 0, 0])
    up = np.array([0, 1, 0])
    K = compute_intrinsics(img_w / img_h, fov, img_h)
    E = compute_extrinsics(eye, front, up)
    P = K @ E
    P = P / P[2, 3]

    pts_2d = pts_3d @ P.T
    pts_2d = homogenize(pts_2d)

    pred_P = pnp_calibration(pts_2d, pts_3d)

    assert np.allclose(P, pred_P)


if __name__ == "__main__":
    main()
