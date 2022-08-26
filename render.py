import numpy as np
from numpy import cross, tan
from numpy.linalg import norm, inv


def normalize(v):
    return v / norm(v)


def compute_extrinsics(eye, front, up):
    # pay attn to whether you are dealing with column vector / row vector
    # a numpy vector by convention is a row vector.
    # this is a little tricky; we write it out for you. read it and understand
    z = normalize(-1 * front)
    x = normalize(cross(up, z))
    y = normalize(cross(z, x))

    # turn row vectors into column vectors. but what do we do after this step?
    x = np.reshape(x, (-1, 1))
    y = np.reshape(y, (-1, 1))
    z = np.reshape(z, (-1, 1))
    eye = np.reshape(eye, (-1, 1))


    # your code here
    cam_2_world = np.block([
        [x, y, z, eye],
        [0, 0, 0, 1]
    ])

    return inv(cam_2_world)


def compute_intrinsics(aspect_ratio, fov, img_height_in_pix):
    ndc = compute_proj_to_normalized(aspect_ratio, fov)
    ndc_to_img = compute_normalized_to_img_trans(aspect_ratio, img_height_in_pix)
    intrinsic = ndc_to_img @ ndc
    return intrinsic


def compute_proj_to_normalized(aspect, fov):
    # compared to standard OpenGL NDC intrinsic,
    # this skips the 3rd row treatment on z. hence the name partial_ndc
    # it's incomplete, but enough for now

    # note that fov is in degrees. you need it in rad. COMPLETED
    # note also that your depth is negative. where should you put a -1 somewhere? COMPLETED
    fov_radians = fov * np.pi / 180
    t = tan(fov_radians / 2)

    partial_ndc_intrinsic = np.array([
        [1/(t * aspect), 0, 0, 0],
        [0, 1/t, 0, 0],
        [0, 0, -1, 0] # negative distance is instantiated here.
    ])
    return partial_ndc_intrinsic


def compute_normalized_to_img_trans(aspect, img_height_in_pix):
    img_h = img_height_in_pix
    img_w = img_height_in_pix * aspect

    # your code here
    ndc_to_img = np.array([
        [img_w/2, 0, img_w/2 - 0.5],
        [0, -img_h/2, img_h/2 - 0.5],
        [0, 0, 1]
    ])
    return ndc_to_img


def unproject(K, pixel_coords, depth=1.0):
    """sometimes also referred to as backproject
        pixel_coords: [n, 2] pixel locations
        depth: [n,] or [,] depth value. of a shape that is broadcastable with pix coords
    """
    K = K[0:3, 0:3]

    pixel_coords = as_homogeneous(pixel_coords)
    pixel_coords = pixel_coords.T  # [2+1, n], so that mat mult is on the left

    # this will give points with z = -1, which is exactly what you want since
    # your camera is facing the -ve z axis
    pts = inv(K) @ pixel_coords

    pts = pts * depth  # [3, n] * [n,] broadcast
    pts = pts.T
    pts = as_homogeneous(pts)
    return pts


def rays_from_img(H, W, K, c2w_pose, normalize_dir=True):
    assert c2w_pose[3, 3] == 1.
    n = H * W
    ys, xs = np.meshgrid(range(H), range(W), indexing="ij")
    xy_coords = np.stack([xs, ys], axis=-1).reshape(n, 2)

    ro = c2w_pose[:, -1]
    pts = unproject(K, xy_coords, depth=1)
    pts = pts @ c2w_pose.T
    rd = pts - ro  # equivalently can subtract [0,0,0,1] before pose transform
    rd = rd[:, :3]
    if normalize_dir:
        rd = rd / np.linalg.norm(rd, axis=-1, keepdims=True)
    ro = np.tile(ro[:3], (n, 1))
    return ro, rd


def rays_through_pixels(K, pixel_coords):
    K = K[0:3, 0:3] # intrinsic matrix has shape [3, 3]
    pixel_coords = as_homogeneous(pixel_coords) # shape [n, 3]
    pts = inv(K) @ pixel_coords.T # shape [3, n]
    pts = pts.T # shape [n, 3]
    pts = as_homogeneous(pts) # shape [n, 4]
    rays = pts - np.array([0, 0, 0, 1]) # shape [n, 4]
    return rays


def homogenize(pts):
    # pts: [n, d], where last dim of the d is the diviser
    pts = pts / pts[:, -1].reshape(-1, 1)
    return pts


def as_homogeneous(pts):
    # pts: [n, d]
    n, d = pts.shape
    points = np.ones((n, d + 1))
    points[:, :d] = pts
    return points


def simple_point_render(pts, img_w, img_h, fov, eye, front, up):
    """
    pts has shape [N, 3]
    """
    canvas = np.ones((img_h, img_w, 3))
    pts = as_homogeneous(pts)
    E = compute_extrinsics(eye, front, up)

    aspect = img_w / img_h
    world_to_ndc = compute_proj_to_normalized(aspect, fov)
    ndc_to_img = compute_normalized_to_img_trans(aspect, img_h)

    pts = pts @ E.T # equivalent to pts = (E @ pts.T).T
    pts = pts @ world_to_ndc.T # equivalent to pts = (world_to_ndc @ pts.T).T
    pts = homogenize(pts) # normalize points

    # now, we must ensure no values are beyond [-1, 1]
    mask = ~(np.abs(pts) > 1.0).any(axis=1)
    pts = pts[mask]

    pts = pts @ ndc_to_img.T # equivalent to pts = (ndc_to_img @ pts.T).T

    pts = np.rint(pts).astype(np.int32) # round each pixel to nearest integer
    xs, ys, _ = pts.T
    canvas[ys, xs] = (1, 0, 0) # color each pixel point red

    return canvas
