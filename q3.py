from pathlib import Path
import json
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from render import as_homogeneous, homogenize

root = Path("./data")


def compute_homography(pts1, pts2):

    n = pts1.shape[0]
    # your code here
    A = []
    for i in range(n):
        p1, p2 = pts1[i]
        p3, p4 = pts2[i]
        A.append([p1, p2, 1, 0, 0, 0, -p3 * p1, -p3 * p2, -p3])
        A.append([0, 0, 0, p1, p2, 1, -p4 * p1, -p4 * p2, -p4])
        # compress A down into a single 3x3 matrix- that's our homography transformation.

    A = np.array(A)
    _, _, H = np.linalg.svd(A)
    # H1to2
    np.reshape()
    return H1to2


def load_corners():
    with open(root / "corners.json", "r") as f:
        corners = json.load(f)
        print(corners)

    points = np.array([
        corners['LL'],
        corners['LR'],
        corners['UR'],
        corners['UL']
    ])
    return points


def warp_image(img, H, target_w, target_h):
    # compared to opencv's warp, this routine does not fill in holes
    h, w = img.shape[:2]
    orig_ys, orig_xs = np.meshgrid(range(h), range(w), indexing="ij")
    orig_ys = orig_ys.reshape(-1)
    orig_xs = orig_xs.reshape(-1)

    coords = np.stack([
        orig_xs, orig_ys, np.ones(h * w)
    ], axis=0)
    coords = H @ coords
    coords = coords[:2] / coords[2]
    coords = np.rint(coords).astype(int)  # round to integers
    xs, ys = coords

    canvas = np.zeros((target_h, target_w, 3))  # background color black

    # throw out those beyond image boundary
    inbound_mask = (ys < target_h) & (ys >= 0) & (xs < target_w) & (xs >= 0)
    ys, xs = ys[inbound_mask], xs[inbound_mask]
    orig_ys, orig_xs = orig_ys[inbound_mask], orig_xs[inbound_mask]

    canvas[ys, xs] = img[orig_ys, orig_xs]
    return canvas


def rectify_image():
    img = np.array(Image.open(root / "stadium.png")) / 255.

    pts1 = load_corners()
    aspect_ratio = 1 / 1.88

    new_height = 500
    new_width = new_height * aspect_ratio

    tl_x = 300
    tl_y = 500
    pts2 = np.array([
        [tl_x, tl_y],
        [tl_x + new_width, tl_y],
        [tl_x + new_width, tl_y + new_height],
        [tl_x, tl_y + new_height],
    ])
    pts2 = np.rint(pts2)

    H = compute_homography(pts1, pts2)
    what = as_homogeneous(pts1) @ H.T
    what = np.rint(homogenize(what)).astype(int)
    assert (what[:, :2] == pts2).all()

    warped = warp_image(img, H, 1000, 1200)

    fig, axes = plt.subplots(1, 2)
    axes[0].imshow(img)
    axes[0].set_title("original")
    axes[1].imshow(warped)
    axes[1].set_title("warped")
    plt.show()


def show_field_corners():
    img = np.array(Image.open(root / "stadium.png"))
    points = load_corners()

    segments = []
    for i in range(4):
        start, end = points[i], points[(i + 1) % 4]
        segments.append([start, end])

    plt.imshow(img)
    line_segments = LineCollection(segments, linestyle='solid', color="cyan")
    ax = plt.gca()
    ax.add_collection(line_segments)
    plt.show()


def main():
    pass
    #show_field_corners()
    rectify_image()


if __name__ == "__main__":
    main()
