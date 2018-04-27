import cv2
import numpy as np
from scipy.special import expit


def rotate_about_center(img, deg, scale=1.0, flags=cv2.INTER_LINEAR):
    # Rotate
    rows, cols = (img.shape[1], img.shape[0])
    M_rotate = cv2.getRotationMatrix2D((cols / 2.0, rows / 2.0), deg, scale)
    dst = cv2.warpAffine(img, M_rotate, (cols, rows), flags=flags)

    return dst


def get_max_positions(hm, scale):
    hm_flattened = np.reshape(hm, (hm.shape[0] * hm.shape[1], hm.shape[2]))
    max_idx = np.argmax(hm_flattened, axis=0)
    max_r, max_c = np.unravel_index(max_idx, hm.shape[0:2])
    joints = np.array((max_c, max_r)).transpose().astype(np.float)
    joints *= scale

    return joints


def draw_result(img, pred, gt=None):

    dst = (img * 255).astype(np.uint8)
    nPts = pred.shape[2]

    joints = get_max_positions(pred, scale = img.shape[0] / pred.shape[0])

    edges = [[0, 1], [1, 2], [2, 6], [6, 3], [3, 4], [4, 5],
             [10, 11], [11, 12], [12, 8], [8, 13], [13, 14], [14, 15],
             [6, 8], [8, 9]]
    for e in edges:
        cv2.line(dst, (int(joints[e[0], 0]), int(joints[e[0], 1])),
                 (int(joints[e[1], 0]), int(joints[e[1], 1])), (0,255,0), 2)
    for j in range(nPts):
        cv2.circle(dst, (int(joints[j, 0]), int(joints[j, 1])), 3, (0,0,255), -1)

    if gt is not None:
        nPts = gt.shape[2]
        joints = get_max_positions(gt, scale=img.shape[0] / gt.shape[0])
        for j in range(nPts):
            cv2.circle(dst, (int(joints[j, 0]), int(joints[j, 1])), 3, (255, 0, 0), -1)

    return dst


def color_heatmap(hm, resize_to=None, apply_sigmoid=False):
    ''' Apply hot colormap for visualization
    :param hm:
    :param resize_to:
    :return:
    '''

    if apply_sigmoid:
        hm = expit(hm)

    if hm.ndim == 3 and hm.shape[2] > 1:
        hm = np.sum(hm, axis=2)
        hm /= np.max(hm)

    hm_uint8 = (np.clip(hm * 255, 0, 255)).astype(np.uint8)
    dst = cv2.applyColorMap(hm_uint8, cv2.COLORMAP_HOT)

    if resize_to is not None:
        dst = cv2.resize(dst, resize_to)
    dst = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)

    return dst
