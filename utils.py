import cv2


def rotate_about_center(img, deg, flags=cv2.INTER_LINEAR):
    # Rotate
    rows, cols = (img.shape[1], img.shape[0])
    M_rotate = cv2.getRotationMatrix2D((cols / 2.0, rows / 2.0), deg, 1)
    dst = cv2.warpAffine(img, M_rotate, (cols, rows), flags=flags)

    return dst