import numpy as np


def norm_bbox(H, W, element):
    xc, yc, width, height = element
#     xc = x1 + width / 2.
#     yc = y1 + height / 2.
    b = [xc / W, yc / H,
         width / W, height / H]
    return b

