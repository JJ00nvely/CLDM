import numpy as np


def norm_bbox(H, W, element):
    xc, yc, width, height = element
    b = [xc / W, yc / H,
         width / W, height / H]
    return b

