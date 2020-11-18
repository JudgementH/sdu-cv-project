#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2020/11/17 19:44

'进行倍数缩放'

__author__ = 'Judgement'

import numpy as np
import cv2


def scale(input: np.ndarray, sx: float, sy: float) -> np.ndarray:
    height, width, channel = input.shape
    output_h, output_w = int(height * sy), int(width * sx)
    output = np.zeros((output_h, output_w, channel), dtype=np.uint8)

    for j in range(output_h):
        for i in range(output_w):
            old_i, old_j = i / sx, j / sy
            old_x1, old_y1 = int(old_i), int(old_j)
            old_x2, old_y2 = old_x1 + 1, old_y1 + 1
            lam_x, lam_y = old_i - old_x1, old_j - old_y1
            if old_x1 == width - 1 or old_y1 == height - 1:
                output[j, i, :] = input[old_y1, old_x1, :]
            else:
                output[j, i, :] = (1 - lam_x) * (1 - lam_y) * input[old_y1, old_x1, :] + \
                                  lam_x * (1 - lam_y) * input[old_y1, old_x2, :] + \
                                  (1 - lam_x) * (lam_y) * input[old_y2, old_x1, :] + \
                                  lam_x * lam_y * input[old_y2, old_x2, :]
    return output


if __name__ == '__main__':
    url = './lab2.png'
    image = cv2.imread(url)
    cv2.imshow('origin', image)
    output = scale(image, 3.6, 3.6)
    cv2.imshow('out', output)
    cv2.waitKey()
    cv2.destroyAllWindows()
