#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2020/11/17 20:35

'图像变形'

__author__ = 'Judgement'

import numpy as np
import cv2


def transform(input: np.ndarray) -> np.ndarray:
    height, width, channel = input.shape
    output = input.copy()
    width_half = width / 2
    height_half = height / 2
    for j in range(height):
        for i in range(width):
            x = (i - width_half) / width_half
            y = (j - height_half) / height_half
            r = np.sqrt(x ** 2 + y ** 2)
            theta = (1 - r) ** 2
            if r < 1:
                f_x = np.cos(theta) * x + np.sin(theta) * y
                f_x = int(f_x * width_half + width_half)
                f_y = np.cos(theta) * y - np.sin(theta) * x
                f_y = int(f_y * height_half + height_half)
                output[j, i, :] = input[f_y, f_x, :]
    return output


if __name__ == '__main__':
    url = './lab2.png'
    image = cv2.imread(url)
    cv2.imshow('origin', image)
    output = transform(image)
    cv2.imshow('transform', output)
    cv2.waitKey()
    cv2.destroyAllWindows()
