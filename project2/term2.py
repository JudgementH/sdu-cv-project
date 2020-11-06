#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2020/10/28 10:00

'实验2.2'

__author__ = 'Judgement'

from time import time

import cv2
import numpy as np
from numba import jit


@jit(nopython=True)
def get_lead_image(image, scale=0.5):
    image_h, image_d, image_c = image.shape
    small_size = (int(image_h * scale), int(image_d * scale), image_c)
    small = np.zeros(shape=small_size, dtype=np.uint8)
    h, w, _ = small.shape
    large_size = (int(h / scale), int(w / scale), _)
    large = image.copy()
    for y in range(h):
        for x in range(w):
            image_x = int(x / scale)
            image_y = int(y / scale)
            small[y][x] = image[image_y][image_x]
    for y in range(large_size[0]):
        for x in range(large_size[1]):
            small_x = x * scale
            small_y = y * scale
            true_x = int(x * scale)
            true_y = int(y * scale)
            lam_x = small_x - true_x
            lam_y = small_y - true_y
            if true_x + 1 == small.shape[1] or true_y + 1 == small.shape[0]:
                large[y][x] = small[true_y][true_x]
                continue
            large[y][x] = (1 - lam_x) * (1 - lam_y) * small[true_y][true_x] + \
                          lam_x * (1 - lam_y) * small[true_y][true_x + 1] + \
                          (1 - lam_x) * lam_y * small[true_y + 1][true_x] + \
                          lam_x * lam_y * small[true_y + 1][true_x + 1]
    return large.astype(np.uint8)


@jit(nopython=True)
def jbf(image_f, image_g, window_size=5, sigma_f=1, sigma_g=1):
    if window_size % 2 == 0:
        raise Exception("窗口大小必须是奇数")
    window_size = window_size // 2
    h, w, c = image_f.shape
    h_, w_, c_ = image_g.shape
    if h != h_ and w != w_ and c != c_:
        raise Exception("原图像和引导图shape不相等")
    result = image_f.copy()
    step_x = w - 2 * window_size
    step_y = h - 2 * window_size

    d_f = np.ones(shape=(window_size * 2 + 1, window_size * 2 + 1))
    for y in range(-window_size, window_size + 1):
        for x in range(-window_size, window_size + 1):
            d_f[y + window_size][x + window_size] = y ** 2 + x ** 2
    for y in range(window_size, window_size + step_y):
        for x in range(window_size, window_size + step_x):
            # result[y][x] = get_pix_jbf(
            #     image_f[y - window_size: y + window_size + 1, x - window_size: x + window_size + 1, :].astype(float),
            #     image_g[y - window_size: y + window_size + 1, x - window_size: x + window_size + 1, :].astype(float),
            #     sigma_f,
            #     sigma_g)
            for ch in range(c):
                window_f = image_f[y - window_size: y + window_size + 1,
                           x - window_size: x + window_size + 1,
                           ch]
                window_g = image_g[y - window_size: y + window_size + 1,
                           x - window_size: x + window_size + 1,
                           ch]
                result_f = np.exp((-d_f) / (2 * sigma_f ** 2))
                d_g = (window_g - result[y][x][ch]) ** 2
                result_g = np.exp((-d_g) / (2 * sigma_g ** 2))
                result[y][x][ch] = np.sum(result_f * result_g * window_f) / np.sum(result_f * result_g)
    return result.astype(np.uint8)


def get_pix_jbf(window_f, window_g, sigma_f, sigma_g):
    center_x = window_f.shape[0] // 2
    center_y = window_f.shape[0] // 2
    center_color = window_g[center_x][center_y]
    result_matrix = window_f
    for y in range(window_f.shape[0]):
        for x in range(window_f.shape[1]):
            d_f = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
            d_g = np.sqrt((window_g[y][x] - center_color) ** 2)
            result_matrix[y][x] = np.exp((-d_f ** 2) / (2 * sigma_f ** 2)) * np.exp((-d_g ** 2) / (2 * sigma_g ** 2))
    sum_upper = np.ones(window_f.shape[2], dtype=np.float)
    sum_under = np.ones(window_f.shape[2], dtype=np.float)
    for c in range(window_f.shape[2]):
        sum_upper[c] = np.sum(result_matrix[:, :, c] * window_g[:, :, c])
        sum_under[c] = np.sum(result_matrix[:, :, c])
    result = sum_upper / sum_under
    return result.astype(np.uint8)


if __name__ == '__main__':
    img = cv2.imread(r"2_2.png")
    time1 = time()
    lead = get_lead_image(img)
    print("引导图计算时间:{time}s".format(time=(time() - time1)))

    time2 = time()
    after = jbf(img, lead, window_size=21, sigma_f=10, sigma_g=10)
    print("jbf计算时间:{time}s".format(time=(time() - time2)))

    cv2.imshow('lead', lead)
    cv2.imshow('jbf', after)
    cv2.waitKey()
    cv2.destroyAllWindows()
