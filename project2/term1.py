#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2020/11/4 21:53

'高斯滤波'

__author__ = 'Judgement'

from time import time

import cv2
import numpy as np


def conv2D(image, kernel):
    image_height, image_width = image.shape

    kernel_height, kernel_width = kernel.shape

    stepX = image_width - kernel_width // 2 * 2
    stepY = image_height - kernel_height // 2 * 2
    result = image
    for row in range(0, stepY):
        for col in range(0, stepX):
            part = image[row:row + kernel_height, col:col + kernel_width]
            part = part * kernel
            sum = np.sum(part)
            result[row + kernel_height // 2, col + kernel_width // 2] = sum
    return result


def gauss_blur(image, sigma=1):
    size = (6 * sigma - 1) // 2 * 2 + 1
    if size % 2 == 0:
        raise Exception("卷积核必须是奇数")
    a, b, c = image[:, :, 0], image[:, :, 1], image[:, :, 2]

    # 构造高斯卷积核
    gauss_kernel = np.zeros((size, size))
    s = size // 2
    for y in range(-s, s + 1):
        for x in range(-s, s + 1):
            gauss_kernel[y + s][x + s] = np.exp(
                -(x ** 2 + y ** 2) / 2 * sigma ** 2) / (2 * np.pi * sigma ** 2)
    gauss_kernel = gauss_kernel / np.sum(gauss_kernel)

    # 进行卷积
    a_ = conv2D(a, gauss_kernel)
    b_ = conv2D(b, gauss_kernel)
    c_ = conv2D(c, gauss_kernel)

    # 合并
    res = cv2.merge((a_, b_, c_))
    return res


def gauss_blur_1D(image, sigma=1):
    size = (6 * sigma - 1) // 2 * 2 + 1
    if size // 2 == 1:
        raise Exception("卷积核必须是奇数")
    a, b, c = image[:, :, 0], image[:, :, 1], image[:, :, 2]

    # 构造高斯卷积核
    gauss_kernel = np.zeros((1, size))
    s = size // 2
    for y in range(-s, s + 1):
        gauss_kernel[0][y + s] = np.exp(-(y ** 2) / (2 * sigma ** 2)) / (sigma * np.sqrt(2 * np.pi))

    gauss_kernel = gauss_kernel / np.sum(gauss_kernel)

    # 进行卷积
    a_ = conv2D(a, gauss_kernel)
    a_ = conv2D(a_, gauss_kernel.T)
    b_ = conv2D(b, gauss_kernel)
    b_ = conv2D(b_, gauss_kernel.T)
    c_ = conv2D(c, gauss_kernel)
    c_ = conv2D(c_, gauss_kernel.T)

    # 合并
    res = cv2.merge((a_, b_, c_))
    return res


if __name__ == '__main__':
    origin = r"F:\Image\2-1\a.jpg"
    img = cv2.imread(origin)
    cv2.imshow('origin',img)

    # 二维
    time1 = time()
    gauss = gauss_blur(img, 1)
    print("耗费时间:{time}s".format(time=(time() - time1)))
    cv2.imshow('gauss2D', gauss)

    time2 = time()
    gauss1D = gauss_blur_1D(img, 1)
    print("耗费时间:{time}s".format(time=(time() - time2)))
    cv2.imshow('gauss1D', gauss1D)

    cv2.waitKey()

