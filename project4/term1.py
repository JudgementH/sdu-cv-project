#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2020/11/18 23:53

'利用传统视觉方法人脸识别'

__author__ = 'Judgement'

import cv2
import numpy as np

import sys

sys.setrecursionlimit(2 ** 31 - 1)


def get_skin(image: np.ndarray) -> np.ndarray:
    '''
    参考文章
    https://blog.csdn.net/qq_22527639/article/details/81501565
    '''
    h, w, c = image.shape
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCR_CB)
    y, cr, cb = (ycrcb[:, :, i] for i in range(3))
    res = np.zeros(shape=cr.shape, dtype=np.uint8)
    for i in range(h):
        for j in range(w):
            if 133 < cr[i, j] < 173 and 77 < cb[i, j] < 127 and y[i, j] >= 70:
                res[i, j] = 255
            else:
                res[i, j] = 0
    return res


def get_connected_component(image: np.ndarray) -> list:
    '''
    输入二值图像0、255 进行连通分支分解
    存放连通分支位置 二维数组存放每个连通分支的 上，右，下，左坐标，
    '''
    height, weight = image.shape
    copy = image.copy()
    components = []
    for i in range(height):
        for j in range(weight):
            if copy[i, j] == 255:
                max_top, max_right, max_bottom, max_left = label_one_component_bfs(copy, j, i, height, 0, 0, weight)
                components.append([max_top, max_right, max_bottom, max_left])
    return components


def label_one_component_dfs(image, x, y, max_top, max_right, max_bottom, max_left):
    height, weight = image.shape
    top = max(y - 1, 0)
    bottom = min(y + 1, height - 1)
    left = max(x - 1, 0)
    right = min(x + 1, weight - 1)

    max_top_ = min(y, max_top)
    max_right_ = max(x, max_right)
    max_bottom_ = max(y, max_bottom)
    max_left_ = min(x, max_left)
    image[y, x] = 160

    if 255 not in image[top: bottom + 1, left: right + 1]:
        return max_top_, max_right_, max_bottom_, max_left_
    for i in range(top, bottom + 1):
        for j in range(left, right + 1):
            if image[i, j] == 255:
                max_top_, max_right_, max_bottom_, max_left_ = label_one_component(image, j, i, max_top_, max_right_,
                                                                                   max_bottom_, max_left_)
    return max_top_, max_right_, max_bottom_, max_left_


def label_one_component_bfs(image, x, y, max_top, max_right, max_bottom, max_left):
    height, weight = image.shape
    queue = []
    queue.append([x, y])
    max_top_ = min(y, max_top)
    max_right_ = max(x, max_right)
    max_bottom_ = max(y, max_bottom)
    max_left_ = min(x, max_left)
    while len(queue) != 0:
        item = queue.pop(0)
        x, y = item[0], item[1]
        image[y, x] = 150

        top = max(y - 1, 0)
        bottom = min(y + 1, height - 1)
        left = max(x - 1, 0)
        right = min(x + 1, weight - 1)

        max_top_ = min(y, max_top_)
        max_right_ = max(x, max_right_)
        max_bottom_ = max(y, max_bottom_)
        max_left_ = min(x, max_left_)

        for i in range(top, bottom + 1):
            for j in range(left, right + 1):
                if image[i, j] == 255:
                    image[i, j] = 1
                    queue.append([j, i])
    return max_top_, max_right_, max_bottom_, max_left_


def have_eyes(image):
    '''
    输入的是皮肤的灰度图，皮肤为255，其余为0
    '''
    eyes_image = 255 - image
    eyes = get_connected_component(eyes_image)
    count = 0
    for eye in eyes:
        max_top, max_right, max_bottom, max_left = eye
        height_com = max_bottom - max_top
        width_com = max_right - max_left
        w_ratio = width_com / image.shape[1]
        h_ratio = height_com / image.shape[0]
        if 0.05 < width_com / image.shape[1] < 1 / 3 and 1 / 200 < h_ratio < 0.2 and width_com >= height_com:
            count += 1

    if 5 >= count >= 1:
        return True
    else:
        return False


def have_mouth(image):
    h, w, c = image.shape
    b, g, r = (image[:, :, i] for i in range(3))

    return True


def is_face(image):
    h, w = image.shape
    n = np.count_nonzero(image)
    ratio = n / (h * w)
    if 0.4 < ratio < 0.8:
        return True
    else:
        return False


if __name__ == '__main__':
    image = cv2.imread('./Orical1.jpg')
    skin = get_skin(image)
    cv2.imshow('skin', skin)
    components = get_connected_component(skin)
    for component in components:
        max_top, max_right, max_bottom, max_left = component
        height_com = max_bottom - max_top
        width_com = max_right - max_left
        face_reg = skin[max_top:max_bottom + 1, max_left:max_right + 1]
        if width_com != 0 and 0.6 < height_com / width_com < 2 and width_com > image.shape[
            1] * 0.05 and height_com > image.shape[0] * 0.05:  # “三庭五眼 ”规则高度和宽度比例应该在（ 0.6, 2）内”
            if have_eyes(face_reg) and is_face(face_reg):
                image = cv2.rectangle(image, (max_left, max_top), (max_right, max_bottom), (0, 255, 0), 2)

    cv2.imshow('res', image)

    cv2.waitKey()
    cv2.destroyAllWindows()
