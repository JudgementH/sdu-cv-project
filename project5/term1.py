#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2020/11/22 17:31

'commit'

__author__ = 'Judgement'

import cv2
import numpy as np
import numba
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d


@numba.jit(nopython=True)
def get_disparity(left, right, shape=(3, 3), d_level=40):
    max_d = d_level
    n, m = shape  # n:height, m:width
    height, width, channel = left.shape
    e = np.zeros(shape=(height, width, max_d))
    for d in range(max_d):
        for i in range(height):
            for j in range(width):

                for x in range(i, i + n + 1):
                    for y in range(j, j + m + 1):
                        x, y = min(x, height - 1), min(y, width - 1)
                        dy = min(y + d, width - 1)
                        for k in range(3):
                            e[i, j, d] += (left[x, dy, k] - right[x, y, k]) ** 2
                e[i, j, d] /= (3 * n * m)

    e_avg = np.zeros(shape=(e.shape))
    for d in range(max_d):
        for i in range(height):
            for j in range(width):

                for x in range(i, i + n + 1):
                    for y in range(j, j + m + 1):
                        x, y = min(x, height - 1), min(y, width - 1)
                        e_avg[i, j, d] += e[x, y, d] / (n * m)

    disparity = np.zeros(shape=(height, width))
    for i in range(height):
        for j in range(width):
            min_ = 2147483647
            for d in range(max_d):
                if min_ > e_avg[i, j, d]:
                    min_ = e_avg[i, j, d]
                    disparity[i, j] = d

    return disparity.astype(np.uint8)


def get_depth_map(image, f=30, T=20):
    '''
    image:视差图可以通过get_disparity得到
    f:两个相机的焦距
    T:两个相机之间距离
    '''
    mid_d = cv2.medianBlur(image, ksize=5)
    depth = mid_d.copy()
    h, w = mid_d.shape
    for i in range(h):
        for j in range(w):
            if mid_d[i, j] != 0:
                depth[i, j] = f * T / mid_d[i, j]
            else:
                depth[i, j] = 0
    return depth


@numba.jit(nopython=True)
def get_disparity_reliable(left, right, disparity, alpha=2., shape=(2, 2)):
    '''
    返回： 可靠的视差图, 可靠度
    image: 视差图d(i,j)
    alpha: 可靠性容忍度，数值越低，可靠度越高
    shape: 获得视差图时的窗口大小和得到不可靠的图片是一致的
    '''

    height, width = disparity.shape
    n, m = shape

    ed = np.zeros(shape=(height, width))
    for i in range(height):
        for j in range(width):

            for x in range(i, i + n + 1):
                for y in range(j, j + m + 1):
                    x, y = min(x, height - 1), min(y, width - 1)
                    dy = min(y + disparity[i, j], width - 1)
                    for k in range(3):
                        ed[i, j] += (left[x, dy, k] - right[x, y, k]) ** 2
            ed[i, j] /= (3 * n * m)
    ve = alpha * np.mean(ed)
    disparity_reliable = disparity.copy()
    sd = 0  # 统计不可靠点数目
    sum_ed = 0
    for i in range(height):
        for j in range(width):
            if ed[i, j] <= ve:
                disparity_reliable[i, j] = disparity[i, j]
                sum_ed += ed[i, j]
            else:
                disparity_reliable[i, j] = 0
                sd += 1

    reliability = 1 / (sd * sum_ed)  # 可靠度

    return disparity_reliable


def show_image(image: np.ndarray, name: str, d_level=40):
    h, w = image.shape
    min_ = np.min(image)
    max_ = np.max(image)
    show = 255 / (max_ - min_) * (image - min_)
    show = show[:h, :w - d_level]
    cv2.imshow(name, show.astype(np.uint8))


if __name__ == '__main__':
    left = cv2.imread('./view1m.png')
    right = cv2.imread('./view5m.png')
    disparity = get_disparity(left, right)
    show_image(disparity, 'disparity')
    disparity_reliable = get_disparity_reliable(left, right, disparity)
    show_image(disparity_reliable, 'disparity_reliable')
    depth = get_depth_map(disparity_reliable)
    show_image(depth, 'depth')

    # plt画深度图
    h, w, c = left.shape

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title('depth_map')

    x, y = np.meshgrid(range(h), range(w))
    #
    # bottom = np.zeros_like(x)
    # x_space = y_space = 1

    # ax.bar3d(x, y, 0, x_space, y_space, z, shade=True)
    ax.plot_surface(x, y, disparity_reliable.T, cmap='bone')
    ax.set_xlim(0, 200)
    ax.set_ylim(0, 200)
    ax.set_zlim(0, 40)

    plt.show()
    cv2.waitKey()
    cv2.destroyAllWindows()
