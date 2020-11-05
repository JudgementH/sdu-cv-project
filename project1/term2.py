#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2020/10/28 9:09

'实验一的2'

__author__ = 'Judgement'

import cv2
import numpy as np

img1 = cv2.imread(r"F:\Image\1-2\a.png", cv2.IMREAD_UNCHANGED)
img2 = cv2.imread(r"F:\Image\1-2\bg.png")
alpha = img1[:, :, 3]
alpha = cv2.merge((alpha, alpha, alpha))
alpha = alpha / 255
mask = 1 - alpha

sec = cv2.multiply(img1[:, :, :3].astype(float), alpha).astype(np.uint8)
res = cv2.multiply(img2.astype(float), mask).astype(np.uint8)
res2 = cv2.add(res, sec)
cv2.imshow('res2', res2)
cv2.waitKey()
cv2.destroyAllWindows()
