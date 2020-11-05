#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2020/10/28 10:00

'实验2.2'

__author__ = 'Judgement'

import cv2

img = cv2.imread(r"F:\Image\a.jpg")
sigma = 3
kernal = cv2.getGaborKernel(6 * sigma, sigma)
print(kernal.shape)
cv2.imshow('img', img)
cv2.waitKey()
