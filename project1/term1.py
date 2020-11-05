#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2020/11/3 20:49

'实验1的1'
from PIL import Image, ImageSequence
import cv2
import numpy
import pyglet

__author__ = 'Judgement'

img1 = r"F:\Image\1-1\Img1.png"
img4 = r"F:\Image\1-1\Img4.gif"

img1 = pyglet.resource.image('Img1.png')
sprite1 = pyglet.sprite.Sprite(img1)
sprite1.x = 80
sprite1.y = 420
sprite1.scale = 0.4

img2 = pyglet.resource.image('Img2.jpg')
img2.scale = 0.4

img3 = pyglet.resource.image('Img3.bmp')
sprite3 = pyglet.sprite.Sprite(img3)
sprite3.x = 160
sprite3.y = 50
sprite3.scale = 0.7

animation = pyglet.resource.animation('Img4.gif')
sprite4 = pyglet.sprite.Sprite(animation)
sprite4.scale = 0.8
sprite4.x = 600
sprite4.y = 50

window = pyglet.window.Window(1200, 800)

# 背景
background = 1, 1, 1, 1
pyglet.gl.glClearColor(*background)


@window.event
def on_draw():
    window.clear()
    sprite1.draw()
    img2.blit(600, 400)
    sprite3.draw()
    sprite4.draw()


pyglet.app.run()
