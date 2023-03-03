import os

import cv2 as cv
import numpy as np
import taichi as ti
import wget

ti.init(arch=ti.gpu)

HEIGHT, WIDTH, CHANNELS = (256, 256, 3)
pixels = ti.field(dtype=float, shape=(WIDTH, HEIGHT, CHANNELS))

@ti.kernel
def intersect():
    return

if __name__ == "__main__":

    gui = ti.GUI("Tiny Taichi Ray Tracer", res=(WIDTH, HEIGHT))

    while gui.running:

        intersect()
        gui.set_image(pixels)
        gui.show()