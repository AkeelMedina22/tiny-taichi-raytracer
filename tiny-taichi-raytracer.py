import os

import cv2 as cv
import numpy as np
import taichi as ti
import wget

ti.init(arch=ti.gpu)

HEIGHT, WIDTH, CHANNELS = (400, 400, 3)
fov = 1.05
pixels = ti.Vector.field(3, dtype=float)
ti.root.dense(ti.ij, (WIDTH, HEIGHT)).place(pixels)


@ti.data_oriented
class Sphere:
    def __init__(self, center, radius, material):
        self.center = center
        self.radius = radius
        self.material = material

    @ti.func
    def intersect(self, orig, dir, t0):
        L = self.center-orig
        tca = ti.math.dot(L, dir)
        d2 = ti.math.dot(L,L) - tca*tca
        sign = int(1)
        if (d2 >= float(self.radius*self.radius)):
            thc = ti.math.dot(L,dir)
            t0 = tca - thc
            t1 = tca + thc
            if t0 < t1:
                t0 = t1
            if t0 < 0:
                sign += -1
        else:
            sign += -1
        return int(sign)
            
sphere = Sphere(ti.Vector([-3, 0, -16]), 2, 'ivory')

@ti.func
def cast_ray(orig, dir, sphere):
    sphere_dist = float('inf')
    output = ti.Vector([0.0,0.0,0.0])
    if sphere.intersect(orig, dir, sphere_dist) == 1:
        output = ti.Vector([0.2, 0.7, 0.8])
    else:
        output = ti.Vector([0.4, 0.4, 0.3])
    return output

@ti.kernel
def render():
    for j in range(HEIGHT):
        for i in range(WIDTH):
            x = (2*(i + 0.5)/WIDTH-1)*ti.tan(fov/2.0)*WIDTH/HEIGHT
            y = -(2*(j + 0.5)/HEIGHT - 1)*ti.tan(fov/2.0)
            dir = ti.Vector([x, y, -1]).normalized()
            pixels[i,j] = cast_ray(ti.Vector([0,0,0]), dir, sphere)


if __name__ == "__main__":

    gui = ti.GUI("Tiny Taichi Ray Tracer", res=(WIDTH, HEIGHT))

    while gui.running:

        render()
        gui.set_image(pixels)
        gui.show()