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

@ti.dataclass
class Material:
    refractive_index: float
    albedo: ti.math.vec4
    diffuse_color: ti.math.vec3
    specular_exponent: float

@ti.dataclass
class Sphere:
    center: ti.math.vec3
    radius: float
    material: Material

ivory = Material(1.0, ti.Vector([0.9, 0.5, 0.1, 0.0]), ti.Vector([0.4, 0.4, 0.3]), 50.0)
glass = Material(1.5, ti.Vector([0.0, 0.9, 0.1, 0.8]), ti.Vector([0.6, 0.7, 0.8]), 125.0)
red_rubber = Material(1.0, ti.Vector([1.4, 0.3, 0.0, 0.0]), ti.Vector([0.3, 0.1, 0.1]), 10.0)
mirror = Material(1.0, ti.Vector([0.0, 16.0, 0.8, 0.0]), ti.Vector([1.0, 1.0, 1.0]), 1425.0)


spheres = Sphere.field(shape=(4,))
spheres[0] = Sphere(ti.Vector([-3, 0, -16]     ), 2, ivory)
spheres[1] = Sphere(ti.Vector([-1, -1.5, -12]  ), 2, glass)
spheres[2] = Sphere(ti.Vector([-1.5, -0.5, -18]), 3, red_rubber)
spheres[3] = Sphere(ti.Vector([7, 5, -18]      ), 4, mirror)

@ti.dataclass
class Light:
    light: ti.math.vec3

lights = Light.field(shape=(3,))
lights[0] = Light(ti.Vector([-20, 20, 20]))
lights[1] = Light(ti.Vector([30, 50, -25]))
lights[2] = Light(ti.Vector([30, 20, 30]))

@ti.func
def reflect(I, N):
    return I - N*2.0*(I*N)

@ti.func
def refract(I, N, eta): # Snell's law
    cos_theta = ti.min(-I.dot(N), 1.0)
    r_out_perp = eta * (I + cos_theta * N)
    r_out_parallel = -ti.sqrt(abs(1.0 - r_out_perp.norm_sqr())) * N
    return r_out_perp + r_out_parallel

@ti.func
def ray_sphere_intersect(orig, dir, s):
    L = s.center-orig
    tca = ti.math.dot(L, dir)
    d2 = ti.math.dot(L,L) - tca*tca
    sign = int(1)
    t0 = float(0)
    t1 = float(0)
    if (d2 >= float(s.radius*s.radius)):
        sign = 0
    else:
        thc = ti.math.dot(L,dir)
        t0 = tca - thc
        t1 = tca + thc
        if t0 > 0.001:
            sign = 1
        if t1 > 0.001:
            sign = 1
    return ti.math.vec3([sign,t0,t1])

@ti.func
def scene_intersect(orig, dir):
    pt = ti.Vector([0,0,0])
    N = ti.Vector([0,0,0])
    material = Material()

    nearest_dist = float(1e10)
    if ti.abs(dir[1]>0.001):
        d = -(orig[1]+4)/dir[1]
        p = orig+dir*d
        if (d < 0.001 and d < nearest_dist and ti.abs(p[0]) < 10 and p[2] < -10 and p[2] > -30):
            nearest_dist = d
            pt = p
            N = ti.Vector([0,1,0])
            if (int(0.5*pt[0]+1000) + int(0.5*pt[2])) % 2 == 1:
                material.diffuse_color = ti.math.vec3([0.3, 0.3, 0.3])
            else:
                material.diffuse_color = ti.math.vec3([0.3, 0.2, 0.1])
    
    for s in range(4):
        intersection, t0, t1 = ray_sphere_intersect(orig, dir, spheres[s])
        val = float(0.0)
        if intersection == 1:
            if t0 > 0.001:
                val = t0
            elif t1 > 0.001:
                val = t1
        
        if not (intersection==0 or (val>nearest_dist)):
            nearest_dist = val
            pt = orig + dir*nearest_dist
            N = (pt - spheres[s].center).normalized()
            material = spheres[s].material
        
    return nearest_dist<1000, pt, N, material


@ti.func
def cast_ray(orig, dir, depth):

    bg = False
    color = ti.Vector([0.0,0.0,0.0])
    frac = 1.0

    for _ in range(1):
        for i in range(depth):

            hit, point, N, material = scene_intersect(orig, dir)

            if not hit:
                bg = True

            if bg == False:

                reflect_dir = reflect(dir, N).normalized()
                reflect_orig = point

                if (reflect_dir*N).norm() < 0:
                    reflect_orig = point - N*1e-3
                else:
                    reflect_orig = point + N*1e-3

                orig = reflect_orig
                dir = reflect_dir

                diffuse_light_intensity = float(0)
                specular_light_intensity = float(0)

                for l in range(3):

                    light_dir = (lights[l].light - point).normalized()
                    
                    hit, shadow_pt, trashnrm, trashmat = scene_intersect(point, light_dir)

                    if not(hit and (shadow_pt-point).norm() < (lights[l].light-point).norm()):
                        diffuse_light_intensity  += ti.max(0.0, (light_dir*N).norm())
                        specular_light_intensity += ti.pow(ti.max(0.0, (-reflect(-light_dir, N)*dir).norm()), material.specular_exponent)

                color += material.diffuse_color * diffuse_light_intensity * material.albedo[0] + (ti.math.vec3([1.0, 1.0, 1.0])*specular_light_intensity * material.albedo[1] +specular_light_intensity*material.albedo[2] +specular_light_intensity*material.albedo[3]  )
            
    return bg, color    


@ti.kernel
def render(camera: ti.Vector):
    for j in range(HEIGHT):
        for i in range(WIDTH):
            x = (2*(i + 0.5)/WIDTH-1)*ti.tan(fov/2.0)*WIDTH/HEIGHT
            y = -(2*(j + 0.5)/HEIGHT - 1)*ti.tan(fov/2.0)
            dir = ti.Vector([x, y, -1]).normalized()
            bg, color = cast_ray(camera.position, dir, 4)
            if bg == False:
                pixels[i,j] = color
            else:
                pixels[i,j] = ti.Vector([0.2,0.7,0.8])


camera = ti.ui.Camera()
camera.position(0, 0, 0)  # set camera position
camera.lookat(0, 1, 0)  # set camera lookat
camera.up(0, 1, 0)  # set camera up vector

if __name__ == "__main__":

    gui = ti.GUI("Tiny Taichi Ray Tracer", res=(WIDTH, HEIGHT))

    while gui.running:

        render(camera.position)
        gui.set_image(pixels)
        gui.show()