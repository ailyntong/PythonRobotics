import random
import numpy as np
import math
import matplotlib.pyplot as plt

DEFAULT_XMIN = 0.0
DEFAULT_XMAX = 60.0
DEFAULT_YMIN = 0.0
DEFAULT_YMAX = 60.0

class ConfigurationSpace:
    def __init__(self, 
                 xmin=DEFAULT_XMIN, xmax=DEFAULT_XMAX, 
                 ymin=DEFAULT_YMIN, ymax=DEFAULT_YMAX, 
                 obstacles=None):
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
        self.obstacles = obstacles

    def make_start(self, pad=0):
        while True:
            x = (random.random() * (self.xmax - self.xmin)) + self.xmin
            y = (random.random() * (self.ymax - self.ymin)) + self.ymin
            if not self.collides(x, y, pad):
                return x, y

    def make_goal(self, pad=0):
        return self.make_start(pad)

    def make_node(self):
        x = random.uniform(self.xmin, self.xmax)
        y = random.uniform(self.ymin, self.ymax)
        return x, y

    def collides(self, x, y, pad=0):
        for ox, oy, r in self.obstacles:
            if self.manhattan_distance(x, y, ox, oy) <= r + pad:
                return True
        return False

    def edge_collides(self, x1, y1, x2, y2, pad=0):
        for cx, cy, r in self.obstacles:
            ax = x1 - cx
            ay = y1 - cy
            bx = x2 - cx
            by = y2 - cy
            r += pad
            # https://math.stackexchange.com/questions/275529/check-if-line-intersects-with-circles-perimeter
            c = ax**2 + ay**2 - r**2
            b = 2 * (ax * (bx - ax) + ay * (by - ay))
            a = (bx - ax)**2 + (by - ay)**2
            disc = b**2 - 4 * a * c
            if disc <= 0:
                # return False
                continue
            sqrtdisc = math.sqrt(disc)
            t1 = (-b + sqrtdisc) / (2 * a)
            t2 = (-b - sqrtdisc) / (2 * a)
            if (0 < t1 < 1) or (0 < t2 < 1):
                return True
        return False

    @staticmethod
    def manhattan_distance(x1, y1, x2, y2):
        return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)

    def display(self):
        for ox, oy, r in self.obstacles:
            self.plot_circle(ox, oy, r)
        plt.axis('equal')
        plt.axis([self.xmin, self.xmax, self.ymin, self.ymax])
        plt.grid(True)

    @staticmethod
    def plot_circle(x, y, r, color='-b'): # pragma: no cover
        deg = list(range(0, 360, 5))
        deg.append(0)
        xl = [x + r * math.cos(np.deg2rad(d)) for d in deg]
        yl = [y + r * math.sin(np.deg2rad(d)) for d in deg]
        plt.plot(xl, yl, color)

def generate_obstacles(num_obstacles, 
                       xmin=DEFAULT_XMIN, xmax=DEFAULT_XMAX, 
                       ymin=DEFAULT_YMIN, ymax=DEFAULT_YMAX, 
                       rmin=1, rmax=3):
    obstacles = []
    for _ in range(num_obstacles):
        r = (random.random() * (rmax - rmin)) + rmin
        x = (random.random() * (xmax - xmin - 2 * r)) + xmin + r
        y = (random.random() * (ymax - ymin - 2 * r)) + xmin + r
        obstacles.append((x, y, r))
    return obstacles

def generate_random_cspace(num_obstacles=10,
                           xmin=DEFAULT_XMIN, xmax=DEFAULT_XMAX, 
                           ymin=DEFAULT_YMIN, ymax=DEFAULT_YMAX, 
                           rmin=5.0, rmax=10.0):
    obstacles = generate_obstacles(num_obstacles, xmin, xmax, ymin, ymax, rmin, rmax)
    c = ConfigurationSpace(xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, obstacles=obstacles)
    return c

def generate_random_params(rr=0.0, num_obstacles=10,
                           xmin=DEFAULT_XMIN, xmax=DEFAULT_XMAX, 
                           ymin=DEFAULT_YMIN, ymax=DEFAULT_YMAX, 
                           rmin=5.0, rmax=10.0):
    cspace = generate_random_cspace(num_obstacles, xmin, xmax, ymin, ymax, rmin, rmax)
    sx, sy = cspace.make_start(rr)
    gx, gy = cspace.make_goal(rr)

    return cspace, sx, sy, gx, gy, rr

def generate_params(obstacles, sx, sy, gx, gy, rr,
                    xmin=DEFAULT_XMIN, xmax=DEFAULT_XMAX, 
                    ymin=DEFAULT_YMIN, ymax=DEFAULT_YMAX):
    cspace = ConfigurationSpace(xmin, xmax, ymin, ymax, obstacles)
    assert not cspace.collides(sx, sy, rr) and not cspace.collides(gx, gy, rr)
    return cspace, sx, sy, gx, gy, rr