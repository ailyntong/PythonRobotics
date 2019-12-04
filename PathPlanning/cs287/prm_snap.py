import random
import numpy as np
import math
import matplotlib.pyplot as plt
import os, sys
import scipy.spatial
import time

sys.path.append(os.path.dirname(os.path.abspath(__file__)) +
                "/../cs287/")

try:
    from config_space import *
    from prm import PRM
except ImportError:
    raise

show_animation = True

class PRMSnap(PRM):
    def plan(self):
        sample_x, sample_y = self.sample_points(self.nsamples)
        valid_x, valid_y = self.filter_samples(sample_x, sample_y)

        if show_animation:
            self.cspace.display()
            plt.plot(self.sx, self.sy, '^r')
            plt.plot(self.gx, self.gy, '^c')
            plt.plot(sample_x, sample_y, '.b')
            # for x, y in zip(valid_x, valid_y):
            #     plt.plot(x, y, '.k')
            #     plt.pause(.0001)
            plt.plot(valid_x, valid_y, '.k')

        roadmap = self.generate_roadmap(valid_x, valid_y)
        # valid_roadmap = self.filter_roadmap(valid_x, valid_y, roadmap)

        # self.draw_roadmap(roadmap, valid_x, valid_y)
        # self.draw_roadmap(valid_roadmap, valid_x, valid_y)

        rx, ry = self.dijkstra_planning(roadmap, valid_x, valid_y)

        if show_animation:
            plt.plot(rx, ry, "-r")

        for i in range(len(rx) - 1):
            c = self.collides_strict(rx[i], ry[i], rx[i+1], ry[i+1])
            if c:
                print(rx[i], ry[i], rx[i+1], ry[i+1], c)
                break

        plt.show()