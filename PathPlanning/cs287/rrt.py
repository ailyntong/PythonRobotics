import math
import random
import sys, os

import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)) +
                "/../cs287/")

try:
    from config_space import *
except ImportError:
    raise

show_animation = True

class RRT:

    class Node:
        def __init__(self, x, y):
            self.x = x
            self.y = y
            self.path_x = []
            self.path_y = []
            self.parent = None

    def __init__(self, cspace, sx, sy, gx, gy, robot_size,
                 expand_dist=3.0, path_resolution=0.5, goal_sample_rate=5, max_iter=2000):
        self.cspace = cspace
        self.start = self.Node(sx, sy)
        self.end = self.Node(gx, gy)
        self.rr = robot_size
        self.expand_dist = expand_dist
        self.path_resolution = path_resolution
        self.goal_sample_rate = goal_sample_rate
        self.max_iter = max_iter
        self.nodes = []

    def planning(self, show_animation=show_animation):
        self.nodes = [self.start]
        for i in range(self.max_iter):
            rnd_node = self.get_random_node()
            nearest_ind = self.get_nearest_node_index(self.nodes, rnd_node)
            nearest_node = self.nodes[nearest_ind]

            new_node = self.steer(nearest_node, rnd_node, self.expand_dist)

            if self.check_collision(new_node):
                self.nodes.append(new_node)

            # if show_animation and i % 5 == 0:
            #     self.draw_graph(rnd_node)

            if self.calc_dist_to_goal(new_node.x, new_node.y) <= self.expand_dist:
                final_node = self.steer(new_node, self.end, self.expand_dist)
                if self.check_collision(final_node):
                    return i, self.generate_final_course(len(self.nodes) - 1)

            # if show_animation and i % 5 == 0:
                # self.draw_graph(rnd_node)

        return -1, None # cannot find path

    def steer(self, from_node, to_node, extend_length=float('inf')):
        new_node = self.Node(from_node.x, from_node.y)
        d, theta = self.calc_distance_and_angle(new_node, to_node)

        new_node.path_x = [new_node.x]
        new_node.path_y = [new_node.y]

        extend_length = min(extend_length, d)

        n_expand = math.floor(extend_length / self.path_resolution)

        for _ in range(n_expand):
            new_node.x += self.path_resolution * np.cos(theta)
            new_node.y += self.path_resolution * np.sin(theta)
            new_node.path_x.append(new_node.x)
            new_node.path_y.append(new_node.y)

        d, _ = self.calc_distance_and_angle(new_node, to_node)
        if d <= self.path_resolution:
            new_node.path_x.append(to_node.x)
            new_node.path_y.append(to_node.y)

        new_node.parent = from_node

        return new_node

    def generate_final_course(self, goal_ind):
        path = [[self.end.x, self.end.y]]
        node = self.nodes[goal_ind]
        while node.parent is not None:
            path.append([node.x, node.y])
            node = node.parent
        path.append([node.x, node.y])

        return path

    def calc_dist_to_goal(self, x, y):
        return self.cspace.manhattan_distance(x, y, self.end.x, self.end.y)

    def get_random_node(self):
        if random.randint(0, 100) > self.goal_sample_rate:
            x, y = self.cspace.make_node()
            rnd = self.Node(x, y)
        else:  # goal point sampling
            rnd = self.Node(self.end.x, self.end.y)
        return rnd

    def draw_graph(self, rnd=None):
        plt.clf()
        if rnd is not None:
            plt.plot(rnd.x, rnd.y, '^k')
        for node in self.nodes:
            if node.parent:
                plt.plot(node.path_x, node.path_y, '-g')
        self.cspace.display()
        plt.plot(self.start.x, self.start.y, 'xr')
        plt.plot(self.end.x, self.end.y, 'xr')
        plt.pause(.01)

    @staticmethod
    def get_nearest_node_index(node_list, rnd_node):
        dlist = [(node.x - rnd_node.x) ** 2 + (node.y - rnd_node.y)
                 ** 2 for node in node_list]
        minind = dlist.index(min(dlist))

        return minind

    def check_collision(self, node):
        for x, y in zip(node.path_x, node.path_y):
            if self.cspace.collides(x, y, self.rr):
                return False # collision
        return True # safe

    @staticmethod
    def calc_distance_and_angle(from_node, to_node):
        dx = to_node.x - from_node.x
        dy = to_node.y - from_node.y
        d = math.sqrt(dx ** 2 + dy ** 2)
        theta = math.atan2(dy, dx)
        return d, theta

def main():
    print("start " + __file__)

    robot_size = 0.5

    # obstacles = [(30, 35, 28),
    #              (0, 0, 15),
    #              (60, 0, 15)]
    obstacles = [
        (20, 20, 4),
        (12, 24, 8),
        (12, 32, 8),
        (12, 40, 8),
        (28, 20, 8),
        (36, 20, 8),
        (32, 40, 4)
    ] 
    params = generate_params(obstacles, 20, 10, 23, 30, robot_size)

    # params = generate_random_params(rr=robot_size)
    rrt = RRT(*params)
    n, path = rrt.planning()

    # # ====Search Path with RRT====
    # obstacleList = [
    #     (5, 5, 1),
    #     (3, 6, 2),
    #     (3, 8, 2),
    #     (3, 10, 2),
    #     (7, 5, 2),
    #     (9, 5, 2),
    #     (8, 10, 1)
    # ]  # [x, y, radius]
    # # Set Initial parameters
    # cspace = ConfigurationSpace(-2, 15, -2, 15, obstacleList)
    # rrt = RRT(cspace=cspace,
    #           start=[0, 0],
    #           goal=[6.0, 10.0])
    # path = rrt.planning()

    if path is None:
        print("Cannot find path")
    else:
        print("found path!!")
        print(n)

        # Draw final path
        if show_animation:
            rrt.draw_graph()
            plt.plot([x for (x, y) in path], [y for (x, y) in path], '-r')
            plt.grid(True)
            plt.pause(0.01)  # Need for Mac
            plt.show()


if __name__ == '__main__':
    main()
