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
except ImportError:
    raise

# parameters
N_SAMPLES = 500
N_KNN = 10
MAX_NEIGHBOR_DIST = 20.0

show_animation = True

class Node:
    def __init__(self, x, y, cost, pind):
        self.x = x
        self.y = y
        self.cost = cost
        self.pind = pind
    
    def __str__(self):
        return str(self.x) + "," + str(self.y) + "," + str(self.cost) + "," + str(self.pind)

class KDTree:
    def __init__(self, data):
        self.tree = scipy.spatial.cKDTree(data)

    def search(self, point, k=1):
        if len(point.shape) >= 2:
            index = []
            dist = []

            for i in point.T:
                idist, iindex = self.tree.query(i, k=k)
                index.append(iindex)
                dist.append(idist)
            
            return index, dist
        
        dist, index = self.tree.query(point, k=k)
        return index, dist

    def search_in_distance(self, point, r):
        return self.tree.query_ball_point(point, r)

class PRM:
    def __init__(self, cspace, start_x, start_y, goal_x, goal_y, robot_radius, nsamples=N_SAMPLES):
        self.cspace = cspace
        self.sx = start_x
        self.sy = start_y
        self.gx = goal_x
        self.gy = goal_y
        self.rr = robot_radius
        self.nsamples=nsamples

    def plan_iter(self, num_init_samples=100, inc=20, limit=2000, show_animation=False):
        num_samples = num_init_samples
        sample_x, sample_y = self.sample_points(num_init_samples - inc)
        valid_x, valid_y = self.filter_samples(sample_x, sample_y)
        while True:
            # print(num_samples)
            new_sample_x, new_sample_y = self.sample_points(inc)
            new_valid_x, new_valid_y = self.filter_samples(new_sample_x, new_sample_y)

            sample_x.extend(new_sample_x)
            sample_y.extend(new_sample_y)
            valid_x.extend(new_valid_x)
            valid_y.extend(new_valid_y)

            if show_animation:
                plt.clf()
                self.cspace.display()
                plt.plot(self.sx, self.sy, '^r')
                plt.plot(self.gx, self.gy, '^c')
                plt.plot(sample_x, sample_y, '.b')
                plt.plot(valid_x, valid_y, '.k')

            roadmap = self.generate_roadmap(valid_x, valid_y)
            rx, ry = self.dijkstra_planning(roadmap, valid_x, valid_y)

            if rx:
                break
            elif num_samples >= limit:
                if show_animation:
                    plt.show()
                return -1

            num_samples += inc

        # print('needed {} samples'.format(num_samples))
        if show_animation:
            plt.plot(rx, ry, "-r")
            plt.show()
        return num_samples
            

    def plan(self, show_animation=show_animation):
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
            plt.show()

        # for i in range(len(rx) - 1):
        #     c = self.collides_strict(rx[i], ry[i], rx[i+1], ry[i+1])
        #     if c:
        #         print(rx[i], ry[i], rx[i+1], ry[i+1], c)
        #         break

        return self.nsamples if rx else -1

    def sample_points(self, n_samples):
        xmax = self.cspace.xmax
        ymax = self.cspace.ymax
        minx = self.cspace.xmin
        ymin = self.cspace.ymin

        sample_x, sample_y = [], []

        for _ in range(n_samples):
            sample_x.append((random.random() * (xmax - minx)) + minx)
            sample_y.append((random.random() * (ymax - ymin)) + ymin)

        sample_x.append(self.sx)
        sample_y.append(self.sy)
        sample_x.append(self.gx)
        sample_y.append(self.gy)

        return sample_x, sample_y

    def filter_samples(self, sample_x, sample_y):
        samples = zip(sample_x, sample_y)
        samples = list(filter(lambda p: not self.cspace.collides(p[0], p[1], self.rr), samples))
        sample_x, sample_y = zip(*samples)
        return list(sample_x), list(sample_y)

    def generate_roadmap(self, sample_x, sample_y):
        roadmap = []
        samples_kdtree = KDTree(np.vstack((sample_x, sample_y)).T)
        nsample = len(sample_x)

        # for i, x, y, in zip(range(len(sample_x)), sample_x, sample_y):
        #     indexes, dists = samples_kdtree.search(np.array([x, y]).reshape(2, 1), k=N_KNN)
        #     edges = indexes[0][1:]
        #     roadmap.append(edges)

        for (i, ix, iy) in zip(range(nsample), sample_x, sample_y):

            index, dists = samples_kdtree.search(
                np.array([ix, iy]).reshape(2, 1), k=nsample)
            inds = index[0]
            edge_id = []

            for ii in range(1, len(inds)):
                nx = sample_x[inds[ii]]
                ny = sample_y[inds[ii]]

                if not self.collides_strict(ix, iy, nx, ny):
                    edge_id.append(inds[ii])
                # edge_id.append(inds[ii])

                if len(edge_id) >= N_KNN:
                    break

            roadmap.append(edge_id)

        return roadmap

    def filter_roadmap(self, sample_x, sample_y, roadmap):
        valid_roadmap = []

        for i, x, y in zip(range(len(sample_x)), sample_x, sample_y):
            edges = roadmap[i]
            filtered_edges = []
            for ii in edges:
                nx = sample_x[ii]
                ny = sample_y[ii]
                if not self.collides(x, y, nx, ny):
                    filtered_edges.append(ii)
            valid_roadmap.append(filtered_edges)

        return valid_roadmap

    def draw_roadmap(self, roadmap, sample_x, sample_y):
        assert(len(roadmap) == len(sample_x))
        lines = []
        for i in range(len(roadmap)):
            for ii in roadmap[i]:
                lines.extend([(sample_x[i], sample_y[i]), (sample_x[ii], sample_y[ii]), '-g'])
        plt.plot(*lines)

    def collides(self, x, y, nx, ny):
        return self.cspace.manhattan_distance(x, y, nx, ny) >= MAX_NEIGHBOR_DIST

    def collides_strict(self, x, y, nx, ny):
        return self.cspace.manhattan_distance(x, y, nx, ny) >= MAX_NEIGHBOR_DIST \
            or self.cspace.edge_collides(x, y, nx, ny, self.rr)

    def dijkstra_planning(self, roadmap, sample_x, sample_y):
        """
        roadmap: ??? [m]
        sample_x: ??? [m]
        sample_y: ??? [m]
        
        @return: Two lists of path coordinates ([x1, x2, ...], [y1, y2, ...]), empty list when no path was found
        """

        nstart = Node(self.sx, self.sy, 0.0, -1)
        ngoal = Node(self.gx, self.gy, 0.0, -1)

        openset, closedset = dict(), dict()
        openset[len(roadmap) - 2] = nstart

        path_found = True
        
        while True:
            if not openset:
                # print("Cannot find path")
                path_found = False
                break

            c_id = min(openset, key=lambda o: openset[o].cost)
            current = openset[c_id]

            # show graph
            if show_animation:# and len(closedset.keys()) % 2 == 0:
                plt.plot(current.x, current.y, "xg")
                # plt.pause(0.001)

            if c_id == (len(roadmap) - 1):
                # print("goal is found!")
                ngoal.pind = current.pind
                ngoal.cost = current.cost
                break

            # Remove the item from the open set
            del openset[c_id]
            # Add it to the closed set
            closedset[c_id] = current

            # expand search grid based on motion model
            for i in range(len(roadmap[c_id])):
                n_id = roadmap[c_id][i]
                dx = sample_x[n_id] - current.x
                dy = sample_y[n_id] - current.y
                d = math.sqrt(dx**2 + dy**2)
                node = Node(sample_x[n_id], sample_y[n_id],
                            current.cost + d, c_id)

                if n_id in closedset:
                    continue
                # Otherwise if it is already in the open set
                if n_id in openset:
                    if openset[n_id].cost > node.cost:
                        openset[n_id].cost = node.cost
                        openset[n_id].pind = c_id
                else:
                    openset[n_id] = node
        
        if path_found is False:
            return [], []

        # generate final course
        rx, ry = [ngoal.x], [ngoal.y]
        pind = ngoal.pind
        while pind != -1:
            n = closedset[pind]
            rx.append(n.x)
            ry.append(n.y)
            pind = n.pind

        return rx, ry

def main():
    print(__file__ + ' start!!!')

    seed = int(random.random() * 1000000000)
    random.seed(seed)
    print(seed)

    # random.seed(132538284)
    # random.seed(364932963)
    # random.seed(215051673)
    # random.seed(591224707)
    # random.seed(899706384)
    # random.seed(435534417)

    # start and goal position
    # sx = 10.0  # [m]
    # sy = 10.0  # [m]
    # gx = 50.0  # [m]
    # gy = 50.0  # [m]
    robot_size = 0.5 # [m]

    obstacles = [(30, 35, 28),
                 (0, 0, 15),
                 (60, 0, 15)]
    params = generate_params(obstacles, 0, 60, 60, 60, robot_size)

    # obstacles = [
    #     (20, 20, 4),
    #     (12, 24, 8),
    #     (12, 32, 8),
    #     (12, 40, 8),
    #     (28, 20, 8),
    #     (36, 20, 8),
    #     (32, 40, 4)
    # ] 
    # params = generate_params(obstacles, 20, 10, 23, 30, robot_size)

    # params = generate_random_params(rr=robot_size)
    prm = PRM(*params)
    n = prm.plan()
    # n = prm.plan_iter(show_animation=True)
    print(n)

if __name__ == '__main__':
    main()