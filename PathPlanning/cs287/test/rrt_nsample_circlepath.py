import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) +
                "/../../cs287/")
from rrt import RRT
from config_space import generate_params

import random
import argparse

def test_samples_needed(args):
    print(__file__ + ' start!!!')

    results = []
    num_failed = 0

    seed = int(random.random() * 1000000000)
    random.seed(seed)

    rr = 0.5
    # obstacles = [(30, 35, 28),
    #             (0, 0, 15),
    #             (60, 0, 15)]
    # params = generate_params(obstacles, 0, 60, 60, 60, rr)

    obstacles = [
        (20, 20, 4),
        (12, 24, 8),
        (12, 32, 8),
        (12, 40, 8),
        (28, 20, 8),
        (36, 20, 8),
        (32, 40, 4)
    ] 
    params = generate_params(obstacles, 20, 10, 23, 30, rr)

    rrt = RRT(*params)

    for i in range(args.iters):
        print('Trial {}'.format(i))
        rrt = RRT(*params)
        n, path = rrt.planning(show_animation=False)
        if n == -1:
            num_failed += 1
        else:
            results.append(n)
        print(n)

    print('average number of samples per trial: {}'.format(sum(results) / len(results)))
    print('percent trials succeeded: {}'.format(1 - (num_failed/args.iters)))
    
    with open(args.filename, 'w') as f:
        f.write('seed: {}\n'.format(seed))
        for r in results:
            f.write(str(r) + '\n')
        f.write('Average: {}\n'.format(sum(results) / len(results)))
        f.write('Percent success: {}\n'.format(1 - (num_failed/args.iters)))


if __name__ == '__main__':
    # filename = 'rrt_nsample_circlepath.txt'
    filename = 'rrt_nsample_spiralpath.txt'

    parser = argparse.ArgumentParser()
    parser.add_argument('--filename', '-f', type=str, default=filename)
    parser.add_argument('--iters', '-t', type=int, default=100)
    parser.add_argument('--num_init_samples', '-n', type=int, default=100)
    parser.add_argument('--increment', '-i', type=int, default=20)
    parser.add_argument('--limit', '-l', type=int, default=2000)
    args = parser.parse_args()

    test_samples_needed(args)