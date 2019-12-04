import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) +
                "/../../cs287/")
from prm import PRM
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
    obstacles = [(30, 35, 28),
                (0, 0, 15),
                (60, 0, 15)]
    params = generate_params(obstacles, 0, 60, 60, 60, rr)
    # obstacles = [
    #     (20, 20, 4),
    #     (12, 24, 8),
    #     (12, 32, 8),
    #     (12, 40, 8),
    #     (28, 20, 8),
    #     (36, 20, 8),
    #     (32, 40, 4)
    # ] 
    # params = generate_params(obstacles, 20, 10, 23, 30, rr)

    #================================================================

    all_results = []
    for n in range(args.num_init_samples, args.limit, args.increment):
        print('{} samples'.format(n))
        results = []
        num_failed = 0
        for i in range(args.iters):
            prm = PRM(*params, nsamples=n)
            res = prm.plan(show_animation=False)
            if res == -1:
                num_failed += 1
            else:
                results.append(res)
        avg = 0 if len(results) == 0 else sum(results) / len(results)
        print('average number of samples per trial: {}'.format(avg))
        p = 1 - num_failed/args.iters
        print('percent success: {}'.format(p))
        all_results.append((results, avg, p))

        if p >= args.threshold:
            break
    
    with open(args.filename, 'w') as f:
        f.write('Seed: {}\n'.format(seed))
        for r, a, p in all_results:
            f.write(str(r) + '\n')
            f.write('Average: {}\n'.format(a))
            f.write('Percent success: {}\n'.format(p))
            f.write('\n')

    #================================================================

    # prm = PRM(*params)

    # for i in range(args.iters):
    #     print('Trial {}'.format(i))
    #     res = prm.plan_iter(num_init_samples=args.num_init_samples, inc=args.increment, limit=args.limit)
    #     if res == -1:
    #         num_failed += 1
    #     else:
    #         results.append(res)
    #     print(res)

    # print('average number of samples per trial: {}'.format(sum(results) / len(results)))
    # print('number of trials failed: {}'.format(num_failed))
    
    # with open(args.filename, 'w') as f:
    #     f.write('Seed: {}\n'.format(seed))
    #     for r in results:
    #         f.write(str(r) + '\n')
    #     f.write('Average: {}\n'.format(sum(results) / len(results)))
    #     f.write('Percent success: {}\n'.format(1 - (num_failed/args.iters)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--filename', '-f', type=str, default='prm_nsample_circlepath.txt')
    parser.add_argument('--iters', '-t', type=int, default=100)
    parser.add_argument('--num_init_samples', '-n', type=int, default=100)
    parser.add_argument('--increment', '-i', type=int, default=20)
    parser.add_argument('--limit', '-l', type=int, default=2000)
    parser.add_argument('--threshold', '-p', type=float, default=.99)
    args = parser.parse_args()

    test_samples_needed(args)