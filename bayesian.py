import joblib
import pandas as pd
import numpy as np
import os
import argparse
import pickle

import skopt
from matplotlib import pyplot as plt

from Executor import Executor
from datetime import datetime
from utils.logger import TrainingLogger
from skopt import gp_minimize, forest_minimize
from skopt.plots import plot_convergence

from utils.plots import plot_rf


def st_knn_join(x):
    executor = Executor(args=args, data_path='resources/generate_100k')

    # 参数设置
    alpha = 200
    beta = 40
    binNum = 200

    if args.knob == 'alpha':
        alpha = x[0]
    elif args.knob == 'beta':
        beta = x[0]
    elif args.knob == 'binNum':
        binNum = x[0]

    time_sum = 0
    logger.print(f'alpha: {alpha}, beta: {beta}, binNum: {binNum}')
    cnt = 5
    for i in range(cnt):
        # alpha = 200, beta = 40, binNum = 200
        file_save_path = f'./results_generate_100k/{alpha}_{beta}_{binNum}/{i}'
        if not os.path.exists(file_save_path):
            executor.execute(alpha=alpha, beta=beta, binNum=binNum, save_path=file_save_path)
        time = executor.parse_result(file_save_path)
        logger.print(f'time: {time}')
        time_sum += time
    logger.print(f'avg_time: {time_sum / 5}\n')
    return time_sum / cnt


def draw_0():
    # Plot f(x) + contours
    x = np.linspace(0, 700, 350).reshape(-1, 1)
    fx = [st_knn_join(x_i) for x_i in x]
    plt.plot(x, fx, "r--", label="True (execute time)")
    plt.fill(np.concatenate([x, x[::-1]]),
             np.concatenate(([fx_i - 1.9600 for fx_i in fx],
                             [fx_i + 1.9600 for fx_i in fx[::-1]])),
             alpha=.2, fc="r", ec="None")
    plt.legend()
    plt.grid()
    plt.show()


def draw(res):
    noise_level = 0.1

    for n_iter in range(5):
        # Plot true function.
        plt.subplot(5, 2, 2 * n_iter + 1)

        if n_iter == 0:
            show_legend = True
        else:
            show_legend = False

        ax = plot_rf(res, n_calls=n_iter,
                     objective=st_knn_join,
                     noise_level=noise_level,
                     show_legend=show_legend, show_title=False,
                     show_next_point=False, show_acq_func=False)
        ax.set_ylabel("")
        ax.set_xlabel("")

        # Plot EI(x)
        plt.subplot(5, 2, 2 * n_iter + 2)
        ax = plot_rf(res, n_calls=n_iter,
                     show_legend=show_legend, show_title=False,
                     show_mu=False, show_acq_func=True,
                     show_observations=False,
                     show_next_point=True)
        ax.set_ylabel("")
        ax.set_xlabel("")

    plt.show()
    plt.savefig(f'{args.save_path}/function.png')


# gp回归
def gp():
    if args.knob == 'alpha':
        bound = [(50, 1000)]
    elif args.knob == 'beta':
        bound = [(5, 100)]
    elif args.knob == 'binNum':
        bound = [(10, 1000)]
    else:
        bound = []
        logger.print(f"knob err: {args.knob}")

    res = gp_minimize(st_knn_join,  # the function to minimize
                      bound,  # the bounds on each dimension of x
                      acq_func="EI",  # the acquisition function
                      n_calls=35,  # the number of evaluations of f
                      n_random_starts=30,  # the number of random initialization points
                      noise=0.1 ** 2,  # the noise level (optional)
                      random_state=123)  # the random seed
    # 保存
    skopt.dump(res, f'{args.save_path}/{args.knob}.pkl')
    fig = plot_convergence(res).get_figure()
    fig.savefig(f'{args.save_path}/output.png')

    return res


# 执行bayes优化
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--comment', type=str, default='bayes_test', required=False,
                        help="Description for current work")
    parser.add_argument('--knob', type=str, default='alpha', required=False,
                        help="Research on specific knob")
    args = parser.parse_args()

    # 初始化日志
    save_path = f'./log/{args.comment}/{str(datetime.now()).replace(":", "_").replace(".", "_")}'

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    logger = TrainingLogger(f'{save_path}/log')
    args.logger = logger
    args.save_path = save_path

    res = gp()
    draw(res)
