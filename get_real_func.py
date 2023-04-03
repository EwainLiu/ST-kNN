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

    # 执行
    executor = Executor(args=args, data_path='resources/generate_10k')
    results_all = executor.execute_specific_knob()

    # 绘图
    df = pd.DataFrame(results_all)
    plt.plot(df[args.knob], df['time'])
    plt.savefig(f'{save_path}/log/real_func.png')
