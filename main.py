import argparse
import os.path

from Executor import Executor
from bayesian import st_knn_join
from utils.logger import TrainingLogger

from datetime import datetime


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--comment', type=str, default='default', required=False,
                        help="Description for current work")
    args = parser.parse_args()

    # 初始化日志
    save_path = f'./log/{args.comment}/{str(datetime.now()).replace(":", "_").replace(".", "_")}'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    logger = TrainingLogger(f'{save_path}/log')

    args.logger = logger
    args.save_path = save_path

    executor = Executor(args=args)
    # executor.start()

