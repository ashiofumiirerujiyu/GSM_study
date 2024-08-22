import os
import random
import logging
import pytz
import torch
from datetime import datetime


class KSTFormatter(logging.Formatter):
    def formatTime(self, record, datefmt=None):
        kst = pytz.timezone('Asia/Seoul')
        log_time = datetime.fromtimestamp(record.created, kst)
        if datefmt:
            return log_time.strftime(datefmt)
        else:
            return log_time.strftime('%Y-%m-%d %H:%M:%S')
        

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def generate_save_path(seed):
    script_path = os.path.abspath(__file__)
    script_directory = os.path.dirname(script_path)

    base_output_path = os.path.join(script_directory, "output")
    if not os.path.exists(base_output_path):
        os.makedirs(base_output_path, exist_ok=True)

    kst = pytz.timezone('Asia/Seoul')
    now = datetime.now(kst)

    date_str = now.strftime("%Y_%m_%d")
    time_str = now.strftime("%H%M%S")

    path = os.path.join(base_output_path, f"{date_str}", f"{time_str}_{seed}")
    os.makedirs(path, exist_ok=True)

    return path
    