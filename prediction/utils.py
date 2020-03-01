import time
import shutil
import torch
import logging
import os
from scipy.special import softmax


class AverageMeter(object):
    def __init__(self, name=''):
        self._name = name
        self.avg = 0.0
        self.sum = 0.0
        self.cnt = 0.0

    def reset(self):
        self.avg = 0.0
        self.sum = 0.0
        self.cnt = 0.0

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt

    def __str__(self):
        return "%s: %.5f" % (self._name, self.avg)

    def get_avg(self):
        return self.avg

    def __repr__(self):
        return self.__str__()


def datestr():
    now = time.gmtime()
    return '{}{:02}{:02}_{:02}{:02}'.format(now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min)


def get_logger(file_path):
    logger = logging.getLogger('net')
    log_format = '%(asctime)s | %(message)s'
    formatter = logging.Formatter(log_format, datefmt='%m/%d %I:%M:%S %p')
    file_handler = logging.FileHandler(file_path)
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.setLevel(logging.INFO)

    return logger


def get_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def get_device_ids():
    return range(
        torch.cuda.device_count()) if torch.cuda.is_available() else -1


def save_checkpoint(state, is_best, path, prefix, filename='checkpoint.pth.tar'):
    prefix_save = os.path.join(path, prefix)
    name = prefix_save + '_' + filename
    torch.save(state, name)
    if is_best:
        shutil.copyfile(name, prefix_save + '_model_best.pth.tar')


def save_description(path, args, model):
    f = open(path, "w")
    for k, v in args.items():
        f.write("{} = {}\n".format(k, v))
    f.write("\n")
    f.write(model.__repr__())
    f.close()


def save(model, model_path):
    
    torch.save(model.state_dict(), model_path)


def load(model, model_path):
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))


def add_text_to_file(text, file_path):
    with open(file_path, 'a') as f:
        f.write(text)


def clear_files_in_the_list(list_of_paths):
    for file_name in list_of_paths:
        open(file_name, 'w').close()


def create_directories_from_list(list_of_directories):
    for directory in list_of_directories:
        if not os.path.exists(directory):
            os.makedirs(directory)

