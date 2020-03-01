
from prediction.dummy_data import get_data
from prediction.data_wrapper import get_loaders
from prediction.model import Network
from prediction.train import Trainer

import time
import numpy as np
import torch
from prediction.utils import create_directories_from_list, get_logger, load
import logging
import argparse
import os
import sys
import glob
import pickle
sys.path.append("../")

import torch.nn as nn
import torch.backends.cudnn as cudnn

parser = argparse.ArgumentParser("drugs")
parser.add_argument('-f', type=str, default='./data/processed/processed_clean_df.csv', help='location of the data corpus')
parser.add_argument('--data', type=str, default='./data/processed/processed_clean_df.csv', help='location of the data corpus')
parser.add_argument('--input_size', nargs='+', type=list, default=[1], help='input size')
parser.add_argument('--hidden_size', type=int, default=100, help='number of hidden states')
parser.add_argument('--num_of_layers', type=int, default=1, help='number of hidden layers')
parser.add_argument('--output_size', nargs='+', type=list, default=[1], help='output size')
parser.add_argument('--test_portion',type = float, default = 0.2, help = 'what should be used for test')
parser.add_argument('--train_window',type = int, default = 12, help = 'training window, how many months should be concatenated in the input')
parser.add_argument('--batch_size', type=int, default=8, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.001, help='init learning rate')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--gpu', type=int, default=-1, help='gpu device id')
parser.add_argument('--epochs', type=int, default=150, help='num of training epochs')
parser.add_argument('--log_freq', type=int, default=10, help='the number of steps after which we log')
parser.add_argument('--layers', type=int, default=8, help='total number of layers')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--debug', type=bool, default=False, help='whether we are in a debug mode')
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--samples', type=int, default=20, help='how many samples to take')
parser.add_argument('--dropout', type=float, default=0.0, help='how many samples to take')

args = parser.parse_args()
name = "drugs-{}-{}".format(args.save, time.strftime("%Y%m%d-%H%M%S"))
args.save = './model/' + name
args.model_path = "./model/" + name + "/" + name + ".model"

NAME = "model/drugs-EXP-20200301-062740/drugs-EXP-20200301-062740.model"

if torch.cuda.is_available():
  args.gpu = 0

def get_model(train_loader=None, test_loader=None):
  if NAME != "":
    model = Network(args.train_window, args.input_size, args.hidden_size, args.num_of_layers, args.batch_size, args.output_size, args.dropout)
    load(model, NAME)
    scaler = pickle.load(open('model/scaler.pck', 'rb'))
    return model, scaler

  create_directories_from_list([args.save])

  logger = get_logger(args.save + "/log.txt")
  logger.info("# START #")

  if not torch.cuda.is_available() and args.gpu == -1:
    logging.info('no gpu device available')

  np.random.seed(args.seed)
  torch.manual_seed(args.seed)

  if torch.cuda.is_available():
    torch.cuda.set_device(args.gpu)
    torch.cuda.manual_seed(args.seed)
    logging.info('gpu device = %d' % args.gpu)
    cudnn.benchmark = True
    cudnn.enabled=True

  logger.info("Args = %s", args)
  logger.info("## Creating model and criterion ##")
  criterion = nn.MSELoss()

  if torch.cuda.is_available():
    criterion = criterion.cuda()

  model =Network(args.train_window, args.input_size, args.hidden_size, args.num_of_layers, args.batch_size, args.output_size, args.dropout)
  if torch.cuda.is_available():
    model = model.cuda()

  logging.info(model.__repr__())

  optimizer = torch.optim.SGD(
      model.parameters(),
      lr=args.learning_rate,
      weight_decay=args.weight_decay)

  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                         T_max=args.epochs//4,
                                                         last_epoch=-1)

  logger.info("## Getting and possibly pre-processing data ##")
  if train_loader is None or test_loader is None:
    train_data, test_data = get_data(args.test_portion)
    train_loader, test_loader = get_loaders(train_data, test_data, args)


  logger.info("## Beginning training ##")

  trainer = Trainer(model, criterion, optimizer, scheduler, logger, args)

  trainer.train_loop(train_loader, test_loader)

  return model



if __name__ == '__main__':
  get_model()
