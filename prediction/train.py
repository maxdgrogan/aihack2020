import torch
from torch.autograd import Variable
from prediction.utils import AverageMeter, save
import time


class Trainer:
    def __init__(self, model, criterion, optimiser, scheduler, logger, args):
        self.train_mse = AverageMeter()
        self.test_mse = AverageMeter()

        self.train_std = AverageMeter()
        self.test_std = AverageMeter()

        self.logger = logger
        self.args = args

        self.criterion = criterion
        self.model = model

        self.optimiser = optimiser
        self.scheduler = scheduler

        self.log_freq = args.log_freq
        self.path_to_save_model = args.model_path

    def train_loop(self, train_loader, test_loader):
        best_mse = float('inf')
        self.logger.info("### Start to train weights for %d epochs #### " % (
            self.args.epochs))
        for epoch in range(self.args.epochs):
            self.logger.info(
                "#### Training weights for epoch %d #### " % (epoch))
            self._training_step(
                train_loader, self.optimiser, epoch, info_for_logger="_train_step_")
            self.scheduler.step()

            mse = self._validate(test_loader, epoch)
            if self.args.debug:
                break
            if best_mse > mse:
                best_mse = mse
                self.logger.info("Best top1 acc by now. Save model")
                save(self.model, self.path_to_save_model)

        self.logger.info("## Finished training ## ")

    def _training_step(self, loader, optimiser, epoch, info_for_logger=""):
        self.model.train()
        start_time = time.time()
        batches = len(loader)
        for step, (X, y) in enumerate(loader):
            if self.args.gpu != -1:
                X, y = X.cuda(), y.cuda()
            N = X.shape[0]

            optimiser.zero_grad()
            y_pred = self.model(X)
 
            loss = self.criterion(y_pred, y)
            std = 0.
            loss.backward()
            optimiser.step()

            self._update(loss.item(), std, N, test_or_train="Train")

            self._intermediate_stats_logging(step, epoch, len_loader=len(
                loader), test_or_train="Train", info_for_logger=info_for_logger)

            if self.args.debug:
                break

        self._epoch_stats_logging(start_time=start_time, epoch=epoch,
                                  test_or_train='Train', info_for_logger=info_for_logger)
        self._reset()

    def _validate(self, loader, epoch):
        batches = len(loader)
        start_time = time.time()
        with torch.no_grad():
            for step, (X, y) in enumerate(loader):
                if self.args.gpu != -1:
                    X, y = X.cuda(), y.cuda()
                N = X.shape[0]
                y_pred = []
                for i in range(self.args.samples):
                    y_pred.append(self.model(X))
                y_pred = torch.stack(y_pred)
                y_pred = y_pred.view(N, self.args.samples)

                loss = 0.
                for i in range(self.args.samples):
                    loss += self.criterion(y_pred[:, i].unsqueeze(dim=1), y)
                loss /= self.args.samples

                std = 0.
                for i in range(self.args.samples):
                    std += torch.std(y_pred[:, i])
                std /= self.args.samples
                std = 0.0 if torch.isnan(std) else std

                self._update(loss, std, N, test_or_train="Test")
                self._intermediate_stats_logging(
                    step, epoch, len_loader=len(loader), test_or_train="Test")

                if self.args.debug:
                    break

        top_mse = self.test_mse.get_avg()
        self._epoch_stats_logging(
            start_time=start_time, epoch=epoch, test_or_train="Test")
        self._reset()

        return top_mse

    def _update(self, loss, std, N, test_or_train):
        if test_or_train == "Train":
            self.train_mse.update(loss, N)
            self.train_std.update(std, N)
        elif test_or_train == "Test":
            self.test_mse.update(loss, N)
            self.test_std.update(std, N)
        else:
            raise NotImplementedError(
                "This split has not been implemented: {}".format(test_or_train))

    def _reset(self):
        for avg in [self.train_mse, self.train_std, self.test_mse, self.test_std]:
            avg.reset()

    def _epoch_stats_logging(self, start_time, epoch, test_or_train, info_for_logger=''):
        if test_or_train == "Train":
            mse = self.train_mse.get_avg()
            std= self.train_std.get_avg()

        elif test_or_train == "Test":
            mse = self.test_mse.get_avg()
            std = self.test_std.get_avg()

        self.logger.info(info_for_logger+test_or_train + ": [{:3d}/{}] Final MSE {:.4f}+-{:.5f} Time {:.2f}".format(
            epoch+1, self.args.epochs, mse, std, time.time() - start_time))

    def _intermediate_stats_logging(self, step, epoch, len_loader, test_or_train, info_for_logger=''):
        if (step > 1 and step % self.log_freq == 0) or step == len_loader - 1:

            if test_or_train == "Train":
                mse = self.train_mse.get_avg()
                std = self.train_std.get_avg()

            elif test_or_train == "Test":
                mse = self.test_mse.get_avg()
                std = self.test_std.get_avg()

            self.logger.info("#### "+test_or_train +
                             ": [{:3d}/{}] Step {:03d}/{:03d} "
                             "MSE {:.4f}+-{:.5f}".format(
                                 epoch + 1, self.args.epochs, step, len_loader - 1, mse,
                                 std))
