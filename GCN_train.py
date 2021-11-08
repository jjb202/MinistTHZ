from DualGCN import DualGCN
from dataset import ISBI_Dataset
from torch.optim.lr_scheduler import MultiStepLR
import torch.nn as nn
import torch
import random
from torch.optim import Adam
from cal_ssim import SSIM
import numpy as np
import argparse
import os
import settings
import torch.optim as optim
import losses
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from warmup_scheduler import GradualWarmupScheduler
logger = settings.logger

######### Set Seeds ###########
random.seed(1234)
np.random.seed(1234)
torch.manual_seed(1234)
torch.cuda.manual_seed_all(1234)


def ensure_dir(dir_path):
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)


class Session:
    def __init__(self):
        self.log_dir = settings.log_dir
        self.model_dir = settings.model_dir
        ensure_dir(settings.model_dir)
        ensure_dir(settings.log_dir)
        logger.info('set model dir as %s' % settings.model_dir)
        logger.info('set log dir as %s' % settings.log_dir)
        # 网络
        if torch.cuda.is_available():
            self.net = DualGCN().cuda()
            self.crit = nn.L1Loss().cuda()
            self.ssim = SSIM().cuda()
            ######### Loss ###########
            self.criterion_char = losses.CharbonnierLoss().cuda()
            self.criterion_edge = losses.EdgeLoss().cuda()
        else:
            self.net = DualGCN()
            self.crit = nn.L1Loss()
            self.ssim = SSIM()

        # 初始化各个参数
        self.step = 0
        self.step_num = settings.iterations
        self.save_steps = settings.save_steps
        self.num_workers = settings.num_workers
        self.batch_size = settings.batch_size
        self.dataloaders = {}
        self.writers = {}
        # 优化器选择Adam
        self.g_opt = Adam(self.net.parameters(), lr=settings.lr, betas=(0.9, 0.999),eps=1e-8)
        # self.g_sche = MultiStepLR(self.g_opt, milestones=[60000, 80000], gamma=0.2)

        scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(self.g_opt, 5000,
                                                            eta_min=1e-6)
        self.g_sche = GradualWarmupScheduler(self.g_opt, multiplier=1, total_epoch=3,
                                       after_scheduler=scheduler_cosine)

    def tensorboard(self, name):
        self.writers[name] = SummaryWriter(os.path.join(self.log_dir, name + '.events'))
        return self.writers[name]

    def write(self, name, out):
        for k, v in out.items():
            self.writers[name].add_scalar(k, v, self.step)
        out['lr'] = self.g_opt.param_groups[0]['lr']
        out['step'] = self.step
        outputs = [
            "{}:{:.4g}".format(k, v)
            for k, v in out.items()
        ]
        logger.info(name + '--' + ' '.join(outputs))

    def get_dataloader(self, dataset_name):
        dataset = ISBI_Dataset(dataset_name)
        if not dataset_name in self.dataloaders:
            self.dataloaders[dataset_name] = DataLoader(dataset, batch_size=self.batch_size[dataset_name],
                                                        shuffle=True, num_workers=self.num_workers[dataset_name],
                                                        drop_last=True)
        return iter(self.dataloaders[dataset_name])

    # 保存网络模型
    def save_checkpoints(self, name, folder):
        ckp_folder = os.path.join(self.model_dir, folder)
        ensure_dir(ckp_folder)
        ckp_path = os.path.join(ckp_folder, name)
        obj = {
            'net': self.net.state_dict(),
            'clock': self.step,
            'opt': self.g_opt.state_dict(),
        }
        torch.save(obj, ckp_path)

    # 下载模型函数
    def load_checkpoints(self, name, folder):
        ckp_path = os.path.join(self.model_dir, folder, name)
        try:
            obj = torch.load(ckp_path)
            logger.info('Load net checkpoint %s' % ckp_path)
        except FileNotFoundError:
            logger.info('No net checkpoint %s!!' % ckp_path)
            return
        self.net.load_state_dict(obj['net'])
        self.g_opt.load_state_dict(obj['opt'])
        self.step = obj['clock']
        self.g_sche.last_epoch = self.step

    # 训练
    def train(self, name, batch):
        self.net.train()
        image, label = batch['image'], batch['label']
        if torch.cuda.is_available():
            image = image.cuda()
            label = label.cuda()
        pred = self.net(image)

        loss_list = [self.crit(pred[0], label)]
        ssim_list = [self.ssim(pred[0], label)]
        losses = {
            'loss_L1%d' % i: loss.item()
            for i, loss in enumerate(loss_list)
        }
        ssimes = {
            'ssim%d' % i: ssim.item()
            for i, ssim in enumerate(ssim_list)
        }
        losses.update(ssimes)
        # 显示loss
        self.write(name, losses)
        # 梯度回传，更新参数
        loss = sum(loss_list)
        self.net.zero_grad()
        loss.backward(retain_graph=True)
        self.g_opt.step()
        return pred[0], loss.item()


# 训练和验证网络，ckp_name 默认加载最新保存的模型，开始训练
# folder指的是你./models下的某个文件夹，该folder内保存的是各种模型
def run_train(ckp_name='latest', folder='mynet'):
    # 创建任务
    sess = Session()
    sess.load_checkpoints(ckp_name, folder)
    sess.tensorboard('train')
    dt_train = sess.get_dataloader('train')
    best_loss = float('inf')
    while sess.step <= sess.step_num:
        sess.g_sche.step()
        try:
            batch_t = next(dt_train)
        except StopIteration:
            dt_train = sess.get_dataloader('train')
            batch_t = next(dt_train)
        pred_t, losses = sess.train('train', batch_t)
        # 保存模型，每save_steps次，保存最新的模型latest
        if sess.step % int(sess.save_steps) == 0:
            sess.save_checkpoints('latest', folder)
        # 保存loss值最小的网络参数
        if losses < best_loss:
            best_loss = losses
            sess.save_checkpoints('best_model', folder)
        sess.step += 1

    logger.info('------------finish training-------------')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', default='latest')
    parser.add_argument('-f', '--folder', default='DualGCN')  # 可以改默认配置，把模型保存在不同的文件夹下
    args = parser.parse_args()
    print(args.folder)
    run_train(args.model, args.folder)

