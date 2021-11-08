from ESRGAN import *
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
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

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
        # 初始化各个参数
        self.step = 0
        self.step_num = settings.iterations
        self.save_steps = settings.save_steps
        self.num_workers = settings.num_workers
        self.batch_size = settings.batch_size
        self.dataloaders = {}
        self.writers = {}
        self.nf = settings.nf
        self.scale_factor = settings.scale_factor
        self.in_channels = settings.in_channels
        self.out_channels = settings.out_channels
        self.lr = settings.esrgan_lr
        # 网络
        if torch.cuda.is_available():
            self.generator = ESRGAN(in_channels=self.in_channels,out_channels=self.out_channels,nf=self.nf,
                              scale_factor=self.scale_factor).cuda()
            self.discriminator = Discriminator().cuda()
            self.content_criterion = nn.L1Loss().cuda()
            self.adversarial_criterion = nn.BCEWithLogitsLoss().cuda()
        else:
            self.generator = ESRGAN(in_channels=self.in_channels,out_channels=self.out_channels,nf=self.nf,
                              scale_factor=self.scale_factor)
            self.discriminator = Discriminator()
            self.content_criterion = nn.L1Loss()
            self.adversarial_criterion = nn.BCEWithLogitsLoss().cuda()

        # 优化器选择Adam
        self.optimizer_generator = Adam(self.generator.parameters(), lr=self.lr, betas=(0.9, 0.999))
        self.optimizer_discriminator = Adam(self.discriminator.parameters(), lr=self.lr, betas=(0.9, 0.999))
        self.g_sche = MultiStepLR(self.optimizer_generator, milestones=[20000, 40000,60000], gamma=0.1)
        self.d_sche = MultiStepLR(self.optimizer_discriminator, milestones=[20000, 40000,60000], gamma=0.1)

    def tensorboard(self, name):
        self.writers[name] = SummaryWriter(os.path.join(self.log_dir, name + '.events'))
        return self.writers[name]

    def write(self, name, out):
        for k, v in out.items():
            self.writers[name].add_scalar(k, v, self.step)
        out['lr'] = self.optimizer_generator.param_groups[0]['lr']
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
        ckp_folder = os.path.join(self.model_dir, folder,"generator")
        ensure_dir(ckp_folder)
        ckp_path = os.path.join(ckp_folder, name)
        obj = {
            'net': self.generator.state_dict(),
            'clock': self.step,
            'opt': self.optimizer_generator.state_dict(),
        }
        torch.save(obj, ckp_path)
        ckp_folder = os.path.join(self.model_dir, folder,"discriminator")
        ensure_dir(ckp_folder)
        ckp_path = os.path.join(ckp_folder, name)
        obj = {
            'net': self.discriminator.state_dict(),
            'clock': self.step,
            'opt': self.optimizer_discriminator.state_dict(),
        }
        torch.save(obj, ckp_path)

    # 下载模型函数
    def load_checkpoints(self, name, folder):
        ckp_path = os.path.join(self.model_dir, folder, "generator",name)
        try:
            obj = torch.load(ckp_path)
            logger.info('Load net checkpoint %s' % ckp_path)
        except FileNotFoundError:
            logger.info('No net checkpoint %s!!' % ckp_path)
            return
        self.generator.load_state_dict(obj['net'])
        self.optimizer_generator.load_state_dict(obj['opt'])
        self.g_sche.last_epoch = self.step

        ckp_path = os.path.join(self.model_dir, folder, name)
        try:
            obj = torch.load(ckp_path)
            logger.info('Load net checkpoint %s' % ckp_path)
        except FileNotFoundError:
            logger.info('No net checkpoint %s!!' % ckp_path)
            return
        self.discriminator.load_state_dict(obj['net'])
        self.optimizer_discriminator.load_state_dict(obj['opt'])
        self.step = obj['clock']
        self.d_sche.last_epoch = self.step

    # 训练
    def train(self, name, batch):
        self.generator.train()
        self.discriminator.train()
        image, label = batch['image'], batch['label']
        real_labels = torch.ones((image.size(0), 1))
        fake_labels = torch.zeros((label.size(0), 1))
        if torch.cuda.is_available():
            image = image.cuda()
            label = label.cuda()
            real_labels = real_labels.cuda()
            fake_labels = fake_labels.cuda()

        ##########################
        #   training generator   #
        ##########################
        self.optimizer_generator.zero_grad()
        fake_resolution = self.generator(image)

        score_real = self.discriminator(label)
        score_fake = self.discriminator(fake_resolution)

        discriminator_rf = score_real - score_fake.mean()
        discriminator_fr = score_fake - score_real.mean()

        adversarial_loss_rf = self.adversarial_criterion(discriminator_rf, fake_labels)
        adversarial_loss_fr = self.adversarial_criterion(discriminator_fr, real_labels)
        adversarial_loss = (adversarial_loss_fr + adversarial_loss_rf) / 2

        content_loss = self.content_criterion(fake_resolution,label)

        generator_loss = adversarial_loss * 5e-3 + content_loss * 0.1
        generator_loss.backward()
        self.optimizer_generator.step()

        losses = {
            'g_loss_L1': content_loss.item(),'g_loss_adv':adversarial_loss.item()
        }

        ##########################
        # training discriminator #
        ##########################

        self.optimizer_discriminator.zero_grad()
        score_real = self.discriminator(label)
        score_fake = self.discriminator(fake_resolution.detach())
        discriminator_rf = score_real - score_fake.mean()
        discriminator_fr = score_fake - score_real.mean()

        adversarial_loss_rf = self.adversarial_criterion(discriminator_rf, real_labels)
        adversarial_loss_fr = self.adversarial_criterion(discriminator_fr, fake_labels)
        discriminator_loss = (adversarial_loss_fr + adversarial_loss_rf) / 2

        discriminator_loss.backward()
        self.optimizer_discriminator.step()
        self.g_sche.step()
        self.d_sche.step()

        d_losses = {
            'd_loss_adv' : discriminator_loss.item()
        }
        losses.update(d_losses)
        # 显示loss
        self.write(name, losses)
        return fake_resolution, content_loss.item()


# 训练和验证网络，ckp_name 默认加载最新保存的模型，开始训练
# folder指的是你./models下的某个文件夹，该folder内保存的是各种模型
def run_train(ckp_name='latest', folder='esrgan'):
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
    parser.add_argument('-f', '--folder', default='esrgan')  # 可以改默认配置，把模型保存在不同的文件夹下
    args = parser.parse_args()
    print(args.folder)
    run_train(args.model, args.folder)

