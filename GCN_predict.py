import glob
import numpy as np
import torch
import os
import cv2
from hinet_arch import HINet
from dataset import TestDataset
import settings
from torch.utils.data import DataLoader
import argparse

logger = settings.logger

def ensure_dir(dir_path):
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)

class Session:
    def __init__(self):
        self.model_dir = settings.model_dir
        self.save_img_dir = settings.save_img_dir
        ensure_dir(settings.model_dir)
        ensure_dir(settings.save_img_dir)
        logger.info('set model dir as %s' % settings.model_dir)
        logger.info('set model dir as %s' % settings.save_img_dir)
        if torch.cuda.is_available():
            self.net = HINet().cuda()
        else:
            self.net =HINet()
        self.dataloaders = {}

    def get_dataloader(self, dataset_name):
        dataset = TestDataset(dataset_name)
        if not dataset_name in self.dataloaders:
            self.dataloaders[dataset_name] = DataLoader(dataset, batch_size=1,
                                                        shuffle=False, num_workers=1,
                                                        drop_last=True)
        return iter(self.dataloaders[dataset_name])

    # 下载模型函数
    def load_checkpoints(self, name,folder):
        ckp_path = os.path.join(self.model_dir,folder, name)
        try:
            obj = torch.load(ckp_path)
            logger.info('Load checkpoint %s' % ckp_path)
        except FileNotFoundError:
            logger.info('No checkpoint %s!!' % ckp_path)
            return
        self.net.load_state_dict(obj['net'])

    def test(self, name, batch):
        image, file_name = batch['image'], batch['idx']
        print(file_name)
        if torch.cuda.is_available():
            image = image.cuda()

        with torch.no_grad():
            output= self.net(image)
        return output, file_name

    def save_image(self, img_lists):
        pred, file_name = img_lists[0], img_lists[1][0]
        pred = (pred[0]+pred[1]).cpu().data
        pred = pred * 255
        pred = np.float32(pred)
        img_file = os.path.join(self.save_img_dir, file_name)
        cv2.imwrite(img_file, pred[0].squeeze())



def run_test(ckp_name,model_folder):
    sess = Session()
    sess.net.eval()
    sess.load_checkpoints(ckp_name,model_folder)
    dt = sess.get_dataloader(settings.save_img_dir.split('/')[1])
    for i, batch in enumerate(dt):
        batch_img, file_name = sess.test('test', batch)
        sess.save_image([batch_img, file_name])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument('-m', '--model', default='latest')
    parser.add_argument('-m', '--model', default='best_model')
    parser.add_argument('-f', '--folder', default='HINet')  # 可以改默认配置，把模型保存在不同的文件夹下
    args = parser.parse_args()
    print(args.folder)
    run_test(args.model,args.folder)