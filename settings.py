import os
import logging
#######################ESRGAN##################
nf = 8
scale_factor = 1
in_channels = 1
out_channels = 1
esrgan_lr = 1e-4
##################### Network parameters ###################################
num_workers = {'train':4,'val':4}
# 训练设置
batch_size = {'train':4,'val':4}
lr = 2e-4          # learning rate
save_steps = 400
iterations = 150000    # iterations default=300000
# 路径
data_dir = './data'
log_dir = './logdir/'#日志文件夹
save_img_dir = './test'
model_dir = './models/'#模型文件夹

# 创建一个logger
logger = logging.getLogger('train')
logger.setLevel(logging.INFO)
#创建一个handler，用于输出到控制台
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
#  定义handler的输出格式formatter
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)
############################################################################