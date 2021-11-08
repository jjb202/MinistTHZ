'''
calculate the PSNR and SSIM.
same as MATLAB's results
'''
import os
import math
import numpy as np
import cv2
import glob
import csv
import sys

sys.path.append('../')

from matplotlib import pyplot as plt

def main(folder_GT,folder_Gen):

    # Configurations
    # GT - Ground-truth;
    # Gen: Generated / Restored / Recovered images
    # folder_GT = '/mnt/cv/code/IMDN/Test_Datasets/Set5'
    # folder_Gen = '/mnt/cv/code/IMDN/results/Set5/x2'

    test_Y =False
    PSNR_all = []
    SSIM_all = []
    img_list = os.listdir(folder_GT)
    file_num = len(img_list)

    for i in  range(file_num):

        path_GT=os.path.join(folder_GT,img_list[i])
        im_GT = cv2.imread(path_GT,0) / 255.
        path_Gen=os.path.join(folder_Gen,img_list[i])
        im_Gen = cv2.imread(path_Gen,0) / 255.

        # calculate PSNR and SSIM
        PSNR = calculate_psnr(im_GT,im_Gen)

        SSIM = calculate_ssim(im_GT*255,im_Gen*255)
        print('the {:3d}th - {:25}. \tPSNR: {:.6f} dB, \tSSIM: {:.6f}'.format(
            i + 1, img_list[i], PSNR, SSIM))

        PSNR_all.append(PSNR)
        SSIM_all.append(SSIM)
    print('Average: PSNR: {:.6f} dB, SSIM: {:.6f}'.format(
        sum(PSNR_all) / len(PSNR_all),
        sum(SSIM_all) / len(SSIM_all)))

    #return [int(folder_Gen.split('/')[-1].split('_')[-1][2:]), sum(PSNR_all) / len(PSNR_all), sum(SSIM_all) / len(SSIM_all)]

'''
def calculate_psnr(img1, img2):
    # img1 and img2 have range [0, 255]
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))
'''
def calculate_psnr(img1, img2):
   mse = np.mean((img1 - img2) ** 2)
   if np.max(img1) > 128.:
       PIXEL_MAX = 255.
   else:
       PIXEL_MAX = 1.
   if mse < 1.0e-10:
      return 100
   return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

def ssim(img1, img2):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def calculate_ssim(img1, img2):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')


def bgr2ycbcr(img, only_y=True):
    '''same as matlab rgb2ycbcr
    only_y: only return Y channel
    Input:
        uint8, [0, 255]
        float, [0, 1]
    '''
    in_img_type = img.dtype
    img.astype(np.float32)
    if in_img_type != np.uint8:
        img *= 255.
    # convert
    if only_y:
        rlt = np.dot(img, [24.966, 128.553, 65.481]) / 255.0 + 16.0
    else:
        rlt = np.matmul(img, [[24.966, 112.0, -18.214], [128.553, -74.203, -93.786],
                              [65.481, -37.797, 112.0]]) / 255.0 + [16, 128, 128]
    if in_img_type == np.uint8:
        rlt = rlt.round()
    else:
        rlt /= 255.
    return rlt.astype(in_img_type)


def draw_nf_rf_vs_psnr_ssim(NF_X, RF_X, NF_Y_PSNR, NF_Y_SSIM, RF_Y_PSNR, RF_Y_SSIM):
    plt.subplot(2, 2, 1)
    plt.title('PSNR VS channel', fontsize=15)  # 标题，并设定字号大小
    plt.xlabel("channel", fontsize=14)  # 设置x轴，并设定字号大小
    plt.ylabel("PSNR", fontsize=14)  # 设置y轴，并设定字号大小
    plt.grid()
    plt.plot(NF_X[:-4], NF_Y_PSNR[:-4], color="deeppink", linewidth=2, marker='^')
    plt.plot(NF_X[-4:], NF_Y_PSNR[-4:], color="g", linewidth=2, marker='^',linestyle=':')

    plt.subplot(2, 2, 2)
    plt.title('SSIM VS channel', fontsize=15)  # 标题，并设定字号大小
    plt.xlabel("channel", fontsize=14)  # 设置x轴，并设定字号大小
    plt.ylabel("SSIM", fontsize=14)  # 设置y轴，并设定字号大小
    plt.grid()
    plt.plot(NF_X[:-4], NF_Y_SSIM[:-4], color="deeppink", linewidth=2, marker='^')
    plt.plot(NF_X[-4:], NF_Y_SSIM[-4:], color="g", linewidth=2, marker='^',linestyle=':')

    plt.subplot(2, 2, 3)
    plt.title('PSNR VS RF', fontsize=15)  # 标题，并设定字号大小
    plt.xlabel("channel", fontsize=14)  # 设置x轴，并设定字号大小
    plt.ylabel("PSNR", fontsize=14)  # 设置y轴，并设定字号大小
    plt.grid()
    plt.plot(RF_X, RF_Y_PSNR, color="deeppink", linewidth=2, marker='^')
    plt.subplot(2, 2, 4)
    plt.title('SSIM VS RF', fontsize=15)  # 标题，并设定字号大小
    plt.xlabel("channel", fontsize=14)  # 设置x轴，并设定字号大小
    plt.ylabel("SSIM", fontsize=14)  # 设置y轴，并设定字号大小
    plt.grid()
    plt.plot(RF_X, RF_Y_SSIM, color="deeppink", linewidth=2, marker='^')
    plt.show()

if __name__ == '__main__':
    folder_GT='./data/test/label'
    folder_Gen='./test'
    # folder_GT='../../dataset/RAIN800_DID_MDN_dataset/Rain12'
    # folder_Gen='../rgb_in_RAIN1400_FSPANet_detal_15/Rain12'
    # folder_GT='../../dataset/RAIN800_DID_MDN_dataset/SPANet_test'
    # folder_Gen='../rgb_in_RAIN1400_FSPANet_detal_15/SPANet_test'
    # folder_GT='../../dataset/RAIN800_DID_MDN_dataset/rain100L_test'
    # folder_Gen='../rgb_in_RAIN1400_FSPANet_detal_15/rain100L_test'
    # folder_GT='../../dataset/RAIN800_DID_MDN_dataset/rain100H_test'
    # folder_Gen='../rgb_in_RAIN1400_FSPANet_detal_15/rain100H_test'
    # folder_GT='../../dataset/RescanDataset/test'
    # folder_Gen='../rgb_in_RAIN1400_FSPANet_detal_15/test'
    # DDN RescanNet PReNet SPANet FSPANet_detail FSPANet_X DSC
    main(folder_GT,folder_Gen)
