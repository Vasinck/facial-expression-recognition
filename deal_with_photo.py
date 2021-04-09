import cv2
import numpy as np

def add_gaussian_noise(image_array, mean=0.0, var=30):
    '''
    给数据添加高斯噪声
    '''
    std = var**0.5
    noisy_img = image_array + np.random.normal(mean, std, image_array.shape)
    noisy_img_clipped = np.clip(noisy_img, 0, 255).astype(np.uint8)
    return noisy_img_clipped

def flip_image(image_array):
    '''
    对图像进行水平翻转
    '''
    return cv2.flip(image_array, 1)

def color2gray(image_array):
    '''
    将BGR格式转换成灰度图片
    '''
    gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    gray_img_3d = image_array.copy()
    gray_img_3d[:, :, 0] = gray
    gray_img_3d[:, :, 1] = gray
    gray_img_3d[:, :, 2] = gray
    return gray_img_3d
