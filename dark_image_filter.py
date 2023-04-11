from cv2 import cv2
import numpy as np
import os


def contrast(img0):
    font = cv2.FONT_HERSHEY_SIMPLEX

    hsldark = cv2.cvtColor(img0, cv2.COLOR_BGR2HLS)
    Lchanneld = hsldark[:, :, 1]
    lvalueld = cv2.mean(Lchanneld)[0]

    return lvalueld


dir_root = '/mnt/新增磁碟區/RGB_IR資料集/VisDrone_DroneVehicle/val/'

rgb_type = 'valimg'
rgb_dir = dir_root + rgb_type

img_number = 1

for fname in sorted(os.listdir(rgb_dir)):
    rgb_img_path = rgb_dir + '/' + fname

    img0 = cv2.imread(rgb_img_path)[100:100+512, 100:100+640]
    light_value = contrast(img0)
    print("{}, name = {}, value = {}".format(img_number, fname, light_value))
    img_number += 1
