from PIL import Image
import numpy as np
import os
from PIL import ImageFile
from cv2 import cv2
ImageFile.LOAD_TRUNCATED_IMAGES = True


def contrast(img0):
    font = cv2.FONT_HERSHEY_SIMPLEX

    hsldark = cv2.cvtColor(img0, cv2.COLOR_BGR2HLS)
    Lchanneld = hsldark[:, :, 1]
    lvalueld = cv2.mean(Lchanneld)[0]

    return lvalueld

if __name__ == '__main__':
    dir_root = '/mnt/新增磁碟區/RGB_IR資料集/VisDrone_DroneVehicle/train/'

    rgb_type = 'trainimg'
    rgb_dir = dir_root + rgb_type

    ir_type = 'trainimgr'
    ir_dir = dir_root + ir_type

    rgb_training_data = []
    thermal_training_data = []
    img_number = 0

    for fname in sorted(os.listdir(rgb_dir)):
        ir_img_path = ir_dir + '/' + fname
        rgb_img_path = rgb_dir + '/' + fname

        light_value = contrast(cv2.imread(rgb_img_path)[100:100+512, 100:100+640])

        if os.path.exists(ir_img_path) and os.path.exists(rgb_img_path) and light_value >= 30:
            print("{}, {}".format(img_number, fname))
            bounding_box = (100, 100, 640 + 100, 512 + 100)

            img_rgb = Image.open(rgb_img_path).crop(bounding_box)
            img_rgb_array = np.array(img_rgb)
            rgb_training_data.append(img_rgb_array)

            img_ir = Image.open(ir_img_path).crop(bounding_box).convert('L')
            img_ir_array = np.array(img_ir)
            thermal_training_data.append(img_ir_array)

            img_number += 1

    np.save(dir_root + "rgb_training_data", rgb_training_data)
    np.save(dir_root + "thermal_training_data", thermal_training_data)