from PIL import Image
import numpy as np
import os
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

dir_root = '/mnt/新增磁碟區/RGB_IR資料集/FLIR_1/train/'

rgb_type = 'RGB'
rgb_dir = dir_root + rgb_type

ir_type = 'thermal_8_bit'
ir_dir = dir_root + ir_type


grayscale_training_data = []
thermal_training_data = []
img_number = 0

for fname in sorted(os.listdir(rgb_dir)):
    ir_img_path = ir_dir + '/' + fname[:-4] + '.jpeg'
    rgb_img_path = rgb_dir + '/' + fname

    img_rgb_size = os.path.getsize(rgb_img_path) / 1024
    if os.path.exists(ir_img_path) and img_rgb_size > 100:
        print("{}, {}".format(img_number, fname))
        img_rgb = Image.open(rgb_img_path)
        img_gray = img_rgb.convert('L')
        img_gray_array = np.array(img_gray)
        grayscale_training_data.append(img_gray_array)

        img_ir = Image.open(ir_img_path)
        img_ir_array = np.array(img_ir)
        thermal_training_data.append(img_ir_array)
        img_number += 1

np.save(dir_root + "grayscale_training_data", grayscale_training_data)
np.save(dir_root + "thermal_training_data", thermal_training_data)