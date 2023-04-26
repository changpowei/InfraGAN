import numpy as np
import os

dir_root = '/mnt/新增磁碟區/RGB_IR資料集/FLIR_1/train/'

testing_dataset = False

if testing_dataset:
    A_data = np.load(os.path.join(dir_root, "grayscale_test_data.npy"), allow_pickle=True)
    B_data = np.load(os.path.join(dir_root, "thermal_test_data.npy"), allow_pickle=True)
else:
    A_data = np.load(os.path.join(dir_root, "grayscale_training_data.npy"), allow_pickle=True)
    B_data = np.load(os.path.join(dir_root, "thermal_training_data.npy"), allow_pickle=True)


assert len(A_data) == len(B_data), "資料集長度必須一樣！！！"

for i in range(len(A_data)):
    A = A_data[i]
    B = B_data[i]

    print("{}, RGB:{}, IR:{}".format(i, A.shape, B.shape))
