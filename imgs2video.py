import cv2
import numpy as np
import os

frameSize = (512, 512)

out = cv2.VideoWriter('./results/infragan_visdrone/output_video.avi',cv2.VideoWriter_fourcc(*'DIVX'), 2, frameSize)

for i in range(5, 185, 5):
    filename = './results/infragan_visdrone/test_{}/{}_fake_B.png'.format(i, i)
    if os.path.isfile(filename):
        img = cv2.imread(filename)
        cv2.putText(img=img, text=str(i), org=(450, 450), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=1,
                    color=(255, 0, 0), thickness=1)
        out.write(img)

out.release()