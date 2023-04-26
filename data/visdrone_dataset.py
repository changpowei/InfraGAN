import os.path
import random
import torchvision.transforms as transforms
import numpy as np
from data.base_dataset import BaseDataset
import os

class VisDroneDataset(BaseDataset):
    def initialize(self, opt, test=False):
        print('VisDroneDataset')
        self.opt = opt
        self.root = opt.dataroot
        self.dir_AB = os.path.join(opt.dataroot, opt.phase)
        if test:
            self.A_data = np.load(os.path.join(self.root, "rgb_testing_data.npy"), allow_pickle=True)
            self.B_data = np.load(os.path.join(self.root, "thermal_testing_data.npy"), allow_pickle=True)
        else:
            self.A_data = np.load(os.path.join(self.root, "rgb_training_data.npy"), allow_pickle=True)
            self.B_data = np.load(os.path.join(self.root, "thermal_training_data.npy"), allow_pickle=True)

    def __getitem__(self, index):
        A = self.A_data[index]#.transpose(2, 0, 1)
        B = self.B_data[index]#.transpose(2, 0, 1)

        # A = A.resize((self.opt.loadSize, self.opt.loadSize), Image.BICUBIC)
        A = transforms.ToTensor()(A.copy()).float()
        B = transforms.ToTensor()(B.copy()).float()


        w_total = A.size(2)
        w = int(w_total)
        h = A.size(1)
        if self.opt.resize_or_crop == "center_crop":
            w_offset = int((w - self.opt.fineSize) / 2 - 1)
            h_offset = 0

        else:
            w_offset = random.randint(0, max(0, w - self.opt.fineSize - 1))
            h_offset = random.randint(0, max(0, h - self.opt.fineSize - 1))

        A = A[:, h_offset:h_offset + self.opt.fineSize,
            w_offset:w_offset + self.opt.fineSize]
        B = B[:, h_offset:h_offset + self.opt.fineSize,
            w_offset:w_offset + self.opt.fineSize]

        A = transforms.Normalize([0.5], [0.5])(A)
        B = transforms.Normalize([0.5], [0.5])(B)



        return {'A': A, 'B': B}

    def __len__(self):
        return len(self.A_data)

    def name(self):
        return 'VisDrone DATASET'
