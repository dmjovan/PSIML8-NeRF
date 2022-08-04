import os, sys
import glob
import numpy as np
from torch.utils.data import Dataset

class CustomDataset(Dataset):

    def __init__(self, data_dir):

        self.images = np.load(os.path.join(data_dir, 'images.npy'))
        self.poses =  np.load(os.path.join(data_dir, 'poses.npy'))
        self.focal =  np.load(os.path.join(data_dir, 'focal.npy'))

        self.img_h, self.img_w = self.images.shape[1:3]