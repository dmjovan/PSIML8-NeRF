import os, sys
import glob
import numpy as np
from skimage.io import imread
from torch.utils.data import Dataset
import cv2
import imageio
from imgviz import label_colormap

class ReplicaDataset(Dataset):

    def __init__(self, data_dir, train_ids, test_ids, img_h=None, img_w=None):

        traj_file = os.path.join(data_dir, "traj_w_c.txt")
        self.rgb_dir = os.path.join(data_dir, "rgb")
        self.depth_dir = os.path.join(data_dir, "depth")  # depth is in mm uint

        self.train_ids = train_ids
        self.train_num = len(train_ids)
        self.test_ids = test_ids
        self.test_num = len(test_ids)

        self.img_h = img_h
        self.img_w = img_w

        self.Ts_full = np.loadtxt(traj_file, delimiter=" ").reshape(-1, 4, 4)

        self.rgb_list = sorted(glob.glob(self.rgb_dir + '/rgb*.png'), key=lambda file_name: int(file_name.split("_")[-1][:-4]))
        self.depth_list = sorted(glob.glob(self.depth_dir + '/depth*.png'), key=lambda file_name: int(file_name.split("_")[-1][:-4]))

        self.train_samples = {'image': [], 'depth': [], 'T_wc': []}
        self.test_samples = {'image': [], 'depth': [], 'T_wc': []}

       # training samples
        for idx in train_ids:
            image = cv2.imread(self.rgb_list[idx])[:,:,::-1] / 255.0  # change from BGR uinit 8 to RGB float
            depth = cv2.imread(self.depth_list[idx], cv2.IMREAD_UNCHANGED) / 1000.0  # uint16 mm depth, then turn depth from mm to meter

            if (self.img_h is not None and self.img_h != image.shape[0]) or (self.img_w is not None and self.img_w != image.shape[1]):
                image = cv2.resize(image, (self.img_w, self.img_h), interpolation=cv2.INTER_LINEAR)
                depth = cv2.resize(depth, (self.img_w, self.img_h), interpolation=cv2.INTER_LINEAR)

            T_wc = self.Ts_full[idx]

            self.train_samples["image"].append(image)
            self.train_samples["depth"].append(depth)
            self.train_samples["T_wc"].append(T_wc)


        # test samples
        for idx in test_ids:
            image = cv2.imread(self.rgb_list[idx])[:,:,::-1] / 255.0  # change from BGR uinit 8 to RGB float
            depth = cv2.imread(self.depth_list[idx], cv2.IMREAD_UNCHANGED) / 1000.0  # uint16 mm depth, then turn depth from mm to meter

            if (self.img_h is not None and self.img_h != image.shape[0]) or (self.img_w is not None and self.img_w != image.shape[1]):
                image = cv2.resize(image, (self.img_w, self.img_h), interpolation=cv2.INTER_LINEAR)
                depth = cv2.resize(depth, (self.img_w, self.img_h), interpolation=cv2.INTER_LINEAR)
                
            T_wc = self.Ts_full[idx]

            self.test_samples["image"].append(image)
            self.test_samples["depth"].append(depth)
            self.test_samples["T_wc"].append(T_wc)

        for key in self.test_samples.keys():  # transform list of np array to array with batch dimension
            self.train_samples[key] = np.asarray(self.train_samples[key])
            self.test_samples[key] = np.asarray(self.test_samples[key])

        self.colour_map_np = label_colormap()[self.semantic_classes]
        self.mask_ids = np.ones(self.train_num)  # init self.mask_ids as full ones
        # 1 means the correspinding label map is used for semantic loss during training, while 0 means no semantic loss is applied on this frame

        print("#####################################################################")
        print("Training Sample Summary:")
        for key in self.train_samples.keys(): 
            print("{} has shape of {}, type {}.".format(key, self.train_samples[key].shape, self.train_samples[key].dtype))
        print("#####################################################################")
        print("Testing Sample Summary:")
        for key in self.test_samples.keys(): 
            print("{} has shape of {}, type {}.".format(key, self.test_samples[key].shape, self.test_samples[key].dtype))