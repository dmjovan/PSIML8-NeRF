import os
import cv2
import glob
import numpy as np
from torch.utils.data import Dataset

class CustomDataset(Dataset):

    def __init__(self, config):
        
        self.img_h=config["experiment"]["height"]
        self.img_w=config["experiment"]["width"]
        
        self.rgb_dir = os.path.join(config["experiment"]["dataset_dir"], "images1")
        self.poses_file = os.path.join(config["experiment"]["dataset_dir"], "poses_old.npy")
        self.camera_file = os.path.join(config["experiment"]["dataset_dir"], 'camera_matrix.npy')

        self.poses =  np.load(self.poses_file).reshape(-1, 4, 4)
        self.K =  np.load(self.camera_file).reshape(3, 3)

        self.train_ids = list(range(0, 40))
        self.train_num = 40

        self.test_ids = list(range(40, 50))
        self.test_num = 10

        self.rgb_list = sorted(glob.glob(self.rgb_dir + '/*.jpg'), key=lambda file_name: int((os.path.split(file_name)[1])[:-4]))

        self.train_samples = {'image': [], 'depth': [], 'pose': []}
        self.test_samples = {'image': [], 'depth': [], 'pose': []}

       # training samples
        for idx in self.train_ids:
            image = cv2.imread(self.rgb_list[idx])[:,:,::-1] / 255.0

            depth = np.random.uniform(low=config["render"]["depth_range"][0]/config["render"]["depth_range"][1], high=1.0, size=image[:,:,0].shape)
            pose = self.poses[idx]

            self.train_samples["image"].append(image)
            self.train_samples["depth"].append(depth)
            self.train_samples["pose"].append(pose)

        # test samples
        for idx in self.test_ids:
            image = cv2.imread(self.rgb_list[idx])[:,:,::-1] / 255.0
            depth = np.random.uniform(low=config["render"]["depth_range"][0]/config["render"]["depth_range"][1], high=1.0, size=image[:,:,0].shape)
            pose = self.poses[idx]

            self.test_samples["image"].append(image)
            self.test_samples["depth"].append(depth)
            self.test_samples["pose"].append(pose)

        for key in self.test_samples.keys():  # transform list of np array to array with batch dimension
            self.train_samples[key] = np.asarray(self.train_samples[key])
            self.test_samples[key] = np.asarray(self.test_samples[key])

        print("#####################################################################")
        print("Training Sample Summary:")
        for key in self.train_samples.keys(): 
            print(f"{key} has shape of {self.train_samples[key].shape}, type {self.train_samples[key].dtype} ")
        
        print("#####################################################################")
        print("Testing Sample Summary:")
        for key in self.test_samples.keys(): 
            print(f"{key} has shape of {self.test_samples[key].shape}, type {self.test_samples[key].dtype}")