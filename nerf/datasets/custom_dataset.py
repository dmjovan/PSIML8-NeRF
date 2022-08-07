import os
import cv2
import glob
import numpy as np
from torch.utils.data import Dataset

class CustomDataset(Dataset):

    def __init__(self, config):
        
        self.img_h=config["experiment"]["height"]
        self.img_w=config["experiment"]["width"]
        
        self.rgb_dir = os.path.join(config["experiment"]["dataset_dir"], "images")
        self.poses_file = os.path.join(config["experiment"]["dataset_dir"], "poses.npy")
        self.focal_file = os.path.join(config["experiment"]["dataset_dir"], 'focal.npy')

        self.poses =  np.load(self.poses_file).reshape(-1, 4, 4)
        self.focal =  np.load(self.focal_file)

        self.train_ids = list(range(0, len(os.listdir(self.rgb_dir)), 2))
        self.train_num = len(self.train_ids)

        self.test_ids = [x + 1 for x in self.train_ids]
        self.test_num = len(self.test_ids)

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
            depth = np.random.uniform(low=config["render"]["depth_range"][0]/config["render"]["depth_range"][1], size=image[:,:,0].shape)
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