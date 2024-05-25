import numpy as np
from PIL import Image
import os, sys, copy
import pickle
from prettytable import PrettyTable
import random
import math as m
import torchvision
import torchvision.transforms.transforms as T
from ..transforms.keypoint_detection import *
import math
import torch
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R

from .util import *

class SYRIP:
    def __init__(self,  root, transforms, heatmap_size, subset='train', 
        gt2d=True, read_confidence=True, sample_interval=None, rep=1, 
        flip=False, cond_3d_prob=0, abs_coord=False, rot=False,
        num_joint=17,norm_2d=False,truncated=False,aug=False, sigma=2, image_size=(256, 256)):
        
        self.root = root
        self.transforms = transforms
        self.heatmap_size = heatmap_size
        self.gt_trainset = None
        self.gt_testset = None
        self.dt_dataset = None
        self.root = None
        self.subset = subset
        self.gt2d = gt2d
        self.sigma = sigma
        self.image_size = image_size
        self.read_confidence = read_confidence
        self.sample_interval = sample_interval
        self.flip = flip
        self.camera_param = None
        self.abs_coord = abs_coord
        self.rot = rot
        self.image_name = []
        self.action = []
        self.num_joint = num_joint
        self.norm_2d = norm_2d
        self.joint_match = {
            17: [i for i in range(17)],
            15: [0, 1, 2, 3, 4, 5, 6, 8, 10, 11, 12, 13, 14, 15, 16]
        }
        self.change_2d = [-1,-3,-5,-6,-4,-2,-7,-9,-11,-12,-10,-8]
        self.change_12 = [2,1,0,3,4,5,-3,-2,-1,-4,-5,-6]
        self.K = []
        self.left_joints = [3,4,5,9,10,11]
        self.right_joints = [0,1,2,6,7,8]
        self.cond_3d_prob = cond_3d_prob
        self.truncated = truncated
        self.aug  = aug
        self.db_2d, self.db_3d, self.frame_name= self.read_data()
    
        if self.sample_interval:
            self._sample(sample_interval)

        self.rep = rep
        if self.rep > 1:
            print(f'stack dataset {self.rep} times for multi-sample eval')

        self.real_data_len = len(self.db_2d)

    def __init__(self,  root, transforms, heatmap_size, subset='train', 
        gt2d=True, read_confidence=True, sample_interval=None, rep=1, 
        flip=False, cond_3d_prob=0, abs_coord=False, rot=False,
        num_joint=17,norm_2d=False,aug=False,cls=False,scale=1.0,normed=False, sigma=2, image_size=(256, 256)):
        
        self.root = root
        self.transforms = transforms
        self.gt_trainset = None
        self.gt_testset = None
        self.dt_dataset = None
        self.num_keypoints = num_joint
        self.subset = subset
        self.sigma = sigma
        self.image_size = image_size
        self.gt2d = gt2d
        self.heatmap_size = heatmap_size
        self.K = []
        self.read_confidence = read_confidence
        self.sample_interval = sample_interval
        self.flip = flip
        self.camera_param = None
        self.abs_coord = abs_coord
        self.rot = rot
        self.image_name = []
        self.action = []
        self.num_joint = num_joint
        self.norm_2d = norm_2d
        self.joint_match = {
            17: [i for i in range(17)],
            15: [0, 1, 2, 3, 4, 5, 6, 8, 10, 11, 12, 13, 14, 15, 16]
        }
        self.change = [0,2,5,11,1,4,10,3,9,12,15,13,18,20,14,19,21]
        self.left_joints = [1, 4, 7, 10,13,16,18,20,22]
        self.right_joints = [2, 5, 8, 11, 14, 17, 19, 21, 23]
        self.cond_3d_prob = cond_3d_prob
        self.change_to_12 = [1, 2, 3, 4, 5, 6,  11, 12, 13, 14, 15, 16]
        self.aug = aug
        self.normed= normed
        
        
        # self.image, self.db_2d, self.db_3d, self.frame_name = self.read_data()
        self.images, self.db_2d, self.db_3d, self.frame_names = self.read_data() #image, pose_2d, pose_3d, frame_name
    
        if self.sample_interval:
            self._sample(sample_interval)

        self.rep = rep
        if self.rep > 1:
            print(f'stack dataset {self.rep} times for multi-sample eval')

        self.real_data_len = len(self.db_2d)
        self.cls = cls
        self.scale = scale
        
        
    def norm(self,pose_3d):
        pose_3d = 2*(pose_3d - pose_3d.min())/(pose_3d.max()-pose_3d.min())-1
        return pose_3d
        import ipdb;ipdb.set_trace()
        

    def __getitem__(self, idx):
        """
        Return: [17, 2], [17, 3] for data and labels
        """
        w = 480
        h = 640
        #  % self.real_data_len
        image = self.images[idx]
        data_2d = self.db_2d[idx]
        data_3d = self.db_3d[idx]
        image_name = self.frame_names[idx]
        K = self.K[idx]

        image, data = self.transforms(image, keypoint2d=data_2d, intrinsic_matrix=K)

        if self.cls:
            data_2d = np.concatenate([data_2d,np.ones((data_2d.shape[0],1))],axis=-1)

        # normalize 2D pose, adapted from SURREAL.
        visible = np.array([1.] * 17, dtype=np.float32)
        visible = visible[:, np.newaxis]

        # 2D heatmap
        target, target_weight = generate_target(data_2d, visible, self.heatmap_size, self.sigma, self.image_size)
        target = torch.from_numpy(target)
        target_weight = torch.from_numpy(target_weight)

        meta = {
            'image': image_name,
            'keypoint2d': data_2d,  # （NUM_KEYPOINTS x 2）
            'keypoint3d': data_3d,  # （NUM_KEYPOINTS x 3）
        }

        # return image, data_2d, data_3d,K,np.array([0,1])
        return image, target, target_weight, meta
    

    def __len__(self,):
        return len(self.db_2d) * self.rep
    

    def _random_flip(self, data, p=0.5):
        """
        Flip with prob p
        data: [17, 2] or [17, 3]
        """
        if np.random.rand(1,)[0] < p:
            data = data.copy()
            data[:, 0] *= -1  # flip x of all joints
            data[self.left_joints+self.right_joints] = data[self.right_joints+self.left_joints]
        return data
    
    
    def _random_rotate(self, data, p=0.5):
        """
        Flip with prob p
        data: [17, 2] or [17, 3]
        """
       
        try:
            data = data.reshape(data.shape[0],-1)
        except: import ipdb;ipdb.set_trace()
        if data.shape[-1]==2:
            ones_column = np.ones((data.shape[0], 1))
            data = np.hstack((data,ones_column))
        if np.random.rand(1,)[0] < p:
            data = data.copy()
            data = R.random().as_matrix().dot(data.T).T
        return data
    
    def save_action(self,action):
       
        self.action = action
        assert len(self.db_3d)==len(self.action)
        return self.action

    def add_noise(self, pose2d, std=5, noise_type='gaussian'):
        """
        pose2d: [B, j, 2]
        """
        if noise_type == 'gaussian':
            noise = std * np.random.randn(*pose2d.shape).astype(np.float32)
            pose2d = pose2d + noise
        elif noise_type == 'uniform':
          
            noise = std * (np.random.rand(*pose2d.shape).astype(np.float32) - 0.5)
            pose2d = pose2d + noise
        else:
            raise NotImplementedError
        return pose2d

    def _sample(self, sample_interval):
        print(f'Class H36MDataset({self.subset}): sample dataset every {sample_interval} frame')
        self.db_2d = self.db_2d[::sample_interval]
        self.db_3d = self.db_3d[::sample_interval]
        self.image_name = self.image_name[::sample_interval]
        self.K = self.K[::sample_interval]

    def read_data(self):
       
        data = np.load('/data/AmitRoyChowdhury/sarosij/MINI-RGBD.npy',allow_pickle=True).item()
        data = data[self.subset]
        
        imgs = []
        pose_2d = []
        pose_3d = []
        frame_name = []
        
        for _, item in enumerate(tqdm(data.keys())):

            img_name = item.split('_')[1] + '_' + item.split('_')[-1].replace('.txt', '.png')
            img_file = os.path.join(self.root, f"{item.split('_')[0]}/rgb", img_name)
            image = copy.deepcopy(Image.open(img_file))
            image = image.resize(self.image_size)
            # image = self.transforms(image)
            
           
            if self.flip and np.random.rand(1,)[0]<0.5:
                pose_3d.append(self._random_rotate(self. _random_flip((data[item]['pose_3d']-data[item]['pose_3d'][0:1]).reshape(-1,3), p=0.5),p=0.5))
                pose_2d.append(data[item]['pose_2d'])
            
            pose_2d.append(data[item]['pose_2d'])
            pose_3d.append(data[item]['pose_3d'])
            K = np.zeros((3, 3), dtype=np.float32)
            fx = 588.67905803875317
            fy = 590.25690113005601
            cx = 322.22048191353628
            cy = 237.46785983766890
            K[0][0] = fx
            K[1][1] = fy
            K[0][2] = cx
            K[1][2] = cy
            K[2][2] = 1
            self.K.append(K)
            frame_name.append(item)

            imgs.append(image)
            #image.close()
                    
        pose_2d = np.array(pose_2d,dtype=np.float32)
        pose_3d = np.array(pose_3d,dtype=np.float32)
        frame_name = np.array(frame_name)
        
        
        if self.num_joint == 17:
            pose_2d = pose_2d[:,self.change]
            pose_3d = pose_3d[:,self.change]
            
     
        if self.aug: 
            aug_data = np.load('aug_mini.npy')
            
            for aug in aug_data:
                aug/= np.random.uniform(0.8,1.2)
            pose_3d = np.concatenate([pose_3d,aug_data],axis=0)

        if len(pose_2d) != len(pose_3d):
            pose_2d = np.zeros_like(pose_3d)
            frame_name = np.zeros_like(pose_3d)
            self.K = np.zeros_like(pose_3d)
            

        if self.num_joint==12:
            pose_2d = pose_2d[:,self.change_to_12,:]
            pose_3d = pose_3d[:,self.change_to_12,:]

        return imgs, pose_2d, pose_3d, frame_name

    @staticmethod
    def get_skeleton():
        return [[0, 1], [1, 2], [2, 3], [0, 4], [4, 5], [5, 6], 
        [0, 7], [7, 8], [8, 9], [9, 10], [8, 11], [11, 12], [12, 13], 
        [8, 14], [14, 15], [15, 16]]