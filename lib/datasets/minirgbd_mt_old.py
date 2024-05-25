import numpy as np
from PIL import Image
import os, sys, copy
import pickle
from prettytable import PrettyTable
from webcolors import name_to_rgb
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

class MiniRGBD_mt:

    # TODO: add image
    head = (9,)
    shoulder = (12, 13)
    elbow = (11, 14)
    wrist = (10, 15)
    hip = (2, 3)
    knee = (1, 4)
    ankle = (0, 5)
    all = (12, 13, 11, 14, 10, 15, 2, 3, 1, 4, 0, 5)
    right_leg = (0, 1, 2, 8)
    left_leg = (5, 4, 3, 8)
    backbone = (8, 9)
    right_arm = (10, 11, 12, 8)
    left_arm = (15, 14, 13, 8)

    # self.keypoints = ['nose', 'leftEye', 'rightEye', 'leftEar', 'rightEar', 'leftShoulder', 
    #               'rightShoulder', 'leftElbow', 'rightElbow', 'leftWrist', 'rightWrist', 
    #               'leftHip', 'rightHip', 'leftKnee', 'rightKnee', 'leftAnkle', 'rightAnkle']

    # self.flip_pairs = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16)]
    # self.parent = [0, 0, 1, 2, 0, 0, 5, 6, 7, 8, 5, 6, 11, 12, 13, 14]
    # self.change_to_17 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]=
    
    def __init__(self,  root, heatmap_size, transforms_base=None, transforms_stu=None, transforms_tea=None, subset='train', 
        gt2d=True, read_confidence=True, sample_interval=None, rep=1, 
        flip=False, cond_3d_prob=0, abs_coord=False, rot=False,
        num_joint=16,norm_2d=False,aug=False,cls=False,scale=1.0,normed=False, sigma=2, image_size=(256, 256)):
        
        assert subset in ['train', 'validate', 'all']

        self.subset = subset
        self.heatmap_size = heatmap_size
        self.root = root
        self.transforms_base = Compose([ResizePad(image_size[0])]) + transforms_base
        self.transforms_stu = transforms_stu
        self.transforms_tea = transforms_tea
        self.k = 1
        self.gt_trainset = None
        self.gt_testset = None
        self.dt_dataset = None
        self.num_keypoints = num_joint
        self.subset = subset
        self.sigma = sigma
        self.image_size = image_size
        self.gt2d = gt2d
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
            16: [i for i in range(16)],
            14: [0, 1, 2, 3, 4, 5, 6, 8, 10, 11, 12, 13, 14] #Difference of 2
        }
        self.keypoints_group = {
            "head": self.head,
            "shoulder": self.shoulder,
            "elbow": self.elbow,
            "wrist": self.wrist,
            "hip": self.hip,
            "knee": self.knee,
            "ankle": self.ankle,
            "all": self.all
        }
        self.change = [0,2,5,11,1,4,10,3,9,12,15,13,18,20,14,19,21]
        #self.change = [0,2,5,11,1,4,10,3,9,12,15,13,14,15] #16 keypoints
        self.left_joints = [1, 4, 7, 10, 13, 16, 18, 20, 22]
        self.right_joints = [2, 5, 8, 11, 14, 17, 19, 21, 23]
        self.cond_3d_prob = cond_3d_prob
        # self.change_to_12 = [5, 7, 9, 11, 13, 15, 6, 8, 10, 12, 14, 16] #12 keypoints: removed head and neck.
        #self.change_to_16 = [0, 5, 7, 9, 11, 13, 15, 2, 4, 6, 8, 10, 12, 14, 16] #16 keypoints: removed head and neck.
        self.change_to_16 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16] #16 keypoints: removed head and neck.
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

    def visualize(self, image, keypoints, filename):
        """Visualize an image with its keypoints, and store the result into a file

        Args:
            image (PIL.Image):
            keypoints (torch.Tensor): keypoints in shape K x 2
            filename (str): the name of file to store
        """
        assert self.colored_skeleton is not None

        image = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR).copy()
        if keypoints is not None:
            for (_, (line, color)) in self.colored_skeleton.items():
                color = name_to_rgb(color) if type(color) == str else color
                for i in range(len(line) - 1):
                    print('keypoints', keypoints)
                    print('line_i', line[i])
                    print('line_i1', line[i + 1])
                    # start, end = keypoints[line[i]], keypoints[line[i + 1]]
                    # cv2.line(image, (int(start[0]), int(start[1])), (int(end[0]), int(end[1])), color=color,
                    #         thickness=3)
            for keypoint in keypoints:
                cv2.circle(image, (int(keypoint[0]), int(keypoint[1])), 3, name_to_rgb('black'), 1)
        cv2.imwrite(filename, image)

    def group_accuracy(self, accuracies):
        """ Group the accuracy of K keypoints into different kinds.

        Args:
            accuracies (list): accuracy of the K keypoints

        Returns:
            accuracy of ``N=len(keypoints_group)`` kinds of keypoints

        """
        grouped_accuracies = dict()
        for name, keypoints in self.keypoints_group.items():
            grouped_accuracies[name] = sum([accuracies[idx] for idx in keypoints]) / len(keypoints)
        return grouped_accuracies
        
        
    def norm(self,pose_3d):
        pose_3d = 2*(pose_3d - pose_3d.min())/(pose_3d.max()-pose_3d.min())-1
        return pose_3d
        import ipdb;ipdb.set_trace()
        

    def __getitem__(self, idx):
        """
        Return: [16, 2], [16, 3] for data and labels
        """
        w = 480
        h = 640
        #  % self.real_data_len
        image = self.images[idx]
        data_2d = self.db_2d[idx]
        data_3d = self.db_3d[idx]
        image_name = self.frame_names[idx]
        K = self.K[idx]
        # print(data_2d.shape)

        image, data = self.transforms_base(image, keypoint2d=data_2d, intrinsic_matrix=None)
        data_2d = data['keypoint2d']

        image_stu, data_stu = self.transforms_stu(image, keypoint2d=data_2d, intrinsic_matrix=None)
        data_2d_stu = data_stu['keypoint2d']
        aug_param_stu = data_stu['aug_param']

        if self.cls:
            data_2d = np.concatenate([data_2d,np.ones((data_2d.shape[0],1))],axis=-1)

        # normalize 2D pose, adapted from SURREAL.
        visible = np.array([1.] * self.num_keypoints, dtype=np.float32)
        visible = visible[:, np.newaxis]

        # 2D heatmap
        # print(data_2d.shape, data_2d_stu.shape, data_2d_stu.shape[0], visible.shape)
        # exit()
        target_stu, target_weight_stu = generate_target(data_2d_stu, visible, self.heatmap_size, self.sigma, self.image_size)
        target_stu = torch.from_numpy(target_stu)
        target_weight_stu = torch.from_numpy(target_weight_stu)


        target_ori, target_weight_ori = generate_target(data_2d, visible, self.heatmap_size, self.sigma, self.image_size)
        target_ori = torch.from_numpy(target_ori)
        target_weight_ori = torch.from_numpy(target_weight_ori)

        meta_stu = {
            'image': image_name,
            'target_small_stu': generate_target(data_2d_stu, visible, (8, 8), self.sigma, self.image_size),
            'keypoint2d_ori': data_2d,
            'target_ori': target_ori,  
            'target_weight_ori': target_weight_ori,
            'keypoint2d_stu': data_2d_stu, # （NUM_KEYPOINTS x 2）
            'aug_param_stu': aug_param_stu,
        }

        images_tea, targets_tea, target_weights_tea, metas_tea = [], [], [], []
        for _ in range(self.k):
            image_tea, data_tea = self.transforms_tea(image, keypoint2d=data_2d, intrinsic_matrix=None)
            keypoint2d_tea = data_tea['keypoint2d']
            aug_param_tea = data_tea['aug_param']

            # 2D heatmap
            target_tea, target_weight_tea = generate_target(keypoint2d_tea, visible, self.heatmap_size, self.sigma, self.image_size)
            target_tea = torch.from_numpy(target_tea)
            target_weight_tea = torch.from_numpy(target_weight_tea)

            meta_tea = {
                'image': image_name,
                'target_small_tea': generate_target(keypoint2d_tea, visible, (8, 8), self.sigma, self.image_size),
                'keypoint2d_tea': keypoint2d_tea,  # （NUM_KEYPOINTS x 2）
                'aug_param_tea': aug_param_tea,
            }
            images_tea.append(image_tea)
            targets_tea.append(target_tea)
            target_weights_tea.append(target_weight_tea) 
            metas_tea.append(meta_tea)

        return image_stu, target_stu, target_weight_stu, meta_stu, images_tea, targets_tea, target_weights_tea, metas_tea
    

    def __len__(self,):
        return len(self.db_2d) * self.rep
    
    def visualize(self, image, keypoints, filename):
        """Visualize an image with its keypoints, and store the result into a file

        Args:
            image (PIL.Image):
            keypoints (torch.Tensor): keypoints in shape K x 2
            filename (str): the name of file to store
        """
        assert self.colored_skeleton is not None

        image = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR).copy()
        if keypoints is not None:
            for (_, (line, color)) in self.colored_skeleton.items():
                color = name_to_rgb(color) if type(color) == str else color
                for i in range(len(line) - 1):
                    start, end = keypoints[line[i]], keypoints[line[i + 1]]
                    cv2.line(image, (int(start[0]), int(start[1])), (int(end[0]), int(end[1])), color=color,
                            thickness=3)
            for keypoint in keypoints:
                cv2.circle(image, (int(keypoint[0]), int(keypoint[1])), 3, name_to_rgb('black'), 1)
        cv2.imwrite(filename, image)
    
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
            
            # self.visualize(image, data[item]['pose_2d'], f'{img_name}.png')
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
            

        # if self.num_joint==12:
        #     pose_2d = pose_2d[:,self.change_to_12,:]
        #     pose_3d = pose_3d[:,self.change_to_12,:]

        if self.num_joint==16:
            pose_2d = pose_2d[:,self.change_to_16,:]
            pose_3d = pose_3d[:,self.change_to_16,:]

        return imgs, pose_2d, pose_3d, frame_name

    @staticmethod
    def get_skeleton():
        return [[0, 1], [1, 2], [2, 3], [0, 4], [4, 5], [5, 6], 
        [0, 7], [7, 8], [8, 9], [9, 10], [8, 11], [11, 12], [12, 13], 
        [8, 14], [14, 15], [15, 16]]