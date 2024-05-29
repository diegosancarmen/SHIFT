import numpy as np
import os
from PIL import Image
import sys, copy
from tqdm import tqdm
import torch
from .keypoint_dataset import Body16KeypointDataset
from ..transforms.keypoint_detection import *
from .util import *
from ._util import download as download_data, check_exits
from ..transforms.keypoint_detection import Compose, ResizePad
from .util import generate_target

class MiniRGBD(Body16KeypointDataset):
    """MiniRGBD dataset for Mean Teacher framework

    Args:
        root (str): Root directory of dataset
        split (str, optional): Split to use. Default: 'train'.
        transforms_base (callable, optional): Base transformations.
        transforms_stu (callable, optional): Student transformations.
        transforms_tea (callable, optional): Teacher transformations.
        image_size (tuple): (width, height) of the image. Default: (256, 256)
        heatmap_size (tuple): (width, height) of the heatmap. Default: (64, 64)
        sigma (int): sigma parameter when generate the heatmap. Default: 2
        vis (bool): If true, visualize the keypoints.
        # MINIRGBD Joints List
        # global
        # leftThigh
        # rightThigh
        # spine
        # leftCalf
        # rightCalf
        # spine1
        # leftFoot
        # rightFoot
        # spine2
        # leftToes
        # rightToes
        # neck
        # leftShoulder
        # rightShoulder
        # head
        # leftUpperArm
        # rightUpperArm
        # leftForeArm
        # rightForeArm
        # leftHand
        # rightHand
        # leftFingers
        # rightFingers
        # noseVertex
    """
    def __init__(self, root, split='train', transforms=None, image_size=(256, 256), **kwargs):
        self.split = split
        normalize = Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        transforms = Compose([
            ResizePad(image_size[0]),
            ToTensor(),
            normalize
        ])

        # Load data
        self.samples = []
        data = np.load('/data/AmitRoyChowdhury/sarosij/MiniRGBD/MiniRGBD.npy', allow_pickle=True).item()
        data = data[split]
        for _, item in enumerate(tqdm(data.keys())):
            img_name = item.split('_')[1] + '_' + item.split('_')[-1].replace('.txt', '.png')
            img_path = os.path.join(root, f"{item.split('_')[0]}/rgb", img_name)
            self.samples.append((img_path, data[item]['pose_2d']))
        #self.images, self.db_2d, self.db_3d, self.frame_names = self.read_data() #image, pose_2d, pose_3d, frame_name
        
        #(2, 5, 11, 1, 4, 10, 3, 9, 12, 15, 13, 18, 20, 14, 19, 21)
        self.joints_index = (7, 4, 1, 2, 5, 8, 6, 9, 12, 15, 20, 18, 16, 17, 19, 21)
        self.visible = np.ones(16, dtype=np.float32)

        super(MiniRGBD, self).__init__(root, samples=self.samples, transforms=transforms, image_size=image_size, **kwargs)

    def __getitem__(self, index):
        sample = self.samples[index]
        image_name = sample[0]
        image = Image.open(image_name)
        keypoint2d = sample[1][self.joints_index, :2]
        image, data = self.transforms(image, keypoint2d=keypoint2d)
        keypoint2d = data['keypoint2d']

        visible = np.array([1.] * 16, dtype=np.float32)
        visible = visible[:, np.newaxis]

        # 2D heatmap
        target, target_weight = generate_target(keypoint2d, visible, self.heatmap_size, self.sigma, self.image_size)
        target = torch.from_numpy(target)
        target_weight = torch.from_numpy(target_weight)

        meta = {
            'image': image_name,
            'keypoint2d': keypoint2d,  # （NUM_KEYPOINTS x 2）
            'keypoint3d': np.zeros((self.num_keypoints, 3)).astype(keypoint2d.dtype),  # （NUM_KEYPOINTS x 3）
        }
        return image, target, target_weight, meta
    
    # def read_data(self, root, split):
    #     data = np.load('/data/AmitRoyChowdhury/sarosij/MINI-RGBD.npy', allow_pickle=True).item()
    #     data = data[split]
        
    #     imgs = []
    #     pose_2d = []
    #     pose_3d = []
    #     frame_name = []
        
    #     for _, item in enumerate(tqdm(data.keys())):
    #         img_name = item.split('_')[1] + '_' + item.split('_')[-1].replace('.txt', '.png')
    #         img_file = os.path.join(root, f"{item.split('_')[0]}/rgb", img_name)
    #         image = copy.deepcopy(Image.open(img_file))
    #         image = image.resize(self.image_size)
            
    #         if self.flip and np.random.rand(1,)[0] < 0.5:
    #             pose_3d.append(self._random_rotate(self._random_flip((data[item]['pose_3d'] - data[item]['pose_3d'][0:1]).reshape(-1, 3), p=0.5), p=0.5))
    #             pose_2d.append(data[item]['pose_2d'])

    #         pose_2d.append(data[item]['pose_2d'])
    #         pose_3d.append(data[item]['pose_3d'])
    #         K = np.zeros((3, 3), dtype=np.float32)
    #         fx = 588.67905803875317
    #         fy = 590.25690113005601
    #         cx = 322.22048191353628
    #         cy = 237.46785983766890
    #         K[0][0] = fx
    #         K[1][1] = fy
    #         K[0][2] = cx
    #         K[1][2] = cy
    #         K[2][2] = 1
    #         self.K.append(K)
    #         frame_name.append(item)

    #         imgs.append(image)
                    
    #     pose_2d = np.array(pose_2d, dtype=np.float32)
    #     pose_3d = np.array(pose_3d, dtype=np.float32)
    #     frame_name = np.array(frame_name)

    #     if len(pose_2d) != len(pose_3d):
    #         pose_2d = np.zeros_like(pose_3d)
    #         frame_name = np.zeros_like(pose_3d)
    #         self.K = np.zeros_like(pose_3d)

    #     return imgs, pose_2d, pose_3d, frame_name
