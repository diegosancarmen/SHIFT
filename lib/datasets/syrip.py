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

class SyRIP(Body16KeypointDataset):
    """SyRIP dataset for Mean Teacher framework

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
    """
    def __init__(self, root, split='train', transforms=None, image_size=(256, 256), **kwargs):
        self.split = split
        normalize = Normalize([0, 0, 0], [1, 1, 1])
        transforms = Compose([
            ResizePad(image_size[0]),
            ToTensor(),
            normalize
        ])
        # self.transforms_base = Compose([ResizePad(image_size[0])]) + transforms_base
        # self.transforms_stu = transforms_stu
        # self.transforms_tea = transforms_tea
        # self.vis = vis

        # Load data
        self.samples = []
        #data = np.load('/data/AmitRoyChowdhury/SyRIP/SyRIP_data_split/300S/SyRIP.npy', allow_pickle=True).item()
        data = np.load('/data/AmitRoyChowdhury/sarosij/SyRIP/prior/SyRIP_200R.npy', allow_pickle=True).item()
        data = data[split]
        for _, item in enumerate(tqdm(data.keys())):
            if split in ['train', 'prior']:
                image_root = "/data/AmitRoyChowdhury/SyRIP/data/syrip/images/train_infant"
                img_path = os.path.join(image_root, f"train{item:05}.jpg")
            elif split == "validate":
                image_root = "/data/AmitRoyChowdhury/SyRIP/data/syrip/images/validate_infant"
                img_path = os.path.join(image_root, f"test{item}.jpg")
                if not os.path.isfile(img_path):
                    print(".", end = " ")
            self.samples.append((img_path, data[item]))

        #self.images, self.db_2d, self.db_3d, self.frame_names = self.read_data() #image, pose_2d, pose_3d, frame_name
        
        self.joints_index = (15, 13, 11, 12, 14, 16, 0, 0, 0, 0, 9, 7, 5, 6, 8, 10)
        self.visible = np.array([1.] * 6 + [0, 0, 0] + [1.] * 7, dtype=np.float32)

        super(SyRIP, self).__init__(root, samples=self.samples, transforms=transforms, image_size=image_size, **kwargs)

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