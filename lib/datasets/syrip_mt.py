import numpy as np
import os
from PIL import Image
import sys, copy
from tqdm import tqdm
import torch
from .keypoint_dataset import Body16KeypointDataset
from ..transforms.keypoint_detection import Compose, ResizePad
from .util import generate_target

class SyRIP_mt(Body16KeypointDataset):
    """miniRGBD dataset for Mean Teacher framework

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
    def __init__(self, root, split='train', transforms_base=None, transforms_stu=None, transforms_tea=None, image_size=(256, 256), heatmap_size=(64, 64), sigma=2, vis=False, **kwargs):
        self.split = split
        self.transforms_base = Compose([ResizePad(image_size[0])]) + transforms_base
        self.transforms_stu = transforms_stu
        self.transforms_tea = transforms_tea
        self.vis = vis
        self.k = 1
                
        self.joints_index = (15, 13, 11, 12, 14, 16, 0, 0, 0, 0, 9, 7, 5, 6, 8, 10)
        self.visible = np.array([1.] * 6 + [0, 0, 0] + [1.] * 7, dtype=np.float32)

        # Load data
        self.samples = []
        data = np.load('/data/AmitRoyChowdhury/InfantUDA/SHIFT/lib/datasets/infant_annotations/SyRIP_all.npy', allow_pickle=True).item()
        data = data[split]
        for _, item in enumerate(tqdm(data.keys())):
            if split in ['train', 'prior']:
                image_root = "/data/AmitRoyChowdhury/SyRIP/data/syrip/images/train_infant/"
                img_path = os.path.join(image_root, f"train{item:05}.jpg")
            elif split == "validate":
                image_root = "/data/AmitRoyChowdhury/SyRIP/data/syrip/images/validate_infant"
                img_path = os.path.join(image_root, f"test{item}.jpg")
                if not os.path.isfile(img_path):
                    print(".", end = " ")
            self.samples.append((img_path, data[item]))
        #self.images, self.db_2d, self.db_3d, self.frame_names = self.read_data() #image, pose_2d, pose_3d, frame_name

        super(SyRIP_mt, self).__init__(root, samples=self.samples, image_size=image_size, heatmap_size=heatmap_size, sigma=sigma, **kwargs)

    def __getitem__(self, index):
        sample = self.samples[index]
        image_name = sample[0]
        #image = copy.deepcopy(Image.open(image_name))
        image = Image.open(image_name).convert('RGB')
        keypoint2d = sample[1][self.joints_index, :2]
        image, data = self.transforms_base(image, keypoint2d=keypoint2d, intrinsic_matrix=None)
        keypoint2d = data['keypoint2d']

        image_stu, data_stu = self.transforms_stu(image, keypoint2d=keypoint2d, intrinsic_matrix=None)
        keypoint2d_stu = data_stu['keypoint2d']
        aug_param_stu = data_stu['aug_param']

        visible = np.array([1.] * 16, dtype=np.float32)
        visible = visible[:, np.newaxis]
        # print('keypoint2d_stu:', keypoint2d_stu.shape, visible.shape)
        # 2D heatmap
        target_stu, target_weight_stu = generate_target(keypoint2d_stu, visible, self.heatmap_size, self.sigma, self.image_size)
        target_stu = torch.from_numpy(target_stu)
        target_weight_stu = torch.from_numpy(target_weight_stu)

        target_ori, target_weight_ori = generate_target(keypoint2d, visible, self.heatmap_size, self.sigma, self.image_size)
        target_ori = torch.from_numpy(target_ori)
        target_weight_ori = torch.from_numpy(target_weight_ori)

        meta_stu = {
            'image': image_name,
            'target_small_stu': generate_target(keypoint2d_stu, visible, (8, 8), self.sigma, self.image_size),
            'keypoint2d_ori': keypoint2d,
            'target_ori': target_ori,
            'target_weight_ori': target_weight_ori,
            'keypoint2d_stu': keypoint2d_stu,
            'aug_param_stu': aug_param_stu,
        }

        images_tea, targets_tea, target_weights_tea, metas_tea = [], [], [], []
        for _ in range(self.k):
            image_tea, data_tea = self.transforms_tea(image, keypoint2d=keypoint2d, intrinsic_matrix=None)
            keypoint2d_tea = data_tea['keypoint2d']
            aug_param_tea = data_tea['aug_param']

            # 2D heatmap
            target_tea, target_weight_tea = generate_target(keypoint2d_tea, visible, self.heatmap_size, self.sigma, self.image_size)
            target_tea = torch.from_numpy(target_tea)
            target_weight_tea = torch.from_numpy(target_weight_tea)

            meta_tea = {
                'image': image_name,
                'target_small_tea': generate_target(keypoint2d_tea, visible, (8, 8), self.sigma, self.image_size),
                'keypoint2d_tea': keypoint2d_tea,
                'aug_param_tea': aug_param_tea,
            }
            images_tea.append(image_tea)
            targets_tea.append(target_tea)
            target_weights_tea.append(target_weight_tea)
            metas_tea.append(meta_tea)

        if self.vis:
            self.visualize(image, keypoint2d, os.path.join(self.root, 'visualization', f'{image_name}.jpg'))

        return image_stu, target_stu, target_weight_stu, meta_stu, images_tea, targets_tea, target_weights_tea, metas_tea