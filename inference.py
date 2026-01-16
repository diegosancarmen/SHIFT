# python inference.py --image-path 'path/to/MINI-RGBD_web/01/rgb/syn_00100.png' --output-dir 'inference' --uda-path 'checkpoints/surreal2mini_1/syn2syn_pose_resnet101/checkpoints_2024-06-09-00_17_17/best.pth' --num-keypoints 16 --image-size 256
import argparse
import os
import re
import cv2
import torch
import warnings
import numpy as np
from webcolors import name_to_rgb
import torchvision.transforms as transforms
from torchvision.transforms import Compose, ToPILImage
from torch.optim import Adam, SGD
from PIL import Image
import matplotlib.pyplot as plt

import lib.models as models
import lib.datasets as datasets
from lib.models.seg_net import SegNet, get_segmentation_masks
from lib.transforms import Denormalize
import lib.transforms.keypoint_detection as T
from lib.keypoint_detection import *
from utils import *

warnings.filterwarnings("ignore", category=UserWarning)

def load_models(uda_path, arch, num_keypoints, device):

    student = models.__dict__[arch](num_keypoints=num_keypoints).to(device)
    student = torch.nn.DataParallel(student).cuda()
    stu_optimizer = Adam(student.parameters(), lr=args.lr)

    checkpoint = torch.load(uda_path, map_location='cpu')
    student.load_state_dict(checkpoint['student'])
    stu_optimizer.load_state_dict(checkpoint['stu_optimizer'])

    dict_path = os.path.join(os.path.dirname(uda_path), os.path.splitext(os.path.basename(uda_path))[0] + '_pt.pth')
    pretrained_dict = torch.load(dict_path, map_location='cpu')['student']
    model_dict = student.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    student.load_state_dict(pretrained_dict, strict=False)
    student.eval()

    seg_model = SegNet(num_classes=21, pretrained=True).to(device)

    return student, seg_model

def preprocess_image(args, transform):
    image = Image.open(args.image_path)
    image = image.resize((args.image_size, args.image_size))
    image = transform(image).unsqueeze(0)
    return image

# Function to visualize and save images
def save_images(image, seg_mask, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # Save original image
    denormalize = Denormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    image_np = denormalize(image.squeeze()).permute(1, 2, 0).cpu().numpy()
    plt.imsave(os.path.join(output_dir, f'original_image_{args.image_id}.png'), image_np)

    # Save segmentation mask
    plt.imsave(os.path.join(output_dir, f'segmentation_mask_{args.image_id}.png'), seg_mask.squeeze().cpu().numpy(), cmap='gray')
    plt.close()

def visualize(dataset, image, seg, keypoint2d, path_kp, path_seg): #image, seg, keypoints, filename_kp, filename_seg
    """
    Args:
        image (tensor): image in shape 3 x H x W
        keypoint2d (tensor): keypoints in shape K x 2
        name: name of the saving image
    """
    tensor_to_image = Compose([
        Denormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ToPILImage()
    ])
    if seg is not None and path_seg is not None:
        dataset.visualize(tensor_to_image(image), seg,
                                    keypoint2d, path_kp, path_seg)
    else:
        dataset.visualize(tensor_to_image(image), None,
                                    keypoint2d, path_kp, None)

def main(args):

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([transforms.ToTensor(), normalize])
    
    # Load models
    student, seg_model = load_models(args.uda_path, args.arch, args.num_keypoints, args.device)
    dataset = datasets.__dict__[args.dataset]
    dataset = dataset(root=args.root, transforms=transform, image_size=args.image_size, heatmap_size=args.heatmap_size)

    # Preprocess image
    image= preprocess_image(args, transform).to(args.device)
    # Extract heatmaps from the student model
    with torch.no_grad():
        heatmaps = student(image)

    pred, _ = get_max_preds(heatmaps.cpu().numpy())
    # Extract segmentation mask
    seg_mask = get_segmentation_masks(seg_model, image)
    seg_mask = torch.argmax(seg_mask, dim=1).float()  # B, H, W

    # Save images
    save_images(image, seg_mask, args.output_dir)
    visualize(dataset, image.squeeze(0), None, pred[0] * args.image_size / args.heatmap_size, os.path.join(args.output_dir, f"predicted_kpts_{args.image_id}.png"), None)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PRDA Inference')
    parser.add_argument('--image-path', type=str, required=True, help='Path to the input image')
    parser.add_argument('-d', '--dataset', default='SURREAL', type=str, help='Dataset name')
    parser.add_argument('--root', default='/data/AmitRoyChowdhury/dripta/surreal_processed', help='root path of the dataset')
    parser.add_argument('--output-dir', type=str, required=True, help='Directory to save the outputs')
    parser.add_argument('--uda-path', type=str, required=True, help='Path to the UDA model checkpoint')
    parser.add_argument('--arch', type=str, default='pose_resnet101', help='Model architecture')
    parser.add_argument('--num-keypoints', type=int, default=16, help='Number of keypoints')
    parser.add_argument('--lr', '--learning-rate', default=0.0001, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--image-size', type=int, default=256, help='Input image size')
    parser.add_argument('--heatmap-size', type=int, default=64, help='Heatmap size')
    args = parser.parse_args()

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #args.kp = re.sub(r'rgb/syn_', 'joints_2Ddep/syn_joints_2Ddep_', re.sub(r'\.png$', '.txt', args.image_path))
    args.image_id = re.search(r'syn_(\d+).png', args.image_path).group(1)
    print('Running Inference....')
    main(args)
    print('Inference completed!')
