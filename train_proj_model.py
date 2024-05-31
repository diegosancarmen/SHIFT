#python train_proj_model.py --log encoder_logs/surreal
import argparse 
import random 
import warnings
import os
import sys
import time
from tqdm import tqdm
import numpy as np 
import torch 
import torch.backends.cudnn as cudnn
from torch.utils.data import TensorDataset, DataLoader
from torchvision.transforms import Compose, ToPILImage

sys.path.append('..')
from lib.meter import AverageMeter, ProgressMeter
from lib.logger import CompleteLogger
from lib.transforms import Denormalize
import lib.models as models
from lib.data import ForeverDataIterator
import lib.transforms.keypoint_detection as T
import lib.datasets as datasets
from lib.models.seg_net import SegNet, get_segmentation_masks
from lib.keypoint_detection import accuracy
from lib.models.loss import *
from utils import *

import warnings

# Ignore UserWarning from torchvision
warnings.filterwarnings("ignore", category=UserWarning)

def train_model(train_source_iter, device, student, seg_model, model, optimizer, criterion, epoch, visualize, args):
    batch_time = AverageMeter('Time', ':4.2f')
    data_time = AverageMeter('Data', ':3.1f')
    losses_s = AverageMeter('Loss', ":.4e")

    iters_per_epoch = args.iters_per_epoch
    progress = ProgressMeter(
        iters_per_epoch,
        [batch_time, data_time, losses_s],
        prefix="Epoch: [{}]".format(epoch))   

    model.train()
    end = time.time()

    scaler = torch.cuda.amp.GradScaler()

    for i in range(iters_per_epoch):
        x_s, label_s, weight_s, meta_s = next(train_source_iter)
        x_s = x_s.to(device)
        label_s = label_s.to(device)
        weight_s = weight_s.to(device)
        # measure data loading time
        data_time.update(time.time() - end)

        optimizer.zero_grad()

        with torch.cuda.amp.autocast():
            y_s = student(x_s)  # B, 16, 64, 64
            seg_x_s = get_segmentation_masks(seg_model, x_s)  # B, 2, 256, 256
            seg_x_s = torch.argmax(seg_x_s, dim=1).float().unsqueeze(1)  # B, 256, 256
            #seg_x_s = (seg_x_s > args.seg_threshold).float().unsqueeze(1)  # GT segmentation mask

            y_s_kp = heatmap_to_keypoints(y_s)
            fake_segmentation = model(y_s_kp)
            loss_s = criterion(fake_segmentation, seg_x_s)  # Supervised loss

        scaler.scale(loss_s).backward()
        scaler.step(optimizer)
        scaler.update()

        _, _, _, pred_s = accuracy(y_s.detach().cpu().numpy(),
                                   label_s.detach().cpu().numpy())

        # Update loss meters
        losses_s.update(loss_s.item(), x_s.size(0))

        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)
            if visualize is not None:
                visualize(x_s[0], seg_x_s[0], pred_s[0] * args.image_size / args.heatmap_size, "{}_{}_pred.jpg".format(args.dset, i), "{}_{}_mask.jpg".format(args.dset, i))
                visualize(x_s[0], None, meta_s['keypoint2d'][0], "{}_{}_label.jpg".format(args.dset, i), None)

def main(args):

    logger = CompleteLogger(args.log + '_' + args.arch)
    logger.write(' '.join(f'{k}={v}' for k, v in vars(args).items()))
    
    image_size = (args.image_size, args.image_size)
    heatmap_size = (args.heatmap_size, args.heatmap_size)
    normalize = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    transform = T.Compose([
        T.ToTensor(),
        normalize
    ])
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')
    cudnn.benchmark = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # create model 
    seg_model = SegNet(num_classes=2, pretrained=True).to(device)
    kp2seg_model = KeypointToSegmentationEncoder(num_keypoints=16, output_size=256).to(device)
    optimizer = torch.optim.Adam(kp2seg_model.parameters(), lr=args.lr, betas=(0.5, 0.999))
    criterion = SegLoss()
    
    # create dataset 
    source_dataset = datasets.__dict__[args.dset]
    dataset = source_dataset(root=args.dset_root, split='train', transforms=transform,
                                          image_size=image_size, heatmap_size=heatmap_size)
    train_source_loader = DataLoader(dataset, batch_size=args.batch_size,
                                     shuffle=True, num_workers=args.workers, pin_memory=True, drop_last=True)
    train_source_iter = ForeverDataIterator(train_source_loader)

    student = models.__dict__[args.arch](num_keypoints=dataset.num_keypoints).cuda()
    N = len(dataset)

    save_dir = os.path.join(args.save_dir, args.dset)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # define visualization function
    tensor_to_image = Compose([
        Denormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ToPILImage()
    ])

    def visualize(image, seg, keypoint2d, name_kp, name_seg): #image, seg, keypoints, filename_kp, filename_seg
        """
        Args:
            image (tensor): image in shape 3 x H x W
            keypoint2d (tensor): keypoints in shape K x 2
            name: name of the saving image
        """
        if seg is not None and name_seg is not None:
            dataset.visualize(tensor_to_image(image), seg,
                                       keypoint2d, logger.get_image_path(name_kp), logger.get_image_path(name_seg))
        else:
            dataset.visualize(tensor_to_image(image), None,
                                       keypoint2d, logger.get_image_path(name_kp), None)

    # train model
    print('Training Encoder...')
    for epoch in range(args.epochs):
        train_model(train_source_iter, device, student, seg_model, kp2seg_model, optimizer, criterion, epoch, visualize, args)
        torch.save(
            {
                'model': kp2seg_model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch
            }, os.path.join(args.save_dir, f'{args.dset}_kp2seg_gan.pt')
        )

if __name__ == '__main__':
    architecture_names = sorted(
        name for name in models.__dict__
        if name.islower() and not name.startswith("__")
        and callable(models.__dict__[name])
    )

    parser = argparse.ArgumentParser(description='Train keypoint to segmentation map encoder')

    parser.add_argument('--dset', type=str, default='SURREAL',
                        help='keypoint to segmentation mask dataset')
    parser.add_argument("--dset-root", type=str, default='/data/AmitRoyChowdhury/dripta/surreal_processed',
                        help="root path of the source dataset")
    parser.add_argument('-a', '--arch', metavar='ARCH', default='pose_resnet101',
                        choices=architecture_names,
                        help='backbone architecture: ' +
                             ' | '.join(architecture_names) +
                             ' (default: pose_resnet101)')
    parser.add_argument('--image-size', type=int, default=256,
                        help='input image size')
    parser.add_argument('--heatmap-size', type=int, default=64,
                        help='output heatmap size')
    parser.add_argument("--batch-size",  type=int, default=64,
                        help='mini-batch size (default: 1024)')
    parser.add_argument("--log", type=str, default='src_only',
                        help="Where to save logs, checkpoints and debugging images.")
    parser.add_argument("--lr", type=float, default=3e-4, 
                        help='learning rate')
    parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
                        help='number of data loading workers (default: 2)')
    parser.add_argument("--seg-threshold", type=float, default=0.5, 
                        help='Binarization Threshold for segmentation mask')
    parser.add_argument("--iters-per-epoch", type=int, default=500, 
                        help='iterations per epoch')
    parser.add_argument("--epochs", type=int, default=150,
                        help='number of total epochs to run')
    parser.add_argument("--seed", type=int, default=0, 
                        help='seed for initializing training.')
    parser.add_argument("--print-freq", type=int, default=100, 
                        help='print frequency (default: 100)')
    parser.add_argument("--save-dir", type=str, default='/data/AmitRoyChowdhury/sarosij/kp2seg_data/',
                        help="where to save encoder model")
    args = parser.parse_args()
    main(args)