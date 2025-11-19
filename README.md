# Exploiting Synthetic Adult Datasets for Infant Pose Estimation

Code for SHIFT: Exploiting Synthetic Adult Datasets for Infant Pose Estimation
Sarosij Bose, Hannah Dela Cruz, Arindam Dutta, Elena Kokkoni, Konstantinos Karydis, Amit K. Roy Chowdhury

# Introduction

<p align="left">
  <img width="650" src="figures/method_comparison.png">
</p>

From left to right: keypoint predictions from a baseline adult human pose estimation model ([Xiao et al., 2018](https://arxiv.org/abs/1807.10221)), predictions from a SOTA UDA pose estimation model ([Kim et al., 2022](https://arxiv.org/abs/2204.00172)), and predictions from our method, SHIFT. Adult pose estimation models fail when directly applied to infant data; similarly, UDAPE struggles to overcome the domain shift between adults and infants. In contrast, SHIFT accounts for the highly self-occluded pose distribution of infants, thereby effectively adapting to the infant domain

# Method

<p align="left">
  <img width="650" src="figures/framework.png">
</p>

SHIFT leverages the mean-teacher framework ([Tarvainen et al., 2017](https://arxiv.org/abs/1703.01780)) to adapt a model pretrained on a labeled adult source dataset $(x_s, y_s)$ to unlabeled infant target images $(x_t)$. To address anatomical variations, SHIFT employs an infant pose prior $\theta_p$ to produce plausibility scores for each prediction of the student model $\mathcal{M}_s$ and, to handle self-occlusions we employ an off-the-model $F_{seg}$ and our learned Kp2Seg module $(G)$ to perform image-pose visibility alignment

# Usage

**Dataset Preparation**

**SURREAL Dataset**
As instructed by [UDA_PoseEstimation](https://github.com/VisionLearningGroup/UDA_PoseEstimation/tree/master), the following datasets can be downloaded automatically:
- [Surreal Dataset](https://www.di.ens.fr/willow/research/surreal/data/)
- [SyRIP Dataset](https://coe.northeastern.edu/Research/AClab/SyRIP/images/)
- [MINI-RGBD Dataset](https://www.iosb.fraunhofer.de/en/competences/image-exploitation/object-recognition/sensor-networks/motion-analysis.html#mini-rgbd)

Save it to `../MINI-RGBD_web`

**Prior Module Training**

See `prior` folder for instructions.

**Keypoint-to-Segmentation Module Training**

```python train_keypoint_to_segmentation.py --dset-root /data/AmitRoyChowdhury/dripta/surreal_processed --dset SURREAL --arch pose_resnet101 --image-size 256 --heatmap-size 64 --batch-size 32 --log path/to/log/directory --lr 0.0003 --workers 2 --seg-threshold 0.5 --iters-per-epoch 500 --epochs 30 --seed 0 --print-freq 100 --save-dir path/to/save/directory```

# Experiments

SURREAL-to-MINIRGBD

```python train_human_to_infant.py path/to/surreal path/to/mini-rgbd -s SURREAL -t MiniRGBD --target-train MiniRGBD_mt --log logs/surreal2minirgbd --prior path/to/prior_stage_3.pt --kp2seg /path/to/kp2seg_data/SURREAL_kp2seg_gan.pt --debug --lambda_c 1 --pretrain-epoch 40 --lambda_s 1e-6 --lambda_p 1e-6 --mode 'all' --rotation_stu 60 --shear_stu -30 30 --translate_stu 0.05 0.05 --scale_stu 0.6 1.3 --color_stu 0.25 --blur_stu 0 --rotation_tea 60 --shear_tea -30 30 --translate_tea 0.05 0.05 --scale_tea 0.6 1.3 --color_tea 0.25 --blur_tea 0 -b 32 --mask-ratio 0.5 --k 1 --s2t-freq 0.5 --s2t-alpha 0 1 --t2s-freq 0.5 --t2s-alpha 0 1 --occlude-rate 0.5 --occlude-thresh 0.9```

SURREAL-to-SyRIP

```# python train_human_to_infant.py path/to/surreal path/to/SyRIP/data/syrip/images -s SURREAL -t SyRIP --target-train SyRIP_mt --log logs/surreal2syrip --prior path/to/prior_stage_3.pt --kp2seg /path/to/kp2seg_data/SURREAL_kp2seg_gan.pt --debug --lambda_c 1 --pretrain-epoch 40 --lambda_s 1e-6 --lambda_p 1e-6 --mode 'all' --rotation_stu 60 --shear_stu -30 30 --translate_stu 0.05 0.05 --scale_stu 0.6 1.3 --color_stu 0.25 --blur_stu 0 --rotation_tea 60 --shear_tea -30 30 --translate_tea 0.05 0.05 --scale_tea 0.6 1.3 --color_tea 0.25 --blur_tea 0 -b 32 --mask-ratio 0.5 --k 1 --s2t-freq 0.5 --s2t-alpha 0 1 --t2s-freq 0.5 --t2s-alpha 0 1 --occlude-rate 0.5 --occlude-thresh 0.9```
<!-- # Acknowledgements -->

