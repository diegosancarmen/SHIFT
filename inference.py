
# python -W ignore scripts/demo_inference.py --cfg configs/halpe_26/resnet/256x192_res50_lr1e-3_1x.yaml --checkpoint pretrained_models/halpe26_fast_res50_256x192.pth --indir /data/AmitRoyChowdhury/OCMotion/video_samples/0019_Camera03_243_frames_2 --outdir /data/AmitRoyChowdhury/OCMotion/video_samples/0019_Camera03_243_frames_2/alphapose
import os
from glob import glob
from tqdm import tqdm
import argparse

def main(args):


    # indir_root = '/data/AmitRoyChowdhury/Humans3.6M/data/val_dataset_sample5'
    # indir_root = '/data/AmitRoyChowdhury/Humans3.6M/data/val_dataset_sample5_com_block_occl/'
    indir_root = '/data/AmitRoyChowdhury/Humans3.6M/data/val_dataset_sample5'
    outdir_root = '/data/AmitRoyChowdhury/rohit/AlphaPose/val_dataset_sample5_MB_CLEAN' 
    split = 'S9'

    ## >/dev/null 2>&1
    os.makedirs(outdir_root, exist_ok=True)
    list_all_video = glob(f'{indir_root}/Images/{split}/*/',recursive=True)

    for indir in tqdm(sorted(list_all_video)):
        outdir = os.path.join( outdir_root,'/'.join(indir.split('/')[-4:]))

        # Run AlphaPose
        op_a = os.system(f"cd /data/AmitRoyChowdhury/rohit/AlphaPose && \
                python -W ignore scripts/demo_inference.py \
                --cfg configs/halpe_26/resnet/256x192_res50_lr1e-3_1x.yaml \
                --checkpoint pretrained_models/halpe26_fast_res50_256x192.pth \
                --indir {indir} --outdir {outdir} >/dev/null 2>&1")
        if op_a != 0:
            print('Alpha Pose Error for:',indir, '\t SKIPPED' )
            continue

        # Run MotionBERT
        op_b = os.system(f'cd /data/AmitRoyChowdhury/rohit/MotionBERT && \
                    python -W ignore infer_wild_mesh.py \
                    --json_path {outdir}/alphapose-results.json \
                    --out_path {outdir} \
                    --img_w 1000 --img_h 1002 --focus 1 \
                    --pixel >/dev/null 2>&1')
        # print(outdir)
        # break
        if op_b != 0:
            print('MotionBERT Error for:',indir, '\t SKIPPED motionBERT (Alphapose sucess)' )
            continue

print('Execution Completed')

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Custom Inference for 2D Infant Pose Estimation.')

    parser.add_argument('--dset', type=str, default='MiniRGBD',
                        help='keypoint to segmentation mask dataset')
    args = parser.parse_args()
    main(args)