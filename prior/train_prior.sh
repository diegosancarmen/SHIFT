#MiniRGBD
#Create poses and corresponding distance value
#python create_data.py --dset MiniRGBD --dset-root '/data/AmitRoyChowdhury/MINI-RGBD_web' --image-size 256 --heatmap-size 64 --k-dist 5  --k-faiss 500 --save-dir '/data/AmitRoyChowdhury/sarosij/prior_data'
#Format the created data
#python prepare_data.py --dset MiniRGBD --clean-pose-file '/data/AmitRoyChowdhury/sarosij/prior_data/MiniRGBD/K_5/raw/clean_poses_MiniRGBD.npy'  --noisy-pose-file '/data/AmitRoyChowdhury/sarosij/prior_data/MiniRGBD/K_5/raw/noisy_poses_MiniRGBD.npy' --save-dir '/data/AmitRoyChowdhury/sarosij/prior_data'
#Train prior
#python train_prior.py --data-root '/data/AmitRoyChowdhury/sarosij/prior_data/MiniRGBD/processed' --out-dir '/data/AmitRoyChowdhury/sarosij/prior_data/MiniRGBD/ckpts'

#SyRIP
#Create poses and corresponding distance value
#python create_data.py --dset SyRIP --dset-root '/data/AmitRoyChowdhury/SyRIP/data/syrip/images' --image-size 256 --heatmap-size 64 --k-dist 5  --k-faiss 500 --save-dir '/data/AmitRoyChowdhury/sarosij/prior_data'
#Format the created data
#python prepare_data.py --dset SyRIP --clean-pose-file '/data/AmitRoyChowdhury/sarosij/prior_data/SyRIP/K_5/raw/clean_poses_SyRIP.npy'  --noisy-pose-file '/data/AmitRoyChowdhury/sarosij/prior_data/SyRIP/K_5/raw/noisy_poses_SyRIP.npy' --save-dir '/data/AmitRoyChowdhury/sarosij/prior_data'
#Train prior
python train_prior.py --data-root '/data/AmitRoyChowdhury/sarosij/prior_data/SyRIP/processed' --out-dir '/data/AmitRoyChowdhury/sarosij/prior_data/SyRIP/ckpts/real'