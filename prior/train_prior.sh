#Create poses and corresponding distance value
#python create_data.py --dset MiniRGBD --dset-root '/data/AmitRoyChowdhury/MINI-RGBD_web' --image-size 256 --heatmap-size 64 --k-dist 5  --k-faiss 500 --save-dir '/data/AmitRoyChowdhury/sarosij/prior_data'
#Format the created data
#python prepare_data.py --clean-pose-file '/data/AmitRoyChowdhury/sarosij/prior_data/clean_poses_MiniRGBD.npy'  --noisy-pose-file '/data/AmitRoyChowdhury/sarosij/prior_data/noisy_poses_MiniRGBD.npy' --save-dir '/data/AmitRoyChowdhury/sarosij/prior_data'
#Train prior
python train_prior.py --data-root '/data/AmitRoyChowdhury/sarosij/prior_data/processed' --out-dir '/data/AmitRoyChowdhury/sarosij/prior_data/ckpts'