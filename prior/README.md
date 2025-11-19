
### Training
- Create poses and corresponding distance value
```python prior/create_data.py --dset SyRIP --dset-root path/to/SyRIP --save-dir path/to/output/dir```

- Format the created data
```python prior/prepare_data.py --dset SyRIP --clean-pose-file path/to/SyRIP/K_5/raw/clean_poses_SyRIP.npy  --noisy-pose-file path/to/SyRIP/K_5/raw/noisy_poses_SyRIP.npy --save-dir path/to/output/dir```

- Train prior
```python prior/train_prior.py --data-root path/to/prior/processed --out-dir path/to/output/dir```

