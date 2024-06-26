import numpy as np
import json
import os
import random

d = {'train':{},'validate':{}, 'prior':{}}

root = "/data/AmitRoyChowdhury/SyRIP/data/syrip/annotations/"

train_data = os.path.join(root, "200R_1000S/person_keypoints_train_infant.json")
validate_data = os.path.join(root, "validate500/person_keypoints_validate_infant.json")

with open(train_data, "r") as file:
    train_dict = json.load(file)

for annot in train_dict['annotations']:
    key = annot['image_id']
    d['train'][key] = np.array(annot['keypoints']).reshape(17, 3)[:, :2]

with open(validate_data, "r") as file:
    validate_dict = json.load(file)

for annot in validate_dict['annotations']:
    key = annot['image_id']
    d['validate'][key] = np.array(annot['keypoints']).reshape(17, 3)[:, :2]
    
# Step 1: Extract all keys from 'train'
all_synth_keys = sorted(list(d['train'].keys()))[200:]
all_real_keys = sorted(list(d['train'].keys()))[:200]

# Step 2: Randomly select 300 keys
random_keys = random.sample(all_synth_keys, 300)

# # Remove the transferred items from the source key
# for key in random_keys:
#     d['prior'][key] = d['train'][key]
#     del d['train'][key]
    
for key in all_real_keys:
    d['prior'][key] = d['train'][key]
    del d['train'][key]

print(len(list(d['prior'].keys())))
print(len(list(d['train'].keys())))

np.save('/data/AmitRoyChowdhury/sarosij/SyRIP/prior/SyRIP_200R.npy', d)
print('Data Preprocessed!')