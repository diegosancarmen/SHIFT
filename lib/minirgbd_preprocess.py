import os 
import numpy as np
import json
import argparse

def main(args):

    l = os.listdir(args.root)
    d = {'train':{},'validate':{}, 'prior':{}, 'all':{}}
    file_name = []
    file_name_3d = [ ]
    for i in l:
        if i not  in ['01','02','03','04','05','06','07','08','09','10','11','12']:    continue
        if i in ['01','02','03','04','05','06','07','08','09','10']:
            temp_dict = d['train']
            if i in ['01', '04', '05', '08']:
                prior_dict = d['prior'] # 25% of train data.
        elif i in ['11','12']:
            temp_dict = d['validate']
        dict = d['all']
        path_3d  = os.path.join(args.root,i,'joints_3D')
        path_2d  = os.path.join(args.root,i,'joints_2Ddep')

        for j in os.listdir(path_2d):
            with open(os.path.join(path_2d, j),'r') as f:
                file_name.append(j)
                lines = f.readlines()
                pose_2d = []
                for line in lines:  
                    pose_2d.append(np.array([line.split(' ')[0:2]]))
                pose_2d = np.array(pose_2d).reshape(-1,2).astype('float32')
                
                if str(i+'_'+j) not in temp_dict.keys():  
                    temp_dict[i+'_'+j] = {}
                    dict[i+'_'+j] = {}
                temp_dict[i+'_'+j]['pose_2d'] = pose_2d
                dict[i+'_'+j]['pose_2d'] = pose_2d

                if int(i)%2 == 1 and int(i) < 11:
                    if str(i+'_'+j) not in prior_dict.keys():
                        prior_dict[i+'_'+j] = {}
                    prior_dict[i+'_'+j]['pose_2d'] = pose_2d
        
        for j in os.listdir(path_3d):
            with open(os.path.join(path_3d, j),'r') as f:
                file_name_3d.append(j)
                lines = f.readlines()
                pose_3d = []
                for line in lines:
                    pose_3d.append(np.array([line.split(' ')[0:3]]))
                pose_3d = np.array(pose_3d).reshape(-1,3).astype('float32')
                temp_string = (i+'_'+j).replace('joints_3D','joints_2Ddep')
                if temp_string not in temp_dict.keys():
                    temp_dict[temp_string] = {}
                    dict[temp_string] = {}
                temp_dict[temp_string]['pose_3d'] = pose_3d
                dict[temp_string]['pose_3d'] = pose_3d

                if int(i)%2 == 1 and int(i) < 11:
                    if temp_string not in prior_dict.keys():
                        prior_dict[temp_string] = {}
                    prior_dict[temp_string]['pose_3d'] = pose_3d

    np.save(os.path.join(args.save_dir, f'{args.dset}.npy'),d)
    print('Data Preprocessed!')

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Preprocess Target dataset')
    parser.add_argument('--dset', type=str, default='MiniRGBD',
                        help='target pose dataset')
    parser.add_argument('--root', type=str, default='/data/AmitRoyChowdhury/MINI-RGBD_web', help='root path of the target dataset')
    parser.add_argument('--save-dir', type=str, default='/data/AmitRoyChowdhury/sarosij/MiniRGBD', help='save path of the source dataset')
    
    args = parser.parse_args()
    main(args)