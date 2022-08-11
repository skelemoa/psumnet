import numpy as np

from torch.utils.data import Dataset

from feeders import tools
import os
import h5py
import pickle
import torch
from .bone_pairs import ntu_pairs, kinect_leg_pairs, kinect_hand_pairs


class Feeder(Dataset):
    def __init__(self, data_path, label_path=None, p_interval=1, split='train', random_choose=False, random_shift=False,
                 random_move=False, random_rot=False, window_size=-1, normalization=False, debug=False, use_mmap=False, stream = "body"):
        
        self.debug = True
        self.data_path = data_path
        self.label_path = label_path
        self.split = split
        self.random_choose = random_choose
        self.random_shift = random_shift
        self.random_move = random_move
        self.window_size = window_size
        self.normalization = normalization
        self.use_mmap = use_mmap
        self.p_interval = p_interval
        self.random_rot = random_rot
        self.stream = stream
        
        self.data = np.load(data_path)

        # label_path = '/ssd_scratch/cvit/neel1998/ntu60_kinect_xsub/{}_label_60.pkl'.format(self.split)
        if os.path.exists(label_path):
            with open(label_path, 'rb') as f:
                self.pickle_file = pickle.load(f)
                self.sample_name = self.pickle_file[0][0]
                self.label = self.pickle_file[0][1]
        else:
            logging.info('')
            logging.error('Error: Do NOT exist data files: {}!'.format(label_path))
            logging.info('Please generate data first!')
            raise ValueError()
        if self.debug:
            self.sample_name = self.sample_name[:300]
            self.label = self.label[:300]

        print(split + " data samples:" + str(len(self.sample_name)))

    def __len__(self):
        return len(self.sample_name)

    def __iter__(self):
        return self

    def __getitem__(self, idx):

        label = self.label[idx]
        name = self.sample_name[idx].split(".")[0]
        
        data_numpy = self.data[idx]
        data_numpy = np.array(data_numpy)
        
        valid_frame_num = np.sum(data_numpy.sum(0).sum(-1).sum(-1) != 0)
        data_numpy = tools.valid_crop_resize(data_numpy, valid_frame_num, self.p_interval, self.window_size)
        if self.random_rot:
            data_numpy = tools.random_rot(data_numpy)
        

        C, T, V, M = data_numpy.shape
        pairs = ntu_pairs
        
        if self.stream == "hand":
        
            hand_data = np.zeros((C,T,13,M))
            hand_data[:,:,0,:] = data_numpy[:,:,20,:]
            hand_data[:,:,1,:] = data_numpy[:,:,4,:]
            hand_data[:,:,2,:] = data_numpy[:,:,5,:]
            hand_data[:,:,3,:] = data_numpy[:,:,6,:]
            hand_data[:,:,4,:] = data_numpy[:,:,7,:]
            hand_data[:,:,5,:] = data_numpy[:,:,21,:]
            hand_data[:,:,6,:] = data_numpy[:,:,22,:]
            hand_data[:,:,7,:] = data_numpy[:,:,8,:]
            hand_data[:,:,8,:] = data_numpy[:,:,9,:]
            hand_data[:,:,9,:] = data_numpy[:,:,10,:]
            hand_data[:,:,10,:] = data_numpy[:,:,11,:]
            hand_data[:,:,11,:] = data_numpy[:,:,23,:]
            hand_data[:,:,12,:] = data_numpy[:,:,24,:]

            data_numpy = hand_data
            pairs = kinect_hand_pairs

        elif self.stream == "leg":
        
            leg_data = np.zeros((C,T,9,M))
            leg_data[:,:,0,:] = data_numpy[:,:,0,:]
            leg_data[:,:,1,:] = data_numpy[:,:,12,:]
            leg_data[:,:,2,:] = data_numpy[:,:,13,:]
            leg_data[:,:,3,:] = data_numpy[:,:,14,:]
            leg_data[:,:,4,:] = data_numpy[:,:,15,:]
            leg_data[:,:,5,:] = data_numpy[:,:,16,:]
            leg_data[:,:,6,:] = data_numpy[:,:,17,:]
            leg_data[:,:,7,:] = data_numpy[:,:,18,:]
            leg_data[:,:,8,:] = data_numpy[:,:,19,:]

            data_numpy = leg_data
            pairs = kinect_leg_pairs


        bone_data_numpy = np.zeros_like(data_numpy)
        for v1, v2 in pairs:
            bone_data_numpy[:, :, v1 - 1] = data_numpy[:, :, v1 - 1] - data_numpy[:, :, v2 - 1]
        
        joint_vel_data = np.zeros_like(data_numpy)
        bone_vel_data = np.zeros_like(bone_data_numpy)

        joint_vel_data[:, :-1] = data_numpy[:, 1:] - data_numpy[:, :-1]
        joint_vel_data[:, -1] = 0

        bone_vel_data[:, :-1] = bone_data_numpy[:, 1:] - bone_data_numpy[:, :-1]
        bone_vel_data[:, -1] = 0

        final_data = np.concatenate([data_numpy, bone_data_numpy, joint_vel_data, bone_vel_data],0)

        return final_data, label, idx

    def top_k(self, score, top_k):
        rank = score.argsort()
        hit_top_k = [l in rank[i, -top_k:] for i, l in enumerate(self.label)]
        return sum(hit_top_k) * 1.0 / len(hit_top_k)

    

def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod
