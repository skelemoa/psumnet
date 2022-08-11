import numpy as np

from torch.utils.data import Dataset

from feeders import tools
import os
import h5py
import pickle
import torch
from .bone_pairs import shrec_pairs


class Feeder(Dataset):
    def __init__(self, data_path, label_path=None, p_interval=1, split='train', random_choose=False, random_shift=False,
                 random_move=False, random_rot=False, window_size=-1, normalization=False, debug=False, use_mmap=False, stream = "body"):
        
        self.debug = debug
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
        
        with open(data_path, 'rb') as f:
            self.data = pickle.load(f)

        # label_path = '/ssd_scratch/cvit/neel1998/ntu60_kinect_xsub/{}_label_60.pkl'.format(self.split)
        if os.path.exists(label_path):
            with open(label_path, 'rb') as f:
                self.pickle_file = pickle.load(f)
                self.sample_name = self.pickle_file[0]
                self.label = self.pickle_file[1]
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
        
        d = self.data[idx]
        
        C,T,V,M = d.shape

        max_temp_step = 180

        data_numpy = np.zeros([C,max_temp_step,V,M])
        data_numpy[:,:T,:,:] = d


        pairs = shrec_pairs
      

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
