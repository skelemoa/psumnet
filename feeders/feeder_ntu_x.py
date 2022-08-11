import numpy as np

from torch.utils.data import Dataset

from feeders import tools
import os
import h5py
import pickle
from .bone_pairs import ntux_body_pairs, ntux_hand_pairs, ntux_leg_pairs

train_subj_ids = [1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34, 35, 38, 45, 46, 47, 49, 50, 52, 53, 54, 55, 56, 57, 58, 59, 70, 74, 78, 80, 81, 82, 83, 84, 85, 86, 89, 91, 92, 
93, 94, 95, 97, 98, 100, 103]

training_cameras = [2, 3]


class Feeder(Dataset):
	def __init__(self, data_path, label_path=None, p_interval=1, split='train', random_choose=False, random_shift=False,
				 random_move=False, random_rot=False, window_size=-1, normalization=False, debug=False, use_mmap=False,
				 bone=False, vel=False, stream="body"):

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
		self.bone = bone
		self.vel = vel
		self.stream = stream
		
		self.label_path = self.data_path + '/{}_label_60.pkl'.format(self.split)
		if os.path.exists(self.label_path):
			with open(self.label_path, 'rb') as f:
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


		subj_id = int(name[name.index("P") + 1 : name.index("P") + 4])

		# c_id = int(name[name.index("C") + 1 : name.index("C") + 4])
		a_id = int(name[name.index("A") + 1 : name.index("A") + 4]) 


		if subj_id in train_subj_ids:
		   d = h5py.File(self.data_path + "/train/" + name + ".h5",'r')
		else:
		   d = h5py.File(self.data_path + "/test/" + name + ".h5",'r')
		
		data = d[list(d.keys())[-1]][:]
		data_numpy = data[:, :, :67, :]

		valid_frame_num = np.sum(data_numpy.sum(0).sum(-1).sum(-1) != 0)
		data_numpy = tools.valid_crop_resize(data_numpy, valid_frame_num, self.p_interval, self.window_size)
		
		if self.random_rot:
			data_numpy = tools.random_rot(data_numpy)

		C, T, V, M = data_numpy.shape
		
		r_hand_base = 46
		l_hand_base = 25

		if self.stream == "body":

			body_data = np.zeros((C, T, 37, M))
			body_data[:,:,:25,:] = data_numpy[:,:,:25,:]
			
			body_data[:,:,25,:] = data_numpy[:,:,r_hand_base,:]
			body_data[:,:,26,:] = data_numpy[:,:,r_hand_base + 4,:]
			body_data[:,:,27,:] = data_numpy[:,:,r_hand_base + 8,:]
			body_data[:,:,28,:] = data_numpy[:,:,r_hand_base + 12,:]
			body_data[:,:,29,:] = data_numpy[:,:,r_hand_base + 16,:]
			body_data[:,:,30,:] = data_numpy[:,:,r_hand_base + 20,:]

			body_data[:,:,31,:] = data_numpy[:,:,l_hand_base,:]
			body_data[:,:,32,:] = data_numpy[:,:,l_hand_base + 4,:]
			body_data[:,:,33,:] = data_numpy[:,:,l_hand_base + 8,:]
			body_data[:,:,34,:] = data_numpy[:,:,l_hand_base + 12,:]
			body_data[:,:,35,:] = data_numpy[:,:,l_hand_base + 16,:]
			body_data[:,:,36,:] = data_numpy[:,:,l_hand_base + 20,:]

			data_numpy = body_data
			pairs = ntux_body_pairs

		elif self.stream == "hand":
			hand_data = np.zeros((C,T,48,M))
			hand_data[:,:,:3,:] = data_numpy[:,:,[2,3,4],:] #right arm  
			hand_data[:,:,3:6,:] = data_numpy[:,:,[5,6,7],:] #left arm
			hand_data[:,:,6:27,:] = data_numpy[:,:,46:,:]  # right fingers
			hand_data[:,:,27:,:] = data_numpy[:,:,25:46,:]  # left fingers
			data_numpy = hand_data
			pairs = ntux_hand_pairs

		elif self.stream == "leg":
			leg_data = np.zeros((C,T,13,M))

			leg_data[:,:,0,:] = data_numpy[:,:,8,:]
			leg_data[:,:,1:4,:] = data_numpy[:,:,9:12,:] 
			leg_data[:,:,4,:] = data_numpy[:,:,24,:] 
			leg_data[:,:,5,:] = data_numpy[:,:,22,:] 
			leg_data[:,:,6,:] = data_numpy[:,:,23,:]

			leg_data[:,:,7:10,:] = data_numpy[:,:,12:15,:]
			leg_data[:,:,10,:] = data_numpy[:,:,21,:] 
			leg_data[:,:,11,:] = data_numpy[:,:,19,:] 
			leg_data[:,:,12,:] = data_numpy[:,:,20,:] 

			data_numpy = leg_data
			pairs = ntux_leg_pairs

		
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
