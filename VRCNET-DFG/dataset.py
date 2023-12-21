import pickle

import torch
import numpy as np
import torch.utils.data as data
import h5py
import os
from tqdm import tqdm
import json, sys
# sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

class ShapeNetH5(data.Dataset):
    def __init__(self, train=True, npoints=2048, novel_input=True, novel_input_only=False):

        self.points = npoints

        self.complete_path = 'data/complete_pc'
        self.dict={'mug':'1.57 0 -1.57', 'bowl':'1.57 0 -1.57', 'knife':'0 -1.57 0', 'mouse':'0 -1.57 3.14', 'bottle':'1.57 0 -1.57',
                   'bucket':'0 -1.57 3.14', 'camera':'1.57 0 0', 'pliers':'1.57 0 -1.57', 'remote':'3.14 0 0', 'teapot':'1.57 0 -1.57',
                   'stapler':'3.14 0 -1.57', 'eyeglass':'-1.57 -1.57 0', 'scissors':'-1.57 3.14 0', 'spraycan':'0 0 0', 'headphone':'0 0 0',
                   'lightbulb':'1.57 0 0', 'wineglass':'1.57 0 0', 'flashlight':'0 -1.57 1.57', 'screwdriver':'-1.57 0 -1.57', 'spraybottle':'1.57 0 0',
                   'drill_hairdryer':'-1.57 -1.57 0'}
        if train:
            self.files = self.make_dataset_train()
        else:
            self.files = self.make_dataset_test()

        self.size = len(self.files)

    def __len__(self):
        return self.size

    def __getitem__(self, index):

        file = self.files[index]
        pc = np.load(file)
        pcs = pc['pc'][...]

        choice = np.random.choice(len(pcs), self.points, replace=True)
        partial = pcs[choice, :]
        r = pc['rotation']
        t = pc['translation']
        f = open('obj_data.json', 'r')
        x_a = float(self.dict[file.split('/')[-3]].split(' ')[0]) * (np.pi / 2 / 1.57)
        y_a = float(self.dict[file.split('/')[-3]].split(' ')[1]) * (np.pi / 2 / 1.57)
        z_a = float(self.dict[file.split('/')[-3]].split(' ')[2]) * (np.pi / 2 / 1.57)
        a_list = [x_a, y_a, z_a]
        path = self.complete_path + '/' + file.split('/')[-3] + '/' + file.split('/')[-2][:-4] + '.npy'
        point_cloud = np.load(path)
        choice22 = np.random.choice(len(point_cloud), self.points, replace=True)
        point_cloud = point_cloud[choice22, :]
        point_cloud = self.pc_rotate(point_cloud, a_list)
        # complete = point_cloud @ r.T + t

        #######for issacgym experiment######
        complete = point_cloud
        partial = (partial-t) @ r
        ####################################
        # import trimesh
        # obj = trimesh.PointCloud(partial, colors=[1, 95, 107, 255])
        # full = trimesh.PointCloud(complete, colors=[212, 106, 126, 255])
        # trimesh.Scene([obj, full]).show()


        label = 1

        return label, partial, complete, file

    def make_dataset_train(self):  

        files = []
        file_root = 'data/render_pc_for_completion/mug/train'
        for category in tqdm(os.listdir(file_root)):
            for instance in os.listdir(os.path.join(file_root,category)):
                for each_partical_pc in os.listdir(os.path.join(file_root,category,instance)):
                    files.append(os.path.join(file_root,category,instance,each_partical_pc))
        return files

    def make_dataset_test(self): 

        files = []
        file_root = 'data/render_pc_for_completion/mug/test'
        for category in tqdm(os.listdir(file_root)):
            for instance in os.listdir(os.path.join(file_root,category)):
                for each_partical_pc in os.listdir(os.path.join(file_root,category,instance)):
                    files.append(os.path.join(file_root,category,instance,each_partical_pc))
        return files

    def pc_rotate(self, pc, a_list):
        rotation_anglex = a_list[0]
        cosvalx = np.cos(rotation_anglex)
        sinvalx = np.sin(rotation_anglex)
        rotation_angley = a_list[1]
        cosvaly = np.cos(rotation_angley)
        sinvaly = np.sin(rotation_angley)
        rotation_anglez = a_list[2]
        cosvalz = np.cos(rotation_anglez)
        sinvalz = np.sin(rotation_anglez)
        Rx = np.array([[1, 0, 0],
                       [0, cosvalx, -sinvalx],
                       [0, sinvalx, cosvalx]])
        Ry = np.array([[cosvaly, 0, sinvaly],
                       [0, 1, 0],
                       [-sinvaly, 0, cosvaly]])
        Rz = np.array([[cosvalz, -sinvalz, 0],
                       [sinvalz, cosvalz, 0],
                       [0, 0, 1]])
        rotation_matrix = np.dot(Rz, np.dot(Ry, Rx))
        return np.dot(pc, rotation_matrix)

    
