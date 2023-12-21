import pickle

import torch
import numpy as np
import torch.utils.data as data
import h5py
import os
from tqdm import tqdm
from xml.dom.minidom import parse
import xml.dom.minidom
from datasets.utils import *
import shutil
class Refinedataset(data.Dataset):
    def __init__(self, train=True, npoints=2048, novel_input=True, novel_input_only=False):
        self.points = npoints
        if train:
            self.files = self.make_dataset_train()
        else:
            self.files = self.make_dataset_test()

        self.size = len(self.files)

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        file = self.files[index]
        with open(file, 'rb') as filename:
            pcs, graspparts, obj_name, j_p, _, hand_mesh_points = pickle.load(filename)
        new_file = file.replace('test_result_sim_refine', 'test_result_sim')
        xml_path = new_file[:-4]+'.xml'
        hand_a, hand_r, hand_t, file_name, trans_obj_i = self.reading_xml(xml_path)
        input_rwxyz = torch.zeros(4)  # .cuda()
        input_rwxyz[1:] = hand_r.reshape(4)[:3]
        input_rwxyz[0] = hand_r.reshape(4)[3]
        handrta = torch.cat((input_rwxyz, hand_t, hand_a.squeeze(0)), dim=0)

        return handrta, torch.from_numpy(pcs), file, trans_obj_i

    def make_dataset_train(self): # 

        files = []

        file_root = 'test_result_sim_refine'
        for category in tqdm(os.listdir(file_root)):
            if category=='success.txt':
                continue
            for instance in os.listdir(os.path.join(file_root, category)):
                if instance=='sift':
                    shutil.rmtree(os.path.join(file_root, category, instance))
                    continue
                pc_path = os.path.join(file_root, category, instance)
                for each_grasps in os.listdir(pc_path):
                    if each_grasps=='success.txt':
                        os.remove(os.path.join(file_root, category, instance, each_grasps))
                        continue
                    if each_grasps.endswith('.pkl'):
                        files.append(os.path.join(file_root, category, instance, each_grasps))
        return files

    def make_dataset_test(self): # 

        files = []

        file_root = 'test_result_sim_refine'
        for category in tqdm(os.listdir(file_root)):
            if category=='success.txt':
                continue
            for instance in os.listdir(os.path.join(file_root, category)):
                if instance=='sift':
                    shutil.rmtree(os.path.join(file_root, category, instance))
                    continue
                pc_path = os.path.join(file_root, category, instance)
                for each_grasps in os.listdir(pc_path):
                    if each_grasps=='success.txt':
                        os.remove(os.path.join(file_root, category, instance, each_grasps))
                        continue
                    if each_grasps.endswith('.pkl'):
                        files.append(os.path.join(file_root, category, instance, each_grasps))
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



    def reading_xml(self, filename):
        DOMTree = xml.dom.minidom.parse(filename)
        collection = DOMTree.documentElement
        graspableBody = collection.getElementsByTagName('graspableBody')[0]
        filenames = filename
        file_name_i = graspableBody.getElementsByTagName('filename')[0].childNodes[0].data
        file_name = file_name_i.split('/')[-1].replace('.obj.smoothed','')[:-4]
        # file_name = file_name_i.split('/')[-1].split('_')[0]#.replace('.obj.smoothed','')[:-6]

        robot = collection.getElementsByTagName("robot")[0]
        robot_name = robot.getElementsByTagName('filename')[0].childNodes[0].data
        a = robot.getElementsByTagName('dofValues')[0]
        a = a.childNodes[0].data
        trans = robot.getElementsByTagName('transform')[0]
        rt = trans.getElementsByTagName('fullTransform')[0]
        rt = rt.childNodes[0].data

        rt_obj_ = graspableBody.getElementsByTagName('transform')[0]
        rt_obj = rt_obj_.getElementsByTagName('fullTransform')[0]
        rt_obj = rt_obj.childNodes[0].data
        r_obj = rt_obj.split(')')[0].split('(')[1].split(' ')
        r_obj = np.array(r_obj, dtype='float32')
        r_obj = torch.tensor(r_obj, dtype=torch.float32)  # .unsqueeze(0)
        rs_obj = r_obj.clone()
        for i in range(4):
            rs_obj[0] = r_obj[1]
            rs_obj[1] = r_obj[2]
            rs_obj[2] = r_obj[3]
            rs_obj[3] = r_obj[0]

        r = rt.split(')')[0].split('(')[1].split(' ')
        t = rt.split('[')[1].split(']')[0].split(' ')

        a = a.replace('   ', ' ')
        a = np.array(a.split(' '), dtype='float32')
        a = torch.tensor(a, dtype=torch.float32).unsqueeze(0)
        if robot_name[-6:-4]=='lw':
            a[:, 1] = -a[:, 1]
        elif robot_name[-6:-4] != 'nd':
            raise ('robot_name has wrong, place call zhutq')
        r = np.array(r, dtype='float32')
        r = torch.tensor(r, dtype=torch.float32)  # .unsqueeze(0)
        rs = r.clone()
        for i in range(4):
            rs[0] = r[1]
            rs[1] = r[2]
            rs[2] = r[3]
            rs[3] = r[0]

        t = np.array(t, dtype='float32')
        t = torch.tensor(t, dtype=torch.float32) * 0.001  # .unsqueeze(0)

        return a, rs, t, file_name, rs_obj
