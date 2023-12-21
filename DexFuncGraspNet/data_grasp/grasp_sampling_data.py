import os
import torch
from data_grasp.base_dataset import BaseDataset, NoPositiveGraspsException
import numpy as np
from datasets.utils import *
import copy

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


class GraspSamplingData(BaseDataset):
    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        self.opt = opt
        self.device = torch.device('cuda:{}'.format(
            opt.gpu_ids[0])) if opt.gpu_ids else torch.device('cpu')

        self.files, self.names = self.make_dataset(opt)
        self.obj = []
        self.rtj = []
        self.instance_name = []
        self.grasp_name = []
        for j, file in enumerate(self.files):
            for i in range(file[1]['rtj'].shape[0]):
                self.rtj.append(file[1]['rtj'][i])
                self.obj.append(file[0])
                self.instance_name.append(str(self.names[j])[:-4])
                self.grasp_name.append(str(file[1]['name'][i]))

        self.size = len(self.rtj)

        self.dict = {'mug': '1.57 0 -1.57', 'bowl': '1.57 0 -1.57', 'knife': '0 -1.57 0', 'mouse': '0 -1.57 3.14',
                     'bottle': '1.57 0 -1.57',
                     'bucket': '0 -1.57 3.14', 'camera': '1.57 0 0', 'pliers': '1.57 0 -1.57', 'remote': '3.14 0 0',
                     'teapot': '1.57 0 -1.57',
                     'stapler': '3.14 0 -1.57', 'eyeglass': '-1.57 -1.57 0', 'scissors': '-1.57 3.14 0',
                     'spraycan': '0 0 0', 'headphone': '0 0 0',
                     'lightbulb': '1.57 0 0', 'wineglass': '1.57 0 0', 'flashlight': '0 -1.57 1.57',
                     'screwdriver': '-1.57 0 -1.57', 'spraybottle': '1.57 0 0',
                     'drill_hairdryer': '-1.57 -1.57 0'}

        self.idx_category = {'mug': '0', 'bowl': '1', 'knife': '2', 'mouse': '3', 'bottle': '4', 'bucket': '5',
                            'camera': '6', 'pliers': '7', 'remote': '8','teapot': '9', 'stapler': '10', 'eyeglass': '11',
                            'scissors': '12', 'spraycan': '13', 'headphone': '14','lightbulb': '15', 'wineglass': '16', 'flashlight': '17',
                            'screwdriver': '18', 'spraybottle': '19', 'drill_hairdryer': '20'}
        
        self.vrc_result = '../VRCNET-DFG/data/results/mug/test'#for simulation test


    def __getitem__(self, index):
        
        #########################################################
        rtj = self.rtj[index]
        pcs = self.obj[index]['points']
        category = self.obj[index]['obj_class']
        obj_name = self.instance_name[index]##['obj_name']
        input_rwxyz = torch.tensor(rtj[:4]) #wxyz
        hand_t = torch.tensor(rtj[4:7]) * 0.001
        hand_a = torch.tensor(rtj[7:])
        choice = np.random.choice(len(pcs), 2500, replace=True)
        pcs = pcs[choice, :]
        meta = {}
        ###############################################


        meta['r'] = input_rwxyz
        input_rxyzw = torch.zeros(4)  # .cuda()
        input_rxyzw[:3] = input_rwxyz.reshape(4)[1:]
        input_rxyzw[3] = input_rwxyz.reshape(4)[0]
        device = 'cpu'
        meta['pc'] = pcs * 0.001
        
        meta['t'] = hand_t
        meta['angles'] = hand_a

        ################used for issacgym simulation################
        category_idx = category
        x_a = float(self.dict[category_idx].split(' ')[0]) * (np.pi / 2 / 1.57)
        y_a = float(self.dict[category_idx].split(' ')[1]) * (np.pi / 2 / 1.57)
        z_a = float(self.dict[category_idx].split(' ')[2]) * (np.pi / 2 / 1.57)
        a_list = [x_a, y_a, z_a]
        meta['pc'] = self.pc_rotate(meta['pc'], a_list)
        R_tran = self.init_R(a_list).to(device).float()
        new_t_ = torch.matmul(R_tran.T, meta['t'].float() * 1000.0).T.reshape(3)
        R_hand = torch.matmul(R_tran.T, r_to_transform(input_rxyzw))
        input_rxyzw = trans_2_quat_gpu(R_hand, device=device)
        input_rwxyz = torch.zeros(4).to(device)
        input_rwxyz[1:] = input_rxyzw.reshape(4)[:3]
        input_rwxyz[0] = input_rxyzw.reshape(4)[3]
        meta['r'] = input_rwxyz
        meta['t'] = new_t_

        hand_t = meta['t'].reshape(3).to(device).float() #* 1000.0
        hand_r = meta['r'].reshape(4).to(device).float()#wxyz
        hand_a = meta['angles'].reshape(22).to(device).float()
        new_j_p, transformed_pts = trans_xml_pkl(hand_a, hand_t, hand_r, device=device)
        meta['hand_pc'] = transformed_pts.clone().squeeze(0).detach().cpu().numpy() * 0.001
        meta['hand_keypoints'] = new_j_p.clone().squeeze(0).detach().cpu().numpy()
        ###############################################################


        meta['name'] = category +'/' + obj_name+'.obj'
        meta['ro'] = self.get_rotation(a_list, np.eye(3))
        meta['file'] = self.grasp_name[index]

        #相机坐标系下的抓取 随机一个camera的RT 为了真实场景下实验服务 真实场景得到部分点云，减去质心的距离
        # camera_pc_path_list = os.listdir(os.path.join(self.pc_p, meta['name']))            
        # random = np.random.randint(low=0, high=len(camera_pc_path_list)-1, size=1, dtype=int)
        # camera_corrds_info = np.load(os.path.join(self.pc_p, meta['name'], camera_pc_path_list[random[0]]))
        # R = camera_corrds_info['rotation']
        # T = camera_corrds_info['translation']
        # cam_r = torch.from_numpy(R).to(device)
        # cam_rr = torch.from_numpy(R.T).to(device)
        # new_t = torch.matmul(new_t_.unsqueeze(0), cam_rr.clone()) #+ torch.from_numpy(T).to(device).unsqueeze(0)*1000.0
        # r = quat_mul_tensor(trans_2_quat_gpu(cam_r, device=device), input_rxyzw.reshape(4)).reshape(4) #xyzw
        # input_rwxyz = torch.zeros(4).to(device)
        # input_rwxyz[1:] = r.clone().reshape(4)[:3]
        # input_rwxyz[0] = r.clone().reshape(4)[3]
        # meta['pc'] = meta['pc'] @ R.T #+ T
        # meta['r'] = input_rwxyz
        # meta['t'] = new_t.squeeze()
        # meta['angles'] = hand_a
        # new_j_p, transformed_pts = trans_xml_pkl(meta['angles'].clone(), meta['t'].clone(), meta['r'].clone(), device=device)
        # meta['hand_pc'] = transformed_pts.clone().squeeze(0).detach().cpu().numpy() * 0.001
        # meta['hand_keypoints'] = new_j_p.clone().squeeze(0).detach().cpu().numpy()
        #相机坐标系下的抓取 随机一个camera的RT 取消注释即可

        # meta['instance']= self.instances[index]
        
        if self.opt.is_train == False:
            #将vrcnet输出的点云进行读取,测试时,直接选第一个,单位m 物体已经正放置在桌面上了
            vrc_cp_path = os.listdir(os.path.join(self.vrc_result, category, obj_name+'.obj'))
            random = np.random.randint(low=0, high=len(vrc_cp_path)-1, size=1, dtype=int)
            meta['pc'] = np.load(os.path.join(self.vrc_result, category, obj_name+'.obj', vrc_cp_path[random[0]]))
        if torch.isnan(meta['r']).any() or torch.isinf(meta['r']).any():
            meta['r'] = torch.where(torch.isnan(meta['r']), torch.full_like(meta['r'], 0), meta['r'])
            meta['r'] = torch.where(torch.isinf(meta['r']), torch.full_like(meta['r'], 1), meta['r'])
            print('nan/inf in data')
        # import trimesh
        # obj = trimesh.PointCloud(meta['pc'], colors=[1, 95, 107, 255])
        # hand = trimesh.PointCloud(meta['hand_pc'], colors=[212, 106, 126, 255])
        # trimesh.Scene([obj, hand]).show()

        return meta
        
    def __len__(self):
        return self.size

    def get_rotation(self, a_list, R):
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
        trans_r = trans_2_quat_cpu(np.dot(rotation_matrix, R).T)
        trans_r_ = copy.deepcopy(trans_r)
        trans_r_[0] = trans_r[3]
        trans_r_[1:] = trans_r[:3]
        return trans_r_



    def init_R(self, a_list):
        a_list = torch.Tensor(a_list)
        rotation_anglex = a_list[0]
        cosvalx = torch.cos(rotation_anglex)
        sinvalx = torch.sin(rotation_anglex)
        rotation_angley = a_list[1]
        cosvaly = torch.cos(rotation_angley)
        sinvaly = torch.sin(rotation_angley)
        rotation_anglez = a_list[2]
        cosvalz = torch.cos(rotation_anglez)
        sinvalz = torch.sin(rotation_anglez)
        Rx = torch.Tensor([[1, 0, 0],
                    [0, cosvalx, -sinvalx],
                    [0, sinvalx, cosvalx]])
        Ry = torch.Tensor([[cosvaly, 0, sinvaly],
                    [0, 1, 0],
                    [-sinvaly, 0, cosvaly]])
        Rz = torch.Tensor([[cosvalz, -sinvalz, 0],
                    [sinvalz, cosvalz, 0],
                    [0, 0, 1]])
        rotation_matrix = torch.matmul(Rz, torch.matmul(Ry, Rx))
        return rotation_matrix

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
    

    

