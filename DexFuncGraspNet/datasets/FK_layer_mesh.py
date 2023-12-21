import torch
import torch.nn as nn
import numpy as np
import time, sys
import copy
import open3d as o3d
sys.path.append('../utils/')
# from lib.fk.utils import get_4x4_matrix
def get_4x4_matrix(quat, pos):
    if isinstance(quat, list):
        quat = np.array(quat, np.float64) *1000.0
        quat = quat[[3, 0, 1, 2]]
    if isinstance(pos, list):
        pos = np.array(pos, np.float64)
    t = np.eye(4).astype(np.float64)
    t[:3, 3] = pos *1000.0
    t[:3, :3] = o3d.geometry.get_rotation_matrix_from_quaternion(quat)
    return t
class Rodrigues:
    def __init__(self, rot_axis_list,device):
        SHADOW_HAND_DOF_NUM = 22
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        rot_axis = torch.FloatTensor(rot_axis_list).to(self.device).reshape(22, 3, 1)   # [22, 3, 1]
        rot_axis_T = torch.transpose(rot_axis, 1, 2) # [22, 1, 3]
        self.nnT = torch.bmm(rot_axis, rot_axis_T)  # [22, 3, 3]
        self.skew = torch.zeros_like(self.nnT).to(self.device) # [22, 3, 3]
        self.skew[:, 0, 1] = -rot_axis[:, 2, 0]
        self.skew[:, 0, 2] = rot_axis[:, 1, 0]
        self.skew[:, 1, 0] = rot_axis[:, 2, 0]
        self.skew[:, 1, 2] = -rot_axis[:, 0, 0]
        self.skew[:, 2, 0] = -rot_axis[:, 1, 0]
        self.skew[:, 2, 1] = rot_axis[:, 0, 0]

    '''
    description: 
    param {*} self
    param {*} theta : [batch, 22] 关节角度(rad)
    return {*} mat: [batch, 22, 3, 3]
    '''
    def rot_axis_to_mat(self, theta):
        batch_size = theta.size()[0]
        cos_theta = torch.cos(theta).reshape(batch_size, 22, 1, 1).repeat(1, 1, 3, 3)  # [batch, 22, 3, 3]
        sin_theta = torch.sin(theta).reshape(batch_size, 22, 1, 1).repeat(1, 1, 3, 3)  # [batch, 22, 3, 3]
        I = torch.eye(3).to(self.device).reshape(1, 1, 3, 3).repeat(batch_size, 22, 1, 1) # [batch, 22, 3, 3]
        nnT = self.nnT.unsqueeze(0).repeat(batch_size, 1, 1, 1) # [batch, 22, 3, 3]
        skew = self.skew.unsqueeze(0).repeat(batch_size, 1, 1, 1) # [batch, 22, 3, 3]
        mat = cos_theta * I + (1 - cos_theta) * nnT + sin_theta * skew
        return mat

class FK_layer(nn.Module):
    def __init__(self, base, rotations, device=None, F=1, J=22):
        super(FK_layer, self).__init__()
        """
        base: [F,7] or [F,1,4,4], torch.FloatTensor
        rotations: [F, 22], torch.FloatTensor
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device is not None:
            self.device = device
        if base.shape[-1] == 7 and len(base.shape) == 2:
            self.base_M = self.transforms_base(base)   # [F,1,4,4]
        elif base.shape[-1] == 4 and len(base.shape) == 4:
            self.base_M = base
        
        self.non_zero_joints_index = list(torch.where(rotations[0]>5e-6)[0].cpu().numpy())
        self.rotations = rotations
        self.delta_max, self.delta_min = self.get_scope(rotations)
        self.F, self.J = F, J

        self.body_name_list = ['palm',  # 1
                                   'knuckle', 'F3', 'F2', 'F1',  # 2, 3, 4, 5,
                                   'knuckle', 'F3', 'F2', 'F1',  # 6, 7, 8, 9,
                                   'knuckle', 'F3', 'F2', 'F1',  # 10, 11, 12, 13,
                                   'lfmetacarpal', 'knuckle', 'F3', 'F2', 'F1', # 14, 15, 16, 17, 18,
                                   'TH3_z', '', 'TH2_z', '', 'TH1_z',] # 19, 20, 21, 22, 23

        self.BODY_NUM = len(self.body_name_list) # 1 palm + 22 joints = 23

        TH3_z_q = [ 0, 0.3824995, 0, 0.9239557 ]
        body_t_list = [[0, 0, 0], 
                       [0.033, 0, 0.095], [0, 0, 0], [0, 0, 0.045], [0, 0, 0.025],
                       [0.011, 0, 0.099], [0, 0, 0], [0, 0, 0.045], [0, 0, 0.025],
                       [-0.011, 0, 0.095],[0, 0, 0], [0, 0, 0.045], [0, 0, 0.025],
                       [-0.034, 0, 0.021], [-0.001, 0, 0.067], [0, 0, 0], [0, 0, 0.045], [0, 0, 0.025],
                       [0.034, -0.009, 0.029], [0, 0, 0], [0, 0, 0.038], [0, 0, 0], [0, 0, 0.032]]

        M_sh = []  # Local transform matrix for shadow hand
        for i in range(1, self.BODY_NUM):
            if self.body_name_list[i] == 'TH3_z':
                local = get_4x4_matrix(TH3_z_q, body_t_list[i]).astype(np.float64)
            else:
                local = get_4x4_matrix([0,0,0,1], body_t_list[i]).astype(np.float64)
            M_sh.append(local)
        self.M_sh = torch.FloatTensor(M_sh).to(self.device) # [22, 4, 4]

        rot_axis_list = [[0, 1, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0],
                        [0, 1, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0],
                        [0, 1, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0],
                        [0.571 ,0 ,0.821], [0, 1, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0],
                        [0, 0, -1], [1, 0, 0], [1, 0, 0], [0, 1, 0], [0, 1, 0]]
        self.rodrigues = Rodrigues(rot_axis_list, self.device)

        self.parents = [-1,
                    0, 1, 2, 3,
                    0, 5, 6, 7,
                    0, 9, 10, 11,
                    0, 13, 14, 15, 16,
                    0, 18, 19, 20, 21]

        self.pts_dict = np.load('/home/hjl/Functional_Graspnet/datasets/sampled_pts_of_shadow_hand.npy', allow_pickle=True).item()
        for name in self.pts_dict:
            self.pts_dict[name] = torch.from_numpy(self.pts_dict[name]).float().to(self.device) * 1000.0
        
        self.root_offset = nn.Embedding(self.F, 3)
        self.emb = nn.Embedding(self.F, self.J)
        nn.init.zeros_(self.emb.weight)
        nn.init.zeros_(self.root_offset.weight)

    def get_scope(self, rotations):
        theta_up_limit  = [  0.349, 1.571, 1.571, 1.571,
                             0.349, 1.571, 1.571, 1.571, 
                             0.349, 1.571, 1.571, 1.571,
                             0.785, 0.349, 1.571, 1.571, 1.571,
                             1.047, 1.222, 0.209, 0.524, 0.000]
        theta_low_limit = [ -0.349, 0.000, 0.000, 0.000,
                            -0.349, 0.000, 0.000, 0.000,
                            -0.349, 0.000, 0.000, 0.000,
                             0.000,-0.349, 0.000, 0.000, 0.000,
                            -1.047, 0.000,-0.209,-0.524,-1.571]
        max = torch.FloatTensor(theta_up_limit).to(self.device)#.cuda()
        min = torch.FloatTensor(theta_low_limit).to(self.device)#.cuda()
        delta_max = max - rotations
        delta_min = min - rotations
        return delta_max, delta_min

    def set_rotations_into_window(self):
        self.emb.weight.data[:, :self.J] = torch.where(self.emb.weight.data[:, :self.J] > self.delta_max, self.delta_max, self.emb.weight.data[:, :self.J])
        self.emb.weight.data[:, :self.J] = torch.where(self.emb.weight.data[:, :self.J] < self.delta_min, self.delta_min, self.emb.weight.data[:, :self.J])

    def transforms_blank(self, shape0, shape1):
        """
        transforms : (F, J, 4, 4) ndarray
            Array of identity transforms for
            each frame F and joint J
        """
        diagonal = torch.eye(4).to(self.device)
        ts = diagonal.expand(shape0, shape1, 4, 4)
        return ts

    def transforms_base(self, base):
        """
        base:(F, 7)—— x y z qw qx qy qz
        """
        base = base.unsqueeze(-2)
        rotations = base[:, :, 3:]
        q_length = torch.sqrt(torch.sum(rotations.pow(2), dim=-1))  # [F,J,1]
        qw = rotations[..., 0] / q_length  # [F,J,1]
        qx = rotations[..., 1] / q_length  # [F,J,1]
        qy = rotations[..., 2] / q_length  # [F,J,1]
        qz = rotations[..., 3] / q_length  # [F,J,1]
        """Unit quaternion based rotation matrix computation"""
        x2 = qx + qx  # [F,J,1]
        y2 = qy + qy
        z2 = qz + qz
        xx = qx * x2
        yy = qy * y2
        wx = qw * x2
        xy = qx * y2
        yz = qy * z2
        wy = qw * y2
        xz = qx * z2
        zz = qz * z2
        wz = qw * z2

        dim0 = torch.stack([1.0 - (yy + zz), xy - wz, xz + wy], -1)
        dim1 = torch.stack([xy + wz, 1.0 - (xx + zz), yz - wx], -1)
        dim2 = torch.stack([xz - wy, yz + wx, 1.0 - (xx + yy)], -1)
        R = torch.stack([dim0, dim1, dim2], -2).to(self.device)   # [F,1,3,3]

        T = base[..., :3].unsqueeze(-1).to(self.device)  # (F, 1, 3, 1)
        zeros = torch.zeros([int(base.shape[0]), 1, 1, 3]).to(self.device)  # (F, 1, 1, 3)
        ones = torch.ones([int(base.shape[0]), 1, 1, 1]).to(self.device)  # (F, 1, 1, 1)
        base_M = torch.cat([torch.cat([R, zeros], -2), torch.cat([T, ones], -2)], -1) # (F, 1, 4, 4)

        return base_M   # [F,1,4,4]

    def transforms_rotations(self, rotations):
        M_r = torch.eye(4).to(self.device).reshape(1, 1, 4, 4).repeat(self.F, self.J, 1, 1)
        M_r[:, :, :3, :3] = self.rodrigues.rot_axis_to_mat(rotations)
        return M_r   # [F,J,4,4]

    def joint2index(self, joints):
        """
        Food:        (0,  1),  2,     3  --            (FF4, FF3),   FF2,     FF1
        Middle:      (4,  5),  6,     7  --            (MF4, MF3),   MF2,     MF1
        Ring:        (8,  9), 10,    11  --            (RF4, RF3),   RF2,     RF1
        little: 12,  (13,14), 15,    16  --    LF5,    (LF4, LF3),   LF2,     LF1
        Thumb:       (17,18),(19,20),21  --            (TF5, TF4),(TF3, TF2), TF1
        """
        assert self.J == 22, 'J should be 22'

        # joints_1_index = [1, 5, 9, 14, 18]
        # joints_2_index = [2, 6, 10, 15, 19, 20]
        # joints_3_index = [3, 7, 11, 16, 21]
        
        joints_1_index = [0, 1, 4, 5, 8, 9, 13, 14, 17, 18]
        joints_2_index = [2, 6, 10, 12, 15, 19, 20]
        joints_3_index = [3, 7, 11, 16, 21]
        # joints_1_index = list(set(self.non_zero_joints_index) & set(joints_1_index))
        # joints_2_index = list(set(self.non_zero_joints_index) & set(joints_2_index))
        # joints_3_index = list(set(self.non_zero_joints_index) & set(joints_3_index))
        
        index = torch.zeros([3, self.J], dtype=torch.bool).to(self.device)
        index[0, joints_1_index] = True
        index[1, joints_2_index] = True
        index[2, joints_3_index] = True
        if isinstance(joints, int):
            assert joints in [1, 2, 3], 'joints should be in [1, 2, 3]'
            return index[joints-1].expand(self.F, self.J)
        elif isinstance(joints, list):
            index_ = torch.zeros(self.J, dtype=torch.bool).to(self.device)
            for j in joints:
                assert j in [1, 2, 3], 'joints should be in [1, 2, 3]'
                index_ = index_ | index[j - 1]
            return index_.expand(self.F, self.J)


    def forward(self, refine_joints=None, refine_part=''):
        """
        rotations: [F,J];
        refine_index: int or list in [1, 2, 3], representing 1st, 2nd and 3rd joints, default=None
        """
        F, J = self.F, self.J
        delta_positions = self.root_offset.weight
        delta_rotations = self.emb.weight
        
        base = self.base_M.clone()
        rotations = self.rotations.clone()

        base[:,:,:3,3] += delta_positions.reshape(self.F, 1, 3)
        refine_index = self.joint2index([1,2,3])
        delta_rotations = delta_rotations * refine_index
        rotations += delta_rotations # has grad
        
        # transforms_local
        M_r = self.transforms_rotations(rotations).to(self.device) # [F,J,4,4]
        M_sh = self.M_sh.unsqueeze(0).repeat(F, 1, 1, 1)  # [F,J,4,4]
        locals = torch.matmul(M_sh, M_r)  # [F,J,4,4]
        
        # transforms_global
        globals_blank = self.transforms_blank(F, J)  # [F,J,4,4]
        
        globals = torch.cat([base, globals_blank], 1)  # [F,J+1,4,4]
        globals_detached_base = torch.cat([base.detach(), globals_blank], 1)  # [F,J+1,4,4]
        for i in range(1, len(self.parents)):  # 从1号而非0号开始, [1,23]
            globals[:, i] = torch.matmul(globals[:, self.parents[i]], locals[:, i-1]) # [F, 4, 4]
            globals_detached_base[:, i] = torch.matmul(globals_detached_base[:, self.parents[i]], locals[:, i-1]) # [F, 4, 4]
        positions = globals[:, :, :3, 3]  # [F,J+1,4,4] --> [F,J+1,3]

        transformed_finger_pts_dict = dict()
        transformed_pts_list = []
        transformed_tips_list = []
        for i in range(self.BODY_NUM):
            if self.body_name_list[i] in ['palm', 'lfmetacarpal', 'F3', 'F2', 'F1', 'TH3_z', 'TH2_z', 'TH1_z']:
                pts = self.pts_dict[self.body_name_list[i]].unsqueeze(0).repeat(F, 1, 1)  # [F, 100 / 400, 3]
                if i == 18 or i == 20:
                    t = globals[:, i+1] # [F, 4, 4]
                else:
                    t = globals[:, i]
                R_mat = t[:,:3,:3].transpose(1, 2).contiguous() # [F, 3, 3]
                t_mat = t[:,:3,3].unsqueeze(1) # [F, 1, 3]
                pts_trans = torch.matmul(pts, R_mat) + t_mat # [F, 100 / 400, 3], 400 only for palm
                if self.body_name_list[i] in ['F1', 'TH1_z']:
                    transformed_tips_list.append(pts_trans[:, -1, :].unsqueeze(1))
                # else:
                #     transformed_pts_list.append(pts)
                transformed_pts_list.append(pts_trans)

        transformed_pts = torch.cat(transformed_pts_list, axis=1) # [F, 2000, 3] or [F, 1500, 3] (remove F1 & TH1 link)
        transformed_tips = torch.cat(transformed_tips_list, axis=1)  # [F, 5, 3]
        
        
        positions = torch.cat([positions, transformed_tips], axis=1)
        return positions, transformed_pts

if __name__ == "__main__":
    """
    Food:        (0,  1),  2,     3  --            (FF4, FF3),   FF2,     FF1
    Middle:      (4,  5),  6,     7  --            (MF4, MF3),   MF2,     MF1
    Ring:        (8,  9), 10,    11  --            (RF4, RF3),   RF2,     RF1
    little: 12,  (13,14), 15,    16  --    LF5,    (LF4, LF3),   LF2,     LF1
    Thumb:       (17,18),(19,20),21  --            (TF5, TF4),(TF3, TF2), TF1
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    base_M = torch.eye(4).reshape(1,1,4,4).to(device)

    rotations = [   0, 0, 0, 0,
                    0, 0, 0, 0,
                    0, 0, 0, 0,
                 0, 0, 0, 0, 0,
                 0, 0, 0, 0, -1.571]
                
    rotations = torch.FloatTensor(rotations).reshape(1, -1).to(device)

    fk_layers = FK_layer(base_M, rotations)
    fk_layers.to(device)

    print("Forward start")
    t_preforward = time.time()
    positions, transformed_pts = fk_layers()
    print("Forward end, time: {:.2f}s".format(time.time() - t_preforward))
    print(positions.shape, transformed_pts.size())
