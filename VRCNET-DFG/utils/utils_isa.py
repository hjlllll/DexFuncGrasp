import math
import numpy as np
import torch
# from trans import *
# from FK_layer_mesh import FK_layer
import pickle
# from FK_model_opt import fk_run

def trans_2_quat(R):
    quat = torch.zeros(1, 4).cuda()
    quat[0][3] = 0.5 * ((1 + R[0][0][0] + R[0][1][1] + R[0][2][2]).sqrt())
    quat[0][0] = (R[0][2][1] - R[0][1][2]) / (4 * quat[0][0])
    quat[0][1] = (R[0][0][2] - R[0][2][0]) / (4 * quat[0][0])
    quat[0][3] = (R[0][1][0] - R[0][0][1]) / (4 * quat[0][0])
    return quat
#
# def rt_to_transform(rotations,translations):# input [qw,qx,qy,qz] [x,y,z]
#     """
#     rotations : (F, J, 4) for each frame F and joint J
#     """
#     q_length = torch.sqrt(torch.sum(rotations.pow(2), dim=-1))  # [F,J,1]
#     qw = rotations[..., 3] / q_length  # [F,J,1]
#     qx = rotations[..., 0] / q_length  # [F,J,1]
#     qy = rotations[..., 1] / q_length  # [F,J,1]
#     qz = rotations[..., 2] / q_length  # [F,J,1]
#     """Unit quaternion based rotation matrix computation""" """坐标系的定义和构建隐含在这部分了 """
#     x2 = qx + qx  # [F,J,1]
#     y2 = qy + qy
#     z2 = qz + qz
#     xx = qx * x2
#     yy = qy * y2
#     wx = qw * x2
#     xy = qx * y2
#     yz = qy * z2
#     wy = qw * y2
#     xz = qx * z2
#     zz = qz * z2
#     wz = qw * z2
#
#     dim0 = torch.stack([1.0 - (yy + zz), xy - wz, xz + wy,translations[..., 0]], -1)
#     dim1 = torch.stack([xy + wz, 1.0 - (xx + zz), yz - wx,translations[..., 1]], -1)
#     dim2 = torch.stack([xz - wy, yz + wx, 1.0 - (xx + yy),translations[..., 2]], -1)
#     dim3 = torch.tensor([0.0,0.0,0.0,1.0])
#     m = torch.stack([dim0, dim1, dim2, dim3], -2)
#
#     return m  # [F,J,4,4]

# def r_to_transform(rotations):# input [qw,qx,qy,qz] [x,y,z]
#     """
#     rotations : (F, J, 4) for each frame F and joint J
#     """
#     q_length = torch.sqrt(torch.sum(rotations.pow(2), dim=-1))  # [F,J,1]
#     qw = rotations[..., 3] / q_length  # [F,J,1]
#     qx = rotations[..., 0] / q_length  # [F,J,1]
#     qy = rotations[..., 1] / q_length  # [F,J,1]
#     qz = rotations[..., 2] / q_length  # [F,J,1]
#     """Unit quaternion based rotation matrix computation""" """坐标系的定义和构建隐含在这部分了 """
#     x2 = qx + qx  # [F,J,1]
#     y2 = qy + qy
#     z2 = qz + qz
#     xx = qx * x2
#     yy = qy * y2
#     wx = qw * x2
#     xy = qx * y2
#     yz = qy * z2
#     wy = qw * y2
#     xz = qx * z2
#     zz = qz * z2
#     wz = qw * z2
#
#     dim0 = torch.stack([1.0 - (yy + zz), xy - wz, xz + wy], -1)
#     dim1 = torch.stack([xy + wz, 1.0 - (xx + zz), yz - wx], -1)
#     dim2 = torch.stack([xz - wy, yz + wx, 1.0 - (xx + yy)], -1)
#     m = torch.stack([dim0, dim1, dim2], -2)
#
#     return m  # [F,J,3,3]

def save_pcd_pkl(points, graspparts, obj_name, j_p, obj_index, num, path_pkl, shadowhandmesh_points):
    '''
    save pkl and visualize by open3d with pcd file
    '''
    # path_pcd_pkl = 'Refine_Pose'
    # path_pcd_pkl = 'Grasp_Pose'
    path_pcd_pkl = path_pkl
    # path = path_pcd_pkl + '/{}_{}.pkl'.format(obj_index, num)
    with open(path_pcd_pkl, 'wb') as file:
        pickle.dump((points,graspparts,obj_name,j_p,num,shadowhandmesh_points),file)
        # print('saving pcd_pkl in directory...')

# def trans_xml_pkl(input_a,input_t,input_r):
#     view_mat_mesh = torch.tensor([[0.0, 0.0, 1.0],
#                                   [-1.0, 0, 0.0],
#                                   [0.0, -1.0, 0]])
#     a = trans_2_quat_gpu(view_mat_mesh)
#     input_rxyzw = torch.zeros(4).cuda()
#     input_rxyzw[:3] = input_r.reshape(4)[1:]
#     input_rxyzw[3] = input_r.reshape(4)[0]
#     r = quat_mul_tensor(input_rxyzw.reshape(4), trans_2_quat_gpu(view_mat_mesh)).reshape(4)
#     # joints = torch.zeros((1,22)).cuda()
#     index = [0, 1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19, 20, 21]
#     # joints[:, index] = input_a.clone() #* 1.5708
#     # indexs = [3, 7,  11, 15]
#     # joints[:, indexs] = 0.0
#     # #[4,8,12,16]
#     # angle_2_pair = torch.ones([2, joints.shape[0]]).cuda()
#     # angle_1_pair = torch.zeros([2, joints.shape[0]]).cuda()
#     # angle_2_pair[0] = input_a[:, 3] #* 1.5708
#     # angle_1_pair[0] = input_a[:, 3] -1 #* 1.5708 - 1.5708
#     # joints[:, 3] = torch.min(angle_2_pair, 0)[0]
#     # joints[:, 4] = torch.max(angle_1_pair, 0)[0]
#     # angle_2_pair[0] = input_a[:, 6] #* 1.5708
#     # angle_1_pair[0] = input_a[:, 6] -1 #* 1.5708 - 1.5708
#     # joints[:, 7] = torch.min(angle_2_pair, 0)[0]
#     # joints[:, 8] = torch.max(angle_1_pair, 0)[0]
#     # angle_2_pair[0] = input_a[:, 9] #* 1.5708
#     # angle_1_pair[0] = input_a[:, 9] -1 #* 1.5708 - 1.5708
#     # joints[:, 11] = torch.min(angle_2_pair, 0)[0]
#     # joints[:, 12] = torch.max(angle_1_pair, 0)[0]
#     # angle_2_pair[0] = input_a[:, 12] #* 1.5708
#     # angle_1_pair[0] = input_a[:, 12] -1 #* 1.5708 - 1.5708
#     # joints[:, 15] = torch.min(angle_2_pair, 0)[0]
#     # joints[:, 16] = torch.max(angle_1_pair, 0)[0]
#     base_M = torch.from_numpy(get_4x4_matrix_cuda(r, input_t)).float().reshape(1, 1, 4, 4).cuda()
#     angles = input_a.reshape(1, -1).cuda() #* 1.5708
#     fk_angles = torch.zeros((1,22))
#     fk_angles[:, :4] = angles[:, 13:17]
#     fk_angles[:, 4:8] = angles[:, 9:13]
#     fk_angles[:, 8:12] = angles[:, 5:9]
#     fk_angles[:, 12:17] = angles[:, :5]
#     fk_angles[:, 17:] = angles[:, 17:]
#     fk_angles[:, [0, 4]] = -fk_angles[:, [0, 4]]  ########mention######
#     fk_angles[:, -2] = -fk_angles[:, -2]
#     fk_angles[:, -1] = -fk_angles[:, -1]
#     fk_angles = fk_angles.reshape(1, -1).cuda()
#     fk_layers = FK_layer(base_M, fk_angles)
#     fk_layers.cuda()
#     positions, transformed_pts = fk_layers()
#     inputa = torch.zeros(1,18).cuda()
#     input_a = input_a.reshape(1,-1)
#     input_a[:, [3, 7, 11, 15]] += input_a[:, [4, 8, 12, 16]]
#     inputa = input_a[:, index]
#
#     new_j_p = fk_run(input_r.reshape(1,-1), input_t.reshape(1,-1), inputa/1.5708).squeeze(0)[:28]  # [:, idx_close]  # [1, 1+J+A, 3]
#
#     return new_j_p, transformed_pts


