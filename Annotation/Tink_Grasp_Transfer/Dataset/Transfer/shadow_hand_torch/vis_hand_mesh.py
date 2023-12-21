'''
Author: Yan Jianhang
Date: 2022-04-11 20:26:09
LastEditors: Yan Jianhang
LastEditTime: 2022-05-10 09:54:28
FilePath: /STransGrasp/lib/fk/vis_hand_mesh.py
Description:  FK of shadow hand in isaacgym. (numpy version)
'''

import sys
import copy
import open3d.visualization as vis
import open3d as o3d
import numpy as np
sys.path.append('./')
from lib.fk.utils import draw_3Daxes, o3d_read_mesh

def get_4x4_matrix(quat, pos):
    if isinstance(quat, list):
        quat = np.array(quat, np.float64)
        quat = quat[[3, 0, 1, 2]]
    if isinstance(pos, list):
        pos = np.array(pos, np.float64)
    t = np.eye(4).astype(np.float64)
    t[:3, 3] = pos
    t[:3, :3] = o3d.geometry.get_rotation_matrix_from_quaternion(quat)
    return t

def rot_axis_to_mat_Rodrigues(axis, theta):

    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    if isinstance(axis, list):
        axis = np.array(axis)
    axis = axis.reshape(3, 1)
    skew = np.array([[0, -axis[2,0], axis[1,0]], [axis[2,0], 0, -axis[0,0]], [-axis[1,0], axis[0,0], 0]])
    mat = cos_theta * np.eye(3) + (1 - cos_theta) * np.matmul(axis, axis.T) + sin_theta * skew
    return mat

if __name__ == "__main__":
    DOF_NUM = 22
    theta_up_limit = [  0.349, 1.571, 1.571, 1.571,
                        0.349, 1.571, 1.571, 1.571, 
                        0.349, 1.571, 1.571, 1.571,
                        0.785, 0.349, 1.571, 1.571, 1.571,
                        1.047, 1.222, 0.209, 0.524, 0]
    theta_low_limit = [      -0.349, 0.000, 0.000, 0.000,
                             -0.349, 0.000, 0.000, 0.000,
                             -0.349, 0.000, 0.000, 0.000,
                             0.000,-0.349, 0.000, 0.000, 0.000,
                             -1.047, 0.000,-0.209,-0.524, -1.571]

    rotations = [0, 0, 0, 0, 0,
                    0, 0, 0, 0,
                    0, 0, 0, 0,
                    0, 0, 0, 0,
                0, 0, 0, 0, 0]
    show_list = []
    
    mesh_root = 'assets/mjcf/open_ai_assets/stls/hand/'

    body_name_list = ['palm',          'knuckle', 'F3', 'F2', 'F1', 
                                       'knuckle', 'F3', 'F2', 'F1',
                                       'knuckle', 'F3', 'F2', 'F1',
                                       'lfmetacarpal', 'knuckle', 'F3', 'F2', 'F1',
                                       'TH3_z', '', 'TH2_z', '', 'TH1_z',]
    TH3_z_q = [ 0, 0.3824995, 0, 0.9239557 ]
    body_t_list = [[0, 0, 0], 
                                  [0.033, 0, 0.095], [0, 0, 0], [0, 0, 0.045], [0, 0, 0.025],
                                  [0.011, 0, 0.099], [0, 0, 0], [0, 0, 0.045], [0, 0, 0.025],
                                  [-0.011, 0, 0.095],[0, 0, 0], [0, 0, 0.045], [0, 0, 0.025],
                                  [-0.034, 0, 0.021], [-0.001, 0, 0.067], [0, 0, 0], [0, 0, 0.045], [0, 0, 0.025],
                                  [0.034, -0.009, 0.029], [0, 0, 0], [0, 0, 0.038], [0, 0, 0], [0, 0, 0.032]]

    rot_axis_list = [[0, 1, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0],
                    [0, 1, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0],
                    [0, 1, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0],
                    [0.011, 0, 0.099], [0, 1, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0],
                     [0, 0, -1], [1, 0, 0], [1, 0, 0], [0, 1, 0], [0, 1, 0]]
    
    joint_locals = np.eye(4)[np.newaxis, :]
    joint_locals = np.repeat(joint_locals, DOF_NUM, axis=0)
    for i in range(DOF_NUM):
        joint_locals[i][:3,:3] = rot_axis_to_mat_Rodrigues(rot_axis_list[i], rotations[i])

    body_parents = [-1,
                0, 1, 2, 3,
                0, 5, 6, 7,
                0, 9, 10, 11,
                0, 13, 14, 15, 16,
                0, 18, 19, 20, 21]
    joint_parents = []
    
    body_nums = len(body_parents)
    body_globals = np.eye(4)[np.newaxis, :]
    body_globals = np.repeat(body_globals, body_nums, axis=0)


    for i in range(1, body_nums):
        if body_name_list[i] == 'TH3_z':
            local = get_4x4_matrix(TH3_z_q, body_t_list[i]).astype(np.float64)
        else:
            local = get_4x4_matrix([0,0,0,1], body_t_list[i]).astype(np.float64)
        local = np.matmul(local, joint_locals[i-1])
        body_globals[i] = np.matmul(body_globals[body_parents[i]], local)

    # -------------- save pts of shadow hand------------------
    # pts_dict_of_shadow_hand = dict()
    # for name in ['palm', 'lfmetacarpal', 'F3', 'F2', 'F1', 'TH3_z', 'TH2_z', 'TH1_z']:
    #     mesh = o3d_read_mesh(mesh_root + '{}.stl'.format(name), scale=0.001)
    #     sampled_pts_num = 400 if name == 'palm' else 100
    #     pcd = mesh.sample_points_poisson_disk(number_of_points=sampled_pts_num, init_factor=10)
    #     pts = np.asarray(pcd.points)
    #     if name == 'F1':
    #         tip_point = np.array([0, 0, 0.0295287])[np.newaxis, ...]
    #         nearest_pt_idx = np.argmin(np.linalg.norm(pts - tip_point, axis=1, keepdims=False))
    #         print("F1 tip point nearest point is : {}".format(pts[nearest_pt_idx]))
    #         pts[nearest_pt_idx] = pts[-1]
    #         pts[-1] = tip_point
    #     if name == 'TH1_z':
    #         tip_point = np.array([0, 0, 0.03494929])[np.newaxis, ...]
    #         nearest_pt_idx = np.argmin(np.linalg.norm(pts - tip_point, axis=1, keepdims=False))
    #         print("TH1 tip point nearest point is : {}".format(pts[nearest_pt_idx]))
    #         pts[nearest_pt_idx] = pts[-1]
    #         pts[-1] = tip_point
    #     pts_dict_of_shadow_hand[name] = pts
    # np.save('./assets/sampled_pts_of_shadow_hand.npy', pts_dict_of_shadow_hand)

    pts_dict = np.load('./assets/sampled_pts_of_shadow_hand.npy', allow_pickle=True).item()
    for i in range(body_nums):
        show_list.append(draw_3Daxes(body_globals[i], size=0.03))
        if body_name_list[i] in ['palm', 'lfmetacarpal', 'F3', 'F2', 'F1', 'TH3_z', 'TH2_z', 'TH1_z']:
            pcd = o3d.geometry.PointCloud()
            pts = pts_dict[body_name_list[i]]
            if i == 18 or i == 20:
                t = body_globals[i+1]
            else:
                t = body_globals[i]
            pcd.points = o3d.utility.Vector3dVector(pts)
            pcd.transform(t)
            if body_name_list[i] in ['F1', 'TH1_z']:
                colors = np.zeros_like(pts)
                colors[-1] = np.array([1, 0, 0])
                pcd.colors = o3d.utility.Vector3dVector(colors)
            show_list.append(pcd)

    # --------------------init version of show (mesh format)--------------------
    # for i in range(body_nums):
    #     show_list.append(draw_3Daxes(body_globals[i], size=0.03))
    #     if body_name_list[i] != '':
    #         if i == 18 or i == 20:
    #             mesh = o3d_read_mesh(mesh_root + '{}.stl'.format(body_name_list[i]), scale=0.001)
    #             mesh.transform(body_globals[i+1])
    #         else:
    #             mesh = o3d_read_mesh(mesh_root + '{}.stl'.format(body_name_list[i]), scale=0.001)
    #             mesh.transform(body_globals[i])
                
    #         # pcd = mesh.sample_points_uniformly(number_of_points=1000)
    #         if body_name_list[i] == 'palm':
    #             pcd = mesh.sample_points_poisson_disk(number_of_points=400, init_factor=10)
    #             show_list.append(pcd)
    #         elif body_name_list[i] != 'knuckle':
    #             pcd = mesh.sample_points_poisson_disk(number_of_points=100, init_factor=10)
    #             show_list.append(pcd)
    #         else:
    #             continue

    vis.draw(show_list)