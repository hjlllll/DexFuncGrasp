# ----------------------------------------------
# Written by Kailin Li (kailinli@sjtu.edu.cn)
# ----------------------------------------------
import glob
import hashlib
import json
import os
import pickle
import re

import numpy as np
import open3d as o3d
import torch
import trimesh
# from liegroups import SO3
# from manotorch.manolayer import ManoLayer, MANOOutput
# from manotorch.utils.anchorutils import anchor_load_driver, recover_anchor
# from manotorch.utils.quatutils import quaternion_to_angle_axis
from termcolor import cprint
from trimesh.base import Trimesh
from utils.trans import *
from contact_utils import cal_dist, process_contact_info
from vis_contact_info import open3d_show

mesh_path_matcher = re.compile(r"interpolate/.+-.+/interp[0-9]{2}.ply$")


import trimesh
import trimesh.transformations as tra
from xml.dom.minidom import parse
import xml.dom.minidom
import copy

# class ShadowGripper():
#     """An object representing a Shadow gripper."""
#
#     def __init__(self ,data ,filename, q=None, num_contact_points_per_finger=10, root_folder=''):
#         """
#         Create a ShadowHand gripper object.
#
#         Keyword Arguments:
#             q {list of int} -- configuration (default: {None})
#             num_contact_points_per_finger {int} -- contact points per finger (default: {10})
#             root_folder {str} -- base folder for model files (default: {''})
#         """
#         self.joint_limits = [0.0, 0.04]
#         self.default_pregrasp_configuration = 0.04
#
#         if q is None:
#             q = self.default_pregrasp_configuration
#
#         self.q = q
#
#         #object
#
#         # self.obj = trimesh.load(filename)
#
#         # self.joints = reading_xml()
#         #lf5-lf1,rf4-rf1-->th5-th1
#
#
#         joint = data[7:]
#         for i in range(len(joint)):
#             joint[i] = joint[i] / np.pi
#         forearm = root_folder + 'gripper_models/shadowhand/forearm_electric.stl'
#         wrist = root_folder + 'gripper_models/shadowhand/wrist.stl'
#         palm = root_folder + 'gripper_models/shadowhand/palm.stl'
#         lfknuckle = root_folder + 'gripper_models/shadowhand/knuckle.stl'
#         LFJ4 = root_folder + 'gripper_models/shadowhand/lfmetacarpal.stl'
#         LFJ3 = root_folder + 'gripper_models/shadowhand/F3.stl'
#         LFJ2 = root_folder + 'gripper_models/shadowhand/F2.stl'
#         LFJ1 = root_folder + 'gripper_models/shadowhand/F1.stl'
#         rfknuckle = root_folder + 'gripper_models/shadowhand/knuckle.stl'
#         RFJ3 = root_folder + 'gripper_models/shadowhand/F3.stl'
#         RFJ2 = root_folder + 'gripper_models/shadowhand/F2.stl'
#         RFJ1 = root_folder + 'gripper_models/shadowhand/F1.stl'
#         mfknuckle = root_folder + 'gripper_models/shadowhand/knuckle.stl'
#         MFJ3 = root_folder + 'gripper_models/shadowhand/F3.stl'
#         MFJ2 = root_folder + 'gripper_models/shadowhand/F2.stl'
#         MFJ1 = root_folder + 'gripper_models/shadowhand/F1.stl'
#         ffknuckle = root_folder + 'gripper_models/shadowhand/knuckle.stl'
#         FFJ3 = root_folder + 'gripper_models/shadowhand/F3.stl'
#         FFJ2 = root_folder + 'gripper_models/shadowhand/F2.stl'
#         FFJ1 = root_folder + 'gripper_models/shadowhand/F1.stl'
#         TH3 = root_folder + 'gripper_models/shadowhand/TH3_z.stl'
#         TH2 = root_folder + 'gripper_models/shadowhand/TH2_z.stl'
#         TH1 = root_folder + 'gripper_models/shadowhand/TH1_z.stl'
#
#
#         self.forearm = trimesh.load(forearm)
#         self.wrist = trimesh.load(wrist)
#         self.palm = trimesh.load(palm)
#         self.LFJ4 = trimesh.load(LFJ4)
#         self.lfknuckle = trimesh.load(lfknuckle)
#         self.LFJ3 = trimesh.load(LFJ3)
#         self.LFJ2 = trimesh.load(LFJ2)
#         self.LFJ1 = trimesh.load(LFJ1)
#         self.rfknuckle = trimesh.load(rfknuckle)
#         self.RFJ3 = trimesh.load(RFJ3)
#         self.RFJ2 = trimesh.load(RFJ2)
#         self.RFJ1 = trimesh.load(RFJ1)
#         self.mfknuckle = trimesh.load(mfknuckle)
#         self.MFJ3 = trimesh.load(MFJ3)
#         self.MFJ2 = trimesh.load(MFJ2)
#         self.MFJ1 = trimesh.load(MFJ1)
#         self.ffknuckle = trimesh.load(ffknuckle)
#         self.FFJ3 = trimesh.load(FFJ3)
#         self.FFJ2 = trimesh.load(FFJ2)
#         self.FFJ1 = trimesh.load(FFJ1)
#         self.TH3 = trimesh.load(TH3)
#         self.TH2 = trimesh.load(TH2)
#         self.TH1 = trimesh.load(TH1)
#         self.palm.vertices *= 0.001
#         self.wrist.vertices *= 0.001
#         self.LFJ4.vertices *= 0.001
#         self.lfknuckle.vertices *= 0.001
#         self.LFJ3.vertices *= 0.001
#         self.LFJ2.vertices *= 0.001
#         self.LFJ1.vertices *= 0.001
#         self.rfknuckle.vertices *= 0.001
#         self.RFJ3.vertices *= 0.001
#         self.RFJ2.vertices *= 0.001
#         self.RFJ1.vertices *= 0.001
#         self.mfknuckle.vertices *= 0.001
#         self.MFJ3.vertices *= 0.001
#         self.MFJ2.vertices *= 0.001
#         self.MFJ1.vertices *= 0.001
#         self.ffknuckle.vertices *= 0.001
#         self.FFJ3.vertices *= 0.001
#         self.FFJ2.vertices *= 0.001
#         self.FFJ1.vertices *= 0.001
#         self.TH3.vertices *= 0.001
#         self.TH2.vertices *= 0.001
#         self.TH1.vertices *= 0.001
#         # self.obj.vertices *= 0.001
#
#         # transform fingers relative to the base
#         self.forearm.apply_translation([0 ,-0.01, 0])
#         self.wrist.apply_translation([0 ,0 ,0.256])
#         palm_t = np.array([0,0,0])
#
#         palm_t = np.array(data[4:7].cpu())
#         datas = data[:4].clone()# qw, qx, qy, qz
#         # datas[0] = data[1]
#         # datas[1] = data[2]
#         # datas[2] = data[3]
#         # datas[3] = data[0]
#         # self.palm.apply_transform(tra.quaternion_matrix(data[:4].cpu()))
#         # palm_r = tra.quaternion_matrix(data[:4].cpu())
#
#         zeros = tra.quaternion_matrix(datas.cpu()) #  qw, qx, qy, qz,
#         # zeros = np.array([0,1,0,0])#.cuda()
#         # zeros = tra.quaternion_matrix(zeros)
#         palm_r_ = tra.euler_matrix(-0.5 * np.pi, 0 , 0)
#         palm_r = np.dot(zeros,palm_r_)
#         palm_r__ = tra.euler_matrix(0,  0.5* np.pi, 0)
#         palm_r___ = np.dot(palm_r,palm_r__)
#         # palm_r____ = tra.euler_matrix(0.5*np.pi, 0, 0)
#         # palm_r_____ = np.dot(palm_r___, palm_r____)
#         # palm_r = np.dot(palm_r,palm_r___)
#         palm_r = palm_r___
#         self.palm.apply_transform(palm_r)
#         # palm_t = np.array([0,0,0])
#         self.palm.apply_translation(palm_t)
#
#         #LF
#         lfj4_t = np.array([-0.017-0.016 ,0 ,0.044-0.023])#why?
#         lfj4_r = tra.euler_matrix(0.571 * joint[0] * np.pi ,0 ,0.821 * joint[0] * np.pi)
#         lfj4_r = np.dot(palm_r, lfj4_r)
#         lfj4_t = np.dot(palm_r[:3,:3], lfj4_t)
#         self.LFJ4.apply_transform(lfj4_r)
#         self.LFJ4.apply_translation(palm_t + lfj4_t)
#         lfj3_t = np.array([-0.017+0.016 ,0 ,0.044+0.023])
#         lfj3_t = np.dot(lfj4_r[:3,:3],lfj3_t)
#         lfj3_r1 = tra.euler_matrix(0, joint[1] * np.pi, 0)
#         lfj3_r2 = tra.euler_matrix(joint[2] * np.pi, 0, 0)
#         lfj3_r = np.dot(lfj3_r2, lfj3_r1)
#         lfj3_r = np.dot(lfj4_r,lfj3_r)
#         self.LFJ3.apply_transform(lfj3_r)
#         self.LFJ3.apply_translation(palm_t + lfj4_t + lfj3_t)
#         lfknuckle_t = lfj3_t.copy()
#         self.lfknuckle.apply_transform(lfj3_r)
#         self.lfknuckle.apply_translation(palm_t + lfj4_t + lfknuckle_t)
#         lfj2_t = np.array([0, 0, 0.045])
#         lfj2_r = tra.euler_matrix(joint[3] * np.pi, 0, 0)
#         lfj2_r_ = np.dot(lfj3_r, lfj2_r)
#         self.LFJ2.apply_transform(lfj2_r_)
#         lfj2_t_ = np.dot(lfj3_r[:3, :3], lfj2_t)
#         self.LFJ2.apply_translation(palm_t + lfj4_t + lfj3_t + lfj2_t_)
#         lfj1_t = np.array([0, 0, 0.025])
#         lfj1_r = tra.euler_matrix(joint[4] * np.pi, 0, 0)
#         lfj1_r_ = np.dot(lfj2_r_, lfj1_r)
#         self.LFJ1.apply_transform(lfj1_r_)
#         lfj1_r_ = np.dot(lfj2_r_[:3, :3], lfj1_t)
#         self.LFJ1.apply_translation(palm_t + lfj4_t + lfj3_t + lfj2_t_ + lfj1_r_)
#
#         # RF
#         rfj3_t = np.array([-0.011, 0, 0.095])
#         rfj3_r1 = tra.euler_matrix(0, joint[5] * np.pi, 0)
#         rfj3_r2 = tra.euler_matrix(joint[6] * np.pi, 0, 0)
#         rfj3_r = np.dot(rfj3_r2, rfj3_r1)
#         rfj3_r = np.dot(palm_r, rfj3_r)
#         rfj3_t = np.dot(palm_r[:3, :3], rfj3_t)
#         self.RFJ3.apply_transform(rfj3_r)
#         self.RFJ3.apply_translation(palm_t + rfj3_t)
#         rfknuckle_t = rfj3_t.copy()
#         self.rfknuckle.apply_transform(rfj3_r)
#         self.rfknuckle.apply_translation(palm_t + rfknuckle_t)
#         rfj2_t = np.array([0, 0, 0.045])
#         rfj2_r = tra.euler_matrix(joint[7] * np.pi, 0, 0)
#         rfj2_r_ = np.dot(rfj3_r, rfj2_r)
#         self.RFJ2.apply_transform(rfj2_r_)
#         rfj2_t_ = np.dot(rfj3_r[:3, :3], rfj2_t)
#         self.RFJ2.apply_translation(palm_t + rfj3_t + rfj2_t_)
#         rfj1_t = np.array([0, 0, 0.025])
#         rfj1_r = tra.euler_matrix(joint[8] * np.pi, 0, 0)
#         rfj1_r_ = np.dot(rfj2_r_, rfj1_r)
#         self.RFJ1.apply_transform(rfj1_r_)
#         rfj1_r_ = np.dot(rfj2_r_[:3, :3], rfj1_t)
#         self.RFJ1.apply_translation(palm_t + rfj3_t + rfj2_t_ + rfj1_r_)
#
#         # MF
#         mfj3_t = np.array([0.011, 0, 0.095])
#         mfj3_r1 = tra.euler_matrix(0, joint[9] * np.pi, 0)
#         mfj3_r2 = tra.euler_matrix(joint[10] * np.pi, 0, 0)
#         mfj3_r = np.dot(mfj3_r2, mfj3_r1)
#         mfj3_r = np.dot(palm_r, mfj3_r)
#         mfj3_t = np.dot(palm_r[:3, :3], mfj3_t)
#         self.MFJ3.apply_transform(mfj3_r)
#         self.MFJ3.apply_translation(palm_t + mfj3_t)
#         mfknuckle_t = mfj3_t.copy()
#         self.mfknuckle.apply_transform(mfj3_r)
#         self.mfknuckle.apply_translation(palm_t + mfknuckle_t)
#         mfj2_t = np.array([0, 0, 0.045])
#         mfj2_r = tra.euler_matrix(joint[11] * np.pi, 0, 0)
#         mfj2_r_ = np.dot(mfj3_r, mfj2_r)
#         self.MFJ2.apply_transform(mfj2_r_)
#         mfj2_t_ = np.dot(mfj3_r[:3, :3], mfj2_t)
#         self.MFJ2.apply_translation(palm_t + mfj3_t + mfj2_t_)
#         mfj1_t = np.array([0, 0, 0.025])
#         mfj1_r = tra.euler_matrix(joint[12] * np.pi, 0, 0)
#         mfj1_r_ = np.dot(mfj2_r_, mfj1_r)
#         self.MFJ1.apply_transform(mfj1_r_)
#         mfj1_r_ = np.dot(mfj2_r_[:3, :3], mfj1_t)
#         self.MFJ1.apply_translation(palm_t + mfj3_t + mfj2_t_ + mfj1_r_)
#
#         #FF
#         ffj3_t = np.array([0.033 ,0 ,0.095])
#         ffj3_r1 = tra.euler_matrix(0, joint[13] * np.pi, 0)
#         ffj3_r2 = tra.euler_matrix(joint[14] * np.pi, 0, 0)
#         ffj3_r = np.dot(ffj3_r2,ffj3_r1)
#         ffj3_r = np.dot(palm_r, ffj3_r)
#         ffj3_t = np.dot(palm_r[:3, :3], ffj3_t)
#         self.FFJ3.apply_transform(ffj3_r)
#         self.FFJ3.apply_translation(palm_t + ffj3_t)
#         ffknuckle_t = ffj3_t.copy()
#         self.ffknuckle.apply_transform(ffj3_r)
#         self.ffknuckle.apply_translation(palm_t + ffknuckle_t)
#         ffj2_t = np.array([0,0,0.045])
#         ffj2_r = tra.euler_matrix(joint[15] * np.pi, 0, 0)
#         ffj2_r_= np.dot(ffj3_r,ffj2_r)
#         self.FFJ2.apply_transform(ffj2_r_)
#         ffj2_t_ = np.dot(ffj3_r[:3,:3],ffj2_t)
#         self.FFJ2.apply_translation(palm_t + ffj3_t + ffj2_t_)
#         ffj1_t = np.array([0,0,0.025])
#         ffj1_r = tra.euler_matrix(joint[16] * np.pi, 0, 0)
#         ffj1_r_= np.dot(ffj2_r_,ffj1_r)
#         self.FFJ1.apply_transform(ffj1_r_)
#         ffj1_r_ = np.dot(ffj2_r_[:3,:3],ffj1_t)
#         self.FFJ1.apply_translation(palm_t + ffj3_t + ffj2_t_ + ffj1_r_)
#
#         #TH
#         thj3_t = np.array([0.034 ,-0.009 ,0.029])
#         thj3_r0 = tra.euler_matrix(0 ,0.785 ,0)
#         thj3_r2 = tra.euler_matrix(0,0, -joint[17] * np.pi)
#         thj3_r1 = tra.euler_matrix(joint[18] * np.pi, 0, 0)
#         thj3_r = np.dot(thj3_r2, thj3_r1)
#         thj3_r = np.dot(thj3_r0, thj3_r)
#         thj3_r = np.dot(palm_r, thj3_r)
#         thj3_t = np.dot(palm_r[:3, :3], thj3_t)
#         self.TH3.apply_transform(thj3_r)
#         self.TH3.apply_translation(palm_t + thj3_t)
#         thj2_t = np.array([0, 0, 0.038])
#         thj2_r2 = tra.euler_matrix(joint[19] * np.pi, 0, 0)
#         thj2_r1 = tra.euler_matrix(0, -joint[20] * np.pi, 0) #+ ---> -
#         thj2_r = np.dot(thj2_r2, thj2_r1)
#         thj2_r = np.dot(thj3_r,thj2_r)
#         thj2_t = np.dot(thj3_r[:3,:3],thj2_t)
#         self.TH2.apply_transform(thj2_r)
#         self.TH2.apply_translation(palm_t + thj3_t + thj2_t)
#         thj1_t = np.array([0 ,0 ,0.032])
#         thj1_r = tra.euler_matrix(0, -joint[21] * np.pi, 0)#+ ---> -
#         thj1_t = np.dot(thj2_r[:3,:3],thj1_t)
#         thj1_r = np.dot(thj2_r,thj1_r)
#         self.TH1.apply_transform(thj1_r)
#         self.TH1.apply_translation(palm_t + thj3_t + thj2_t + thj1_t)
#
#
#         # self.fingers = trimesh.util.concatenate([self.finger_l, self.finger_r])
#         self.hand = trimesh.util.concatenate([self.palm,
#                                               self.LFJ4,self.lfknuckle,self.LFJ3,self.LFJ2,self.LFJ1,
#                                               self.rfknuckle,self.RFJ3,self.RFJ2,self.RFJ1,
#                                               self.mfknuckle,self.MFJ3,self.MFJ2,self.MFJ1,
#                                               self.ffknuckle,self.FFJ3,self.FFJ2,self.FFJ1,
#                                               self.TH3, self.TH2, self.TH1])
#                                               # ,self.obj])
#
#         self.ray_origins = []
#         self.ray_directions = []
#         for i in np.linspace(-0.01, 0.02, num_contact_points_per_finger):
#             self.ray_origins.append(
#                 np.r_[self.palm.bounding_box.centroid + [0, 0, i], 1])
#             self.ray_origins.append(
#                 np.r_[self.palm.bounding_box.centroid + [0, 0, i], 1])
#             # self.ray_directions.append(
#             #     np.r_[-self.palm.bounding_box.primitive.transform[:3, 0]])
#             # self.ray_directions.append(
#             #     np.r_[+self.palm.bounding_box.primitive.transform[:3, 0]])
#
#         self.ray_origins = np.array(self.ray_origins)
#         self.ray_directions = np.array(self.ray_directions)
#
#         self.standoff_range = np.array([max(self.palm.bounding_box.bounds[0, 2],
#                                             self.forearm.bounding_box.bounds[1, 2]),
#                                         self.palm.bounding_box.bounds[1, 2]])
#         self.standoff_range[0] += 0.001
#
#     def get_meshes(self):
#         """Get list of meshes that this gripper consists of.
#
#         Returns:
#             list of trimesh -- visual meshes
#         """
#         return self.hand
#         # return [self.palm, self.forearm, self.hand]
#     # def get_obbs(self):
#     #     """Get list of obstacle meshes.
#     #
#     #     Returns:
#     #         list of trimesh -- bounding boxes used for collision checking
#     #     """
#     #     return [self.palm.bounding_box, self.forearm.bounding_box]
#
#     # def get_closing_rays(self, transform):
#     #     """Get an array of rays defining the contact locations and directions on the hand.
#     #
#     #     Arguments:
#     #         transform {[nump.array]} -- a 4x4 homogeneous matrix
#     #
#     #     Returns:
#     #         numpy.array -- transformed rays (origin and direction)
#     #     """
#     #     return transform[:3, :].dot(
#     #         self.ray_origins.T).T, transform[:3, :3].dot(self.ray_directions.T).T

from util_shadow import ShadowGripper


def load_pointcloud(mesh_path: str, sample=5000):
    match_res = mesh_path_matcher.findall(mesh_path)
    pc_cache = mesh_path.replace("interpolate/", "interpolate_pc_cache/").replace(".ply", f"_pc{sample}.pkl")
    if len(match_res) > 0:
        # is interp ply
        if os.path.exists(pc_cache):
            print(f"load obj point cloud from {pc_cache}")
            pc_data = pickle.load(open(pc_cache, "rb"))
            obj_pc = o3d.geometry.PointCloud()
            obj_pc.points = o3d.utility.Vector3dVector(pc_data["points"])
            obj_pc.normals = o3d.utility.Vector3dVector(pc_data["normals"])
            return obj_pc
    mesh = trimesh.load(mesh_path, process=False, force="mesh", skip_materials=True)

    # bbox_center = (mesh.vertices.min(0) + mesh.vertices.max(0)) / 2
    # mesh.vertices = mesh.vertices - bbox_center

    o3d_obj_mesh = o3d.geometry.TriangleMesh()
    o3d_obj_mesh.triangles = o3d.utility.Vector3iVector(mesh.faces)
    o3d_obj_mesh.vertices = o3d.utility.Vector3dVector(mesh.vertices)
    o3d_obj_mesh.compute_triangle_normals()
    o3d_obj_mesh.compute_vertex_normals()
    obj_pc = o3d_obj_mesh.sample_points_poisson_disk(sample, seed=0)
    if len(match_res) > 0:
        print(f"dump obj point cloud to {pc_cache}")
        os.makedirs(os.path.split(pc_cache)[0], exist_ok=True)
        pickle.dump({"points": np.asarray(obj_pc.points), "normals": np.asarray(obj_pc.normals)}, open(pc_cache, "wb"))
    return obj_pc


def to_pointcloud(mesh, sample=5000):
    o3d_obj_mesh = o3d.geometry.TriangleMesh()
    o3d_obj_mesh.triangles = o3d.utility.Vector3iVector(mesh.faces)
    o3d_obj_mesh.vertices = o3d.utility.Vector3dVector(mesh.vertices)
    o3d_obj_mesh.compute_triangle_normals()
    o3d_obj_mesh.compute_vertex_normals()
    return o3d_obj_mesh.sample_points_poisson_disk(sample, seed=0)


def cal_contact_info(
    hand_verts,
    hand_faces,
    obj_verts,
):
    print('hand_faces:', hand_faces.shape)
    print('hand_verts:', hand_verts.shape)
    print('obj_verts:', obj_verts.shape)
    # **** mesh order of shadow hand in trimesh

    #         16-15-14-\
    #                   \
    #   13-- 12-- 11 ----0 palm
    #  10 -- 9 -- 8 ----/
    #   7 -- 6 -- 5 ---/
    #   4 - 3 - 2 -1--/
    # thumb = np.array([0,1,2,3,4,5,6,23,25,38,39,41,42,43,44,45,46,47,48,49,51,52,18])+10127 #23
    # ff = np.array([0,1,2,3,4,5,6,7,10,11,12,13,14,15,20,21,22,23,40,41,44,45,46,48,49,53,58,
    #                60,61,65,66,67,68,95,96,115,116,117,118,133,134,156,157,158,159,160,161,162,163])+9105#49
    # mf = np.array([0, 1, 2, 3, 4, 5, 6, 7, 10, 11, 12, 13, 14, 15, 20, 21, 22, 23, 40, 41, 44, 45, 46, 48, 49, 53, 58,
    #                60, 61, 65, 66, 67, 68, 95, 96, 115, 116, 117, 118, 133, 134, 156, 157, 158, 159, 160, 161, 162,163]) + 7845
    # rf = np.array([0, 1, 2, 3, 4, 5, 6, 7, 10, 11, 12, 13, 14, 15, 20, 21, 22, 23, 40, 41, 44, 45, 46, 48, 49, 53, 58,
    #                60, 61, 65, 66, 67, 68, 95, 96, 115, 116, 117, 118, 133, 134, 156, 157, 158, 159, 160, 161, 162,163]) + 6585
    # lf = np.array([0, 1, 2, 3, 4, 5, 6, 7, 10, 11, 12, 13, 14, 15, 20, 21, 22, 23, 40, 41, 44, 45, 46, 48, 49, 53, 58,
    #                60, 61, 65, 66, 67, 68, 95, 96, 115, 116, 117, 118, 133, 134, 156, 157, 158, 159, 160, 161, 162,163]) + 5325
    # finger = [thumb,ff,mf,rf,lf]
    # for i in finger:
    #     with open('./tink/gripper_models/assets/hand_id.txt','ab') as f:
    #         np.savetxt(f,i,fmt='%0.0f')

    hand_index = np.loadtxt(os.path.join("./tink/gripper_models/assets/hand_id.txt"), dtype=np.int32)
    # print(hand_index.shape)# [23,49,49,49,49] 219
    # hand_palm_vertex_index = np.loadtxt(os.path.join("./assets/hand_palm_full.txt"), dtype=np.int32)
    # face_vertex_index, anchor_weight, merged_vertex_assignment, anchor_mapping = anchor_load_driver("./assets")
    # n_regions = len(np.unique(merged_vertex_assignment))

    n_regions = 5 # five fingertips of shadow
    hand_mesh = o3d.geometry.TriangleMesh()
    hand_mesh.triangles = o3d.utility.Vector3iVector(hand_faces)
    hand_mesh.vertices = o3d.utility.Vector3dVector(hand_verts)
    hand_mesh.compute_triangle_normals()
    hand_mesh.compute_vertex_normals()
    hand_normals = np.asarray(hand_mesh.vertex_normals)

    hand_verts_selected = hand_verts[hand_index]
    hand_normals_selected = hand_normals[hand_index]
    merged_vertex_assignment_selected = hand_index
    # merged_vertex_assignment_selected = hand_verts[:,0]#[:obj_verts.shape[0],0]
    # merged_vertex_assignment_selected = merged_vertex_assignment#[hand_palm_vertex_index]

    # anchor_pos = recover_anchor(hand_verts, face_vertex_index, anchor_weight)
    anchor_pos = hand_faces
    contact_info_list = cal_dist(
        hand_verts=hand_verts_selected,
        hand_normals=hand_normals_selected,
        obj_verts=obj_verts,
        hand_verts_region=merged_vertex_assignment_selected,
        n_regions=n_regions,
        anchor_pos=anchor_pos,
        # anchor_mapping=anchor_mapping,
    )
    # vertex_contact, hand_region, anchor_id, anchor_dist, anchor_elasti, anchor_padding_mask = process_contact_info(
    #     contact_info_list,
    #     # anchor_mapping,
    #     pad_vertex=True,
    #     pad_anchor=True,
    #     dist_th=1000.0,
    #     elasti_th=0.0,)

    vertex_contact, hand_region, anchor_id = process_contact_info(
        contact_info_list,
        # anchor_mapping,
        pad_vertex=True,
        pad_anchor=True,
        dist_th=1000.0,
        elasti_th=0.0, )

    return {
        "vertex_contact": vertex_contact,
        "hand_region": hand_region,
        "anchor_id": anchor_id,
        # "anchor_dist": anchor_dist,
        # "anchor_elasti": anchor_elasti,
        # "anchor_padding_mask": anchor_padding_mask,
    }



def get_hand_parameter(pose_path):

    pose = pickle.load(open(pose_path, "rb"))
    hand_pose, hand_shape, hand_tsl = pose["hand_pose"], pose["hand_shape"].numpy(), pose["hand_tsl"].numpy()
    hand_pose = quaternion_to_angle_axis(hand_pose.reshape(16, 4)).reshape(48).numpy()
    obj_rot, obj_tsl = pose["obj_transf"][:3, :3].numpy(), pose["obj_transf"][:3, 3].T.numpy()

    hand_gr = SO3.exp(hand_pose[:3]).as_matrix()
    hand_gr = obj_rot.T @ hand_gr
    hand_gr = SO3.log(SO3.from_matrix(hand_gr, normalize=True))
    hand_pose[:3] = hand_gr
    hand_tsl = obj_rot.T @ (hand_tsl - obj_tsl)

    return hand_pose, hand_shape, hand_tsl

def reading_xml(filename):
    DOMTree = xml.dom.minidom.parse(filename)
    collection = DOMTree.documentElement
    graspableBody = collection.getElementsByTagName('graspableBody')[0]
    filenames = filename
    file_name_i = graspableBody.getElementsByTagName('filename')[0].childNodes[0].data
    file_name = file_name_i.split('/')[-1].replace('.obj.smoothed', '')[:-4]
    # file_name = file_name_i.split('/')[-1].split('_')[0]#.replace('.obj.smoothed','')[:-6]

    # obj_name = filenames.split('.xml')[0].split('_')[5]
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
    # for i in range(4):
    rs_obj[0] = r_obj[1]
    rs_obj[1] = r_obj[2]
    rs_obj[2] = r_obj[3]
    rs_obj[3] = r_obj[0]

    r = rt.split(')')[0].split('(')[1].split(' ')
    t = rt.split('[')[1].split(']')[0].split(' ')

    a = a.replace('   ', ' ')
    a = np.array(a.split(' '), dtype='float32')
    a = torch.tensor(a, dtype=torch.float32).unsqueeze(0)
    if robot_name[-6:-4] == 'lw':
        a[:, 1] = -a[:, 1]
    elif robot_name[-6:-4] != 'nd':
        raise ('robot_name has wrong, place call zhutq')
    r = np.array(r, dtype='float32')
    r = torch.tensor(r, dtype=torch.float32)  # .unsqueeze(0)
    rs = copy.deepcopy(r)
    # for i in range(4):
    rs[0] = r[1]
    rs[1] = r[2]
    rs[2] = r[3]
    rs[3] = r[0]

    t = np.array(t, dtype='float32')
    t = torch.tensor(t, dtype=torch.float32) #* 0.001  # .unsqueeze(0)

    return a, rs, t, file_name, rs_obj


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    # parser.add_argument("--data", "-d", type=str, required=True)
    # parser.add_argument("--source", "-s", type=str, required=True)
    parser.add_argument("--tag", "-t", type=str, default="trans") #debug
    parser.add_argument("--pose_path", "-p", type=str)  # grasp pose path
    parser.add_argument("--vis", action="store_true")
    parser.add_argument('--idx', type=int, default=19, help="0-20 id of hjl new dataset, means category")

    args = parser.parse_args()

    #reading object mesh
    # real_meta = json.load(open("./DeepSDF_OakInk/data/meta/object_id.json", "r"))
    # virtual_meta = json.load(open("./DeepSDF_OakInk/data/meta/virtual_object_id.json", "r"))

    f = open('../Dataset/Obj_Data/obj_data.json', 'r')
    content = json.loads(f.read())
    category = content[str(args.idx)]['category']
    data_root = '../Dataset/Obj_Data'
    real_meta_root = os.path.join(data_root, 'split', 'split.json')
    # real_meta = json.load(open('../Dataset/Objects/bottle/split/split.json', "r"))
    # real_meta = json.load(open(real_meta_root, 'r'))
    # virtual_meta = json.load(open(real_meta_root, 'r'))
    # virtual_meta = json.load(open('../Dataset/Objects/bottle/split/split.json', "r"))


    # if args.source in real_meta['52']['real']:
        # obj_name = real_meta['52']['real'][args.source]

    # if args.source in real_meta[str(args.idx)]['real']:
    #     obj_name = args.source
    #
    #     # obj_path = "DeepSDF_OakInk/data/OakInkObjects"
    #     # obj_path = '../Dataset/Objects'
    #     obj_path = '../Dataset/Obj_Data'
    #
    # else:
    #     obj_name = virtual_meta[args.source]["virtual"]
    #     obj_path = "DeepSDF_OakInk/data/OakInkVirtualObjects"
    # obj_mesh_path = os.path.join(obj_path,category, 'obj', "{}.obj".format(obj_name))
    # # obj_mesh_path = glob.glob(os.path.join(obj_path, 'bottle', 'obj', "{}.obj".format(obj_name))) #+ glob.glob(
    # #     # os.path.join(obj_path, obj_name, 'ply', "*.ply"))
    # # assert len(obj_mesh_path) == 1
    # obj_pc = load_pointcloud(obj_mesh_path)


    #######change to new hjl data#######
    #从 split_interpolate.json中提取 相当于从grasps文件夹中提取抓取
    f = open('../Dataset/Obj_Data/obj_data.json', 'r')
    content = json.loads(f.read())
    category = content[str(args.idx)]['category']
    data = os.path.join(data_root, category, 'split')
    data_root = '../Dataset/Obj_Data'
    split_interpolate_file = open(os.path.join(data_root, category, 'split', 'split_interpolate.json'))
    grasp_file = json.loads(split_interpolate_file.read())[str(args.idx)]['real']
    grasp_root = '../Dataset/Grasps'
    grasp_list = []
    for file in grasp_file:
        file = str(args.idx) + '_' + file
        grasp_list.append(os.path.join(grasp_root, file, 'new'))
    #grasp_list是同一类物体的不同实例的抓取文件夹
    #进入循环
    for grasp_instance in grasp_list:
        grasps = os.listdir(grasp_instance)
        for each_grasp in grasps:
            grasp_id = each_grasp.split('_')[1].split('.')[0]
            each_grasp = os.path.join(grasp_instance, each_grasp)
            if each_grasp.endswith('.xml'):
                a, rs, t, file_name, rs_obj = reading_xml(each_grasp)  # x y z w
                with open(each_grasp[:-4] + '.pkl', 'rb') as filename:
                    points, graspparts, obj_name, j_p, num, hand_mesh_points = pickle.load(filename)

                view_mat_mesh = torch.tensor([[0.0, 1.0, 0.0],
                                              [-1.0, 0, 0.0],
                                              [0.0, 0.0, 1.0]])


                rs = quat_mul(rs.clone().cpu().numpy().reshape(4), trans_2_quat_cpu(view_mat_mesh)).reshape(4)
                input_rwxyz = torch.zeros(4)
                input_rwxyz[1:] = torch.from_numpy(rs).reshape(4)[:3]
                input_rwxyz[0] = torch.from_numpy(rs).reshape(4)[3]

                joint = torch.zeros(29)
                joint[4:7] = t
                joint[:4] = input_rwxyz
                joint[7:] = a
                shadowhand = ShadowGripper(root_folder='./tink/', filename='')
                shadow_mesh = shadowhand.get_meshes(joint)

                source = obj_name[:-4]



                obj_mesh_path = os.path.join(data_root, category, 'obj', "{}".format(obj_name))
                obj_pc = load_pointcloud(obj_mesh_path)

                # saving contact_info
                contact_info = cal_contact_info(
                    np.array(shadow_mesh.vertices.squeeze()), shadow_mesh.faces, np.asarray(obj_pc.points)*1000.0)
                contactinfo_path = os.path.join(data, "contact", f"{source}", f"{args.tag}")
                os.makedirs(contactinfo_path, exist_ok=True)
                pickle.dump(contact_info,
                            open(os.path.join(contactinfo_path, "{}_{}_contact_info.pkl".format(source, grasp_id)),
                                 "wb"))

                if args.vis:
                    open3d_show(
                        # obj_points=obj_pc.points,
                        obj_points=None,
                        obj_verts=obj_pc.points,
                        obj_normals=None,
                        contact_info=contact_info,
                        hand_verts=shadow_mesh.vertices.squeeze(),
                        hand_faces=shadow_mesh.faces,
                        show_hand_normals=True,
                    )







    #reading shadowhand pose
    # joint = torch.Tensor([0,1,0,0,-0.12,-0.04,0.13,0.260677, -0.12708819, 0.44815338, 1.5708, 0.05383605403900149,
    #          0.0098000765, 0.5904129, 1.0705094, 0, 0.010692179,
    #          0.5704359, 1.5708, 0.21206278247833255, 0.014512147, 0.3004437,
    #          1.383557, 0,0.14294301 ,0.7534741 ,0.086198345 ,0.2972183, 0.8795107])
    # shadowhand = ShadowGripper(root_folder='./tink/',data=joint,filename='')
    # shadow_mesh = shadowhand.get_meshes()
    # open3d_show(
    #     obj_verts=obj_pc.points,
    #     obj_normals=obj_pc.normals,
    #     contact_info=None,
    #     hand_verts=shadow_mesh.vertices.squeeze(),
    #     hand_faces=shadow_mesh.faces,
    #     show_hand_normals=False,)


    #
    # hand_pose, hand_shape, hand_tsl = get_hand_parameter(args.pose_path)
    # hash_hand = hashlib.md5(pickle.dumps(np.concatenate([hand_pose, hand_shape, hand_tsl]))).hexdigest()
    #
    # contactinfo_path = os.path.join(args.data, "contact", f"{args.source}", f"{args.tag}_{hash_hand[:10]}")
    # os.makedirs(contactinfo_path, exist_ok=True)

    # if os.path.exists(os.path.join(contactinfo_path, "contact_info.pkl")):
    #     cprint(f"{contactinfo_path} exists, skip.", "yellow")
    #     exit(0)

    # pickle.dump(
    #     {"pose": hand_pose, "shape": hand_shape, "tsl": hand_tsl},
    #     open(os.path.join(contactinfo_path, "hand_param.pkl"), "wb"),
    # )
    #
    # with open(os.path.join(contactinfo_path, "source.txt"), "w") as f:
    #     f.write(args.pose_path)

    # mano_layer = ManoLayer(center_idx=0, mano_assets_root="assets/mano_v1_2")
    # mano_output: MANOOutput = mano_layer(
    #     torch.from_numpy(hand_pose).unsqueeze(0), torch.from_numpy(hand_shape).unsqueeze(0))
    # hand_faces = mano_layer.th_faces.numpy()

    # contact_info = cal_contact_info(
    #     mano_output.verts.squeeze().numpy() + hand_tsl[None], hand_faces, np.asarray(obj_pc.points))



    # saving contact_info
    # contact_info = cal_contact_info(
    #     np.array(shadow_mesh.vertices.squeeze()), shadow_mesh.faces, np.asarray(obj_pc.points))
    # contactinfo_path = os.path.join(args.data, "contact", f"{args.source}", f"{args.tag}")
    # os.makedirs(contactinfo_path, exist_ok=True)
    # pickle.dump(contact_info, open(os.path.join(contactinfo_path, "contact_info.pkl"), "wb"))

    # file = '../Dataset/Objects/bottle/split/contact/bottle26/trans/bottle12/contact_info.pkl'
    # contact_info = pickle.load(open(file,'rb'))

    # if args.vis:
    #     open3d_show(
    #         obj_verts=obj_pc.points,
    #         obj_normals=obj_pc.normals,
    #         contact_info=contact_info,
    #         hand_verts=shadow_mesh.vertices.squeeze(),
    #         hand_faces=shadow_mesh.faces,
    #         show_hand_normals=True,
    #     )
