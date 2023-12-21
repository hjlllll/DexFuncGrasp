from xml.dom.minidom import parse
import xml.dom.minidom
import torch
import numpy as np
import os

def getfiles(dirPath, fileType, fileList = []):
    for f in os.listdir(dirPath):
        path = os.path.join(dirPath, f)
        if os.path.isdir(path):
            getfiles(path, fileType)
        if os.path.isfile(path):
            w = [i in f for i in fileType]
            if all(w):
                fileList.append(path)
    return fileList

def reading_xml(filename):
    DOMTree = xml.dom.minidom.parse(filename)
    collection = DOMTree.documentElement
    graspableBody = collection.getElementsByTagName('graspableBody')[0]
    filenames = filename
    file_name_i = graspableBody.getElementsByTagName('filename')[0].childNodes[0].data
    file_name = file_name_i.split('/')[-1].replace('.obj.smoothed', '')[:-4]
    # file_name = file_name_i.split('/')[-1].split('_')[0]#.replace('.obj.smoothed','')[:-6]

    obj_name = filenames.split('.xml')[0].split('_')[5]
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
    if robot_name[-6:-4] == 'lw':
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

    return a, r, t, file_name, rs_obj


import math
import numpy as np
import torch
from datasets.trans import *
from datasets.FK_layer_mesh import FK_layer
import pickle
from datasets.FK_model_opt import fk_run
from datasets.FK_layer_mesh_physic import FK_layer_physics
def trans_2_quat(R):
    # quat = torch.zeros(1, 4).cuda()
    # quat[0][3] = 0.5 * ((1 + R[0][0][0] + R[0][1][1] + R[0][2][2]).sqrt())
    # quat[0][0] = (R[0][2][1] - R[0][1][2]) / (4 * quat[0][0])
    # quat[0][1] = (R[0][0][2] - R[0][2][0]) / (4 * quat[0][0])
    # quat[0][3] = (R[0][1][0] - R[0][0][1]) / (4 * quat[0][0])
    quat = torch.zeros(4).cuda()
    # quat = np.zeros(4)
    quat[3] = 0.5 * (torch.sqrt(1 + R[0][0] + R[1][1] + R[2][2]))
    quat[0] = (R[2][1] - R[1][2]) / (4 * quat[3])
    quat[1] = (R[0][2] - R[2][0]) / (4 * quat[3])
    quat[2] = (R[1][0] - R[0][1]) / (4 * quat[3])
    return quat

def rt_to_transform(rotations,translations):# input [qw,qx,qy,qz] [x,y,z]
    """
    rotations : (F, J, 4) for each frame F and joint J
    """
    q_length = torch.sqrt(torch.sum(rotations.pow(2), dim=-1))  # [F,J,1]
    qw = rotations[..., 3] / q_length  # [F,J,1]
    qx = rotations[..., 0] / q_length  # [F,J,1]
    qy = rotations[..., 1] / q_length  # [F,J,1]
    qz = rotations[..., 2] / q_length  # [F,J,1]
    """Unit quaternion based rotation matrix computation""" """坐标系的定义和构建隐含在这部分了 """
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

    dim0 = torch.stack([1.0 - (yy + zz), xy - wz, xz + wy,translations[..., 0]], -1)
    dim1 = torch.stack([xy + wz, 1.0 - (xx + zz), yz - wx,translations[..., 1]], -1)
    dim2 = torch.stack([xz - wy, yz + wx, 1.0 - (xx + yy),translations[..., 2]], -1)
    dim3 = torch.tensor([0.0,0.0,0.0,1.0])
    m = torch.stack([dim0, dim1, dim2, dim3], -2)

    return m  # [F,J,4,4]

def r_to_transform(rotations):# input [qx,qy,qz,qw] [x,y,z]
    """
    rotations : (F, J, 4) for each frame F and joint J
    """
    q_length = torch.sqrt(torch.sum(rotations.pow(2), dim=-1))  # [F,J,1]
    qw = rotations[..., 3] / q_length  # [F,J,1]
    qx = rotations[..., 0] / q_length  # [F,J,1]
    qy = rotations[..., 1] / q_length  # [F,J,1]
    qz = rotations[..., 2] / q_length  # [F,J,1]
    """Unit quaternion based rotation matrix computation""" """坐标系的定义和构建隐含在这部分了 """
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
    m = torch.stack([dim0, dim1, dim2], -2)

    return m  # [F,J,3,3]

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


def trans_xml_pkl(input_a,input_t,input_r,device=None):
    view_mat_mesh = torch.tensor([[0.0, 0.0, 1.0],
                                  [-1.0, 0, 0.0],
                                  [0.0, -1.0, 0]])
    input_rxyzw = torch.zeros(4).to(device)#.cuda()
    input_rxyzw[:3] = input_r.reshape(4)[1:]
    input_rxyzw[3] = input_r.reshape(4)[0]
    r = quat_mul_tensor(input_rxyzw.reshape(4), trans_2_quat_gpu(view_mat_mesh,device)).reshape(4)

    index = [0, 1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19, 20, 21]
    base_M = torch.from_numpy(get_4x4_matrix_cuda(r, input_t)).float().reshape(1, 1, 4, 4).to(device)#.cuda()
    angles = input_a.reshape(1, -1).to(device)#.cuda() 
    fk_angles = torch.zeros((1,22))
    fk_angles[:, :4] = angles[:, 13:17]
    fk_angles[:, 4:8] = angles[:, 9:13]
    fk_angles[:, 8:12] = angles[:, 5:9]
    fk_angles[:, 12:17] = angles[:, :5]
    fk_angles[:, 17:] = angles[:, 17:]
    fk_angles[:, [0, 4]] = -fk_angles[:, [0, 4]]  ##############
    fk_angles[:, -2] = -fk_angles[:, -2]
    fk_angles[:, -1] = -fk_angles[:, -1]
    fk_angles = fk_angles.reshape(1, -1).to(device)#.cuda()
    fk_layers = FK_layer(base_M, fk_angles, device=device)
    fk_layers.to(device)#.cuda()
    positions, transformed_pts = fk_layers()
    input_a = input_a.reshape(1,-1)
    input_a[:, [3, 7, 11, 15]] += input_a[:, [4, 8, 12, 16]]
    inputa = input_a[:, index]

    new_j_p = fk_run(input_r.reshape(1,-1), input_t.reshape(1,-1), inputa/1.5708, device=device).squeeze(0)[:28]  # [:, idx_close]  # [1, 1+J+A, 3]

    return new_j_p, transformed_pts

def trans_xml_pkl_new(input_a,input_t,input_r,device=None):
    F = input_a.shape[0]
    view_mat_mesh = torch.tensor([[0.0, 0.0, 1.0],
                                  [-1.0, 0, 0.0],
                                  [0.0, -1.0, 0]])
    input_rxyzw = torch.zeros((F, 4)).to(device)#.cuda()
    input_rxyzw[:, :3] = input_r[:, 1:]
    input_rxyzw[:, 3] = input_r[:, 0]
    r = quat_mul_tensor(input_rxyzw, trans_2_quat_gpu(view_mat_mesh, device).reshape(1, 4).repeat(F, 1))#.reshape(4)

    index = [0, 1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19, 20, 21]
    base_M = get_4x4_matrix_tensor_batch(r, input_t, device).reshape(F, 1, 4, 4).to(device)#.cuda()
    angles = input_a.reshape(F, -1).to(device)#.cuda() #* 1.5708
    fk_angles = torch.zeros((F,22))
    fk_angles[:, :4] = angles[:, 13:17]
    fk_angles[:, 4:8] = angles[:, 9:13]
    fk_angles[:, 8:12] = angles[:, 5:9]
    fk_angles[:, 12:17] = angles[:, :5]
    fk_angles[:, 17:] = angles[:, 17:]
    fk_angles[:, [0, 4]] = -fk_angles[:, [0, 4]]  ########mention######
    fk_angles[:, -2] = -fk_angles[:, -2]
    fk_angles[:, -1] = -fk_angles[:, -1]
    fk_angles = fk_angles.reshape(F, -1).to(device)#.cuda()
    fk_layers = FK_layer(base_M, fk_angles, device=device, F=F)
    fk_layers.to(device)#.cuda()
    positions, transformed_pts = fk_layers()
    input_a = input_a.reshape(F,-1)
    input_a[:, [3, 7, 11, 15]] += input_a[:, [4, 8, 12, 16]]
    inputa = input_a[:, index]

    new_j_p = fk_run(input_r.reshape(F,-1), input_t.reshape(F,-1), inputa/1.5708, device=device)[:, :28]  # [:, idx_close]  # [1, 1+J+A, 3]

    return new_j_p, transformed_pts

def trans18_to_22_batch(input_a):
    joints = torch.zeros((input_a.shape[0], 22)).cuda()
    index = [0, 1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19, 20, 21]
    joints[:, index] = input_a.cuda().clone()  # * 1.5708
    indexs = [3, 7, 11, 15]
    joints[:, indexs] = 0.0
    angle_2_pair = torch.ones([2, input_a.shape[0]]).cuda()
    angle_1_pair = torch.zeros([2, input_a.shape[0]]).cuda()
    angle_2_pair[0] = input_a[:, 3]  # * 1.5708
    angle_1_pair[0] = input_a[:, 3] - 1  # * 1.5708 - 1.5708
    joints[:, 3] = torch.min(angle_2_pair, 0)[0]
    joints[:, 4] = torch.max(angle_1_pair, 0)[0]
    angle_2_pair[0] = input_a[:, 6]  # * 1.5708
    angle_1_pair[0] = input_a[:, 6] - 1  # * 1.5708 - 1.5708
    joints[:, 7] = torch.min(angle_2_pair, 0)[0]
    joints[:, 8] = torch.max(angle_1_pair, 0)[0]
    angle_2_pair[0] = input_a[:, 9]  # * 1.5708
    angle_1_pair[0] = input_a[:, 9] - 1  # * 1.5708 - 1.5708
    joints[:, 11] = torch.min(angle_2_pair, 0)[0]
    joints[:, 12] = torch.max(angle_1_pair, 0)[0]
    angle_2_pair[0] = input_a[:, 12]  # * 1.5708
    angle_1_pair[0] = input_a[:, 12] - 1  # * 1.5708 - 1.5708
    joints[:, 15] = torch.min(angle_2_pair, 0)[0]
    joints[:, 16] = torch.max(angle_1_pair, 0)[0]

    return joints

def get_4x4_matrix_tensor_batch(quat, pos, device):

    t = torch.eye(4).repeat(quat.shape[0],1,1,1)
    q_length = torch.sqrt(torch.sum(quat.pow(2), dim=-1))  # [F,J,1]
    qw = quat[..., 3] / q_length  # [F,J,1]
    qx = quat[..., 0] / q_length  # [F,J,1]
    qy = quat[..., 1] / q_length  # [F,J,1]
    qz = quat[..., 2] / q_length  # [F,J,1]
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
    t[:, :, :3, 3] = pos.unsqueeze(1)#.expand(-1, 1, -1)#.transpose(1,0)
    t[:, :, :3, :3] = torch.stack([dim0, dim1, dim2], -2).unsqueeze(1).to(device)#.cuda()#.expand(-1, 1, -1, -1).cuda()#.transpose(1,0).cuda()  # [F,1,3,3]
    return t


def trans_xml_pkl_batch(input_a,input_t,input_r,device):
    view_mat_mesh = torch.tensor([[0.0, 0.0, 1.0],
                                  [-1.0, 0, 0.0],
                                  [0.0, -1.0, 0]])

    mesh_r = torch.zeros(input_a.shape[0],4).to(device)
    mesh_r[:, :3] = input_r.clone()[:, 1:].to(device)
    mesh_r[:, 3] = input_r.clone()[:, 0].to(device)

    r = quat_mul_tensor(mesh_r, trans_2_quat_gpu(view_mat_mesh,device).reshape(1,4).repeat(input_a.shape[0],1))#.reshape(4)


    base_M = get_4x4_matrix_tensor_batch(r, input_t, device).reshape(r.shape[0], 1, 4, 4).to(device)#.cuda()

    joints = trans18_to_22_batch(input_a)

    angles = joints.to(device) * 1.5708
    fk_angles = torch.zeros((r.shape[0], 22)).to(device)#.cuda()
    fk_angles[:, :4] = angles.clone()[:, 13:17]
    fk_angles[:, 4:8] = angles.clone()[:, 9:13]
    fk_angles[:, 8:12] = angles.clone()[:, 5:9]
    fk_angles[:, 12:17] = angles.clone()[:, :5]
    fk_angles[:, 17:] = angles.clone()[:, 17:]
    fk_angles[:, [0, 4]] = -fk_angles[:, [0, 4]]  ########mention######
    fk_angles[:, -2] = -fk_angles[:, -2]
    fk_angles[:, -1] = -fk_angles[:, -1]
    fk_layers = FK_layer_physics()
    positions, transformed_pts = fk_layers.run(base_M, fk_angles,F=fk_angles.shape[0])

    outputs_F = fk_run(input_r, input_t, input_a, device)[:, :28]  # * 0.001

    return outputs_F, transformed_pts#, input_rwxyz.reshape(1,-1)


def farthest_point_sample_tensor(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids

def index_points(points, idx):
    """

    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points

def farthest_point_sample(point, npoint):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    xyz = point[:,:3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point


def farthest_points(data,
                    nclusters,
                    dist_func,
                    return_center_indexes=False,
                    return_distances=False,
                    verbose=False):
    """
      Performs farthest point sampling on data points.
      Args:
        data: numpy array of the data points.
        nclusters: int, number of clusters.
        dist_dunc: distance function that is used to compare two data points.
        return_center_indexes: bool, If True, returns the indexes of the center of
          clusters.
        return_distances: bool, If True, return distances of each point from centers.

      Returns clusters, [centers, distances]:
        clusters: numpy array containing the cluster index for each element in
          data.
        centers: numpy array containing the integer index of each center.
        distances: numpy array of [npoints] that contains the closest distance of
          each point to any of the cluster centers.
    """
    if nclusters >= data.shape[0]:
        if return_center_indexes:
            return np.arange(data.shape[0],
                             dtype=np.int32), np.arange(data.shape[0],
                                                        dtype=np.int32)

        return np.arange(data.shape[0], dtype=np.int32)

    clusters = np.ones((data.shape[0],), dtype=np.int32) * -1
    distances = np.ones((data.shape[0],), dtype=np.float32) * 1e7
    centers = []
    for iter in range(nclusters):
        index = np.argmax(distances)
        centers.append(index)
        shape = list(data.shape)
        for i in range(1, len(shape)):
            shape[i] = 1

        broadcasted_data = np.tile(np.expand_dims(data[index], 0), shape)
        new_distances = dist_func(broadcasted_data, data)
        distances = np.minimum(distances, new_distances)
        clusters[distances == new_distances] = iter
        if verbose:
            print('farthest points max distance : {}'.format(
                np.max(distances)))

    if return_center_indexes:
        if return_distances:
            return clusters, np.asarray(centers, dtype=np.int32), distances
        return clusters, np.asarray(centers, dtype=np.int32)

    return clusters


def distance_by_translation_point(p1, p2):
    """
      Gets two nx3 points and computes the distance between point p1 and p2.
    """
    return np.sqrt(np.sum(np.square(p1 - p2), axis=-1))


def regularize_pc_point_count(pc, npoints, use_farthest_point=False):
    """
      If point cloud pc has less points than npoints, it oversamples.
      Otherwise, it downsample the input pc to have npoint points.
      use_farthest_point: indicates whether to use farthest point sampling
      to downsample the points. Farthest point sampling version runs slower.
    """
    if pc.shape[0] > npoints:
        if use_farthest_point:
            _, center_indexes = farthest_points(pc,
                                                npoints,
                                                distance_by_translation_point,
                                                return_center_indexes=True)
        else:
            center_indexes = np.random.choice(range(pc.shape[0]),
                                              size=npoints,
                                              replace=False)
        pc = pc[center_indexes, :]
    else:
        required = npoints - pc.shape[0]
        if required > 0:
            index = np.random.choice(range(pc.shape[0]), size=required)
            pc = np.concatenate((pc, pc[index, :]), axis=0)
    return pc
def fps(points, k):
    sample_points = np.zeros((k, 3))


    point = points[0]
    points = points[:-1,:]
    sample_points[0,:] = point
    distance = np.sum((points - point)**2, axis=1)
    for i in range(k):
        distance = np.minimum(distance, np.sum((points - point) ** 2, axis=1))  ## 前缀思想更新最小值
        index = np.argmax(distance)
        point = points[index]
        sample_points[i, :] = point
        mask = np.ones((points.shape[0]), dtype=bool)
        mask[index] = False
        points = points[mask]
        distance = distance[mask]
    return sample_points


def random_rotate_debug_y():
    rotation_angle = np.random.uniform() * 2 * np.pi
    cosval = np.cos(rotation_angle)
    sinval = np.sin(rotation_angle)
    rotation_matrix = np.array([[cosval, 0, sinval],
                                [0, 1, 0],
                                [-sinval, 0, cosval]])
    return rotation_matrix

def random_rotate_debug():
    rotation_anglex = np.random.uniform() * 2 * np.pi
    cosvalx = np.cos(rotation_anglex)
    sinvalx = np.sin(rotation_anglex)
    rotation_angley = np.random.uniform() * 2 * np.pi
    cosvaly = np.cos(rotation_angley)
    sinvaly = np.sin(rotation_angley)
    rotation_anglez = np.random.uniform() * 2 * np.pi
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
    rotation_matrix = np.dot(Rz, np.dot(Ry,Rx))
    return rotation_matrix
def random_rotate_debug_small(angle_clip=0.18):
    angles_sigma = 0.06 # 权重,控制角度大小
    angles = np.clip(angles_sigma * np.random.randn(3), -angle_clip, angle_clip)
    cosvalx = np.cos(angles[0])
    sinvalx = np.sin(angles[0])
    cosvaly = np.cos(angles[1])
    sinvaly = np.sin(angles[1])
    cosvalz = np.cos(angles[2])
    sinvalz = np.sin(angles[2])
    Rx = np.array([[1, 0, 0],
                   [0, cosvalx, -sinvalx],
                   [0, sinvalx, cosvalx]])
    Ry = np.array([[cosvaly, 0, sinvaly],
                   [0, 1, 0],
                   [-sinvaly, 0, cosvaly]])
    Rz = np.array([[cosvalz, -sinvalz, 0],
                   [sinvalz, cosvalz, 0],
                   [0, 0, 1]])
    rotation_matrix = np.dot(Rz, np.dot(Ry,Rx))
    return rotation_matrix

def rotate_point_cloud_debug(batch_data, rotation_matrix):
    """ Randomly rotate the point clouds to augument the dataset
        rotation is per shape based along up direction
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """

    shape_pc = batch_data#[k, ...]
    rotated_data = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
    return rotated_data

def trans_2_quat_cpu_debug(R):
    # quat = torch.zeros(4)#.cuda()
    quat = np.zeros(4)
    quat[3] = 0.5 * (np.sqrt(1 + R[0][0] + R[1][1] + R[2][2]))
    quat[0] = (R[2][1] - R[1][2]) / (4 * quat[3])
    quat[1] = (R[0][2] - R[2][0]) / (4 * quat[3])
    quat[2] = (R[1][0] - R[0][1]) / (4 * quat[3])
    return quat #qx, qy, qz, qw

def quat_mul_tensor_debug(a, b):
    assert a.shape == b.shape
    shape = a.shape
    a = a.reshape(-1, 4)
    b = b.reshape(-1, 4)
    x1, y1, z1, w1 = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    x2, y2, z2, w2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    ww = (z1 + x1) * (x2 + y2)
    yy = (w1 - y1) * (w2 + z2)
    zz = (w1 + y1) * (w2 - z2)
    xx = ww + yy + zz
    qq = 0.5 * (xx + (z1 - x1) * (x2 - y2))
    w = qq - ww + (z1 - y1) * (y2 - z2)
    x = qq - xx + (x1 + w1) * (x2 + w2)
    y = qq - yy + (w1 - x1) * (y2 + z2)
    z = qq - zz + (z1 + y1) * (w2 - x2)

    quat = np.stack([x, y, z, w], axis=-1)  # .view(shape)

    return quat