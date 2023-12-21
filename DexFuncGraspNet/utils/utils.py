import numpy as np
import copy
import os
import math
import time
import trimesh.transformations as tra
import json
from utils import sample
import torch
import yaml
from easydict import EasyDict as edict
from xml.dom.minidom import parse
import xml.dom.minidom

from datasets.utils import *

def trans_rtj_point(outputs_base, outputs_a):
    outputs_rotation = torch.zeros([outputs_a.shape[0], 27]).type_as(outputs_a)  # .cuda()
    outputs_rotation[:, 0:3] = outputs_a[:, 0:3]
    angle_2_pair = torch.ones([2, outputs_a.shape[0]])  # .cuda()
    angle_1_pair = torch.zeros([2, outputs_a.shape[0]])  # .cuda()
    outputs_rotation[:, 4] = outputs_rotation[:, 3] * 0.8
    outputs_rotation[:, 6:8] = outputs_a[:, 4:6]

    outputs_rotation[:, 9] = outputs_rotation[:, 8] * 0.8
    outputs_rotation[:, 11:13] = outputs_a[:, 7:9]

    outputs_rotation[:, 14] = outputs_rotation[:, 13] * 0.8
    outputs_rotation[:, 16:18] = outputs_a[:, 10:12]

    outputs_rotation[:, 18] = outputs_rotation[:, 17] * 0.8
    outputs_rotation[:, 21:26] = outputs_a[:, 13:]  # all
    return outputs_base, outputs_rotation

def reading_xml(filename):
    DOMTree = xml.dom.minidom.parse(filename)
    collection = DOMTree.documentElement
    graspableBody = collection.getElementsByTagName('graspableBody')[0]
    filenames = filename
    file_name_i = graspableBody.getElementsByTagName('filename')[0].childNodes[0].data
    file_name = file_name_i.split('/')[-1].replace('.obj.smoothed', '')[:-4]

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

    rs_obj = r_obj.copy()
    for i in range(4):
        rs_obj[0] = r_obj[1]
        rs_obj[1] = r_obj[2]
        rs_obj[2] = r_obj[3]
        rs_obj[3] = r_obj[0]

    r = rt.split(')')[0].split('(')[1].split(' ')
    t = rt.split('[')[1].split(']')[0].split(' ')

    a = a.replace('   ', ' ')
    a = np.array(a.split(' '), dtype='float32')
    if robot_name[-6:-4] == 'lw':
        a[:, 1] = -a[:, 1]
    elif robot_name[-6:-4] != 'nd':
        raise ('robot_name has wrong, place call zhutq')
    r = np.array(r, dtype='float32')
    rs = r.copy()
    for i in range(4):
        rs[0] = r[1]
        rs[1] = r[2]
        rs[2] = r[3]
        rs[3] = r[0]

    t = np.array(t, dtype='float32')* 0.001

    return a, rs, t, file_name, rs_obj



def quaternion_mult(q, r):
    """
    Multiply quaternion(s) q with quaternion(s) r.
    Expects two equally-sized tensors of shape (*, 4), where * denotes any number of dimensions.
    Returns q*r as a tensor of shape (*, 4).
    """
    assert q.shape[-1] == 4
    assert r.shape[-1] == 4

    original_shape = q.shape

    # Compute outer product
    terms = torch.bmm(r.view(-1, 4, 1), q.view(-1, 1, 4))

    w = terms[:, 0, 0] - terms[:, 1, 1] - terms[:, 2, 2] - terms[:, 3, 3]
    x = terms[:, 0, 1] + terms[:, 1, 0] - terms[:, 2, 3] + terms[:, 3, 2]
    y = terms[:, 0, 2] + terms[:, 1, 3] + terms[:, 2, 0] - terms[:, 3, 1]
    z = terms[:, 0, 3] - terms[:, 1, 2] + terms[:, 2, 1] + terms[:, 3, 0]
    return torch.stack((w, x, y, z), dim=1).view(original_shape)


def conj_quaternion(q):
    """
      Conjugate of quaternion q.
    """
    q_conj = q.clone()
    q_conj[:, :, 1:] *= -1
    return q_conj


def rotate_point_by_quaternion(point, q, device="cpu"):
    """
      Takes in points with shape of (batch_size x n x 3) and quaternions with
      shape of (batch_size x n x 4) and returns a tensor with shape of 
      (batch_size x n x 3) which is the rotation of the point with quaternion
      q. 
    """
    shape = point.shape
    q_shape = q.shape

    assert (len(shape) == 3), 'point shape = {} q shape = {}'.format(
        shape, q_shape)
    assert (shape[-1] == 3), 'point shape = {} q shape = {}'.format(
        shape, q_shape)
    assert (len(q_shape) == 3), 'point shape = {} q shape = {}'.format(
        shape, q_shape)
    assert (q_shape[-1] == 4), 'point shape = {} q shape = {}'.format(
        shape, q_shape)
    assert (q_shape[1] == shape[1]), 'point shape = {} q shape = {}'.format(
        shape, q_shape)

    q_conj = conj_quaternion(q)
    r = torch.cat([
        torch.zeros(
            (shape[0], shape[1], 1), dtype=point.dtype).to(device), point
    ],
                  dim=-1)
    final_point = quaternion_mult(quaternion_mult(q, r), q_conj)
    final_output = final_point[:, :,
                               1:]  #torch.slice(final_point, [0, 0, 1], shape)
    return final_output


def tc_rotation_matrix(az, el, th, batched=False):
    if batched:

        cx = torch.cos(torch.reshape(az, [-1, 1]))
        cy = torch.cos(torch.reshape(el, [-1, 1]))
        cz = torch.cos(torch.reshape(th, [-1, 1]))
        sx = torch.sin(torch.reshape(az, [-1, 1]))
        sy = torch.sin(torch.reshape(el, [-1, 1]))
        sz = torch.sin(torch.reshape(th, [-1, 1]))

        ones = torch.ones_like(cx)
        zeros = torch.zeros_like(cx)

        rx = torch.cat([ones, zeros, zeros, zeros, cx, -sx, zeros, sx, cx],
                       dim=-1)
        ry = torch.cat([cy, zeros, sy, zeros, ones, zeros, -sy, zeros, cy],
                       dim=-1)
        rz = torch.cat([cz, -sz, zeros, sz, cz, zeros, zeros, zeros, ones],
                       dim=-1)

        rx = torch.reshape(rx, [-1, 3, 3])
        ry = torch.reshape(ry, [-1, 3, 3])
        rz = torch.reshape(rz, [-1, 3, 3])

        return torch.matmul(rz, torch.matmul(ry, rx))
    else:
        cx = torch.cos(az)
        cy = torch.cos(el)
        cz = torch.cos(th)
        sx = torch.sin(az)
        sy = torch.sin(el)
        sz = torch.sin(th)

        rx = torch.stack([[1., 0., 0.], [0, cx, -sx], [0, sx, cx]], dim=0)
        ry = torch.stack([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dim=0)
        rz = torch.stack([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dim=0)

        return torch.matmul(rz, torch.matmul(ry, rx))


def mkdir(path):
    if not os.path.isdir(path):
        os.makedirs(path)




def rot_and_trans_to_grasps(euler_angles, translations, selection_mask):
    grasps = []
    refine_indexes, sample_indexes = np.where(selection_mask)
    for refine_index, sample_index in zip(refine_indexes, sample_indexes):
        rt = tra.euler_matrix(*euler_angles[refine_index, sample_index, :])
        rt[:3, 3] = translations[refine_index, sample_index, :]
        grasps.append(rt)
    return grasps


def convert_qt_to_rt(grasps):
    Ts = grasps[:, 4:]
    Rs = qeuler(grasps[:, :4], "zyx")
    return Rs, Ts


def qeuler(q, order, epsilon=0):
    """
    Convert quaternion(s) q to Euler angles.
    Expects a tensor of shape (*, 4), where * denotes any number of dimensions. # w x y z
    Returns a tensor of shape (*, 3).
    """
    assert q.shape[-1] == 4

    original_shape = list(q.shape)
    original_shape[-1] = 3
    q = q.view(-1, 4)

    q0 = q[:, 0] # w
    q1 = q[:, 1] # x
    q2 = q[:, 2] # y
    q3 = q[:, 3] # z

    if order == 'xyz':
        x = torch.atan2(2 * (q0 * q1 - q2 * q3), 1 - 2 * (q1 * q1 + q2 * q2))
        y = torch.asin(
            torch.clamp(2 * (q1 * q3 + q0 * q2), -1 + epsilon, 1 - epsilon))
        z = torch.atan2(2 * (q0 * q3 - q1 * q2), 1 - 2 * (q2 * q2 + q3 * q3))
    elif order == 'yzx':
        x = torch.atan2(2 * (q0 * q1 - q2 * q3), 1 - 2 * (q1 * q1 + q3 * q3))
        y = torch.atan2(2 * (q0 * q2 - q1 * q3), 1 - 2 * (q2 * q2 + q3 * q3))
        z = torch.asin(
            torch.clamp(2 * (q1 * q2 + q0 * q3), -1 + epsilon, 1 - epsilon))
    elif order == 'zxy':
        x = torch.asin(
            torch.clamp(2 * (q0 * q1 + q2 * q3), -1 + epsilon, 1 - epsilon))
        y = torch.atan2(2 * (q0 * q2 - q1 * q3), 1 - 2 * (q1 * q1 + q2 * q2))
        z = torch.atan2(2 * (q0 * q3 - q1 * q2), 1 - 2 * (q1 * q1 + q3 * q3))
    elif order == 'xzy':
        x = torch.atan2(2 * (q0 * q1 + q2 * q3), 1 - 2 * (q1 * q1 + q3 * q3))
        y = torch.atan2(2 * (q0 * q2 + q1 * q3), 1 - 2 * (q2 * q2 + q3 * q3))
        z = torch.asin(
            torch.clamp(2 * (q0 * q3 - q1 * q2), -1 + epsilon, 1 - epsilon))
    elif order == 'yxz':
        x = torch.asin(
            torch.clamp(2 * (q0 * q1 - q2 * q3), -1 + epsilon, 1 - epsilon))
        y = torch.atan2(2 * (q1 * q3 + q0 * q2), 1 - 2 * (q1 * q1 + q2 * q2))
        z = torch.atan2(2 * (q1 * q2 + q0 * q3), 1 - 2 * (q1 * q1 + q3 * q3))
    elif order == 'zyx':
        x = torch.atan2(2 * (q0 * q1 + q2 * q3), 1 - 2 * (q1 * q1 + q2 * q2))
        y = torch.asin(
            torch.clamp(2 * (q0 * q2 - q1 * q3), -1 + epsilon, 1 - epsilon))
        z = torch.atan2(2 * (q0 * q3 + q1 * q2), 1 - 2 * (q2 * q2 + q3 * q3))
    else:
        raise ValueError("Invalid order " + order)

    return torch.stack((x, y, z), dim=1).view(original_shape)





def denormalize_grasps(grasps, mean=0, std=1):
    temp = 1 / std
    for grasp in grasps:
        grasp[:3, 3] = (std * grasp[:3, 3] + mean)


def quat2mat(quat):
    """Convert quaternion coefficients to rotation matrix.
    Args:
        quat: first three coeff of quaternion of rotation. fourth is then computed to have a norm of 1 -- size = [B, 3]
    Returns:
        Rotation matrix corresponding to the quaternion -- size = [B, 3, 3]
    """
    norm_quat = torch.cat([quat[:, :1].detach() * 0 + 1, quat], dim=1)
    norm_quat = norm_quat / norm_quat.norm(p=2, dim=1, keepdim=True)
    w, x, y, z = norm_quat[:, 0], norm_quat[:, 1], norm_quat[:,
                                                             2], norm_quat[:,
                                                                           3]

    B = quat.size(0)

    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z

    rotMat = torch.stack([
        w2 + x2 - y2 - z2, 2 * xy - 2 * wz, 2 * wy + 2 * xz, 2 * wz + 2 * xy,
        w2 - x2 + y2 - z2, 2 * yz - 2 * wx, 2 * xz - 2 * wy, 2 * wx + 2 * yz,
        w2 - x2 - y2 + z2
    ],
                         dim=1).reshape(B, 3, 3)
    return rotMat


def qrot(q, v):
    """
    Rotate vector(s) v about the rotation described by quaternion(s) q.
    Expects a tensor of shape (*, 4) for q and a tensor of shape (*, 3) for v,
    where * denotes any number of dimensions.
    Returns a tensor of shape (*, 3).
    """
    assert q.shape[-1] == 4
    assert v.shape[-1] == 3
    assert q.shape[:-1] == v.shape[:-1]

    original_shape = list(v.shape)
    q = q.view(-1, 4)
    v = v.view(-1, 3)

    qvec = q[:, 1:]
    uv = torch.cross(qvec, v, dim=1)
    uuv = torch.cross(qvec, uv, dim=1)
    return (v + 2 * (q[:, :1] * uv + uuv)).view(original_shape)


def trans_2_quat_cpu(R):
    quat = torch.zeros(4)#.cuda()
    # quat = np.zeros(4)
    quat[3] = 0.5 * (torch.sqrt(1 + R[0][0] + R[1][1] + R[2][2]))  # qw,qx,qy,qz
    quat[0] = (R[2][1] - R[1][2]) / (4 * quat[3])
    quat[1] = (R[0][2] - R[2][0]) / (4 * quat[3])
    quat[2] = (R[1][0] - R[0][1]) / (4 * quat[3])
    rs = quat.clone()
    for i in range(4):
        rs[0] = quat[3]
        rs[1] = quat[0]
        rs[2] = quat[1]
        rs[3] = quat[2]
    return rs  # qw, qx, qy, qz

def q2R(quat):
    '''
    Args: quat: [x,y,z,w]
    Return: R_matrix
    '''
    quats = quat.clone()
    for i in range(4):
        quats[:,0] = quat[:,3]
        quats[:,1] = quat[:,0]
        quats[:,2] = quat[:,1]
        quats[:,3] = quat[:,2]

    bs, _ = quat.size()
    quat = quats # now : [w, x, y, z]
    quat = quat / (torch.norm(quat, dim=1).view(bs, 1))
    R_martix = torch.cat(((1.0 - 2.0*(quat[:, 2]**2 + quat[:, 3]**2)).view(bs, 1),\
            (2.0*quat[:, 1]*quat[:, 2] - 2.0*quat[:, 0]*quat[:, 3]).view(bs, 1), \
            (2.0*quat[:, 0]*quat[:, 2] + 2.0*quat[:, 1]*quat[:, 3]).view(bs, 1), \
            (2.0*quat[:, 1]*quat[:, 2] + 2.0*quat[:, 3]*quat[:, 0]).view(bs, 1), \
            (1.0 - 2.0*(quat[:, 1]**2 + quat[:, 3]**2)).view(bs, 1), \
            (-2.0*quat[:, 0]*quat[:, 1] + 2.0*quat[:, 2]*quat[:, 3]).view(bs, 1), \
            (-2.0*quat[:, 0]*quat[:, 2] + 2.0*quat[:, 1]*quat[:, 3]).view(bs, 1), \
            (2.0*quat[:, 0]*quat[:, 1] + 2.0*quat[:, 2]*quat[:, 3]).view(bs, 1), \
            (1.0 - 2.0*(quat[:, 1]**2 + quat[:, 2]**2)).view(bs, 1)), dim=1).contiguous().view(bs, 3, 3)
    return R_martix