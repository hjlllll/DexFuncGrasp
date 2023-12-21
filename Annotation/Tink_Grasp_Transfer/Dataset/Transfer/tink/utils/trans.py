import numpy as np
import torch
import math
#旋转矩阵转四元数需要pyquaternion包
from pyquaternion import Quaternion
#四元数转旋转矩阵需要scipy
from scipy.spatial.transform import Rotation as R
import open3d as o3d
# from tink.FK_model_opt import fk_run
from FK_model_opt import fk_run

from utils.FK_layer_mesh_physic import FK_layer
def trans_2_quat_cpu(R):
    # quat = torch.zeros(4)#.cuda()
    quat = np.zeros(4)
    quat[3] = 0.5 * (np.sqrt(1 + R[0][0] + R[1][1] + R[2][2]))
    quat[0] = (R[2][1] - R[1][2]) / (4 * quat[3])
    quat[1] = (R[0][2] - R[2][0]) / (4 * quat[3])
    quat[2] = (R[1][0] - R[0][1]) / (4 * quat[3])
    return quat #qx, qy, qz, qw


def quat_mul_tensor(a, b):
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

    quat = torch.stack([x, y, z, w], dim=-1).view(shape)

    return quat


def trans_2_quat_gpu(R):
    # quat = torch.zeros(4)#.cuda()
    quat = torch.zeros(4)#.cuda()
    quat[3] = 0.5 * (torch.sqrt(1 + R[0][0] + R[1][1] + R[2][2]))
    quat[0] = (R[2][1] - R[1][2]) / (4 * quat[3])
    quat[1] = (R[0][2] - R[2][0]) / (4 * quat[3])
    quat[2] = (R[1][0] - R[0][1]) / (4 * quat[3])
    return quat #qx, qy, qz, qw

def trans_2_quat_cpu(R):
    # quat = torch.zeros(4)#.cuda()
    quat = np.zeros(4)
    quat[3] = 0.5 * (np.sqrt(1 + R[0][0] + R[1][1] + R[2][2]))
    quat[0] = (R[2][1] - R[1][2]) / (4 * quat[3])
    quat[1] = (R[0][2] - R[2][0]) / (4 * quat[3])
    quat[2] = (R[1][0] - R[0][1]) / (4 * quat[3])
    return quat #qx, qy, qz, qw

def isRotationMatrix(R):
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype=R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6
# rotationMatrixToEulerAngles 用于旋转矩阵转欧拉角
def rotationMatrixToEulerAngles(R):
    assert (isRotationMatrix(R))
    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6
    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0
    return np.array([x, y, z])


# eulerAnglesToRotationMatrix欧拉角转旋转矩阵
def eulerAnglesToRotationMatrix(theta):
    R_x = np.array([[1, 0, 0],
                    [0, math.cos(theta[0]), -math.sin(theta[0])],
                    [0, math.sin(theta[0]), math.cos(theta[0])]
                    ])
    R_y = np.array([[math.cos(theta[1]), 0, math.sin(theta[1])],
                    [0, 1, 0],
                    [-math.sin(theta[1]), 0, math.cos(theta[1])]
                    ])

    R_z = np.array([[math.cos(theta[2]), -math.sin(theta[2]), 0],
                    [math.sin(theta[2]), math.cos(theta[2]), 0],
                    [0, 0, 1]
                    ])
    R = np.dot(R_z, np.dot(R_y, R_x))
    return R

#旋转矩阵转四元数
def rotateToQuaternion(rotateMatrix):
    q = Quaternion(matrix=rotateMatrix)
    return q

# rotationMat_1 = eulerAnglesToRotationMatrix([1.57,0,0])
# print ("\nR1 :\n{0}".format(rotationMat_1))
# a = np.random.randn(200,3)
# print(np.dot(a,rotationMat_1).shape)
def quat_mul(a, b):
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

    quat = np.stack([x, y, z, w], axis=-1)#.view(shape)

    return quat


def transforms_base(base):
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

def get_4x4_matrix_cuda(quat, pos):
    # if isinstance(quat, list):
    #     quat = np.array(quat, np.float64).cpu()
    quat = quat.detach().cpu().numpy()
    quat = quat[[3, 0, 1, 2]]
    if isinstance(pos, list):
        pos = np.array(pos, np.float64)
    pos = pos.clone().detach().cpu().numpy()
    t = np.eye(4).astype(np.float64)
    t[:3, 3] = pos
    t[:3, :3] = o3d.geometry.get_rotation_matrix_from_quaternion(quat)
    return t

def get_4x4_matrix_tensor(quat, pos):
    # if isinstance(quat, list):
    #     quat = np.array(quat, np.float64).cpu()
    # quat = quat.detach().cpu().numpy()
    # quat = quat[[3, 0, 1, 2]]
    # if isinstance(pos, list):
    #     pos = np.array(pos, np.float64)
    # pos = pos.clone().detach().cpu().numpy()
    t = torch.eye(4).expand(1,1,-1,-1)
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
    # R = torch.stack([dim0, dim1, dim2], -2).cuda()  # [F,1,3,3]
    t[:, :, :3, 3] = pos.expand(1, -1, -1)
    t[:, :, :3, :3] = torch.stack([dim0, dim1, dim2], -2).cuda()  # [F,1,3,3]
    return t


def get_4x4_matrix_tensor_batch(quat, pos):
    # if isinstance(quat, list):
    #     quat = np.array(quat, np.float64).cpu()
    # quat = quat.detach().cpu().numpy()
    # quat = quat[[3, 0, 1, 2]]
    # if isinstance(pos, list):
    #     pos = np.array(pos, np.float64)
    # pos = pos.clone().detach().cpu().numpy()
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
    # R = torch.stack([dim0, dim1, dim2], -2).cuda()  # [F,1,3,3]
    t[:, :, :3, 3] = pos.unsqueeze(1)#.expand(-1, 1, -1)#.transpose(1,0)
    t[:, :, :3, :3] = torch.stack([dim0, dim1, dim2], -2).unsqueeze(1).cuda()#.expand(-1, 1, -1, -1).cuda()#.transpose(1,0).cuda()  # [F,1,3,3]
    return t

def trans18_to_22(input_a):
    joints = torch.zeros((1, 22)).cuda()
    index = [0, 1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19, 20, 21]
    joints[:, index] = input_a.cuda().clone()  # * 1.5708
    indexs = [3, 7, 11, 15]
    joints[:, indexs] = 0.0
    angle_2_pair = torch.ones([2, 1]).cuda()
    angle_1_pair = torch.zeros([2, 1]).cuda()
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


def trans_xml_pkl(input_a,input_t,input_r):
    view_mat_mesh = torch.tensor([[0.0, 0.0, 1.0],
                                  [-1.0, 0, 0.0],
                                  [0.0, -1.0, 0]])
    # a = trans_2_quat_gpu(view_mat_mesh)
    # input_rxyzw = torch.zeros(4)#.cuda()
    # input_rxyzw[:3] = input_r.reshape(4)[1:]
    # input_rxyzw[3] = input_r.reshape(4)[0]
    mesh_r = torch.zeros(4)
    mesh_r[:3] = input_r.clone().reshape(4)[1:]
    mesh_r[3] = input_r.clone().reshape(4)[0]


    r = quat_mul_tensor(mesh_r.reshape(4), trans_2_quat_gpu(view_mat_mesh)).reshape(4)
    # joints = torch.zeros((1,22)).cuda()
    # index = [0, 1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19, 20, 21]
    # base_M = torch.from_numpy(get_4x4_matrix_cuda(r, input_t)).float().reshape(1, 1, 4, 4).cuda()

    base_M = get_4x4_matrix_tensor(r, input_t).expand(1, 1, -1, -1).cuda()#.reshape(1, 1, 4, 4).cuda()

    # angles = input_a.reshape(1, -1).cuda() # * 1.5708

    joints = trans18_to_22(input_a)

    angles = joints.reshape(1, -1).cuda() * 1.5708
    fk_angles = torch.zeros((1, 22)).cuda()
    fk_angles[:, :4] = angles.clone()[:, 13:17]
    fk_angles[:, 4:8] = angles.clone()[:, 9:13]
    fk_angles[:, 8:12] = angles.clone()[:, 5:9]
    fk_angles[:, 12:17] = angles.clone()[:, :5]
    fk_angles[:, 17:] = angles.clone()[:, 17:]
    fk_angles[:, [0, 4]] = -fk_angles[:, [0, 4]]  ########mention######
    fk_angles[:, -2] = -fk_angles[:, -2]
    fk_angles[:, -1] = -fk_angles[:, -1]
    # fk_angles = fk_angles.reshape(1, -1)
    fk_layers = FK_layer()
    # fk_layers.cuda()
    positions, transformed_pts = fk_layers.run(base_M, fk_angles)
    # inputa = torch.zeros(1,18).cuda()
    # input_a = input_a.reshape(1,-1)
    # input_a[:, [3, 7, 11, 15]] += inputa[:, [4, 8, 12, 16]]
    # inputa = input_a[:, index]
    outputs_F = fk_run(input_r, input_t, input_a)[:, :28]  # * 0.001

    return outputs_F, transformed_pts, joints.squeeze()#, input_rwxyz.reshape(1,-1)

def trans_xml_pkl_batch(input_a,input_t,input_r):
    view_mat_mesh = torch.tensor([[0.0, 0.0, 1.0],
                                  [-1.0, 0, 0.0],
                                  [0.0, -1.0, 0]])
    # a = trans_2_quat_gpu(view_mat_mesh)
    # input_rxyzw = torch.zeros(4)#.cuda()
    # input_rxyzw[:3] = input_r.reshape(4)[1:]
    # input_rxyzw[3] = input_r.reshape(4)[0]
    mesh_r = torch.zeros(input_a.shape[0],4)
    mesh_r[:, :3] = input_r.clone()[:, 1:]
    mesh_r[:, 3] = input_r.clone()[:, 0]

    r = quat_mul_tensor(mesh_r, trans_2_quat_gpu(view_mat_mesh).reshape(1,4).repeat(input_a.shape[0],1))#.reshape(4)

    # joints = torch.zeros((1,22)).cuda()
    # index = [0, 1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19, 20, 21]
    # base_M = torch.from_numpy(get_4x4_matrix_cuda(r[0], input_t[0])).float().reshape(1, 1, 4, 4).repeat(input_a.shape[0],1,1,1).cuda()

    base_M = get_4x4_matrix_tensor_batch(r, input_t).reshape(r.shape[0], 1, 4, 4).cuda()
    # angles = input_a.reshape(1, -1).cuda() # * 1.5708

    joints = trans18_to_22_batch(input_a)

    angles = joints.cuda() * 1.5708
    fk_angles = torch.zeros((r.shape[0], 22)).cuda()
    fk_angles[:, :4] = angles.clone()[:, 13:17]
    fk_angles[:, 4:8] = angles.clone()[:, 9:13]
    fk_angles[:, 8:12] = angles.clone()[:, 5:9]
    fk_angles[:, 12:17] = angles.clone()[:, :5]
    fk_angles[:, 17:] = angles.clone()[:, 17:]
    fk_angles[:, [0, 4]] = -fk_angles[:, [0, 4]]  ########mention######
    fk_angles[:, -2] = -fk_angles[:, -2]
    fk_angles[:, -1] = -fk_angles[:, -1]
    # fk_angles = fk_angles.reshape(1, -1)
    fk_layers = FK_layer()
    # fk_layers.cuda()
    positions, transformed_pts = fk_layers.run(base_M, fk_angles,F=fk_angles.shape[0])
    # inputa = torch.zeros(1,18).cuda()
    # input_a = input_a.reshape(1,-1)
    # input_a[:, [3, 7, 11, 15]] += inputa[:, [4, 8, 12, 16]]
    # inputa = input_a[:, index]
    outputs_F = fk_run(input_r, input_t, input_a)[:, :28]  # * 0.001

    return outputs_F, transformed_pts, joints.squeeze()#, input_rwxyz.reshape(1,-1)


