import numpy as np
import torch
import math
#旋转矩阵转四元数需要pyquaternion包
from pyquaternion import Quaternion
#四元数转旋转矩阵需要scipy
from scipy.spatial.transform import Rotation as R
import open3d as o3d

def quat_conjugate(a):
    shape = a.shape
    a = a.reshape(-1, 4)
    return torch.cat((-a[:, :3], a[:, -1:]), dim=-1).view(shape)

def trans_2_quat_cpu(R):
    # quat = torch.zeros(4)#.cuda()
    quat = np.zeros(4)
    quat[3] = 0.5 * (np.sqrt(1 + R[0][0] + R[1][1] + R[2][2]))
    quat[0] = (R[2][1] - R[1][2]) / (4 * quat[3])
    quat[1] = (R[0][2] - R[2][0]) / (4 * quat[3])
    quat[2] = (R[1][0] - R[0][1]) / (4 * quat[3])
    return quat #qx, qy, qz, qw

def trans_2_quat_gpu(R,device):
    # quat = torch.zeros(4)#.cuda()
    quat = torch.zeros(4).to(device)#.cuda()
    quat[3] = 0.5 * (torch.sqrt(1 + R[0][0] + R[1][1] + R[2][2]))
    quat[0] = (R[2][1] - R[1][2]) / (4 * quat[3])
    quat[1] = (R[0][2] - R[2][0]) / (4 * quat[3])
    quat[2] = (R[1][0] - R[0][1]) / (4 * quat[3])
    return quat #qx, qy, qz, qw

def trans_2_quat_gpu_batch(R, device):
    # quat = torch.zeros(4)#.cuda()
    quat = torch.zeros((R.shape[0],4)).to(device)#.cuda()
    quat[:, 3] = 0.5 * (torch.sqrt(1 + R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]))
    quat[:, 0] = (R[:, 2, 1] - R[:, 1, 2]) / (4 * quat[:, 3])
    quat[:, 1] = (R[:, 0, 2] - R[:, 2, 0]) / (4 * quat[:, 3])
    quat[:, 2] = (R[:, 1, 0] - R[:, 0, 1]) / (4 * quat[:, 3])
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

def quat_apply(a, b):
    shape = b.shape
    a = a.reshape(-1, 4)
    b = b.reshape(-1, 3)
    xyz = a[:, :3]
    t = xyz.cross(b, dim=-1) * 2
    return (b + a[:, 3:] * t + xyz.cross(t, dim=-1)).view(shape)

def get_3x3_matrix_cuda(quat, pos):
    # if isinstance(quat, list):
    #     quat = np.array(quat, np.float64).cpu()
    quat = quat.detach().cpu().numpy()
    quat = quat[[3, 0, 1, 2]]
    if isinstance(pos, list):
        pos = np.array(pos, np.float64)
    # pos = pos.clone().detach().cpu().numpy()
    t = np.eye(3).astype(np.float64)
    # t[:3, 3] = pos
    t[:3, :3] = o3d.geometry.get_rotation_matrix_from_quaternion(quat)
    return t


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


