# -*- coding: UTF-8 -*-
import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class Shadowhand_FK(object):
    """
    self.run()为主函数
    输入为base的姿态 (F, 7): x y z qw qx qy qz
         rotations (F, J): J个关节的旋转角度，弧度制
    输出：各个关节的空间坐标（包括base的），与base相同参考系

    使用示例：
    from FK_model import Shadowhand_FK
    fk = Shadowhand_FK()
    j_p = fk.run(base, rotations)

    关节值定义为26个，不包括基座，即 J=26
    0号坐标是基座，直接由网络预测，不参与计算
    """
    def __init__(self, npy_dir='/media/song/17e0f03b-07a2-440f-b0be-fb2d073c636c/song/a_2022_summer/dataset_generater/utils/'):
        self.npy_dir = npy_dir

    def transforms_multiply(self, t0s, t1s):
        return torch.matmul(t0s, t1s)

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
        base：（F, 7）—— x y z qw qx qy qz
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
        """
        角度输入暂定为弧度制，最好加上非线性处理，使角度值不超过限制
        rotations : (F, J) , Angle for each frame F and joint J
        M_r: 将角度转为绕z轴的旋转矩阵
        """
        m11 = torch.cos(rotations)  # (F, J)
        m12 = -torch.sin(rotations)  # (F, J)
        m21 = torch.sin(rotations)  # (F, J)
        m22 = torch.cos(rotations)  # (F, J)
        mr = torch.stack([torch.stack([m11, m21], -1), torch.stack([m12, m22], -1)], -1)  # (F, J, 2, 2)
        zeros = torch.zeros([int(rotations.shape[0]), int(rotations.shape[1]), 2, 2]).to(self.device)  # (F, J, 2, 2)
        eyes = torch.eye(2).expand(int(rotations.shape[0]), int(rotations.shape[1]), 2, 2).to(self.device)  # (F, J, 2, 2)
        M_r = torch.cat([torch.cat([mr, zeros], -2), torch.cat([zeros, eyes], -2)], -1)  # (F, J, 4, 4)

        return M_r   # [F,J,4,4]

    def transforms_local(self, M_sh, rotations):
        M_r = self.transforms_rotations(rotations).to(self.device)# [F,J,4,4]
        # print(M_r.shape)
        # print(M_sh.shape)
        M_sh = M_sh.expand(int(rotations.shape[0]), int(rotations.shape[1]), 4, 4).to(self.device)
        transforms = self.transforms_multiply(M_sh, M_r)  # [F,J,4,4]
        # print("transforms.shape:", transforms.shape)
        return transforms

    def transforms_global(self, base, parents, M_sh, rotations, ass_idx=(0,1,2,3)):
        locals = self.transforms_local(M_sh, rotations)  # 角度+预设生成旋转矩阵 [F,J,4,4]
        globals = self.transforms_blank(int(rotations.shape[0]), int(rotations.shape[1]))  # [F,J,4,4]
        base_M = self.transforms_base(base)   # [F,1,4,4]

        globals = torch.cat([base_M, globals], 1)  # 0号坐标是基座，直接由网络预测，不参与计算，但是需要给定值 # [F,1+J,4,4]
        globals = torch.split(globals, 1, 1)  # 因为torch.split输出是tuple型，后续无法迭代，所以需要变成list型
        globals = list(globals)  # list长度为J+1，每个元素[F, 1, 4, 4]
        # print(len(globals), locals.shape)

        # Chose ass key joints
        # 在正向运动学计算过程中，总共涉及到28个关节点
        # # ↓ index：28位list，对应着28个关键点，每位数字代表是否围绕该关节点构造辅助点组，-1则不需要，非-1数字表示使用辅助点库M_ass中的第几组，目前，M_ass共6组，每组4个，来自两种手指的三个link
        # # ↓ 一般只对每根手指的三个link构建辅助点[-1,    -1, -1, -1, 0, 1, 2,    -1, -1, 0,  1,  2,    -1, -1,  0,  1,  2,    -1, -1,  0,  1,  2,    -1, -1,  3, -1,  4,  5]
        index = [-1,    -1, -1, -1, 0, 1, 2,    -1, -1, 0,  1,  2,    -1, -1,  0,  1,  2,    -1, -1,  0,  1,  2,    -1, -1,  3, -1,  4,  5]
        # index = [-1,   -1, -1, -1, -1, -1, 2,   -1, -1, -1, -1, 2,   -1, -1, -1, -1, 2,   -1, -1, -1, -1, 2,   -1, -1, -1, -1, -1, 5]
        j_num = sum(i>-1 for i in index)  # # ← 统计需要围绕几个关键点构建辅助点组
        #ass_idx = [0,1,2,3]  # # ← 每组辅助点，组内选择的序号！！！
        ass_num = len(ass_idx)
        ass = self.transforms_blank(int(rotations.shape[0]), j_num*ass_num)  # 构建辅助点数组，点数A=j_num×ass_num，总尺寸[F, A, 4, 4]
        ass = torch.split(ass, 1, 1)  # 因为torch.split输出是tuple型，后续无法迭代，所以需要变成list型
        ass = list(ass)  # list长度为A，每个元素[F, 1, 4, 4]
        M_ass = np.load(self.npy_dir+'/M_ass.npy')  # [6,4,4,4] 加载辅助点信息
        M_ass = torch.from_numpy(M_ass).float().to(self.device)
        j_idx = -1

        # Calculate key points
        for i in range(1, len(parents)):  # 从1号而非0号开始, range(1,27)，因为0号是基准，已经有了
            # # ↓ 这里实质就是通过不断右乘新矩阵得到本关节相对初始坐标系的变换关系，恰好4×4矩阵最右上角三个数就是本关节在初始坐标系的坐标
            globals[i] = self.transforms_multiply(globals[parents[i]][:, 0], locals[:, i-1])[:, None, :, :]  # 一次右乘（A*M=B）：以A坐标系为基础，进行M的变换，得到B
            if index[i] != -1  and ass_num != 0:
                j_idx = j_idx + 1
                for ass_i in range(ass_num):
                    ass[ass_num*j_idx+ass_i] = self.transforms_multiply(globals[parents[i]][:, 0], M_ass[index[i], ass_idx[ass_i]])[:, None, :, :]

        globals = torch.cat(globals, 1)  # [F,1+J,4,4]
        ass = torch.cat(ass, 1)  # [F,A,4,4]
        globals_ass = torch.cat([globals, ass], 1)  # [F,1+J+A,4,4]

        return globals_ass   # [F,1+J+A,4,4]
    '''
    对比说明
                  plam           little                  ring                  middle                   index                     thumb
    key_points  = [ 0,     1,  2,  3, 4, 5, 6,     7,  8, 9, 10, 11,     12, 13, 14, 15, 16,     17, 18, 19, 20, 21,     22, 23, 24, 25, 26, 27]
    parents     = [-1,     0,  1,  2, 3, 4, 5,     0,  7, 8,  9, 10,      0, 12, 13, 14, 15,      0, 17, 18, 19, 20,      0, 22, 23, 24, 25, 26]
    index (ass) = [-1,    -1, -1, -1, 0, 1, 2,    -1, -1, 0,  1,  2,     -1, -1,  0,  1,  2,     -1, -1,  0,  1,  2,     -1, -1,  3, -1,  4,  5]
    '''
    def run(self, base, rotations, ass_idx=(0,1,2,3)):  # base:[F,1,3], rotations:[F,J,4]
        self.device = base.device
        # # ↓ 28个关键点所对应的运动链父结点
        parents = [-1, 0, 1, 2, 3, 4, 5, 0, 7, 8, 9, 10, 0, 12, 13, 14, 15, 0, 17, 18, 19, 20, 0, 22, 23, 24, 25, 26]   # 依据graspit!中shadowhand_simple顺序,图见OneNote-学习笔记-机器人学-坐标变换-shadowhand
        M_sh = np.load(self.npy_dir+'M_shadowhand.npy')  # 加载shadowhand的运动学关系
        M_sh = torch.from_numpy(M_sh).float()
        # print(M_sh.shape)
        positions = self.transforms_global(base, parents, M_sh, rotations, ass_idx)[:, :, :, 3]  # [F,1+J+A,1,4] --> [F,1+J+A,4]
        return positions[:, :, :3]  # positions[:, :, :3] / positions[:, :, 3, None]   # [F,1+J+A,3]

def show_data(points, graspparts, j_pp=None):
    print(points.shape)
    colpart = []
    for i in range(graspparts.shape[0]):
        hh = 0
        for cc in colpart:
            if not (graspparts[i]==cc).all():
                hh += 1
        if hh==len(colpart):
            colpart.append(graspparts[i])
    col = np.random.random([len(colpart), 3])
    print('len(col):', len(colpart))
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 1, 1, projection='3d')
    ax1.scatter([100, 0, 0], [0, 100, 0], [0, 0, 100], c=np.array([[1, 0, 0]]), marker='o', linewidths=1)
    # for i in range(points.shape[0]):
    #     for j, cc in enumerate(colpart):
    #         if (graspparts[i]==cc).all():
    #             col_idx = j
    #     ax1.scatter(points[i, 0], points[i, 1], points[i, 2], c=col[col_idx].reshape(1,-1), marker='.')  # [0.5, 0.5, 0.5]

    if j_pp is not None:
        Joints = j_pp.squeeze(0)[:28]
        ax1.scatter(Joints[:,0], Joints[:,1], Joints[:,2], c=np.array([[0, 0, 0]]), marker='x', linewidths=2)
        Ass = j_pp.squeeze(0)[28:]
        ax1.scatter(Ass[:,0], Ass[:,1], Ass[:,2], c=np.array([[0, 1, 0]]), marker='x', linewidths=2)
        lines = [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [0, 7], [7, 8], [8, 9], [9, 10], [10, 11], [0, 12], [12, 13], [13, 14], [14, 15], [15, 16],
                 [0, 17], [17, 18], [18, 19], [19, 20], [20, 21], [0, 22], [22, 23], [23, 24], [24, 25], [25, 26],[26,27]]
        for line in lines:
            x = [Joints[line[0]][0], Joints[line[1]][0]]
            y = [Joints[line[0]][1], Joints[line[1]][1]]
            z = [Joints[line[0]][2], Joints[line[1]][2]]
            ax1.plot(x, y, z,  color='r', linewidth=2)

    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.legend()
    plt.axis('off')
    plt.show()


if __name__ == '__main__':
    # fk = Shadowhand_FK()
    base = np.array([[0,0,0,1,0,0,0]])
    rotations = np.array([[0,0,0,1.5,
                            0,0,1.5,
                            0,0,0,
                            0,0,0,
                            0,0,0,0,0]])
    # 3 + 4
    rotations = torch.tensor(rotations, dtype=torch.float32).cuda()
    base = torch.tensor(base, dtype=torch.float32).cuda()
    outputs_base = base
    outputs_a = rotations

    # 17(18) -> 27(J)
    outputs_rotation = torch.zeros([outputs_a.shape[0], 27]).type_as(outputs_a)  # .cuda()
    # 20210706:因为graspit中shadowhand模型和运动学与真实手不一致，因此与预训练fk_cpu.py用的模型存在不同，
    #          目前有两种策略（详见onenote）：①网络预测指尖两关节和，两处模型不同，让他猜；②网络分别预测两关节，用loss进行约束。
    outputs_rotation[:, 0:3] = outputs_a[:, 0:3]
    angle_2_pair = torch.ones([2, outputs_a.shape[0]]).cuda()
    angle_1_pair = torch.zeros([2, outputs_a.shape[0]]).cuda()
    angle_2_pair[0] = outputs_a[:, 3]
    angle_1_pair[0] = outputs_a[:, 3] - 1
    outputs_rotation[:, 3] = torch.min(angle_2_pair, 0)[0]
    outputs_rotation[:, 4] = torch.max(angle_1_pair, 0)[0]
    outputs_rotation[:, 6:8] = outputs_a[:, 4:6]
    angle_2_pair[0] = outputs_a[:, 6]
    angle_1_pair[0] = outputs_a[:, 6] - 1
    outputs_rotation[:, 8] = torch.min(angle_2_pair, 0)[0]
    outputs_rotation[:, 9] = torch.max(angle_1_pair, 0)[0]
    outputs_rotation[:, 11:13] = outputs_a[:, 7:9]
    angle_2_pair[0] = outputs_a[:, 9]
    angle_1_pair[0] = outputs_a[:, 9] - 1
    outputs_rotation[:, 13] = torch.min(angle_2_pair, 0)[0]
    outputs_rotation[:, 14] = torch.max(angle_1_pair, 0)[0]
    outputs_rotation[:, 16:18] = outputs_a[:, 10:12]
    angle_2_pair[0] = outputs_a[:, 12]
    angle_1_pair[0] = outputs_a[:, 12] - 1
    outputs_rotation[:, 18] = torch.min(angle_2_pair, 0)[0]
    outputs_rotation[:, 19] = torch.max(angle_1_pair, 0)[0]
    outputs_rotation[:, 21:26] = outputs_a[:, 13:]  # all
    fk = Shadowhand_FK()
    j_p = fk.run(outputs_base, outputs_rotation * 1.5708)  # [F, J+10, 3]  #原始J+1个关键点，加上10个关键点
    # j_p = fk.run(base, rotations)
    # j_p = fk.run(base, rotations)
    print(j_p.shape)  # jp[6, 11, 16, 21, 26]分别是LF,RF,MF,FF,TH的顶点坐标
    # m = np.zeros((5,3))
    # m[0] = j_p[0][6]
    # m[1] = j_p[0][11]
    # m[2] = j_p[0][16]
    # m[3] = j_p[0][21]
    # m[4] = j_p[0][26]
    # m = m/100.0
    # print(m)
    base = base.cpu().detach().numpy()
    rotations = rotations.cpu().detach().numpy()
    j_p = j_p.cpu().detach().numpy()
    show_data(base, rotations, j_p)