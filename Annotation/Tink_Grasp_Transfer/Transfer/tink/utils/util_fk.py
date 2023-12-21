import torch
#FK part
def trans_rtj_point(base, rotations):
    # 3 + 4
    rotations = torch.tensor(rotations, dtype=torch.float32).cuda()
    base = torch.tensor(base, dtype=torch.float32).cuda()
    outputs_base = base
    outputs_a = rotations

    # 17(18) -> 27(J)
    outputs_rotation = torch.zeros([outputs_a.shape[0], 27]).type_as(outputs_a)  # .cuda()
    # 20210706:因为graspit中shadowhand模型和运动学与真实手不一致，因此与预训练fk_cpu.py用的模型存在不同，
    #          目前有两种策略（详见onenote）：①网络预测指尖两关节和，两处模型不同，让他猜；②网络分别预测两关节，用loss进行约束。using①
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
    return outputs_base, outputs_rotation