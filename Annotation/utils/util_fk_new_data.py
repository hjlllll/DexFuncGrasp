import torch
#FK part
def trans_rtj_point(outputs_base, outputs_a):
    # 3 + 4
    outputs_a = torch.tensor(outputs_a, dtype=torch.float32).cuda()
    outputs_base = torch.tensor(outputs_base, dtype=torch.float32).cuda()

    # 17(18) -> 27(J)
    outputs_rotation = torch.zeros([outputs_a.shape[0], 27]).type_as(outputs_a)  # .cuda()
    #          
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
    # return outputs_rotation

def trans_rtj_point_train(outputs_base, outputs_a):
    # 3 + 4


    # 17(18) -> 27(J)
    outputs_rotation = torch.zeros([outputs_a.shape[0], 27]).type_as(outputs_a)  # .cuda()
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
    return outputs_rotation
