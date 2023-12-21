'''
20230321
refinement after generation and point cloud completion

1 refine loss-away rtj
2 refine loss-away + loss-close rt
3 refine loss-away + loss-close + loss-norm rtj

'''
import numpy as np
import torch
from datasets.utils import *
from tqdm import tqdm
import os
from dataset_refine import *
from utils_isa import *
import trimesh
from write_xml_new_data import *
from utils_isa import *


class Modelopt:
    def __init__(self, on_gpu=True):

        # Choose to train on CPU or GPU
        if on_gpu and torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        return

    
    def grasping_opt(self, input_rta, points, file, trans_obj_i):
        input_r = input_rta[:, :4].to(self.device)
        input_t = input_rta[:, 4:7].to(self.device)*1000.0
        input_a = (input_rta[:, 7:]/1.5708).to(self.device)
        input_a[:, [2, 3, 5, 6, 8, 9, 11, 12]] /=1.15

        input_a.requires_grad = True
        optimizer0 = torch.optim.Adam([input_a], lr=5e-3)
     
        opt_sign = 'abc'
      
        input_r_initial = input_rta[:,0:4] 
        input_t_initial = input_rta[:,4:4 + 3] * 1000.0  
        input_a_initial = input_rta[:,4 + 3:4 + 3 + 18 ] / 1.5708  
        _, transformed_pts_initial = trans_xml_pkl_batch(input_a_initial.clone().detach(), input_t_initial.clone().detach(), input_r_initial.clone().detach() , device='cuda')
        for step in tqdm(range(200)):
            points = points.to(self.device)

            _,transformed_pts = trans_xml_pkl_batch(input_a, input_t, input_r, device='cuda')
            id_list = torch.load('hand_point_idx.pth')
            all = np.arange(0,2000)
            hand_index = np.concatenate(
                (id_list['thumb'], id_list['firfinger'], id_list['midfinger'], id_list['ringfinger'], id_list['litfinger']),
                axis=0)
            away_index = np.setdiff1d(all, hand_index)
           
            transformed_pts *= 0.001
            away_points = transformed_pts[:, away_index, :]# b, n2, 3
            away_points = index_points(away_points, farthest_point_sample_tensor(away_points, 300))
            close_points = torch.cat([transformed_pts[:, id_list['thumb'],:].unsqueeze(1), transformed_pts[:, id_list['firfinger'],:].unsqueeze(1)
                                        , transformed_pts[:, id_list['midfinger'],:].unsqueeze(1), transformed_pts[:, id_list['ringfinger'],:].unsqueeze(1),
                                    transformed_pts[:, id_list['litfinger'],:].unsqueeze(1)], dim=1)
            close_distance = ((close_points.unsqueeze(3).expand(-1, -1, -1, points.shape[1], -1)*1000 -
                            points.unsqueeze(1).expand(-1, close_points.shape[1], -1, -1).
                            unsqueeze(2).expand(-1, -1, close_points.shape[2], -1, -1)*1000)**2).sum(-1).sqrt()   
            close_distances = torch.min(close_distance, -1)[0] # [b, 5, 22]
            close_distances = torch.min(close_distances, -1)[0].cuda() 
            # [b, 2500, 300]
            away_distance = ((away_points.unsqueeze(1).repeat(1, points.shape[1], 1, 1) - points.unsqueeze(2).
                            repeat(1, 1, away_points.shape[1],1)) ** 2).sum(-1).sqrt()
            away_distances = torch.log2(0.002/ (torch.min(away_distance, -2)[0] + 0.000001))  # [b, 300]
            away_distances[away_distances <= 0] = 0
            close0 = close_distances[:,0].sum() / close_distances.shape[0]
            close1 = close_distances[:,1].sum() / close_distances.shape[0]
            close2 = close_distances[:,2].sum() / close_distances.shape[0]
            close3 = close_distances[:,3].sum() / close_distances.shape[0]
            close4 = close_distances[:,4].sum() / close_distances.shape[0]
            loss_away = away_distances.sum()/away_distances.shape[0]  
            loss_norm = ((input_a_initial.clone().detach()[:, [1, 4, 7, 10, 13]].to(self.device) - input_a[:, [1, 4, 7, 10, 13]])**2).sum()
            loss = 5*close0 + 5*close1 + 5*close2 + 2.5 * close3 + 20*loss_away + 100*loss_norm

            optimizer = optimizer0
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % 10 == 0:
                print('step {}--{}: f(x)={:.4f}, thumb={:.4f}, little={:.4f}, loss_away={:.4f} , loss_norm={:.4f} '
                      .format(step, optimizer.defaults['lr'], loss.item(), close0.item(), close4.item(), loss_away.item(), loss_norm.item()))

            if loss < 0.001:
                break

        print('finial : step {}: f(x)={}'.format(step, loss.item()))

        points = points.to(self.device)
        _, transformed_pts = trans_xml_pkl_batch(input_a, input_t, input_r, device='cuda')
        id_list = torch.load('hand_point_idx.pth')
        all = np.arange(0, 2000)
        hand_index = np.concatenate(
            (id_list['thumb'], id_list['firfinger'], id_list['midfinger'], id_list['ringfinger'], id_list['litfinger']),
            axis=0)
        away_index = np.setdiff1d(all, hand_index)
        for i in range(points.shape[0]):

            # obj = trimesh.PointCloud(points[i].cpu().numpy(), colors=[212, 106, 126, 255])
            # hand0 = trimesh.PointCloud(transformed_pts_initial[i].detach().cpu().numpy()*0.001, colors=[35, 25, 226, 255])
            # hand = trimesh.PointCloud(transformed_pts[i].detach().cpu().numpy()*0.001, colors=[1, 95, 107, 255])
            # trimesh.Scene([obj, hand, hand0]).show()

            #######saving results########
            save_xml_name = file[i][:-4] + '.xml'
            category = save_xml_name.split('/')[-3]
            obj_name = save_xml_name.split('/')[-4] + '.xml'
            # print(save_xml_name)
            xyzw = input_r[i].clone().detach().cpu().numpy()#.reshape(4)
            input_rwxyz = np.zeros(4)
            input_rwxyz[1:] = xyzw.reshape(4)[:3]
            input_rwxyz[0] = xyzw.reshape(4)[3]
            input_rwxyzo = np.zeros(4)
            input_rwxyzo[1:] = trans_obj_i[i].clone().detach().cpu().numpy().reshape(4)[:3]
            input_rwxyzo[0] = trans_obj_i[i].clone().detach().cpu().numpy().reshape(4)[3]
            write_xml_new_data(category=category, obj_name=obj_name, r=xyzw,
                               r_o=input_rwxyzo, t=input_t[i].clone().detach().cpu().numpy(),
                               a=input_a[i].clone().detach().cpu().numpy() * 1.5708,
                               path=save_xml_name,
                               mode='train', rs=(21, 'real'))




if __name__ == '__main__':
    import os
    import shutil

    src = "test_result_sim" # 源文件目录
    det = "test_result_sim_refine" # 目的文件目录\
    for sr in os.listdir(src):
        de = os.path.join(det, sr)
        srs = os.path.join(src, sr)
        for s in os.listdir(os.path.join(srs)):
            d = os.path.join(de, s)
            if not os.path.exists(d):
                os.makedirs(d)                    
            ss = os.path.join(srs, s)
            for root, _, fnames in os.walk(ss):
                for fname in sorted(fnames):  # sorted函数把遍历的文件按文件名排序
                    fpath = os.path.join(root, fname)
                    shutil.copy(fpath, d)  # 完成文件拷贝
                    print(fname + " has been copied!")

    ###########refine form test_result_sim to test_result_sim_complete#############
    points_num = 2048
    batchsize = 64
    workers = 4
    dataset_test = Refinedataset(train=False, npoints=points_num)
    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=batchsize, shuffle=True,
                                                  num_workers=int(workers))
    dataset_length = len(dataset_test)
    for i, data in enumerate(dataloader_test):
        input_rta, pc, file, trans_obj_i = data


        opt = Modelopt()
        index = [0, 1, 2, 3, 4, 5, 6, 0 + 7, 1 + 7, 2 + 7, 3 + 7, 5 + 7, 6 + 7, 7 + 7, 9 + 7, 10 + 7, 11 + 7, 13 + 7, 14 + 7, 15 + 7, 17 + 7, 18 + 7, 19 + 7, 20 + 7, 21 + 7]
        input_rta[:, [3+ 7, 7+ 7, 11+ 7, 15+ 7]] += input_rta[:, [4+ 7, 8+ 7, 12+ 7, 16+ 7]]
        input_rta = input_rta[:, index]
        opt.grasping_opt(input_rta, pc, file, trans_obj_i)