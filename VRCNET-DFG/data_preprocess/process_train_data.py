import os, argparse
import torch
import numpy as np
from tqdm import tqdm
from utils import *
import pickle
import json
import trimesh
'''
对于每个instance的30个部分点云，每个部分点云配对sample的指定数量的抓取并储存
'''

parser = argparse.ArgumentParser(description='render point cloud')
parser.add_argument('--category', type=str, default='mug')
parser.add_argument('--mode', type=str, default='train')
parser.add_argument('--render_num', type=int, default=30)
parser.add_argument('--vis', action='store_true', default=False)
parser.add_argument('--category_id', type=int, default=0)
args = parser.parse_args()


def init_R(a_list):
    a_list = torch.Tensor(a_list)
    rotation_anglex = a_list[0]
    cosvalx = torch.cos(rotation_anglex)
    sinvalx = torch.sin(rotation_anglex)
    rotation_angley = a_list[1]
    cosvaly = torch.cos(rotation_angley)
    sinvaly = torch.sin(rotation_angley)
    rotation_anglez = a_list[2]
    cosvalz = torch.cos(rotation_anglez)
    sinvalz = torch.sin(rotation_anglez)
    Rx = torch.Tensor([[1, 0, 0],
                   [0, cosvalx, -sinvalx],
                   [0, sinvalx, cosvalx]])
    Ry = torch.Tensor([[cosvaly, 0, sinvaly],
                   [0, 1, 0],
                   [-sinvaly, 0, cosvaly]])
    Rz = torch.Tensor([[cosvalz, -sinvalz, 0],
                   [sinvalz, cosvalz, 0],
                   [0, 0, 1]])
    rotation_matrix = torch.matmul(Rz, torch.matmul(Ry, Rx))
    return rotation_matrix


def main():
    vis = args.vis
    cate = args.category
    mode = args.mode
    # render_num = args.render_num
    f = open('/media/hjl/Samsung_T5/1dataset/obj_data.json', 'r')
    content = json.loads(f.read())
    category = content[str(args.category_id)]['category']
    functional_root = '/media/hjl/Samsung_T5/1dataset/functional_data_new'
    save_root = os.path.join('/media/hjl/Samsung_T5/1dataset/render_pc/{}'.format(cate), mode)
    # obj_root = os.path.join('obj/{}'.format(cate), mode)
    # inst_names = os.listdir(save_root)
    grasps_path = '/media/hjl/Samsung_T5/1dataset/Grasps'
    print('---Render {} {} models---'.format(cate, mode))
    # for i, inst_name in enumerate(inst_names):
    inst_name = category
    print(args.category_id, '  ', inst_name)
    grasps_sources = os.listdir(grasps_path)
    grasps_lists = []
    for grasps_source in grasps_sources:
        if grasps_source.split('_')[0] == str(args.category_id):
            grasps_lists.append(grasps_source)
    length = 0
    for i, grasps_list in enumerate(grasps_lists):
        grasps_path_intsance = os.path.join(grasps_path, grasps_list, 'sift')
       
        for ply_file in os.listdir(os.path.join(save_root, inst_name)):
            print(ply_file)
            if not os.path.exists(os.path.join(grasps_path_intsance, ply_file[:-4], 'new')):
                continue
            grasps_file_list = []
            grasps_file_list = getfiles(os.path.join(grasps_path_intsance, ply_file[:-4], 'new'), ['.xml'], grasps_file_list)
            grasp_size = len(grasps_file_list)
            print(grasp_size)
            for each_cam_points in tqdm(os.listdir(os.path.join(save_root, inst_name, ply_file))):
                if each_cam_points.endswith('npz'):
                    camera_pc_path = os.path.join(save_root, inst_name, ply_file, each_cam_points)
                    camera_corrds_info = np.load(camera_pc_path)
                    R = camera_corrds_info['rotation']
                    T = camera_corrds_info['translation']
                    point_camera_corrds = camera_corrds_info['pc']
                    if i == 1:
                        if not os.path.exists(os.path.join(functional_root, inst_name, ply_file, each_cam_points[:-4])):
                            length = 0
                        else:
                            length = len(os.listdir(os.path.join(functional_root, inst_name, ply_file, each_cam_points[:-4])))
                    for k, grasp_name in enumerate(grasps_file_list):
                        a, rs, t, file_name, rs_obj = reading_xml(grasp_name)
                        if file_name != ply_file[:-4]:
                            print(111111111111111)
                        hand_t = t.reshape(3).cuda() * 1000.0 #+ torch.from_numpy(T).cuda()*1000.0
                        hand_r = rs.reshape(4).cuda() #wxyz
                        hand_a = a.reshape(22).cuda()
                        cam_r = torch.from_numpy(R).cuda()
                        # cam_r0 = torch.eye(3, 3).cuda()
                        cam_rr = torch.from_numpy(R.T).cuda()
                        _, transformed_pts0 = trans_xml_pkl(hand_a.clone(), hand_t.clone(), hand_r.clone(), device='cuda')

                        hand_r_rxyzw = torch.zeros(4).cuda()
                        hand_r_rxyzw[:3] = hand_r.clone().reshape(4)[1:]
                        hand_r_rxyzw[3] = hand_r.clone().reshape(4)[0]
                        idx = 0
                        for item in content:
                            if content[item]['category'] == inst_name:
                                idx = int(item)
                                break
                        with open('/media/hjl/Samsung_T5/1dataset/category_angle.txt') as file:
                            category_a = file.read().splitlines()
                            x_a = float(category_a[idx].split(' ')[1]) * (np.pi / 2 / 1.57)
                            y_a = float(category_a[idx].split(' ')[2]) * (np.pi / 2 / 1.57)
                            z_a = float(category_a[idx].split(' ')[3]) * (np.pi / 2 / 1.57)
                            a_list = [x_a, y_a, z_a]
                        R_tran = init_R(a_list).cuda()
                        # new_t_ = torch.matmul(hand_t, R_tran.T).reshape(3)
                        new_t_ = torch.matmul(R_tran.T, hand_t).T.reshape(3)
                        R_hand = torch.matmul(R_tran.T, r_to_transform(hand_r_rxyzw))
                        input_rxyzw = trans_2_quat_gpu(R_hand, device='cuda')
                        input_rwxyz = torch.zeros(4).cuda()
                        input_rwxyz[1:] = input_rxyzw.reshape(4)[:3]
                        input_rwxyz[0] = input_rxyzw.reshape(4)[3]
                        new_j_p, transformed_pts00 = trans_xml_pkl(hand_a.clone(), new_t_.squeeze().clone(), input_rwxyz.clone(), device='cuda')

                        new_t = torch.matmul(new_t_.unsqueeze(0), cam_rr.clone()) + torch.from_numpy(T).cuda().unsqueeze(0)*1000.0
                        r = quat_mul_tensor(trans_2_quat_gpu(cam_r, device='cuda'), input_rxyzw.reshape(4)).reshape(4) #xyzw
                        input_rwxyz = torch.zeros(4).cuda()
                        input_rwxyz[1:] = r.clone().reshape(4)[:3]
                        input_rwxyz[0] = r.clone().reshape(4)[3]




                        new_j_p, transformed_pts = trans_xml_pkl(hand_a.clone(), new_t.squeeze().clone(), input_rwxyz.clone(), device='cuda')
                        with open(grasp_name[:-4]+'.pkl', 'rb') as filename:
                            points, graspparts, obj_name, j_p, _, hand_mesh_points = pickle.load(filename)
                            sa_r = os.path.join(functional_root, inst_name, ply_file, each_cam_points[:-4])
                            os.makedirs(sa_r, exist_ok=True)
                            camera_hand_mesh0 = transformed_pts00.clone().squeeze(0).detach().cpu().numpy() @ R.T + T*1000.0
                            camera_hand_mesh = transformed_pts.squeeze().detach().cpu().numpy()
                            camera_key_mesh = new_j_p.cpu().detach().numpy().squeeze()# @ R.T + T*1000.0
                            if i==1:
                                k += length

                            save_func_root = os.path.join(sa_r, 'Func_Grasp_{0}'.format(str(k).zfill(3)))
                            if os.path.exists(save_func_root+'.npz'):
                                print('{} already exists'.format(save_func_root))
                                continue
                            else:
                                if str(input_rwxyz.reshape(4)[0].item()) == 'nan':
                                    print('nan in data')
                                else:
                                    np.savez(save_func_root, cam_r=R, cam_t=T,
                                          hand_key=camera_key_mesh,
                                         hand_r=input_rwxyz.clone().cpu(), hand_t=new_t.squeeze().clone().cpu(), hand_a=a)

                            # visualize grasps and partical pointcloud in camera corrds
                            # camera_hand_mesh = transformed_pts.clone().squeeze(0).detach().cpu().numpy() @ R.T + T*1000.0
                            # camera_key_mesh = new_j_p.cpu().detach().numpy().squeeze() @ R.T + T*1000.0
                            
                            # obj = trimesh.PointCloud(point_camera_corrds, colors=[1, 95, 107, 255])
                            # hand = trimesh.PointCloud(camera_hand_mesh * 0.001, colors=[212, 106, 126, 255])
                            # hand_ori = trimesh.PointCloud(hand_mesh_points * 0.001, colors=[0, 255, 0, 255])
                            # hand0 = trimesh.PointCloud(camera_hand_mesh0 * 0.001, colors=[212, 106, 126, 255])
                            
                            # src_pc = np.dot(points, R_tran.cpu().numpy()) * 0.001
                            # src_pc = src_pc @ R.T + T #* 1000.0
                            # src = trimesh.PointCloud(src_pc, colors=[0, 0, 255, 255])
                            # obj_root = os.path.join('/media/hjl/Samsung_T5/1dataset/Obj_Data/')
                            # mesh = trimesh.load(os.path.join(obj_root, inst_name, 'obj', ply_file))
                            # trimesh.Scene([obj, src, hand, hand0, mesh, hand_ori]).show()


if __name__ == '__main__':
    main()