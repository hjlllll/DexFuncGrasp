# ----------------------------------------------
# Written by Kailin Li (kailinli@sjtu.edu.cn)
# ----------------------------------------------
import glob
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import torch
from torch.nn import parameter
import trimesh
# from manotorch.manolayer import ManoLayer
# from manotorch.utils.anchorutils import anchor_load, get_region_palm_mask, masking_load_driver
# from manotorch.utils.quatutils import angle_axis_to_quaternion, quaternion_to_angle_axis
from termcolor import colored, cprint
from tqdm import trange

from cal_contact_info import to_pointcloud
from hand_optimizer import GeOptimizer, init_runtime_viz
from info_transform import get_obj_path
from sdf_loss import load_obj_latent, load_sdf_decoder
from util_shadow import ShadowGripper
from utils.read_xml import reading_xml
import json
from utils.trans import *

from utils.write_xml import write_xml_new_data, save_pcd_pkl

def scatter_array(ax, array, c=None, cmap=None):
    ax.scatter3D(array[:, 0], array[:, 1], array[:, 2], c=c, cmap=cmap)
    return ax


if __name__ == "__main__":
    import argparse
    from scipy.io import loadmat

    parser = argparse.ArgumentParser(description="Eval Hand Manipulation")


    parser.add_argument("--resume", type=str, default="latest")

    parser.add_argument("--lr", default=5e-4, type=float)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--iters", default=400, type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--center_obj", action="store_false")

    parser.add_argument("--fix_tsl", action="store_true")

    parser.add_argument("--vis", action="store_true")

    parser.add_argument("--all", action="store_true")
    parser.add_argument('--idx', type=int, default=0, help="0-20 id of hjl new dataset, means category")
    arg = parser.parse_args()
    sdf_samples_subdir = "sdf/free_space_pts"
    data_root = '../Dataset/Obj_Data'
    grasp_root = '../Dataset/Grasps'
    f = open('../Dataset/Obj_Data/obj_data.json', 'r')
    content = json.loads(f.read())
    category = content[str(arg.idx)]['category']
    data = os.path.join(data_root, category, 'split')
    split_inter = os.path.join(data, "split_interpolate.json")
    handle_list = []
    device = torch.device("cuda")
    if arg.all:
        split = json.load(open(split_inter, "r"))
        for cid in split.keys():
            real_ids = split[cid]["real"]
            virtual_ids = split[cid]["virtual"]
            for v_oid in virtual_ids:
                for r_oid in real_ids:
                    handle_list.append((v_oid, r_oid))
    for idx, (target_instance, soruce_idx) in enumerate(handle_list):
        cprint(f"pose refinement: {soruce_idx} -> {target_instance}", "blue")
        info_transferred_path = os.path.join(data_root, category, 'split', 'contact')
        info_transferred_list = os.listdir(os.path.join(info_transferred_path, target_instance))
        vertex_contacts,contact_regions,anchor_ids,ts,input_rwxyz0s, angles = torch.zeros(1,1),0,0,0,0,0
        batch = 0
        mesh = trimesh.load(get_obj_path(target_instance, split_inter, str(arg.idx), data_root, category), process=False, force="mesh", skip_materials=True)
        obj_pc = to_pointcloud(mesh)
        obj_verts_np, obj_normals_np = np.asarray(obj_pc.points) * 1000.0, np.asarray(obj_pc.normals)
        obj_verts, obj_normals = torch.tensor(obj_verts_np, dtype=torch.float32).to(device), torch.tensor(obj_normals_np).to(device)

        for info_transferred_instance in info_transferred_list:
            if info_transferred_instance[:-(int(len(info_transferred_instance.split('_')[-1])))-1] != soruce_idx:
                continue
            if not os.path.exists(os.path.join(info_transferred_path, target_instance, info_transferred_instance, 'contact_info.pkl')):
                continue
            contact_info = pickle.load(open(os.path.join(info_transferred_path, target_instance, info_transferred_instance, 'contact_info.pkl'), "rb"))
            len_num = len(info_transferred_instance.split('_')[-1])
            transfered_path = os.path.join(grasp_root, str(arg.idx)+'_'+info_transferred_instance[:-len_num-1], 'transferred')
            trans_contact_path = os.path.join(transfered_path, target_instance)
            os.makedirs(trans_contact_path, exist_ok=True)
            # mesh = trimesh.load(get_obj_path(target_instance, split_inter, str(arg.idx), data_root, category), process=False, force="mesh", skip_materials=True)
            # obj_pc = to_pointcloud(mesh)
            # obj_verts_np, obj_normals_np = np.asarray(obj_pc.points) * 1000.0, np.asarray(obj_pc.normals)
            # obj_verts, obj_normals = torch.tensor(obj_verts_np, dtype=torch.float32).to(device), torch.tensor(obj_normals_np).to(device)
            vertex_contact = torch.from_numpy(contact_info["vertex_contact"]).long().to(device).unsqueeze(0)
            contact_region = torch.from_numpy(contact_info["hand_region"]).long().to(device).unsqueeze(0) #[5000]
            anchor_id = torch.from_numpy(contact_info["anchor_id"]).long().to(device).unsqueeze(0)
            z = torch.zeros(anchor_id.shape[0]).long().cuda()
            if anchor_id.equal(z):
                print('jump this instance')
                continue
            batch += 1

            # read source grasps
            each_grasp = '{}_{}.xml'.format(str(arg.idx), info_transferred_instance.split('_')[-1])
            each_grasp_filename = os.path.join(os.path.join(grasp_root, str(arg.idx) + '_' + soruce_idx, 'new'), each_grasp)
            a, rs, t, file_name, rs_obj = reading_xml(each_grasp_filename)  # x y z w
            t *= 0.001  # 为了保持rta三个offset在同一数量级
            input_rwxyz0 = np.zeros(4)
            input_rwxyz0[1:] = rs.clone().detach().cpu().numpy().reshape(4)[:3]
            input_rwxyz0[0] = rs.clone().detach().cpu().numpy().reshape(4)[3]
            input_rwxyz0 = torch.from_numpy(input_rwxyz0).float().reshape(4)
            index = [0, 1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19, 20, 21]
            a[:, [3, 7, 11, 15]] += a[:, [4, 8, 12, 16]]
            angle = a[:, index].clone()

            if batch == 1:
                vertex_contacts = vertex_contact
                contact_regions = contact_region
                anchor_ids = anchor_id
                ts = t.unsqueeze(0)
                angles = angle
                input_rwxyz0s = input_rwxyz0.unsqueeze(0)
            else:
                vertex_contacts = torch.cat((vertex_contacts, vertex_contact), dim=0)
                contact_regions = torch.cat((contact_regions, contact_region), dim=0)
                anchor_ids = torch.cat((anchor_ids, anchor_id), dim=0)
                ts = torch.cat((ts, t.unsqueeze(0)), dim=0)
                angles = torch.cat((angles, angle), dim=0)
                input_rwxyz0s = torch.cat((input_rwxyz0s, input_rwxyz0.unsqueeze(0)), dim=0)
        obj_verts = obj_verts.unsqueeze(0).repeat(vertex_contacts.shape[0],1,1)
        obj_normals = obj_normals.unsqueeze(0).repeat(vertex_contacts.shape[0],1,1)

        if vertex_contacts.shape[0] == 1:
            continue
        # print(vertex_contacts.shape)
        # print(ts.shape,angles.shape,input_rwxyz0s.shape)


        batch_ = ts.shape[0]
        per_bat = 32
        for j in range((batch_//per_bat)+1):
            begin = j * per_bat
            per_batch = (j+1) * per_bat

            opt1 = GeOptimizer(
                device=device,
                info=[ts[begin:per_batch], input_rwxyz0s[begin:per_batch], angles[begin:per_batch]],  # 3 + 4 + 18 = 25
                lr=arg.lr,
                n_iter=arg.iters,
                verbose=False,
                lambda_contact_loss=10.0,  # 单位mm
                lambda_repulsion_loss=0,  # 单位m/100
            )
            # mano_out = opt.mano_layer(hand_pose[None], hand_shape[None])
            # shadow_out = ShadowGripper(root_folder='./tink/', data=joint, filename='')
            # shadow_mesh = shadow_out.get_meshes()

            shadow_mesh = opt1.shadow_mesh
            hand_point = np.zeros((27, 3))
            hand_mesh_point = np.zeros((2000, 3))
            center = np.zeros(3, dtype=np.float32)
            path = os.path.join(data.split('/split')[0], sdf_samples_subdir, target_instance + '.obj.mat')
            sdf_scale = loadmat(path)
            rescale = sdf_scale['scale']  # scale是一个小于1的数， 是物体放大的系数, mesh_to_sdf 相当于将物体放入一个单位圆或者正方体,所以物体是放大了的
            if arg.vis:
                runtime_viz = init_runtime_viz(
                    # hand_verts=mano_out.verts.cpu().squeeze().numpy(),
                    # hand_faces=opt.mano_layer.th_faces.cpu().numpy(),
                    hand_verts=shadow_mesh.vertices.squeeze(),
                    hand_faces=shadow_mesh.faces,
                    hand_point=hand_point,
                    hand_mesh_point=hand_mesh_point,
                    obj_verts=obj_verts_np,
                    obj_normals=obj_normals_np,
                    contact_info=contact_info,
                )
            else:
                runtime_viz = None
            opt1.set_opt_val(
                vertex_contact=vertex_contacts[begin:per_batch],
                contact_region=contact_regions[begin:per_batch],
                anchor_id=anchor_ids[begin:per_batch],
                # anchor_elasti=anchor_elasti,
                # anchor_padding_mask=anchor_padding_mask,
                # hand_shape_init=hand_shape,
                # hand_tsl_init=hand_tsl,
                # hand_pose_init=([i for i in range(16)], hand_pose),
                hand_pose_init=([i for i in range(29)], opt1.joint),
                obj_verts_3d_gt=obj_verts[begin:per_batch],
                obj_normals_gt=obj_normals[begin:per_batch],
                sdf_decoder=load_sdf_decoder(data, arg.resume),
                sdf_latent=load_obj_latent(data, target_instance),
                sdf_center=torch.tensor(center).to(device),
                sdf_rescale=torch.tensor(rescale).to(device),
                runtime_vis=runtime_viz,
            )
            opt1.optimize(progress=True)
            # print(opt1.joint)
            optimized = opt1.joint.clone().detach()

            opt2 = GeOptimizer(
                device=device,
                info=[optimized[:, 4:7], optimized[:, :4], optimized[:, 7:]],  # 3 + 4 + 18 = 25
                lr=arg.lr,
                # n_iter=int(arg.iters/2),
                n_iter=1,
                verbose=False,
                lambda_contact_loss=10.0,  # 单位mm
                lambda_repulsion_loss=8e8,  # 单位m/100
            )
            shadow_mesh = opt2.shadow_mesh
            hand_point = np.zeros((27, 3))
            hand_mesh_point = np.zeros((2000, 3))
            center = np.zeros(3, dtype=np.float32)
            path = os.path.join(data.split('/split')[0], sdf_samples_subdir, target_instance + '.obj.mat')
            sdf_scale = loadmat(path)
            rescale = sdf_scale['scale']  # scale是一个小于1的数， 是物体放大的系数, mesh_to_sdf 相当于将物体放入一个单位圆或者正方体,所以物体是放大了的
            if arg.vis:
                runtime_viz = init_runtime_viz(
                    # hand_verts=mano_out.verts.cpu().squeeze().numpy(),
                    # hand_faces=opt.mano_layer.th_faces.cpu().numpy(),
                    hand_verts=shadow_mesh.vertices.squeeze(),
                    hand_faces=shadow_mesh.faces,
                    hand_point=hand_point,
                    hand_mesh_point=hand_mesh_point,
                    obj_verts=obj_verts_np,
                    obj_normals=obj_normals_np,
                    contact_info=contact_info,
                )
            else:
                runtime_viz = None
            opt2.set_opt_val(
                vertex_contact=vertex_contacts[begin:per_batch],
                contact_region=contact_regions[begin:per_batch],
                anchor_id=anchor_ids[begin:per_batch],
                # anchor_elasti=anchor_elasti,
                # anchor_padding_mask=anchor_padding_mask,
                # hand_shape_init=hand_shape,
                # hand_tsl_init=hand_tsl,
                # hand_pose_init=([i for i in range(16)], hand_pose),
                hand_pose_init=([i for i in range(29)], opt2.joint),
                obj_verts_3d_gt=obj_verts[begin:per_batch],
                obj_normals_gt=obj_normals[begin:per_batch],
                sdf_decoder=load_sdf_decoder(data, arg.resume),
                sdf_latent=load_obj_latent(data, target_instance),
                sdf_center=torch.tensor(center).to(device),
                sdf_rescale=torch.tensor(rescale).to(device),
                runtime_vis=runtime_viz,
            )
            opt2.optimize(progress=True)

            # saving file after optimizing hand pose.....
            for i in range(opt2.joint.shape[0]):
                optimized_all_tensor = opt2.joint[i].clone().detach()
                optimized_all = opt2.joint[i].clone().detach().cpu().numpy()
                optimized_angles = trans18_to_22(opt2.joint[i][7:].reshape(1, 18).clone().detach()).cpu().numpy().reshape(22)
                info_transferred_list_batch = info_transferred_list[begin:per_batch]
                each_grasp2 = '{}_{}.xml'.format(str(arg.idx), info_transferred_list_batch[i].split('_')[-1])
                new_name = os.path.join(os.path.join(grasp_root, str(arg.idx) + '_' + soruce_idx, 'transferred', target_instance), each_grasp2)
                rs_wxyz = np.zeros(4)
                rs_wxyz[1:] = rs_obj.clone().detach().cpu().numpy()[:3]
                rs_wxyz[0] = rs_obj.clone().detach().cpu().numpy()[3]
                write_xml_new_data(category=category, obj_name=target_instance,
                                   r=optimized_all[:4],
                                   r_o=rs_wxyz,
                                   t=optimized_all[4:7]*1000.0,
                                   a=optimized_angles,
                                   path=new_name,
                                   mode='train', rs=(21, 'directly22angles'))
                new_name_pkl = new_name[:-4] + '.pkl'
                batch_feature = np.repeat(np.array([[1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0]]),
                                          obj_verts_np.shape[0], axis=0)

                new_j_p, transformed_points,_ = trans_xml_pkl(optimized_all_tensor[7:].unsqueeze(0) / 1.5708,
                                                              optimized_all_tensor[4: 7].unsqueeze(0) *1000.0,
                                                              optimized_all_tensor[: 4].unsqueeze(0))
                num = int(each_grasp2.split('_')[1].split('.')[0])
                save_pcd_pkl(obj_verts_np, batch_feature, str(target_instance),
                             new_j_p.squeeze(0).detach().cpu().numpy(),
                             str(arg.idx), num, new_name_pkl,
                             transformed_points.clone().squeeze(0).detach().cpu().numpy())

                # hand_pose = quaternion_to_angle_axis(hand_pose).reshape(48).numpy()

                # pickle.dump(
                #     {"pose": hand_pose, "shape": hand_shape, "tsl": hand_tsl},
                #     open(os.path.join(trans_contact_path, "hand_param.pkl"), "wb"),
                # )
                print(new_name)
                print("finish")

    #
    # interpolate_path = os.path.join(arg.data, "interpolate", f"{arg.source}-{arg.target}")
    # contact_info = pickle.load(open(os.path.join(arg.contact_path, arg.target, "contact_info.pkl"), "rb"))
    # trans_contact_path = os.path.join(arg.contact_path, arg.target)
    #
    # # if not arg.overwrite and os.path.exists(os.path.join(trans_contact_path, "hand_param.pkl")):
    # #     cprint(f"{os.path.join(trans_contact_path, 'hand_param.pkl')} exists, skip.", "yellow")
    # #     exit(0)
    #
    # try:
    #     mesh = trimesh.load(get_obj_path(arg.target), process=False, force="mesh", skip_materials=True)
    #
    #     if arg.center_obj:
    #         bbox_center = (mesh.vertices.min(0) + mesh.vertices.max(0)) / 2
    #         mesh.vertices = mesh.vertices - bbox_center
    #
    #     obj_pc = to_pointcloud(mesh)
    # except:
    #     pc_data = pickle.load(open(os.path.join(arg.data, "virtual_pc_cache", f"{arg.target}.pkl"), "rb"))
    #     obj_pc = o3d.geometry.PointCloud()
    #     obj_pc.points = o3d.utility.Vector3dVector(pc_data["points"])
    #     obj_pc.normals = o3d.utility.Vector3dVector(pc_data["normals"])
    #
    # # rescale = pickle.load(open(os.path.join(arg.data, "rescale.pkl"), "rb"))
    # # rescale = rescale["max_norm"] * rescale["scale"]
    #
    # if arg.center_obj:
    #     # * if the object is centered, we do not need to center it again.
    #     center = np.zeros(3, dtype=np.float32)
    # else:
    #     center = pickle.load(
    #         open(list(glob.iglob(f"{os.path.join(arg.data, 'SdfSamples_resize', arg.target)}/*.pkl"))[0], "rb")
    #     )
    #
    # device = torch.device("cuda")
    # obj_verts_np, obj_normals_np = np.asarray(obj_pc.points), np.asarray(obj_pc.normals)
    # obj_verts, obj_normals = torch.tensor(obj_verts_np).to(device), torch.tensor(obj_normals_np).to(device)
    #
    # # hand_param = pickle.load(open(os.path.join(arg.contact_path, "hand_param.pkl"), "rb"))
    # #
    # # if arg.fix_tsl:
    # #     contact_region_center = np.asfarray(obj_pc.points, dtype=np.float32)[contact_info["vertex_contact"] == 1].mean(0)
    # #     raw_mesh = trimesh.load(get_obj_path(arg.source), process=False, force="mesh", skip_materials=True)
    # #     raw_obj_pc = to_pointcloud(raw_mesh)
    # #     raw_contact_info = pickle.load(open(os.path.join(arg.contact_path, "contact_info.pkl"), "rb"))
    # #
    # #     raw_contact_region_center = np.asfarray(raw_obj_pc.points, dtype=np.float32)[
    # #         raw_contact_info["vertex_contact"] == 1
    # #     ].mean(0)
    # #
    # #     hand_param["tsl"] = hand_param["tsl"] + (contact_region_center - raw_contact_region_center)
    #
    # # hand_pose = angle_axis_to_quaternion(torch.tensor(hand_param["pose"].reshape(-1, 3))).to(device)
    # # hand_shape = torch.tensor(hand_param["shape"]).to(device)
    # # hand_tsl = torch.tensor(hand_param["tsl"]).to(device)
    #
    # vertex_contact = torch.from_numpy(contact_info["vertex_contact"]).long().to(device)
    # contact_region = torch.from_numpy(contact_info["hand_region"]).long().to(device)
    # anchor_id = torch.from_numpy(contact_info["anchor_id"]).long().to(device)
    # # anchor_elasti = torch.from_numpy(contact_info["anchor_elasti"]).float().to(device)
    # # anchor_padding_mask = torch.from_numpy(contact_info["anchor_padding_mask"]).long().to(device)
    # #
    # # hra, hpv = masking_load_driver("./assets/anchor", "./assets/hand_palm_full.txt")
    #
    # # hand_region_assignment = torch.from_numpy(hra).long().to(device)
    # # hand_palm_vertex_mask = torch.from_numpy(hpv).long().to(device)
    # # joint = torch.zeros(29)
    # # joint = torch.Tensor(
    # #     [0, 1, 0, 0, -0.12, -0.04, 0.13, 0.260677, -0.12708819, 0.44815338, 1.5708, 0.05383605403900149,
    # #      0.0098000765, 0.5904129, 1.0705094, 0, 0.010692179,
    # #      0.5704359, 1.5708, 0.21206278247833255, 0.014512147, 0.3004437,
    # #      1.383557, 0, 0.14294301, 0.7534741, 0.086198345, 0.2972183, 0.8795107])
    # # joint.requires_grad = True
    # opt = GeOptimizer(
    #     device=device,
    #     lr=arg.lr,
    #     n_iter=arg.iters,
    #     verbose=False,
    #     lambda_contact_loss=500.0,
    #     lambda_repulsion_loss=200.0,
    # )
    # # mano_out = opt.mano_layer(hand_pose[None], hand_shape[None])
    # # shadow_out = ShadowGripper(root_folder='./tink/', data=joint, filename='')
    # # shadow_mesh = shadow_out.get_meshes()
    #
    # shadow_mesh = opt.shadow_mesh
    # hand_point = np.zeros((27,3))
    # if arg.vis:
    #     runtime_viz = init_runtime_viz(
    #         # hand_verts=mano_out.verts.cpu().squeeze().numpy(),
    #         # hand_faces=opt.mano_layer.th_faces.cpu().numpy(),
    #         hand_verts=shadow_mesh.vertices.squeeze(),
    #         hand_faces=shadow_mesh.faces, hand_point=hand_point,
    #         obj_verts=obj_verts_np,
    #         obj_normals=obj_normals_np,
    #         contact_info=contact_info,
    #     )
    # else:
    #     runtime_viz = None
    # opt.set_opt_val(
    #     vertex_contact=vertex_contact,
    #     contact_region=contact_region,
    #     anchor_id=anchor_id,
    #     # anchor_elasti=anchor_elasti,
    #     # anchor_padding_mask=anchor_padding_mask,
    #     # hand_shape_init=hand_shape,
    #     # hand_tsl_init=hand_tsl,
    #     # hand_pose_init=([i for i in range(16)], hand_pose),
    #     hand_pose_init=([i for i in range(29)], opt.joint),
    #     obj_verts_3d_gt=obj_verts,
    #     obj_normals_gt=obj_normals,
    #     sdf_decoder=load_sdf_decoder(arg.data, arg.resume),
    #     sdf_latent=load_obj_latent(arg.data, arg.target),
    #     sdf_center=torch.tensor(center).to(device),
    #     # sdf_rescale=torch.tensor(rescale).to(device),
    #     runtime_vis=runtime_viz,
    # )
    # opt.optimize(progress=True)
    #
    # hand_pose, hand_shape, hand_tsl = opt.recover_hand_param()
    # hand_pose, hand_shape, hand_tsl = hand_pose.cpu(), hand_shape.cpu().numpy(), hand_tsl.cpu().numpy()
    # # hand_pose = quaternion_to_angle_axis(hand_pose).reshape(48).numpy()
    #
    # # pickle.dump(
    # #     {"pose": hand_pose, "shape": hand_shape, "tsl": hand_tsl},
    # #     open(os.path.join(trans_contact_path, "hand_param.pkl"), "wb"),
    # # )
    # print("finish")
