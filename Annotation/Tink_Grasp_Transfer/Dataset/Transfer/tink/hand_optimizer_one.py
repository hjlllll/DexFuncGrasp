from pprint import pprint

import numpy as np
import open3d as o3d
import torch
# from manotorch.anchorlayer import AnchorLayer
# from manotorch.axislayer import AxisLayer
# from manotorch.manolayer import ManoLayer
# from manotorch.utils.quatutils import (
#     angle_axis_to_quaternion,
#     normalize_quaternion,
#     quaternion_mul,
#     quaternion_to_angle_axis,
# )
# from manotorch.utils.rodrigues import rodrigues
from termcolor import colored
from tqdm import trange

from grasp_loss import FieldLoss, HandLoss, ObjectLoss
from sdf_loss import sdf_loss
from vis_contact_info import create_vertex_color
from util_shadow import ShadowGripper
import os
from FK_model_opt import fk_run
from utils.trans import *

from utils.FK_layer_mesh import FK_layer
class GeOptimizer:
    def __init__(
        self,
        device,
        info,
        lr=1e-2,
        n_iter=2500,
        verbose=True,
        # mano_root="assets/mano_v1_2",
        shadow_root = './tink/',
        anchor_path="assets/anchor",
        # values to initialize coef_val
        lambda_contact_loss=20.0,
        lambda_repulsion_loss=10.0,
    ):
        self.device = device
        self.lr = lr
        self.n_iter = n_iter

        # options
        self.verbose = verbose
        self.runtime_vis = None

        # layers and loss utils
        # self.mano_layer = ManoLayer(
        #     rot_mode="quat",
        #     mano_assets_root=mano_root,
        #     center_idx=0,
        #     flat_hand_mean=True,
        # ).to(self.device)


        #joint len=29
        self.joint = torch.from_numpy(np.zeros(25)).float()#.requires_grad_(True) # [7+18]=[25]
        self.joint[4:7] = info[0]  # 3
        self.joint[:4] = info[1]  # 4
        self.joint[7:] = info[2]  # 18
        self.joint.requires_grad = True
        # self.joint[[3 + 7, 7 + 7, 11 + 7, 15 + 7]] += self.joint[[4 + 7, 8 + 7, 12 + 7, 16 + 7]]
        # self.r_graspit = info[3].reshape(1,4)
        # self.r_xyzw = info[4].reshape(1,4)
        # self.r_graspit.requires_grad = True
        # self.r_xyzw.requires_grad = True
        self.shadowhand_layer = ShadowGripper(root_folder=shadow_root, filename='')
        # mesh_joint = self.joint.clone().detach()
        # mesh_joint[4:7] = mesh_joint[4:7]*1000.0
        mesh_joint = torch.zeros(29)
        self.shadow_mesh = self.shadowhand_layer.get_meshes(data=mesh_joint)
        #
        # view_mat_mesh = torch.tensor([[0.0, 0.0, 1.0],
        #                               [-1.0, 0, 0.0],
        #                               [0.0, -1.0, 0]])
        # index = [0, 1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19, 20, 21]
        # input_r = self.r_graspit
        # # input_r = var_hand_pose_assembled[0: 4].unsqueeze(0)
        # input_t = self.joint.unsqueeze(0)[:, 4: 4 + 3] * 1000.0
        # # input_r = input_r.detach()
        # # input_t = input_t.detach() * 1000.0#* 5.0 / 1000.0
        # # input_t = input_t #* 1000.0
        # # var_hand_pose_assembled[:,[3+7,7+7,11+7,15+7]] += var_hand_pose_assembled[:,[4+7,8+7,12+7,16+7]]
        # input_a = self.joint.unsqueeze(0)[:, 4 + 3: 4 + 3 + 22]#[:, index] / 1.5708
        # r = quat_mul_tensor(input_r.reshape(4), trans_2_quat_gpu(view_mat_mesh)).reshape(4)
        #
        # base_M = get_4x4_matrix_tensor(r, input_t).expand(1, 1, -1, -1).cuda()  # .reshape(1, 1, 4, 4).cuda()
        #
        # # angles = input_a.reshape(1, -1).cuda() # * 1.5708
        # fk_angles = torch.zeros((1, 22)).cuda()
        # fk_angles[:, :4] = input_a.clone()[:, 13:17]
        # fk_angles[:, 4:8] = input_a.clone()[:, 9:13]
        # fk_angles[:, 8:12] = input_a.clone()[:, 5:9]
        # fk_angles[:, 12:17] = input_a.clone()[:, :5]
        # fk_angles[:, 17:] = input_a.clone()[:, 17:]
        # fk_angles[:, [0, 4]] = -fk_angles[:, [0, 4]]  ########mention######
        # # fk_angles[:, 8] = -fk_angles[:, 8]  ########mention######
        # fk_angles[:, -2] = -fk_angles[:, -2]
        # fk_angles[:, -1] = -fk_angles[:, -1]
        # pre_rotations = torch.FloatTensor(fk_angles.clone().detach().cpu().numpy()).reshape(1, -1).cuda()
        # self.mesh_layer = FK_layer(base_M, pre_rotations)
        # self.mesh_layer.cuda()

        # self.anchor_layer = AnchorLayer(anchor_path).to(self.device)
        # self.axis_layer = AxisLayer().to(self.device)

        # opt val dict, const val dict
        self.opt_val = {}
        self.const_val = {}
        self.ctrl_val = {}
        self.coef_val = {
            "lambda_contact_loss": lambda_contact_loss,
            "lambda_repulsion_loss": lambda_repulsion_loss,
        }

        # creating slots for optimizer and scheduler
        self.optimizer = None
        self.optimizing = True
        self.scheduler = None

    def set_opt_val(
        self,
        # static val
        vertex_contact,  # TENSOR[NVERT, ] {0, 1}
        contact_region,  # TENSOR[NVERT, 1], int
        anchor_id,  # TENSOR[NVERT, 4]: int
        # anchor_elasti,  # TENSOR[NVERT, 4]
        # anchor_padding_mask,  # TENSOR[NVERT, 4] {0, 1}
        # dynamic val: hand
        hand_shape_gt=None,  # TENSOR[10, ]
        # hand_tsl_gt=None,  # TENSOR[3, ]
        hand_pose_gt=None,  # (LIST[NPROV, ]: int {0..16}, TENSOR[NPROV, 4])
        hand_shape_init=None,  # TENSOR[10, ]
        # hand_tsl_init=None,  # TENSOR[3, ]
        hand_pose_init=None,  # (LIST[NPROV, ]: int {0..16}, TENSOR[NPROV, 4])
        # dynamic val: obj
        obj_verts_3d_gt=None,
        obj_normals_gt=None,
        # sdf
        sdf_decoder=None,
        sdf_latent=None,
        sdf_center=None,
        sdf_rescale=None,
        # runtime viz
        runtime_vis=None,
    ):
        # ====== clear memory
        self.opt_val = {}
        self.const_val = {}
        self.ctrl_val = {
            "optimize_hand_shape": False,
            # "optimize_hand_tsl": False,
            "optimize_hand_pose": False,
            "optimize_obj": False,
        }

        self.sdf_decoder = sdf_decoder
        self.sdf_latent = sdf_latent
        self.sdf_center = sdf_center
        self.sdf_rescale = sdf_rescale

        # ============ process static values >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        # region
        vertex_contact = vertex_contact.long()
        anchor_id = anchor_id.long()
        # anchor_padding_mask = anchor_padding_mask.long()

        # boolean index contact_region, anchor_id, anchor_elasti && anchor_padding_mask
        obj_contact_region = contact_region[vertex_contact == 1]  # TENSOR[NCONT, ]
        indexed_anchor_id = anchor_id[vertex_contact == 1]  # TENSOR[NCONT, 4]
        # anchor_elasti = anchor_elasti[vertex_contact == 1, :]  # TENSOR[NCONT, 4]
        # anchor_padding_mask = anchor_padding_mask[vertex_contact == 1, :]  # TENSOR[NCONT, 4]

        # boolean mask indexing anchor_id, anchor_elasti && obj_vert_id
        # indexed_anchor_id = anchor_id[vertex_contact == 1]  # TENSOR[NVALID, ]

        # self.const_val["indexed_anchor_id"] = indexed_anchor_id
        self.const_val["indexed_vertex_id"] = indexed_anchor_id

        # self.const_val["indexed_anchor_elasti"] = anchor_elasti[anchor_padding_mask == 1]  # TENSOR[NVALID, ]

        # vertex_id = torch.arange(anchor_id.shape[0])[:, None].repeat_interleave(
        #     anchor_padding_mask.shape[1], dim=1
        # )  # TENSOR[NCONT, 4]
        # self.const_val["indexed_vertex_id"] = vertex_id[anchor_padding_mask == 1]  # TENSOR[NVALID, ]

        # tip_anchor_mask = torch.zeros(indexed_anchor_id.shape[0]).bool().to(self.device)
        # tip_anchor_list = [2, 3, 4, 9, 10, 11, 15, 16, 17, 22, 23, 24, 29, 30, 31]
        # for tip_anchor_id in tip_anchor_list:
        #     tip_anchor_mask = tip_anchor_mask | (self.const_val["indexed_anchor_id"] == tip_anchor_id)
        # self.const_val["indexed_elasti_k"] = torch.where(
        #     tip_anchor_mask, torch.Tensor([1.0]).to(self.device), torch.Tensor([0.1]).to(self.device)
        # ).to(self.device)

        # hand faces & edges
        # self.const_val["hand_faces"] = self.shadow_mesh.faces
        # self.const_val["static_verts"] = self.get_static_hand_verts()
        # self.const_val["hand_edges"] = HandLoss.get_edge_idx(self.const_val["hand_faces"])
        # self.const_val["static_edge_len"] = HandLoss.get_edge_len(
        #     self.const_val["static_verts"], self.const_val["hand_edges"]
        # )
        # endregion
        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

        # ============ dynamic val: hand >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        # region
        # ====== hand_shape
        if hand_shape_gt is not None and hand_shape_init is not None:
            raise RuntimeError("both hand_shape gt and init are provided")
        elif hand_shape_gt is not None and hand_shape_init is None:
            self.const_val["hand_shape_gt"] = hand_shape_gt
            self.ctrl_val["optimize_hand_shape"] = False
        elif hand_shape_gt is None and hand_shape_init is not None: ##########
            self.opt_val["hand_shape_var"] = hand_shape_init.detach().clone().requires_grad_(True)
            self.const_val["hand_shape_init"] = hand_shape_init
            self.ctrl_val["optimize_hand_shape"] = True
        else:
            # hand_tsl_gt is None and hand_tsl_init is None:
            self.opt_val["hand_shape_var"] = torch.zeros(29, dtype=torch.float, requires_grad=True, device=self.device)
            self.const_val["hand_shape_init"] = torch.zeros(29, dtype=torch.float, device=self.device)
            self.ctrl_val["optimize_hand_shape"] = True

        # ====== hand_tsl
        # if hand_tsl_gt is not None and hand_tsl_init is not None:
        #     raise RuntimeError("both hand_tsl gt and init are provided")
        # elif hand_tsl_gt is not None and hand_tsl_init is None:
        #     self.const_val["hand_tsl_gt"] = hand_tsl_gt
        #     self.ctrl_val["optimize_hand_tsl"] = False
        # elif hand_tsl_gt is None and hand_tsl_init is not None:
        #     self.opt_val["hand_tsl_var"] = hand_tsl_init.detach().clone().requires_grad_(True)
        #     self.const_val["hand_tsl_init"] = hand_tsl_init
        #     self.ctrl_val["optimize_hand_tsl"] = True
        # else:
        #     # hand_tsl_gt is None and hand_tsl_init is None:
        #     self.opt_val["hand_tsl_var"] = torch.zeros(3, dtype=torch.float, requires_grad=True, device=self.device)
        #     self.const_val["hand_tsl_init"] = torch.zeros(3, dtype=torch.float, device=self.device)
        #     self.ctrl_val["optimize_hand_tsl"] = True

        # ====== hand pose
        # this is complex! need special care
        if hand_pose_gt is not None and hand_pose_init is not None:
            # full gt and init provided
            gt_pose_idx, gt_pose_val = hand_pose_gt
            init_pose_idx, init_pose_val = hand_pose_init
            if len(set(gt_pose_idx).intersection(set(init_pose_idx))) > 0:
                raise RuntimeError("repeat hand_pose gt & init provided")
            if set(gt_pose_idx).union(set(init_pose_idx)) != set(range(16)):
                raise RuntimeError("hand_pose: not enough gt & init")
            self.const_val["hand_pose_gt_idx"] = gt_pose_idx
            self.const_val["hand_pose_gt_val"] = gt_pose_val
            self.const_val["hand_pose_var_idx"] = init_pose_idx
            self.opt_val["hand_pose_var_val"] = init_pose_val.clone().detach().requires_grad_(True)
            self.const_val["hand_pose_init_val"] = init_pose_val
            self.ctrl_val["optimize_hand_pose"] = True
        elif hand_pose_gt is not None and hand_pose_init is None:
            gt_pose_idx, gt_pose_val = hand_pose_gt
            self.const_val["hand_pose_gt_idx"] = gt_pose_idx
            self.const_val["hand_pose_gt_val"] = gt_pose_val
            if set(gt_pose_idx) == set(range(16)):
                # full gt provided
                self.const_val["hand_pose_var_idx"] = []
                self.opt_val["hand_pose_var_val"] = torch.zeros((0, 4), dtype=torch.float, device=self.device)
                self.ctrl_val["optimize_hand_pose"] = False
            else:
                # partial gt provided
                var_pose_idx = self.get_var_pose_idx(gt_pose_idx)
                n_var_pose = len(var_pose_idx)
                init_val = np.array([[0.9999, 0.0, -0.0101, 0.0]] * n_var_pose).astype(np.float32)
                self.const_val["hand_pose_var_idx"] = var_pose_idx
                self.opt_val["hand_pose_var_val"] = torch.tensor(
                    init_val, dtype=torch.float, requires_grad=True, device=self.device
                )
                init_val_true = np.array([[1.0, 0.0, 0.0, 0.0]] * n_var_pose).astype(np.float32)
                self.const_val["hand_pose_init_val"] = torch.tensor(init_val_true, dtype=torch.float, device=self.device)
                self.ctrl_val["optimize_hand_pose"] = True

        elif hand_pose_gt is None and hand_pose_init is not None:###########################3
            # full init provided
            init_pose_idx, init_pose_val = hand_pose_init
            # init_pose_val.requires_grad = True
            if set(init_pose_idx) != set(range(29)):
                raise RuntimeError("hand_pose: not enough init")
            self.const_val["hand_pose_gt_idx"] = []
            self.const_val["hand_pose_gt_val"] = torch.zeros(29, dtype=torch.float).to(self.device)
            self.const_val["hand_pose_var_idx"] = init_pose_idx
            self.opt_val["hand_pose_var_val"] = init_pose_val
            self.const_val["hand_pose_init_val"] = init_pose_val.detach().clone().requires_grad_(True)
            self.ctrl_val["optimize_hand_pose"] = True
        else:
            # hand_pose_gt is None and hand_pose_init is None:
            # nothing provided
            self.const_val["hand_pose_gt_idx"] = []
            self.const_val["hand_pose_gt_val"] = torch.zeros((0, 4), dtype=torch.float).to(self.device)
            self.const_val["hand_pose_var_idx"] = list(range(16))
            n_var_pose = 16
            init_val = np.array([[0.9999, 0.0, -0.0101, 0.0]] * n_var_pose).astype(np.float32)
            self.opt_val["hand_pose_var_val"] = torch.tensor(
                init_val, dtype=torch.float, requires_grad=True, device=self.device
            )
            init_val_true = np.array([[1.0, 0.0, 0.0, 0.0]] * n_var_pose).astype(np.float32)
            self.const_val["hand_pose_init_val"] = torch.tensor(init_val_true, dtype=torch.float, device=self.device)
            self.ctrl_val["optimize_hand_pose"] = True
        # endregion
        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

        # ============ dynamic val: obj >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        # region
        self.const_val["index"] = vertex_contact == 1
        self.const_val["obj_verts_3d_gt"] = obj_verts_3d_gt[vertex_contact == 1, :]
        self.const_val["obj_normals_gt"] = obj_normals_gt[vertex_contact == 1, :]
        self.ctrl_val["optimize_obj"] = False
        self.const_val["full_obj_verts_3d"] = obj_verts_3d_gt
        self.const_val["full_obj_normals"] = obj_normals_gt

        self.const_val["thumb"] = contact_region == 1
        self.const_val["firfinger"] = contact_region == 2
        self.const_val["midfinger"] = contact_region == 3
        self.const_val["ringfinger"] = contact_region == 4
        self.const_val["litfinger"] = contact_region == 5



        # endregion
        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        # ============ construct optimizer & scheduler >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        # region
        # ====== optimizer
        if (
            self.ctrl_val["optimize_hand_shape"]
            or self.ctrl_val["optimize_hand_tsl"]
            or self.ctrl_val["optimize_hand_pose"]
            or self.ctrl_val["optimize_obj"]
        ):
            # dispatch lr to different param
            param = []
            # if self.ctrl_val["optimize_hand_shape"]:
            #     param.append({"params": [self.opt_val["hand_shape_var"]]})
            # if self.ctrl_val["optimize_hand_tsl"]:
            #     param.append({"params": [self.opt_val["hand_tsl_var"]], "lr": self.lr})
            if self.ctrl_val["optimize_hand_pose"]:
                param.append({"params": [self.opt_val["hand_pose_var_val"]]})

            # if self.ctrl_val["optimize_obj"]:
            #     param.append({"params": [self.opt_val["obj_rot_var"]]})
            #     param.append({"params": [self.opt_val["obj_tsl_var"]], "lr": self.lr})
            self.optimizer = torch.optim.Adam(param, lr=self.lr)

            # self.optimizer = torch.optim.SGD(self.mesh_layer.root_offset.parameters(), lr=self.lr*10)
            # self.optimizer2 = torch.optim.SGD(self.mesh_layer.emb.parameters(), lr=self.lr * 0.001)
            self.optimizing = True
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, min_lr=1e-5, mode="min", factor=0.5, patience=20, verbose=False
            )
        else:
            self.optimizing = False
        # endregion
        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

        # ====== runtime viz
        self.runtime_vis = runtime_vis

        # ====== verbose
        if self.verbose:
            print("Optimizing: ", self.optimizing)
            pprint(self.ctrl_val)
            pprint(list(self.opt_val.keys()))
            pprint(list(self.const_val.keys()))
            pprint(self.coef_val)

    @staticmethod
    def get_var_pose_idx(sel_pose_idx):
        # gt has 16 pose
        all_pose_idx = set(range(16))
        sel_pose_idx_set = set(sel_pose_idx)
        var_pose_idx = all_pose_idx.difference(sel_pose_idx_set)
        return list(var_pose_idx)

    def get_static_hand_verts(self):
        # init_val_pose = np.array([[1.0, 0.0, 0.0, 0.0]] * 16).astype(np.float32)
        init_val_pose = np.array([1.0, 0.0, 0.0, 0.0,0 ,0, 0,
                                  0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]).astype(np.float32)
        vec_pose = torch.tensor(init_val_pose)#.reshape(-1).unsqueeze(0).float().to(self.device)
        # vec_shape = torch.zeros(1, 10).float().to(self.device)
        # shadow_layer = ShadowGripper(root_folder='./tink/',  filename='')
        shadow_out = self.shadow_layer.get_meshes(data=vec_pose)
        v = shadow_out.vertices.squeeze()
        return v

    @staticmethod
    def assemble_pose_vec(gt_idx, gt_pose, var_idx, var_pose):
        idx_tensor = torch.cat((torch.Tensor(gt_idx).long(), torch.Tensor(var_idx).long()))
        pose_tensor = torch.cat((gt_pose, var_pose), dim=0)
        pose_tensor = pose_tensor[torch.argsort(idx_tensor)]
        return pose_tensor

    @staticmethod
    def transf_vectors(vectors, tsl, rot):
        """
        vectors: [K, 3], tsl: [3, ], rot: [3, ]
        return: [K, 3]
        """
        rot_matrix = rodrigues(rot.unsqueeze(0)).squeeze(0).reshape((3, 3))
        vec = (rot_matrix @ vectors.T).T
        vec = vec + tsl
        return vec

    def loss_fn(self, opt_val, const_val, ctrl_val, coef_val, iter_bar):
        # var_hand_pose_assembled = self.assemble_pose_vec(
        #     const_val["hand_pose_gt_idx"],
        #     const_val["hand_pose_gt_val"],
        #     const_val["hand_pose_var_idx"],
        #     opt_val["hand_pose_var_val"],
        # )
        var_hand_pose_assembled = opt_val["hand_pose_var_val"].unsqueeze(0)#.float()

        # dispatch hand var
        # vec_pose = var_hand_pose_assembled.unsqueeze(0)
        if ctrl_val["optimize_hand_shape"]:
            vec_shape = opt_val["hand_shape_var"].unsqueeze(0)
        else:
            vec_shape = const_val["hand_shape_gt"].unsqueeze(0)
        # if ctrl_val["optimize_hand_tsl"]:
        #     vec_tsl = opt_val["hand_tsl_var"].unsqueeze(0)
        # else:
        #     vec_tsl = const_val["hand_tsl_gt"].unsqueeze(0)

        # use for debug
        # var_hand_pose_assembled[4 +3: 4 + 3+22] = torch.zeros(22)
        # var_hand_pose_assembled[9] = 1
        # var_hand_pose_assembled[8] = 0.3
        # var_hand_pose_assembled[7] = 0.749

        # rebuild shadow
        index = [0, 1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19, 20, 21]
        # input_r = self.r_graspit
        input_r = var_hand_pose_assembled[:, 0: 4]#.unsqueeze(0)
        input_t = var_hand_pose_assembled[:, 4: 4 + 3] *1000.0
        # input_r = input_r.detach()
        # input_t = input_t.detach() * 1000.0#* 5.0 / 1000.0
        # input_t = input_t #* 1000.0
        # var_hand_pose_assembled[:,[3+7,7+7,11+7,15+7]] += var_hand_pose_assembled[:,[4+7,8+7,12+7,16+7]]
        # input_a = var_hand_pose_assembled[:, 4 + 3: 4 + 3 + 22][:, index]/1.5708
        input_a = var_hand_pose_assembled[:, 4 + 3: ] / 1.5708
        #
        # # change to new mesh-point fk-layer
        # # thumb_id = torch.tensor([0,1,2,3,4,5,52,54,42,43,55,53,11,96,35,46,37,40,39,56,53,28]) + 1900
        # # ff_id = torch.tensor([21,23,0,81,69,86,93,14,20,80,1,2,95,83,10,17,61,85,90,19,31,79]) + 600
        # # mf_id = torch.tensor([21,23,0,81,69,86,93,14,20,80,1,2,95,83,10,17,61,85,90,19,31,79]) + 900
        # # rf_id = torch.tensor([21,23,0,81,69,86,93,14,20,80,1,2,95,83,10,17,61,85,90,19,31,79]) + 1200
        # # lf_id = torch.tensor([21,23,0,81,69,86,93,14,20,80,1,2,95,83,10,17,61,85,90,19,31,79]) + 1600
        # # id_list = {'thumb': thumb_id, 'firfinger': ff_id, 'midfinger': mf_id, 'ringfinger': rf_id, 'litfinger': lf_id}
        # # torch.save(id_list, './tink/gripper_models/assets/hand_point_idx.pth')

        id_list = torch.load('./tink/gripper_models/assets/hand_point_idx.pth')
        id_all = torch.cat((id_list['thumb'],id_list['firfinger'],id_list['midfinger'],id_list['ringfinger'],id_list['litfinger']),dim=0)
        # # transformed_pts = torch.zeros((1,2000,3)).cuda()
        # mesh_r = torch.zeros(4)
        # mesh_r[:3] = input_r.clone().reshape(4)[1:]
        # mesh_r[3] = input_r.clone().reshape(4)[0]
        outputs_F, transformed_pts, joints = trans_xml_pkl(input_a, input_t, input_r)

        # print(transformed_pts.shape)
        # _, transformed_pts = self.mesh_layer()

        transformed_pts_thumb = torch.clone(transformed_pts[:, id_list['thumb'], :])
        transformed_pts_ff = torch.clone(transformed_pts[:, id_list['firfinger'], :])
        transformed_pts_mf = torch.clone(transformed_pts[:, id_list['midfinger'], :])
        transformed_pts_rf = torch.clone(transformed_pts[:, id_list['ringfinger'], :])
        transformed_pts_lf = torch.clone(transformed_pts[:, id_list['litfinger'], :])
        # # transformed_pts_5 [5, 22, 3]
        transformed_pts_5 = torch.cat((transformed_pts_thumb,transformed_pts_ff,transformed_pts_mf,transformed_pts_rf,transformed_pts_lf),0)
        # transformed_pts_5 = torch.zeros((5,22,3)).cuda()
        # transformed_pts_5 = transformed_pts.squeeze(0)[id_all, :].cuda()

        idx_close = (np.array([27, 21, 16, 11, 6])).tolist()  # 取出5根手指尖
        # outputs_F = fk_run(input_r, input_t, input_a)[:, :28] #* 0.001
        outputs_FK = outputs_F[:, idx_close].squeeze().float().cuda()  # [5, 3]
        # var_hand_pose_assembled = torch.tensor(var_hand_pose_assembled)
        # shadow_out = ShadowGripper(root_folder='./tink/',  filename='')

        obj_points = const_val["obj_verts_3d_gt"].clone().float().cuda() #* 1000.0 #接触的点
        features = torch.zeros(const_val["full_obj_verts_3d"].shape[0], 5)
        features[const_val["thumb"], 0] = 1
        features[const_val["firfinger"], 1] = 1
        features[const_val["midfinger"], 2] = 1
        features[const_val["ringfinger"], 3] = 1
        features[const_val["litfinger"], 4] = 1
        thumb_contact_num = torch.where(features[:,0]==1)[0].shape[0]
        firfinger_contact_num = torch.where(features[:,1]==1)[0].shape[0]
        midfinger_contact_num = torch.where(features[:,2]==1)[0].shape[0]
        ringfinger_contact_num = torch.where(features[:,3]==1)[0].shape[0]
        litfinger_contact_num = torch.where(features[:,4]==1)[0].shape[0]

        features = features[const_val["index"]].float().cuda()
        # batch_distance [obj_num, 5, 22]
        batch_distance = ((obj_points.unsqueeze(1).expand(-1, transformed_pts_5.unsqueeze(0).shape[1], -1)
                           .unsqueeze(2).expand(-1,-1,transformed_pts_5.unsqueeze(0).shape[2],-1)
                           - transformed_pts_5.unsqueeze(0).expand(obj_points.shape[0], -1, -1, -1)) ** 2).sum(-1).sqrt()  # [obj_n, fk_n]
        #取22个距离的均值, 以及接触的点的均值
        # batch_dis_close = batch_distance.sum(-1) * features / batch_distance.shape[2]
        # batch_dis_close[features == 0] = float("inf")
        # loss_close = torch.min(batch_dis_close, -2)[0]
        # loss_close[loss_close == float("inf")] = 0
        # contact_loss = loss_close.sum()
        num_all = torch.tensor([[thumb_contact_num, firfinger_contact_num, midfinger_contact_num,
                                 ringfinger_contact_num, litfinger_contact_num]]).float(). \
            expand(obj_points.shape[0], -1).unsqueeze(2).expand(-1, -1, batch_distance.shape[2]).cuda()
        num_all[torch.where(num_all==0)] = 1
        batch_dis_close = batch_distance * features.unsqueeze(2).expand(-1,-1,batch_distance.shape[2])
        batch_dis_close[features.unsqueeze(2).expand(-1,-1,batch_distance.shape[2]) == 0] = 0
        # loss_close = (batch_dis_close/batch_distance.shape[2]).sum(0)/num_all
        loss_close = batch_dis_close/num_all
        contact_loss = loss_close.sum()#/loss_close.shape[0]
        # print(transformed_pts[0].shape, transformed_pts_5[0].shape,  obj_points.shape)
        # dist = torch.sum(torch.pow(transformed_pts[0][:45] - obj_points[:45], 2), dim=1)  # TENSOR[NVALID, ]
        # contact_loss = torch.mean(dist, dim=0)

        # obj_points = const_val["obj_verts_3d_gt"].clone().float().cuda()
        # features = torch.zeros(const_val["full_obj_verts_3d"].shape[0], 5)
        # features[const_val["thumb"], 0] = 1
        # features[const_val["firfinger"], 1] = 1
        # features[const_val["midfinger"], 2] = 1
        # features[const_val["ringfinger"], 3] = 1
        # features[const_val["litfinger"], 4] = 1
        # features = features[const_val["index"]].float().cuda()
        # batch_distance = ((obj_points.unsqueeze(1).expand(-1, outputs_FK.unsqueeze(0).shape[1],
        #                                                   -1) - outputs_FK.unsqueeze(0).expand(obj_points.shape[0], -1,
        #                                                                                        -1)) ** 2).sum(
        #     -1).sqrt()  # [obj_n, fk_n]
        # batch_dis_close = batch_distance * features
        # batch_dis_close[features == 0] = float("inf")
        # loss_close = torch.min(batch_dis_close, -2)[0]
        # loss_close[loss_close == float("inf")] = 0
        # contact_loss = loss_close.sum()

        angle_lower = torch.tensor(
            [0., -20., 0., 0., -20., 0., 0., -20., 0., 0., -20., 0., 0., -60., 0., -12, -30.,
             0.]).cuda() / 90.0  # * 1.5708 / 1.5708
        angle_upper = torch.tensor(
            [45., 20., 90., 180., 20., 90., 180., 20., 90., 180., 20., 90., 180., 60., 70., 12., 30.,
             90.]).cuda() / 90.0  # * 1.5708 / 1.5708
        input_a = input_a.cuda()
        angle_lower_pair = torch.zeros(
            [2, input_a.reshape(-1).shape[0]]).cuda()  # [2,12*18]  outputs_a = [12,18]
        angle_upper_pair = torch.zeros([2, input_a.reshape(-1).shape[0]]).cuda()
        angle_lower_pair[0] = angle_lower.repeat(input_a.shape[0]) - input_a.reshape(
            -1)  # [12*18] - [12*18]
        angle_upper_pair[0] = input_a.reshape(-1) - angle_upper.repeat(input_a.shape[0])
        loss_angles = (torch.max(angle_lower_pair, 0)[0] + torch.max(angle_upper_pair, 0)[0]).sum()

        hand_self_distance0 = (
                    (outputs_F[:, [16]].unsqueeze(2).expand(-1, -1, outputs_F[:, [27, 21, 11, 6]].shape[1], -1)
                     - outputs_F[:, [27, 21, 11, 6]].unsqueeze(1).expand(-1, outputs_F[:, [16]].shape[1], -1,
                                                                          -1)) ** 2).sum(
            -1).sqrt().reshape(outputs_F.shape[0], -1)
        hand_self_distance01 = (
                    (outputs_F[:, [15]].unsqueeze(2).expand(-1, -1, outputs_F[:, [26, 20, 10, 5]].shape[1], -1)
                     - outputs_F[:, [26, 20, 10, 5]].unsqueeze(1).expand(-1, outputs_F[:, [15]].shape[1], -1,
                                                                          -1)) ** 2).sum(
            -1).sqrt().reshape(outputs_F.shape[0], -1)
        # 拇指指尖+无名指指尖 vs 食指指尖+小拇指指尖
        hand_self_distance1 = ((outputs_F[:, [27, 11]].unsqueeze(2).expand(-1, -1, outputs_F[:, [21, 6]].shape[1],
                                                                            -1) - outputs_F[:, [21, 6]].unsqueeze(
            1).expand(-1, outputs_F[:, [27, 11]].shape[1], -1, -1)) ** 2).sum(-1).sqrt().reshape(
            outputs_F.shape[0], -1)
        # 拇指指尖+小拇指指尖 vs 食指指尖+无名指指尖
        hand_self_distance2 = ((outputs_F[:, [27, 6]].unsqueeze(2).expand(-1, -1, outputs_F[:, [21, 11]].shape[1], -1)
                                - outputs_F[:, [21, 11]].unsqueeze(1).expand(-1, outputs_F[:, [27, 6]].shape[1], -1,
                                                                              -1)) ** 2).sum(-1).sqrt().reshape(
            outputs_F.shape[0], -1)
        # 小拇指2\3关节 vs 其他指尖
        hand_self_distance03 = (
                    (outputs_F[:, [5, 4]].unsqueeze(2).expand(-1, -1, outputs_F[:, [21, 16, 11]].shape[1], -1)
                     - outputs_F[:, [21, 16, 11]].unsqueeze(1).expand(-1, outputs_F[:, [5, 4]].shape[1], -1,
                                                                       -1)) ** 2).sum(-1).sqrt().reshape(
            outputs_F.shape[0], -1)  # 21,20,16,11,9,10,14,15

        hand_self_distance5 = (
                    (outputs_F[:, [26, 5]].unsqueeze(2).expand(-1, -1, outputs_F[:, [20, 10, 15]].shape[1], -1)
                     - outputs_F[:, [20, 10, 15]].unsqueeze(1).expand(-1, outputs_F[:, [26, 5]].shape[1], -1,
                                                                       -1)) ** 2).sum(
            -1).sqrt().reshape(outputs_F.shape[0], -1)

        hand_self_distance = torch.cat([hand_self_distance0, hand_self_distance1, hand_self_distance2,
                                        hand_self_distance01, hand_self_distance03, hand_self_distance5], 1).reshape(
            -1)  # /240
        hand_self_pair = torch.zeros([2, hand_self_distance.shape[0]]).cuda()
        hand_self_pair[0] = 22 - hand_self_distance
        loss_handself = torch.max(hand_self_pair, 0)[0].sum() / (outputs_F.shape[0] * 28)




        # mesh_joint = var_hand_pose_assembled.squeeze().clone().detach()
        view_mat_mesh = torch.tensor([[0.0, 1.0, 0.0],
                                      [-1.0, 0, 0.0],
                                      [0.0, 0.0, 1.0]])
        mesh_r = torch.zeros(4)
        mesh_r[:3] = input_r.clone().reshape(4)[1:]
        mesh_r[3] = input_r.clone().reshape(4)[0]
        rr = quat_mul(mesh_r.clone().detach().cpu().numpy().reshape(4), trans_2_quat_cpu(view_mat_mesh)).reshape(4)
        input_rwxyz = torch.zeros(4)
        input_rwxyz[1:] = torch.from_numpy(rr).reshape(4)[:3]
        input_rwxyz[0] = torch.from_numpy(rr).reshape(4)[3]
        mesh_joint = torch.zeros(29)
        mesh_joint[4:7] = var_hand_pose_assembled.squeeze().clone().detach()[4:7]*1000.0
        mesh_joint[:4] = input_rwxyz
        mesh_joint[7:] = joints.clone().detach() * 1.5708
        shadow_mesh = self.shadowhand_layer.get_meshes(data=mesh_joint)
        # shadow_mesh = self.shadowhand_layer.get_meshes(data=opt_val["hand_pose_var_val"].clone().detach())
        # mano_out = self.mano_layer(vec_pose, vec_shape)
        # rebuild_verts, rebuild_joints, rebuild_transf = shadow_mesh.vertices, shadow_mesh.joints, mano_out.transforms_abs
        rebuild_verts = shadow_mesh.vertices
        rebuild_verts_squeezed = shadow_mesh.vertices.squeeze()

        # rebuild_joints = rebuild_joints + vec_tsl
        # rebuild_verts = rebuild_verts + vec_tsl
        # rebuild_transf = rebuild_transf + torch.cat(
        #     [
        #         torch.cat([torch.zeros(3, 3).to(self.device), vec_tsl.view(3, -1)], dim=1),
        #         torch.zeros(1, 4).to(self.device),
        #     ],
        #     dim=0,
        # )

        # rebuild anchor
        # rebuild_anchor = self.anchor_layer(rebuild_verts)
        # rebuild_anchor = rebuild_anchor.contiguous()  # TENSOR[1, 32, 3]
        # rebuild_anchor = rebuild_anchor.squeeze(0)  # TENSOR[32, 3]
        # anchor_pos = rebuild_anchor[const_val["indexed_anchor_id"]]  # TENSOR[NVALID, 3]

        # hand_index = np.loadtxt(os.path.join("./tink/gripper_models/assets/hand_id.txt"), dtype=np.int32)
        # anchor_pos = torch.from_numpy(rebuild_verts.squeeze()[hand_index]).requires_grad_(True).float()

        # rebuild_verts_squeezed = torch.tensor(rebuild_verts_squeezed)
        # anchor_pos = rebuild_verts_squeezed[const_val["indexed_vertex_id"]]
        # dispatch obj var
        if ctrl_val["optimize_obj"]:
            obj_verts = self.transf_vectors(
                const_val["obj_verts_3d_can"],
                opt_val["obj_tsl_var"],
                opt_val["obj_rot_var"],
            )
            full_obj_verts = self.transf_vectors(
                const_val["full_obj_verts_3d"],
                opt_val["obj_tsl_var"],
                opt_val["obj_rot_var"],
            )
            full_obj_normals = self.transf_vectors(
                const_val["full_obj_normals"],
                torch.zeros(3, dtype=torch.float, device=self.device),
                opt_val["obj_rot_var"],
            )
        else:
            obj_verts = const_val["obj_verts_3d_gt"][:outputs_FK.shape[0]]
            # obj_verts = torch.tensor(obj_verts, dtype=torch.float32)
            obj_verts = obj_verts.clone()
            full_obj_verts = const_val["full_obj_verts_3d"]
            full_obj_normals = const_val["full_obj_normals"]

        # contact loss

        # contact_loss = FieldLoss.contact_loss(
        #     outputs_FK,
        #     # obj_verts[const_val["indexed_vertex_id"]],
        #     obj_verts,
        #     # const_val["indexed_anchor_elasti"],
        #     # const_val["indexed_elasti_k"],
        # )

        # sdf_model_loss = sdf_loss(
        #     self.sdf_decoder, self.sdf_latent, rebuild_verts_squeezed, self.sdf_center, self.sdf_rescale
        # )
        sdf_model_loss = sdf_loss(
            self.sdf_decoder, self.sdf_latent, transformed_pts.squeeze(), self.sdf_center, self.sdf_rescale
        )


        if ctrl_val["optimize_hand_pose"]:
            # get hand loss
            # quat_norm_loss = HandLoss.pose_quat_norm_loss(var_hand_pose_assembled)
            # # var_hand_pose_normalized = normalize_quaternion(var_hand_pose_assembled)
            # pose_reg_loss = HandLoss.pose_reg_loss(
            #     var_hand_pose_assembled[const_val["hand_pose_var_idx"]], const_val["hand_pose_init_val"]
            # )
            #
            # # b_axis, u_axis, l_axis = self.axis_layer(rebuild_joints, rebuild_transf)
            #
            # angle_axis = quaternion_to_angle_axis(var_hand_pose_assembled.reshape((16, 4)))
            # angle_axis = angle_axis[1:, :]  # ignore global rot [15, 3]
            # axis = angle_axis / torch.norm(angle_axis, dim=1, keepdim=True)
            # angle = torch.norm(angle_axis, dim=1, keepdim=False)
            # # limit angle
            # angle_limit_loss = HandLoss.rotation_angle_loss(angle)
            #
            # # joint_b_axis_loss = HandLoss.joint_b_axis_loss(b_axis, axis)
            # # joint_u_axis_loss = HandLoss.joint_u_axis_loss(u_axis, axis)
            # # joint_l_limit_loss = HandLoss.joint_l_limit_loss(l_axis, axis)
            #
            # edge_loss = HandLoss.edge_len_loss(
            #     rebuild_verts_squeezed, const_val["hand_edges"], const_val["static_edge_len"]
            # )

            quat_norm_loss = torch.Tensor([0.0]).to(self.device)
            pose_reg_loss = torch.Tensor([0.0]).to(self.device)
            # angle_limit_loss = torch.Tensor([0.0]).to(self.device)
            angle_limit_loss = loss_angles
            handself_loss = loss_handself
        else:
            quat_norm_loss = torch.Tensor([0.0]).to(self.device)
            pose_reg_loss = torch.Tensor([0.0]).to(self.device)
            angle_limit_loss = torch.Tensor([0.0]).to(self.device)
            # joint_b_axis_loss = torch.Tensor([0.0]).to(self.device)
            # joint_u_axis_loss = torch.Tensor([0.0]).to(self.device)
            # joint_l_limit_loss = torch.Tensor([0.0]).to(self.device)
            # edge_loss = torch.Tensor([0.0]).to(self.device)
            pose_reg_loss_to_zero = torch.Tensor([0.0]).to(self.device)
            handself_loss = torch.Tensor([0.0]).to(self.device)
        if ctrl_val["optimize_hand_shape"]:
            # shape_reg_loss = HandLoss.shape_reg_loss(opt_val["hand_shape_var"], const_val["hand_shape_init"])
            shape_reg_loss = torch.Tensor([0.0]).to(self.device)

        else:
            shape_reg_loss = torch.Tensor([0.0]).to(self.device)

        # if ctrl_val["optimize_hand_tsl"]:
        #     # hand_tsl_loss = HandLoss.hand_tsl_loss(opt_val["hand_tsl_var"], const_val["hand_tsl_init"])
        #     hand_tsl_loss = torch.Tensor([0.0]).to(self.device)
        #
        # else:
        #     hand_tsl_loss = torch.Tensor([0.0]).to(self.device)

        if ctrl_val["optimize_obj"]:
            # obj_transf_loss = ObjectLoss.obj_transf_loss(
            #     opt_val["obj_tsl_var"], opt_val["obj_rot_var"], const_val["obj_tsl_init"], const_val["obj_rot_init"]
            # )
            obj_transf_loss = torch.Tensor([0.0]).to(self.device)

        else:
            obj_transf_loss = torch.Tensor([0.0]).to(self.device)
        # print(contact_loss)

        # 分段loss 20221206
        # if iter_bar <= 400:
        #     coef_val["lambda_repulsion_loss"] = 0.0
        # if iter_bar > 400 and iter_bar <=600:
        #     coef_val["lambda_contact_loss"] = 0.0
        #     coef_val["lambda_repulsion_loss"] = 10.0

        loss = (
            # ============= HAND ANATOMICAL LOSS
            0.1 * quat_norm_loss
            + 1 * angle_limit_loss
            + 1 * handself_loss
            # + 1.0 * edge_loss
            # + 0.1 * joint_b_axis_loss
            # + 0.1 * joint_u_axis_loss
            # + 0.1 * joint_l_limit_loss
            # ============= ELAST POTENTIONAL ENERGY
            + coef_val["lambda_contact_loss"] * contact_loss
            + coef_val["lambda_repulsion_loss"] * sdf_model_loss
            # ============= OFFSET LOSS
            + 1.0 * pose_reg_loss
            # + 1.0 * shape_reg_loss
            # + 1.0 * hand_tsl_loss
            + 1.0 * obj_transf_loss
        )

        # debug: runtime viz
        if self.runtime_vis:
            if self.ctrl_val["optimize_obj"]:
                full_obj_verts = self.transf_vectors(
                    self.const_val["full_obj_verts_3d"],
                    self.opt_val["obj_tsl_var"].detach(),
                    self.opt_val["obj_rot_var"].detach(),
                )
            else:
                full_obj_verts = self.const_val["full_obj_verts_3d"]

            # if not ctrl_val["optimize_hand_pose"]:
                # b_axis, u_axis, l_axis = self.axis_layer(rebuild_joints, rebuild_transf)  # mend this up
            # self.runtime_show(rebuild_verts, b_axis, u_axis, l_axis, rebuild_transf, full_obj_verts)
            rebuild_verts = torch.from_numpy(rebuild_verts).clone().detach()#.requires_grad_(True)
            self.runtime_show(rebuild_verts, full_obj_verts, outputs_F, transformed_pts, iter_bar)

        return (
            loss,
            {
                "quat_norm_loss": quat_norm_loss.detach().cpu().item(),
                "angle_limit_loss": angle_limit_loss.detach().cpu().item(),
                # "edge_loss": edge_loss.detach().cpu().item(),
                # "joint_b_axis_loss": joint_b_axis_loss.detach().cpu().item(),
                # "joint_u_axis_loss": joint_u_axis_loss.detach().cpu().item(),
                # "joint_l_limit_loss": joint_l_limit_loss.detach().cpu().item(),
                "contact_loss": contact_loss.detach().cpu().item(),
                # "repulsion_loss": repulsion_loss.detach().cpu().item(),
                "repulsion_loss": sdf_model_loss.detach().cpu().item(),
                "pose_reg_loss": pose_reg_loss.detach().cpu().item(),
                # "hand_tsl_loss": hand_tsl_loss.detach().cpu().item(),
                "obj_transf_loss": obj_transf_loss.detach().cpu().item(),
            },
        )

    def optimize(self, progress=False):
        if progress:
            bar = trange(self.n_iter, position=3)
            bar_hand = trange(0, position=2, bar_format="{desc}")
            bar_contact = trange(0, position=1, bar_format="{desc}")
            bar_axis = trange(0, position=0, bar_format="{desc}")
        else:
            bar = range(self.n_iter)

        loss = torch.Tensor([1000.0]).to(self.device)
        loss_dict = {}
        # for _ in bar:
        for iter_bar in bar:
            # for name, param in self.mesh_layer.named_parameters():
            #     print('name :', name)
            #     print('param:', param)
            #     print('grad required:', param.requires_grad)
            #     print('grad value   :', param.grad)
            loss, loss_dict = self.loss_fn(self.opt_val, self.const_val, self.ctrl_val, self.coef_val, iter_bar)
            if self.optimizing:
                self.optimizer.zero_grad()
                # self.optimizer2.zero_grad()
            print(loss_dict)
            if self.optimizing:
                loss.backward(retain_graph=True)
                self.optimizer.step()
                # self.optimizer2.step()
                self.scheduler.step(loss)

            if progress:
                bar.set_description("TOTAL LOSS {:4e}".format(loss.item()))
                try:
                    bar_hand.set_description(
                        colored("HAND_REGUL_LOSS: ", "yellow")
                        + "QN={:.3e} PR={:.3e} EG={:.3e}".format(
                            loss_dict["quat_norm_loss"],  # QN
                            loss_dict["pose_reg_loss"],  # PR
                            # loss_dict["edge_loss"],  # Edge
                        )
                    )
                except:
                    pass
                try:
                    bar_contact.set_description(
                        colored("HO_CONTACT_LOSS: ", "blue")
                        + "Conta={:.3e}, Repul={:.3e}, OT={:.3e}".format(
                            loss_dict["contact_loss"],  # Conta
                            loss_dict["repulsion_loss"],  # Repul
                            loss_dict["obj_transf_loss"],  # OT
                        )
                    )
                except:
                    pass
                # try:
                #     bar_axis.set_description(
                #         colored("ANGLE_LOSS: ", "cyan")
                #         + "AL={:.3e} JB={:.3e} JU={:.3e} JL={:.3e}".format(
                #             loss_dict["angle_limit_loss"],  # AL
                #             loss_dict["joint_b_axis_loss"],  # JB
                #             loss_dict["joint_u_axis_loss"],  # JU
                #             loss_dict["joint_l_limit_loss"],  # JL
                #         )
                #     )
                # except:
                #     pass
        return loss.item(), loss_dict

    def recover_hand(self, squeeze_out=True):
        vars_hand_pose_assembled = self.assemble_pose_vec(
            self.const_val["hand_pose_gt_idx"],
            self.const_val["hand_pose_gt_val"],
            self.const_val["hand_pose_var_idx"],
            self.opt_val["hand_pose_var_val"],
        ).detach()
        vars_hand_pose_normalized = normalize_quaternion(vars_hand_pose_assembled)
        vec_pose = vars_hand_pose_normalized.unsqueeze(0)
        if self.ctrl_val["optimize_hand_shape"]:
            vec_shape = self.opt_val["hand_shape_var"].detach().unsqueeze(0)
        else:
            vec_shape = self.const_val["hand_shape_gt"].unsqueeze(0)
        if self.ctrl_val["optimize_hand_tsl"]:
            vec_tsl = self.opt_val["hand_tsl_var"].detach().unsqueeze(0)
        else:
            vec_tsl = self.const_val["hand_tsl_gt"].unsqueeze(0)

        device = vec_pose.device
        mano_out = self.mano_layer(vec_pose, vec_shape)
        rebuild_verts, rebuild_joints, rebuild_transf = mano_out.verts, mano_out.joints, mano_out.transforms_abs

        rebuild_verts = rebuild_verts + vec_tsl
        rebuild_joints = rebuild_joints + vec_tsl
        rebuild_transf = rebuild_transf + torch.cat(
            [
                torch.cat((torch.zeros((3, 3), device=device), vec_tsl.T), dim=1),
                torch.zeros((1, 4), device=device),
            ],
            dim=0,
        )
        if squeeze_out:
            rebuild_verts, rebuild_joints, rebuild_transf = (
                rebuild_verts.squeeze(0),
                rebuild_joints.squeeze(0),
                rebuild_transf.squeeze(0),
            )
        return rebuild_verts, rebuild_joints, rebuild_transf

    def recover_hand_pose(self):
        vars_hand_pose_assembled = self.assemble_pose_vec(
            self.const_val["hand_pose_gt_idx"],
            self.const_val["hand_pose_gt_val"],
            self.const_val["hand_pose_var_idx"],
            self.opt_val["hand_pose_var_val"],
        ).detach()
        vars_hand_pose_normalized = normalize_quaternion(vars_hand_pose_assembled)
        return vars_hand_pose_normalized

    def recover_hand_param(self):
        hand_pose = self.recover_hand_pose()
        if self.ctrl_val["optimize_hand_shape"]:
            hand_shape = self.opt_val["hand_shape_var"].detach()
        else:
            hand_shape = self.const_val["hand_shape_gt"]
        if self.ctrl_val["optimize_hand_tsl"]:
            hand_tsl = self.opt_val["hand_tsl_var"].detach()
        else:
            hand_tsl = self.const_val["hand_tsl_gt"]
        return hand_pose, hand_shape, hand_tsl

    def recover_obj(self):
        if self.ctrl_val["optimize_obj"]:
            obj_verts = self.transf_vectors(
                self.const_val["full_obj_verts_3d"],
                self.opt_val["obj_tsl_var"].detach(),
                self.opt_val["obj_rot_var"].detach(),
            )
        else:
            obj_verts = self.const_val["full_obj_verts_3d"]
        return obj_verts

    def obj_rot_np(self):
        if self.ctrl_val["optimize_obj"]:
            res = self.opt_val["obj_rot_var"].detach().cpu().numpy()
            return res
        else:
            raise RuntimeError("not optimizing obj, cannot get obj_rot")

    def obj_tsl_np(self):
        if self.ctrl_val["optimize_obj"]:
            res = self.opt_val["obj_tsl_var"].detach().cpu().numpy()
            return res
        else:
            raise RuntimeError("not optimizing obj, cannot get obj_tsl")

    # def runtime_show(self, hand_verts, b_axis, u_axis, l_axis, hand_transf, obj_verts):
    def runtime_show(self, hand_verts, obj_verts, hand_point, transformed_pts, iter_bar):

        # hand_transf = hand_transf.detach().cpu().squeeze(0).numpy()


        while True:
            # self.runtime_vis["hand_mesh"].vertices = o3d.utility.Vector3dVector(
            #     np.array(hand_verts.detach().cpu().squeeze(0))
            # )
            self.runtime_vis["hand_mesh"].vertices = o3d.utility.Vector3dVector(
                np.array(hand_verts.detach().squeeze())
            )
            self.runtime_vis["hand_mesh"].compute_vertex_normals()
            self.runtime_vis["obj_mesh"].points = o3d.utility.Vector3dVector(
                np.array(obj_verts.detach().cpu().squeeze(0))
            )
            self.runtime_vis["hand_point"].points = o3d.utility.Vector3dVector(
                np.array(hand_point.detach().cpu().squeeze(0)))

            self.runtime_vis["hand_mesh_point"].points = o3d.utility.Vector3dVector(
                np.array(transformed_pts.detach().cpu().squeeze(0)))

            lines_shadow = [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [0, 7], [7, 8], [8, 9], [9, 10], [10, 11],
                            [0, 12], [12, 13], [13, 14], [14, 15], [15, 16],
                            [0, 17], [17, 18], [18, 19], [19, 20], [20, 21], [0, 22], [22, 23], [23, 24], [24, 25],
                            [25, 26], [26, 27]]
            self.runtime_vis["hand_line"].lines = o3d.utility.Vector2iVector(lines_shadow)
            self.runtime_vis["hand_line"].points = o3d.utility.Vector3dVector(
                np.array(hand_point.detach().cpu().squeeze(0)))


            # self.runtime_vis["obj_mesh"].compute_vertex_normals()
            self.runtime_vis["window"].update_geometry(self.runtime_vis["hand_mesh"])
            self.runtime_vis["window"].update_geometry(self.runtime_vis["obj_mesh"])
            self.runtime_vis["window"].update_geometry(self.runtime_vis["hand_point"])
            self.runtime_vis["window"].update_geometry(self.runtime_vis["hand_mesh_point"])
            self.runtime_vis["window"].update_geometry(self.runtime_vis["hand_line"])
            self.runtime_vis["window"].poll_events()
            self.runtime_vis["window"].update_renderer()
            # if iter_bar == 15:
            #     self.runtime_vis["window"].destroy_window()
            if not self.runtime_vis["window"].poll_events():
                break

        return


def caculate_align_mat(vec):
    vec = vec / np.linalg.norm(vec)
    z_unit_Arr = np.array([0, 0, 1])

    z_mat = np.array(
        [
            [0, -z_unit_Arr[2], z_unit_Arr[1]],
            [z_unit_Arr[2], 0, -z_unit_Arr[0]],
            [-z_unit_Arr[1], z_unit_Arr[0], 0],
        ]
    )

    z_c_vec = np.matmul(z_mat, vec)
    z_c_vec_mat = np.array(
        [
            [0, -z_c_vec[2], z_c_vec[1]],
            [z_c_vec[2], 0, -z_c_vec[0]],
            [-z_c_vec[1], z_c_vec[0], 0],
        ]
    )

    if np.dot(z_unit_Arr, vec) == -1:
        qTrans_Mat = -np.eye(3, 3)
    elif np.dot(z_unit_Arr, vec) == 1:
        qTrans_Mat = np.eye(3, 3)
    else:
        qTrans_Mat = np.eye(3, 3) + z_c_vec_mat + np.matmul(z_c_vec_mat, z_c_vec_mat) / (1 + np.dot(z_unit_Arr, vec))

    return qTrans_Mat


def init_runtime_viz(
    hand_verts,
    hand_faces,
    hand_point,
    hand_mesh_point,
    obj_verts,
    obj_normals,
    contact_info,
    cam_extr=None,
):

    hand_mesh_cur = o3d.geometry.TriangleMesh()
    hand_mesh_cur.triangles = o3d.utility.Vector3iVector(hand_faces)
    hand_mesh_cur.vertices = o3d.utility.Vector3dVector(hand_verts)
    # hand_mesh_cur.compute_vertex_normals()
    #choosing color
    hand_mesh_cur.vertex_colors = o3d.utility.Vector3dVector(
        np.array([[149.0 / 255.0, 163.0 / 255.0, 166.0 / 255.0]] * hand_verts.shape[0]))
    o3d_obj_pc = o3d.geometry.PointCloud()
    o3d_obj_pc.points = o3d.utility.Vector3dVector(obj_verts)
    o3d_obj_pc.normals = o3d.utility.Vector3dVector(obj_normals)
    # o3d_obj_pc.paint_uniform_color([1 / 255, 95 / 255, 107 / 255])
    if contact_info is not None:
        o3d_obj_pc.colors = o3d.utility.Vector3dVector(create_vertex_color(contact_info, mode="contact_region"))

    hand_points = o3d.geometry.PointCloud()
    hand_points.points = o3d.utility.Vector3dVector(hand_point)
    hand_mesh_points = o3d.geometry.PointCloud()
    hand_mesh_points.points = o3d.utility.Vector3dVector(hand_mesh_point)
    hand_mesh_points.paint_uniform_color([212 / 255, 106 / 255, 126 / 255])

    line_pcd_shadow = o3d.geometry.LineSet()


    vis_cur = o3d.visualization.VisualizerWithKeyCallback()
    vis_cur.create_window(window_name="Runtime ShadowHand", width=1080, height=1080)
    vis_cur.add_geometry(o3d_obj_pc)
    vis_cur.add_geometry(hand_points)
    vis_cur.add_geometry(hand_mesh_points)
    vis_cur.add_geometry(line_pcd_shadow)
    vis_cur.add_geometry(hand_mesh_cur)
    # vis_cur.get_render_option().point_size = 1.2

    vis_cur.poll_events()
    runtime_vis = {
        "hand_mesh": hand_mesh_cur,
        "obj_mesh": o3d_obj_pc,
        "window": vis_cur,
        "hand_point": hand_points,
        "hand_mesh_point": hand_mesh_points,
        "hand_line": line_pcd_shadow,
    }
    # if cam_extr is not None:
    #     ctl = runtime_vis["window"].get_view_control()
    #     parameters = ctl.convert_to_pinhole_camera_parameters()
    #     parameters.extrinsic = cam_extr
    #     ctl.convert_from_pinhole_camera_parameters(parameters)


    return runtime_vis


def update_runtime_viz(runtime_vis, hand_verts_curr):
    runtime_vis["hand_mesh"].vertices = o3d.utility.Vector3dVector(hand_verts_curr)
    runtime_vis["hand_mesh"].compute_vertex_normals()

    runtime_vis["window"].update_geometry(runtime_vis["hand_mesh"])

    runtime_vis["window"].poll_events()
    runtime_vis["window"].update_renderer()
    runtime_vis["window"].reset_view_point(True)
