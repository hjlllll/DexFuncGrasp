# ----------------------------------------------
# Written by Kailin Li (kailinli@sjtu.edu.cn)
# ----------------------------------------------
import glob
import hashlib
import json
import os
import pickle
import re

import numpy as np
import open3d as o3d
import torch
import trimesh
# from liegroups import SO3
# from manotorch.manolayer import ManoLayer, MANOOutput
# from manotorch.utils.anchorutils import anchor_load_driver, recover_anchor
# from manotorch.utils.quatutils import quaternion_to_angle_axis
from termcolor import cprint
from trimesh.base import Trimesh

from contact_utils import cal_dist, process_contact_info
from vis_contact_info import open3d_show

mesh_path_matcher = re.compile(r"interpolate/.+-.+/interp[0-9]{2}.ply$")


def load_pointcloud(mesh_path: str, sample=5000):
    match_res = mesh_path_matcher.findall(mesh_path)
    pc_cache = mesh_path.replace("interpolate/", "interpolate_pc_cache/").replace(".ply", f"_pc{sample}.pkl")
    if len(match_res) > 0:
        # is interp ply
        if os.path.exists(pc_cache):
            print(f"load obj point cloud from {pc_cache}")
            pc_data = pickle.load(open(pc_cache, "rb"))
            obj_pc = o3d.geometry.PointCloud()
            obj_pc.points = o3d.utility.Vector3dVector(pc_data["points"])
            obj_pc.normals = o3d.utility.Vector3dVector(pc_data["normals"])
            return obj_pc
    mesh = trimesh.load(mesh_path, process=False, force="mesh", skip_materials=True)

    bbox_center = (mesh.vertices.min(0) + mesh.vertices.max(0)) / 2
    mesh.vertices = mesh.vertices - bbox_center

    o3d_obj_mesh = o3d.geometry.TriangleMesh()
    o3d_obj_mesh.triangles = o3d.utility.Vector3iVector(mesh.faces)
    o3d_obj_mesh.vertices = o3d.utility.Vector3dVector(mesh.vertices)
    o3d_obj_mesh.compute_triangle_normals()
    o3d_obj_mesh.compute_vertex_normals()
    obj_pc = o3d_obj_mesh.sample_points_poisson_disk(sample, seed=0)
    if len(match_res) > 0:
        print(f"dump obj point cloud to {pc_cache}")
        os.makedirs(os.path.split(pc_cache)[0], exist_ok=True)
        pickle.dump({"points": np.asarray(obj_pc.points), "normals": np.asarray(obj_pc.normals)}, open(pc_cache, "wb"))
    return obj_pc


def to_pointcloud(mesh, sample=5000):
    o3d_obj_mesh = o3d.geometry.TriangleMesh()
    o3d_obj_mesh.triangles = o3d.utility.Vector3iVector(mesh.faces)
    o3d_obj_mesh.vertices = o3d.utility.Vector3dVector(mesh.vertices)
    o3d_obj_mesh.compute_triangle_normals()
    o3d_obj_mesh.compute_vertex_normals()
    # return o3d_obj_mesh.sample_points_poisson_disk(sample, init_factor=10) #open3d 0.16.0
    return o3d_obj_mesh.sample_points_poisson_disk(sample, seed=0) #open3d 0.15.2


def cal_contact_info(
    hand_verts,
    hand_faces,
    obj_verts,
):
    hand_palm_vertex_index = np.loadtxt(os.path.join("./assets/hand_palm_full.txt"), dtype=np.int32)
    face_vertex_index, anchor_weight, merged_vertex_assignment, anchor_mapping = anchor_load_driver("./assets")
    n_regions = len(np.unique(merged_vertex_assignment))

    hand_mesh = o3d.geometry.TriangleMesh()
    hand_mesh.triangles = o3d.utility.Vector3iVector(hand_faces)
    hand_mesh.vertices = o3d.utility.Vector3dVector(hand_verts)
    hand_mesh.compute_triangle_normals()
    hand_mesh.compute_vertex_normals()
    hand_normals = np.asarray(hand_mesh.vertex_normals)

    hand_verts_selected = hand_verts[hand_palm_vertex_index]
    hand_normals_selected = hand_normals[hand_palm_vertex_index]
    merged_vertex_assignment_selected = merged_vertex_assignment[hand_palm_vertex_index]

    anchor_pos = recover_anchor(hand_verts, face_vertex_index, anchor_weight)
    contact_info_list = cal_dist(
        hand_verts=hand_verts_selected,
        hand_normals=hand_normals_selected,
        obj_verts=obj_verts,
        hand_verts_region=merged_vertex_assignment_selected,
        n_regions=n_regions,
        anchor_pos=anchor_pos,
        anchor_mapping=anchor_mapping,
    )
    vertex_contact, hand_region, anchor_id, anchor_dist, anchor_elasti, anchor_padding_mask = process_contact_info(
        contact_info_list,
        anchor_mapping,
        pad_vertex=True,
        pad_anchor=True,
        dist_th=1000.0,
        elasti_th=0.0,
    )
    return {
        "vertex_contact": vertex_contact,
        "hand_region": hand_region,
        "anchor_id": anchor_id,
        "anchor_dist": anchor_dist,
        "anchor_elasti": anchor_elasti,
        "anchor_padding_mask": anchor_padding_mask,
    }



def get_hand_parameter(pose_path):

    pose = pickle.load(open(pose_path, "rb"))
    hand_pose, hand_shape, hand_tsl = pose["hand_pose"], pose["hand_shape"].numpy(), pose["hand_tsl"].numpy()
    hand_pose = quaternion_to_angle_axis(hand_pose.reshape(16, 4)).reshape(48).numpy()
    obj_rot, obj_tsl = pose["obj_transf"][:3, :3].numpy(), pose["obj_transf"][:3, 3].T.numpy()

    hand_gr = SO3.exp(hand_pose[:3]).as_matrix()
    hand_gr = obj_rot.T @ hand_gr
    hand_gr = SO3.log(SO3.from_matrix(hand_gr, normalize=True))
    hand_pose[:3] = hand_gr
    hand_tsl = obj_rot.T @ (hand_tsl - obj_tsl)

    return hand_pose, hand_shape, hand_tsl


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", "-d", type=str, required=True)
    parser.add_argument("--source", "-s", type=str, required=True)
    parser.add_argument("--tag", "-t", type=str, default="debug")
    parser.add_argument("--pose_path", "-p", type=str)  # grasp pose path
    parser.add_argument("--vis", action="store_true")
    args = parser.parse_args()

    #reading object mesh
    real_meta = json.load(open("./DeepSDF_OakInk/data/meta/object_id.json", "r"))
    virtual_meta = json.load(open("./DeepSDF_OakInk/data/meta/virtual_object_id.json", "r"))

    if args.source in real_meta:
        obj_name = real_meta[args.source]["name"]
        obj_path = "DeepSDF_OakInk/data/OakInkObjects"
    else:
        obj_name = virtual_meta[args.source]["name"]
        obj_path = "DeepSDF_OakInk/data/OakInkVirtualObjects"
    obj_mesh_path = glob.glob(os.path.join(obj_path, obj_name, "align_ds", "*.obj")) + glob.glob(
        os.path.join(obj_path, obj_name, "align_ds", "*.ply")
    )
    assert len(obj_mesh_path) == 1
    obj_pc = load_pointcloud(obj_mesh_path[0])

    hand_pose, hand_shape, hand_tsl = get_hand_parameter(args.pose_path)
    hash_hand = hashlib.md5(pickle.dumps(np.concatenate([hand_pose, hand_shape, hand_tsl]))).hexdigest()

    contactinfo_path = os.path.join(args.data, "contact", f"{args.source}", f"{args.tag}_{hash_hand[:10]}")
    os.makedirs(contactinfo_path, exist_ok=True)

    # if os.path.exists(os.path.join(contactinfo_path, "contact_info.pkl")):
    #     cprint(f"{contactinfo_path} exists, skip.", "yellow")
    #     exit(0)

    pickle.dump(
        {"pose": hand_pose, "shape": hand_shape, "tsl": hand_tsl},
        open(os.path.join(contactinfo_path, "hand_param.pkl"), "wb"),
    )

    with open(os.path.join(contactinfo_path, "source.txt"), "w") as f:
        f.write(args.pose_path)

    mano_layer = ManoLayer(center_idx=0, mano_assets_root="assets/mano_v1_2")
    mano_output: MANOOutput = mano_layer(
        torch.from_numpy(hand_pose).unsqueeze(0), torch.from_numpy(hand_shape).unsqueeze(0)
    )
    hand_faces = mano_layer.th_faces.numpy()
    contact_info = cal_contact_info(
        mano_output.verts.squeeze().numpy() + hand_tsl[None], hand_faces, np.asarray(obj_pc.points)
    )
    pickle.dump(contact_info, open(os.path.join(contactinfo_path, "contact_info.pkl"), "wb"))

    if args.vis:
        open3d_show(
            obj_verts=obj_pc.points,
            obj_normals=obj_pc.normals,
            contact_info=contact_info,
            hand_verts=mano_output.verts.squeeze().numpy() + hand_tsl[None],
            hand_faces=hand_faces,
            show_hand_normals=True,
        )
