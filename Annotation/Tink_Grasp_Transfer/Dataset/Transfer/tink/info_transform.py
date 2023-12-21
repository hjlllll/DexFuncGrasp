# ----------------------------------------------
# Written by Kailin Li (kailinli@sjtu.edu.cn)
# ----------------------------------------------
import glob
import json
import os
import pickle
import re
from ast import parse

import numpy as np
import trimesh
from scipy.spatial.distance import cdist
from termcolor import cprint
from tqdm import tqdm

from cal_contact_info import load_pointcloud, to_pointcloud
from vis_contact_info import open3d_show


def get_obj_path(oid, split_inter, id, data_root, category, use_downsample=True, key="align"):
    # obj_suffix_path = "align_ds" if use_downsample else "align"
    # real_meta = json.load(open("./DeepSDF_OakInk/data/meta/object_id.json", "r"))
    # virtual_meta = json.load(open("./DeepSDF_OakInk/data/meta/virtual_object_id.json", "r"))
    # real_meta = json.load(open('../Dataset/Objects/bottle/split/split.json', "r"))
    # virtual_meta = json.load(open('../Dataset/Objects/bottle/split/split.json', "r"))

    real_meta = json.load(open(split_inter, "r"))
    virtual_meta = json.load(open(split_inter, "r"))

    # if oid in real_meta:
    #     obj_name = real_meta[oid]["name"]
    #     obj_path = "DeepSDF_OakInk/data/OakInkObjects"
    if oid in real_meta[id]['real']:
        # obj_name = real_meta['52']['real'][args.source]
        obj_name = oid

        # obj_path = "DeepSDF_OakInk/data/OakInkObjects"
        # obj_path = '../Dataset/Objects'
    # else:
    if oid in virtual_meta[id]['virtual']:

        obj_name = oid
        # obj_name = virtual_meta[oid]["name"]
        # obj_path = "DeepSDF_OakInk/data/OakInkVirtualObjects"
    obj_mesh_path = os.path.join(data_root, category, 'obj', "{}.obj".format(obj_name))
    # obj_mesh_path = os.path.join(data_root, category, 'split','Reconstructions','Meshes', "{}.obj.ply".format(obj_name))

    # obj_mesh_path = list(
    #     glob.glob(os.path.join(obj_path, obj_name, obj_suffix_path, "*.obj"))
    #     + glob.glob(os.path.join(obj_path, obj_name, obj_suffix_path, "*.ply"))
    # )
    # if len(obj_mesh_path) > 1:
    #     obj_mesh_path = [p for p in obj_mesh_path if key in os.path.split(p)[1]]
    # assert len(obj_mesh_path) == 1
    return obj_mesh_path


def cal_closest_idx(source, target):
    region = np.max([np.abs(source.min()), source.max(), np.abs(target.min()), target.max()])
    k = 6
    cell_size = region * 2 / k
    res = np.zeros(len(target), dtype=int)
    for x in range(k):
        for y in range(k):
            for z in range(k):

                def select_region(d, i):
                    return (target[:, d] > i * cell_size - region - 1e-6) & (
                        target[:, d] < (i + 1) * cell_size - region + 1e-6
                    )

                def search_region(d, i):
                    return (source[:, d] > (i - 1) * cell_size - region - 1e-6) & (
                        source[:, d] < (i + 2) * cell_size - region + 1e-6
                    )

                target_filter = select_region(0, x) & select_region(1, y) & select_region(2, z)
                source_filter = search_region(0, x) & search_region(1, y) & search_region(2, z)
                target_cell = target[target_filter]
                source_cell = source[source_filter]
                if len(target_cell) == 0 or len(source_cell) == 0:
                    continue
                res_cell = np.argwhere(source_filter)[np.argmin(cdist(target_cell, source_cell), axis=1)].ravel()
                res[target_filter] = res_cell
    return res


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    # parser.add_argument("--data", "-d", type=str, required=True)
    # parser.add_argument("--contact_path", "-p", type=str, required=True)
    # parser.add_argument("--source", "-s", type=str, required=True)
    # parser.add_argument("--target", "-t", type=str)
    parser.add_argument('--idx', type=int, default=0, help="0-20 id of hjl new dataset, means category")
    parser.add_argument("--all", action="store_true")

    arg = parser.parse_args()

    data_root = '../Dataset/Obj_Data'
    f = open('../Dataset/Obj_Data/obj_data.json', 'r')
    content = json.loads(f.read())
    category = content[str(arg.idx)]['category']
    data = os.path.join(data_root, category, 'split')
    split_inter = os.path.join(data, "split_interpolate.json")
    handle_list = []
    if arg.all:
        split = json.load(open(split_inter, "r"))
        for cid in split.keys():
            real_ids = split[cid]["real"]
            virtual_ids = split[cid]["virtual"]
            for r_oid in real_ids:
                for v_oid in virtual_ids:
                    handle_list.append((r_oid, v_oid))
    for idx, (soruce_idx, target_instance) in enumerate(handle_list):
        cprint(f"handling grasp info transformation: {soruce_idx} -> {target_instance}", "blue")
        interpolate_path = os.path.join(data, "interpolate", f"{soruce_idx}-{target_instance}")
        contact_path = os.path.join(data_root, category, 'split', 'contact')
        contact_grasps = os.listdir(os.path.join(contact_path, soruce_idx, 'trans'))
        inter_paths = os.listdir(interpolate_path)
        inter_paths.sort()
        interpolate_pc_path = os.path.join(data, "interpolate_pc_cache", f"{soruce_idx}-{target_instance}")
        if os.path.exists(interpolate_pc_path):
            continue

        # inter_paths = os.listdir(interpolate_path)
        # inter_paths.sort()

        inter_pc = [load_pointcloud(os.path.join(interpolate_path, p)) for p in inter_paths]

        # rescale = pickle.load(open(os.path.join(arg.data, "rescale.pkl"), "rb"))
        # rescale = rescale["max_norm"] * rescale["scale"]

        source_mesh = trimesh.load(get_obj_path(soruce_idx, split_inter, str(arg.idx), data_root, category),
                                   process=False, force="mesh", skip_materials=True)
        target_mesh = trimesh.load(get_obj_path(target_instance, split_inter, str(arg.idx), data_root, category),
                                   process=False, force="mesh", skip_materials=True)
        source_mesh.vertices = (
                source_mesh.vertices - (
                np.array(source_mesh.vertices).max(0) + np.array(source_mesh.vertices).min(0)) / 2
        )  # center
        source_mesh.vertices = source_mesh.vertices  # / rescale
        target_mesh.vertices = (
                target_mesh.vertices - (
                np.array(target_mesh.vertices).max(0) + np.array(target_mesh.vertices).min(0)) / 2
        )  # center
        target_mesh.vertices = target_mesh.vertices  # / rescale

        obj_pointclouds = [to_pointcloud(source_mesh)] + inter_pc + [to_pointcloud(target_mesh)]
        for each_grasp_contact in contact_grasps:
            contact_info = pickle.load(open(os.path.join(contact_path, soruce_idx, 'trans', each_grasp_contact), "rb"))
            trans_contact_path = os.path.join(contact_path, target_instance, each_grasp_contact[:-17])
            os.makedirs(trans_contact_path, exist_ok=True)
            # out_path = [f"contact_{os.path.splitext(os.path.split(p)[1])[0]}" for p in inter_paths] + ["contact_info"]#########
            out_path = ["contact_info"]

            for i, (source, target) in tqdm(enumerate(zip(obj_pointclouds[:-1], obj_pointclouds[1:]))):
                target_idx = cal_closest_idx(np.asarray(source.points), np.asarray(target.points))

                res = {
                    "vertex_contact": contact_info["vertex_contact"][target_idx],
                    "hand_region": contact_info["hand_region"][target_idx],
                    "anchor_id": contact_info["anchor_id"][target_idx],
                    # "anchor_dist": contact_info["anchor_dist"][target_idx],
                    # "anchor_elasti": contact_info["anchor_elasti"][target_idx],
                    # "anchor_padding_mask": contact_info["anchor_padding_mask"][target_idx],
                }
                contact_info = res
                # if os.path.exists(os.path.join(trans_contact_path, f"{out_path[i]}.pkl")):###########
                #     print('already exists',os.path.join(trans_contact_path, f"{out_path[i]}.pkl"))
                #     continue
                # pickle.dump(contact_info, open(os.path.join(trans_contact_path, f"{out_path[i]}.pkl"), "wb"))########
            pickle.dump(contact_info, open(os.path.join(trans_contact_path, f"{out_path[0]}.pkl"), "wb"))


    # interpolate_path = os.path.join(data, "interpolate", f"{arg.source}-{arg.target}")
    # contact_info = pickle.load(open(os.path.join(arg.contact_path, "contact_info.pkl"), "rb"))

    # trans_contact_path = os.path.join(arg.contact_path, arg.target)
    # os.makedirs(trans_contact_path, exist_ok=True)

    # if os.path.exists(os.path.join(trans_contact_path, "contact_info.pkl")):
    #     cprint(f"{os.path.join(trans_contact_path, 'contact_info.pkl')} exists, skip.", "yellow")
    #     exit(0)

    # inter_paths = os.listdir(interpolate_path)
    # inter_paths.sort()
    #
    # inter_pc = [load_pointcloud(os.path.join(interpolate_path, p)) for p in inter_paths]
    #
    # # rescale = pickle.load(open(os.path.join(arg.data, "rescale.pkl"), "rb"))
    # # rescale = rescale["max_norm"] * rescale["scale"]
    #
    # source_mesh = trimesh.load(get_obj_path(arg.source), process=False, force="mesh", skip_materials=True)
    # target_mesh = trimesh.load(get_obj_path(arg.target), process=False, force="mesh", skip_materials=True)
    # source_mesh.vertices = (
    #     source_mesh.vertices - (np.array(source_mesh.vertices).max(0) + np.array(source_mesh.vertices).min(0)) / 2
    # )  # center
    # source_mesh.vertices = source_mesh.vertices #/ rescale
    # target_mesh.vertices = (
    #     target_mesh.vertices - (np.array(target_mesh.vertices).max(0) + np.array(target_mesh.vertices).min(0)) / 2
    # )  # center
    # target_mesh.vertices = target_mesh.vertices #/ rescale
    #
    # obj_pointclouds = [to_pointcloud(source_mesh)] + inter_pc + [to_pointcloud(target_mesh)]
    #
    # out_path = [f"contact_{os.path.splitext(os.path.split(p)[1])[0]}" for p in inter_paths] + ["contact_info"]
    #
    # for i, (source, target) in tqdm(enumerate(zip(obj_pointclouds[:-1], obj_pointclouds[1:]))):
    #
    #     target_idx = cal_closest_idx(np.asarray(source.points), np.asarray(target.points))
    #
    #     res = {
    #         "vertex_contact": contact_info["vertex_contact"][target_idx],
    #         "hand_region": contact_info["hand_region"][target_idx],
    #         "anchor_id": contact_info["anchor_id"][target_idx],
    #         # "anchor_dist": contact_info["anchor_dist"][target_idx],
    #         # "anchor_elasti": contact_info["anchor_elasti"][target_idx],
    #         # "anchor_padding_mask": contact_info["anchor_padding_mask"][target_idx],
    #     }
    #     contact_info = res
    #     pickle.dump(contact_info, open(os.path.join(trans_contact_path, f"{out_path[i]}.pkl"), "wb"))
