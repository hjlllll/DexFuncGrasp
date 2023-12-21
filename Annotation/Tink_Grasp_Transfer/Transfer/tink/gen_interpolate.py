# ----------------------------------------------
# Written by Kailin Li (kailinli@sjtu.edu.cn)
# ----------------------------------------------
import json
import os
import sys

sys.path.append("../Transfer")

import deep_sdf.workspace as ws
import numpy as np
import torch
from deep_sdf.mesh import create_mesh
from tqdm import tqdm
from termcolor import cprint
from scipy.io import loadmat

if __name__ == "__main__":


    import argparse

    parser = argparse.ArgumentParser()
    # parser.add_argument("--data", "-d", type=str, required=True)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--source", "-s", type=str, required=False)
    parser.add_argument("--target", "-t", type=str, required=False)
    parser.add_argument("--interpolate", "-i", default=10, type=int)
    parser.add_argument("--resume", type=str, default="latest")
    parser.add_argument('--idx', type=int, default=0, help="0-20 id of hjl new dataset, means category")
    data_root = '../Dataset/Obj_Data'

    arg = parser.parse_args()
    f = open('../Dataset/Obj_Data/obj_data.json', 'r')
    content = json.loads(f.read())
    category = content[str(arg.idx)]['category']
    data = os.path.join(data_root, category, 'split')

    # write split.json
    obj_path = os.path.join(data_root, category, 'obj')
    dicts = {}
    # for category in categories:
    obj_list = os.listdir(obj_path)
    source = os.listdir('../Dataset/Grasps')
    real_list = []
    for file in source:
        if file.split("_")[0] == str(arg.idx):
            real_list.append(file)
    virtual_list = list(set(obj_list) - set(real_list))

    virtual_instance_list = []
    for name in virtual_list:
        virtual_instance_list.append(name[:-4])



    real_instance_list = []
    for name in real_list:
        real_instance_list.append(name[len(str(arg.idx))+1:])
    new_virtual_list = list(set(virtual_instance_list) - set(real_instance_list))

    obj_dict = {'real': real_instance_list,
                'virtual': new_virtual_list}
    dicts[str(arg.idx)] = obj_dict
    d = json.dumps(dicts, indent=1)
    if not os.path.exists(os.path.join(data_root, category, 'split')):
        os.makedirs(os.path.join(data_root, category, 'split'))
    f = open(os.path.join(data_root, category, 'split', 'split_interpolate.json'), 'w')
    f.write(d)
    f.close()

    assert arg.all or (arg.source and arg.target)

    handle_list = []

    if arg.all:
        split = json.load(open(os.path.join(data, "split_interpolate.json"), "r"))
        for cid in split.keys():
            real_ids = split[cid]["real"]
            virtual_ids = split[cid]["virtual"]
            for r_oid in real_ids:
                for v_oid in virtual_ids:
                    handle_list.append((r_oid, v_oid))
    else:
        handle_list.append((arg.source, arg.target))

    print(len(handle_list))

    for idx, (soruce_idx, target_idx) in enumerate(handle_list):

        cprint(f"handling {soruce_idx} -> {target_idx}", "blue")

        specs = json.load(open(os.path.join(data, "specs.json"), "r"))
        arch = __import__("networks." + specs["NetworkArch"], fromlist=["Decoder"])
        latent_size = specs["CodeLength"]
        decoder = arch.Decoder(latent_size, **specs["NetworkSpecs"])
        decoder = torch.nn.DataParallel(decoder)
        saved_model_state = torch.load(os.path.join(data, "network", ws.model_params_subdir, arg.resume + ".pth"))
        saved_model_epoch = saved_model_state["epoch"]
        decoder.load_state_dict(saved_model_state["model_state_dict"])

        decoder = decoder.module.cuda()

        sdf_code_path = os.path.join(data, ws.reconstructions_subdir, ws.reconstruction_codes_subdir)
        sdf_mesh_path = os.path.join(data, ws.reconstructions_subdir, ws.reconstruction_meshes_subdir)
        source = f"{soruce_idx}.obj.pth"
        target = f"{target_idx}.obj.pth"
        interpolate_path = os.path.join(data, "interpolate", f"{soruce_idx}-{target_idx}")
        if os.path.exists(interpolate_path) and len(os.listdir(interpolate_path)) == arg.interpolate:
            continue

        os.makedirs(interpolate_path, exist_ok=True)

        k = arg.interpolate

        source_id = soruce_idx + '.obj.mat'
        target_id = target_idx + '.obj.mat'
        filename_source = os.path.join(data.split('/split')[0], ws.sdf_samples_subdir, source_id)
        filename_target = os.path.join(data.split('/split')[0], ws.sdf_samples_subdir, target_idx)
        sdf_scale1 = loadmat(filename_source)
        scale1 = sdf_scale1['scale']
        sdf_scale2 = loadmat(filename_source)
        scale2 = sdf_scale2['scale']
        scale = (scale1 + scale2) / 2
        with torch.no_grad():
            latent_source = torch.load(os.path.join(sdf_code_path, source))
            latent_target = torch.load(os.path.join(sdf_code_path, target))
            if not os.path.isfile(os.path.join(sdf_mesh_path, source.replace(".pth", ".ply"))):
                create_mesh(
                    decoder,
                    latent_source[0].type(torch.FloatTensor).cuda(),
                    os.path.join(sdf_mesh_path, source),
                    N=256,
                    max_batch=int(2 ** 18),
                    scale=scale,
                )
            if not os.path.isfile(os.path.join(sdf_mesh_path, target.replace(".pth", ".ply"))):
                create_mesh(
                    decoder,
                    latent_target[0].type(torch.FloatTensor).cuda(),
                    os.path.join(sdf_mesh_path, target),
                    N=256,
                    max_batch=int(2 ** 18),
                    scale=scale,
                )

            latent_source = latent_source.cpu().numpy()[0][0]
            latent_target = latent_target.cpu().numpy()[0][0]

        inter_list = []
        for i in range(256):
            y = np.array([latent_source[i], latent_target[i]])
            x = np.array([0, k + 1])
            xl = np.arange(1, k + 1)
            interp = np.interp(xl, x, y)
            inter_list.append(interp)
        interpolation = np.vstack(inter_list)

        with torch.no_grad():
            for i in range(k):
                interp_obj = os.path.join(interpolate_path, f"interp{i + 1:02d}")

                latent = torch.from_numpy(interpolation[:, i].T)
                create_mesh(
                    decoder,
                    latent.type(torch.FloatTensor).cuda(),
                    interp_obj,
                    N=256,
                    max_batch=int(2 ** 18),
                    scale=scale,
                )
