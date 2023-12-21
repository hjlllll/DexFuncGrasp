#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

import glob
import logging
import os
import random

import numpy as np
import torch
import torch.utils.data

import deep_sdf.workspace as ws
from scipy.io import loadmat


def get_instance_filenames(data_source, split):
    # npzfiles = []
    # for sid in split:
    #     obj_ids = {}
    #     if "real" in split[sid]:
    #         obj_ids.update({oid: True for oid in split[sid]["real"]})
    #     if "virtual" in split[sid]:
    #         obj_ids.update({oid: False for oid in split[sid]["virtual"]})
    #
    #     for oid, is_real in obj_ids.items():
    #         instance_filename = os.path.join(oid + ".npz")
    #         if not os.path.isfile(os.path.join(data_source, ws.sdf_samples_subdir, instance_filename)):
    #             raise RuntimeError('Requested non-existent file "' + instance_filename + "'")
    #         npzfiles += [instance_filename]
    # return npzfiles

    matfiles = []
    for sid in split:
        obj_ids = {}
        if "real" in split[sid]:
            obj_ids.update({oid: True for oid in split[sid]["real"]})
        if "virtual" in split[sid]:
            obj_ids.update({oid: False for oid in split[sid]["virtual"]})

        for oid, is_real in obj_ids.items():
            instance_filename = os.path.join(oid + ".obj.mat")
            if not os.path.isfile(os.path.join(data_source.split('/split')[0], ws.sdf_samples_subdir, instance_filename)):
                raise RuntimeError('Requested non-existent file "' + instance_filename + "'")
            matfiles += [instance_filename]
    return matfiles


class NoMeshFileError(RuntimeError):
    """Raised when a mesh file is not found in a shape directory"""

    pass


class MultipleMeshFileError(RuntimeError):
    """ "Raised when a there a multiple mesh files in a shape directory"""

    pass


def find_mesh_in_directory(shape_dir):
    mesh_filenames = (
        list(glob.iglob(shape_dir + "/**/*.obj"))
        + list(glob.iglob(shape_dir + "/*.obj"))
        + list(glob.iglob(shape_dir + "/**/*.ply"))
        + list(glob.iglob(shape_dir + "/*.ply"))
    )
    if len(mesh_filenames) == 0:
        raise NoMeshFileError()
    elif len(mesh_filenames) > 1:
        mesh_filenames = [name for name in mesh_filenames if "_align" in name]
        if len(mesh_filenames) > 1:
            raise MultipleMeshFileError()
    return mesh_filenames[0]


def remove_nans(tensor):
    tensor_nan = torch.isnan(tensor[:, 3])
    return tensor[~tensor_nan, :]


def read_sdf_samples_into_ram(filename):
    # npz = np.load(filename)
    # pos_tensor = torch.from_numpy(npz["pos"])
    # neg_tensor = torch.from_numpy(npz["neg"])
    sdf_ob = loadmat(filename)
    sdf_obj = sdf_ob['p_sdf']

    pos_tensor = remove_nans(torch.from_numpy(sdf_obj[np.where(sdf_obj[:, 3] > 0)[0], :]))
    neg_tensor = remove_nans(torch.from_numpy(sdf_obj[np.where(sdf_obj[:, 3] < 0)[0], :]))

    return [pos_tensor, neg_tensor]


def unpack_sdf_samples(filename, subsample=None):
    sdf_ob = loadmat(filename)
    sdf_obj = sdf_ob['p_sdf']
    # npz = np.load(filename)
    if subsample is None:
        # return npz
        return sdf_obj
    # pos_tensor = remove_nans(torch.from_numpy(npz["pos"]))
    # neg_tensor = remove_nans(torch.from_numpy(npz["neg"]))
    pos_tensor = remove_nans(torch.from_numpy(sdf_obj[np.where(sdf_obj[:,3]>0)[0],:]))
    neg_tensor = remove_nans(torch.from_numpy(sdf_obj[np.where(sdf_obj[:,3]<0)[0],:]))

    # split the sample into half
    half = int(subsample / 2)

    random_pos = (torch.rand(half) * pos_tensor.shape[0]).long()
    random_neg = (torch.rand(half) * neg_tensor.shape[0]).long()

    sample_pos = torch.index_select(pos_tensor, 0, random_pos)
    sample_neg = torch.index_select(neg_tensor, 0, random_neg)

    samples = torch.cat([sample_pos, sample_neg], 0)

    return samples


def unpack_sdf_samples_from_ram(data, subsample=None):
    if subsample is None:
        return data
    pos_tensor = data[0]
    neg_tensor = data[1]

    # split the sample into half
    half = int(subsample / 2)

    pos_size = pos_tensor.shape[0]
    neg_size = neg_tensor.shape[0]

    pos_start_ind = random.randint(0, pos_size - half)
    sample_pos = pos_tensor[pos_start_ind : (pos_start_ind + half)]

    if neg_size <= half:
        random_neg = (torch.rand(half) * neg_tensor.shape[0]).long()
        sample_neg = torch.index_select(neg_tensor, 0, random_neg)
    else:
        neg_start_ind = random.randint(0, neg_size - half)
        sample_neg = neg_tensor[neg_start_ind : (neg_start_ind + half)]

    samples = torch.cat([sample_pos, sample_neg], 0)

    return samples


class SDFSamples(torch.utils.data.Dataset):
    def __init__(
        self,
        data_source,
        split,
        subsample,
        load_ram=False,
        print_filename=False,
        num_files=1000000,
    ):
        self.subsample = subsample

        self.data_source = data_source
        self.npyfiles = get_instance_filenames(data_source, split)

        logging.info("using " + str(len(self.npyfiles)) + " shapes from data source " + data_source)

        self.load_ram = load_ram

        if load_ram:
            self.loaded_data = []
            for f in self.npyfiles:
                filename = os.path.join(self.data_source.split('/split')[0], ws.sdf_samples_subdir, f)
                npz = np.load(filename)
                # sdf_ob = loadmat(filename)
                # sdf_obj = sdf_ob['p_sdf']
                # print(sdf_obj)

                pos_tensor = remove_nans(torch.from_numpy(npz["pos"]))
                neg_tensor = remove_nans(torch.from_numpy(npz["neg"]))
                self.loaded_data.append(
                    [
                        pos_tensor[torch.randperm(pos_tensor.shape[0])],
                        neg_tensor[torch.randperm(neg_tensor.shape[0])],
                    ]
                )

    def __len__(self):
        return len(self.npyfiles)

    def __getitem__(self, idx):
        filename = os.path.join(self.data_source.split('/split')[0], ws.sdf_samples_subdir, self.npyfiles[idx])
        if self.load_ram:
            return (
                unpack_sdf_samples_from_ram(self.loaded_data[idx], self.subsample),
                idx,
            )
        else:
            return unpack_sdf_samples(filename, self.subsample), idx
