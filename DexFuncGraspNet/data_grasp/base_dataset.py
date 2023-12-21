import torch.utils.data as data
import numpy as np
import os


class NoPositiveGraspsException(Exception):
    """raised when there's no positive grasps for an object."""
    pass


class BaseDataset(data.Dataset):
    def __init__(self,
                 opt):
        super(BaseDataset, self).__init__()
        self.opt = opt
    

    def make_dataset(self,opt): # 

        if opt.is_train == False:
            grasp_path = 'Grasps_Dataset/test'
        else:
            grasp_path = 'Grasps_Dataset/train'
        ofiles = os.listdir(grasp_path)
        npyfile = []
        name = []
        for ofile in ofiles:
            grasp_paths = str(os.path.join(grasp_path, ofile))
            my_files = os.listdir(grasp_paths)
            for my_file in my_files:
                my_path = str(os.path.join(grasp_paths, my_file))           
                with open(my_path, 'rb') as file:
                    loaded_npy = np.load(file, allow_pickle=True)
                npyfile.append(loaded_npy)
                name.append(my_file)
        return npyfile, name


def collate_fn(batch):
    """Creates mini-batch tensors
    We should build custom collate_fn rather than using default collate_fn
    """
    batch = list(filter(lambda x: x is not None, batch))  #
    meta = {}
    keys = batch[0].keys()
    for key in keys:
        meta.update({key: np.concatenate([d[key] for d in batch])})
    return meta
