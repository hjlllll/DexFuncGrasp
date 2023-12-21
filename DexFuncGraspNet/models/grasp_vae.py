import torch
import networks
from options.train_options import TrainOptions
from options.train_options import BaseOptions
opt = TrainOptions().parse()
# print(opt)
gpu_ids = opt.gpu_ids
print(gpu_ids)
device = torch.device('cuda:{}'.format(
            gpu_ids[0]))
net = networks.define_classifier(opt, gpu_ids, opt.arch,
                                              opt.init_type, opt.init_gain,
                                              device)
pcs = torch.randn(8,446,3).to(device)
z = None
qt, confidence,z = net.module.generate_grasps(pcs, z=z)
print(qt.shape,confidence,z.shape)