import torch.utils.data
from data_grasp.base_dataset import collate_fn


def CreateDataset(opt):
    """loads datasets class"""
    if opt.arch == 'vae' or opt.arch == 'gan':
        from data_grasp.grasp_sampling_data import GraspSamplingData
        dataset = GraspSamplingData(opt)
    # else:
    #     from data.grasp_evaluator_data import GraspEvaluatorData
    #     dataset = GraspEvaluatorData(opt)
    return dataset


class DataLoader:
    """multi-threaded data loading"""
    def __init__(self, opt):
        self.opt = opt
        self.dataset = CreateDataset(opt)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=opt.num_grasps_per_object,
            # batch_size=1,
            shuffle=not opt.serial_batches,
            # shuffle=False,
            drop_last= False, ##when test ---> false
            num_workers=int(opt.num_threads),
            )
            # collate_fn=collate_fn)

    def __len__(self):
        return min(len(self.dataset), self.opt.max_dataset_size)

    def __iter__(self):
        for i, data in enumerate(self.dataloader):
            if i * self.opt.batch_size >= self.opt.max_dataset_size:
                break
            yield data

if __name__ == '__main__':
    from options.train_options import TrainOptions
    from options.test_options import TestOptions
    opt = TrainOptions().parse()
    data = DataLoader(opt=opt)
    train_loader = data.dataloader
    import time
    for i, batch in enumerate(train_loader):
        print(batch['pc'].shape)