import os

from options.test_options import TestOptions
from data_grasp import DataLoader
from models import create_model
from utils.writer import Writer
from write_xml_new_data import *
from utils_isa import *
import numpy as np
def save_result(data, predicted_cp, transformed_pts):
    ##data['pc'] data['name']
    for i in range(data['pc'].shape[0]):
        transformed_pt = transformed_pts[i]
        obj_name = data['name'][i].split('/')[1]
        category = data['name'][i].split('/')[0]
        obj_points = data['pc'][i]
        r_o = data['ro'][i]
        grip_r = predicted_cp[i, :4].reshape(4)
        grip_t = predicted_cp[i, 4:7].reshape(3) * 1000.0
        grip_a = predicted_cp[i, 7:].reshape(18)
        result_path = 'test_result_sim'
        result_path_cate = os.path.join(result_path, category, obj_name[:-4])
        if not os.path.exists(result_path_cate):
            os.makedirs(result_path_cate)
        instance = data['file'][i]
        print(result_path_cate)
        result_path_xml = os.path.join(result_path_cate, instance+'.xml')
        if torch.isnan(r_o).any():
            r_o = torch.tensor([0,-0.7071,0.7071,0]).cuda()
        write_xml_new_data(category=category, obj_name=obj_name, r=grip_r.clone().cpu().numpy().reshape(4),
                           r_o=r_o.clone().cpu().numpy().reshape(4), t=grip_t.clone().cpu().numpy(),
                           a=grip_a.clone().cpu().numpy(),
                           path=result_path_xml,
                           mode='train', rs=(21, 'real'))
        new_name_pkl = os.path.join(result_path_cate, instance + '.pkl')
        graspparts = np.array([[1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0]]).repeat(obj_points.shape[0], 0)
        new_j_p = np.zeros([28,3])
        save_pcd_pkl(obj_points.detach().cpu().numpy(), graspparts, str(obj_name),
                             new_j_p, 0, 0, new_name_pkl,
                             transformed_pt.clone().detach().cpu().numpy())



def run_test(epoch=-1, name=""):
    print('Running Test')
    opt = TestOptions().parse()
    opt.serial_batches = True  # no shuffle
    opt.num_threads = 0
    opt.name = ""
    dataset = DataLoader(opt)
    model = create_model(opt)
    writer = Writer(opt)
    # test
    writer.reset_counter()

    for i, data in enumerate(dataset):
        model.set_input(data)
        
        reconstruction_pc_loss, predicted_cp, transformed_pts = model.test(mode='test')
        print('reconstructionpc_loss', reconstruction_pc_loss)

        #######save the result#########
        save_result(data, predicted_cp, transformed_pts)


if __name__ == '__main__':
    run_test()
