import  os
import  sys
# 打印文件绝对路径（absolute path）
print (os.path.abspath(__file__))                                                          
# 打印文件的目录路径（文件的上一层目录），这个时候是在 bin 这一层。
print (os.path.dirname( os.path.abspath(__file__) ))  
print (os.path.dirname(os.path.dirname( os.path.abspath(__file__) )))  
BASE_DIR=  os.path.dirname(os.path.dirname( os.path.abspath(__file__) ))                   
# 将这个路径添加到环境变量中。
sys.path.append( BASE_DIR  ) 

import json
from HandModel import HandModel


def get_handmodel(robot, batch_size, device, hand_scale=1.):
    urdf_assets_meta = json.load(open("/media/hjl/Samsung_T5/song/a_2022_summer/dataset_generater/dexterous_hands/data/urdf/urdf_assets_meta.json"))
    urdf_path = urdf_assets_meta['urdf_path'][robot]
    meshes_path = urdf_assets_meta['meshes_path'][robot]
    hand_model = HandModel(robot, urdf_path, meshes_path, batch_size=batch_size, device=device, hand_scale=hand_scale)
    return hand_model


if __name__ == '__main__':
    import torch
    from utils_hand.visualize_plotly import plot_point_cloud
    from plotly import graph_objects as go
    # robot_name = 'ezgripper'
    # robot_name = 'shadowhand'  # shadowhand dof #24
    robot_name = 'hithand'
    # robot_name = 'allegro'  # barrett dof #8 allegro dof#16
    # robot_name = 'robotiq_3finger'
    # robot_name = 'barrett'
    init_opt_q = torch.zeros(1, 7 + 20, device='cuda')
    # init_opt_q = torch.zeros(1, 9 + 11, device='cuda')
    init_opt_q[:, :3] = torch.tensor([0.2, 0.02, -0.3], device='cuda')
    # init_opt_q[:, 3:7] = torch.tensor([1., 0., 0., 0.], device='cuda')
    init_opt_q[:, 3:7] = torch.tensor([0.6628859, 0.21884856, -0.3959573, -0.5965781], device='cuda')

    hand_model = get_handmodel(robot_name, 1, 'cuda', 1.)
    init_opt_q[:, 7:].copy_(hand_model.revolute_joints_q_lower.repeat(1, 1))
    surface_points, surface_normals = hand_model.get_surface_points_and_normals(q=init_opt_q)

    # plot hand mesh and surface points + normals
    vis_data = hand_model.get_plotly_data(q=init_opt_q, color='pink')
    # vis_data.append(plot_point_cloud(pts=surface_points.cpu().squeeze(0), color='red'))
    # for i in range(10):
    #     vis_data.append(plot_point_cloud(pts=(surface_points + 0.001 * i * surface_normals).cpu().squeeze(0), color='blue'))

    fig = go.Figure(data=vis_data)
    fig.show()
