import torch
import pickle

def trans_2_quat(R):
    quat = torch.zeros(1, 4).cuda()
    quat[0][3] = 0.5 * ((1 + R[0][0][0] + R[0][1][1] + R[0][2][2]).sqrt())
    quat[0][0] = (R[0][2][1] - R[0][1][2]) / (4 * quat[0][0])
    quat[0][1] = (R[0][0][2] - R[0][2][0]) / (4 * quat[0][0])
    quat[0][3] = (R[0][1][0] - R[0][0][1]) / (4 * quat[0][0])
    return quat


def save_pcd_pkl(points, graspparts, obj_name, j_p, obj_index, num, path_pkl, shadowhandmesh_points):
    '''
    save pkl and visualize by open3d with pcd file
    '''
    # path_pcd_pkl = 'Refine_Pose'
    # path_pcd_pkl = 'Grasp_Pose'
    path_pcd_pkl = path_pkl
    with open(path_pcd_pkl, 'wb') as file:
        pickle.dump((points,graspparts,obj_name,j_p,num,shadowhandmesh_points),file)

