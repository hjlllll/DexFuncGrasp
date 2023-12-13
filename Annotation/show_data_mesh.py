'''
saving pcd
x y z qw qx qy qz: 7  /  rotations: 18
points of shadow
one object one directory
'''

import pickle
import os
import open3d as o3d
import numpy as np
import random
import time

def save_pcd_pkl_and_mesh(points, graspparts, obj_name, interpolate, obj_index, num, path_pkl, interpolate_shadowhandmesh):
    '''
    save pkl and visualize by open3d with pcd file
    '''
    # path_pcd_pkl = 'Refine_Pose'
    # path_pcd_pkl = 'Grasp_Pose'
    path_pcd_pkl = path_pkl
    path = path_pcd_pkl + '/{}_{}/{}_{}.pkl'.format(obj_index, str(obj_name), obj_index, num)
    with open(path, 'wb') as file:
        pickle.dump((points,graspparts,obj_name,interpolate,num,interpolate_shadowhandmesh),file)
        # print('saving pcd_pkl in directory...')

def save_pcd_pkl(points, graspparts, obj_name, j_p, obj_index, num, path_pkl):
    '''
    save pkl and visualize by open3d with pcd file
    '''
    # path_pcd_pkl = 'Refine_Pose'
    # path_pcd_pkl = 'Grasp_Pose'
    path_pcd_pkl = path_pkl
    path = path_pcd_pkl + '/{}_{}/{}_{}.pkl'.format(obj_index, str(obj_name), obj_index, num)
    with open(path, 'wb') as file:
        pickle.dump((points,graspparts,obj_name,j_p, num),file)
        # print('saving pcd_pkl in directory...')


def visualize_contact_points(points, graspparts,j_p):
    col = [[1, 1, 1, 0.8],
           [0, 1, 0, 0.8],
           [0, 0, 1, 0.8],
           [1, 1, 0, 0.8],
           [1, 0, 1, 0.8],
           [0, 1, 1, 0.8],
           [1, 0, 0, 0.8],
           [0.5, 0.5, 0, 0.8],
           [0.6, 0.3, 0.3, 0.8],
           [1, 0, 0, 0.8],
           [1, 0, 0, 0.8],
           [1, 0, 0, 0.8],
           [1, 0, 0, 0.8]]
    index_ = [0, 3, 6, 9, 12]  # finger tip index
    grasppart = graspparts[:, np.array(index_)]
    th = np.where(grasppart[:, 0] == 1)
    ff = np.where(grasppart[:, 1] == 1)
    mf = np.where(grasppart[:, 2] == 1)
    rf = np.where(grasppart[:, 3] == 1)
    lf = np.where(grasppart[:, 4] == 1)
    th_pt = points[np.array(th)]  # [1,n1,3]
    ff_pt = points[np.array(ff)]  # [1,n2,3]
    mf_pt = points[np.array(mf)]
    rf_pt = points[np.array(rf)]
    lf_pt = points[np.array(lf)]
    index_5fingers = [6, 11, 16, 21, 27]
    fingers = j_p[np.array(index_5fingers)]  # [5, 3]
    fi_th = np.expand_dims(fingers[4].reshape(1, 3), 1).repeat(th_pt.shape[1], 1)
    fi_ff = np.expand_dims(fingers[3].reshape(1, 3), 1).repeat(ff_pt.shape[1], 1)
    fi_mf = np.expand_dims(fingers[2].reshape(1, 3), 1).repeat(mf_pt.shape[1], 1)
    fi_rf = np.expand_dims(fingers[1].reshape(1, 3), 1).repeat(rf_pt.shape[1], 1)
    fi_lf = np.expand_dims(fingers[0].reshape(1, 3), 1).repeat(lf_pt.shape[1], 1)
    # 选出大拇指最近的点集
    dis = 15
    contact_th = th_pt.squeeze(0)[
        np.array(np.where(np.sqrt(np.sum((th_pt - fi_th) ** 2, axis=2)).squeeze() <= dis))].squeeze(0)
    contact_ff = ff_pt.squeeze(0)[
        np.array(np.where(np.sqrt(np.sum((ff_pt - fi_ff) ** 2, axis=2)).squeeze() <= dis))].squeeze(0)
    contact_mf = mf_pt.squeeze(0)[
        np.array(np.where(np.sqrt(np.sum((mf_pt - fi_mf) ** 2, axis=2)).squeeze() <= dis))].squeeze(0)
    contact_rf = rf_pt.squeeze(0)[
        np.array(np.where(np.sqrt(np.sum((rf_pt - fi_rf) ** 2, axis=2)).squeeze() <= dis))].squeeze(0)
    contact_lf = lf_pt.squeeze(0)[
        np.array(np.where(np.sqrt(np.sum((lf_pt - fi_lf) ** 2, axis=2)).squeeze() <= dis))].squeeze(0)
    pcd_finger = [contact_th, contact_ff, contact_mf, contact_rf, contact_lf]
    # contact = []
    # for i in range(5):
    #     pcd_f = o3d.geometry.PointCloud()
    #     pcd_f.points = o3d.utility.Vector3dVector(pcd_finger[i])
    #     color = col[i + 1]
    #     part_color = [x * color[3] for x in color[:3]]
    #     # pcd_f.paint_uniform_color(part_color)
    #     contact.append(pcd_f)
    #     # pcd_f.paint_uniform_color([1, 1, 0])
    # contact_0, contact_1, contact_2, contact_3, contact_4 = \
    #     contact[0], contact[1], contact[2], contact[3], contact[4]
    return pcd_finger #contact_0,contact_1,contact_2,contact_3,contact_4

def visual_grasp_pcd(file_path = 'visual_dict', is_AUTO=False):
    grasp_file = file_path
    dirs = os.listdir(grasp_file)
    for dir in dirs:
        dir = grasp_file + '/' + dir
        files = os.listdir(dir)
        for file in files:
            if file.endswith('.pkl'):
                print(file)
                file = dir + '/' + file
                with open(file, 'rb') as filename:
                    points, graspparts, obj_name, j_p, _,  hand_mesh_points = pickle.load(filename)

                    colpart = np.unique(graspparts, axis=0)

                    col = [[1, 1, 1, 0.8],
                           [0, 1, 0, 0.8],
                           [0, 0, 1, 0.8],
                           [1, 1, 0, 0.8],
                           [1, 0, 1, 0.8],
                           [0, 1, 1, 0.8],
                           [1, 0, 0, 0.8],
                           [0.5, 0.5, 0, 0.8],
                           [0.6, 0.3, 0.3, 0.8],
                           [1, 0, 0, 0.8],
                           [1, 0, 0, 0.8],
                           [1, 0, 0, 0.8],
                           [1, 0, 0, 0.8]]
                    print('colpart:\n', colpart)
                    print('colpart.shape:', colpart.shape)
                    points_list = []
                    viewers = o3d.visualization.Visualizer()
                    viewers.create_window(window_name="vis",width=1000, height=800, left=1400, top=100)
                    for j in range(colpart.shape[0]):
                        ppp = []
                        for i in range(points.shape[0]):
                            if (graspparts[i] == colpart[j]).all():
                                ppp.append(points[i])
                        ppp = np.concatenate(ppp, 0).reshape(-1, 3)
                        points_list.append(ppp)
                    pcds = []
                    for i, ppp in enumerate(points_list):
                        pcd = o3d.geometry.PointCloud()
                        pcd.points = o3d.utility.Vector3dVector(ppp)
                        color = col[i]
                        part_color = [x * color[3] for x in color[:3]]
                        pcd.paint_uniform_color(part_color)
                        pcds.append(pcd)
                        viewers.add_geometry(pcd)
                    line_pcd_shadow = o3d.geometry.LineSet()
                    lines_shadow = [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [0, 7], [7, 8], [8, 9], [9, 10],
                                    [10, 11], [0, 12], [12, 13], [13, 14], [14, 15], [15, 16],
                                    [0, 17], [17, 18], [18, 19], [19, 20], [20, 21], [0, 22], [22, 23], [23, 24],
                                    [24, 25], [25, 26], [26, 27]]
                    line_pcd_shadow.lines = o3d.utility.Vector2iVector(lines_shadow)
                    line_pcd_shadow.points = o3d.utility.Vector3dVector(j_p)
                    colors_shadow = [[0, 0.8, 0], [0, 0.8, 0], [0.8, 0.8, 0], [0.8, 0, 0], [0, 0, 0.8], [0.8, 0, 0.8],
                                     [0, 0.8, 0], [0.8, 0.8, 0], [0.8, 0, 0], [0, 0, 0.8], [0.8, 0, 0.8],
                                     [0, 0.8, 0], [0.8, 0.8, 0], [0.8, 0, 0], [0, 0, 0.8], [0.8, 0, 0.8],
                                     [0, 0.8, 0], [0.8, 0.8, 0], [0.8, 0, 0], [0, 0, 0.8], [0.8, 0, 0.8],
                                     [0, 0.8, 0], [0, 0.8, 0], [0.8, 0.8, 0], [0.8, 0, 0], [0, 0, 0.8], [0.8, 0, 0.8]]
                    line_pcd_shadow.colors = o3d.utility.Vector3dVector(colors_shadow)
                    pcds.append(line_pcd_shadow)
                    pointcloud_shadow = o3d.geometry.PointCloud()
                    pointcloud_shadow.points = o3d.utility.Vector3dVector(j_p)
                    pointcloud_shadow.paint_uniform_color([1, 1, 1])
                    pointcloud_shadow_mesh = o3d.geometry.PointCloud()
                    pointcloud_shadow_mesh.points = o3d.utility.Vector3dVector(hand_mesh_points)
                    pointcloud_shadow_mesh.paint_uniform_color([212/255, 106/255, 126/255])
                    render_option: o3d.visualization.RenderOption = viewers.get_render_option()  # 设置点云渲染参数
                    viewers.get_render_option().point_size = 2
                    render_option.background_color = np.array([0, 0, 0])
                    viewers.add_geometry(line_pcd_shadow)
                    viewers.add_geometry(pointcloud_shadow)
                    viewers.add_geometry(pointcloud_shadow_mesh)

                    index_ = [0, 3, 6, 9, 12]  # finger tip index
                    grasppart = graspparts[:, np.array(index_)]
                    th = np.where(grasppart[:, 0] == 1)
                    ff = np.where(grasppart[:, 1] == 1)
                    mf = np.where(grasppart[:, 2] == 1)
                    rf = np.where(grasppart[:, 3] == 1)
                    lf = np.where(grasppart[:, 4] == 1)
                    th_pt = points[np.array(th)] # [1,n1,3]
                    ff_pt = points[np.array(ff)] # [1,n2,3]
                    mf_pt = points[np.array(mf)]
                    rf_pt = points[np.array(rf)]
                    lf_pt = points[np.array(lf)]
                    index_5fingers = [6, 11, 16, 21, 27]
                    fingers = j_p[np.array(index_5fingers)]  # [5, 3]
                    fi_th = np.expand_dims(fingers[4].reshape(1, 3), 1).repeat(th_pt.shape[1], 1)
                    fi_ff = np.expand_dims(fingers[3].reshape(1, 3), 1).repeat(ff_pt.shape[1], 1)
                    fi_mf = np.expand_dims(fingers[2].reshape(1, 3), 1).repeat(mf_pt.shape[1], 1)
                    fi_rf = np.expand_dims(fingers[1].reshape(1, 3), 1).repeat(rf_pt.shape[1], 1)
                    fi_lf = np.expand_dims(fingers[0].reshape(1, 3), 1).repeat(lf_pt.shape[1], 1)
                    # 选出大拇指最近的点集
                    dis = 15
                    contact_th = th_pt.squeeze(0)[np.array(np.where(np.sqrt(np.sum((th_pt-fi_th)**2, axis=2)).squeeze() <= dis))].squeeze(0)
                    contact_ff = ff_pt.squeeze(0)[np.array(np.where(np.sqrt(np.sum((ff_pt-fi_ff)**2, axis=2)).squeeze() <= dis))].squeeze(0)
                    contact_mf = mf_pt.squeeze(0)[np.array(np.where(np.sqrt(np.sum((mf_pt-fi_mf)**2, axis=2)).squeeze() <= dis))].squeeze(0)
                    contact_rf = rf_pt.squeeze(0)[np.array(np.where(np.sqrt(np.sum((rf_pt-fi_rf)**2, axis=2)).squeeze() <= dis))].squeeze(0)
                    contact_lf = lf_pt.squeeze(0)[np.array(np.where(np.sqrt(np.sum((lf_pt-fi_lf)**2, axis=2)).squeeze() <= dis))].squeeze(0)
                    #让这些物体上的点突出显示
                    viewers2 = o3d.visualization.Visualizer()
                    viewers2.create_window(window_name="vis_{}.".format(file.split('/')[2]), width=1000, height=800, left=400, top=100)
                    pcd2 = o3d.geometry.PointCloud()
                    pcd2.points = o3d.utility.Vector3dVector(points)
                    # pcd2.paint_uniform_color([.8,.8,.8])
                    pcd2.paint_uniform_color([1/225,95/225,107/225])
                    pcd_finger = [contact_th,contact_ff,contact_mf,contact_rf,contact_lf]
                    for i in range(5):
                        pcd_f = o3d.geometry.PointCloud()
                        pcd_f.points = o3d.utility.Vector3dVector(pcd_finger[i])
                        color = col[i+1]
                        part_color = [x * color[3] for x in color[:3]]
                        pcd_f.paint_uniform_color(part_color)
                        viewers2.add_geometry(pcd_f)
                    viewers2.add_geometry(pcd2)
                    render_option: o3d.visualization.RenderOption = viewers2.get_render_option()  # 设置点云渲染参数
                    viewers2.get_render_option().point_size = 4
                    viewers2.add_geometry(line_pcd_shadow)
                    viewers2.add_geometry(pointcloud_shadow)
                    viewers2.add_geometry(pointcloud_shadow_mesh)

                    if is_AUTO:
                        t = 0
                        while True:
                            t += 1
                            viewers2.update_renderer()
                            viewers2.poll_events()
                            if t == 200:
                                break


                    else:
                        viewers2.run()


if __name__ == '__main__':
    is_AUTO = False # True False means auto visualize
    visual_grasp_pcd(is_AUTO=is_AUTO)


