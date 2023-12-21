
import numpy as np
from mesh_to_sdf import get_surface_point_cloud,sample_sdf_near_surface,scale_to_unit_sphere_dif,scale_to_unit_cube,scale_to_mesh,scale_to_unit_sphere
import trimesh
import pyrender
import scipy.io as sio
import os
# from mesh_to_sdf.utils import show_point
import tqdm

# CLASS_TO_ID = {'mug':'03797390'}
#
# def remv_models(data_dir, category, mode = 'train'):
#     remv_models = []
#
#     with open('{0}/removed_models/{1}_{2}.txt'.format(data_dir, CLASS_TO_ID[category], mode), 'r') as f:
#         m = f.readline().rstrip('\n')
#         while m != '':
#             remv_models.append(m)
#             m = f.readline().rstrip('\n')
#
#     obj_path = os.path.join(data_dir, category, mode)
#     for file in os.listdir(obj_path):
#         if file in remv_models:
#             os.system('rm -rf {0}/{1}'.format(obj_path, file))

def sample_sdf(file_path):

    mesh = trimesh.load(file_path)
    # mesh = scale_to_mesh(mesh)
    # mesh = scale_to_unit_sphere(mesh)
    mesh, scale = scale_to_unit_sphere_dif(mesh)

    # mesh = scale_to_unit_cube(mesh)
    # points_with_sdf, points_with_normal = sample_surface_sdf(mesh, number_of_points=250000)


    surface = get_surface_point_cloud(mesh, surface_point_method='scan', scan_count=100, sample_point_count=500000,
                                    calculate_normals=True)
    surface_p = surface.points
    surface_n = surface.normals
    surface_data = np.concatenate((surface_p, surface_n), axis=1)
    # showpoint  = np.array(surface_p)
    # show_point(showpoint)
    # surface_data = {'p': surface_data_}
    #
    # free_p, free_sdf = sample_sdf_near_surface(mesh, surface_point_method='scan', sign_method='depth',
    #                                            sample_point_count=500000)
    # free_p, free_sdf = sample_sdf_near_surface(mesh, surface_point_method='scan', sign_method='depth',
    #                                            sample_point_count=500000)
    free_p, free_sdf = sample_sdf_near_surface(mesh, sample_point_count=500000)
    free_sdf = free_sdf.reshape(-1, 1)
    free_data = np.concatenate((free_p, free_sdf), axis=1)
    # free_data = {'p_sdf': free_data_}
    return free_data, surface_data, scale


def show_sdf(points_free, sdf_free, points_surface=None):

    colors = np.zeros(points_free.shape)
    colors[sdf_free < 0, 2] = 1
    colors[sdf_free > 0, 0] = 1

    scene = pyrender.Scene()
    cloud_free = pyrender.Mesh.from_points(points_free, colors=colors)
    scene.add(cloud_free)

    if points_surface is not None:
        colors = np.zeros(points_surface.shape)
        colors[:, 1] = 1
        cloud_surface = pyrender.Mesh.from_points(points_surface, colors=colors)
        scene.add(cloud_surface)

    pyrender.Viewer(scene, use_raymond_lighting=True, point_size=2)


def write_split(obj_path, category, split_path, mode):
    obj_path = os.path.join(obj_path, category, mode)
    inst_list = os.listdir(obj_path)
    with open(os.path.join(split_path, mode, category + '.txt'), 'w') as f:
        for inst_name in inst_list:
            for file in os.listdir(os.path.join(obj_path, inst_name)):
                # f.write(inst_name+'/'+file.split('.')[0]+'\n')
                f.write(inst_name+'\n')

    # file_list = os.listdir(os.path.join(obj_path, category, 'eval'))
    # with open(os.path.join(split_path, 'eval', category + '.txt'), 'w') as f:
    #     for file in file_list:
    #         f.write(file+'\n')


def gen_single_dif_data(file_path, save_path, file_name, vis=False):
    if not os.path.exists(os.path.join(save_path, 'free_space_pts')):
        os.makedirs(os.path.join(save_path, 'free_space_pts'))
    if not os.path.exists(os.path.join(save_path, 'surface_pts_n_normal')):
        os.makedirs(os.path.join(save_path, 'surface_pts_n_normal'))
    points_with_sdf, points_with_normal = sample_sdf(file_path)
    if vis:
        show_sdf(points_with_sdf[0], points_with_sdf[1], points_with_normal.points)
    write_mat_data(save_path, file_name, points_with_sdf, points_with_normal)


def write_mat_data(save_path, file, points_with_sdf, points_with_normal, scale):
    # pts_sdf = np.hstack((points_with_sdf[0], points_with_sdf[1][:, np.newaxis]))
    # pts_normal = np.hstack((points_with_normal.points, points_with_normal.normals))
    pts_sdf = points_with_sdf
    pts_normal = points_with_normal
    free_points_path = os.path.join(save_path, 'free_space_pts', file + '.mat')
    surface_points_path = os.path.join(save_path, 'surface_pts_n_normal', file + '.mat')

    sio.savemat(free_points_path, {'p_sdf': pts_sdf, 'scale': scale})
    sio.savemat(surface_points_path, {'p': pts_normal})

def cond_mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

if __name__ == '__main__':

    import sys
    sys.path.append('./')
    import argparse
    import json
    import shutil
    parser = argparse.ArgumentParser(description='generate sdf')
    parser.add_argument('--idx', type=int, default=0, help="0-20 id of hjl new dataset, means category")
    args = parser.parse_args()




    data_root = '../Dataset/Obj_Data'
    # category = 'bottle'
    # category = args.idx
    f = open('../Dataset/Obj_Data/obj_data.json','r')
    content = json.loads(f.read())
    category = content[str(args.idx)]['category']

    # mode = 'train'
    # split_path = 'dif/split'

    save_path = os.path.join(data_root, category, 'sdf')
    obj_path = os.path.join(data_root, category, 'obj')



    # write split.json
    dicts = {}
    # for category in categories:
    obj_list = os.listdir(obj_path)
    instance_list = []
    for name in obj_list:
        instance_list.append(name[:-4])

    obj_dict = {'real': instance_list,
                'virtual': []}
    dicts[str(args.idx)] = obj_dict
    d = json.dumps(dicts, indent=1)
    if not os.path.exists(os.path.join(data_root, category, 'split')):
        os.makedirs(os.path.join(data_root, category, 'split'))
    f = open(os.path.join(data_root, category, 'split', 'split.json'), 'w')
    f.write(d)
    f.close()

    #write specs.json
    original_json = os.path.join(data_root, 'specs.json')
    new_json = os.path.join(data_root, category, 'split', 'specs.json')
    shutil.copy(original_json, new_json)
    spec = open(new_json,'r')
    dict = json.loads(spec.read())
    dict['Description'] = category
    dict['TrainSplit'] = os.path.join(data_root, category, 'split', 'split.json')
    ds = json.dumps(dict, indent=1)
    f1 = open(new_json, 'w')
    f1.write(ds)
    f1.close()

    if not os.path.exists(os.path.join(save_path, 'free_space_pts')):
        os.makedirs(os.path.join(save_path, 'free_space_pts'))
    if not os.path.exists(os.path.join(save_path, 'surface_pts_n_normal')):
        os.makedirs(os.path.join(save_path, 'surface_pts_n_normal'))

    for i, file in enumerate(os.listdir(obj_path)):
        print(i, '    ', file)

        ####generating sdf...####
        # for j in tqdm(range(1)):
        for j in range(1):
            if os.path.exists(os.path.join(save_path, 'free_space_pts', file + '.mat')):
                continue
            # points_with_sdf, points_with_normal = sample_sdf(os.path.join(obj_path, file, '0.obj'))
            # show_sdf(points_with_sdf[0], points_with_sdf[1], points_with_normal.points)
            points_with_sdf, points_with_normal, scale = sample_sdf(os.path.join(obj_path, file))
            write_mat_data(save_path, file, points_with_sdf, points_with_normal, scale)

        ####visualize sdf....####
        # pts_free = sio.loadmat(os.path.join(save_path, 'free_space_pts', file + '.mat'))['p_sdf']
        # # surface pts n normal
        # pts_normal = sio.loadmat(os.path.join(save_path, 'surface_pts_n_normal', file + '.mat'))['p']
        # show_sdf(pts_free[:,:3], pts_free[:,3])
        # show_sdf(pts_free[:,:3], pts_free[:,3], pts_normal[:,:3])