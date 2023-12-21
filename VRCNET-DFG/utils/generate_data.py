import os
import open3d as o3d
import numpy as np
path = '/media/hjl/Samsung_T5/1dataset/Obj_Data'
save_path = '/media/hjl/Samsung_T5/1dataset/complete_pc/'
for file in os.listdir(path):
    new_path = os.path.join(path, file, 'obj')
    for obj in os.listdir(new_path):
        mesh = o3d.io.read_triangle_mesh(os.path.join(new_path, obj))  # 加载mesh
        sample_num = 5000
        pcd = o3d.geometry.TriangleMesh.sample_points_uniformly(mesh, number_of_points=sample_num)  # 采样点云
        points = np.asarray(pcd.points)
        if not os.path.exists(os.path.join(save_path, file)):
            os.mkdir(os.path.join(save_path, file))
        np.save(os.path.join(save_path, file, obj[:-4]), points)

