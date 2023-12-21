# coding=utf-8

import os
import numpy as np
from xml.dom import minidom
import trimesh
import shutil
import zipfile
from plyfile import PlyData, PlyElement


def getfiles(dirPath, fileType):
    fileList = []
    for root, dirs, files in os.walk(dirPath):
        for f in files:
            w = [i in f for i in fileType]
            if all(w):
                fileList.append(os.path.join(root, f))
            # else:
            #     print('Not a need file:', f)
    return fileList


def read_name(xml_path):
    dom = minidom.parse(xml_path)
    pose_tag = dom.getElementsByTagName('filename')
    obj_name = os.path.splitext(pose_tag[0].firstChild.data)[0].split('/')[-1]
    return obj_name


def read_off(object_dir, object_name):
    file = open(object_dir + '/' + object_name, 'r')
    if 'OFF' != file.readline().strip():
        raise ('Not a valid OFF header')
    n_verts, n_faces, n_dontknow = tuple([int(s) for s in file.readline().strip().split(' ')])
    verts = [[float(s) for s in file.readline().strip().split(' ')] for i_vert in range(n_verts)]
    verts = np.asarray(verts)
    faces = [[int(s) for s in file.readline().strip().split(' ')] for i_face in range(n_faces)]
    faces = np.asarray(faces)[:, 1:]
    return verts, faces


def save_off(output_dir, verts, faces, obj_name):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    OFF_FILE_PATH = os.path.join(output_dir, os.path.splitext(obj_name)[0] + '.off')
    if os.path.exists(OFF_FILE_PATH):
        raise ("The file :'{}' already exists".format(OFF_FILE_PATH))
        # os.remove(OFF_FILE_PATH)
    # 写文件句柄
    handle = open(OFF_FILE_PATH, 'a')
    # 得到点云点数
    vert_num = verts.shape[0]
    face_num = faces.shape[0]

    # pcd头部（重要）
    handle.write('OFF')
    string = '\n' + str(vert_num) + ' ' + str(face_num) + ' ' + str(0)
    handle.write(string)

    # 依次写入点
    for i in range(vert_num):
        string = '\n' + str(verts[i, 0]) + ' ' + str(verts[i, 1]) + ' ' + str(verts[i, 2])
        handle.write(string)
    # 依次写入面
    for i in range(face_num):
        string = '\n3 ' + str(faces[i, 0]) + ' ' + str(faces[i, 1]) + ' ' + str(faces[i, 2])
        handle.write(string)
    handle.close()


def save_pcd(PCD_DIR_PATH, verts, obj_name):
    # 存放路径
    # PCD_DIR_PATH = os.path.join(os.path.abspath('.'), 'pcd')
    if not os.path.exists(PCD_DIR_PATH):
        os.makedirs(PCD_DIR_PATH)
    PCD_FILE_PATH = os.path.join(PCD_DIR_PATH, obj_name + '.obj.pcd')
    if os.path.exists(PCD_FILE_PATH):
        raise ("The file :'{}' already exists".format(PCD_FILE_PATH))
        # os.remove(PCD_FILE_PATH)
    # 写文件句柄
    handle = open(PCD_FILE_PATH, 'a')
    # 得到点云点数
    vert_num = verts.shape[0]

    # pcd头部（重要）
    handle.write(
        '# .PCD v0.7 - Point Cloud Data file format\nVERSION 0.7\nFIELDS x y z\nSIZE 4 4 4\nTYPE F F F\nCOUNT 1 1 1')
    string = '\nWIDTH ' + str(vert_num)
    handle.write(string)
    handle.write('\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0')
    string = '\nPOINTS ' + str(vert_num)
    handle.write(string)
    handle.write('\nDATA ascii')

    # 依次写入点
    for i in range(vert_num):
        string = '\n' + str(verts[i, 0]) + ' ' + str(verts[i, 1]) + ' ' + str(verts[i, 2])
        handle.write(string)
    handle.close()


def export_ply_points_faces(pc, fc, filename):
    vertex = np.zeros(pc.shape[0], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
    # face = np.zeros(fc.shape[0], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
    face = np.zeros(fc.shape[0], dtype=[('vertex_indices', 'i4', (3,))])
    for i in range(pc.shape[0]):
        vertex[i] = (pc[i][0], pc[i][1], pc[i][2])
    # for i in range(fc.shape[0]):
    # face[i] = (fc[i][0], fc[i][1], fc[i][2])
    for i in range(face.shape[0]):
        face[i][0] = (fc[i, 0], fc[i, 1], fc[i, 2])  # face[i][0] = fc[i,1:]
    ply_out = PlyData([PlyElement.describe(vertex, 'vertex', comments=['vertices']),
                       PlyElement.describe(face, 'face', comments=['faces'])])

    ply_filename = filename[:-4] + '.ply'
    ply_out.write(ply_filename)


# Convert ModelNet from .off to .ply
def write_ply_points_faces_from_off(object_dir, object, plyname):
    points, faces = read_off(object_dir, object)
    # out = os.path.join(plyname)
    export_ply_points_faces(points, faces, plyname)

def write_obj_points_faces_from_off(object_dir, object, obj_name):
    points, faces = read_off(object_dir, object)
    # out = os.path.join(plyname)

    try:
        pc_array = np.array([[x, y, z] for x, y, z, _, _, _ in points])
    except:
        pc_array = np.array([[x, y, z] for x, y, z in points])
    face_array = np.array([face for face in faces], dtype=np.int)
    pc_array = pc_array * 0.001
    # face_array = face_array * 0.001
    write_obj(pc_array, face_array, obj_name)

def write_obj(verts, faces, obj_path):
    """
    Write .obj file
    """
    # obj_filename = obj_path[:-4] + '.obj'
    # obj_path.write(obj_filename)

    assert obj_path[-4:] == '.obj'
    with open(obj_path, 'w') as fp:
        for v in verts:
            fp.write('v %f %f %f\n' % (v[0], v[1], v[2]))
        for f in faces + 1:
            fp.write('f %d %d %d\n' % (f[0], f[1], f[2]))


def ply2obj(ply_name, obj_name):
    plydata = PlyData.read(ply_name)
    pc = plydata['vertex'].data
    # print(pc)
    faces = plydata['face'].data
    # print(faces)
    try:
        pc_array = np.array([[x, y, z] for x, y, z, _, _, _ in pc])
    except:
        pc_array = np.array([[x, y, z] for x, y, z in pc])
    face_array = np.array([face[0] for face in faces], dtype=np.int)
    write_obj(pc_array, face_array, obj_name)


if __name__ == "__main__":

    category = 'bottle'

    # outputdir_obj = '../Dataset/Objects/' + category + '/obj'
    # object_dir_off = '../Dataset/Objects/' + category + '/original/zhu'

    outputdir_obj = '../Dataset/Objects/' + category + '/all_obj'
    object_dir_off = '../Dataset/Objects/' + category + '/original/all'
    # off --------> obj
    objects = os.listdir(object_dir_off)
    for object in objects:
        write_obj_points_faces_from_off(object_dir_off, object, outputdir_obj + '/' + object.split('.off')[0] + '.obj')




    # outputdir_obj = '../Dataset/Objects/bottle/obj'
    # object_dir_off = '../Dataset/Objects/bottle/original/zhu'
    #
    # off -------> ply
    # objects = os.listdir(object_dir)
    # for object in objects:
    #     write_ply_points_faces_from_off(object_dir_off, object, os.path.join(outputdir, object))

    # ply -----> obj
    # objects = os.listdir(outputdir)
    # for object in objects:
    #     ply2obj(outputdir+'/'+object.split('.off')[0]+'.ply',
    #             outputdir_obj+'/'+object.split('.off')[0]+'.obj')

    # off -------->stl
    # mesh_org = trimesh.load(os.path.join(object_dir, object))
    # vertices_org, _ = trimesh.sample.sample_surface_even(mesh_org, 20000)
    # vertices_org = np.array(vertices_org)
    # # save_pcd(output_dir, vertices_org, object_name)
    # mesh_org.vertices = mesh_org.vertices / 1000.0
    # mesh_org.export(outputdir+'/'+object.split('.off')[0]+'.stl', 'stl')  # 'dae'



