import os
from os import listdir
from xml.dom.minidom import parse
import xml.dom.minidom
import numpy as np
import torch
import copy
def reading_xml(filename):
    DOMTree = xml.dom.minidom.parse(filename)
    collection = DOMTree.documentElement
    graspableBody = collection.getElementsByTagName('graspableBody')[0]
    filenames = filename
    file_name_i = graspableBody.getElementsByTagName('filename')[0].childNodes[0].data
    file_name = file_name_i.split('/')[-1].replace('.obj.smoothed', '')[:-4]
    # file_name = file_name_i.split('/')[-1].split('_')[0]#.replace('.obj.smoothed','')[:-6]

    # obj_name = filenames.split('.xml')[0].split('_')[5]
    robot = collection.getElementsByTagName("robot")[0]
    robot_name = robot.getElementsByTagName('filename')[0].childNodes[0].data
    a = robot.getElementsByTagName('dofValues')[0]
    a = a.childNodes[0].data
    trans = robot.getElementsByTagName('transform')[0]
    rt = trans.getElementsByTagName('fullTransform')[0]
    rt = rt.childNodes[0].data

    rt_obj_ = graspableBody.getElementsByTagName('transform')[0]
    rt_obj = rt_obj_.getElementsByTagName('fullTransform')[0]
    rt_obj = rt_obj.childNodes[0].data
    r_obj = rt_obj.split(')')[0].split('(')[1].split(' ')
    r_obj = np.array(r_obj, dtype='float32')
    r_obj = torch.tensor(r_obj, dtype=torch.float32)  # .unsqueeze(0)
    rs_obj = r_obj.clone()
    # for i in range(4):
    rs_obj[0] = r_obj[1]
    rs_obj[1] = r_obj[2]
    rs_obj[2] = r_obj[3]
    rs_obj[3] = r_obj[0]

    r = rt.split(')')[0].split('(')[1].split(' ')
    t = rt.split('[')[1].split(']')[0].split(' ')

    a = a.replace('   ', ' ')
    a = np.array(a.split(' '), dtype='float32')
    a = torch.tensor(a, dtype=torch.float32).unsqueeze(0)
    if robot_name[-6:-4] == 'lw':
        a[:, 1] = -a[:, 1]
    elif robot_name[-6:-4] != 'nd':
        raise ('robot_name has wrong, place call zhutq')
    r = np.array(r, dtype='float32')
    r = torch.tensor(r, dtype=torch.float32)  # .unsqueeze(0)
    rs = copy.deepcopy(r)
    # for i in range(4):
    rs[0] = r[1]
    rs[1] = r[2]
    rs[2] = r[3]
    rs[3] = r[0]

    t = np.array(t, dtype='float32')
    t = torch.tensor(t, dtype=torch.float32) #* 0.001  # .unsqueeze(0)

    return a, rs, t, file_name, rs_obj

if __name__ == '__main__':
    a, r, t, obj_name = reading_xml('/home/hjl/isaacgym/python/train_set_experiment/grasp/train/epoch20_82_4_stapler.xml')
    print(a.shape,r.shape,t.shape)

