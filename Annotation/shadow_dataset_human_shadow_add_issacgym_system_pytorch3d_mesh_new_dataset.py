import mediapipe as mp
import sys
import argparse

import numpy as np
import torch
import warnings
warnings.filterwarnings("ignore")
from utils.util_camera import DLT, get_projection_matrix

import copy
from utils.FK_model import Shadowhand_FK
from utils.trans import *
from utils.util_teachnet_new_data import *
import cv2
import pyrealsense2 as rs
from utils.util_fk_new_data import *
from show_data_mesh import *
import json
import time
from utils.write_xml_new_data import write_xml_new_data
from FK_layer_mesh import FK_layer
import torch
import time

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
frame_shape = [720, 1280]


def run_mp(input_stream1, input_stream2, P0, P1):
    #input video stream
    cap0 = cv2.VideoCapture(input_stream1)
    cap1 = cv2.VideoCapture(input_stream2)

    caps = [cap0, cap1]

    #set camera resolution if using webcam to 1280x720. Any bigger will cause some lag for hand detection
    for cap in caps:
        cap.set(3, frame_shape[1])
        cap.set(4, frame_shape[0])

    #create hand keypoints detector object.
    hands0 = mp_hands.Hands(min_detection_confidence=0.5, max_num_hands =1, min_tracking_confidence=0.5)
    hands1 = mp_hands.Hands(min_detection_confidence=0.5, max_num_hands =1, min_tracking_confidence=0.5)

    #containers for detected keypoints for each camera
    kpts_cam0 = []
    kpts_cam1 = []
    kpts_3d = []
    num = 0
    t = time.time()
    j_ps = np.zeros((5,28,3))
    angle = torch.zeros((5,1,18))
    hand = np.zeros((5,21,3))
    shadow_mesh = np.zeros((5,2000,3))
    while True:

        #read frames from stream
        ret0, frame0 = cap0.read()
        ret1, frame1 = cap1.read()

        if not ret0 or not ret1:
            print('----warning! device not all detected and break program----')
            break

        #crop to 720x720.
        #Note: camera calibration parameters are set to this resolution.If you change this, make sure to also change camera intrinsic parameters
        if frame0.shape[1] != 720:
            frame0 = frame0[:,frame_shape[1]//2 - frame_shape[0]//2:frame_shape[1]//2 + frame_shape[0]//2]
            frame1 = frame1[:,frame_shape[1]//2 - frame_shape[0]//2:frame_shape[1]//2 + frame_shape[0]//2]

        # the BGR image to RGB.
        frame0 = cv2.cvtColor(frame0, cv2.COLOR_BGR2RGB)
        frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
        # frame0 = cv.flip(frame0, 1)  # this allows us to flip the image horizontaly
        # frame1 = cv.flip(frame1, 1)  # this allows us to flip the image horizontaly

        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        frame0.flags.writeable = False
        frame1.flags.writeable = False
        results0 = hands0.process(frame0)
        results1 = hands1.process(frame1)

        #prepare list of hand keypoints of this frame
        #frame0 kpts
        frame0_keypoints = []
        if results0.multi_hand_landmarks:
            for hand_landmarks in results0.multi_hand_landmarks:
                for p in range(21):
                    #print(p, ':', hand_landmarks.landmark[p].x, hand_landmarks.landmark[p].y)
                    pxl_x = int(round(frame0.shape[1]*hand_landmarks.landmark[p].x))
                    pxl_y = int(round(frame0.shape[0]*hand_landmarks.landmark[p].y))
                    kpts = [pxl_x, pxl_y]
                    frame0_keypoints.append(kpts)

        #no keypoints found in frame:
        else:
            #if no keypoints are found, simply fill the frame data with [-1,-1] for each kpt
            frame0_keypoints = [[-1, -1]]*21

        kpts_cam0.append(frame0_keypoints)

        #frame1 kpts
        frame1_keypoints = []
        if results1.multi_hand_landmarks:
            for hand_landmarks in results1.multi_hand_landmarks:
                for p in range(21):
                    #print(p, ':', hand_landmarks.landmark[p].x, hand_landmarks.landmark[p].y)
                    pxl_x = int(round(frame1.shape[1]*hand_landmarks.landmark[p].x))
                    pxl_y = int(round(frame1.shape[0]*hand_landmarks.landmark[p].y))
                    kpts = [pxl_x, pxl_y]
                    frame1_keypoints.append(kpts)

        else:
            #if no keypoints are found, simply fill the frame data with [-1,-1] for each kpt
            frame1_keypoints = [[-1, -1]]*21

        #update keypoints container
        kpts_cam1.append(frame1_keypoints)


        #calculate 3d position
        frame_p3ds = []
        for uv1, uv2 in zip(frame0_keypoints, frame1_keypoints):
            if uv1[0] == -1 or uv2[0] == -1:
                # _p3d = [-1, -1, -1]
                _p3d = [-7+ random.random(), 33+ random.random(), 8+ random.random()]

            else:
                _p3d = DLT(P0, P1, uv1, uv2) #calculate 3d position of keypoint
            frame_p3ds.append(_p3d)

        '''
        This contains the 3d position of each keypoint in current frame.
        For real time application, this is what you want.
        '''
        frame_p3ds = np.array(frame_p3ds).reshape((21, 3))
        pcd = copy.deepcopy(frame_p3ds) / 10.0
        pcd = np.matmul(view_mat, pcd.T).T
        # cam0 to our world coordinate
        pcd[:,2] += 5
        pcd[:,1] += 0.4
        pcd[:,0] -= 0.25

        #pcd : hand point
        #####################
        #trans to shadowhand#
        ##  using FK model  #
        #####################

        base = np.array([[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]])
        hand[:4] = hand[1:]
        hand[4] = pcd
        pcd = np.mean(hand, axis=0)
        base[:, :3] = pcd[0] * 200.0


        #############angles##################################################
        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()

        frames = align_to_color.process(frames)

        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue
        # Convert images to numpy arrays

        depth_image = np.asanyarray(depth_frame.get_data())
        # color_image = np.asanyarray(color_frame.get_data())
        depth_image = cv2.convertScaleAbs(depth_image, alpha=0.03)

        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        # depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        # Show images
        cv2.namedWindow('depth_img')
        cv2.resizeWindow("depth_img", 100, 100)
        cv2.moveWindow("depth_img", 1900, 0)
        new_img = seg_hand_depth(depth_image, 500, 1000, 10, 100, 4, 4, 250, True, 300)
        new_img = new_img.astype(np.float32)
        new_img = new_img / 255. * 2. - 1
        cv2.imshow('depth_img', new_img)
        new_img = new_img[np.newaxis, np.newaxis, ...]
        new_img = torch.from_numpy(new_img)
        new_img = new_img.cuda()

        goal = joint_cal(new_img, model, isbio=False)
        # GOAL ： FF LF MF RF TH     FK_model： LF RF MF FF TH
        goals = goal[2:]
        fk_angles = copy.deepcopy(goals)
        fk_angles[:5] = goals[4:9]
        fk_angles[5:9] = goals[13:17]
        fk_angles[13:17] = goals[:4]
        fk_angles = torch.tensor(fk_angles).cuda()#22
        inedx = [4,8,12,16] #delete the last angles of four fingers except thumb
        index = [3,7,12,15]
        ind = torch.tensor([0,1,2,3,5,6,7,9,10,11,13,14,15,17,18,19,20,21])
        fk_angles[index] += fk_angles[inedx]

        fk_angle = fk_angles[ind]
        rotation = fk_angle.reshape(1,18)
        angle[:4] = angle.clone()[1:]
        angle[4] = rotation
        rotations = torch.mean(angle, dim=0, keepdim=False)

        z = pcd[9] - pcd[0]
        z = z / np.linalg.norm(z)
        y = np.cross(pcd[13] - pcd[0], pcd[9] - pcd[0])
        y = y / np.linalg.norm(y)
        x = np.cross(y, z)
        x = x / np.linalg.norm(x)
        pose_new = np.concatenate((x, y, z), axis=0)
        pose_new = pose_new.reshape((3, 3))
        # pose_new = np.eye(3,3)
        pose_new_shadow = copy.deepcopy(pose_new)
        pose_new_shadow = np.matmul(pose_new_shadow.T,view_mat_shadow)

        # qx, qy, qz, qw
        quat = trans_2_quat_cpu(pose_new_shadow)
        quat_qwxyz = copy.deepcopy(quat)
        quat_qwxyz[0] = quat[3]
        quat_qwxyz[1:] = quat[:3]
        base[:, 3:7] = quat_qwxyz.reshape(1,4)

        outputs_base, outputs_rotation = trans_rtj_point(base, rotations/1.5708)
        fk = Shadowhand_FK()

        # 输入为base的姿态 (F, 7): x y z qw qx qy qz
        j_p = fk.run(outputs_base, outputs_rotation * 1.5708)  # [F, J+10, 3]  #原始J+1个关键点，加上10个关键点
        j_p = j_p[:,:28].squeeze(0)
        j_p = j_p.cpu().detach().numpy() #* 0.01
        # print(j_p)
        num+=1
        if num == 1:
            colpart = np.unique(graspparts, axis=0)
            print('colpart:\n', colpart)
            print('白,绿,蓝...')
        draw_pic(points, graspparts, j_p, hand_verts)



        #####add point cloud of shadowhand
        angles = torch.zeros(22)
        inds = torch.tensor([4, 8, 12, 16])
        fk_angles[inds] = 0.0
        angles[:4] = fk_angles[13:17]
        angles[4:8] = fk_angles[9:13]
        angles[8:12] = fk_angles[5:9]
        angles[12:17] = fk_angles[:5]
        angles[17:] = fk_angles[17:]
        angles[[0,4]] = -angles[[0,4]]########mention######

        angles[-2] = -angles[-2]
        angles[-1] = -angles[-1]
        t = pcd[0] * 200.0


        view_mat_mesh = np.array([[0.0, 0.0, 1.0],
                                    [-1.0, 0, 0.0],
                                    [0.0, -1.0, 0]])
        r = quat_mul(quat, trans_2_quat_cpu(view_mat_mesh)).reshape(4)

        base_M = torch.from_numpy(get_4x4_matrix([r[0],r[1],r[2],r[3]], t)).float().reshape(1, 1, 4, 4).cuda()
        angles = torch.FloatTensor(angles).reshape(1, -1).cuda()
        fk_layers = FK_layer(base_M, angles)
        fk_layers.cuda()
        _, transformed_pts = fk_layers()
        shadowmesh_points = transformed_pts.clone().squeeze(0).detach().cpu().numpy()
        # mean blur均值滤波, 使轨迹平滑
        shadow_mesh[:4] = shadow_mesh[1:]
        shadow_mesh[4] = shadowmesh_points
        interpolate_shadowhandmesh = np.mean(shadow_mesh, axis=0)
        transformed_pt.points = o3d.utility.Vector3dVector(interpolate_shadowhandmesh)
        # transformed_pt.paint_uniform_color([248/255, 213/255, 97/255])
        transformed_pt.paint_uniform_color([212 / 255, 106 / 255, 126 / 255])
        viewers3.update_geometry(transformed_pt)

        lines_shadow = [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [0, 7], [7, 8], [8, 9], [9, 10], [10, 11],
                        [0, 12], [12, 13], [13, 14], [14, 15], [15, 16],
                        [0, 17], [17, 18], [18, 19], [19, 20], [20, 21], [0, 22], [22, 23], [23, 24], [24, 25],
                        [25, 26], [26, 27]]

        colors_shadow = [[0, 0.8, 0], [0, 0.8, 0], [0.8, 0.8, 0], [0.8, 0, 0], [0, 0, 0.8], [0.8, 0, 0.8],
                         [0, 0.8, 0], [0.8, 0.8, 0], [0.8, 0, 0], [0, 0, 0.8], [0.8, 0, 0.8],
                         [0, 0.8, 0], [0.8, 0.8, 0], [0.8, 0, 0], [0, 0, 0.8], [0.8, 0, 0.8],
                         [0, 0.8, 0], [0.8, 0.8, 0], [0.8, 0, 0], [0, 0, 0.8], [0.8, 0, 0.8],
                         [0, 0.8, 0], [0, 0.8, 0], [0.8, 0.8, 0], [0.8, 0, 0], [0, 0, 0.8], [0.8, 0, 0.8]]

        # mean blur均值滤波, 使轨迹平滑
        j_ps[:4] = j_ps[1:]
        j_ps[4] = j_p
        interpolate = np.mean(j_ps,axis=0)

        pointcloud_shadow.points = o3d.utility.Vector3dVector(interpolate)
        pointcloud_shadow.paint_uniform_color([1,1,1])

        #####################
        line_pcd_shadow.lines = o3d.utility.Vector2iVector(lines_shadow)
        line_pcd_shadow.colors = o3d.utility.Vector3dVector(colors_shadow)
        line_pcd_shadow.points = o3d.utility.Vector3dVector(interpolate)

        viewers0.update_geometry(pointcloud_shadow)
        viewers0.update_geometry(line_pcd_shadow)
        viewers0.poll_events()
        viewers0.update_renderer()

        viewers1.update_geometry(pointcloud_shadow)
        viewers1.update_geometry(line_pcd_shadow)
        viewers1.poll_events()
        viewers1.update_renderer()


        viewers3.update_renderer()
        viewers3.poll_events()


        kpts_3d.append(frame_p3ds)
        # Draw the hand annotations on the image.
        frame0.flags.writeable = True
        frame1.flags.writeable = True
        frame0 = cv2.cvtColor(frame0, cv2.COLOR_RGB2BGR)
        frame1 = cv2.cvtColor(frame1, cv2.COLOR_RGB2BGR)

        if results0.multi_hand_landmarks:
          for hand_landmarks in results0.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame0, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        if results1.multi_hand_landmarks:
          for hand_landmarks in results1.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame1, hand_landmarks, mp_hands.HAND_CONNECTIONS)


        frame0 = cv2.resize(frame0,(420,420))
        frame1 = cv2.resize(frame1,(420,420))

        cv2.namedWindow('cam1')
        cv2.resizeWindow("cam1", 420, 420)
        cv2.namedWindow('cam0')
        cv2.resizeWindow("cam0", 420, 420)
        cv2.moveWindow("cam1", 2400, 0)
        cv2.moveWindow("cam0", 2400, 500)
        cv2.imshow('cam1', frame1)
        cv2.imshow('cam0', frame0)

        k = cv2.waitKey(1)
        if k & 0xFF == 27:
            viewers0.destroy_window()
            viewers1.destroy_window()
            # viewers2.destroy_window()
            viewers3.destroy_window()
            break #27 is ESC key.

        elif k & 0xFF == ord(' '):
            # saving xml file for dataset
            print('Start saving XML file for grasping...')
            grip_r = base[:, 3:7].reshape(4)
            obj_r = trans_r_
            grip_t = base[:, :3].reshape(3)
            grip_a = rotations.detach().cpu().numpy() #* 1.5708
            grip_a = grip_a.reshape(18)

            results_xml_path = 'Grasp_Pose'
            if os.path.exists(results_xml_path + '/{}_{}'.format(obj_index,str(obj_name[:-4]))):
                print('folder have been created')
                dirs = os.listdir(results_xml_path + '/{}_{}'.format(obj_index,str(obj_name[:-4])))
                list = []
                for dir in dirs:
                    if dir.endswith('.pkl'):
                        num_ = int(dir.split('_')[1].split('.')[0])
                        list.append(num_)
                list_ = sorted(list)
                print(list_)
                num = list_[len(list_)-1]
                num += 1


            else:
                os.makedirs(results_xml_path + '/{}_{}'.format(obj_index,str(obj_name[:-4])))
                print('creating obj path')
                num = 0

            #saving xml file in grasp pose folder
            write_xml_new_data(category=category,obj_name=obj_name, r=grip_r, r_o=obj_r, t=grip_t, a=grip_a,
                      path=results_xml_path + '/{}_{}/{}_{}.xml'.format(obj_index,str(obj_name[:-4]),obj_index, num),
                      mode='train', rs=(21, 'real'))

            ##saving pcd and visualize in open3d
            path_pcd_pkl = 'Grasp_Pose'
            save_pcd_pkl_and_mesh(points, graspparts, obj_name[:-4], interpolate, obj_index, num, path_pcd_pkl, interpolate_shadowhandmesh)


    cv2.destroyAllWindows()

    for cap in caps:
        cap.release()

    return np.array(kpts_cam0), np.array(kpts_cam1), np.array(kpts_3d)


def draw_pic(points, graspparts,j_p, verts):
    colpart = np.unique(graspparts, axis=0)
    col = [[1, 1, 1, 0.8],#白，绿，蓝
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


    points_list = []
    for j in range(colpart.shape[0]):
        ppp = []
        for i in range(points.shape[0]):
            if (graspparts[i]==colpart[j]).all():
                ppp.append(points[i])
        ppp = np.concatenate(ppp, 0).reshape(-1, 3)
        points_list.append(ppp)

    p_sum=0
    # pcds = []
    for i, ppp in enumerate(points_list):
        color = col[i]
        pcd_all[i].points = o3d.utility.Vector3dVector(ppp)
        part_color = [x * color[3] for x in color[:3]]
        pcd_all[i].paint_uniform_color(part_color)
        p_sum += ppp.shape[0]

    viewers0.update_geometry(pcd_all[0])
    viewers0.update_geometry(pcd_all[1])
    viewers0.update_geometry(pcd_all[2])
    viewers0.update_geometry(pcd_all[3])
    viewers0.update_geometry(pcd_all[4])
    viewers0.update_geometry(pcd_all[5])
    viewers0.update_geometry(pcd_all[6])
    viewers0.update_geometry(pcd_all[7])
    viewers0.update_geometry(FOR1)

    viewers1.update_geometry(pcd_all[0])
    viewers1.update_geometry(pcd_all[1])
    viewers1.update_geometry(pcd_all[2])
    viewers1.update_geometry(pcd_all[3])
    viewers1.update_geometry(pcd_all[4])
    viewers1.update_geometry(pcd_all[5])
    viewers1.update_geometry(pcd_all[6])
    viewers1.update_geometry(pcd_all[7])
    viewers1.update_geometry(FOR1)

    point_object.points = o3d.utility.Vector3dVector(points)
    point_object.paint_uniform_color([1/225,95/225,107/225])
    pcd_finger = visualize_contact_points(points, graspparts, j_p)
    contact_0.points = o3d.utility.Vector3dVector(pcd_finger[0])
    contact_1.points = o3d.utility.Vector3dVector(pcd_finger[1])
    contact_2.points = o3d.utility.Vector3dVector(pcd_finger[2])
    contact_3.points = o3d.utility.Vector3dVector(pcd_finger[3])
    contact_4.points = o3d.utility.Vector3dVector(pcd_finger[4])
    contact_0.paint_uniform_color([0, 0.8, 0])
    contact_1.paint_uniform_color([0, 0, 0.8])
    contact_2.paint_uniform_color([0.8, 0.8, 0])
    contact_3.paint_uniform_color([0.8, 0, 0.8])
    contact_4.paint_uniform_color([0.8, 0, 0])
    viewers3.update_geometry(contact_0)
    viewers3.update_geometry(contact_1)
    viewers3.update_geometry(contact_2)
    viewers3.update_geometry(contact_3)
    viewers3.update_geometry(contact_4)
    viewers3.update_geometry(point_object)
    viewers3.update_geometry(FOR1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--idx', type=int, default=4, help="0-20 id of hjl new dataset, means category")
    parser.add_argument('--instance', type=int, default=0, help="id of each instance of its category, use random instance is alright")
    parser.add_argument('--cam_1', type=int, default=6, help='your camera1 id')
    parser.add_argument('--cam_2', type=int, default=4, help='your camera2 id')
    args = parser.parse_args()

    category_dict = {'mug': '1.57 0 -1.57', 'bowl': '1.57 0 -1.57', 'knife': '0 -1.57 0', 'mouse': '0 -1.57 3.14',
                    'bottle': '1.57 0 -1.57',
                    'bucket': '1.57 0 0', 'camera': '1.57 0 0', 'pliers': '1.56 0 -1.58', 'remote': '3.14 0 0',
                    'teapot': '1.57 0 -1.57',
                    'stapler': '1.57 0 -1.57', 'eyeglass': '-1.57 -1.57 0', 'scissors': '-1.57 3.14 0',
                    'spraycan': '1.57 0 0', 'headphone': '1.57 0 0',
                    'lightbulb': '1.57 0 0', 'wineglass': '1.57 0 0', 'flashlight': '0 -1.57 1.57',
                    'screwdriver': '-1.57 0 1.57', 'spraybottle': '1.57 0 0',
                    'drill_hairdryer': '-1.57 -1.57 0'}

    dataset_path = 'IsaacGym/assets/urdf/off/Obj_Data' 
    category_id = str(args.idx) #id.split(' ')[0]
    obj_index = int(category_id)
    # from 0-20 for 21 categories
    f = open('obj_data.json','r')
    content = json.loads(f.read())
    category = content[category_id]['category']
    # each category select 2 instance for grasp annotation
    instance_id = args.instance 
    instance_name = content[category_id]['instance'][int(instance_id)]
    f.close()
    print('\033[1;35myou are collecting grasp of category:{} idx:{} instance:{} \033[0m'.format(category, category_id, instance_name))

    mesh = o3d.io.read_triangle_mesh(os.path.join(dataset_path,category,instance_name))  # 加载mesh
    sample_num = 4000

    pcd = o3d.geometry.TriangleMesh.sample_points_uniformly(mesh, number_of_points=sample_num)  # 采样点云
    points = np.asarray(pcd.points) * 1000.0
    graspparts = np.repeat(np.array([[1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0]]), points.shape[0], axis=0)
    obj_name = instance_name

    angles = category_dict[category]
    x_a = float(angles.split(' ')[0]) * (np.pi / 2 / 1.57)
    y_a = float(angles.split(' ')[1]) * (np.pi / 2 / 1.57)
    z_a = float(angles.split(' ')[2]) * (np.pi / 2 / 1.57)
    a_list = [x_a, y_a, z_a]
    r  = eulerAnglesToRotationMatrix(a_list)

    a = np.dot(r,r.T)
    trans_r = trans_2_quat_cpu(r.T)  # qx, qy, qz, qw
    trans_r_ = copy.deepcopy(trans_r)
    trans_r_[0] = trans_r[3]
    trans_r_[1:] = trans_r[:3]
    #物体的旋转四元数
    trans_r_= trans_r_.reshape(4)  # qw，qx, qy, qz
    points = np.dot(points, r)

    FOR1 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=100, origin=[0, 0, 0])
    viewers0 = o3d.visualization.Visualizer()
    viewers0.create_window(width=400, height=350, left=1900, top=220, window_name='viewer_0')
    viewers1 = o3d.visualization.Visualizer()
    viewers1.create_window(width=400, height=350, left=1900, top=620, window_name='viewer_1')
    viewers3 = o3d.visualization.Visualizer()
    viewers3.create_window(width=1080, height=800, left=1900, top=1100, window_name='viewer_runtime')

    pointcloud = o3d.geometry.PointCloud()
    pointcloud_shadow = o3d.geometry.PointCloud()
    pcd0 = o3d.geometry.PointCloud()
    pcd1 = o3d.geometry.PointCloud()
    pcd2 = o3d.geometry.PointCloud()
    pcd3 = o3d.geometry.PointCloud()
    pcd4 = o3d.geometry.PointCloud()
    pcd5 = o3d.geometry.PointCloud()
    pcd6 = o3d.geometry.PointCloud()
    pcd7 = o3d.geometry.PointCloud()
    pcd_all = []
    pcd_all.append(pcd0)
    pcd_all.append(pcd1)
    pcd_all.append(pcd2)
    pcd_all.append(pcd3)
    pcd_all.append(pcd4)
    pcd_all.append(pcd5)
    pcd_all.append(pcd6)
    pcd_all.append(pcd7)
    point_size = 3
    render_option: o3d.visualization.RenderOption = viewers0.get_render_option()  # 设置点云渲染参数
    viewers0.get_render_option().point_size = point_size
    render_option.background_color = np.array([0, 0, 0])  # 设置背景色
    viewers0.add_geometry(FOR1)
    viewers0.add_geometry(pointcloud)
    viewers0.add_geometry(pointcloud_shadow)
    viewers0.add_geometry(pcd_all[0])
    viewers0.add_geometry(pcd_all[1])
    viewers0.add_geometry(pcd_all[2])
    viewers0.add_geometry(pcd_all[3])
    viewers0.add_geometry(pcd_all[4])
    viewers0.add_geometry(pcd_all[5])
    viewers0.add_geometry(pcd_all[6])
    viewers0.add_geometry(pcd_all[7])

    ##########add two viewer########
    render_option1: o3d.visualization.RenderOption = viewers1.get_render_option()  # 设置点云渲染参数
    viewers1.get_render_option().point_size = point_size
    render_option1.background_color = np.array([0, 0, 0])  # 设置背景色（这里为黑色）
    viewers1.add_geometry(FOR1)
    viewers1.add_geometry(pointcloud)
    viewers1.add_geometry(pointcloud_shadow)
    viewers1.add_geometry(pcd_all[0])
    viewers1.add_geometry(pcd_all[1])
    viewers1.add_geometry(pcd_all[2])
    viewers1.add_geometry(pcd_all[3])
    viewers1.add_geometry(pcd_all[4])
    viewers1.add_geometry(pcd_all[5])
    viewers1.add_geometry(pcd_all[6])
    viewers1.add_geometry(pcd_all[7])

    points_box = np.load('points_box.npy')
    mesh_box_points = o3d.geometry.PointCloud()
    mesh_box_points.points = o3d.utility.Vector3dVector(points_box)
    mesh_box_points.paint_uniform_color([255/255, 255/255, 255/255])

    render_option3: o3d.visualization.RenderOption = viewers3.get_render_option()  # 设置点云渲染参数
    point_size3 = 1.5
    viewers3.get_render_option().point_size = point_size3
    viewers3.add_geometry(FOR1)
    viewers3.add_geometry(pointcloud_shadow)
    contact_0 = o3d.geometry.PointCloud()
    contact_1 = o3d.geometry.PointCloud()
    contact_2 = o3d.geometry.PointCloud()
    contact_3 = o3d.geometry.PointCloud()
    contact_4 = o3d.geometry.PointCloud()
    point_object = o3d.geometry.PointCloud()
    viewers3.add_geometry(point_object)
    viewers3.add_geometry(contact_0)
    viewers3.add_geometry(contact_1)
    viewers3.add_geometry(contact_2)
    viewers3.add_geometry(contact_3)
    viewers3.add_geometry(contact_4)
    viewers3.add_geometry(mesh_box_points)



    ##########add two viewer########
    FOR1 = o3d.geometry.TriangleMesh.scale(FOR1, 0.1, [0, 0, 0])

    input_stream1 = args.cam_1
    input_stream2 = args.cam_2

    if len(sys.argv) == 3:
        input_stream1 = int(sys.argv[1])
        input_stream2 = int(sys.argv[2])

    # view for hand visualize
    view_mat = np.array([[1.0, 0.0, 0.0],
                         [0.0, -1.0, 0],
                         [0.0, 0, -1.0]])

    # projection matrices
    view_mat_trans = np.array([[1.0, 0.0, 0.0],
                               [0.0, 0, -1.0],
                               [0.0, -1.0, 0]])
    view_mat_shadow = np.array([[0.0, -1.0, 0.0],
                               [0.0, 0, -1.0],
                               [1.0, 0.0, 0]])
    P0 = get_projection_matrix(0)
    P1 = get_projection_matrix(1)
    line_pcd = o3d.geometry.LineSet()
    line_pcd_shadow = o3d.geometry.LineSet()
    viewers0.add_geometry(line_pcd)
    viewers0.add_geometry(line_pcd_shadow)
    viewers1.add_geometry(line_pcd)
    viewers1.add_geometry(line_pcd_shadow)

    transformed_pt = o3d.geometry.PointCloud()
    viewers3.add_geometry(transformed_pt)


    ##########################---load_teachnet_part---#########################################
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline.start(config)
    align_to_color = rs.align(rs.stream.color)
    model_path = 'weights_teachnet/new_early_teach_teleop.model'
    model = torch.load(model_path, map_location='cuda')
    model = model.cuda()
    print('load model {}'.format(model_path))
    ##########################---load_teachnet_part---###################################


    ctr = viewers0.get_view_control()
    ctr.set_lookat(np.array([0, 0.0, 300.0]))
    ctr.set_up((1, 0, 0))  # set the positive direction of the x-axis as the up direction
    ctr.set_front((0, 0, 1))  # set the positive direction of the x-axis toward you

    ctr1 = viewers1.get_view_control()
    ctr1.set_lookat(np.array([0, 300.0, 0]))
    ctr1.set_up((1, 0, 0))  # set the positive direction of the x-axis as the up direction
    ctr1.set_front((0, 1, 0))  # set the positive direction of the x-axis toward you

    ctr3 = viewers3.get_view_control()
    # ctr3.set_lookat(np.array([200, 300, 0]))
    # ctr3.set_up((0, 1, 0))  # set the positive direction of the x-axis as the up direction
    ctr3.set_front((-math.sin(np.pi/8)*0.8, 0.65, -math.cos(np.pi/8)*0.8))  # set the positive direction of the x-axis toward you
    hand_verts =0
    kpts_cam0, kpts_cam1, kpts_3d = run_mp(input_stream1, input_stream2, P0, P1)


