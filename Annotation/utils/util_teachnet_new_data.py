import torch
# from model_minimal.detnet import detnet
import open3d
from numba import jit
from model.model import *
import cv2

def trans_2_quat(R):
    quat = torch.zeros(4).cuda()
    quat[3] = 0.5 * ((1 + R[0][0] + R[1][1] + R[2][2]).sqrt()) # qw,qx,qy,qz
    quat[0] = (R[2][1] - R[1][2]) / (4 * quat[3])
    quat[1] = (R[0][2] - R[2][0]) / (4 * quat[3])
    quat[2] = (R[1][0] - R[0][1]) / (4 * quat[3])
    return quat #qx, qy, qz, qw

####minimal_hand part####
# def load_minimal_model():
#     device = torch.device('cuda:0')
#     module = detnet().to(device)
#     print('load model_minimal start')
#     check_point = torch.load('/home/hjl/paper_code/Minimal-Hand-pytorch/my_results/checkpoints/ckp_detnet_106.pth',
#                              map_location=device)
#     model_state = module.state_dict()
#     state = {}
#     for k, v in check_point.items():
#         if k in model_state:
#             state[k] = v
#         else:
#             print(k, ' is NOT in current model_minimal')
#     model_state.update(state)
#     module.load_state_dict(model_state)
#     print('load model_minimal finished')
#     return module


def init_vis(viewers):
    viewers.create_window(width=480, height=480, left=1500, top=0, window_name='hand_joint')
    pointcloud = open3d.geometry.PointCloud()
    line_pcd = open3d.geometry.LineSet()
    FOR1 = open3d.geometry.TriangleMesh.create_coordinate_frame(size=1.5, origin=[0, 0, 0])
    viewers.add_geometry(line_pcd)
    viewers.add_geometry(pointcloud)
    viewers.add_geometry(FOR1)
    FOR2 = open3d.geometry.TriangleMesh.create_coordinate_frame()
    viewers.add_geometry(FOR2)
    return pointcloud,line_pcd,FOR1,FOR2
####minimal_hand part####

####teachnet_part####
@jit(nopython=True)
def surround(i, j, xl, yl, add=1):
    sur = []
    if i - add >= 0:
        sur.append([i - add, j])
    if j - add >= 0:
        sur.append([i, j - add])
    if i + add < xl:
        sur.append([i + add, j])
    if j + add < yl:
        sur.append([i, j + add])
    return sur
@jit(nopython=True)
def inner(inner_edge, img, zero_as_infty, fore_thresh, mask, gap, thresh, x, y, w, l, add):
    for i, j in zip(x, y):
        sur = surround(i, j, w, l, add)
        for s in sur:
            xx, yy = s
            if gap < abs(img[xx, yy] - img[i, j]):
                if zero_as_infty or abs(img[xx, yy] - img[i, j]) < thresh:
                    if img[xx, yy] > img[i, j]:
                        if img[i, j] <= fore_thresh:
                            mask[xx, yy] = 0
                            new_inner_edge = np.array([i, j]).reshape(1, 2)
                            inner_edge = np.vstack((inner_edge, new_inner_edge))
                    else:
                        if img[xx, yy] <= fore_thresh:
                            mask[i, j] = 0
                            new_inner_edge = np.array([xx, yy]).reshape(1, 2)
                            inner_edge = np.vstack((inner_edge, new_inner_edge))
    return inner_edge, mask
def seg_hand_depth(img, gap=100, thresh=500, padding=10, output_size=100, scale=10, add=5, box_z=250,
                   zero_as_infty=False, fore_p_thresh=300, label=None):
    img = img.astype(np.float32)
    img[np.where(img > 450)] = 0
    if zero_as_infty:
        # TODO: for some sensor that maps infty as 0, we should override them
        thresh = np.inf
        his = np.histogram(img[img != 0])
        sum_p = 0
        for i in range(len(his[0])):
            sum_p += his[0][i]
            if his[0][i] == 0 and sum_p > fore_p_thresh:
                fore_thresh = his[1][i]
                break
        else:
            fore_thresh = np.inf
    else:
        fore_thresh = np.inf
    mask = np.ones_like(img)
    w, l = img.shape
    x = np.linspace(0, w - 1, w // scale)
    y = np.linspace(0, l - 1, l // scale)
    grid = np.meshgrid(x, y)
    x = grid[0].reshape(-1).astype(np.int32)
    y = grid[1].reshape(-1).astype(np.int32)
    inner_edge = []
    if zero_as_infty:
        img[img == 0] = np.iinfo(np.uint16).max

    # morphlogy
    open_mask = np.zeros_like(img)
    open_mask[img != np.iinfo(np.uint16).max] = 1
    tmp = open_mask.copy()
    tmp = cv2.morphologyEx(tmp, cv2.MORPH_OPEN, np.ones((3, 3)))
    open_mask -= tmp
    img[open_mask.astype(np.bool_)] = np.iinfo(np.uint16).max

    inner_edge = np.array([1, 1]).reshape(1, 2)
    inner_edge = inner_edge.astype(np.int64)
    inner_edge, mask = inner(inner_edge, img, zero_as_infty, fore_thresh, mask, gap, thresh, x, y, w, l, add)
    inner_edge = inner_edge[1:]

    mask = mask.astype(np.bool_)
    edge_x, edge_y = np.where(mask == 0)
    if edge_x.size == 0 or edge_y.size == 0:
        x_min, x_max, y_min, y_max = 0, 479, 56, 618
        # print(1)
    else:
        x_min, x_max = np.min(edge_x), np.max(edge_x)
        y_min, y_max = np.min(edge_y), np.max(edge_y)
        # print(x_min, x_max, y_min, y_max)

    x_min = max(0, x_min - padding)
    x_max = min(x_max + padding, w - 1)
    y_min = max(0, y_min - padding)
    y_max = min(y_max + padding, l - 1)
    if x_max - x_min > y_max - y_min:
        delta = (x_max - x_min) - (y_max - y_min)
        y_min -= delta / 2
        y_max += delta / 2
    else:
        delta = (y_max - y_min) - (x_max - x_min)
        x_min -= delta / 2
        x_max += delta / 2
    x_min = int(max(0, x_min))
    x_max = int(min(x_max, w - 1))
    y_min = int(max(0, y_min))
    y_max = int(min(y_max, l - 1))

    edge_depth = []
    for (x, y) in inner_edge:
        edge_depth.append(img[x, y])
    avg_depth = np.sum(edge_depth) / float(len(edge_depth))
    # avg_depth = 40
    depth_min = max(avg_depth - box_z / 2, 0)
    depth_max = avg_depth + box_z / 2
    seg_area = img.copy()
    seg_area[seg_area < depth_min] = depth_min
    seg_area[seg_area > depth_max] = depth_max
    # normalized
    seg_area = ((seg_area - avg_depth) / (box_z / 2))  # [-1, 1]
    seg_area = ((seg_area + 1) / 2.) * 255.  # [0, 255]

    output = seg_area[x_min:x_max, y_min:y_max]
    output = cv2.resize(output, (output_size, output_size)).astype(np.uint16)
    # rgb = rgb.copy()
    # rgb = rgb[x_min-10:x_max+100, y_min-10:y_max+100]
    # rgb = cv2.resize(rgb, (output_size*2, output_size*2))
    if label is not None:
        label = label.astype(np.float32)
        label[:, 0] = label[:, 0] - y_min
        label[:, 1] = label[:, 1] - x_min
        label[:, 0] *= (float(output_size) / (y_max - y_min + 1))
        label[:, 1] *= (float(output_size) / (x_max - x_min + 1))
        # if need normalized label
        # label[:, 2] = (label[:, 2] - avg_depth) / (box_z / 2)
        # label[:, 2] = ((label[:, 2] + 1) / 2.) * 255.  # [0, 255]
        label = label.round().astype(np.int32)
        return output, label, np.array([x_max, x_min, y_max, y_min])
    else:
        # return output, rgb
        return output
def clip(x, maxv=None, minv=None):
    if maxv is not None and x > maxv:
        x = maxv
    if minv is not None and x < minv:
        x = minv
    return x
def joint_cal(img, model, isbio=False):
    # start = rospy.Time.now().to_sec()

    # run the model
    feature = test(model, img).cuda()

    # Apply joint-space velocity limit
    # previous_joint = np.zeros([22])
    # max_joint_velocity = 1.0472
    # joint_diff = np.absolute(feature - previous_joint)
    # speed_factor = np.ones_like(feature).astype(np.float32)
    # index = joint_diff > max_joint_velocity
    # speed_factor[index] = max_joint_velocity / joint_diff[index]
    # feature = feature * speed_factor + previous_joint * (1-speed_factor)

    previous_joint = torch.zeros([22]).cuda()
    max_joint_velocity = 1.0472
    joint_diff = torch.absolute(feature - previous_joint)
    speed_factor = torch.ones_like(feature, dtype=torch.float32)#.astype(np.float32)
    index = joint_diff > max_joint_velocity
    speed_factor[index] = max_joint_velocity / joint_diff[index]
    feature = feature * speed_factor + previous_joint * (1 - speed_factor)


    joint = [0.0, 0.0]
    joint += feature.tolist()
    joint[22] *= 6
    if isbio:
        joint[5] = 0.3498509706185152
        joint[10] = 0.3498509706185152
        joint[14] = 0.3498509706185152
        joint[18] = 0.3498509706185152
        joint[23] = 0.3498509706185152

    # joints crop
    joint[2] = clip(joint[2], 0.349, -0.349)
    joint[3] = clip(joint[3], 1.57, 0)
    joint[4] = clip(joint[4], 1.57, 0)
    joint[5] = clip(joint[5], 1.57, 0)

    joint[6] = clip(joint[6], 0.785, 0)

    joint[7] = clip(joint[7], 0.349, -0.349)
    joint[8] = clip(joint[8], 1.57, 0)
    joint[9] = clip(joint[9], 1.57, 0)
    joint[10] = clip(joint[10], 1.57, 0)

    joint[11] = clip(joint[11], 0.349, -0.349)
    joint[12] = clip(joint[12], 1.57, 0)
    joint[13] = clip(joint[13], 1.57, 0)
    joint[14] = clip(joint[14], 1.57, 0)

    joint[15] = clip(joint[15], 0.349, -0.349)
    joint[16] = clip(joint[16], 1.57, 0)
    joint[17] = clip(joint[17], 1.57, 0)
    joint[18] = clip(joint[18], 1.57, 0)

    # joint[19] = clip(joint[19], 1.047, -1.047)
    joint[19] = clip(joint[19], 0.6, -1.047)
    # joint[20] = clip(joint[20], 1.222, 0)
    # joint[20] = clip(joint[20], 1.222, 0.500)
    joint[20] = clip(joint[20], 1.222, 1.122)
    joint[21] = clip(joint[21], 0.209, -0.209)
    joint[22] = clip(joint[22], 0.524, -0.524)
    joint[23] = clip(joint[23], 1.57, 0)

    return joint
def test(model, img):
    model.eval()
    torch.set_grad_enabled(False)

    # assert(img.shape == (input_size, input_size))
    # img = img[np.newaxis, np.newaxis, ...].cpu()
    # img = torch.Tensor(img)
    img = img.cuda()

    # human part
    embedding_human, joint_human = model(img, is_human=True)
    joint_upper_range = torch.tensor([0.349, 1.571, 1.571, 1.571, 0.785, 0.349, 1.571, 1.571,
                                      1.571, 0.349, 1.571, 1.571, 1.571, 0.349, 1.571, 1.571,
                                      1.571, 1.047, 1.222, 0.209, 0.524, 1.571])
    joint_lower_range = torch.tensor([-0.349, 0, 0, 0, 0, -0.349, 0, 0, 0, -0.349, 0, 0, 0,
                                      -0.349, 0, 0, 0, -1.047, 0, -0.209, -0.524, 0])
    joint_upper_range = joint_upper_range.cuda()
    joint_lower_range = joint_lower_range.cuda()
    joint_human = joint_human * (joint_upper_range - joint_lower_range) + joint_lower_range
    return joint_human[0]#.cpu().data.numpy()[0]
####teachnet_part####


