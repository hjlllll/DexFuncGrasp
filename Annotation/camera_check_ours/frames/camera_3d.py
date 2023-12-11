import cv2 as cv
import glob
import numpy as np
import matplotlib.pyplot as plt


def calibrate_camera(images_folder):
    images_names = glob.glob(images_folder)
    images = []
    for imname in images_names:
        im = cv.imread(imname, 1)
        images.append(im)

    # plt.figure(figsize = (10,10))
    # ax = [plt.subplot(2,2,i+1) for i in range(4)]
    #
    # for a, frame in zip(ax, images):
    #     a.imshow(frame[:,:,[2,1,0]])
    #     a.set_xticklabels([])
    #     a.set_yticklabels([])
    # plt.subplots_adjust(wspace=0, hspace=0)
    # plt.show()

    # criteria used by checkerboard pattern detector.
    # Change this if the code can't find the checkerboard
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    rows = 5  # number of checkerboard rows.
    columns = 8  # number of checkerboard columns.
    world_scaling = 1.  # change this to the real world square size. Or not.

    # coordinates of squares in the checkerboard world space
    objp = np.zeros((rows * columns, 3), np.float32)
    objp[:, :2] = np.mgrid[0:rows, 0:columns].T.reshape(-1, 2)
    objp = world_scaling * objp

    # frame dimensions. Frames should be the same size.
    width = images[0].shape[1]
    height = images[0].shape[0]

    # Pixel coordinates of checkerboards
    imgpoints = []  # 2d points in image plane.

    # coordinates of the checkerboard in checkerboard world space.
    objpoints = []  # 3d point in real world space

    for frame in images:
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        # find the checkerboard
        ret, corners = cv.findChessboardCorners(gray, (rows, columns), None)

        if ret == True:
            # Convolution size used to improve corner detection. Don't make this too large.
            conv_size = (11, 11)

            # opencv can attempt to improve the checkerboard coordinates
            corners = cv.cornerSubPix(gray, corners, conv_size, (-1, -1), criteria)
            cv.drawChessboardCorners(frame, (rows, columns), corners, ret)
            cv.imshow('img', frame)
            cv.waitKey(500)

            objpoints.append(objp)
            imgpoints.append(corners)

    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, (width, height), None, None)
    """
    设置标定板的世界坐标
    """
    world_point = np.zeros((rows * columns, 3), np.float32)
    world_point[:, :2] = np.mgrid[:rows, :columns].T.reshape(-1, 2)
    axis = np.float32([[3, 0, 0], [0, 3, 0], [0, 0, -3]]).reshape(-1, 3)
    img = images[0]
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    corner = imgpoints[0]
    corners_ex = cv.cornerSubPix(gray, corner, (11, 11), (-1, -1), criteria)
    _, rvec, tvec, inliers = cv.solvePnPRansac(world_point, corners_ex, mtx, dist)
    rotation_m, _ = cv.Rodrigues(rvec)
    rotation_t = np.hstack([rotation_m, tvec])
    rotation_t_homogeneous_mtx = np.vstack([rotation_t, np.array([0, 0, 0, 1])])
    RT = np.zeros([3, 4])
    for i in range(3):
        for j in range(3):
            RT[i][j] = rotation_m[i][j]
    for i in range(3):
        RT[i][3] = mtx[1][i]
    print('RT:', RT)
    print('rmse:', ret)
    print('camera matrix:\n', mtx)
    print('distortion coeffs:', dist)
    print('Rs:\n', rvecs)
    print('Ts:\n', tvecs)

    return mtx, dist


def stereo_calibrate(mtx1, dist1, mtx2, dist2, frames_folder):
    # read the synched frames
    images_names = glob.glob(frames_folder)
    images_names = sorted(images_names)
    c1_images_names = images_names[:len(images_names) // 2]
    c2_images_names = images_names[len(images_names) // 2:]

    c1_images = []
    c2_images = []
    for im1, im2 in zip(c1_images_names, c2_images_names):
        _im = cv.imread(im1, 1)
        c1_images.append(_im)

        _im = cv.imread(im2, 1)
        c2_images.append(_im)

    # change this if stereo calibration not good.
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.0001)

    rows = 5  # number of checkerboard rows.
    columns = 8  # number of checkerboard columns.
    world_scaling = 1.  # change this to the real world square size. Or not.

    # coordinates of squares in the checkerboard world space
    objp = np.zeros((rows * columns, 3), np.float32)
    objp[:, :2] = np.mgrid[0:rows, 0:columns].T.reshape(-1, 2)
    objp = world_scaling * objp

    # frame dimensions. Frames should be the same size.
    width = c1_images[0].shape[1]
    height = c1_images[0].shape[0]

    # Pixel coordinates of checkerboards
    imgpoints_left = []  # 2d points in image plane.
    imgpoints_right = []

    # coordinates of the checkerboard in checkerboard world space.
    objpoints = []  # 3d point in real world space

    for frame1, frame2 in zip(c1_images, c2_images):
        gray1 = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
        gray2 = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)
        c_ret1, corners1 = cv.findChessboardCorners(gray1, (5, 8), None)
        c_ret2, corners2 = cv.findChessboardCorners(gray2, (5, 8), None)

        if c_ret1 == True and c_ret2 == True:
            corners1 = cv.cornerSubPix(gray1, corners1, (11, 11), (-1, -1), criteria)
            corners2 = cv.cornerSubPix(gray2, corners2, (11, 11), (-1, -1), criteria)

            cv.drawChessboardCorners(frame1, (5, 8), corners1, c_ret1)
            cv.imshow('img', frame1)

            cv.drawChessboardCorners(frame2, (5, 8), corners2, c_ret2)
            cv.imshow('img2', frame2)
            cv.waitKey(500)

            objpoints.append(objp)
            imgpoints_left.append(corners1)
            imgpoints_right.append(corners2)

    stereocalibration_flags = cv.CALIB_FIX_INTRINSIC
    ret, CM1, dist1, CM2, dist2, R, T, E, F = cv.stereoCalibrate(objpoints, imgpoints_left, imgpoints_right, mtx1,
                                                                 dist1,
                                                                 mtx2, dist2, (width, height), criteria=criteria,
                                                                 flags=stereocalibration_flags)

    print(ret)
    return R, T




def triangulate(mtx1, mtx2, R, T):
    uvs1 = [[458, 86], [451, 164], [287, 181],
            [196, 383], [297, 444], [564, 194],
            [562, 375], [596, 520], [329, 620],
            [488, 622], [432, 52], [489, 56]]

    uvs2 = [[540, 311], [603, 359], [542, 378],
            [525, 507], [485, 542], [691, 352],
            [752, 488], [711, 605], [549, 651],
            [651, 663], [526, 293], [542, 290]]

    uvs1 = np.array(uvs1)
    uvs2 = np.array(uvs2)

    frame1 = cv.imread('testing/_C1.png')
    frame2 = cv.imread('testing/_C2.png')

    plt.imshow(frame1[:, :, [2, 1, 0]])
    plt.scatter(uvs1[:, 0], uvs1[:, 1])
    plt.show()  # this call will cause a crash if you use cv.imshow() above. Comment out cv.imshow() to see this.

    plt.imshow(frame2[:, :, [2, 1, 0]])
    plt.scatter(uvs2[:, 0], uvs2[:, 1])
    plt.show()  # this call will cause a crash if you use cv.imshow() above. Comment out cv.imshow() to see this

    # RT matrix for C1 is identity.
    RT1 = np.concatenate([np.eye(3), [[0], [0], [0]]], axis=-1)
    P1 = mtx1 @ RT1  # projection matrix for C1

    # RT matrix for C2 is the R and T obtained from stereo calibration.
    RT2 = np.concatenate([R, T], axis=-1)
    P2 = mtx2 @ RT2  # projection matrix for C2


    def DLT(P1, P2, point1, point2):

        A = [point1[1] * P1[2, :] - P1[1, :],
             P1[0, :] - point1[0] * P1[2, :],
             point2[1] * P2[2, :] - P2[1, :],
             P2[0, :] - point2[0] * P2[2, :]
             ]
        A = np.array(A).reshape((4, 4))
        # print('A: ')
        # print(A)

        B = A.transpose() @ A
        from scipy import linalg
        U, s, Vh = linalg.svd(B, full_matrices=False)

        print('Triangulated point: ')
        print(Vh[3, 0:3] / Vh[3, 3])
        return Vh[3, 0:3] / Vh[3, 3]

    p3ds = []
    for uv1, uv2 in zip(uvs1, uvs2):
        _p3d = DLT(P1, P2, uv1, uv2)
        p3ds.append(_p3d)
    p3ds = np.array(p3ds)

    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim3d(-15, 5)
    ax.set_ylim3d(-10, 10)
    ax.set_zlim3d(10, 30)

    connections = [[0, 1], [1, 2], [2, 3], [3, 4], [1, 5], [5, 6], [6, 7], [1, 8], [1, 9], [2, 8], [5, 9], [8, 9],
                   [0, 10], [0, 11]]
    for _c in connections:
        print(p3ds[_c[0]])
        print(p3ds[_c[1]])
        ax.plot(xs=[p3ds[_c[0], 0], p3ds[_c[1], 0]], ys=[p3ds[_c[0], 1], p3ds[_c[1], 1]],
                zs=[p3ds[_c[0], 2], p3ds[_c[1], 2]], c='red')
    ax.set_title('This figure can be rotated.')
    # uncomment to see the triangulated pose. This may cause a crash if youre also using cv.imshow() above.
    plt.show()


mtx1, dist1 = calibrate_camera(images_folder='D2/*')
mtx2, dist2 = calibrate_camera(images_folder='J2/*')



R, T = stereo_calibrate(mtx1, dist1, mtx2, dist2, 'synched/*')
print("rt:",R,T)
# print('mtx1:',mtx1, 'dist1:',dist1, 'mtx2:',mtx2, 'dist2:',dist2, 'r:',R,'t:', T)

# this call might cause segmentation fault error. This is due to calling cv.imshow() and plt.show()
triangulate(mtx1, mtx2, R, T)
