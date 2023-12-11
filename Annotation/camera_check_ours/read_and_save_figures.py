import cv2 as cv
import copy
cap0 = cv.VideoCapture(0)
cap1 = cv.VideoCapture(6)
i = 0
while True:
    ret0, frame0 = cap0.read()
    ret1, frame1 = cap1.read()
    frame00 = copy.deepcopy(frame0)
    frame10 = copy.deepcopy(frame1)
    # frame_shape = [1280, 720]
    # if frame0.shape[1] != 720:
    #     frame0 = frame0[:, frame_shape[1] // 2 - frame_shape[0] // 2:frame_shape[1] // 2 + frame_shape[0] // 2]
    #     frame1 = frame1[:, frame_shape[1] // 2 - frame_shape[0] // 2:frame_shape[1] // 2 + frame_shape[0] // 2]
    # frame0 = cv.cvtColor(frame0, cv.COLOR_BGR2RGB)
    # frame1 = cv.cvtColor(frame1, cv.COLOR_BGR2RGB)
    rows = 5  # number of checkerboard rows.
    columns = 8
    gray0 = cv.cvtColor(frame0, cv.COLOR_BGR2GRAY)
    gray1 = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)

    # find the checkerboard
    ret0, corners0 = cv.findChessboardCorners(gray0, (rows, columns), None)
    cv.drawChessboardCorners(frame0, (5, 8), corners0, ret1)
    ret1, corners1 = cv.findChessboardCorners(gray1, (rows, columns), None)
    cv.drawChessboardCorners(frame1, (5, 8), corners1, ret1)
    cv.imshow('cam0', frame0)
    cv.imshow('cam1', frame1)

    k = cv.waitKey(1)
    if k & 0xFF == ord('s'):
        save_1_file = 'frames/D2/' + 'd' +str(i) + '.png'
        save_1_files = 'frames/synched/' + 'd' +str(i) + '.png'
        cv.imwrite(save_1_file,frame00)
        cv.imwrite(save_1_files,frame00)
        save_2_file = 'frames/J2/' + 'j' +str(i) + '.png'
        save_2_files = 'frames/synched/' + 'j' +str(i) + '.png'
        cv.imwrite(save_2_file, frame10)
        cv.imwrite(save_2_files, frame10)
        i += 1
    if k & 0xFF == 27: break #27 is ESC key.
