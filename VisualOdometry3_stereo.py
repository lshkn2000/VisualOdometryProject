import os
import glob
import numpy as np
import math
import cv2 as cv
from matplotlib import pyplot as plt

##################plot graph#######################
position_figure = plt.figure()
position_axes = position_figure.add_subplot(1, 1, 1)
# error_figure = plt.figure()
# rotation_error_axes = error_figure.add_subplot(1, 1, 1)
# rotation_error_list = []
# frame_index_list = []

position_axes.set_aspect('equal', adjustable='box')
####################################################
###############ground truth###################################
gt_file = glob.glob('./poses/01.txt')
ground_truth_exist = True
ground_truth = []
with open(*gt_file) as f:
    gt_lines = f.readlines()

    for gt_line in gt_lines:
        pose = np.array(gt_line.split()).reshape((3, 4)).astype(np.float32)
        ground_truth.append(pose)

#######################################################################
# parameters for lucas kanade optical flow
lk_params = dict(winSize=(21, 21), criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 30, 0.03))
#######################################################################
#####################camera setting information ###############################
camera_matrix = np.array([[718.8560, 0.0, 607.1928],
                          [0.0, 718.8560, 185.2157],
                          [0.0, 0.0, 1.0]])
f_l = camera_matrix[0][0]
f_r = camera_matrix[0][0]
c_x_l = camera_matrix[0][2]
c_y_l = camera_matrix[1][2]
c_x_r = camera_matrix[0][2]
c_y_r = camera_matrix[1][2]
baseline = -3.861448000000e+02

# projection Matrix  Left
P1 = np.array([[718.8560, 0.0, 607.1928, 0.0],
               [0.0, 718.8560, 185.2157, 0.0],
               [0, 0, 1, 0]])
# projection Matrix  RIght
P2 = np.array([[718.8560, -386.1448, 607.1928, 0.0],
               [0.0, 718.8560, 185.2157, 0.0],
               [0.0, 0.0, 1.0, 0.0]])
# decomposeProjection Matrix
k_l, r_l, t_l, _, _, _, _ = cv.decomposeProjectionMatrix(P1)
t_l = (t_l / t_l[3])[:3]
k_r, r_r, t_r, _, _, _, _ = cv.decomposeProjectionMatrix(P2)
t_r = (t_r / t_r[3])[:3]


##################################################################################
def extract_feature(img, detector='ORB', mask=None):
    # keypoint detection and feature description
    # 1. AKAZE
    if detector == 'AKAZE':
        detector = cv.AKAZE_create()
    # 2, ORB
    if detector == 'ORB':
        detector = cv.ORB_create()
    # 3. KAZE
    if detector == 'KAZE':
        detector = cv.KAZE_create()
    # 4. SIFT
    if detector == 'SIFT':
        detector = cv.SIFT_create()
        # kp_sift, des_sift =sift.detectAndCompute(img, None)
    kp, des = detector.detectAndCompute(img, mask)

    return kp, des


def feature_matching(des1, des2, detector='ORB', k=2):
    if detector == "ORB":
        bf = cv.BFMatcher_create(cv.NORM_HAMMING2, crossCheck=False)
    elif detector == "SIFT":
        bf = cv.BFMatcher_create(cv.NORM_L2, crossCheck=False)
    matches = bf.knnMatch(des1, des2, k)

    matches = sorted(matches, key=lambda x: x[0].distance)

    filtered_matches = []

    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            filtered_matches.append(m)

    return filtered_matches


###################################################################################
def Depth_map(img_L, img_R, intrinsic_matrix = camera_matrix):
    # disparity 계산 + 3d point 계산
    stereo = cv.StereoBM_create(numDisparities=16, blockSize=15)
    disparity = stereo.compute(img_L, img_R).astype(np.float32)  # Disparity boundary is (-16 ~ 496)
    disparity_Norm = np.divide(disparity, 16.0)  # # Disparity boundary is (-1 ~ 496/16)

    # 참고 https://pythonq.com/so/opencv/706363 , https://stackoverflow.com/questions/41503561/whats-the-difference-between-reprojectimageto3dopencv-and-disparity-to-3d-coo
    # depth 계산 원리
    # Z = np.divide((baseline * f_l), disparity)
    # plt.imshow(Z,'gray')
    f_l = intrinsic_matrix[0][0]
    f_r = intrinsic_matrix[0][0]
    c_x_l = intrinsic_matrix[0][2]
    c_y_l = intrinsic_matrix[1][2]
    c_x_r = intrinsic_matrix[0][2]
    c_y_r = intrinsic_matrix[1][2]
    baseline = 0.54

    disparity_Norm[disparity_Norm == 0.0] = 0.1
    disparity_Norm[disparity_Norm == -1.0] = 0.1
    depth_map = np.ones(disparity_Norm.shape)
    depth_map = f_l * baseline / disparity_Norm
    # Q = np.float32([[1, 0, 0, -c_x_l],
    #                 [0, 1, 0, -c_y_l],
    #                 [0, 0, 0, f_l],
    #                 [0, 0, -1 / baseline, 0]])
    #
    # depth_map = cv.reprojectImageTo3D(disparity_Norm, Q)

    return disparity_Norm, depth_map


################################################################################
def PNP_estimate_motion(prev_kp_L, kp_L, matches, depth_map, depth_limit=3000, intrinsic_matrix = camera_matrix):
    R = np.eye(3)
    T = np.zeros((3, 1))

    prev_img_L_point = np.float32([prev_kp_L[m.queryIdx].pt for m in matches])
    img_L_point = np.float32([kp_L[m.trainIdx].pt for m in matches])

    f_l = intrinsic_matrix[0][0]
    f_r = intrinsic_matrix[0][0]
    c_x_l = intrinsic_matrix[0][2]
    c_y_l = intrinsic_matrix[1][2]
    c_x_r = intrinsic_matrix[0][2]
    c_y_r = intrinsic_matrix[1][2]
    object_points = np.zeros((0, 3))
    delete = []
    if depth_map is not None:
        for i, (u, v) in enumerate(prev_img_L_point):
            z = depth_map[int(v), int(u)]
            if z > depth_limit:
                delete.append(i)
                continue

            x = z * (u - c_x_l) / f_l
            y = z * (v - c_y_l) / f_l
            object_points = np.vstack([object_points, np.array([x, y, z])])

        prev_img_L_point = np.delete(prev_img_L_point, delete, 0)
        img_L_point = np.delete(img_L_point, delete, 0)

        # PNP + RANSAC
        _, rvec, T, inliers = cv.solvePnPRansac(object_points, img_L_point, intrinsic_matrix, None)

        # Rodrigues공식으로 3x3으로 변환
        R = cv.Rodrigues(rvec)[0]

    return R, T, prev_img_L_point, img_L_point


################################################################################
prev_img_L = None
prev_kp_L = None

current_pos = np.zeros((3, 1))
current_rot = np.eye(3)
T_tot = np.eye(4) # homogeneous transformation matrix

output = len(glob.glob('./image_0/*.png'))  # 사진 개수

for index in range(output):
    img_file_L = os.path.join('./image_0/', '{:06d}.png').format(index)
    name_L = './image_0/' + img_file_L
    img_L = cv.imread(name_L, 0)

    img_file_R = os.path.join('./image_1/', '{:06d}.png').format(index)
    name_R = './image_1/'+img_file_R
    img_R = cv.imread(name_R, 0)
    # print("check: {} , img : {} ".format(index, img))
    # cvtColor는 안함  이미 변환되어 있으니까

    # disparity 계산 + 3d point 계산
    disparity, depth_map = Depth_map(img_L, img_R, intrinsic_matrix=camera_matrix)

    # 1. feature Detection 이전사진과 현재사진을 이용
    kp_L, des_L = extract_feature(img_L, "SIFT", None)

    if prev_img_L is None:
        prev_img_L = img_L
        prev_kp_L = kp_L
        prev_des_L = des_L
        continue

    points = np.array(list(map(lambda x: [x.pt], prev_kp_L)), dtype=np.float32)

    # 2. 두 이미지 사이의 매칭
    matches = feature_matching(prev_des_L, des_L, "SIFT", k=2)

    # 3. Pose 추정
    R, t, prev_img_L_point, img_L_point = PNP_estimate_motion(prev_kp_L, kp_L, matches, depth_map, depth_limit=1000, intrinsic_matrix=camera_matrix)
    # H = np.eye(4) # H = [R|T]
    # H[:3,:3], H[:3, 3] = R, t.T
    # T_tot = T_tot.dot(np.linalg.inv(H))
    # R, t = T_tot[:3,:3], T_tot[:3, 3]
    # t = t.reshape(3,1)


    #trajectory = np.zeros((index, 3, 4))
    #trajectory[i + 1, :, :] = T_tot[:3, :]
    ####################
    #     # 2.feature tracking (KLT : Kanade-Lucas-Tomasi)
    #     pt1, st, err = cv2.calcOpticalFlowPyrLK(prev_img_L,img_L, points,None, **lk_params)

    #     # Essenstial Matrix
    #     E, mask = cv.findEssentialMat(pt1, points, camera_matrix, cv.RANSAC, 0.999, 1.0, None)

    #     points, R, t, mask = cv.recoverPose(E, pt1, points, camera_matrix)
    #####################
    scale = 1.0

    # ground truth를 기반으로 scale 계산
    if ground_truth_exist:
        gt_pose = [ground_truth[index][0, 3], ground_truth[index][2, 3]]
        pre_gt_pose = [ground_truth[index - 1][0, 3], ground_truth[index - 1][2, 3]]
        scale = math.sqrt(math.pow((gt_pose[0] - pre_gt_pose[0]), 2.0) + math.pow((gt_pose[1] - pre_gt_pose[1]), 2.0))

    # 우선 ground truth가 없다면 scale 계산은 하지 않고
    current_pos += current_rot.dot(t) * scale
    current_rot = R.dot(current_rot)

    # ground truth plot
    position_axes.scatter(ground_truth[index][0, 3], ground_truth[index][2, 3], s=2, c='red')

    # odometry plot
    position_axes.scatter(current_pos[0][0], -current_pos[2][0], s=2, c='gray')
    plt.pause(.01)

    plot_img = cv.drawKeypoints(img_L, kp_L, None)

    # cv.imshow('feature', plot_img)
    # cv.waitKey(1)

    prev_img_L = img_L
    prev_kp_L = kp_L
    prev_des_L = des_L

position_figure.savefig("position_plot_stereo.png")