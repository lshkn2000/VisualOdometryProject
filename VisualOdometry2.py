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
#######################################################################
# parameters for lucas kanade optical flow
lk_params = dict(winSize=(21, 21),criteria=(cv.TERM_CRITERIA_EPS |cv.TERM_CRITERIA_COUNT, 30, 0.03))
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
##################################################################################

prev_img = None

current_pos = np.zeros((3, 1))
current_rot = np.eye(3)

output = len(glob.glob('./image_0/*.png'))  # 사진 개수

# keypoint detection and feature description
# 5. fast featureDetector
feature_detector = cv.FastFeatureDetector_create(threshold=25, nonmaxSuppression=True)

for index in range(output):
    img_file = os.path.join('./image_0/', '{:06d}.png').format(index)
    img = cv.imread('./image_0/{}'.format(img_file))
    # print("check: {} , img : {} ".format(index, img))
    # cvtColor는 안함  이미 변환되어 있으니까

    kp = feature_detector.detect(img, None)
    if prev_img is None:
        prev_img = img
        prev_kp = kp
        continue

    points = np.array(list(map(lambda x: [x.pt], prev_kp)), dtype=np.float32)

    #############################points matching ##################################
    #     # BFMatcher with default params
    #     # brute-force matcher
    #     # 사이즈가 커지면 속도 저하
    #     bf = cv.BFMatcher()
    #     matches_akaze = bf.knnMatch(prev_des_akaze, des_akaze, k=2)
    #     matches_orb = bf.knnMatch(prev_des_orb, des_orb, k=2)
    #     matches_kaze = bf.knnMatch(prev_des_kaze, des_kaze, k=2)
    #     matches_sift = bf.knnMatch(prev_des_sift, des_sift, k=2)

    #     # Apply ratio test
    #     good_matches_akaze = []
    #     good_matches_orb = []
    #     good_matches_kaze = []
    #     good_matches_sift = []

    #     pt1_akaze, pt2_akaze = [], []
    #     pt1_orb, pt2_orb = [],[]
    #     pt1_kaze, pt2_kaze = [],[]
    #     pt1_sift, pt2_sift = [],[]

    #     for m, n in matches_akaze:
    #         if m.distance < 0.75 * n.distance:
    #             good_matches_akaze.append([m])
    #             pt1_akaze.append(prev_keypoint_akaze[m.queryIdx].pt)
    #             pt2_akaze.append(kp_akaze[m.trainIdx].pt)
    #     for m,n in matches_orb:
    #         if m.distance < 0.75*n.distance:
    #             good_matches_orb.append([m])
    #             pt1_orb.append(prev_keypoint_orb[m.queryIdx].pt)
    #             pt2_orb.append(kp_orb[m.trainIdx].pt)
    #     for m,n in matches_kaze:
    #         if m.distance < 0.75*n.distance:
    #             good_matches_kaze.append([m])
    #             pt1_kaze.append(prev_keypoint_kaze[m.queryIdx].pt)
    #             pt2_kaze.append(kp_kaze[m.trainIdx].pt)
    #     for m,n in matches_sift:
    #         if m.distance < 0.75*n.distance:
    #             good_matches_sift.append([m])
    #             pt1_sift.append(prev_keypoint_sift[m.queryIdx].pt)
    #             pt2_sift.append(kp_sift[m.trainIdx].pt)

    #     pt1_akaze, pt2_akaze = np.float32(pt1_akaze), np.float32(pt2_akaze)
    #     pt1_orb, pt2_orb = np.float32(pt1_orb), np.float32(pt2_orb)
    #     pt1_kaze, pt2_kaze = np.float32(pt1_kaze), np.float32(pt2_kaze)
    #     pt1_sift, pt2_sift = np.float32(pt1_sift), np.float32(pt2_sift)
    #################################################################################

    p1, st, err = cv.calcOpticalFlowPyrLK(prev_img, img, points,None, **lk_params)

    # Essenstial Matrix
    E, mask = cv.findEssentialMat(p1, points, camera_matrix, cv.RANSAC, 0.999, 1.0, None)

    points, R, t, mask = cv.recoverPose(E, p1, points, camera_matrix)

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
    position_axes.scatter(current_pos[0][0], current_pos[2][0], s=2, c='gray')
    plt.pause(.01)

    plot_img = cv.drawKeypoints(img, kp, None)

    # cv.imshow('feature', plot_img)
    # cv.waitKey(1)

    prev_img = img
    prev_kp = kp

position_figure.savefig("position_plot_2.png")