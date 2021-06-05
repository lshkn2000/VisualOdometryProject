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
prev_img = None

current_pos_akaze = np.zeros((3, 1))
current_rot_akaze = np.eye(3)
current_pos_orb = np.zeros((3, 1))
current_rot_orb = np.eye(3)
current_pos_kaze = np.zeros((3, 1))
current_rot_kaze = np.eye(3)
current_pos_sift = np.zeros((3, 1))
current_rot_sift = np.eye(3)

camera_matrix = np.array([[718.8560, 0.0, 607.1928],
                          [0.0, 718.8560, 185.2157],
                          [0.0, 0.0, 1.0]])

output = len(glob.glob('./image_0/*.png'))  # 사진 개수

for index in range(output):
    img_file = os.path.join('./image_0/', '{:06d}.png').format(index)
    img = cv.imread('./image_0/{}'.format(img_file))
    # print("check: {} , img : {} ".format(index, img))
    # cvtColor는 안함  이미 변환되어 있으니까

    # keypoint detection and feature description
    # 1. AKAZE
    akaze = cv.AKAZE_create()
    kp_akaze = akaze.detect(img, None)  # find keypoints
    kp_akaze, des_akaze = akaze.compute(img, kp_akaze)  # compute descriptors with AKAZE
    # kp_akaze, des_akaze =akaze.detectAndCompute(img, None)
    # print("Keypoints AKAZE : {}".format(len(kp_akaze)))

    # 2, ORB
    orb = cv.ORB_create()
    kp_orb = orb.detect(img, None)  # find keypoints
    kp_orb, des_orb = orb.compute(img, kp_orb)  # compute descriptors with ORB
    # kp_orb, des_orb =orb.detectAndCompute(img, None)
    # print("Keypoints ORB : {}".format(len(kp_orb)))

    # 3. KAZE
    kaze = cv.KAZE_create()
    kp_kaze = kaze.detect(img, None)
    kp_kaze, des_kaze = kaze.compute(img, kp_kaze)
    # kp_kaze, des_kaze =kaze.detectAndCompute(img, None)
    # print("Keypoints KAZE : {}".format(len(kp_kaze)))

    # 4. SIFT
    sift = cv.SIFT_create()
    kp_sift = sift.detect(img, None)
    kp_sift, des_sift = sift.compute(img, kp_sift)
    # print("Keypoints SIFT : {}".format(len(kp_sift)))
    # kp_sift, des_sift =sift.detectAndCompute(img, None)

    if prev_img is None:
        prev_img = img
        prev_keypoint_akaze = kp_akaze
        prev_keypoint_orb = kp_orb
        prev_keypoint_kaze = kp_kaze
        prev_keypoint_sift = kp_sift
        #
        prev_des_akaze = des_akaze
        prev_des_orb = des_orb
        prev_des_kaze = des_kaze
        prev_des_sift = des_sift
        continue

    points_akaze = np.array(list(map(lambda x: [x.pt], prev_keypoint_akaze)), dtype=np.float32).squeeze()
    points_orb = np.array(list(map(lambda x: [x.pt], prev_keypoint_orb)), dtype=np.float32).squeeze()
    points_kaze = np.array(list(map(lambda x: [x.pt], prev_keypoint_kaze)), dtype=np.float32).squeeze()
    points_sift = np.array(list(map(lambda x: [x.pt], prev_keypoint_sift)), dtype=np.float32).squeeze()

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

    pt1_akaze, st, err = cv.calcOpticalFlowPyrLK(prev_img, img, points_akaze, None)
    pt2_akaze = points_akaze
    pt1_orb, st, err = cv.calcOpticalFlowPyrLK(prev_img, img, points_orb, None)
    pt2_orb = points_orb
    pt1_kaze, st, err = cv.calcOpticalFlowPyrLK(prev_img, img, points_kaze, None)
    pt2_kaze = points_kaze
    pt1_sift, st, err = cv.calcOpticalFlowPyrLK(prev_img, img, points_sift, None)
    pt2_sift = points_sift

    # Essenstial Matrix
    E_akaze, mask_akaze = cv.findEssentialMat(pt1_akaze, pt2_akaze, camera_matrix, cv.RANSAC, 0.999, 1.0, None)
    E_orb, mask_orb = cv.findEssentialMat(pt1_orb, pt2_orb, camera_matrix, cv.RANSAC, 0.999, 1.0, None)
    E_kaze, mask_kaze = cv.findEssentialMat(pt1_kaze, pt2_kaze, camera_matrix, cv.RANSAC, 0.999, 1.0, None)
    E_sift, mask_sift = cv.findEssentialMat(pt1_sift, pt2_sift, camera_matrix, cv.RANSAC, 0.999, 1.0, None)

    pt1_akaze, R_akaze, t_akaze, mask_akaze = cv.recoverPose(E_akaze, pt2_akaze, pt1_akaze, camera_matrix)
    pt1_orb, R_orb, t_orb, mask_orb = cv.recoverPose(E_orb, pt2_orb, pt1_orb, camera_matrix)
    pt1_kaze, R_kaze, t_kaze, mask_kaze = cv.recoverPose(E_kaze, pt2_kaze, pt1_kaze, camera_matrix)
    pt1_sift, R_sift, t_sift, mask_sift = cv.recoverPose(E_sift, pt2_sift, pt1_sift, camera_matrix)

    scale = 1.0

    # ground truth를 기반으로 scale 계산
    if ground_truth_exist:
        gt_pose = [ground_truth[index][0, 3], ground_truth[index][2, 3]]
        pre_gt_pose = [ground_truth[index - 1][0, 3], ground_truth[index - 1][2, 3]]
        scale = math.sqrt(math.pow((gt_pose[0] - pre_gt_pose[0]), 2.0) + math.pow((gt_pose[1] - pre_gt_pose[1]), 2.0))

    # 우선 ground truth가 없다면 scale 계산은 하지 않고
    current_pos_akaze += current_rot_akaze.dot(t_akaze) * scale
    current_rot_akaze = R_akaze.dot(current_rot_akaze)

    current_pos_orb += current_rot_orb.dot(t_orb) * scale
    current_rot_orb = R_orb.dot(current_rot_orb)

    current_pos_kaze += current_rot_kaze.dot(t_kaze) * scale
    current_rot_kaze = R_kaze.dot(current_rot_kaze)

    current_pos_sift += current_rot_sift.dot(t_sift) * scale
    current_rot_sift = R_sift.dot(current_rot_sift)

    # ground truth plot
    position_axes.scatter(ground_truth[index][0, 3], ground_truth[index][2, 3], s=0.5, c='red')

    # odometry plot
    position_axes.scatter(-current_pos_akaze[0][0], -current_pos_akaze[2][0], s=2, c='gray')
    position_axes.scatter(-current_pos_orb[0][0], -current_pos_orb[2][0], s=2, c='red')
    position_axes.scatter(-current_pos_kaze[0][0], -current_pos_kaze[2][0], s=2, c='blue')
    position_axes.scatter(-current_pos_sift[0][0], -current_pos_sift[2][0], s=2, c='black')
    plt.pause(.01)

    plot_img = cv.drawKeypoints(img, kp_akaze, None)

    # cv.imshow('feature', plot_img)
    # cv.waitKey(1)

    prev_img = img
    prev_keypoint_akaze = kp_akaze
    prev_keypoint_orb = kp_orb
    prev_keypoint_kaze = kp_kaze
    prev_keypoint_sift = kp_sift

    prev_des_akaze = des_akaze
    prev_des_orb = des_orb
    prev_des_kaze = des_kaze
    prev_des_sift = des_sift

position_figure.savefig("position_plot.png")