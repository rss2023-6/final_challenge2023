#!/usr/bin/env python
import numpy as np
import cv2
# from cv_bridge import CvBridge, CvBridgeError

def homography_matrix():
  PTS_IMAGE_PLANE =   [[180, 172],
                    [315,226],
                    [331, 170],
                    [494, 159],
                    [432, 210],
                    [217, 220],]
  
  PTS_GROUND_PLANE = [[110.0, 50.0],
                    [30.0, 0.0],
                    [110, 0.0],
                    [110.0, -50.0],
                    [30.0, -10.0],
                    [30.0, 10.0]]
  
  
  # [[10.0, 0.0],
  #                     [20.0, 0.0],
  #                     [25.0, 0.0],
  #                     [10.0, -8.0],
  #                     [20.0, -10.0],
  #                     [20.0, 10.0],]
  METERS_PER_INCH = 0.0254

  if not len(PTS_GROUND_PLANE) == len(PTS_IMAGE_PLANE):
      rospy.logerr("ERROR: PTS_GROUND_PLANE and PTS_IMAGE_PLANE should be of same length")

  #Initialize data into a homography matrix

  np_pts_ground = np.array(PTS_GROUND_PLANE)
  np_pts_ground = np_pts_ground * METERS_PER_INCH
  np_pts_ground = np.float32(np_pts_ground[:, np.newaxis, :])

  np_pts_image = np.array(PTS_IMAGE_PLANE)
  np_pts_image = np_pts_image * 1.0
  np_pts_image = np.float32(np_pts_image[:, np.newaxis, :])

  h, err = cv2.findHomography(np_pts_image, np_pts_ground)
  return h

def transform_image(img):
    height, width, channels = img.shape
    height_cutoff = int(height*0.4)
    img[:int(height*0.4), 0:width] = 0
    width = 400
    height = 300
    img_out = cv2.warpPerspective(img, M, (width, height))
    # img_out = cv2.flip(img_out, 0)
    # img_out = cv2.flip(img_out, 1)
    img = img_out
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)[1]
    plt.figure()
    plt.imshow(thresh)
    return thresh