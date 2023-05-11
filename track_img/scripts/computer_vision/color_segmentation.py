from __future__ import division
import cv2
import numpy as np
import pdb
import math
import rospy

#################### X-Y CONVENTIONS #########################
# 0,0  X  > > > > >
#
#  Y
#
#  v  This is the image. Y increases downwards, X increases rightwards
#  v  Please return bounding boxes as ((xmin, ymin), (xmax, ymax))
#  v
#  v
#  v
###############################################################

def image_print(img):
	"""
	Helper function to print out images, for debugging. Pass them in as a list.
	Press any key to continue.
	"""
	cv2.imshow("image", img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

def get_slope(pt1,pt2):
    try:
        return (pt1[1]-pt2[1])/(pt1[0]-pt2[0])
    except:
        print("undefined")
        return None
    
def get_length(pt1, pt2):
    return np.sqrt((pt1[1]-pt2[1])**2+(pt1[0]-pt2[0])**2)

def get_intersect(a1, a2, b1, b2):
    """ 
    Returns the point of intersection of the lines passing through a2,a1 and b2,b1.
    a1: [x, y] a point on the first line
    a2: [x, y] another point on the first line
    b1: [x, y] a point on the second line
    b2: [x, y] another point on the second line
    """
    s = np.vstack([a1,a2,b1,b2])        # s for stacked
    h = np.hstack((s, np.ones((4, 1)))) # h for homogeneous
    l1 = np.cross(h[0], h[1])           # get first line
    l2 = np.cross(h[2], h[3])           # get second line
    x, y, z = np.cross(l1, l2)          # point of intersection
    if z == 0:                          # lines are parallel
        return (float('inf'), float('inf'))
    return (int(x/z), int(y/z))

def angle_bisector_equation(p1, p2, q1, q2):
    """
    Find the equation of the angle bisector between two lines given two points on each line.

    Parameters:
    p1 (tuple): (x, y) coordinates of the first point on the first line
    p2 (tuple): (x, y) coordinates of the second point on the first line
    q1 (tuple): (x, y) coordinates of the first point on the second line
    q2 (tuple): (x, y) coordinates of the second point on the second line

    Returns:
    tuple: a tuple containing the slope and y-intercept of the angle bisector
    """
    m1 = get_slope(p1,p2)
    b1 = p1[1]*1.0 - (m1 * p1[0]*1.0)
    m2 = get_slope(q1, q2)
    b2 = q1[1]*1.0 - (m2 * q1[0]*1.0)
    
    A1 = -1.0*m1
    B1 = 1.0
    C1 = -1.0*b1 
    
    A2 = -1.0*m2
    B2 = 1.0
    C2 = -1.0*b2

    sqrt1 = math.sqrt(A1**2+B1**2)
    sqrt2= math.sqrt(A2**2+B2**2)
    denom = (B2*sqrt1-B1*sqrt2)
    
    # new_slope = (A1*math.sqrt(A2**2+B2**2) - A2*math.sqrt(A1**2+B1**2))/(B2*math.sqrt(A1**2+B1**2)-B1*math.sqrt(A2**2+B2**2))
    # new_yintercept = (C1*math.sqrt(A2**2+B2**2) - C2*math.sqrt(A1**2+B1**2))/(B2*math.sqrt(A1**2+B1**2)-B1*math.sqrt(A2**2+B2**2))
    new_slope = (A1*sqrt2 - A2*sqrt1)/denom
    new_yintercept = (C1*sqrt2 - C2*sqrt1)/denom
    return new_slope, new_yintercept

def best_lines_bisector_line(fd_linesp, shape):
    #rospy.logerr("fd{}".format(fd_linesp))
    x_max,y_max = shape[1],shape[0]
    ####testing
    all_lines = np.array(fd_linesp)
    #print(all_lines)
    start_pts = all_lines[:, :2]
    end_pts = all_lines[:, 2:]

    # Compute angles for all lines
    angles = np.arctan2(end_pts[:, 1] - start_pts[:, 1], end_pts[:, 0] - start_pts[:, 0])
    sort_ind = np.argsort(angles)
    sorted_lines = all_lines[sort_ind]
    
    # Find lines with minimum and maximum angles
    pos_idx = np.where(angles[sort_ind]> 0.3)#np.argmin(angles) #right
    neg_idx = np.where(angles[sort_ind]< -0.3)#np.argmax(angles) #left

    print(pos_idx)
    print(neg_idx)
    # Get start and end points of the two lines
    A,B = (-5000,y_max),(x_max/2,0) #left side
    C,D = (5000,y_max),(x_max/2,0) #right side
    if np.size(pos_idx, axis=None): #if no positive slopes, use right image edge
        med_pos = pos_idx[0][-1]
        #med_pos = int(round(np.median(pos_idx)))
#         print("median pos index:", med_pos)
        C,D = (sorted_lines[med_pos][0],sorted_lines[med_pos][1]),(sorted_lines[med_pos][2],sorted_lines[med_pos][3])

    if np.size(neg_idx, axis=None):
        neg_pos = neg_idx[0][0]
        #neg_pos = int(round(np.median(neg_idx)))
        A,B = (sorted_lines[neg_pos][0],sorted_lines[neg_pos][1]),(sorted_lines[neg_pos][2],sorted_lines[neg_pos][3])

    m_1 = get_slope(A, B)
    b_1 = A[1]*1.0 - m_1*A[0]*1.0
    X_1 = (376.0 - b_1*1.0)/(1.0*m_1)
    #rospy.logerr("mbx1{} {} {}".format(m_1, b_1, X_1))
    
    m_2 = get_slope(C, D)
    b_2 = C[1]*1.0 - m_2*C[0]*1.0
    X_2 = (376 - b_2)/(1.0*m_2)
    
    #rospy.logerr("mbx2 {} {} {}".format(m_2, b_2, X_2))
    return X_1, X_2

    # Compute intersection point of the two lines
    intersection_pt = get_intersect(A, B, C, D)
    slope, y_intercept = angle_bisector_equation(A, B, C, D)
    avg_y = y_max #(B[0] + D[0]) / 2
    a_x = (avg_y - y_intercept) / slope
    end_intersection_pt = (int(a_x),int(avg_y))#(int(avg_y), int(a_x))


    # quad_slope, quad_y_intercept = angle_bisector_equation(A,B,intersection_pt, end_intersection_pt)
#     a_quad_x = (avg_y - quad_y_intercept) / quad_slope
#     end_quad_intersection_pt = (int(a_quad_x),int(avg_y))

    # print(intersection_pt, end_intersection_pt)
    return intersection_pt, end_intersection_pt

def get_strong_hough_lines(lines):
    lines_arr = np.array(lines)

    # Initialize an empty array to store strong lines
    strong_lines = np.empty((0, 2))

    # Iterate over each line
    for line in lines_arr:
        rho, theta = line[0]

        # Check if the line satisfies the conditions
        if np.all(np.abs(strong_lines[:, 0] - rho) > 50) and np.all(np.abs(strong_lines[:, 1] - theta) > 0.1 * np.abs(theta)):
            strong_lines = np.vstack((strong_lines, line[0]))

    return strong_lines

def polar2cartesian(rho, theta_rad, rotate90 = False):
    """
    Converts line equation from polar to cartesian coordinates

    Args:
        rho: input line rho
        theta_rad: input line theta
        rotate90: output line perpendicular to the input line

    Returns:
        m: slope of the line
           For horizontal line: m = 0
           For vertical line: m = np.nan
        b: intercept when x=0
    """
    x = np.cos(theta_rad) * rho
    y = np.sin(theta_rad) * rho
    m = np.nan
    if not np.isclose(x, 0.0):
        m = y / x
    if rotate90:
        if m is np.nan:
            m = 0.0
        elif np.isclose(m, 0.0):
            m = np.nan
        else:
            m = -1.0 / m
    b = 0.0
    if m is not np.nan:
        b = y - m * x

    return m, b
        


def cd_color_segmentation(img, template, visualize =False):
    """
    Implement the cone detection using color segmentation algorithm
    Input:
        img: np.3darray; the input image with a cone to be detected. BGR.
        template_file_path; Not required, but can optionally be used to automate setting hue filter values.
    Return:
        point: that is some look ahead distance on the angle bisector line between the two track lines
    """
    ########## YOUR CODE STARTS HERE ##########
    ### Cut off anything not the floor
    start_point1 = (0, 0) #top rectangle 
    end_point1 = (676, 150) #top rectangle: just edit y_coordinate for line_follower

    thickness = -1 
    color = (0, 0, 0) # Black color in BGR
    img_line= cv2.rectangle(img, start_point1, end_point1, color, thickness)


    ### change image to HSV and detect white 
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower_white = np.array([0,0,200])
    upper_white = np.array([179,255,255])

    mask = cv2.inRange(img_hsv, lower_white, upper_white)  
    masked_img = cv2.bitwise_and(img_hsv, img_hsv, mask=mask).astype('uint8')

    ### Remove any small pieces 
    hsv_image = masked_img#cv2.cvtColor(img_masked2, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv_image)
    ret, th1 = cv2.threshold(h,180,255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    kernel = np.ones((1,1), dtype = "uint8")/9
    bilateral = cv2.bilateralFilter(th1, 9 , 75, 75)
    erosion = cv2.erode(bilateral, kernel, iterations = 1)

        ##finding the area of all connected white pixels in the image
    pixel_components, output, stats, centroids =cv2.connectedComponentsWithStats(erosion, connectivity=8)
    area = stats[1:, -1]
    pixel_components = pixel_components - 1
    min_size = 50
    img2 = np.zeros((output.shape))

        ##Removing the small white pixel area below the minimum size
    for i in range(0, pixel_components):
        if area[i] >= min_size:
            img2[output == i + 1] = 255

    # try:
    ### get hough transform and filter by slope and length 
    src = np.uint8(img2)
    src = cv2.GaussianBlur(src, (5, 5), 0)
    dst = cv2.Canny(src, 10, 600, None, 3)

    minLineLength = 70
    maxLineGap = 10
    linesP = cv2.HoughLinesP(dst, 1, np.pi/180, threshold=50, minLineLength=minLineLength, maxLineGap=maxLineGap)

    #lines = cv2.HoughLines(dst, 1, np.pi/180, threshold=50)
    #filteredLines = get_strong_hough_lines(lines)

  
    filtered_linesp = []

    #for line in lines:
    #    m, b = polar2cartesian(line[0][0], line[0][1], True)
    #    filtered_linesp.append([0, b, -b/m, 0])

    if linesP is not None:
        for line in linesP:
            x1, y1, x2, y2 = line[0]
            if abs(y2 - y1) > 0.2 * abs(x2 - x1) and get_length((x1, y1), (x2, y2)) >= minLineLength:
               filtered_linesp.append([x1, y1, x2, y2])

    print(filtered_linesp)


    if len(filtered_linesp) == 0:
       for line in linesP:
            x1, y1, x2, y2 = line[0]
            if get_length((x1, y1), (x2, y2)) >= minLineLength:
                filtered_linesp.append([x1, y1, x2, y2])

    #If no line is detected, add line done the center
    if len(filtered_linesp) == 0:
         filtered_linesp.append([335, 0, 336, 376])

    X_1, X_2 = best_lines_bisector_line(filtered_linesp, img.shape)
    #rospy.logerr("X1{}".format(X_1))
    #rospy.logerr("X2{}".format(X_2))
    left = transformUvToXy(X_1, 376)
    right = transformUvToXy(X_2, 376)

    print(left, right)


    #print(g)
    if visualize:
        cv2.circle(img, (int(X_1), 376), radius=10, color=(225, 0, 255))
        cv2.circle(img, (int(X_2), 376), radius=10, color=(225, 0, 255))
        #cv2.line(img, intersection_pt, end_intersection_pt, (0,200,255), 3, cv2.LINE_AA)
        image_print(img)

    return left, right
    # except:
    # 	return None

def transformUvToXy(u, v):
        """
        u and v are pixel coordinates.
        The top left pixel is the origin, u axis increases to right, and v axis
        increases down.

        Returns a normal non-np 1x2 matrix of xy displacement vector from the
        camera to the point on the ground plane.
        Camera points along positive x axis and y axis increases to the left of
        the camera.

        Units are in meters.
        """

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
        
        METERS_PER_INCH = 0.0254

        np_pts_ground = np.array(PTS_GROUND_PLANE)
        np_pts_ground = np_pts_ground * METERS_PER_INCH
        np_pts_ground = np.float32(np_pts_ground[:, np.newaxis, :])

        np_pts_image = np.array(PTS_IMAGE_PLANE)
        np_pts_image = np_pts_image * 1.0
        np_pts_image = np.float32(np_pts_image[:, np.newaxis, :])

        h, err = cv2.findHomography(np_pts_image, np_pts_ground)

        homogeneous_point = np.array([[u], [v], [1]])
        xy = np.dot(h, homogeneous_point)
        scaling_factor = 1.0 / xy[2, 0]
        homogeneous_xy = xy * scaling_factor
        x = homogeneous_xy[0, 0]
        y = homogeneous_xy[1, 0]
        return x, y


#if __name__ == '__main__':
#    _img = cv2.imread("C:\\Users\\vanwi\OneDrive\\Documents\\MIT\\Senior\\6.4200\\racecar_docker\\home\\racecar_ws\\src\\final_challenge2023\\track_img\\0.34.png")
    #_img = cv2.imread("../../images/0.34.png")
#    cd_color_segmentation(_img, "", True)
