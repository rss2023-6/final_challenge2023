import cv2
import numpy as np
import pdb
import math

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
    b1 = p1[1] - (m1 * p1[0])
    m2 = get_slope(q1, q2)
    b2 = q1[1] - (m2 * q1[0])
    
    A1 = -m1
    B1 = 1
    C1 = -b1 
    
    A2 = -m2
    B2 = 1
    C2 = -b2
    
    new_slope = (A1*np.sqrt(A2**2+B2**2) - A2*np.sqrt(A1**2+B1**2))/(B2*np.sqrt(A1**2+B1**2)-B1*np.sqrt(A2**2+B2**2))
    new_yintercept = (C1*np.sqrt(A2**2+B2**2) - C2*np.sqrt(A1**2+B1**2))/(B2*np.sqrt(A1**2+B1**2)-B1*np.sqrt(A2**2+B2**2))
    return new_slope, new_yintercept

def best_lines_bisector_line(fd_linesp):
    all_angles = dict()
    for i in range(len(fd_linesp)):
        l = fd_linesp[i]
        start_pt = (l[0], l[1])
        end_pt = (l[2], l[3])
        angle = np.arctan(get_slope(start_pt, end_pt))
        all_angles[(start_pt, end_pt)] = angle
#         print(start_pt, end_pt, angle)
    max_key = max(all_angles, key=all_angles.get)
    min_key = min(all_angles, key=all_angles.get)
    
    #Line AB, Line CD
    A = min_key[0]#(29, 342)
    B = min_key[1]#(298, 160)
    C = max_key[0]#(403, 157) 
    D = max_key[1]#(654, 243)
    intersection_pt = get_intersect(A, B, C, D)
    
#     print(intersection_pt)
    
    #get end pt 
    slope, y_intercept = angle_bisector_equation(A, B, C,D)
    avg_y = (B[0]+D[0])/2
    a_x = (avg_y-y_intercept)/slope
    end_intersection_pt = (int(avg_y),int(a_x))
    
#     print(end_intersection_pt)
    
    return intersection_pt, end_intersection_pt

def cd_color_segmentation(img, template):
	"""
	Implement the cone detection using color segmentation algorithm
	Input:
		img: np.3darray; the input image with a cone to be detected. BGR.
		template_file_path; Not required, but can optionally be used to automate setting hue filter values.
	Return:
		bbox: ((x1, y1), (x2, y2)); the bounding box of the cone, unit in px
				(x1, y1) is the top left of the bbox and (x2, y2) is the bottom right of the bbox
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
	area = stats[1:, -1]; pixel_components = pixel_components - 1
	min_size = 50
	img2 = np.zeros((output.shape))

		##Removing the small white pixel area below the minimum size
	for i in range(0, pixel_components):
		if area[i] >= min_size:
			img2[output == i + 1] = 255


	try:
		### get hough transform and filter by slope and length 
		src = np.uint8(img2)
		dst = cv2.Canny(src, 50, 200, None, 3)
		linesP = cv2.HoughLinesP(dst, 1, np.pi / 180, 50, None, 50, 10) #probabilistic
		
		filtered_linesp = []
		if linesP is not None:
			for i in range(0, len(linesP)):
				l = linesP[i][0]
				start_pt = (l[0], l[1])
				end_pt = (l[2], l[3])
				if abs(get_slope(start_pt, end_pt)) >0.20:
					if get_length(start_pt, end_pt) >= 70:
						filtered_linesp.append([l[0], l[1],l[2], l[3]])
		
	#         print(filtered_linesp)
		intersection_pt, end_intersection_pt = best_lines_bisector_line(filtered_linesp)
		avg_pt = int((intersection_pt[0]+end_intersection_pt[0])/2), int((intersection_pt[1]+end_intersection_pt[1])/2)
	#         print(avg_pt)
		return avg_pt
	except:
		return None

