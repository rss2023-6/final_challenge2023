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
    
    new_slope = (A1*math.sqrt(A2**2+B2**2) - A2*math.sqrt(A1**2+B1**2))/(B2*math.sqrt(A1**2+B1**2)-B1*math.sqrt(A2**2+B2**2))
    new_yintercept = (C1*math.sqrt(A2**2+B2**2) - C2*math.sqrt(A1**2+B1**2))/(B2*math.sqrt(A1**2+B1**2)-B1*math.sqrt(A2**2+B2**2))
    return new_slope, new_yintercept

def best_lines_bisector_line(fd_linesp,img_shape):
    all_lines = np.array(fd_linesp)
    start_pts = all_lines[:, :2]
    end_pts = all_lines[:, 2:]
    
    # Compute angles for all lines
    angles = np.arctan2(end_pts[:, 1] - start_pts[:, 1], end_pts[:, 0] - start_pts[:, 0])
    
    # Find lines with minimum and maximum angles
    min_idx = np.argmin(angles)
    max_idx = np.argmax(angles)
    
    # Get start and end points of the two lines
    A, B = start_pts[min_idx], end_pts[min_idx]
    C, D = start_pts[max_idx], end_pts[max_idx]
    
    # Compute intersection point of the two lines
    intersection_pt = get_intersect(A, B, C, D)
    
    # Compute end intersection point of the two lines
    slope, y_intercept = angle_bisector_equation(A, B, C, D)
    avg_y = img_shape[0]#(B[0] + D[0]) / 2
    a_x = (avg_y - y_intercept) / slope
    end_intersection_pt = (int(avg_y), int(a_x))
    
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
	# try:
	src = np.uint8(img2)
	src = cv2.GaussianBlur(src, (5, 5), 0)
	dst = cv2.Canny(src, 50, 200, None, 3)

	minLineLength = 70
	maxLineGap = 10
	linesP = cv2.HoughLinesP(dst, 1, np.pi/180, threshold=50, minLineLength=minLineLength, maxLineGap=maxLineGap)
	filtered_linesp = []

	if linesP is not None:
		for line in linesP:
			x1, y1, x2, y2 = line[0]
			if abs(y2 - y1) > 0.2 * abs(x2 - x1) and get_length((x1, y1), (x2, y2)) >= minLineLength:
				filtered_linesp.append([x1, y1, x2, y2])

	intersection_pt, end_intersection_pt = best_lines_bisector_line(filtered_linesp, img.shape)
	avg_pt = int((intersection_pt[0]+end_intersection_pt[0])/2), int((intersection_pt[1]+end_intersection_pt[1])/2)

	return avg_pt
	# except:
	# 	return None


if __name__ == '__main__':
	_img = cv2.imread("/Users/kristinezheng/racecar_docker/home/racecar_ws/src/final_challenge2023/track_img/c5.png")

	cd_color_segmentation(_img, "", True)
		
# import cv2
# import numpy as np
# import pdb

# #################### X-Y CONVENTIONS #########################
# # 0,0  X  > > > > >
# #
# #  Y
# #
# #  v  This is the image. Y increases downwards, X increases rightwards
# #  v  Please return bounding boxes as ((xmin, ymin), (xmax, ymax))
# #  v
# #  v
# #  v
# ###############################################################

# def image_print(img):
# 	"""
# 	Helper function to print out images, for debugging. Pass them in as a list.
# 	Press any key to continue.
# 	"""
# 	cv2.imshow("image", img)
# 	cv2.waitKey(0)
# 	cv2.destroyAllWindows()

# def cd_color_segmentation(img, template):
# 	"""
# 	Implement the cone detection using color segmentation algorithm
# 	Input:
# 		img: np.3darray; the input image with a cone to be detected. BGR.
# 		template_file_path; Not required, but can optionally be used to automate setting hue filter values.
# 	Return:
# 		bbox: ((x1, y1), (x2, y2)); the bounding box of the cone, unit in px
# 				(x1, y1) is the top left of the bbox and (x2, y2) is the bottom right of the bbox
# 	"""
# 	########## YOUR CODE STARTS HERE ##########

# 	# #line following box 
# 	# start_point1 = (0, 0) #top rectangle 
# 	# end_point1 = (672, 200) #top rectangle: just edit y_coordinate for line_follower

# 	# thickness = -1 
# 	# color = (0, 0, 0) # Black color in BGR
# 	# img_line= cv2.rectangle(img, start_point1, end_point1, color, thickness)

# 	# start_point2 = (0, 300) #bottom rectangle: just edit y_coordinate for line follower
# 	# end_point2 = (672, 376) #bottom rectangle 
# 	# img_line= cv2.rectangle(img_line, start_point2, end_point2, color, thickness)
	
# 	#convert to hsv
# 	img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) #for line following, convert img to img_line

# 	#filter oranges
# 	light_orange = (0, 210, 170) #H,S,V
# 	dark_orange = (50, 255, 255) #H,S,V
# 	mask = cv2.inRange(img_hsv, light_orange, dark_orange)
# 	masked_img = cv2.bitwise_and(img_hsv, img_hsv, mask=mask)

# 	#bounding box
# 	h, s, v = cv2.split(masked_img)
# 	ret, th1 = cv2.threshold(h,180,255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
# 	kernel = np.ones((1,1), dtype = "uint8")/9
# 	bilateral = cv2.bilateralFilter(th1, 9 , 75, 75)
# 	erosion = cv2.erode(bilateral, kernel, iterations = 1)


# 	#finding the area of all connected white pixels in the image
# 	pixel_components, output, stats, centroids =cv2.connectedComponentsWithStats(erosion, connectivity=8)
# 	area = stats[1:, -1]; pixel_components = pixel_components - 1
# 	min_size = 200 #1000

# 	img2 = np.zeros((output.shape))
# 	#Removing the small white pixel area below the minimum size
# 	for i in range(0, pixel_components):
# 		if area[i] >= min_size:
# 			img2[output == i + 1] = 255

# 	img3 = img2.astype(np.uint8) 

# 	# find contours and bounding rectangle in the thresholded image
# 	cnts, hierachy = cv2.findContours(img3.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2:]
# 	try: 
# 		rect = cv2.boundingRect(cnts[0])
# 		x,y,w,h = rect #(x,y) be the top-left coordinate of the rectangle and (w,h) be its width and height.

# 		#visualize
# 		# img_final = img.copy()
# 		# #img_final = cv2.drawContours(img_final,cnts[0],0,(0,255,255),2)
# 		# img_final = cv2.rectangle(img_final, (x,y),(x+w,y+h),(0,255,0),2)
# 		# image_print(img_final)

# 	except:
# 		x,y,w,h = 0,0,0,0

# 	bounding_box = ((x,y),(x+w,y+h))


# 	########### YOUR CODE ENDS HERE ###########

# 	# Return bounding box
# 	return bounding_box
