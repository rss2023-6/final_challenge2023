import cv2
import imutils
import numpy as np
import pdb

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
	Helper function to print out images, for debugging.
	Press any key to continue.
	"""
	winname = "Image"
	cv2.namedWindow(winname)        # Create a named window
	cv2.moveWindow(winname, 40,30)  # Move it to (40,30)
	cv2.imshow(winname, img)
	cv2.waitKey()
	cv2.destroyAllWindows()

def cd_sift_ransac(img, template):
	"""
	Implement the cone detection using SIFT + RANSAC algorithm
	Input:
		img: np.3darray; the input image with a cone to be detected
	Return:
		bbox: ((x1, y1), (x2, y2)); the bounding box of the cone, unit in px
				(x1, y1) is the bottom left of the bbox and (x2, y2) is the top right of the bbox
	"""
	# Minimum number of matching features
	MIN_MATCH = 10
	# Create SIFT
	sift = cv2.xfeatures2d.SIFT_create()

	# Compute SIFT on template and test image
	kp1, des1 = sift.detectAndCompute(template,None)
	# plt.figure()
	# plt.imshow(cv2.drawKeypoints(cv2.cvtColor(template, cv2.COLOR_BGR2GRAY), kp1,template))
	kp2, des2 = sift.detectAndCompute(img,None)
	# plt.figure()
	# plt.imshow(cv2.drawKeypoints(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), kp2,img))

	# Find matches
	bf = cv2.BFMatcher()
	matches = bf.knnMatch(des1,des2,k=2)
 
	plt.figure()
	plt.imshow(cv2.drawMatches(template,kp1,img,kp2,matches[:10], None,flags=2))

	# Find and store good matches
	good = []
	for m,n in matches:
		if m.distance < 0.75*n.distance:
			good.append(m)
	
	# plt.figure()
	# plt.imshow(cv2.drawKeypoints(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), good, img))

	# If enough good matches, find bounding box
	if len(good) > MIN_MATCH:
		src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
		dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

		# Create mask
		M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
		matchesMask = mask.ravel().tolist()
		h, w, c = template.shape
		pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)

		########## YOUR CODE STARTS HERE ##########
		x_min = y_min = float('inf')
		x_max = y_max = 0
		n = len(src_pts)
		for i in range(n):
				if(True):
						point = np.array([np.append(src_pts[i], [1])])
						out = np.dot(M, np.transpose(point))
						# print(out)
						out = out / out[2][0]
						x = out[0][0]
						y = out[1][0]
						if(x < x_min):
								x_min = x
						if(x > x_max):
								x_max = x
						if(y < y_min):
								y_min = y
						if(y > y_max):
								y_max = y

		########### YOUR CODE ENDS HERE ###########
		# Return bounding box
		return ((x_min, y_min), (x_max, y_max))
	else:

		print("[SIFT] not enough matches; matches: ", len(good))
		# Return bounding box of area 0 if no match found
		return ((0,0), (0,0))

def cd_template_matching(img, template):
	"""
	Implement the cone detection using template matching algorithm
	Input:
		img: np.3darray; the input image with a cone to be detected
	Return:
		bbox: ((x1, y1), (x2, y2)); the bounding box of the cone, unit in px
				(x1, y1) is the bottom left of the bbox and (x2, y2) is the top right of the bbox
	"""
	template_canny = cv2.Canny(template, 50, 200)
	# plt.figure()
	# plt.imshow(template_canny, cmap='gray')
 
	# Perform Canny Edge detection on test image
	grey_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	img_canny = cv2.Canny(grey_img, 50, 200)
	
	plt.figure()
	plt.imshow(img_canny, cmap='gray')
	plt.figure()

	# Get dimensions of template
	(img_height, img_width) = img_canny.shape[:2]

	# Keep track of best-fit match
	best_match = None

	# Loop over different scales of image
	maximum = 0
	for scale in np.linspace(1.5, .5, 50):
		# Resize the image
		resized_template = imutils.resize(template_canny, width = int(template_canny.shape[1] * scale))

		(h,w) = resized_template.shape[:2]
		# Check to see if test image is now smaller than template image
		if resized_template.shape[0] > img_height or resized_template.shape[1] > img_width:
			continue

		########## YOUR CODE STARTS HERE ##########
		# Use OpenCV template matching functions to find the best match
		# across template scales.
		# MIN = float('inf')
		# bounding_box = ((0,0),(0,0))
		methods = ['TM_CCOEFF_NORMED'] # 'TM_SQDIFF_NORMED', 'TM_CCOEFF_NORMED', 'TM_CCORR', 'TM_CCORR', 'TM_CCOEFF', 'TM_CCORR','TM_CCORR_NORMED', 'TM_SQDIFF', 'TM_SQDIFF_NORMED'
		for m in methods:
				res = cv2.matchTemplate(img_canny.copy(), resized_template.copy(), getattr(cv2, m)) #,getattr(cv2, m)
				min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

				if m in [getattr(cv2, 'TM_SQDIFF'), getattr(cv2, 'TM_SQDIFF_NORMED')]:
						top_left = min_loc
						if(min_val < MIN):
								bounding_box = ((top_left[0], top_left[1]), (top_left[0] + w, top_left[1] + h))
								MIN = min_val
				else:
						top_left = max_loc
						if(max_val > maximum):
								bounding_box = ((top_left[0], top_left[1]), (top_left[0] + w, top_left[1] + h))
								# print(bounding_box)
								# print(max_val)
								# print(maximum)
								# print()
								maximum = max_val
		# Remember to resize the bounding box using the highest scoring scale
		# x1,y1 pixel will be accurate, but x2,y2 needs to be correctly scaled
		########### YOUR CODE ENDS HERE ###########
	return bounding_box
