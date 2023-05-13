import cv2
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
	Helper function to print out images, for debugging. Pass them in as a list.
	Press any key to continue.
	"""
	cv2.imshow("image", img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

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

	# #line following box 
	# start_point1 = (0, 0) #top rectangle 
	# end_point1 = (672, 200) #top rectangle: just edit y_coordinate for line_follower

	# thickness = -1 
	# color = (0, 0, 0) # Black color in BGR
	# img_line= cv2.rectangle(img, start_point1, end_point1, color, thickness)

	# start_point2 = (0, 300) #bottom rectangle: just edit y_coordinate for line follower
	# end_point2 = (672, 376) #bottom rectangle 
	# img_line= cv2.rectangle(img_line, start_point2, end_point2, color, thickness)
	
	#convert to hsv
	img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) #for line following, convert img to img_line

	#filter oranges
	light_orange = (0, 210, 170) #H,S,V
	dark_orange = (50, 255, 255) #H,S,V
	mask = cv2.inRange(img_hsv, light_orange, dark_orange)
	masked_img = cv2.bitwise_and(img_hsv, img_hsv, mask=mask)

	#bounding box
	h, s, v = cv2.split(masked_img)
	ret, th1 = cv2.threshold(h,180,255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
	kernel = np.ones((1,1), dtype = "uint8")/9
	bilateral = cv2.bilateralFilter(th1, 9 , 75, 75)
	erosion = cv2.erode(bilateral, kernel, iterations = 1)


	#finding the area of all connected white pixels in the image
	pixel_components, output, stats, centroids =cv2.connectedComponentsWithStats(erosion, connectivity=8)
	area = stats[1:, -1]; pixel_components = pixel_components - 1
	min_size = 200 #1000

	img2 = np.zeros((output.shape))
	#Removing the small white pixel area below the minimum size
	for i in range(0, pixel_components):
		if area[i] >= min_size:
			img2[output == i + 1] = 255

	img3 = img2.astype(np.uint8) 

	# find contours and bounding rectangle in the thresholded image
	cnts, hierachy = cv2.findContours(img3.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2:]
	try: 
		rect = cv2.boundingRect(cnts[0])
		x,y,w,h = rect #(x,y) be the top-left coordinate of the rectangle and (w,h) be its width and height.

		#visualize
		# img_final = img.copy()
		# #img_final = cv2.drawContours(img_final,cnts[0],0,(0,255,255),2)
		# img_final = cv2.rectangle(img_final, (x,y),(x+w,y+h),(0,255,0),2)
		# image_print(img_final)

	except:
		x,y,w,h = 0,0,0,0

	bounding_box = ((x,y),(x+w,y+h))


	########### YOUR CODE ENDS HERE ###########

	# Return bounding box
	return bounding_box
