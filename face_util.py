import imutils
import cv2
import dlib
import scipy
import math
import numpy

def get_mouth_loc_with_height(image):
	x1,y1,w1,h1,h_in,y = 1,1,1,1,1,1
	image = imutils.resize(image, width=500)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	# detect faces in the grayscale image
	rects = detector(gray, 1)
	if(len(rects) > 0):
		# loop over the face detections
		for (i, rect) in enumerate(rects):
			# determine the facial landmarks for the face region, then
			# convert the landmark (x, y)-coordinates to a NumPy array
			shape = predictor(gray, rect)
			shape = shape_to_np(shape)
			x_lowest_in_face, y_lowest_in_face = shape[9]
			# loop over the face parts individually
			for (name, (i, j)) in face_utils.FACIAL_LANDMARKS_IDXS.items():
				if(name == "mouth"):
					# extract the ROI of the face region as a separate image
					(x1, y1, w1, h1) = cv2.boundingRect(np.array([shape[i:j]]))
				if(name == "inner_mouth"):
					# extract the ROI of the face region as a separate image
					(x, y, w, h) = cv2.boundingRect(np.array([shape[i:j]]))
					h_in = h
		return {"mouth_x":x1,
		"mouth_y":y1,"mouth_w":w1,"mouth_h":h1, "image_ret":image, "height_of_inner_mouth":h_in,
		"inner_mouth_y":y, "y_lowest_in_face":y_lowest_in_face, "shape": shape}
	else:
		return {"error":"true", "message":"No Face Found!"}


def draw_mouth(image, shape):
	# draw mouth points
	(j, k) = FACIAL_LANDMARKS_IDXS["mouth"]
	pts_mouth = shape[j:k]
	for (x, y) in pts_mouth:
		cv2.circle(image, (x, y), 1, (0, 255, 0), -1)

	# return the output image
	return image


def mouth_aspect_ratio(shape):
	# grab the indexes of the facial landmarks for the mouth
	(mStart, mEnd) = (49, 68)
	mouth = shape[mStart:mEnd]
	# compute the euclidean distances between the two sets of
	# vertical mouth landmarks (x, y)-coordinates
	A = dist.euclidean(mouth[2], mouth[10]) # 51, 59
	B = dist.euclidean(mouth[4], mouth[8]) # 53, 57

	# compute the euclidean distance between the horizontal
	# mouth landmark (x, y)-coordinates
	C = dist.euclidean(mouth[0], mouth[6]) # 49, 55

	# compute the mouth aspect ratio
	mar = (A + B) / (2.0 * C)

	# return the mouth aspect ratio
	return mar