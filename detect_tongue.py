enhanced = cv2.detailEnhance(frame, sigma_s=10, sigma_r=0.15)
frame = imutils.resize(frame, width=500)


result = get_mouth_loc_with_height(enhanced)
message = 'Face detected!'
if(checkKey(result,"error")):
    message = result['message']
    cv2.imshow("output", frame)
    cv2.putText(frame, message, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
            0.7, (0, 0, 0), 1)
    continue
else:
    mouth_x=result['mouth_x']
    mouth_y=result['mouth_y']
    mouth_w=result['mouth_w']
    mouth_h=result['mouth_h']
    image_ret=result['image_ret']
    height_of_inner_mouth=result['height_of_inner_mouth']
    inner_mouth_y=result['inner_mouth_y']
    y_lowest_in_face=result['y_lowest_in_face']
    shape=result['shape']
    frame = draw_mouth(frame, shape)
    mouthMAR = mouth_aspect_ratio(shape)
    if(mouthMAR > MOUTH_AR_THRESH):
        cv2.putText(frame, "Mouth is open! " + str(mouthMAR), (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
            0.7, (0, 0, 0), 1)
    else:
        cv2.putText(frame, "Mouth is closed! " + str(mouthMAR), (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
            0.7, (0, 0, 0), 1)







# detect blob
# Initiate ORB detector
orb = cv2.ORB_create()
# find the keypoints with ORB
kp = orb.detect(roi, None)

# compute the descriptors with ORB
kp, des = orb.compute(roi, kp)

x_fin, y_fin = 1, 1
if (kp):
    x_fin, y_fin = kp[0].pt
    for point in kp:
        x, y = point.pt
        if (y_fin < y):
            y_fin = y
            x_fin = x

print(x_fin, y_fin)

# convert
x_fin = math.floor((x_fin * x1) / x2)
y_fin = math.floor((y_fin * y1) / y2)
cv2.putText(frame, "tongue tip location: " + str(x_fin) + "," + str(y_fin), (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
            0.7, (0, 0, 0), 1)

# If y_fin<=1 , then no blob keypoint is found
if (mouthMAR > MOUTH_AR_THRESH and y_fin > 1):
    color = (0, 0, 255)
    image = cv2.circle(frame[inner_mouth_y:mouth_y + mouth_h, mouth_x:mouth_x + mouth_w], (x_fin, y_fin), 4, color,
                       thickness=-1)
    cv2.imshow("output", frame)
else:
    cv2.imshow("output", frame)