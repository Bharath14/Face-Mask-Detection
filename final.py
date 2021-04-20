import cv2 as cv
import numpy as np
import time
from tensorflow.keras.models import load_model
#import matplotlib.pyplot as plt

def rescaleFrame(frame, scale = 0.5):
	#works for images,videos and live video
	width = int(frame.shape[1]*scale)
	height = int(frame.shape[0]*scale)

	dimensions = (width,height)

	return cv.resize(frame,dimensions,interpolation=cv.INTER_AREA)

#image = cv.imread('person.jpg')
#image = rescaleFrame(image)
#cv.imshow("image_before",image)
def crop(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
	label = str(classes[class_id])
	color = COLORS[class_id]
	blank = np.zeros((img.shape[0],img.shape[1]), dtype='uint8')
	rectangle1 = cv.rectangle(blank.copy(),(x,y), (x_plus_w,y_plus_h), 255, -1)
	rectangle1 = cv.resize(rectangle1, img.shape[1::-1])
	blank = np.zeros(img.shape[:2], dtype='uint8')
	b,g,r = cv.split(img)
	blue = cv.merge([b,blank,blank])
	green = cv.merge([blank,g,blank])
	red = cv.merge([blank,blank,r])
	b=cv.bitwise_and(rectangle1,b)
	g=cv.bitwise_and(rectangle1,g)
	r=cv.bitwise_and(rectangle1,r)
	merged = cv.merge([b,g,r])
	return merged



#video
vid = cv.VideoCapture(0)
cascPathface = "./haarcascade_frontalface_default.xml"
faceCascade = cv.CascadeClassifier(cascPathface)
model = load_model('alex_kaggle.h5')
labels_dict={0:'MASK',1:'NO MASK'}
color_dict={0:(0,255,0),1:(0,0,255)}
while True:
	ret, image = vid.read()
	ng=np.zeros(image.shape[:2],dtype='uint8')
	#cv.imshow("video",image)

	Width = image.shape[1]
	Height = image.shape[0]

	classes = None
	with open('object-detection-opencv/yolov3.txt', 'r') as f:
		classes = [line.strip() for line in f.readlines()]

	net = cv.dnn.readNet('yolov3.weights', 'object-detection-opencv/yolov3.cfg')
	net.setInput(cv.dnn.blobFromImage(image, 0.00392, (416,416), (0,0,0), True, crop=False))

	layer_names = net.getLayerNames()
	output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
	outs = net.forward(output_layers)

	class_ids = []
	confidences = []
	boxes = []
	Width = image.shape[1]
	Height = image.shape[0]
	for out in outs:
		for detection in out:
			scores = detection[5:]
			class_id = np.argmax(scores)
			confidence = scores[class_id]
			if confidence > 0.5:
				center_x = int(detection[0] * Width)
				center_y = int(detection[1] * Height)
				w = int(detection[2] * Width)
				h = int(detection[3] * Height)
				x = center_x - w / 2
				y = center_y - h / 2
				class_ids.append(class_id)
				confidences.append(float(confidence))
				boxes.append([x, y, w, h])
			COLORS = np.random.uniform(0, 255, size=(len(classes), 3))
			indices = cv.dnn.NMSBoxes(boxes, confidences, 0.1, 0.1)
	for i in indices:
		i = i[0]
		box = boxes[i]
		if class_ids[i]==0:
			label = str(classes[class_id]) 
			color = COLORS[class_id]
			cv.rectangle(image, (round(box[0]),round(box[1])), (round(box[0]+box[2]),round(box[1]+box[3])), (0, 1, 0), 2)
			cv.putText(image, label, (round(box[0])-10,round(box[1])-10), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
			human = crop(image, class_ids[i], confidences[i], round(box[0]), round(box[1]), round(box[0]+box[2]), round(box[1]+box[3]))
	gray = cv.cvtColor(human, cv.COLOR_BGR2GRAY)
	faces = faceCascade.detectMultiScale(gray,
		 scaleFactor=1.1,
		 minNeighbors=5,
		 minSize=(50, 50),
		 flags=cv.CASCADE_SCALE_IMAGE)
	for (x,y,w,h) in faces:
		face_img = gray[y:y+w,x:x+w]
		resized = cv.resize(face_img , (128,128))
		normalized = resized/255.0
		reshaped = np.reshape(normalized,(1,128,128,1))
		result = model.predict(reshaped)
		#print(result)
		label = np.argmax(result, axis =1)[0]
		#print(result[0][label])
		cv.rectangle(image,(x,y), (x+w,y+h),color_dict[label],2)
		cv.rectangle(image,(x,y-40), (x+w,y),color_dict[label],-1)
		cv.putText(image, labels_dict[label]+"  "+str(result[0][label]), (x, y-10),cv.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
		#face = crop(image,class_ids[i],confidences[i],x,y,x+w,y+h)
	cv.imshow('Live',image)
	#cv.imshow("crop_image",ng)

	if cv.waitKey(1) & 0xFF == ord('q'):
		break
cv.destroyAllWindows()
vid.release()
