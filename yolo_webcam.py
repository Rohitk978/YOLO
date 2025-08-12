from ultralytics import YOLO
import cv2
import cvzone
import math

# For web Cam.
# cap = cv2.VideoCapture(0) 
# # setting height and width.
# cap.set(3,1280)
# cap.set(4,720) 

# For Video.
cap = cv2.VideoCapture("./videos/cars.mp4")

# creating a yolo model.
model = YOLO("yolo_weights/yolov8l.pt")
# classname = "coco.names"
with open('coco.names', 'r') as f:
    classes = [line.strip() for line in f.readlines()]
while(True):
    success , img = cap.read()
    results = model(img,stream=True)
    for r in results:
        boxes = r.boxes
        for box in boxes:
        #    For creating bounding box.
           x1,y1,x2,y2 =  box.xyxy[0]
           x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)
           # cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,255),3)
           w,h = x2-x1,y2-y1
           cvzone.cornerRect(img,(x1,y1,w,h))
        #    For the Confidence.
           conf = math.ceil((box.conf[0]*100)/100)
           cvzone.putTextRect(img,f"{conf}",(max(0,x1),max(35,y1)))
        #    For the class name.
           cls = int(box.cls[0])
           cvzone.putTextRect(img,f"{classes[cls]} {conf}",(max(0,x1),max(35,y1)))
    cv2.imshow("Image",img)
    cv2.waitKey(1)