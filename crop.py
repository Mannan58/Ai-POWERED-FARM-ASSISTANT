import cv2
from ultralytics import YOLO

camera = cv2.VideoCapture(0)
model = YOLO("yolo26n.pt")

while True:
     success, frmae = camera.read()
     
     cv2.imshow("live",frmae)
     
     key = cv2.waitKey(1) & 0xFF
     
     if key == ord("c"):
         result = model(frmae)
         for i in range(len(result[0].boxes)):
             class_id = int(result[0].boxes.cls[i])
             class_name = result[0].names[class_id]
             
             if class_name == "banana":
                 box = result[0].boxes.xyxy[i]
                 x1,y1,x2,y2 = map(int,box)
                 crop_image = frmae[y1:y2,x1:x2]
                 cv2.imshow("croped_image",crop_image)
     if key == ord("q"):
         break
camera.release()
cv2.destroyAllWindows()