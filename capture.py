import cv2
from ultralytics import YOLO

camera= cv2.VideoCapture(0)
model = YOLO("yolo26n.pt")

block = 1

while True:
    success , frame = camera.read()
    cv2.imshow("live",frame)
    
    key = cv2.waitKey(1) &0xFF
    objects={}
    if key == ord("s"):
        result = model(frame)
        for i in range(len(result[0].boxes)):
            class_id = int(result[0].boxes.cls[i])
            class_name = result[0].names[class_id]
            
            # if class_name == "banana":
            if class_name in objects:
                    objects[class_name] +=1
            else:
                    objects[class_name] = 1
            
        text = str(objects) 
            
        yolo_image= result[0].plot()
        cv2.putText(yolo_image,text,(20,40),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
        file_name = f"images/block_{block}.jpg"
        cv2.imwrite(file_name,yolo_image)
                
        cv2.imshow("photo_captured",frame)
        block+=1
        
    if key == ord("q"):
        break
    
camera.release()
cv2.destroyAllWindows()