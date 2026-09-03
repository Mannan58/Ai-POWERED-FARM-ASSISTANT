import cv2
from ultralytics import YOLO

camera = cv2.VideoCapture(0)
model = YOLO("yolo26n.pt")


while True:
    
    success, frame = camera.read()
    if not success:
        print("no frame")
        break
    
    results = model(frame)
    objects = {}
    for i in range(len(results[0].boxes)): #for every box in one frme
        class_id = int(results[0].boxes.cls[i])
        class_name = results[0].names[class_id]
        
        if class_name in objects:
            objects[class_name] += 1
        else:
            objects[class_name]=1   #if not object append the key and then put the value 1
    print(objects)
    
    text = str(objects)
    
    anottated_frame = results[0].plot()
    
    cv2.putText(anottated_frame,text,(20,40),cv2.FONT_HERSHEY_PLAIN,1.5,(0, 0, 255),2)
    
    cv2.imshow("live",anottated_frame)
    key = cv2.waitKey(1) & 0xFF
    if key == (ord("q")):
        break

camera.release()
cv2.destroyAllWindows()   
    