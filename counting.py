import cv2
from ultralytics import YOLO

camera = cv2.VideoCapture(1)

model = YOLO("sam_model.pt")


objects = {"leaf":0,"flower":0,"fruit":0}
while True:
    success, frame = camera.read()
    if not success:
        print("camera not detected")
        
    cv2.imshow("Live",frame)
    key = cv2.waitKey(1)&0xFF
    if key == ord("s"):
        results = model(frame)
        cv2.imshow("photo captured",frame)
        
        for result in results:
            result.show()
        
        for i in range(len(results[0].boxes)):
            class_id = int(results[0].boxes.cls[i])
            class_name = results[0].names[class_id]
            objects[class_name] += 1
    if key == ord("q"):
        break
print(objects)
camera.release()
cv2.destroyAllWindows()    