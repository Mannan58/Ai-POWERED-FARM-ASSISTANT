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
    
    for result in results:
        result.show()
    anottated_frame = results[0].plot()
    cv2.imshow("live",anottated_frame)
    key = cv2.waitKey(1) & 0xFF
    if key == (ord("q")):
        break

    
    