from ultralytics import YOLO

model = YOLO("yolo26n.pt") # pretrained model from yolo to detect 80 commom daily life things

results = model("photo.jpg")
apple_count = 0
for result in results:
    for i in range(len(result.boxes)):
     class_id = int(result.boxes.cls[i]) # get the id of the class detected in integer form
     name = result.names[class_id] #get that id only of the dictionary
     confi = float(result.boxes.conf[i])
     if name == "apple":
       apple_count += 1
     print(name)
     print(class_id)
    print("="*20)
    result.show()    
print(apple_count)   