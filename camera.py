import cv2
camera = cv2.VideoCapture(0)
cv2.namedWindow("live",cv2.WINDOW_NORMAL)
cv2.resizeWindow("live",800,500)

while True:
    success, frame = camera.read()
    if not success:
        print("no frmae captured")
    cv2.imshow("live",frame)
    key = cv2.waitKey(1) & 0xFF
    if(key == ord("0")):
        break
    

cv2.imwrite("image.jpg",frame)
print(frame.shape)

camera.release()
cv2.destroyAllWindows()