import cv2

for i in range(5):
    cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
    if cap.isOpened():
        print(f"Camera found at index {i}")
        ret, frame = cap.read()
        print("Frame:", ret)
        if ret:
            cv2.imshow("Camera", frame)
            cv2.waitKey(0)
        cap.release()
    else:
        print(f"Index {i} not working")

cv2.destroyAllWindows()