import cv2

# Load the Haar cascade (use OpenCV's built-in path to avoid missing file issues)
haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Open the webcam
cam = cv2.VideoCapture(0)

while True:  # infinite loop
    ret, img = cam.read()
    if not ret:
        print("Failed to grab frame")
        break

    grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = haar_cascade.detectMultiScale(grayImg, 1.3, 4)

    # Draw rectangle around detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow("Face Detection", img)  # display the frame

    key = cv2.waitKey(10) 
    if key == ord('q'):  # press 'q' to quit
        break

cam.release()
cv2.destroyAllWindows()
