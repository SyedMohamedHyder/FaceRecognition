import cv2
import find_face

webcam = cv2.VideoCapture(0)

ret, frame = webcam.read()
while ret and not(cv2.waitKey(5) & 0xFF == ord('d')):
    find_face.recognize(frame)
    cv2.imshow('Recognize People', frame)
    ret, frame = webcam.read()

webcam.release()
cv2.destroyAllWindows()