import os
import cv2

name = input('Enter name of the person : ')
target_dir = os.path.join('dataset_images', name)
os.mkdir(target_dir)

image_num = 0

webcam = cv2.VideoCapture(0)

ret, frame = webcam.read()
while ret and not(cv2.waitKey(15) & 0xFF == ord('d')):
    cv2.imshow('Video Recording', frame)
    if cv2.waitKey(15) & 0xFF == ord(' '):
        cv2.imshow('Captured Image', frame)
        cv2.waitKey(500)
        cv2.destroyWindow('Captured Image')
        filename = os.path.join(target_dir, f'{name}{image_num}.jpg')
        cv2.imwrite(filename, frame)
        image_num += 1
    ret, frame = webcam.read()

webcam.release()
cv2.destroyAllWindows()
