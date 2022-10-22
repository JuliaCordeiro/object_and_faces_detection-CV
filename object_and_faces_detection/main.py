import cv2
import constants

image = cv2.imread('./object_and_faces_detection/images/people/camila_achutti.jpg')

cv2.rectangle(image, (20, 20), (120, 120), constants.white, 1)

cv2.imshow('Retangle', image)

cv2.waitKey(0)