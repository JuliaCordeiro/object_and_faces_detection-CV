import cv2
import constants

image = cv2.imread('./images/people/camila_achutti.jpg')

cv2.rectangle(image, (20, 20), (120, 120), constants.white, 1)

cv2.imshow('Rectangle', image)

cv2.waitKey(0)