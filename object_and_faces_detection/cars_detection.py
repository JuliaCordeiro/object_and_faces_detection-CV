import cv2
from base_functions import (
  bgr_to_gray,
  retangle_design
)

CAR_HAAR = cv2.CascadeClassifier('./object_and_faces_detection/haar_cascade/car.xml')

def cars_detection(image):
  cars = CAR_HAAR.detectMultiScale(
    image = image, 
    scaleFactor = 1.1, 
    minNeighbors = 1)
  print(cars)
  return cars

def detection_process(image):
  gray_image = bgr_to_gray(image)

  cars = cars_detection(gray_image)
  image = retangle_design(cars, image)

  cv2.imshow(f'{str(len(cars))} cars(s) founded', image)

traffic = cv2.imread('./object_and_faces_detection/images/cars/traffic.jpg')

detection_process(traffic)

cv2.waitKey(0)
cv2.destroyAllWindows()