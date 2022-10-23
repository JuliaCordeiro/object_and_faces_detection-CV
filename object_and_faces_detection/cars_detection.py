import cv2
from base_functions import (
  bgr_to_gray,
  bgr_to_rgb,
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
  gray_image = bgr_to_rgb(image)

  cars = cars_detection(gray_image)
  image = retangle_design(cars, image)

  cv2.imshow(f'{str(len(cars))} cars(s) founded', image)

def detection_on_video(video):
  while True:
    ret, frame = video.read()

    if type(frame) == type(None):
      break

    gray_frame = bgr_to_rgb(frame)

    cars = cars_detection(gray_frame)
    frame = retangle_design(cars, frame)

    cv2.imshow('Traffic', frame)

    if cv2.waitKey(33) == 27:
      break

traffic = cv2.imread('./object_and_faces_detection/images/cars/traffic.jpg')
traffic_video = cv2.VideoCapture('./object_and_faces_detection/videos/cars/traffic.mp4')

detection_process(traffic)
detection_on_video(traffic_video)

cv2.waitKey(0)
cv2.destroyAllWindows()