import cv2
import constants

CAR_HAAR = cv2.CascadeClassifier('./object_and_faces_detection/haar_cascade/car.xml')

def resize_image(image):
  scale_percent = 30
  width = int(image.shape[1] * scale_percent / 100)
  height = int(image.shape[0] * scale_percent / 100)
  dimension = (width, height)
  image_resize = cv2.resize(image, dimension, interpolation = cv2.INTER_AREA)
  return image_resize

def bgr_to_gray(image):
  gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  return gray_image

def retangle_design(param, image):
  for(x, y, width, height) in param:
    cv2.rectangle(image, (x, y), (x + width, y + height), constants.white, 2)
  return image

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