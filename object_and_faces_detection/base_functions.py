import cv2
import constants

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

def bgr_to_rgb(image):
  rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  return rgb_image

def retangle_design(param, image):
  for(x, y, width, height) in param:
    cv2.rectangle(image, (x, y), (x + width, y + height), constants.white, 2)
  return image