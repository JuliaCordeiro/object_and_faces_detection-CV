import cv2
import constants

FACE_HAAR = cv2.CascadeClassifier('./object_and_faces_detection/haar_cascade/frontalface.xml')
CAT_HAAR = cv2.CascadeClassifier('./object_and_faces_detection/haar_cascade/frontalcatface.xml')

def resize_image(image):
  scale_percent = 30
  width = int(image.shape[1] * scale_percent / 100)
  height = int(image.shape[0] * scale_percent / 100)
  dimension = (width, height)
  image_resize = cv2.resize(image, dimension, interpolation = cv2.INTER_AREA)
  return image_resize

def make_image_gray(image):
  gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  return gray_image

def retangle_design(param, image):
  for(x, y, w, h) in param:
    cv2.rectangle(image, (x, y), (x + w, y + h), constants.white, 3)
  return image

def person_detection(image):
  faces = FACE_HAAR.detectMultiScale(
    image, 
    scaleFactor = 1.1, 
    minNeighbors = 2,
    minSize = (25, 25))
  return faces
  
def cat_detection(image):
  cats = CAT_HAAR.detectMultiScale(
    image, 
    scaleFactor = 1.1, 
    minNeighbors = 3,
    minSize = (75, 75))
  return cats

def detection_process(image):
  gray_image = make_image_gray(image)

  faces = person_detection(gray_image)
  image = retangle_design(faces, image)

  cats = cat_detection(gray_image)
  image = retangle_design(cats, image)

  cv2.imshow(f'{str(len(faces))} face(s) and {str(len(cats))} cats(s) founded', image)

person = cv2.imread('./object_and_faces_detection/images/people/steve_jobs.jpg')
people = cv2.imread('./object_and_faces_detection/images/people/lucifer_cast.jpg')
cat = cv2.imread('./object_and_faces_detection/images/cats/cat_04.jpg')
person_and_cat = cv2.imread('./object_and_faces_detection/images/person_with_cat.jpg')

detection_process(cat)

cv2.waitKey(0)
cv2.destroyAllWindows()