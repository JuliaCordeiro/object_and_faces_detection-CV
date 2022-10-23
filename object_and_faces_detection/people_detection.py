import cv2
from base_functions import (
  bgr_to_rgb,
  retangle_design
)

FACE_HAAR = cv2.CascadeClassifier('./object_and_faces_detection/haar_cascade/frontalface.xml')

def person_detection(image):
  faces = FACE_HAAR.detectMultiScale(
    image = image, 
    scaleFactor = 1.1, 
    minNeighbors = 4)
  print(faces)
  return faces

def detection_process(image):
  rgb_image = bgr_to_rgb(image)

  faces = person_detection(rgb_image)
  image = retangle_design(faces, image)

  cv2.imshow(f'{str(len(faces))} face(s) founded', image)

person1 = cv2.imread('./object_and_faces_detection/images/people/bill_gates.jpg')
person2 = cv2.imread('./object_and_faces_detection/images/people/camila_achutti.jpg')
person3 = cv2.imread('./object_and_faces_detection/images/people/jeff_bezos.jpg')
person4 = cv2.imread('./object_and_faces_detection/images/people/steve_jobs.jpg')
people = cv2.imread('./object_and_faces_detection/images/people/lucifer_cast.jpg')

detection_process(people)

cv2.waitKey(0)
cv2.destroyAllWindows()