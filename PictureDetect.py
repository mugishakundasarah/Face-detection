import cv2

# Load the image
image = cv2.imread("team.jpg")

# Create a CascadeClassifier object to detect faces
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Detect faces in the image
faces = face_cascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5)

# Loop over the faces and crop them
for i, (x, y, w, h) in enumerate(faces):
    # Create a new image containing only the face
    face = image[y:y+h, x:x+w]

    # Save the face to disk
    cv2.imwrite(f"face_{i}.jpg", face)
