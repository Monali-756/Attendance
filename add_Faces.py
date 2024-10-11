import cv2
import pickle
import numpy as np
import os

# Constants
IMAGE_SIZE = (50, 50)
MAX_FACES = 100
DATA_DIR = 'data/'

# Create data directory if it doesn't exist
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

video = cv2.VideoCapture(0)
facedetect = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')

faces_data = []
i = 0
name = input("Enter Your Name: ")

while True:
    ret, frame = video.read()
    if not ret:
        print("Failed to capture image")
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, 1.3, 5)
    
    for (x, y, w, h) in faces:
        crop_img = frame[y:y+h, x:x+w, :]
        resized_img = cv2.resize(crop_img, IMAGE_SIZE)
        
        if len(faces_data) < MAX_FACES and i % 10 == 0:
            faces_data.append(resized_img)
        
        i += 1
        cv2.putText(frame, f'Faces Captured: {len(faces_data)}', (50, 50), 
                    cv2.FONT_HERSHEY_COMPLEX, 1, (50, 50, 255), 1)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (50, 50, 255), 1)
    
    cv2.imshow("Frame", frame)
    
    k = cv2.waitKey(1)
    if k == ord('q') or len(faces_data) == MAX_FACES:
        break

video.release()
cv2.destroyAllWindows()

faces_data = np.asarray(faces_data)
faces_data = faces_data.reshape(MAX_FACES, -1)

# Save names
if 'names.pkl' not in os.listdir(DATA_DIR):
    names = [name] * MAX_FACES
    with open(os.path.join(DATA_DIR, 'names.pkl'), 'wb') as f:
        pickle.dump(names, f)
else:
    with open(os.path.join(DATA_DIR, 'names.pkl'), 'rb') as f:
        names = pickle.load(f)
    names.extend([name] * MAX_FACES)  # Correctly extend the list
    with open(os.path.join(DATA_DIR, 'names.pkl'), 'wb') as f:
        pickle.dump(names, f)

# Save faces data
if 'faces_data.pkl' not in os.listdir(DATA_DIR):
    with open(os.path.join(DATA_DIR, 'faces_data.pkl'), 'wb') as f:
        pickle.dump(faces_data, f)
else:
    with open(os.path.join(DATA_DIR, 'faces_data.pkl'), 'rb') as f:
        faces = pickle.load(f)
    faces = np.append(faces, faces_data, axis=0)
    with open(os.path.join(DATA_DIR, 'faces_data.pkl'), 'wb') as f:
        pickle.dump(faces, f)

print("Data saved successfully.")
