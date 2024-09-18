import cv2
import numpy as np
from keras.models import model_from_json
from keras.preprocessing import image
import face_recognition

# Load the model architecture from JSON file
with open("fer.json", "r") as json_file:
    loaded_model_json = json_file.read()

# Load the model from JSON file
model = model_from_json(loaded_model_json)

# Load the model weights from the HDF5 file
model.load_weights("best_model.keras")

# Define the emotion labels
emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

# Load an image for testing
img_path = "t2.jpg"  # Replace with your image path
img = cv2.imread(img_path)

# Convert BGR image to RGB (face_recognition uses RGB)
rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Detect faces in the image using face_recognitiona
face_locations = face_recognition.face_locations(rgb_img)

for top, right, bottom, left in face_locations:
    cv2.rectangle(img, (left, top), (right, bottom), (255, 0, 0), thickness=1)
    roi_gray = cv2.cvtColor(img[top:bottom, left:right], cv2.COLOR_BGR2GRAY)
    roi_gray = cv2.resize(roi_gray, (48, 48))

    img_pixels = image.img_to_array(roi_gray)
    img_pixels = np.expand_dims(img_pixels, axis=0)
    img_pixels /= 255.0

    predictions = model.predict(img_pixels)
    max_index = np.argmax(predictions[0])
    predicted_emotion = emotion_labels[max_index]

    # Display the emotion prediction
    cv2.putText(
        img,
        predicted_emotion,
        (left, top - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )

img = cv2.resize(img, (640, 640))
# Display the image with predictions
cv2.imshow("Emotion Detection", img)

cv2.waitKey(0)
cv2.destroyAllWindows()
