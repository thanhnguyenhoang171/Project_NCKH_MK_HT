import cv2
import numpy as np
from keras.models import model_from_json
from keras.preprocessing import image
import face_recognition

# Load the model architecture from JSON file
with open("train2_epoch200_remove_earlystp\\fer.json", "r") as json_file:
    loaded_model_json = json_file.read()

# Load the model from JSON file
model = model_from_json(loaded_model_json)
a
# Load the model weights from the HDF5 file
model.load_weights("train2_epoch200_remove_earlystp\\best_model.keras")

# Define the emotion labels
emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

# Access the camera
cap = cv2.VideoCapture(
    "v1.mp4"
)  # Use 0 for the default camera, change if you have multiple cameras

while True:
    ret, frame = cap.read()  # Capture frame-by-frame
    if not ret:
        continue

    # Detect faces in the frame using face_recognition
    rgb_frame = frame[:, :, ::-1]  # Convert BGR to RGB
    face_locations = face_recognition.face_locations(rgb_frame)

    for top, right, bottom, left in face_locations:
        cv2.rectangle(frame, (left, top), (right, bottom), (255, 0, 0), thickness=2)
        roi_gray = cv2.cvtColor(frame[top:bottom, left:right], cv2.COLOR_BGR2GRAY)

        roi_gray = cv2.resize(roi_gray, (48, 48))

        img_pixels = image.img_to_array(roi_gray)
        img_pixels = np.expand_dims(img_pixels, axis=0)
        img_pixels /= 255.0

        predictions = model.predict(img_pixels)
        max_index = np.argmax(predictions[0])
        predicted_emotion = emotion_labels[max_index]

        # Display the emotion prediction
        cv2.putText(
            frame,
            predicted_emotion,
            (left, top - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    # Display the frame
    cv2.imshow("Emotion Detection", frame)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
