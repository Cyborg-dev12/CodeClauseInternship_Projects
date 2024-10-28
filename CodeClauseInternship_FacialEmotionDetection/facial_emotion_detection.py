import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model

data_dir = 'FacialEmotionDetection\\train'
batch_size = 64
img_size = (48, 48)

datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_data = datagen.flow_from_directory(data_dir, target_size=img_size, color_mode='grayscale',
                                         batch_size=batch_size, class_mode='categorical', subset='training')
val_data = datagen.flow_from_directory(data_dir, target_size=img_size, color_mode='grayscale',
                                       batch_size=batch_size, class_mode='categorical', subset='validation')

model = Sequential([
    Conv2D(64, (3, 3), activation='relu', input_shape=(48, 48, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(256, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(7, activation='softmax')  
])


model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])


epochs = 25
model.fit(train_data, validation_data=val_data, epochs=epochs)



model = load_model('emotion_detection_model.h5')
model.save('emotion_detection_model.keras')
def load_model():
    try:
        model = load_model('emotion_detection_model.keras')
        model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

        print("Model loaded successfully.")
        return model
    except Exception as e:
        print("Error loading model:", e)
        return None

def detect_emotions():
    model = load_model()
    if model is None:
        print("Model not available. Exiting.")
        return

    emotion_dict = {0: "Angry", 1: "Disgust", 2: "Fear", 3: "Happy", 4: "Sad", 5: "Surprise", 6: "Neutral"}

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        if len(faces) == 0:
            print("No faces detected.")

        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            face_resized = cv2.resize(face, (48, 48))
            face_normalized = face_resized / 255.0
            face_reshaped = np.reshape(face_normalized, (1, 48, 48, 1))

            try:
                prediction = model.predict(face_reshaped)
                emotion_index = int(np.argmax(prediction))
                emotion_label = emotion_dict[emotion_index]

                cv2.putText(frame, emotion_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            except Exception as e:
                print("Prediction error:", e)

        cv2.imshow('Emotion Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def test_preprocessing():
    test_img = np.random.rand(100, 100)
    resized_img = cv2.resize(test_img, (48, 48))
    assert resized_img.shape == (48, 48), "Resizing failed"
    normalized_img = resized_img / 255.0
    assert np.max(normalized_img) <= 1.0, "Normalization failed"
    print("Preprocessing test passed.")

def test_model_loading():
    model = load_model()
    assert model is not None, "Model loading failed"
    print("Model loading test passed.")

def test_detection():
    try:
        detect_emotions()
        print("Detection test executed successfully.")
    except Exception as e:
        print("Detection test failed:", e)

if __name__ == '__main__':

    test_preprocessing()
    test_model_loading()
