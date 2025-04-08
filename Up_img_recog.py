import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
from scipy.spatial.distance import cdist

# Constants
MODEL_FILE = "hand_gesture_model.tflite"
NORMALIZER_FILE = "normalizer_params.npz"
CLASS_NAMES_FILE = "class_names.npy"
BASE_DISTANCES = 210  

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.6
)

class FeatureNormalizer:
    """Feature Normalization"""
    def __init__(self):
        self.mean = None
        self.std = None
    
    def fit(self, X):
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
        self.std[self.std == 0] = 1
    
    def transform(self, X):
        return (X - self.mean) / self.std
    
    def save(self, path):
        np.savez(path, mean=self.mean, std=self.std)
    
    @classmethod
    def load(cls, path):
        data = np.load(path)
        normalizer = cls()
        normalizer.mean = data['mean']
        normalizer.std = data['std']
        return normalizer

def predict_from_image(image_path):
    """Predict hand gesture from a single image"""
    print("Processing image for gesture recognition...")

    # Load Model
    interpreter = tf.lite.Interpreter(model_path=MODEL_FILE)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Load Normalizer & Class Names
    normalizer = FeatureNormalizer.load(NORMALIZER_FILE)
    class_names = np.load(CLASS_NAMES_FILE, allow_pickle=True)

    # Read Image
    image = cv2.imread(image_path)
    image = cv2.flip(image,1)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Process Image with MediaPipe
    results = hands.process(rgb_image)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw landmarks
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Extract landmarks
            landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])

            # Compute distance matrix and extract features
            dist_matrix = cdist(landmarks, landmarks, 'euclidean')
            features = dist_matrix[np.triu_indices_from(dist_matrix, k=1)]

            if len(features) == BASE_DISTANCES:
                # Normalize features
                features = normalizer.transform(features[np.newaxis, :]).astype(np.float32)
                
                # Run model inference
                interpreter.set_tensor(input_details[0]["index"], features)
                interpreter.invoke()
                predictions = interpreter.get_tensor(output_details[0]["index"])[0]

                # Get the predicted class and confidence
                class_id = np.argmax(predictions)
                confidence = predictions[class_id]
                class_name = class_names[class_id]

                # Display prediction on the image
                cv2.putText(image, f"{class_name} ({confidence:.2f})", (20, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

    else:
        print("No hand detected.")

    # Show Final Output
    cv2.imshow("Hand Gesture Recognition", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Run Prediction on an Image
if __name__ == "__main__":
    predict_from_image(r"C:\Users\acer\Pictures\Camera Roll\img1.jpg")  # Change path as needed
