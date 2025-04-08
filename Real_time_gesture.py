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
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.5
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

def run_real_time():
    """Real-time Hand Gesture Recognition with Hand Landmarks"""
    print(" Starting real-time hand gesture recognition...")

    # Load Model
    interpreter = tf.lite.Interpreter(model_path=MODEL_FILE)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Load Normalizer & Class Names
    normalizer = FeatureNormalizer.load(NORMALIZER_FILE)
    class_names = np.load(CLASS_NAMES_FILE, allow_pickle=True)

    # Start Webcam
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
    cap.set(cv2.CAP_PROP_FPS, 120)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw MediaPipe Hand Landmarks as it was before
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Extract landmarks as numpy array
                landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])

                # Compute distance matrix and extract features
                dist_matrix = cdist(landmarks, landmarks, 'euclidean')
                features = dist_matrix[np.triu_indices_from(dist_matrix, k=1)]

                if len(features) == BASE_DISTANCES:
                    # Normalize and classify
                    features = normalizer.transform(features[np.newaxis, :]).astype(np.float32)
                    interpreter.set_tensor(input_details[0]["index"], features)
                    interpreter.invoke()
                    predictions = interpreter.get_tensor(output_details[0]["index"])[0]

                    class_id = np.argmax(predictions)
                    confidence = predictions[class_id]
                    class_name = class_names[class_id]

                    # Display prediction
                    cv2.putText(frame, f"{class_name} ({confidence:.2f})", (10, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Show the frame
        cv2.imshow("Hand Gesture Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_real_time()
