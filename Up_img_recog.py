import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
from scipy.spatial.distance import cdist

# === CONSTANTS ===
SINGLE_MODEL = "hand_gesture_model_single.tflite"
DUAL_MODEL = "hand_gesture_model_dual.tflite"
SINGLE_NORM = "normalizer_params_single.npz"
DUAL_NORM = "normalizer_params_dual.npz"
LABELS_SINGLE = "labels_single.npy"
LABELS_DUAL = "labels_dual.npy"
CONFIDENCE_THRESHOLD = 0.7

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=2,
    min_detection_confidence=0.5
)

def load_resources():
    """Load models, normalizers, and labels"""
    return {
        'interpreters': {
            'single': tf.lite.Interpreter(SINGLE_MODEL),
            'dual': tf.lite.Interpreter(DUAL_MODEL)
        },
        'normalizers': {
            'single': np.load(SINGLE_NORM),
            'dual': np.load(DUAL_NORM)
        },
        'labels': {
            'single': np.load(LABELS_SINGLE, allow_pickle=True).item(),
            'dual': np.load(LABELS_DUAL, allow_pickle=True).item()
        }
    }

def extract_features(landmarks_list):
    """Extract and combine features from hands"""
    features = []
    for landmarks in landmarks_list:
        dist_matrix = cdist(landmarks, landmarks, 'euclidean')
        features.append(dist_matrix[np.triu_indices_from(dist_matrix, k=1)])
    return np.concatenate(features) if len(features) > 1 else features[0]

def predict(image_path):
    """Predict gesture from image with hand detection"""
    resources = load_resources()
    [interpreter.allocate_tensors() for interpreter in resources['interpreters'].values()]
    
    image = cv2.imread(image_path)
    image = cv2.flip(image, 1)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    results = hands.process(rgb_image)
    prediction_text = "No hands detected"
    hand_type_text = ""
    confidence = 0.0

    if results.multi_hand_landmarks:
        landmarks_list = []
        hand_labels = []
        
        # Process each hand and its handedness
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            # Draw landmarks
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Get landmarks
            landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
            landmarks_list.append(landmarks)
            
            # Get hand label (Left/Right)
            hand_labels.append(handedness.classification[0].label)

        # Determine hand type display
        num_hands = len(landmarks_list)
        if num_hands == 1:
            hand_type_text = f"{hand_labels[0]} Hand"
            mode = 'single'
        else:
            hand_type_text = "Both Hands"
            mode = 'dual'

        try:
            # Prepare features
            features = extract_features(landmarks_list[:2])
            mean = resources['normalizers'][mode]['mean']
            std = resources['normalizers'][mode]['std']
            
            # Normalize and predict
            features = (features - mean) / std
            features = np.expand_dims(features, 0).astype(np.float32)
            
            interpreter = resources['interpreters'][mode]
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            
            interpreter.set_tensor(input_details[0]['index'], features)
            interpreter.invoke()
            output = interpreter.get_tensor(output_details[0]['index'])[0]

            # Get prediction
            max_conf = np.max(output)
            label_idx = np.argmax(output)
            confidence = max_conf
            prediction_text = resources['labels'][mode][label_idx] if max_conf >= CONFIDENCE_THRESHOLD else "Other"
            
        except Exception as e:
            prediction_text = "Prediction error"
            print(f"Error: {e}")

    # Draw results
    y_pos = 40
    cv2.putText(image, f"Gesture: {prediction_text}", (20, y_pos), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    cv2.putText(image, f"Hands: {hand_type_text}", (20, y_pos+40), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
    cv2.putText(image, f"Confidence: {confidence:.2f}", (20, y_pos+80), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 255, 100), 2)

    cv2.imshow("Hand Gesture Recognition", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    predict(r"C:\Users\acer\Pictures\Camera Roll\img1.jpg")
