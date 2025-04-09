import cv2
import numpy as np
import mediapipe as mp
import time
import tensorflow as tf

# === CONFIDENCE THRESHOLD ===
CONFIDENCE_THRESHOLD = 0.75 

# === LOAD TFLITE MODELS ===
interpreter_single = tf.lite.Interpreter(model_path="hand_gesture_model_single.tflite")
interpreter_single.allocate_tensors()
input_details_single = interpreter_single.get_input_details()
output_details_single = interpreter_single.get_output_details()

interpreter_dual = tf.lite.Interpreter(model_path="hand_gesture_model_dual.tflite")
interpreter_dual.allocate_tensors()
input_details_dual = interpreter_dual.get_input_details()
output_details_dual = interpreter_dual.get_output_details()

# === LOAD NORMALIZERS ===
norm_single = np.load("normalizer_params_single.npz")
mean_single, std_single = norm_single['mean'], norm_single['std']

norm_dual = np.load("normalizer_params_dual.npz")
mean_dual, std_dual = norm_dual['mean'], norm_dual['std']

# === LOAD LABELS ===
labels_single = np.load("labels_single.npy", allow_pickle=True).item()
labels_dual = np.load("labels_dual.npy", allow_pickle=True).item()

# === UTILITY ===
def extract_features(landmarks):
    from scipy.spatial.distance import cdist
    dist_matrix = cdist(landmarks, landmarks, 'euclidean')
    return dist_matrix[np.triu_indices_from(dist_matrix, k=1)]

def predict(features, mode="single"):
    if mode == "single":
        features = (features - mean_single) / std_single
        features = np.expand_dims(features, axis=0).astype(np.float32)
        interpreter_single.set_tensor(input_details_single[0]['index'], features)
        interpreter_single.invoke()
        output = interpreter_single.get_tensor(output_details_single[0]['index'])
        max_conf = np.max(output)
        label_idx = np.argmax(output)
        return labels_single[label_idx] if max_conf >= CONFIDENCE_THRESHOLD else "Other"
    
    elif mode == "dual":
        features = (features - mean_dual) / std_dual
        features = np.expand_dims(features, axis=0).astype(np.float32)
        interpreter_dual.set_tensor(input_details_dual[0]['index'], features)
        interpreter_dual.invoke()
        output = interpreter_dual.get_tensor(output_details_dual[0]['index'])
        max_conf = np.max(output)
        label_idx = np.argmax(output)
        return labels_dual[label_idx] if max_conf >= CONFIDENCE_THRESHOLD else "Other"

# === INIT MEDIAPIPE + CAMERA ===
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

cap = cv2.VideoCapture(0)
prev_time = 0

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    image = cv2.flip(frame, 1)
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    multi_hand_landmarks = results.multi_hand_landmarks
    multi_handedness = results.multi_handedness

    hand_type_text = ""
    prediction_text = ""

    if multi_hand_landmarks:
        landmarks_all = []

        for hand_landmarks in multi_hand_landmarks:
            lm = np.array([[pt.x, pt.y, pt.z] for pt in hand_landmarks.landmark])
            feats = extract_features(lm)
            landmarks_all.append(feats)

        if len(landmarks_all) == 1:
            prediction_text = predict(landmarks_all[0], mode="single")
            hand_type_text = multi_handedness[0].classification[0].label + " Hand"

        elif len(landmarks_all) >= 2:
            combined_feats = np.concatenate(landmarks_all[:2])
            prediction_text = predict(combined_feats, mode="dual")
            hand_type_text = "Both Hands"

        # Draw landmarks
        for handLms in multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, handLms, mp_hands.HAND_CONNECTIONS)

        # Display text
        cv2.putText(image, f"Gesture: {prediction_text}", (10, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(image, f"Hand: {hand_type_text}", (10, 80), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    # FPS Display
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if prev_time != 0 else 0
    prev_time = curr_time
    cv2.putText(image, f"FPS: {int(fps)}", (10, 120), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 255, 100), 2)

    cv2.imshow("Real-Time Hand Gesture Recognition", image)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
        break

cap.release()
cv2.destroyAllWindows()
