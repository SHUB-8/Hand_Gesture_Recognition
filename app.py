from flask import Flask, render_template, Response, request, jsonify
import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
from scipy.spatial.distance import cdist
import os
import base64
import time

app = Flask(__name__, static_folder="static", template_folder="templates")

# Global variables
camera = None
processing_active = False

# Constants
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
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.5
)

class GestureRecognizer:
    def __init__(self):
        self.resources = self.load_resources()
        self.initialize_models()
        
    def load_resources(self):
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
    
    def initialize_models(self):
        for mode in ['single', 'dual']:
            self.resources['interpreters'][mode].allocate_tensors()
    
    def extract_features(self, landmarks_list):
        features = []
        for landmarks in landmarks_list:
            dist_matrix = cdist(landmarks, landmarks, 'euclidean')
            features.append(dist_matrix[np.triu_indices_from(dist_matrix, k=1)])
        return np.concatenate(features) if len(features) > 1 else features[0]

    def predict(self, features, mode="single"):
        mean = self.resources['normalizers'][mode]['mean']
        std = self.resources['normalizers'][mode]['std']
        interpreter = self.resources['interpreters'][mode]
        
        features = (features - mean) / std
        features = np.expand_dims(features, axis=0).astype(np.float32)
        
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        interpreter.set_tensor(input_details[0]['index'], features)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])[0]
        
        max_conf = np.max(output)
        label_idx = np.argmax(output)
        return {
            'label': self.resources['labels'][mode][label_idx] if max_conf >= CONFIDENCE_THRESHOLD else "Other",
            'confidence': float(max_conf)
        }

recognizer = GestureRecognizer()

def process_frame(frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    hand_type_text = ""
    prediction_text = ""
    confidence = 0.0

    if results.multi_hand_landmarks:
        landmarks_list = []
        hand_labels = []
        
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
            landmarks_list.append(landmarks)
            hand_labels.append(handedness.classification[0].label)

        num_hands = len(landmarks_list)
        mode = 'dual' if num_hands >= 2 else 'single'
        hand_type_text = "Both Hands" if num_hands >= 2 else f"{hand_labels[0]} Hand"

        try:
            features = recognizer.extract_features(landmarks_list[:2])
            prediction = recognizer.predict(features, mode)
            prediction_text = prediction['label']
            confidence = prediction['confidence']
        except Exception as e:
            prediction_text = "Prediction Error"
            print(f"Processing error: {e}")

    y_pos = 30
    cv2.putText(frame, f"Gesture: {prediction_text}", (10, y_pos), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f"Hands: {hand_type_text}", (10, y_pos+30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    cv2.putText(frame, f"Confidence: {confidence:.2f}", (10, y_pos+60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 255, 100), 2)

    return frame

def generate_frames():
    global camera, processing_active
    
    while processing_active:
        success, frame = camera.read()
        if not success:
            break
        
        frame = cv2.flip(frame, 1)
        processed_frame = process_frame(frame)
        
        ret, buffer = cv2.imencode('.jpg', processed_frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

def get_gesture_references():
    gestures = []
    ref_path = os.path.join(app.static_folder, 'ref_images')
    
    if os.path.exists(ref_path):
        for folder in sorted(os.listdir(ref_path)):
            folder_path = os.path.join(ref_path, folder)
            if os.path.isdir(folder_path):
                images = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                if images:
                    gestures.append({
                        'name': f"Gesture {folder}",
                        'folder': folder,
                        'image': images[0]
                    })
    return gestures

@app.route('/')
def index():
    return render_template('index.html', gestures=get_gesture_references())

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_recognition', methods=['POST'])
def start_recognition():
    global camera, processing_active
    
    if camera is None:
        camera = cv2.VideoCapture(0)
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    processing_active = True
    return jsonify({"status": "started", "message": "Recognition started successfully"})

@app.route('/stop_recognition', methods=['POST'])
def stop_recognition():
    global camera, processing_active
    
    processing_active = False
    if camera is not None:
        camera.release()
        camera = None
    
    return jsonify({"status": "stopped", "message": "Recognition stopped successfully"})

@app.route('/upload_image', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({"success": False, "error": "No image provided"})
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({"success": False, "error": "No image selected"})
    
    try:
        image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
        image = cv2.flip(image, 1)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        with mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.6) as static_hands:
            results = static_hands.process(rgb_image)
            prediction_text = "No hands detected"
            hand_type_text = ""
            confidence = 0.0

            if results.multi_hand_landmarks:
                landmarks_list = []
                hand_labels = []
                
                for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                    mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
                    landmarks_list.append(landmarks)
                    hand_labels.append(handedness.classification[0].label)

                num_hands = len(landmarks_list)
                mode = 'dual' if num_hands >= 2 else 'single'
                hand_type_text = "Both Hands" if num_hands >= 2 else f"{hand_labels[0]} Hand"

                try:
                    features = recognizer.extract_features(landmarks_list[:2])
                    prediction = recognizer.predict(features, mode)
                    prediction_text = prediction['label']
                    confidence = prediction['confidence']
                except Exception as e:
                    prediction_text = "Prediction Error"
                    print(f"Image processing error: {e}")

            y_pos = 40
            cv2.putText(image, f"Gesture: {prediction_text}", (20, y_pos), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            cv2.putText(image, f"Hands: {hand_type_text}", (20, y_pos+40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
            cv2.putText(image, f"Confidence: {confidence:.2f}", (20, y_pos+80), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 255, 100), 2)

        _, buffer = cv2.imencode('.jpg', image)
        return jsonify({
            "success": True,
            "image": f"data:image/jpeg;base64,{base64.b64encode(buffer).decode('utf-8')}"
        })
        
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/about')
def about():
    return render_template('about.html', gestures=get_gesture_references())

@app.route('/documentation')
def documentation():
    return render_template('documentation.html', gestures=get_gesture_references())


@app.errorhandler(404)
def page_not_found(e):
    return render_template('index.html', gestures=get_gesture_references()), 404

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)

