from flask import Flask, render_template, Response, request, jsonify
import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
from scipy.spatial.distance import cdist
import os
import base64
from threading import Thread
import time

app = Flask(__name__, static_folder="static", template_folder="templates")

# Global variables
camera = None
processing_active = False

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

# Load model and normalizer
try:
    interpreter = tf.lite.Interpreter(model_path=MODEL_FILE)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    normalizer = FeatureNormalizer.load(NORMALIZER_FILE)
    class_names = np.load(CLASS_NAMES_FILE, allow_pickle=True)
    print("Model and supporting files loaded successfully")
except Exception as e:
    print(f"Error loading models: {e}")

def process_frame(frame):
    """Process a single frame for hand gesture recognition"""
    # Convert BGR to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process the frame with MediaPipe
    results = hands.process(rgb_frame)
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw landmarks on the frame
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
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
                
                # Get prediction results
                class_id = np.argmax(predictions)
                confidence = float(predictions[class_id])
                class_name = str(class_names[class_id])
                
                # Add prediction text to the frame
                cv2.putText(frame, f"{class_name} ({confidence:.2f})", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    return frame

def generate_frames():
    """Generate frames from webcam for streaming"""
    global camera, processing_active
    
    while processing_active:
        success, frame = camera.read()
        if not success:
            break
        
        # Flip frame horizontally for mirror effect
        frame = cv2.flip(frame, 1)
        
        # Process frame
        processed_frame = process_frame(frame)
        
        # Encode the frame
        ret, buffer = cv2.imencode('.jpg', processed_frame)
        frame_bytes = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_recognition', methods=['POST'])
def start_recognition():
    """Start the recognition process"""
    global camera, processing_active
    
    if camera is None:
        camera = cv2.VideoCapture(0)
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    processing_active = True
    return jsonify({"status": "started", "message": "Recognition started successfully"})

@app.route('/stop_recognition', methods=['POST'])
def stop_recognition():
    """Stop the recognition process"""
    global camera, processing_active
    
    processing_active = False
    
    if camera is not None:
        camera.release()
        camera = None
    
    return jsonify({"status": "stopped", "message": "Recognition stopped successfully"})

@app.route('/upload_image', methods=['POST'])
def upload_image():
    """Handle image upload and recognition"""
    if 'image' not in request.files:
        return jsonify({"success": False, "error": "No image provided"})
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({"success": False, "error": "No image selected"})
    
    try:
        # Read image file
        image_data = file.read()
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Process image using MediaPipe static mode
        with mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=1,
            min_detection_confidence=0.6
        ) as static_hands:
            # Convert BGR to RGB
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Process the image
            results = static_hands.process(rgb_image)
            
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
                        
                        # Get prediction results
                        class_id = np.argmax(predictions)
                        confidence = float(predictions[class_id])
                        class_name = str(class_names[class_id])
                        
                        # Add prediction text to the image
                        cv2.putText(image, f"{class_name} ({confidence:.2f})", (20, 50),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
            else:
                cv2.putText(image, "No hand detected", (20, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        
        # Encode the processed image
        ret, buffer = cv2.imencode('.jpg', image)
        img_str = base64.b64encode(buffer).decode('utf-8')
        
        return jsonify({
            "success": True,
            "image": f"data:image/jpeg;base64,{img_str}"
        })
        
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)