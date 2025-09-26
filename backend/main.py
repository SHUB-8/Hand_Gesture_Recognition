import fastapi
import socketio
import numpy as np
import json
import tensorflow.lite as tflite
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List

# Initialize FastAPI and Socket.IO
app = fastapi.FastAPI()
sio = socketio.AsyncServer(async_mode="asgi", cors_allowed_origins=[])

# Wrap the FastAPI app with Socket.IO and CORS middleware
# This allows both HTTP and WebSocket requests to be handled by the same server.
socket_app = CORSMiddleware(
    socketio.ASGIApp(sio, app),
    allow_origins=["*"],  # Allow all origins for simplicity in development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic model for validating the structure of incoming image prediction requests
class Features(BaseModel):
    features: List[float]

# --- Load Machine Learning Models and Resources ---

# Load single-hand gesture recognition model and its associated resources
interpreter_single = tflite.Interpreter(model_path="hand_gesture_model_single.tflite")
interpreter_single.allocate_tensors()
normalizer_single = np.load("normalizer_params_single.npz")
labels_single = np.load("labels_single.npy", allow_pickle=True).item()
input_details_single = interpreter_single.get_input_details()
output_details_single = interpreter_single.get_output_details()

# Load dual-hand gesture recognition model and its associated resources
interpreter_dual = tflite.Interpreter(model_path="hand_gesture_model_dual.tflite")
interpreter_dual.allocate_tensors()
normalizer_dual = np.load("normalizer_params_dual.npz")
labels_dual = np.load("labels_dual.npy", allow_pickle=True).item()
input_details_dual = interpreter_dual.get_input_details()
output_details_dual = interpreter_dual.get_output_details()

def normalize(features, mean, std):
    """Normalizes feature data using pre-calculated mean and std."""
    return (features - mean) / std

# --- API Endpoints ---

@app.get("/")
def read_root():
    """Root endpoint for basic health checks."""
    return {"Hello": "World"}

@app.post("/predict_image")
def predict_image(payload: Features, num_hands: int = 1):
    """
    Handles gesture prediction for a single uploaded image.
    Receives hand landmark features and the number of hands detected.
    """
    features = np.array(payload.features)
    
    if num_hands == 1:
        # Process with the single-hand model
        normalized_features = normalize(features, normalizer_single['mean'], normalizer_single['std'])
        input_data = np.expand_dims(normalized_features, axis=0).astype(np.float32)
        interpreter_single.set_tensor(input_details_single[0]['index'], input_data)
        interpreter_single.invoke()
        output_data = interpreter_single.get_tensor(output_details_single[0]['index'])
        predicted_class = np.argmax(output_data)
        prediction = labels_single[predicted_class]
        confidence = float(output_data[0][predicted_class])
    else:
        # Process with the dual-hand model
        normalized_features = normalize(features, normalizer_dual['mean'], normalizer_dual['std'])
        input_data = np.expand_dims(normalized_features, axis=0).astype(np.float32)
        interpreter_dual.set_tensor(input_details_dual[0]['index'], input_data)
        interpreter_dual.invoke()
        output_data = interpreter_dual.get_tensor(output_details_dual[0]['index'])
        predicted_class = np.argmax(output_data)
        prediction = labels_dual[predicted_class]
        confidence = float(output_data[0][predicted_class])
    
    return {"prediction": prediction, "confidence": confidence}

# --- WebSocket Event Handlers ---

@sio.on('connect')
async def connect(sid, environ):
    print("Client connected:", sid)

@sio.on('features')
async def features(sid, data):
    """
    Handles real-time gesture prediction for a single hand.
    Receives features and handedness from the client.
    """
    features = np.array(data['features'])
    hand = data['hand']
    
    # Normalize features and predict using the single-hand model
    normalized_features = normalize(features, normalizer_single['mean'], normalizer_single['std'])
    input_data = np.expand_dims(normalized_features, axis=0).astype(np.float32)
    interpreter_single.set_tensor(input_details_single[0]['index'], input_data)
    interpreter_single.invoke()
    output_data = interpreter_single.get_tensor(output_details_single[0]['index'])
    predicted_class = np.argmax(output_data)
    prediction = labels_single[predicted_class]
    confidence = float(output_data[0][predicted_class])
    
    # Send the prediction, confidence, and handedness back to the client
    await sio.emit('prediction', {'prediction': prediction, 'confidence': confidence, 'hand': hand}, to=sid)

@sio.on('features_dual')
async def features_dual(sid, data):
    """
    Handles real-time gesture prediction for two hands.
    Receives combined features and handedness for both hands.
    """
    features = np.array(data['features'])
    hands = data['hands']

    # Normalize features and predict using the dual-hand model
    normalized_features = normalize(features, normalizer_dual['mean'], normalizer_dual['std'])
    input_data = np.expand_dims(normalized_features, axis=0).astype(np.float32)
    interpreter_dual.set_tensor(input_details_dual[0]['index'], input_data)
    interpreter_dual.invoke()
    output_data = interpreter_dual.get_tensor(output_details_dual[0]['index'])
    predicted_class = np.argmax(output_data)
    prediction = labels_dual[predicted_class]
    confidence = float(output_data[0][predicted_class])

    # Send the prediction, confidence, and handedness back to the client
    await sio.emit('prediction', {'prediction': prediction, 'confidence': confidence, 'hands': hands}, to=sid)

@sio.on('disconnect')
def disconnect(sid):
    print('Client disconnected:', sid)
