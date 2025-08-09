import fastapi
import socketio
import numpy as np
import json
import tensorflow.lite as tflite
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List

app = fastapi.FastAPI()
sio = socketio.AsyncServer(async_mode="asgi", cors_allowed_origins="*")
socket_app = socketio.ASGIApp(sio, app)

class Features(BaseModel):
    features: List[float]

# Load single-hand model and resources
interpreter_single = tflite.Interpreter(model_path="hand_gesture_model_single.tflite")
interpreter_single.allocate_tensors()
normalizer_single = np.load("normalizer_params_single.npz")
labels_single = np.load("labels_single.npy", allow_pickle=True).item()
input_details_single = interpreter_single.get_input_details()
output_details_single = interpreter_single.get_output_details()

# Load dual-hand model and resources
interpreter_dual = tflite.Interpreter(model_path="hand_gesture_model_dual.tflite")
interpreter_dual.allocate_tensors()
normalizer_dual = np.load("normalizer_params_dual.npz")
labels_dual = np.load("labels_dual.npy", allow_pickle=True).item()
input_details_dual = interpreter_dual.get_input_details()
output_details_dual = interpreter_dual.get_output_details()

def normalize(features, mean, std):
    return (features - mean) / std

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/predict_image")
def predict_image(payload: Features, num_hands: int = 1):
    features = np.array(payload.features)
    
    if num_hands == 1:
        normalized_features = normalize(features, normalizer_single['mean'], normalizer_single['std'])
        input_data = np.expand_dims(normalized_features, axis=0).astype(np.float32)
        interpreter_single.set_tensor(input_details_single[0]['index'], input_data)
        interpreter_single.invoke()
        output_data = interpreter_single.get_tensor(output_details_single[0]['index'])
        predicted_class = np.argmax(output_data)
        prediction = labels_single[predicted_class]
    else:
        normalized_features = normalize(features, normalizer_dual['mean'], normalizer_dual['std'])
        input_data = np.expand_dims(normalized_features, axis=0).astype(np.float32)
        interpreter_dual.set_tensor(input_details_dual[0]['index'], input_data)
        interpreter_dual.invoke()
        output_data = interpreter_dual.get_tensor(output_details_dual[0]['index'])
        predicted_class = np.argmax(output_data)
        prediction = labels_dual[predicted_class]
    
    return {"prediction": prediction}

@sio.on('connect')
async def connect(sid, environ):
    print("connect ", sid)

@sio.on('features')
async def features(sid, data):
    features = np.array(data)
    normalized_features = normalize(features, normalizer_single['mean'], normalizer_single['std'])
    input_data = np.expand_dims(normalized_features, axis=0).astype(np.float32)
    interpreter_single.set_tensor(input_details_single[0]['index'], input_data)
    interpreter_single.invoke()
    output_data = interpreter_single.get_tensor(output_details_single[0]['index'])
    predicted_class = np.argmax(output_data)
    prediction = labels_single[predicted_class]
    await sio.emit('prediction', {'prediction': prediction}, to=sid)

@sio.on('features_dual')
async def features_dual(sid, data):
    features = np.array(data)
    normalized_features = normalize(features, normalizer_dual['mean'], normalizer_dual['std'])
    input_data = np.expand_dims(normalized_features, axis=0).astype(np.float32)
    interpreter_dual.set_tensor(input_details_dual[0]['index'], input_data)
    interpreter_dual.invoke()
    output_data = interpreter_dual.get_tensor(output_details_dual[0]['index'])
    predicted_class = np.argmax(output_data)
    prediction = labels_dual[predicted_class]
    await sio.emit('prediction', {'prediction': prediction}, to=sid)

@sio.on('disconnect')
def disconnect(sid):
    print('disconnect ', sid)
