import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Multiply, BatchNormalization, Dropout
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split

# Constants
NUM_CLASSES = 30  
MODEL_FILE = "hand_gesture_model.tflite"
NORMALIZER_FILE = "normalizer_params.npz"
DATASET_FILE = "gesture_dataset.npz"

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

def build_model(input_shape, num_classes):
    """Define & Compile Model"""
    inputs = Input(shape=input_shape)

    x = Dense(256, activation='relu')(inputs)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    x = Dense(512, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)

    attention = Dense(512, activation='sigmoid')(x)
    x = Multiply()([x, attention])

    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs, outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(0.0001),
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def train_model():
    """Train Model"""
    print("\n=== Checking for Existing Model ===")
    
    if os.path.exists(MODEL_FILE):
        print(f"\n Deleting existing model: {MODEL_FILE}")
        os.remove(MODEL_FILE)
    
    print("\n=== Loading Dataset ===")
    data = np.load(DATASET_FILE)
    X, y = data['X'], data['y']

    normalizer = FeatureNormalizer()
    normalizer.fit(X)
    normalizer.save(NORMALIZER_FILE)
    X_norm = normalizer.transform(X)

    X_train, X_val, y_train, y_val = train_test_split(X_norm, y, test_size=0.2, stratify=y, random_state=42)

    model = build_model((X.shape[1],), NUM_CLASSES)
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=100, batch_size=32, verbose=2)

    print("\n Saving model...")
    model.save("hand_gesture_model.h5")

    print("\n Converting Keras model to TensorFlow Lite...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    with open(MODEL_FILE, "wb") as f:
        f.write(tflite_model)
    print(" Model saved as TensorFlow Lite!")

if __name__ == "__main__":
    train_model()
