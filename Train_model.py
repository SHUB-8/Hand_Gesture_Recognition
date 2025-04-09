import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Multiply, BatchNormalization, Dropout
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split


SINGLE_DATASET_FILE = "features_single_hand.npz"
DUAL_DATASET_FILE = "features_dual_hand.npz"
SINGLE_MODEL_FILE = "hand_gesture_model_single.tflite"
DUAL_MODEL_FILE = "hand_gesture_model_dual.tflite"
SINGLE_H5_FILE = "hand_gesture_model_single.h5"
DUAL_H5_FILE = "hand_gesture_model_dual.h5"
SINGLE_NORMALIZER_FILE = "normalizer_params_single.npz"
DUAL_NORMALIZER_FILE = "normalizer_params_dual.npz"
NUM_CLASSES_SINGLE = 30
NUM_CLASSES_DUAL = 5

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

def train_hand_model(dataset_path, model_path_h5, model_path_tflite, normalizer_path, num_classes, hand_type="single"):
    print(f"\n=== Training {hand_type.title()} Hand Model ===")

    if os.path.exists(model_path_tflite):
        print(f"\n Deleting existing model: {model_path_tflite}")
        os.remove(model_path_tflite)

    data = np.load(dataset_path)
    X, y = data['X'], data['y']

    normalizer = FeatureNormalizer()
    normalizer.fit(X)
    normalizer.save(normalizer_path)
    X_norm = normalizer.transform(X)

    X_train, X_val, y_train, y_val = train_test_split(X_norm, y, test_size=0.2, stratify=y, random_state=42)

    model = build_model((X.shape[1],), num_classes)
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=100, batch_size=32, verbose=2)

    print(f"\n Saving model ({hand_type})...")
    model.save(model_path_h5)

    print("\n Converting Keras model to TensorFlow Lite...")
    # Update the converter configuration
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,  # Core TFLite ops
        tf.lite.OpsSet.SELECT_TF_OPS     # Fallback for unsupported ops
    ]
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    with open(model_path_tflite, "wb") as f:
        f.write(tflite_model)

    print(f"Model saved as TensorFlow Lite: {model_path_tflite}")

if __name__ == "__main__":
    # Train single hand model
    train_hand_model(SINGLE_DATASET_FILE, SINGLE_H5_FILE, SINGLE_MODEL_FILE, SINGLE_NORMALIZER_FILE, num_classes=NUM_CLASSES_SINGLE, hand_type="single")

    # Train dual hand model
    train_hand_model(DUAL_DATASET_FILE, DUAL_H5_FILE, DUAL_MODEL_FILE, DUAL_NORMALIZER_FILE, num_classes=NUM_CLASSES_DUAL, hand_type="dual")
