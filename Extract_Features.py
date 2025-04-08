import os
import numpy as np
import cv2
import mediapipe as mp
from tqdm import tqdm
from scipy.spatial.distance import cdist

# Constants
BASE_DISTANCES = 210  
DATASET_FILE = "gesture_dataset.npz"
CLASS_NAMES_FILE = "class_names.npy"

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1,
                        min_detection_confidence=0.7, min_tracking_confidence=0.5)

def extract_features(image):
    """Extract 210 pairwise distances from hand landmarks."""
    results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if not results.multi_hand_landmarks:
        return None

    landmarks = np.array([[lm.x, lm.y, lm.z] for lm in results.multi_hand_landmarks[0].landmark])
    dist_matrix = cdist(landmarks, landmarks, 'euclidean')
    features = dist_matrix[np.triu_indices_from(dist_matrix, k=1)]

    return features if len(features) == BASE_DISTANCES else None

def load_and_save_dataset(data_dir):
    """Load dataset, extract features, and save to file."""
    print("\n=== Extracting Features from Dataset ===")

    class_dirs = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    class_names = class_dirs.copy()
    
    X, y = [], []
    min_samples = min(len(os.listdir(os.path.join(data_dir, d))) for d in class_dirs)

    for class_idx, class_dir in enumerate(class_dirs):
        class_path = os.path.join(data_dir, class_dir)
        image_files = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

        print(f"\nProcessing {class_dir} ({len(image_files)} images)")

        for file in tqdm(image_files[:min_samples], desc=class_dir):  # Balance classes
            img_path = os.path.join(class_path, file)
            image = cv2.imread(img_path)
            if image is None:
                continue

            features = extract_features(image)
            if features is not None:
                X.append(features)
                y.append(class_idx)

    np.savez(DATASET_FILE, X=np.array(X), y=np.array(y))
    np.save(CLASS_NAMES_FILE, class_names)
    print("\n Features extracted and saved!")

if __name__ == "__main__":
    DATA_DIR = r"C:\Users\acer\Desktop\PBL Project\Combined_Dataset"
    load_and_save_dataset(DATA_DIR)
