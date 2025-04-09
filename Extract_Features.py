import cv2
import numpy as np
import os
from scipy.spatial.distance import cdist
import mediapipe as mp
from tqdm import tqdm

# === PATH CONFIG ===
SINGLE_HAND_DIR = r"C:\Users\acer\Desktop\PBL Project\hand_gesture_project _2\Dataset\Single_Hand_Dataset"
DUAL_HAND_DIR = r"C:\Users\acer\Desktop\PBL Project\hand_gesture_project _2\Dataset\Both_Hand_Dataset"
OUT_SINGLE = "features_single_hand.npz"
OUT_DUAL = "features_dual_hand.npz"

# === INIT ===
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=2,
    min_detection_confidence=0.5
)

def extract_features(landmarks):
    dist_matrix = cdist(landmarks, landmarks, 'euclidean')
    return dist_matrix[np.triu_indices_from(dist_matrix, k=1)]

def process_dataset(base_dir, dual_hand=False):
    features = []
    labels = []
    label_map = {}

    # Get all subdirectories (gesture names) and enforce sequential indices
    gesture_names = sorted([d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))])
    
    for label_idx, gesture_name in enumerate(gesture_names):
        gesture_path = os.path.join(base_dir, gesture_name)
        label_map[label_idx] = gesture_name  # Ensure no skipped indices

        print(f"Processing '{gesture_name}' ({'Dual' if dual_hand else 'Single'})")

        for img_name in tqdm(os.listdir(gesture_path)):
            img_path = os.path.join(gesture_path, img_name)
            img = cv2.imread(img_path)
            if img is None:
                continue
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = hands.process(img_rgb)

            if not results.multi_hand_landmarks:
                continue

            hand_feats = []
            for hand_landmarks in results.multi_hand_landmarks:
                lm = np.array([[pt.x, pt.y, pt.z] for pt in hand_landmarks.landmark])
                feats = extract_features(lm)
                hand_feats.append(feats)

            if dual_hand and len(hand_feats) >= 2:
                feature_vector = np.concatenate(hand_feats[:2])  # 420 features (210 + 210)
                features.append(feature_vector)
                labels.append(label_idx)
            elif not dual_hand and len(hand_feats) == 1:
                features.append(hand_feats[0])  # 210 features
                labels.append(label_idx)

    return np.array(features), np.array(labels), label_map

# === PROCESS BOTH DATASETS ===
X_single, y_single, map_single = process_dataset(SINGLE_HAND_DIR, dual_hand=False)
X_dual, y_dual, map_dual = process_dataset(DUAL_HAND_DIR, dual_hand=True)

np.savez(OUT_SINGLE, X=X_single, y=y_single)
np.savez(OUT_DUAL, X=X_dual, y=y_dual)

print(f"\nSaved single-hand features to: {OUT_SINGLE} ({X_single.shape[0]} samples)")
print(f"Saved dual-hand features to: {OUT_DUAL} ({X_dual.shape[0]} samples)")

# Save label maps with ensured indices
np.save("labels_single.npy", map_single)
np.save("labels_dual.npy", map_dual)
