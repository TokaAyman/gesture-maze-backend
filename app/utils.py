# File: gesture-maze-backend/app/utils.py

import numpy as np

def preprocess_landmarks(landmarks_np: np.ndarray) -> np.ndarray:
    """
    landmarks_np: numpy array of shape (21, 3), containing x,y,z for 21 landmarks.
    Returns: numpy array of shape (1, 63), where each (x,y,z) triplet is recentered
             (wrist as origin) and scaled by distance from wrist→middle fingertip, then flattened.
    """
    if landmarks_np.shape != (21, 3):
        raise ValueError(f"Expected input shape (21,3), got {landmarks_np.shape}")
    
    # 1) Wrist (index 0) as origin
    wrist = landmarks_np[0]  # shape (3,)
    
    # 2) Recenter: subtract wrist from all landmarks
    centered = landmarks_np - wrist  # shape (21,3)
    
    # 3) Normalize by distance from wrist to middle fingertip (index 13)
    # Note: In MediaPipe hand landmarks, index 13 is actually the ring finger MCP
    # Middle fingertip is index 12, but based on your training code using x14/y14/z14,
    # that corresponds to index 13 in 0-based indexing
    dist = np.linalg.norm(centered[13])
    if dist == 0:
        dist = 1e-6  # avoid divide-by-zero
    
    normalized = centered / dist  # shape (21,3)
    
    # 4) Flatten into (1,63): x1,y1,z1, x2,y2,z2, ..., x21,y21,z21
    features = normalized.flatten().reshape(1, -1)  # shape (1,63)
    
    return features