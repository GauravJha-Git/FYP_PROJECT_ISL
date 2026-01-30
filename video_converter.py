import cv2
import numpy as np
import os
import mediapipe as mp
import random

# --- CONFIGURATION ---
# Where your raw videos are
RAW_DATA_PATH = 'ISL_Raw_Data'  
# Where the math data will be saved
OUTPUT_PATH = os.path.join('MP_Data') 
# We force every video to be exactly 30 frames
SEQUENCE_LENGTH = 30 
# From 1 video, create 30 slightly different versions
AUGMENTATIONS = 30 

# Setup MediaPipe (Robust Import)
try:
    if hasattr(mp, 'solutions'):
        mp_holistic = mp.solutions.holistic
        mp_drawing = mp.solutions.drawing_utils
    else:
        import mediapipe.python.solutions as solutions
        mp_holistic = solutions.holistic
        mp_drawing = solutions.drawing_utils
    print("MediaPipe loaded successfully.")
except ImportError as e:
    print(f"CRITICAL ERROR: {e}")
    exit()
except AttributeError as e:
    print(f"CRITICAL ERROR: {e}")
    exit()

def extract_keypoints(results):
    # Extract Pose (33 points * 4 values: x,y,z,visibility)
    if results.pose_landmarks:
        pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten()
    else:
        pose = np.zeros(33*4)
        
    # Extract Left Hand (21 points * 3 values: x,y,z)
    if results.left_hand_landmarks:
        lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten()
    else:
        lh = np.zeros(21*3)
        
    # Extract Right Hand
    if results.right_hand_landmarks:
        rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten()
    else:
        rh = np.zeros(21*3)
        
    return np.concatenate([pose, lh, rh])

def augment_data(frames_list):
    """Creates copies of the data with random noise (rotation/zoom simulation)"""
    augmented_sequences = []
    
    for _ in range(AUGMENTATIONS):
        new_sequence = []
        # Random scale factor (Zoom in/out between 85% and 115%)
        scale = random.uniform(0.85, 1.15) 
        
        for frame in frames_list:
            # Simple augmentation: Multiply coordinates by scale
            # (In a real app, we might do rotation matrices, but this works for POC)
            augmented_frame = frame * scale 
            new_sequence.append(augmented_frame)
            
        augmented_sequences.append(new_sequence)
    return augmented_sequences

# --- MAIN EXECUTION ---
if not os.path.exists(RAW_DATA_PATH):
    print(f"ERROR: Folder '{RAW_DATA_PATH}' not found. Please create it.")
    exit()

actions = os.listdir(RAW_DATA_PATH)
print(f"Found actions: {actions}")

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    
    for action in actions:
        video_folder = os.path.join(RAW_DATA_PATH, action)
        if not os.path.isdir(video_folder): continue
        
        video_files = os.listdir(video_folder)
        if len(video_files) == 0: 
            print(f"Skipping {action}: No video found.")
            continue
            
        # Take the first video in the folder
        video_path = os.path.join(video_folder, video_files[0])
        cap = cv2.VideoCapture(video_path)
        
        frames_keypoints = []
        
        # 1. READ VIDEO & EXTRACT SKELETON
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(image)
            keypoints = extract_keypoints(results)
            frames_keypoints.append(keypoints)
        cap.release()
        
        if len(frames_keypoints) == 0:
            print(f"Error: Could not extract data from {action}")
            continue

        # 2. FIT TO 30 FRAMES (Pad or Cut)
        if len(frames_keypoints) < SEQUENCE_LENGTH:
            # If too short, repeat the last frame
            while len(frames_keypoints) < SEQUENCE_LENGTH:
                frames_keypoints.append(frames_keypoints[-1])
        elif len(frames_keypoints) > SEQUENCE_LENGTH:
            # If too long, cut the middle section
            start = (len(frames_keypoints) - SEQUENCE_LENGTH) // 2
            frames_keypoints = frames_keypoints[start : start + SEQUENCE_LENGTH]
            
        # 3. AUGMENT (Create 30 variations)
        augmented_batch = augment_data(frames_keypoints)
        
        # 4. SAVE TO DISK
        for i, sequence in enumerate(augmented_batch):
            save_path = os.path.join(OUTPUT_PATH, action, str(i))
            os.makedirs(save_path, exist_ok=True)
            for frame_num, frame_data in enumerate(sequence):
                np.save(os.path.join(save_path, str(frame_num) + '.npy'), frame_data)
                
        print(f"Successfully created 30 training sequences for: {action}")