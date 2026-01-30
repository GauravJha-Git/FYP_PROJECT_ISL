import numpy as np
import os

DATA_PATH = os.path.join('MP_Data')
actions = os.listdir(DATA_PATH)

print(f"--- DATA VARIANCE REPORT ---")
for action in actions:
    action_path = os.path.join(DATA_PATH, action)
    sequences = os.listdir(action_path)
    
    # Load all frames for this action
    all_frames = []
    for seq in sequences:
        for frame_num in range(30):
            res = np.load(os.path.join(action_path, seq, f"{frame_num}.npy"))
            all_frames.append(res)
            
    all_frames = np.array(all_frames)
    
    # Calculate variance across the dataset for this action
    variance = np.var(all_frames, axis=0).mean()
    print(f"Action: {action} | Overall Variance: {variance:.6f} | Total Frames: {len(all_frames)}")
    
    if variance < 0.0001:
        print(f"WARNING: {action} has extremely low variance. The model might just be memorizing one pose.")
