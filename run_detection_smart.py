import cv2
import numpy as np
import os
import mediapipe as mp

# Standard Import (Requires MediaPipe 0.10.9+ with solutions)
import mediapipe as mp

try:
    mp_holistic = mp.solutions.holistic
    mp_drawing = mp.solutions.drawing_utils
except AttributeError:
    print("ERROR: mp.solutions not found. Ensure you are using a compatible MediaPipe version (e.g. 0.10.9) on Python 3.10.")
    exit()

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# --- 1. SETUP & CONFIGURATION ---
DATA_PATH = os.path.join('MP_Data')
# STRICT: Force sorted order to match training
actions = np.array(sorted(os.listdir(DATA_PATH)))
print(f"Loaded labels: {actions}")

# MANUAL BUILD TO AVOID VERSION CONFLICTS
model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30, 258)))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))

model.load_weights('action.h5')

mp_holistic = mp.solutions.holistic 
mp_drawing = mp.solutions.drawing_utils

# --- 2. HELPER FUNCTIONS ---
def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, lh, rh])

def prob_viz(res, actions, input_frame):
    """
    Draws a dynamic bar chart on the left side of the screen.
    res: list of probabilities
    actions: list of label names
    """
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        # Draw bar
        cv2.rectangle(output_frame, (0, 60+num*40), (int(prob*100), 90+num*40), (245, 117, 16), -1)
        # Draw text
        cv2.putText(output_frame, actions[num], (0, 85+num*40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        
    return output_frame

# --- 3. INITIALIZATION ---
sequence = []
sentence = []
predictions = []
threshold = 0.7

cap = cv2.VideoCapture(0)

# Mouse Callback for STOP Button
def mouse_callback(event, x, y, flags, param):
    global cap
    if event == cv2.EVENT_LBUTTONDOWN:
        # Check if click is inside the STOP button box (500, 0) to (640, 40)
        if 500 <= x <= 640 and 0 <= y <= 40:
            print("Stop button clicked. Exiting...")
            if cap.isOpened():
                cap.release()
            cv2.destroyAllWindows()
            exit()

cv2.namedWindow('ISL Smart Detection')
cv2.setMouseCallback('ISL Smart Detection', mouse_callback)

# --- 4. MAIN LOOP ---
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        # Process Frame
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = holistic.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Draw Landmarks
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        
        # Prediction Logic
        keypoints = extract_keypoints(results)
        sequence.append(keypoints)
        sequence = sequence[-30:] # Keep last 30 frames

        if len(sequence) == 30:
            res = model.predict(np.expand_dims(sequence, axis=0))[0]
            best_idx = np.argmax(res)
            best_lbael = actions[best_idx]
            
            predictions.append(best_idx)
            predictions = predictions[-10:] # Keep last 10 predictions history

            # --- STABILIZATION LOGIC ---
            # 1. Be consistent for last 10 frames
            if np.unique(predictions[-10:])[0] == best_idx: 
                # 2. Confidence check
                if res[best_idx] > threshold: 
                    
                    if len(sentence) > 0: 
                        # 3. Only append if different from logic
                        if best_lbael != sentence[-1]: 
                            sentence.append(best_lbael)
                    else:
                        sentence.append(best_lbael)

            if len(sentence) > 5: 
                sentence = sentence[-5:]

            # --- VISUALIZATION ---
            image = prob_viz(res, actions, image)
            
        # Draw Sentence Box
        cv2.rectangle(image, (0,0), (640, 40), (245, 117, 16), -1)
        cv2.putText(image, ' '.join(sentence), (3,30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Draw Stop Button
        cv2.rectangle(image, (500, 0), (640, 40), (0, 0, 255), -1) # Red box top-right
        cv2.putText(image, 'STOP', (540, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow('ISL Smart Detection', image)
        
        # Check for Close Click
        if cv2.getWindowProperty('ISL Smart Detection', cv2.WND_PROP_VISIBLE) < 1:
            break

        # Keyboard Exit
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
