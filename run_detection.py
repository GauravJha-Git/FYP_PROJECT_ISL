import cv2
import numpy as np
import os
import mediapipe as mp
from tensorflow.keras.models import load_model

# --- 1. CRITICAL: FORCE SAME ORDER AS TRAINING ---
# We read the folder names exactly like the training script did.
DATA_PATH = os.path.join('MP_Data')
actions = np.array(sorted(os.listdir(DATA_PATH)))
print(f"Loaded labels: {actions}") # <--- CHECK THIS IN TERMINAL LATER

model = load_model('action.h5')

mp_holistic = mp.solutions.holistic 
mp_drawing = mp.solutions.drawing_utils

def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, lh, rh])

sequence = []
sentence = []
threshold = 0.7

cap = cv2.VideoCapture(0)

# Mouse Callback for Button
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

cv2.namedWindow('ISL Real-Time')
cv2.setMouseCallback('ISL Real-Time', mouse_callback)

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = holistic.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        
        # Prediction Logic
        keypoints = extract_keypoints(results)
        sequence.append(keypoints)
        sequence = sequence[-30:]

        if len(sequence) == 30:
            res = model.predict(np.expand_dims(sequence, axis=0))[0]
            
            # --- DEBUGGING PRINT ---
            # This prints: [Hello%, NoGesture%, Thanks%]
            print(f"Prediction: {res}  --> Best: {actions[np.argmax(res)]}") 
            
            prediction = actions[np.argmax(res)]
            
            if res[np.argmax(res)] > threshold: 
                if len(sentence) > 0: 
                    if prediction != sentence[-1]: 
                        sentence.append(prediction)
                else:
                    sentence.append(prediction)

            if len(sentence) > 5: 
                sentence = sentence[-5:]

            cv2.rectangle(image, (0,0), (640, 40), (245, 117, 16), -1)
            cv2.putText(image, ' '.join(sentence), (3,30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            
        # Draw Stop Button
        cv2.rectangle(image, (500, 0), (640, 40), (0, 0, 255), -1) # Red box top-right
        cv2.putText(image, 'STOP', (540, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow('ISL Real-Time', image)
        
        # Check for Close Click
        if cv2.getWindowProperty('ISL Real-Time', cv2.WND_PROP_VISIBLE) < 1:
            break
            
        # Keyboard Exit
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()